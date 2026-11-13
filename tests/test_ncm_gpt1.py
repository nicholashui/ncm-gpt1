#!/usr/bin/env python3
"""Unit + smoke tests for GPT-1-scale Next-Concept Modeling.

Run from the project root:
    PYTHONPATH=. .venv/bin/python tests/test_ncm_gpt1.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from models.vqvae import VQVAE
from utils.train_utils import (
    build_generator,
    build_pre_tokenizer,
    build_prefix_ids,
    build_vqvae,
    corrupt_codes,
    generator_ce_loss,
    param_counts,
    prefix_latents,
    set_seed,
    tokenizer_recon_accuracy,
)

PASSED = []
FAILED = []


def ok(name, detail=""):
    PASSED.append(name)
    extra = f"  ({detail})" if detail else ""
    print(f"  PASS  {name}{extra}")


def fail(name, err):
    FAILED.append((name, err))
    print(f"  FAIL  {name}: {err}")


def section(title):
    print(f"\n== {title} ==")


def _smoke_cfg():
    return OmegaConf.load(ROOT / "config/smoke.yaml")


def test_instantiate_smoke():
    section("1. Instantiate smoke VQVAE + NextConceptTransformer")
    set_seed(42)
    cfg = _smoke_cfg()
    vqvae = build_vqvae(cfg, device="cpu")
    net = build_generator(vqvae, cfg, device="cpu")
    t_tot, t_ne = param_counts(vqvae, exclude=["wte", "wpe"])
    g_tot, g_ne = param_counts(net, exclude=["wte", "wle", "wpe"])
    print(f"  smoke tokenizer: {t_tot:,} total, {t_ne:,} non-embedding")
    print(f"  smoke generator: {g_tot:,} total, {g_ne:,} non-embedding")
    assert t_tot > 0 and g_tot > 0
    levels = list(vqvae.vq.levels)
    assert list(levels) == [1, 4, 8, 16], levels
    ok("instantiate smoke", f"tok={t_tot:,} gen={g_tot:,}")
    return cfg, vqvae, net, (t_tot, t_ne, g_tot, g_ne)


def test_tokenizer_forward(vqvae, cfg):
    section("2. Tokenizer forward + backward")
    vqvae.train()
    B, L = 2, int(cfg.tokenizer.vqvae.context_length)
    V = int(cfg.tokenizer.vqvae.in_vocab_size)
    tokens = torch.randint(0, min(V, 50), (B, L))
    logits, vq_loss, utils, global_util, _ = vqvae(tokens)
    assert logits.shape == (B, L, V), logits.shape
    assert torch.isfinite(vq_loss), vq_loss
    pad = 1
    ce = F.cross_entropy(logits.reshape(-1, V), tokens.reshape(-1), ignore_index=pad)
    loss = ce + vq_loss
    vqvae.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad.detach().abs().sum() for p in vqvae.parameters() if p.requires_grad and p.grad is not None]
    assert any(g > 0 for g in grads), "no non-zero gradients"
    acc = tokenizer_recon_accuracy(logits.detach(), tokens, pad)
    ok(
        "tokenizer forward/backward",
        f"logits={tuple(logits.shape)} vq={float(vq_loss.detach()):.4f} acc={acc:.3f} util={float(global_util):.3f}",
    )


def test_encode_shapes(vqvae, cfg):
    section("3. encode / idx_to_ncm_input lengths")
    vqvae.eval()
    B, L = 2, int(cfg.tokenizer.vqvae.context_length)
    V = int(cfg.tokenizer.vqvae.in_vocab_size)
    tokens = torch.randint(0, min(V, 50), (B, L))
    idx = vqvae.encode(tokens)
    levels = list(vqvae.vq.levels)
    assert idx.shape == (B, sum(levels)), (idx.shape, sum(levels))
    ncm_x = vqvae.vq.idx_to_ncm_input(idx)
    assert ncm_x.shape[:2] == (B, sum(levels[1:])), ncm_x.shape
    assert ncm_x.shape[2] == int(cfg.tokenizer.vqvae.embed_dim)
    ok("encode shapes", f"idx={tuple(idx.shape)} ncm_x={tuple(ncm_x.shape)}")
    return idx


def test_tokenizer_step(cfg):
    section("4. One tokenizer train step on smoke_corpus")
    from utils.data import iter_documents_from_file, pack_documents

    set_seed(0)
    pre = build_pre_tokenizer(cfg)
    docs = list(iter_documents_from_file(ROOT / "data/smoke_corpus.txt"))
    L = int(cfg.tokenizer.vqvae.context_length)
    windows, _, _ = pack_documents(docs, pre, L, min_windows=8, repeat=4)
    tokens = windows[:8]
    vqvae = build_vqvae(cfg, device="cpu")
    vqvae.train()
    opt = torch.optim.AdamW((p for p in vqvae.parameters() if p.requires_grad), lr=3e-4)
    V = int(cfg.tokenizer.vqvae.in_vocab_size)
    pad = int(pre.pad_token_id)

    def run():
        logits, vq_loss, *_ = vqvae(tokens)
        ce = F.cross_entropy(logits.reshape(-1, V), tokens.reshape(-1), ignore_index=pad)
        return ce + vq_loss, ce, vq_loss

    loss0, _, _ = run()
    loss0.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        loss1, ce, vq = run()
    assert torch.isfinite(loss0) and torch.isfinite(loss1)
    ok(
        "tokenizer train step",
        f"loss {float(loss0):.4f} -> {float(loss1):.4f} (ce={float(ce):.4f} vq={float(vq):.4f})",
    )


def test_generator_step(cfg):
    section("5. One generator train step with corruption")
    set_seed(1)
    vqvae = build_vqvae(cfg, device="cpu")
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    net = build_generator(vqvae, cfg, device="cpu")
    net.train()
    B, L = 4, int(cfg.tokenizer.vqvae.context_length)
    V = int(cfg.tokenizer.vqvae.in_vocab_size)
    tokens = torch.randint(3, min(V, 80), (B, L))
    levels = list(vqvae.vq.levels)
    vocabs = list(vqvae.vq.vqvae_vocab_sizes)
    with torch.no_grad():
        idx = vqvae.encode(tokens)
        corrupted = corrupt_codes(idx, levels, vocabs, p_max=0.8, p_min=0.2)
        ncm_x = vqvae.vq.idx_to_ncm_input(corrupted)
        prefix = build_prefix_ids(B, 0, prefix_len=1, device=tokens.device)
        z_prefix = prefix_latents(vqvae, prefix, raw_token_prefix=True)
        x = torch.cat([z_prefix, ncm_x], dim=1)
    targets = idx[:, levels[0] :]
    logits = net(x)
    assert logits.shape[0] == B
    assert logits.shape[1] == targets.shape[1] == sum(levels[1:]), (logits.shape, targets.shape)
    loss = generator_ce_loss(logits, targets, levels[1:], vocabs[1:], alpha=0.0)
    assert torch.isfinite(loss), loss
    net.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad.abs().sum() for p in net.parameters() if p.requires_grad and p.grad is not None]
    assert any(g > 0 for g in grads)
    # Confirm the frozen tokenizer has no grads.
    assert all(p.grad is None for p in vqvae.parameters())
    ok("generator train step", f"logits={tuple(logits.shape)} loss={float(loss):.4f}")


def test_generate(cfg):
    section("6. generate() + VQ decode")
    set_seed(2)
    vqvae = build_vqvae(cfg, device="cpu")
    vqvae.eval()
    net = build_generator(vqvae, cfg, device="cpu")
    net.eval()
    levels = list(vqvae.vq.levels)
    B = 2
    l0 = torch.randint(0, int(vqvae.vq.vqvae_vocab_sizes[0]), (B, int(levels[0])))
    prefix = build_prefix_ids(B, 0, prefix_len=1)
    with torch.no_grad():
        codes = net.generate(l0, prefix, temperature=1.0, top_k=0, top_p=1.0)
        assert codes.shape == (B, sum(levels)), codes.shape
        _, zl = vqvae.vq.idx_to_zhats(codes, return_zlevels=True)
        tokens = vqvae.decoder(zl[-1]).argmax(dim=-1)
        assert tokens.shape == (B, int(levels[-1])), tokens.shape
    ok("generate + decode", f"codes={tuple(codes.shape)} tokens={tuple(tokens.shape)}")


def test_measure_gpt1():
    section("param counts for gpt1 config (instantiate only, no training)")
    set_seed(42)
    cfg = OmegaConf.load(ROOT / "config/gpt1.yaml")
    print("  building gpt1 VQVAE...")
    vqvae = build_vqvae(cfg, device="cpu")
    t_tot, t_ne = param_counts(vqvae, exclude=["wte", "wpe"])
    print("  building gpt1 NextConceptTransformer...")
    net = build_generator(vqvae, cfg, device="cpu")
    g_tot, g_ne = param_counts(net, exclude=["wte", "wle", "wpe"])
    print(f"  gpt1 tokenizer: {t_tot:,} total ({t_tot/1e6:.2f}M), {t_ne:,} non-embedding")
    print(f"  gpt1 generator: {g_tot:,} total ({g_tot/1e6:.2f}M), {g_ne:,} non-embedding")
    # Sanity: generator body should be in the same ballpark as GPT-1 (~117M),
    # typically a bit under because there is no 40k-token embedding table.
    assert g_tot > 50_000_000, g_tot
    assert t_tot > 1_000_000, t_tot
    ok("measure gpt1 params", f"tok={t_tot/1e6:.2f}M gen={g_tot/1e6:.2f}M")
    # Free memory before the end-to-end smoke.
    del net, vqvae
    return t_tot, t_ne, g_tot, g_ne


def test_end_to_end():
    section("7. End-to-end smoke: train_tokenizer, train_generator, generate")
    py = sys.executable
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    cmds = [
        [
            py, "train_tokenizer.py",
            "--config", "config/smoke.yaml",
            "--data", "data/smoke_corpus.txt",
            "--max_steps", "8",
            "--output_dir", "artifacts/tokenizer",
        ],
        [
            py, "train_generator.py",
            "--config", "config/smoke.yaml",
            "--data", "data/smoke_corpus.txt",
            "--tokenizer_ckpt", "artifacts/tokenizer/tokenizer.pt",
            "--max_steps", "8",
            "--output_dir", "artifacts/generator",
        ],
        [
            py, "generate.py",
            "--config", "config/smoke.yaml",
            "--tokenizer_ckpt", "artifacts/tokenizer/tokenizer.pt",
            "--generator_ckpt", "artifacts/generator/generator.pt",
            "--n", "2",
            "--seed", "0",
            "--show_levels",
        ],
    ]
    for cmd in cmds:
        print("  $ " + " ".join(cmd), flush=True)
        proc = subprocess.run(
            cmd, cwd=str(ROOT), env=env, text=True, capture_output=True
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        # Print a tail so the parent report can quote last lines.
        tail = "\n".join(out.strip().splitlines()[-20:])
        print(tail)
        if proc.returncode != 0:
            raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{out}")
    ok("end-to-end smoke scripts", "exit 0")


def main():
    print(f"Python {sys.version.split()[0]}  torch {torch.__version__}  device=cpu")
    smoke_counts = gpt1_counts = None
    steps = [
        ("instantiate", lambda: test_instantiate_smoke()),
    ]
    # Run 1, capture models for 2-3, then independent tests.
    try:
        cfg, vqvae, net, smoke_counts = test_instantiate_smoke()
    except Exception as e:
        fail("instantiate smoke", f"{e}\n{traceback.format_exc()}")
        cfg = vqvae = None

    if vqvae is not None:
        for name, fn in [
            ("tokenizer forward", lambda: test_tokenizer_forward(vqvae, cfg)),
            ("encode shapes", lambda: test_encode_shapes(vqvae, cfg)),
        ]:
            try:
                fn()
            except Exception as e:
                fail(name, f"{e}\n{traceback.format_exc()}")

    for name, fn in [
        ("tokenizer step", lambda: test_tokenizer_step(cfg or _smoke_cfg())),
        ("generator step", lambda: test_generator_step(cfg or _smoke_cfg())),
        ("generate", lambda: test_generate(cfg or _smoke_cfg())),
        ("measure gpt1", test_measure_gpt1),
        ("end-to-end", test_end_to_end),
    ]:
        try:
            result = fn()
            if name == "measure gpt1" and result is not None:
                gpt1_counts = result
        except Exception as e:
            fail(name, f"{e}\n{traceback.format_exc()}")

    print("\n======= SUMMARY =======")
    print(f"passed: {len(PASSED)}   failed: {len(FAILED)}")
    if smoke_counts:
        t_tot, t_ne, g_tot, g_ne = smoke_counts
        print(f"smoke tokenizer: {t_tot:,}  generator: {g_tot:,}")
    if gpt1_counts:
        t_tot, t_ne, g_tot, g_ne = gpt1_counts
        print(f"gpt1  tokenizer: {t_tot:,} ({t_tot/1e6:.2f}M)  generator: {g_tot:,} ({g_tot/1e6:.2f}M)")
    for name, err in FAILED:
        print(f"FAILED {name}:\n{err}")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
