#!/usr/bin/env python3
"""Runtime helpers: chunked attention, atomic checkpoints, time budget, packing caps."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch

from models.common import chunked_block_diagonal_sdpa
from omegaconf import OmegaConf

from utils.data import (
    _wait_for_packed_cache,
    acquire_packed_windows,
    corpus_cache_meta,
    pack_documents,
    split_heldout_windows,
)
from utils.train_utils import (
    TimeBudget,
    ThroughputMeter,
    corrupt_codes,
    corrupt_level_rates,
    generator_ce_loss,
    level_weights,
    mid_level_slice,
    load_module_state,
    module_state_dict,
    prune_checkpoints,
    save_checkpoint,
    unwrap_model,
    write_run_status,
)


def test_chunked_matches_dense_mask():
    torch.manual_seed(0)
    B, H, D = 2, 4, 8
    prefix, levels = 1, (1, 4, 8)
    S = prefix + sum(levels)
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)

    # Dense mask identical to NextConceptTransformer block-diagonal + prefix.
    d = torch.cat([torch.full((l,), i) for i, l in enumerate(levels)]).view(1, sum(levels), 1)
    target = torch.where(d == d.transpose(1, 2), 0.0, float("-inf")).reshape(sum(levels), sum(levels))
    P, T = prefix, sum(levels)
    full = torch.cat([
        torch.cat([torch.zeros(P, P), torch.full((P, T), float("-inf"))], dim=1),
        torch.cat([torch.zeros(T, P), target], dim=1),
    ], dim=0).reshape(1, 1, S, S)

    dense = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=full, dropout_p=0.0)
    chunked = chunked_block_diagonal_sdpa(q, k, v, prefix, levels, dropout_p=0.0)
    max_diff = (dense - chunked).abs().max().item()
    assert max_diff < 1e-5, max_diff
    print(f"  PASS  chunked vs dense mask  max_diff={max_diff:.2e}")


def test_atomic_checkpoint(tmp_path: Path):
    path = tmp_path / "ck.pt"
    save_checkpoint(path, {"step": 7, "model": {"w": torch.tensor([1.0])}})
    assert path.is_file()
    assert not path.with_name(path.name + ".tmp").exists()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["step"] == 7
    print("  PASS  atomic checkpoint")


def test_unwrap_and_orig_mod_prefix():
    lin = torch.nn.Linear(4, 4)
    state = {f"_orig_mod.{k}": v.clone() for k, v in lin.state_dict().items()}
    other = torch.nn.Linear(4, 4)
    load_module_state(other, state, strict=True)
    for a, b in zip(lin.parameters(), other.parameters()):
        assert torch.allclose(a, b)

    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self._orig_mod = inner

    wrapped = Wrapper(lin)
    assert unwrap_model(wrapped) is lin
    assert set(module_state_dict(wrapped).keys()) == set(lin.state_dict().keys())
    print("  PASS  unwrap / _orig_mod prefix")


def test_time_budget_and_meter():
    import time
    budget = TimeBudget(max_hours=0.00001, reserve_minutes=0)  # 0.036 seconds
    time.sleep(0.05)
    assert budget.should_stop()
    budget2 = TimeBudget(max_hours=None)
    assert not budget2.should_stop()
    meter = ThroughputMeter()
    meter.update(100, n_steps=2)
    time.sleep(0.01)
    sps, tps = meter.rate()
    assert sps > 0 and tps > 0
    print(f"  PASS  time budget + throughput  {sps:.1f} steps/s")


class _PackTok:
    eos_token_id = 1
    pad_token_id = 0

    def encode(self, text):
        return [2, 3, 4, 5]


def test_pack_max_windows():
    class Tok:
        eos_token_id = 1
        pad_token_id = 0

        def encode(self, text):
            return [2, 3, 4, 5]

    docs = ["aaaa", "bbbb", "cccc", "dddd", "eeee"] * 20
    data, n_docs, n_toks = pack_documents(docs, Tok(), 4, max_windows=6)
    assert len(data) == 6, len(data)
    data2, _, _ = pack_documents(docs, Tok(), 4, respect_sentences=True, max_windows=5)
    assert len(data2) == 5, len(data2)
    print(f"  PASS  pack max_windows  n_docs={n_docs} n_toks={n_toks}")
    one, n_docs, n_toks = pack_documents(["ab"], _PackTok(), 4, max_windows=2)
    assert one.tolist() == [[2, 3, 4, 5], [1, 0, 0, 0]], one.tolist()
    assert n_docs == 1 and n_toks == 5


def test_level_weights_and_corrupt_rates():
    # Predicted levels for the GPT-1 tokenizer (level 0 is not in the CE).
    levels = [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 512]
    ell = torch.tensor(levels, dtype=torch.float32)
    masses = {}
    for alpha in (0.0, 0.5, 1.0):
        w = level_weights(levels, alpha)
        mass = float((ell * w).sum())
        assert abs(mass - 1.0) < 1e-4, (alpha, mass)
        masses[alpha] = float(ell[0] * w[0])
    # alpha=0.5 is the Kaggle experiment: more mass on the shortest level than the ELBO.
    assert masses[0.5] > masses[0.0] > 0.0
    rates = corrupt_level_rates(5, p_max=0.8, p_min=0.2)
    assert rates[0] == 0.8 and abs(rates[-1] - 0.2) < 1e-8
    assert rates == sorted(rates, reverse=True)
    idx = torch.zeros(2, 6, dtype=torch.long)
    rng = torch.Generator()
    rng.manual_seed(0)
    out = corrupt_codes(idx, [1, 2, 3], [8, 8, 8], p_max=1.0, p_min=1.0, generator=rng)
    assert out.shape == idx.shape
    assert int(out.max()) < 8
    print(f"  PASS  level weights + corrupt rates  coarse_mass {masses[0.0]:.4f} -> {masses[0.5]:.4f}")


def test_heldout_and_pack_cache(tmp_path: Path):
    windows = torch.arange(20 * 4).view(20, 4)
    train, val = split_heldout_windows(windows, 5, min_train=4)
    assert train.shape[0] == 15 and val.shape[0] == 5
    assert torch.equal(val, windows[-5:])
    assert torch.equal(train, windows[:-5])
    train2, val2 = split_heldout_windows(windows, 18, min_train=4)
    assert val2.shape[0] == 0 and train2.shape[0] == 20

    class Args:
        hf_dataset = "toy"
        hf_split = "train"
        hf_text_field = "text"
        data = ""
        max_docs = None

    meta = corpus_cache_meta(Args(), 4, 2, False, 1, 1, 0)
    cache = tmp_path / "packed_windows.pt"
    if cache.exists():
        cache.unlink()
    first, n_docs, n_toks, wrote = acquire_packed_windows(
        ["ab", "cd", "ef"],
        _PackTok(),
        4,
        min_windows=0,
        repeat=1,
        respect_sentences=False,
        max_windows=2,
        cache_path=cache,
        cache_meta=meta,
        rank=0,
        world_size=1,
    )
    assert wrote and cache.is_file()
    assert first.shape == (2, 4)
    second, _, _, wrote_again = acquire_packed_windows(
        ["this text must not be read"],
        _PackTok(),
        4,
        min_windows=0,
        repeat=1,
        respect_sentences=False,
        max_windows=2,
        cache_path=cache,
        cache_meta=meta,
        rank=0,
        world_size=1,
    )
    assert not wrote_again
    assert torch.equal(first, second)
    assert n_docs >= 1 and n_toks >= 5
    # A truncated cache must not be treated as a hit.
    cache.write_bytes(b"not-a-torch-cache")
    rebuilt, _, _, wrote_rebuild = acquire_packed_windows(
        ["ab", "cd"],
        _PackTok(),
        4,
        min_windows=0,
        repeat=1,
        respect_sentences=False,
        max_windows=2,
        cache_path=cache,
        cache_meta=meta,
        rank=0,
        world_size=1,
    )
    assert wrote_rebuild and rebuilt.shape == (2, 4)
    assert not cache.with_name(cache.name + ".tmp").exists()
    print(f"  PASS  held-out split + pack cache  n_docs={n_docs}")


def test_slow_pack_does_not_nccl_barrier(tmp_path: Path):
    """Peer ranks poll the cache. A slow pack must not post dist.barrier."""
    import inspect
    import threading
    import time
    import torch.distributed as dist

    src = inspect.getsource(acquire_packed_windows) + "\n" + inspect.getsource(_wait_for_packed_cache)
    assert "dist.barrier" not in src
    assert "torch.distributed" not in src

    barriers = []
    original = dist.barrier

    def boom(*args, **kwargs):
        barriers.append(1)
        raise AssertionError("NCCL barrier during pack")

    dist.barrier = boom
    cache = tmp_path / "slow_packed_windows.pt"
    if cache.exists():
        cache.unlink()
    err = cache.with_name(cache.name + ".error")
    if err.exists():
        err.unlink()
    meta = corpus_cache_meta(
        type("Args", (), {
            "hf_dataset": "toy",
            "hf_split": "train",
            "hf_text_field": "text",
            "data": "",
            "max_docs": None,
        })(),
        4, 2, False, 1, 1, 0,
    )
    results = {}
    errors = {}

    def docs():
        yield "ab"
        time.sleep(0.4)
        yield "cd"

    def run(rank, documents):
        try:
            results[rank] = acquire_packed_windows(
                documents,
                _PackTok(),
                4,
                min_windows=0,
                repeat=1,
                respect_sentences=False,
                max_windows=2,
                cache_path=cache,
                cache_meta=meta,
                rank=rank,
                world_size=2,
                wait_timeout_s=5,
                wait_poll_s=0.05,
                progress_s=0.0,
            )
        except Exception as exc:
            errors[rank] = exc

    import io
    from contextlib import redirect_stdout
    log = io.StringIO()
    try:
        with redirect_stdout(log):
            waiter = threading.Thread(target=run, args=(1, ()))
            waiter.start()
            time.sleep(0.05)
            run(0, docs())
            waiter.join(timeout=5)
    finally:
        dist.barrier = original
    text = log.getvalue()
    assert "no NCCL barrier" in text, text
    assert "packing..." in text, text
    assert "elapsed=" in text, text
    assert not waiter.is_alive(), "waiter still blocked"
    assert not errors, errors
    assert not barriers
    assert results[0][3] is True and results[1][3] is False
    assert torch.equal(results[0][0], results[1][0])
    assert results[0][0].shape == (2, 4)
    print("  PASS  slow pack waits on file, not NCCL")


def test_kaggle_campaign_defaults():
    cfg = OmegaConf.load(ROOT / "config/kaggle.yaml")
    assert int(cfg.train.generator.batch_size) == 2
    assert float(cfg.train.generator.alpha) == 0.5
    assert float(cfg.train.generator.mid_aux) == 0.5
    assert bool(cfg.model.gradient_checkpointing) is True
    assert bool(cfg.model.use_chunked_block_attn) is True
    assert bool(cfg.model.use_flex_attention) is False
    assert int(cfg.data.max_windows) == 524288
    assert int(cfg.train.generator.val_every) == 1000
    assert int(cfg.train.generator.val_windows) == 256
    assert str(cfg.splitter.backend) == "rules"
    assert bool(cfg.splitter.enabled) and bool(cfg.splitter.pack_by_sentence)
    # Local configs stay on the paper ELBO.
    gpt1 = OmegaConf.load(ROOT / "config/gpt1.yaml")
    assert float(gpt1.train.generator.alpha) == 0.0
    print("  PASS  kaggle defaults (batch 2, alpha 0.5, mid_aux 0.5, cap 524288)")


def test_mid_aux_central_third():
    assert mid_level_slice(22) == (7, 14)
    assert mid_level_slice(1) == (0, 1)
    torch.manual_seed(0)
    levels = [4, 9, 16, 25, 36, 49]
    vocabs = [8] * len(levels)
    span = sum(levels)
    logits = torch.randn(2, span, 8, requires_grad=True)
    targets = torch.randint(0, 8, (2, span))
    loss0, parts = generator_ce_loss(
        logits, targets, levels, vocabs, alpha=0.5, mid_aux=0.0, return_parts=True
    )
    loss1, parts1 = generator_ce_loss(
        logits, targets, levels, vocabs, alpha=0.5, mid_aux=0.5, return_parts=True
    )
    assert parts == parts1
    start, end = mid_level_slice(len(parts))
    mid_mean = sum(parts[start:end]) / (end - start)
    assert abs(float(loss1.detach()) - float(loss0.detach()) - 0.5 * mid_mean) < 1e-5
    loss0.backward(retain_graph=True)
    grad0 = logits.grad.detach().abs().clone()
    logits.grad = None
    loss1.backward()
    grad1 = logits.grad.detach().abs()
    offset = sum(levels[:start])
    mid_w = sum(levels[start:end])
    # The auxiliary term increases gradient on the central-third positions.
    assert float(grad1[:, offset:offset + mid_w].sum()) > float(grad0[:, offset:offset + mid_w].sum())
    print(
        f"  PASS  mid_aux central third [{start}:{end})  "
        f"delta={float((loss1 - loss0).detach()):.4f}"
    )


def test_prune_and_status(tmp_path: Path):
    for i in range(4):
        (tmp_path / f"tokenizer_step{i}.pt").write_bytes(b"x")
        os.utime(tmp_path / f"tokenizer_step{i}.pt", (i, i))
    prune_checkpoints(tmp_path, "tokenizer_step*.pt", keep=2)
    left = sorted(p.name for p in tmp_path.glob("tokenizer_step*.pt"))
    assert left == ["tokenizer_step2.pt", "tokenizer_step3.pt"], left
    write_run_status(tmp_path, step=12, max_steps=100, finished=False, extra={"stage": "tokenizer"})
    import json
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["step"] == 12 and status["stage"] == "tokenizer" and not status["finished"]
    print("  PASS  prune + status.json")


def main():
    tmp = ROOT / "artifacts" / "_runtime_test"
    tmp.mkdir(parents=True, exist_ok=True)
    tests = [
        test_chunked_matches_dense_mask,
        lambda: test_atomic_checkpoint(tmp),
        test_unwrap_and_orig_mod_prefix,
        test_time_budget_and_meter,
        test_pack_max_windows,
        test_level_weights_and_corrupt_rates,
        lambda: test_heldout_and_pack_cache(tmp),
        lambda: test_slow_pack_does_not_nccl_barrier(tmp),
        test_kaggle_campaign_defaults,
        test_mid_aux_central_third,
        lambda: test_prune_and_status(tmp),
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {fn}: {exc}")
            import traceback
            traceback.print_exc()
    print(f"runtime tests failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
