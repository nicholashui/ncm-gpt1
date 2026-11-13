#!/usr/bin/env python3
"""Stage 2: train the Next-Concept Transformer on a frozen VQ-VAE tokenizer.

For each batch of pre-tokenized ids (B, L):
  1. idx = vqvae.encode(tokens)                         # (B, sum(levels))
  2. corrupt INPUT codes per level (labels stay clean)
  3. ncm_x = vqvae.vq.idx_to_ncm_input(corrupted_idx)   # (B, sum(levels[1:]), d)
  4. prefix = EOS; z_prefix = encoder.wte(prefix) when raw_token_prefix
     x = cat([z_prefix, ncm_x], dim=1)
  5. logits = net(x)                                    # (B, sum(levels[1:]), max_vocab)
  6. CE against clean idx[:, levels[0]:], weighted by w_k
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from utils.linguistic import pack_kwargs_from_cfg
from utils.data import (
    infinite_loader,
    iter_documents_from_file,
    iter_documents_from_hf,
    make_loader,
    pack_documents,
)
from utils.train_utils import (
    actual_prefix_pad_id,
    add_device_args,
    build_generator,
    build_pre_tokenizer,
    build_prefix_ids,
    configure_cuda,
    corrupt_codes,
    generator_ce_loss,
    load_vqvae,
    lr_at,
    maybe_compile,
    resolve_compile,
    maybe_copy_config,
    optimize,
    param_counts,
    prefix_latents,
    resolve_device,
    save_checkpoint,
    set_lr,
    set_seed,
    setup_amp,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train the NCM generator on a frozen tokenizer")
    p.add_argument("--config", default="config/smoke.yaml")
    p.add_argument("--tokenizer_ckpt", default="artifacts/tokenizer/tokenizer.pt")
    p.add_argument("--data", default=None)
    p.add_argument("--hf_dataset", default=None)
    p.add_argument("--hf_split", default="train")
    p.add_argument("--hf_text_field", default="text")
    p.add_argument("--hf_config", default=None)
    p.add_argument("--output_dir", default="artifacts/generator")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--save_every", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--resume", default=None)
    add_device_args(p)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log_every", type=int, default=1)
    return p.parse_args()


def collect_documents(args):
    docs = []
    if args.data:
        docs.extend(iter_documents_from_file(args.data))
    if args.hf_dataset:
        docs.extend(
            iter_documents_from_hf(
                args.hf_dataset,
                split=args.hf_split,
                text_field=args.hf_text_field,
                config_name=args.hf_config,
            )
        )
    if not docs:
        raise SystemExit("provide --data and/or --hf_dataset")
    return docs


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    set_seed(seed)
    device = resolve_device(args.device)
    configure_cuda(device)
    pin = device.type == "cuda"

    gcfg = cfg.train.generator
    batch_size = int(args.batch_size or gcfg.batch_size)
    max_steps = int(args.max_steps or gcfg.max_steps)
    save_every = int(args.save_every or gcfg.get("save_every", max_steps))
    warmup = int(gcfg.warmup)
    peak_lr = float(gcfg.peak_lr)
    floor_lr = float(gcfg.floor_lr)
    betas = tuple(float(x) for x in gcfg.betas)
    wd = float(gcfg.weight_decay)
    clip = float(gcfg.grad_clip)
    alpha = float(gcfg.get("alpha", 0.0))
    p_max = float(gcfg.get("corrupt_p_max", 0.8))
    p_min = float(gcfg.get("corrupt_p_min", 0.2))
    pack_repeat = int(gcfg.get("pack_repeat", 1))
    raw_prefix = bool(cfg.model.get("raw_token_prefix", True))
    prefix_len = int(cfg.model.prefix_len)

    pre_tok = build_pre_tokenizer(cfg)
    pad_id = actual_prefix_pad_id(cfg, pre_tok)
    # Write the real pad id onto the live config so a saved copy is accurate.
    cfg.model.prefix_pad_token_id = pad_id
    eos_id = int(pre_tok.eos_token_id)
    print(f"pre-tokenizer eos={eos_id} pad={pad_id}")

    docs = collect_documents(args)
    L = int(cfg.tokenizer.vqvae.context_length)
    pack_kw = pack_kwargs_from_cfg(cfg)
    windows, n_docs, n_toks = pack_documents(
        docs, pre_tok, L, min_windows=batch_size, repeat=pack_repeat, **pack_kw
    )
    if pack_kw.get("respect_sentences"):
        print("packing windows at sentence boundaries")
    print(f"packed {n_docs} docs / {n_toks} tokens -> {len(windows)} windows of L={L}")
    loader = make_loader(
        windows, batch_size=batch_size, seed=seed, pin_memory=pin, num_workers=args.workers
    )
    batches = infinite_loader(loader)

    vqvae = load_vqvae(cfg, args.tokenizer_ckpt, device, frozen=True)
    vqvae.eval()
    net = build_generator(vqvae, cfg, device=device)
    use_compile = resolve_compile(device, args.compile)
    net = maybe_compile(net, use_compile, "generator")
    net.train()
    total, non_embd = param_counts(net, exclude=["wte", "wle", "wpe"])
    print(f"generator params: {total:,} total, {non_embd:,} non-embedding")

    levels = list(vqvae.vq.levels)
    vocab_sizes = list(vqvae.vq.vqvae_vocab_sizes)
    v0 = int(vocab_sizes[0])
    target_levels = levels[1:]
    target_vocabs = vocab_sizes[1:]

    opt = torch.optim.AdamW(
        (p for p in net.parameters() if p.requires_grad),
        lr=peak_lr,
        betas=betas,
        weight_decay=wd,
    )
    l0_counts = torch.zeros(v0, dtype=torch.double)
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        net.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        if ckpt.get("inference", {}).get("level0_counts") is not None:
            l0_counts = ckpt["inference"]["level0_counts"].double()
        print(f"resumed generator from {args.resume} at step {start_step}")

    autocast, scaler, amp_dtype = setup_amp(device, args.amp)
    print(f"amp={amp_dtype}  compile={use_compile}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    maybe_copy_config(cfg, out_dir)

    def unigram_payload():
        denom = l0_counts.sum().clamp(min=1.0)
        return {
            "level0_unigram": (l0_counts / denom).float(),
            "level0_counts": l0_counts.long(),
        }

    def dump(step, path):
        save_checkpoint(
            path,
            {
                "model": net.state_dict(),
                "optimizer": opt.state_dict(),
                "step": step,
                # Official eval_shard.py reads sd["model"] and
                # generator.pt:inference.level0_unigram
                "inference": unigram_payload(),
                "config_name": str(cfg.get("name", "")),
            },
        )

    pbar = tqdm(range(start_step, max_steps), initial=start_step, total=max_steps, desc="generator")
    last_log = ""
    for step in pbar:
        tokens = next(batches).to(device, non_blocking=pin)
        B = tokens.shape[0]
        lr = lr_at(step, warmup, peak_lr, floor_lr, max_steps)
        set_lr(opt, lr)
        opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            idx = vqvae.encode(tokens)  # (B, sum(levels)), clean labels
            l0 = idx[:, : levels[0]].reshape(-1).to("cpu")
            l0_counts += torch.bincount(l0, minlength=v0).double()
            corrupted = corrupt_codes(idx, levels, vocab_sizes, p_max=p_max, p_min=p_min)
            ncm_x = vqvae.vq.idx_to_ncm_input(corrupted)  # (B, sum(levels[1:]), d)
            prefix_ids = build_prefix_ids(B, eos_id, prefix_len=prefix_len, device=device)
            z_prefix = prefix_latents(vqvae, prefix_ids, raw_prefix)
            x = torch.cat([z_prefix, ncm_x], dim=1)

        targets = idx[:, levels[0] :]
        with autocast:
            logits = net(x)  # (B, sum(levels[1:]), max_vocab)
            if logits.shape[1] != targets.shape[1]:
                raise RuntimeError(
                    f"logit length {logits.shape[1]} != target length {targets.shape[1]} "
                    f"(predict_level0 / query_level0 should be false)"
                )
            loss = generator_ce_loss(logits, targets, target_levels, target_vocabs, alpha=alpha)

        optimize(loss, opt, net.parameters(), clip, scaler)

        last_log = (
            f"step {step + 1}/{max_steps}  loss={loss.item():.4f}  "
            f"logits={tuple(logits.shape)}  targets={tuple(targets.shape)}  lr={lr:.2e}"
        )
        if (step + 1) % args.log_every == 0 or step == start_step:
            pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{lr:.1e}")
            print(last_log, flush=True)
        if save_every > 0 and (step + 1) % save_every == 0:
            dump(step + 1, out_dir / f"generator_step{step + 1}.pt")

    dump(max_steps, out_dir / "generator.pt")
    print(f"saved {out_dir / 'generator.pt'}")
    print(last_log)


if __name__ == "__main__":
    main()
