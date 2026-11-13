#!/usr/bin/env python3
"""Stage 1: train a multiscale residual VQ-VAE tokenizer.

Build the VQ-VAE from config (do NOT use VQVAETokenizer.from_config, which
always loads a checkpoint). Loss is token CE (PAD ignored) + vq_loss.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Scripts are runnable with cwd or PYTHONPATH equal to the project root.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
import torch.nn.functional as F
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
    build_pre_tokenizer,
    build_vqvae,
    configure_cuda,
    lr_at,
    maybe_compile,
    resolve_compile,
    maybe_copy_config,
    optimize,
    param_counts,
    resolve_device,
    save_checkpoint,
    set_lr,
    set_seed,
    setup_amp,
    tokenizer_recon_accuracy,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train the NCM residual VQ-VAE tokenizer")
    p.add_argument("--config", default="config/smoke.yaml")
    p.add_argument("--data", default=None, help="utf-8 .txt (blank-line documents) or .jsonl with a 'text' field")
    p.add_argument("--hf_dataset", default=None, help="optional HuggingFace dataset, e.g. roneneldan/TinyStories")
    p.add_argument("--hf_split", default="train")
    p.add_argument("--hf_text_field", default="text")
    p.add_argument("--hf_config", default=None, help="optional dataset config name")
    p.add_argument("--output_dir", default="artifacts/tokenizer")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--save_every", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--resume", default=None, help="path to a tokenizer checkpoint to resume")
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

    tcfg = cfg.train.tokenizer
    batch_size = int(args.batch_size or tcfg.batch_size)
    max_steps = int(args.max_steps or tcfg.max_steps)
    save_every = int(args.save_every or tcfg.get("save_every", max_steps))
    warmup = int(tcfg.warmup)
    peak_lr = float(tcfg.peak_lr)
    floor_lr = float(tcfg.floor_lr)
    betas = tuple(float(x) for x in tcfg.betas)
    wd = float(tcfg.weight_decay)
    clip = float(tcfg.grad_clip)
    pack_repeat = int(tcfg.get("pack_repeat", 1))

    pre_tok = build_pre_tokenizer(cfg)
    pad_id = int(pre_tok.pad_token_id)
    print(
        f"pre-tokenizer={cfg.tokenizer.pre_tokenizer.name} "
        f"vocab={getattr(pre_tok, 'vocab_size', '?')} "
        f"eos={pre_tok.eos_token_id} pad={pad_id} "
        f"(config prefix_pad_token_id would be {actual_prefix_pad_id(cfg, pre_tok)})"
    )

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

    vqvae = build_vqvae(cfg, device=device)
    use_compile = resolve_compile(device, args.compile)
    vqvae = maybe_compile(vqvae, use_compile, "tokenizer")
    vqvae.train()
    total, non_embd = param_counts(vqvae, exclude=["wte", "wpe"])
    print(f"tokenizer params: {total:,} total, {non_embd:,} non-embedding")

    opt = torch.optim.AdamW(vqvae.parameters(), lr=peak_lr, betas=betas, weight_decay=wd)
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        vqvae.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        print(f"resumed tokenizer from {args.resume} at step {start_step}")

    autocast, scaler, amp_dtype = setup_amp(device, args.amp)
    print(f"amp={amp_dtype}  compile={use_compile}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    maybe_copy_config(cfg, out_dir)
    vocab = int(cfg.tokenizer.vqvae.in_vocab_size)

    def dump(step, path):
        save_checkpoint(
            path,
            {
                "model": vqvae.state_dict(),
                "optimizer": opt.state_dict(),
                "step": step,
                "config_name": str(cfg.get("name", "")),
            },
        )

    pbar = tqdm(range(start_step, max_steps), initial=start_step, total=max_steps, desc="tokenizer")
    last_log = ""
    for step in pbar:
        tokens = next(batches).to(device, non_blocking=pin)
        lr = lr_at(step, warmup, peak_lr, floor_lr, max_steps)
        set_lr(opt, lr)
        opt.zero_grad(set_to_none=True)
        with autocast:
            logits, vq_loss, utils_per, global_util, _ppl = vqvae(tokens)
            ce = F.cross_entropy(
                logits.float().reshape(-1, vocab),
                tokens.reshape(-1),
                ignore_index=pad_id,
            )
            loss = ce + vq_loss
        optimize(loss, opt, vqvae.parameters(), clip, scaler)

        acc = tokenizer_recon_accuracy(logits.detach(), tokens, pad_id)
        util = float(global_util) if not torch.is_tensor(global_util) else float(global_util)
        last_log = (
            f"step {step + 1}/{max_steps}  loss={loss.item():.4f}  ce={ce.item():.4f}  "
            f"vq={float(vq_loss.detach()):.4f}  acc={acc:.4f}  util={util:.4f}  lr={lr:.2e}"
        )
        if (step + 1) % args.log_every == 0 or step == start_step:
            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                acc=f"{acc:.3f}",
                util=f"{util:.3f}",
                lr=f"{lr:.1e}",
            )
            print(last_log, flush=True)
        if save_every > 0 and (step + 1) % save_every == 0:
            dump(step + 1, out_dir / f"tokenizer_step{step + 1}.pt")

    dump(max_steps, out_dir / "tokenizer.pt")
    print(f"saved {out_dir / 'tokenizer.pt'}")
    print(last_log)


if __name__ == "__main__":
    main()
