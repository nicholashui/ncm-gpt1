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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from utils.linguistic import activate_splitter
from utils.data import acquire_packed_windows, corpus_cache_meta, infinite_loader, iter_all_documents, make_loader
from utils.train_utils import (
    TimeBudget,
    ThroughputMeter,
    TrainLogger,
    actual_prefix_pad_id,
    add_device_args,
    all_reduce_sum,
    build_pre_tokenizer,
    build_vqvae,
    cleanup_distributed,
    configure_cuda,
    cuda_mem_gb,
    dist_print,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
    load_module_state,
    lr_at,
    maybe_compile,
    maybe_copy_config,
    module_state_dict,
    optimize,
    param_counts,
    prune_checkpoints,
    resolve_compile,
    resolve_resume,
    save_checkpoint,
    unwrap_model,
    set_lr,
    set_seed,
    setup_amp,
    tokenizer_recon_accuracy,
    wrap_ddp,
    write_run_status,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train the NCM residual VQ-VAE tokenizer")
    p.add_argument("--config", default="config/smoke.yaml")
    p.add_argument("--data", default=None, help="utf-8 .txt (blank-line documents) or .jsonl with a 'text' field")
    p.add_argument("--hf_dataset", default=None, help="optional HuggingFace dataset, e.g. lucadiliello/bookcorpusopen")
    p.add_argument("--hf_split", default="train")
    p.add_argument("--hf_text_field", default="text")
    p.add_argument("--hf_config", default=None, help="optional dataset config name")
    p.add_argument("--output_dir", default="artifacts/tokenizer")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--save_every", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--resume", default=None, help="checkpoint path, or 'auto' to use output_dir/latest.pt")
    add_device_args(p)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


def _cfg_data(cfg, key, default=None):
    data = cfg.get("data", {}) or {}
    return data.get(key, default) if hasattr(data, "get") else default


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if not args.hf_dataset and _cfg_data(cfg, "hf_dataset"):
        args.hf_dataset = str(_cfg_data(cfg, "hf_dataset"))
        args.hf_split = str(_cfg_data(cfg, "hf_split", args.hf_split))
        args.hf_text_field = str(_cfg_data(cfg, "hf_text_field", args.hf_text_field))
        if _cfg_data(cfg, "hf_config"):
            args.hf_config = str(_cfg_data(cfg, "hf_config"))
    if args.max_docs is None and _cfg_data(cfg, "max_docs") is not None:
        args.max_docs = int(_cfg_data(cfg, "max_docs"))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    device, rank, world_size, local_rank = init_distributed(args.device)
    set_seed(seed + rank)
    configure_cuda(device)
    pin = device.type == "cuda"
    budget = TimeBudget(args.max_hours, args.reserve_minutes)

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
    dist_print(
        f"pre-tokenizer={cfg.tokenizer.pre_tokenizer.name} "
        f"vocab={getattr(pre_tok, 'vocab_size', '?')} "
        f"eos={pre_tok.eos_token_id} pad={pad_id} "
        f"(config prefix_pad_token_id would be {actual_prefix_pad_id(cfg, pre_tok)})"
    )
    if world_size > 1:
        dist_print(f"ddp world_size={world_size}  batch_size={batch_size} per GPU")

    splitter_kw = activate_splitter(cfg, args.output_dir, pre_tok)
    max_windows = args.max_windows if args.max_windows is not None else _cfg_data(cfg, "max_windows")
    max_windows = int(max_windows or 0)
    L = int(cfg.tokenizer.vqvae.context_length)
    cache_meta = corpus_cache_meta(
        args, L, max_windows, splitter_kw["respect_sentences"], pack_repeat, int(pre_tok.eos_token_id), pad_id
    )
    cache_path = Path(args.output_dir).resolve().parent / "packed_windows.pt"
    # No NCCL barrier here. A long pack must not post a collective that the
    # peer rank can time out on. Other ranks poll packed_windows.pt.
    docs = iter_all_documents(args) if rank == 0 else ()
    try:
        windows, n_docs, n_toks, packed_now = acquire_packed_windows(
            docs,
            pre_tok,
            L,
            min_windows=batch_size * world_size,
            repeat=pack_repeat,
            respect_sentences=splitter_kw["respect_sentences"],
            max_windows=max_windows,
            cache_path=cache_path,
            cache_meta=cache_meta,
            rank=rank,
            world_size=world_size,
        )
    except ValueError as exc:
        raise SystemExit(f"provide --data and/or --hf_dataset ({exc})") from exc
    dist_print(
        f"packed {n_docs} docs / {n_toks} tokens -> {len(windows)} windows of L={L} "
        f"(cap={max_windows} cache={'wrote' if packed_now else 'hit'} {cache_path})"
    )
    loader, sampler = make_loader(
        windows,
        batch_size=batch_size,
        seed=seed,
        pin_memory=pin,
        num_workers=args.workers,
        rank=rank,
        world_size=world_size,
    )
    batches = infinite_loader(loader, sampler)

    vqvae = build_vqvae(cfg, device=device)
    total, non_embd = param_counts(vqvae, exclude=["wte", "wpe"])
    dist_print(f"tokenizer params: {total:,} total, {non_embd:,} non-embedding")

    opt = torch.optim.AdamW(vqvae.parameters(), lr=peak_lr, betas=betas, weight_decay=wd)
    start_step = 0
    resume_path = resolve_resume(args.resume, args.output_dir, preferred=("latest.pt", "tokenizer.pt"))
    scaler_state = None
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        load_module_state(vqvae, ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        scaler_state = ckpt.get("scaler")
        dist_print(f"resumed tokenizer from {resume_path} at step {start_step}")

    use_compile = resolve_compile(device, args.compile)
    vqvae = maybe_compile(vqvae, use_compile, "tokenizer")
    vqvae = wrap_ddp(vqvae, local_rank, world_size, find_unused_parameters=True)
    vqvae.train()

    autocast, scaler, amp_dtype = setup_amp(device, args.amp)
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    dist_print(f"amp={amp_dtype}  compile={use_compile}  device={device}")

    out_dir = Path(args.output_dir)
    if is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)
        maybe_copy_config(cfg, out_dir)
    logger = TrainLogger(out_dir, "tokenizer")
    vocab = int(cfg.tokenizer.vqvae.in_vocab_size)

    def dump(step, path):
        payload = {
            "model": module_state_dict(vqvae),
            "optimizer": opt.state_dict(),
            "step": step,
            "config_name": str(cfg.get("name", "")),
        }
        if scaler is not None:
            payload["scaler"] = scaler.state_dict()
        save_checkpoint(path, payload)

    def dump_all(step, finished):
        if is_main_process():
            dump(step, out_dir / "latest.pt")
            dump(step, out_dir / "tokenizer.pt")
            write_run_status(out_dir, step, max_steps, finished, extra={"stage": "tokenizer"})
        if world_size > 1:
            import torch.distributed as dist
            dist.barrier()

    pbar = tqdm(
        range(start_step, max_steps),
        initial=start_step,
        total=max_steps,
        desc="tokenizer",
        disable=not is_main_process(),
    )
    last_log = ""
    meter = ThroughputMeter()
    stopped_early = False
    last_step = start_step
    if pin:
        torch.cuda.reset_peak_memory_stats()
    try:
        for step in pbar:
            last_step = step
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
            grad_norm = optimize(loss, opt, vqvae.parameters(), clip, scaler)
            meter.update(tokens.numel())

            acc = tokenizer_recon_accuracy(logits.detach(), tokens, pad_id)
            util = float(global_util) if not torch.is_tensor(global_util) else float(global_util)
            if (step + 1) % args.log_every == 0 or step == start_step:
                steps_s, toks_s = meter.rate()
                toks_s = all_reduce_sum(toks_s, device)
                mem = cuda_mem_gb()
                util_parts = []
                if torch.is_tensor(utils_per):
                    vals = utils_per.detach().float().flatten().tolist()
                    if vals:
                        util_parts = {"util_first": vals[0], "util_mid": vals[len(vals) // 2], "util_last": vals[-1]}
                vq_mod = unwrap_model(vqvae).vq
                fine_drop_prob = float(getattr(vq_mod, "last_fine_drop_prob", 0.0))
                fine_drop_keep = int(getattr(vq_mod, "last_fine_drop_keep", 0))
                fine_drop_applied = bool(getattr(vq_mod, "last_fine_drop_applied", False))
                n_train = max(int(windows.shape[0]), 1)
                data_epoch = (step + 1) * batch_size * world_size / n_train
                last_log = (
                    f"step {step + 1}/{max_steps}  loss={loss.item():.4f}  ce={ce.item():.4f}  "
                    f"vq={float(vq_loss.detach()):.4f}  acc={acc:.4f}  util={util:.4f}  "
                    f"lr={lr:.2e}  {steps_s:.2f} steps/s  {toks_s:.0f} tok/s  mem={mem:.2f}GiB  "
                    f"grad={grad_norm:.3f}  epoch={data_epoch:.3f}  "
                    f"fine_drop_p={fine_drop_prob:.2f} keep={fine_drop_keep} applied={int(fine_drop_applied)}"
                )
                pbar.set_postfix(
                    loss=f"{loss.item():.3f}",
                    acc=f"{acc:.3f}",
                    sps=f"{steps_s:.2f}",
                    tps=f"{toks_s:.0f}",
                )
                dist_print(last_log, flush=True)
                rec = {
                    "stage": "tokenizer",
                    "step": step + 1,
                    "max_steps": max_steps,
                    "loss": float(loss.item()),
                    "ce": float(ce.item()),
                    "vq": float(vq_loss.detach()),
                    "acc": acc,
                    "util": util,
                    "lr": lr,
                    "steps_per_s": steps_s,
                    "tokens_per_s": toks_s,
                    "mem_gb": mem,
                    "hours": budget.elapsed_hours(),
                    "grad_norm": grad_norm,
                    "data_epoch": data_epoch,
                    "n_windows": int(windows.shape[0]),
                    "fine_drop_prob": fine_drop_prob,
                    "fine_drop_keep": fine_drop_keep,
                    "fine_drop_applied": fine_drop_applied,
                }
                if util_parts:
                    rec.update(util_parts)
                logger.log(**rec)
                meter.reset()
            if save_every > 0 and (step + 1) % save_every == 0:
                if is_main_process():
                    dump(step + 1, out_dir / f"tokenizer_step{step + 1}.pt")
                    dump(step + 1, out_dir / "latest.pt")
                    write_run_status(out_dir, step + 1, max_steps, False, extra={"stage": "tokenizer"})
                    prune_checkpoints(out_dir, "tokenizer_step*.pt", keep=2)
                if world_size > 1:
                    import torch.distributed as dist
                    dist.barrier()
            if budget.should_stop():
                stopped_early = True
                last_step = step + 1
                dist_print(
                    f"time budget reached after {budget.elapsed_hours():.2f} h at step {last_step}; saving"
                )
                break
            last_step = step + 1
    except KeyboardInterrupt:
        stopped_early = True
        dist_print(f"interrupted at step {last_step}; saving")

    finished = (not stopped_early) and last_step >= max_steps
    dump_all(last_step if last_step > 0 else start_step, finished)
    if is_main_process():
        print(f"saved {out_dir / 'tokenizer.pt'}  step={last_step}  finished={finished}")
        if last_log:
            print(last_log)
    cleanup_distributed()


if __name__ == "__main__":
    main()
