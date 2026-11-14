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
from utils.data import infinite_loader, iter_all_documents, make_loader, pack_documents
from utils.train_utils import (
    TimeBudget,
    ThroughputMeter,
    TrainLogger,
    actual_prefix_pad_id,
    add_device_args,
    all_reduce_sum,
    build_generator,
    build_pre_tokenizer,
    build_prefix_ids,
    cleanup_distributed,
    configure_cuda,
    corrupt_codes,
    cuda_mem_gb,
    dist_print,
    generator_ce_loss,
    init_distributed,
    is_main_process,
    load_module_state,
    load_vqvae,
    lr_at,
    maybe_compile,
    maybe_copy_config,
    module_state_dict,
    optimize,
    param_counts,
    prefix_latents,
    prune_checkpoints,
    resolve_compile,
    resolve_resume,
    save_checkpoint,
    set_lr,
    set_seed,
    setup_amp,
    wrap_ddp,
    write_run_status,
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
    cfg.model.prefix_pad_token_id = pad_id
    eos_id = int(pre_tok.eos_token_id)
    dist_print(f"pre-tokenizer eos={eos_id} pad={pad_id}")
    if world_size > 1:
        dist_print(f"ddp world_size={world_size}  batch_size={batch_size} per GPU")

    pack_kw = pack_kwargs_from_cfg(cfg)
    max_windows = args.max_windows if args.max_windows is not None else _cfg_data(cfg, "max_windows")
    if max_windows:
        pack_kw["max_windows"] = int(max_windows)
    L = int(cfg.tokenizer.vqvae.context_length)
    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()
    docs = iter_all_documents(args)
    try:
        windows, n_docs, n_toks = pack_documents(
            docs, pre_tok, L, min_windows=batch_size * world_size, repeat=pack_repeat, **pack_kw
        )
    except ValueError as exc:
        raise SystemExit(f"provide --data and/or --hf_dataset ({exc})") from exc
    if pack_kw.get("respect_sentences"):
        dist_print("packing windows at sentence boundaries")
    dist_print(f"packed {n_docs} docs / {n_toks} tokens -> {len(windows)} windows of L={L}")
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

    vqvae = load_vqvae(cfg, args.tokenizer_ckpt, device, frozen=True)
    vqvae.eval()
    net = build_generator(vqvae, cfg, device=device)
    total, non_embd = param_counts(net, exclude=["wte", "wle", "wpe"])
    dist_print(f"generator params: {total:,} total, {non_embd:,} non-embedding")

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
    resume_path = resolve_resume(args.resume, args.output_dir, preferred=("latest.pt", "generator.pt"))
    scaler_state = None
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        load_module_state(net, ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        if ckpt.get("inference", {}).get("level0_counts") is not None:
            l0_counts = ckpt["inference"]["level0_counts"].double()
        scaler_state = ckpt.get("scaler")
        dist_print(f"resumed generator from {resume_path} at step {start_step}")

    use_compile = resolve_compile(device, args.compile)
    net = maybe_compile(net, use_compile, "generator")
    net = wrap_ddp(net, local_rank, world_size, find_unused_parameters=False)
    net.train()

    autocast, scaler, amp_dtype = setup_amp(device, args.amp)
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    dist_print(f"amp={amp_dtype}  compile={use_compile}  device={device}")

    out_dir = Path(args.output_dir)
    if is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)
        maybe_copy_config(cfg, out_dir)
    logger = TrainLogger(out_dir, "generator")

    def gathered_l0():
        t = l0_counts.clone()
        if world_size > 1:
            import torch.distributed as dist
            x = t.to(device)
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            t = x.cpu()
        return t

    def unigram_payload():
        # all_reduce must run on every rank
        counts = gathered_l0()
        denom = counts.sum().clamp(min=1.0)
        return {
            "level0_unigram": (counts / denom).float(),
            "level0_counts": counts.long(),
        }

    def dump(step, path, inference):
        payload = {
            "model": module_state_dict(net),
            "optimizer": opt.state_dict(),
            "step": step,
            "inference": inference,
            "config_name": str(cfg.get("name", "")),
        }
        if scaler is not None:
            payload["scaler"] = scaler.state_dict()
        save_checkpoint(path, payload)

    def dump_all(step, finished):
        inference = unigram_payload()
        if is_main_process():
            dump(step, out_dir / "latest.pt", inference)
            dump(step, out_dir / "generator.pt", inference)
            write_run_status(out_dir, step, max_steps, finished, extra={"stage": "generator"})
        if world_size > 1:
            import torch.distributed as dist
            dist.barrier()

    pbar = tqdm(
        range(start_step, max_steps),
        initial=start_step,
        total=max_steps,
        desc="generator",
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
            B = tokens.shape[0]
            lr = lr_at(step, warmup, peak_lr, floor_lr, max_steps)
            set_lr(opt, lr)
            opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                idx = vqvae.encode(tokens)
                l0 = idx[:, : levels[0]].reshape(-1).to("cpu")
                l0_counts += torch.bincount(l0, minlength=v0).double()
                corrupted = corrupt_codes(idx, levels, vocab_sizes, p_max=p_max, p_min=p_min)
                ncm_x = vqvae.vq.idx_to_ncm_input(corrupted)
                prefix_ids = build_prefix_ids(B, eos_id, prefix_len=prefix_len, device=device)
                z_prefix = prefix_latents(vqvae, prefix_ids, raw_prefix)
                x = torch.cat([z_prefix, ncm_x], dim=1)

            targets = idx[:, levels[0] :]
            with autocast:
                logits = net(x)
                if logits.shape[1] != targets.shape[1]:
                    raise RuntimeError(
                        f"logit length {logits.shape[1]} != target length {targets.shape[1]} "
                        f"(predict_level0 / query_level0 should be false)"
                    )
                loss, level_ce = generator_ce_loss(
                    logits, targets, target_levels, target_vocabs, alpha=alpha, return_parts=True
                )

            optimize(loss, opt, net.parameters(), clip, scaler)
            meter.update(tokens.numel())

            if (step + 1) % args.log_every == 0 or step == start_step:
                steps_s, toks_s = meter.rate()
                toks_s = all_reduce_sum(toks_s, device)
                mem = cuda_mem_gb()
                ce_first = level_ce[0] if level_ce else float("nan")
                ce_mid = level_ce[len(level_ce) // 2] if level_ce else float("nan")
                ce_last = level_ce[-1] if level_ce else float("nan")
                last_log = (
                    f"step {step + 1}/{max_steps}  loss={loss.item():.4f}  "
                    f"ce[first/mid/last]={ce_first:.4f}/{ce_mid:.4f}/{ce_last:.4f}  "
                    f"lr={lr:.2e}  {steps_s:.2f} steps/s  {toks_s:.0f} tok/s  mem={mem:.2f}GiB"
                )
                pbar.set_postfix(loss=f"{loss.item():.3f}", sps=f"{steps_s:.2f}", tps=f"{toks_s:.0f}")
                dist_print(last_log, flush=True)
                logger.log(
                    stage="generator",
                    step=step + 1,
                    max_steps=max_steps,
                    loss=float(loss.item()),
                    ce_first=ce_first,
                    ce_mid=ce_mid,
                    ce_last=ce_last,
                    lr=lr,
                    steps_per_s=steps_s,
                    tokens_per_s=toks_s,
                    mem_gb=mem,
                    hours=budget.elapsed_hours(),
                    seq_len=int(x.shape[1]),
                )
                meter.reset()
            if save_every > 0 and (step + 1) % save_every == 0:
                inference = unigram_payload()
                if is_main_process():
                    dump(step + 1, out_dir / f"generator_step{step + 1}.pt", inference)
                    dump(step + 1, out_dir / "latest.pt", inference)
                    write_run_status(out_dir, step + 1, max_steps, False, extra={"stage": "generator"})
                    prune_checkpoints(out_dir, "generator_step*.pt", keep=2)
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
        print(f"saved {out_dir / 'generator.pt'}  step={last_step}  finished={finished}")
        if last_log:
            print(last_log)
    cleanup_distributed()


if __name__ == "__main__":
    main()
