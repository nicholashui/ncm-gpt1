#!/usr/bin/env python3
"""Unconditional NCM generation: sample level-0 from the training unigram,
generate remaining levels coarse-to-fine, then decode tokens with the VQ decoder
(argmax, matching official eval).
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

from utils.adaptive import default_adaptive_cfg
from utils.train_utils import (
    add_device_args,
    configure_cuda,
    actual_prefix_pad_id,
    build_generator,
    build_pre_tokenizer,
    build_prefix_ids,
    load_vqvae,
    resolve_device,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(description="Unconditional coarse-to-fine NCM generation")
    p.add_argument("--config", default="config/smoke.yaml")
    p.add_argument("--tokenizer_ckpt", default="artifacts/tokenizer/tokenizer.pt")
    p.add_argument("--generator_ckpt", default="artifacts/generator/generator.pt")
    p.add_argument("--n", type=int, default=2, help="number of samples")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default=None, help="optional; unused for the default prefix_len=1 EOS path")
    p.add_argument("--show_levels", action="store_true", help="print intermediate level decodes")
    p.add_argument("--adaptive", dest="adaptive", action="store_true", default=None, help="expand a level only where still needed")
    p.add_argument("--no-adaptive", dest="adaptive", action="store_false", help="always expand every level and cell")
    add_device_args(p)
    p.add_argument("--cut_at_eos", action="store_true", default=True)
    p.add_argument("--no_cut_at_eos", action="store_false", dest="cut_at_eos")
    return p.parse_args()


def detokenize(pre_tok, ids, eos_id, pad_id, cut_at_eos=True):
    ids = [int(x) for x in ids]
    if cut_at_eos and eos_id in ids:
        ids = ids[: ids.index(eos_id)]
    ids = [t for t in ids if t not in (eos_id, pad_id)]
    return pre_tok.decode(ids, skip_special_tokens=True)


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    set_seed(args.seed)
    device = resolve_device(args.device)
    configure_cuda(device)

    g = cfg.get("generation", {})
    temperature = float(args.temperature if args.temperature is not None else g.get("temperature", 1.0))
    top_k = int(args.top_k if args.top_k is not None else g.get("top_k", 0))
    top_p = float(args.top_p if args.top_p is not None else g.get("top_p", 1.0))

    pre_tok = build_pre_tokenizer(cfg)
    eos_id = int(pre_tok.eos_token_id)
    pad_id = actual_prefix_pad_id(cfg, pre_tok)
    cfg.model.prefix_pad_token_id = pad_id

    vqvae = load_vqvae(cfg, args.tokenizer_ckpt, device, frozen=True)
    vqvae.eval()
    net = build_generator(vqvae, cfg, device=device)
    ckpt = torch.load(args.generator_ckpt, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model"], strict=True)
    net.eval()

    unigram = None
    if isinstance(ckpt, dict) and "inference" in ckpt and "level0_unigram" in ckpt["inference"]:
        unigram = ckpt["inference"]["level0_unigram"].float().to(device)
        if unigram.sum() <= 0:
            unigram = None
    if unigram is None:
        v0 = int(vqvae.vq.vqvae_vocab_sizes[0])
        unigram = torch.ones(v0, device=device) / v0
        print("warning: no level0_unigram in checkpoint; using uniform")

    levels = list(vqvae.vq.levels)
    l0_len = int(levels[0])
    B = int(args.n)
    prefix_len = int(cfg.model.prefix_len)

    with torch.no_grad():
        l0 = torch.multinomial(unigram.cpu(), B * l0_len, replacement=True).view(B, l0_len).to(device)
        prefix = build_prefix_ids(B, eos_id, prefix_len=prefix_len, device=device)
        if args.prompt:
            # Optional conditional hook: fill the short prefix with prompt tokens.
            # The default GPT-1 / smoke models use prefix_len=1 (a single EOS).
            ids = pre_tok.encode(args.prompt)
            ids = (ids + [pad_id] * prefix_len)[:prefix_len]
            prefix = torch.tensor([ids] * B, dtype=torch.long, device=device)
            print(f"(using --prompt prefix of length {prefix_len}; unconditional l0 is still sampled)")
        ad = default_adaptive_cfg(g.get("adaptive", {}))
        if args.adaptive is not None:
            ad["enabled"] = bool(args.adaptive)
        codes, info = net.generate(
            l0, prefix, temperature=temperature, top_k=top_k, top_p=top_p,
            adaptive=ad, return_info=True,
        )
        _, zl = vqvae.vq.idx_to_zhats(codes, return_zlevels=True)
        tokens = vqvae.decoder(zl[-1]).argmax(dim=-1)

    expected = sum(int(x) for x in levels)
    if codes.shape[1] != expected:
        raise RuntimeError(f"generated codes length {codes.shape[1]} != sum(levels)={expected}")
    if tokens.shape[1] != int(levels[-1]):
        raise RuntimeError(f"decoded tokens length {tokens.shape[1]} != L={levels[-1]}")

    print(f"codes={tuple(codes.shape)}  tokens={tuple(tokens.shape)}  temp={temperature}")
    if info.get("used_adaptive"):
        stopped = info.get("stopped_after", [])
        skips = info.get("spatial_skip_frac", [])
        mean_stop = sum(stopped) / max(len(stopped), 1)
        mean_skip = sum(skips) / max(len(skips), 1) if skips else 0.0
        print(
            f"adaptive=on  stopped_after={stopped}  mean_stop={mean_stop:.1f}/{len(levels)}  "
            f"spatial_skip_frac={mean_skip:.3f}"
        )
    else:
        print("adaptive=off  (every level and cell expanded)")
    for i in range(B):
        text = detokenize(pre_tok, tokens[i].tolist(), eos_id, pad_id, cut_at_eos=args.cut_at_eos)
        print(f"\n===== sample {i} =====")
        print(text if text else "(empty after EOS/PAD cut)")
        if args.show_levels:
            for k, z_k in enumerate(zl):
                logits_k = vqvae.decoder(z_k)
                ids_k = logits_k[i].argmax(dim=-1).tolist()
                text_k = detokenize(pre_tok, ids_k, eos_id, pad_id, cut_at_eos=False)
                print(f"  -- level {k} (ell={levels[k]}): {text_k!r}")


if __name__ == "__main__":
    main()
