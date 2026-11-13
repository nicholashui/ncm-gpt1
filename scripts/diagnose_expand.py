#!/usr/bin/env python3
"""Show residual energy and decode confidence by level, to see where expansion saturates."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from utils.adaptive import pool_to_cells
from utils.train_utils import build_pre_tokenizer, load_vqvae, resolve_device, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/gpt1.yaml")
    p.add_argument("--tokenizer_ckpt", default="artifacts/tokenizer/tokenizer.pt")
    p.add_argument("--text", default="data/smoke_corpus.txt")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    set_seed(42)
    cfg = OmegaConf.load(args.config)
    device = resolve_device(args.device)
    pre = build_pre_tokenizer(cfg)
    vqvae = load_vqvae(cfg, args.tokenizer_ckpt, device, frozen=True)
    vqvae.eval()
    L = int(cfg.tokenizer.vqvae.context_length)
    raw = Path(args.text).read_text(encoding="utf-8")
    ids = pre.encode(raw)[:L]
    if len(ids) < L:
        pad = int(getattr(pre, "pad_token_id", getattr(pre, "eos_token_id", 0)) or 0)
        ids = ids + [pad] * (L - len(ids))
    tokens = torch.tensor(ids, device=device).view(1, L)

    with torch.no_grad():
        z = vqvae._encode_pre_quant(tokens)
        idx, z_levels = vqvae.vq.z_to_idx(z, return_zlevels=True)
        z_hats = vqvae.vq.idx_to_zhats(idx, return_zlevels=False)[0]
        print(f"{'level':>6} {'ell':>6} {'resid':>10} {'acc':>8} {'conf':>8} {'skip@0.2':>9}")
        gold = tokens
        prev = z.new_zeros(z.shape)
        for k, ell in enumerate(vqvae.vq.levels):
            zhat = z_hats[k]
            residual = (z - zhat).float()
            energy = residual.pow(2).mean().sqrt().item()
            logits = vqvae.decoder(zhat)
            pred = logits.argmax(-1)
            acc = float((pred == gold).float().mean())
            conf = float(logits.softmax(-1).max(-1).values.mean())
            # fraction of cells at this level whose residual energy is < 0.2 of the mean
            down = F.interpolate(residual.transpose(1, 2), size=int(ell), mode="area").transpose(1, 2)
            cell_e = down.pow(2).sum(-1)[0]
            skip = float((cell_e < 0.2 * cell_e.mean().clamp(min=1e-12)).float().mean())
            print(f"{k:6d} {int(ell):6d} {energy:10.4f} {acc:8.3f} {conf:8.3f} {skip:9.3f}")
            prev = zhat


if __name__ == "__main__":
    main()
