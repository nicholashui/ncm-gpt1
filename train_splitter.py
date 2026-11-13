#!/usr/bin/env python3
"""Train a tiny BoundaryTagger from rule-based sentence/phrase/word labels."""

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

from models.splitter import BoundaryTagger
from utils.data import iter_documents_from_file
from utils.linguistic import collect_spans, split_text
from utils.train_utils import add_device_args, build_pre_tokenizer, configure_cuda, resolve_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train a sentence/phrase/word boundary tagger")
    p.add_argument("--config", default="config/gpt1.yaml")
    p.add_argument("--data", default=None, help="text or jsonl; default is cfg.train.data or smoke corpus")
    p.add_argument("--out", default="artifacts/splitter/splitter.pt")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    add_device_args(p)
    return p.parse_args()


def label_window(ids, tokenizer, eos_id, pad_id):
    L = len(ids)
    y = torch.zeros(L, 3, dtype=torch.float32)
    if not ids or all(t in {eos_id, pad_id} for t in ids):
        return y
    text = tokenizer.decode(ids, skip_special_tokens=True) if hasattr(tokenizer, "decode") else ""
    if not text.strip():
        return y
    sentences = split_text(text, tokenizer)
    for k, kind in enumerate(("sentence", "phrase", "word")):
        for s, e in collect_spans(sentences, kind):
            if 0 <= s < L:
                y[s, k] = 1.0
    y[0, :] = 1.0
    return y


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    set_seed(42)
    device = resolve_device(args.device)
    configure_cuda(device)
    pre = build_pre_tokenizer(cfg)
    eos_id = int(pre.eos_token_id)
    pad_id = int(pre.pad_token_id)
    data_path = args.data
    if data_path is None:
        data_path = str(cfg.get("data", {}).get("path", "data/smoke_corpus.txt")) if cfg.get("data") else "data/smoke_corpus.txt"
        if not Path(data_path).exists():
            data_path = "data/smoke_corpus.txt"
    docs = list(iter_documents_from_file(data_path))
    L = int(args.context)
    windows, labels = [], []
    for text in docs:
        ids = list(pre.encode(text) or [])
        if not ids:
            continue
        ids = ids + [eos_id]
        for i in range(0, len(ids), L):
            chunk = ids[i:i + L]
            if len(chunk) < L:
                chunk = chunk + [pad_id] * (L - len(chunk))
            windows.append(chunk)
            labels.append(label_window(chunk, pre, eos_id, pad_id))
    if not windows:
        raise SystemExit(f"no windows from {data_path}")
    x = torch.tensor(windows, dtype=torch.long)
    y = torch.stack(labels, dim=0)
    model = BoundaryTagger(int(pre.vocab_size), args.emb_dim, args.hidden, pad_id=pad_id).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    bs = min(8, len(x))
    for step in range(int(args.steps)):
        idx = torch.randint(0, len(x), (bs,))
        xb, yb = x[idx].to(device), y[idx].to(device)
        logits = model(xb)
        mask = (xb != pad_id).unsqueeze(-1).float()
        loss = (F.binary_cross_entropy_with_logits(logits, yb, reduction="none") * mask).sum() / mask.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == args.steps:
            print(f"step {step+1}/{args.steps}  loss={loss.item():.4f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "vocab_size": int(pre.vocab_size),
                "emb_dim": int(args.emb_dim),
                "hidden": int(args.hidden),
                "pad_id": pad_id,
            },
        },
        out,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
