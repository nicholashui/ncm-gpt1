#!/usr/bin/env python3
"""Print the sentence → phrase → word tree for a text or file."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from utils.linguistic import format_tree, split_text, split_text_tree, collect_spans


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--file", default="data/smoke_corpus.txt")
    p.add_argument("--config", default=None, help="optional; if set, also align to the pre-tokenizer")
    args = p.parse_args()
    if args.text is not None:
        text = args.text
    else:
        text = Path(args.file).read_text(encoding="utf-8")
    text = text.strip()
    if len(text) > 4000:
        text = text[:4000] + "…"
    tree = split_text_tree(text)
    print(format_tree(tree))
    print(f"\n{len(tree)} sentences, "
          f"{sum(len(s['phrases']) for s in tree)} phrases, "
          f"{sum(len(w) for s in tree for w in (p['words'] for p in s['phrases']))} words")
    if args.config:
        from omegaconf import OmegaConf
        from utils.train_utils import build_pre_tokenizer
        pre = build_pre_tokenizer(OmegaConf.load(args.config))
        spans = split_text(text, pre)
        print("\nToken spans:")
        for kind in ("sentence", "phrase", "word"):
            pairs = collect_spans(spans, kind)
            print(f"  {kind}: {pairs[:24]}{' …' if len(pairs) > 24 else ''}  (n={len(pairs)})")


if __name__ == "__main__":
    main()
