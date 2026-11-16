#!/usr/bin/env python3
"""Orchestrate tokenizer then generator training under a wall-clock budget.

Designed for a Kaggle T4 x2 session (max 12 hours, ~30 GPU hours/week):

    python scripts/train_kaggle.py --hours 11.5 --output_dir /kaggle/working/artifacts

The script copies a previous run (attached as a Kaggle dataset) into the output
dir, trains the tokenizer until it finishes or time runs out, then the generator.
Re-run the same command in the next session with --resume auto (the default).
Do not reinstall torch on Kaggle; only the Python deps in requirements-kaggle.txt
(or the few packages listed below) should be installed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    # Open recreation of BookCorpus, the original GPT-1 training domain.
    ("lucadiliello/bookcorpusopen", "train", "text", None),
    # Pile sample (includes books) if BookCorpus is unavailable.
    ("JeanKaddour/minipile", "train", "text", None),
    # Last-resort small corpus so a session can still start.
    ("roneneldan/TinyStories", "train", "text", None),
]


def parse_args():
    p = argparse.ArgumentParser(description="Time-budgeted NCM training for Kaggle T4 x2")
    p.add_argument("--config", default=str(ROOT / "config/kaggle.yaml"))
    p.add_argument("--hours", type=float, default=11.5, help="wall-clock budget for this session")
    p.add_argument("--reserve_minutes", type=float, default=15.0)
    p.add_argument("--output_dir", default=os.environ.get("KAGGLE_OUTPUT", "artifacts/kaggle"))
    p.add_argument("--input_dir", default=None, help="previous /kaggle/input/.../artifacts to copy")
    p.add_argument("--nproc", type=int, default=None, help="GPUs (default: all visible CUDA devices)")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--hf_dataset", default=None)
    p.add_argument("--hf_split", default="train")
    p.add_argument("--hf_text_field", default="text")
    p.add_argument("--hf_config", default=None)
    p.add_argument("--max_windows", type=int, default=None)
    p.add_argument("--max_docs", type=int, default=None)
    p.add_argument("--no_compile", action="store_true", default=True,
                   help="leave torch.compile off (safer on T4 + DDP; default)")
    p.add_argument("--compile", action="store_true", help="enable torch.compile")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="recompute generator blocks to fit T4 memory")
    return p.parse_args()


def read_status(stage_dir: Path):
    path = stage_dir / "status.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def find_input_artifacts(explicit=None):
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.is_dir():
        return None
    for cand in kaggle_input.rglob("tokenizer"):
        if (cand / "latest.pt").is_file() or (cand / "tokenizer.pt").is_file() or (cand / "status.json").is_file():
            return cand.parent
    for cand in kaggle_input.rglob("latest.pt"):
        return cand.parent.parent
    return None


def copy_prior(src: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    for stage in ("tokenizer", "generator", "splitter"):
        s, d = src / stage, dest / stage
        if not s.exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        for item in s.iterdir():
            target = d / item.name
            if item.is_file() and not target.exists():
                shutil.copy2(item, target)
    cached = src / "packed_windows.pt"
    if cached.is_file() and not (dest / "packed_windows.pt").exists():
        shutil.copy2(cached, dest / "packed_windows.pt")


def gpu_count(nproc):
    if nproc:
        return int(nproc)
    try:
        import torch
        return max(int(torch.cuda.device_count()), 1)
    except Exception:
        return 1


def remaining_hours(start, budget):
    import time
    used = (time.monotonic() - start) / 3600.0
    return max(budget - used, 0.05)


def launch(stage, args, out_dir: Path, hours: float, extra: list[str]):
    nproc = gpu_count(args.nproc)
    script = ROOT / f"train_{stage}.py"
    stage_dir = out_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    cmd = []
    if nproc > 1:
        cmd += [sys.executable, "-m", "torch.distributed.run",
                f"--nproc_per_node={nproc}", "--standalone", str(script)]
    else:
        cmd += [sys.executable, str(script)]
    cmd += [
        "--config", str(args.config),
        "--output_dir", str(stage_dir),
        "--resume", "auto",
        "--max_hours", f"{hours:.4f}",
        "--reserve_minutes", str(args.reserve_minutes),
        "--log_every", "50",
        "--no-compile" if not args.compile else "--compile",
    ]
    if args.batch_size:
        cmd += ["--batch_size", str(args.batch_size)]
    if args.gradient_checkpointing and stage == "generator":
        cmd += ["--gradient_checkpointing"]
    if args.max_windows:
        cmd += ["--max_windows", str(args.max_windows)]
    if args.max_docs:
        cmd += ["--max_docs", str(args.max_docs)]
    if stage == "generator":
        tok = out_dir / "tokenizer" / "tokenizer.pt"
        latest = out_dir / "tokenizer" / "latest.pt"
        ckpt = tok if tok.is_file() else latest
        cmd += ["--tokenizer_ckpt", str(ckpt)]
    cmd += extra
    print("launch:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONPATH", str(ROOT))
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


def probe_dataset(name, split, field, config_name):
    try:
        from datasets import load_dataset
        kwargs = {"streaming": True}
        ds = load_dataset(name, config_name, split=split, **kwargs) if config_name else load_dataset(name, split=split, **kwargs)
        row = next(iter(ds))
        text = row.get(field) or row.get("text") or row.get("content")
        return isinstance(text, str) and bool(text.strip())
    except Exception as exc:
        print(f"dataset {name} unavailable ({exc!r})", flush=True)
        return False


def resolve_corpus(args):
    if args.hf_dataset:
        choices = [(args.hf_dataset, args.hf_split, args.hf_text_field, args.hf_config)]
    else:
        choices = list(DATASETS)
    for name, split, field, config in choices:
        print(f"probing dataset {name} ...", flush=True)
        if probe_dataset(name, split, field, config):
            extra = ["--hf_dataset", name, "--hf_split", split, "--hf_text_field", field]
            if config:
                extra += ["--hf_config", config]
            print(f"using {name}", flush=True)
            return extra, name
    raise SystemExit("no usable HuggingFace dataset found (need internet on the Kaggle session)")


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prior = find_input_artifacts(args.input_dir)
    if prior:
        print(f"copying prior artifacts from {prior}", flush=True)
        copy_prior(prior, out)

    corpus_args, corpus_name = resolve_corpus(args)
    (out / "corpus.json").write_text(json.dumps({"hf_dataset": corpus_name}, indent=2) + "\n", encoding="utf-8")

    import time
    t0 = time.monotonic()
    budget = float(args.hours)

    tok_status = read_status(out / "tokenizer")
    if not (tok_status and tok_status.get("finished")):
        hours = remaining_hours(t0, budget)
        print(f"starting tokenizer with {hours:.2f} h left", flush=True)
        rc = launch("tokenizer", args, out, hours, corpus_args)
        if rc != 0:
            raise SystemExit(f"tokenizer exited with {rc}")
        tok_status = read_status(out / "tokenizer")
    else:
        print(f"tokenizer already finished at step {tok_status.get('step')}", flush=True)

    if not (tok_status and tok_status.get("finished")):
        print("tokenizer not finished in this session; rerun to continue", flush=True)
        return

    gen_status = read_status(out / "generator")
    if gen_status and gen_status.get("finished"):
        print(f"generator already finished at step {gen_status.get('step')}", flush=True)
        return

    hours = remaining_hours(t0, budget)
    if hours < 0.25:
        print("not enough time left to start the generator; rerun in the next session", flush=True)
        return
    print(f"starting generator with {hours:.2f} h left", flush=True)
    rc = launch("generator", args, out, hours, corpus_args)
    if rc != 0:
        raise SystemExit(f"generator exited with {rc}")
    gen_status = read_status(out / "generator")
    if gen_status and gen_status.get("finished"):
        print("both stages finished", flush=True)
    else:
        print("generator not finished in this session; rerun to continue", flush=True)


if __name__ == "__main__":
    main()
