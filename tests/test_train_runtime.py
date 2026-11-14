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
from utils.data import pack_documents
from utils.train_utils import (
    TimeBudget,
    ThroughputMeter,
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
