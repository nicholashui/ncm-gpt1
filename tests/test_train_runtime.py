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
from omegaconf import OmegaConf

from utils.data import (
    acquire_packed_windows,
    corpus_cache_meta,
    pack_documents,
    split_heldout_windows,
)
from utils.train_utils import (
    TimeBudget,
    ThroughputMeter,
    corrupt_codes,
    corrupt_level_rates,
    level_weights,
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


class _PackTok:
    eos_token_id = 1
    pad_token_id = 0

    def encode(self, text):
        return [2, 3, 4, 5]


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
    one, n_docs, n_toks = pack_documents(["ab"], _PackTok(), 4, max_windows=2)
    assert one.tolist() == [[2, 3, 4, 5], [1, 0, 0, 0]], one.tolist()
    assert n_docs == 1 and n_toks == 5


def test_level_weights_and_corrupt_rates():
    # Predicted levels for the GPT-1 tokenizer (level 0 is not in the CE).
    levels = [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 512]
    ell = torch.tensor(levels, dtype=torch.float32)
    masses = {}
    for alpha in (0.0, 0.5, 1.0):
        w = level_weights(levels, alpha)
        mass = float((ell * w).sum())
        assert abs(mass - 1.0) < 1e-4, (alpha, mass)
        masses[alpha] = float(ell[0] * w[0])
    # alpha=0.5 is the Kaggle experiment: more mass on the shortest level than the ELBO.
    assert masses[0.5] > masses[0.0] > 0.0
    rates = corrupt_level_rates(5, p_max=0.8, p_min=0.2)
    assert rates[0] == 0.8 and abs(rates[-1] - 0.2) < 1e-8
    assert rates == sorted(rates, reverse=True)
    idx = torch.zeros(2, 6, dtype=torch.long)
    rng = torch.Generator()
    rng.manual_seed(0)
    out = corrupt_codes(idx, [1, 2, 3], [8, 8, 8], p_max=1.0, p_min=1.0, generator=rng)
    assert out.shape == idx.shape
    assert int(out.max()) < 8
    print(f"  PASS  level weights + corrupt rates  coarse_mass {masses[0.0]:.4f} -> {masses[0.5]:.4f}")


def test_heldout_and_pack_cache(tmp_path: Path):
    windows = torch.arange(20 * 4).view(20, 4)
    train, val = split_heldout_windows(windows, 5, min_train=4)
    assert train.shape[0] == 15 and val.shape[0] == 5
    assert torch.equal(val, windows[-5:])
    assert torch.equal(train, windows[:-5])
    train2, val2 = split_heldout_windows(windows, 18, min_train=4)
    assert val2.shape[0] == 0 and train2.shape[0] == 20

    class Args:
        hf_dataset = "toy"
        hf_split = "train"
        hf_text_field = "text"
        data = ""
        max_docs = None

    meta = corpus_cache_meta(Args(), 4, 2, False, 1, 1, 0)
    cache = tmp_path / "packed_windows.pt"
    if cache.exists():
        cache.unlink()
    first, n_docs, n_toks, wrote = acquire_packed_windows(
        ["ab", "cd", "ef"],
        _PackTok(),
        4,
        min_windows=0,
        repeat=1,
        respect_sentences=False,
        max_windows=2,
        cache_path=cache,
        cache_meta=meta,
        rank=0,
        world_size=1,
    )
    assert wrote and cache.is_file()
    assert first.shape == (2, 4)
    second, _, _, wrote_again = acquire_packed_windows(
        ["this text must not be read"],
        _PackTok(),
        4,
        min_windows=0,
        repeat=1,
        respect_sentences=False,
        max_windows=2,
        cache_path=cache,
        cache_meta=meta,
        rank=0,
        world_size=1,
    )
    assert not wrote_again
    assert torch.equal(first, second)
    assert n_docs >= 1 and n_toks >= 5
    print(f"  PASS  held-out split + pack cache  n_docs={n_docs}")


def test_kaggle_campaign_defaults():
    cfg = OmegaConf.load(ROOT / "config/kaggle.yaml")
    assert int(cfg.train.generator.batch_size) == 4
    assert float(cfg.train.generator.alpha) == 0.5
    assert bool(cfg.model.gradient_checkpointing) is True
    assert bool(cfg.model.use_chunked_block_attn) is True
    assert bool(cfg.model.use_flex_attention) is False
    assert int(cfg.data.max_windows) == 524288
    assert int(cfg.train.generator.val_every) == 1000
    assert int(cfg.train.generator.val_windows) == 256
    assert str(cfg.splitter.backend) == "rules"
    assert bool(cfg.splitter.enabled) and bool(cfg.splitter.pack_by_sentence)
    # Local configs stay on the paper ELBO.
    gpt1 = OmegaConf.load(ROOT / "config/gpt1.yaml")
    assert float(gpt1.train.generator.alpha) == 0.0
    print("  PASS  kaggle defaults (batch 4, alpha 0.5, cap 524288, T4 safety flags)")


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
        test_level_weights_and_corrupt_rates,
        lambda: test_heldout_and_pack_cache(tmp),
        test_kaggle_campaign_defaults,
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
