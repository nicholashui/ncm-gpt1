"""Text loading and packing into fixed-length token windows."""

from __future__ import annotations

import array
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler


class PackedTokenDataset(Dataset):
    def __init__(self, windows: torch.Tensor):
        self.windows = windows.long()

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, idx):
        return self.windows[idx]


def iter_documents_from_file(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {path}")
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text", obj.get("content", ""))
                if isinstance(text, str) and text.strip():
                    yield text
        return
    buf = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                if buf:
                    yield "\n".join(buf)
                    buf = []
            else:
                buf.append(line.rstrip("\n"))
    if buf:
        yield "\n".join(buf)


def iter_documents_from_hf(name, split="train", text_field="text", config_name=None, streaming=True, max_docs=None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The optional 'datasets' package is required for --hf_dataset. "
            "Install it with: pip install datasets"
        ) from exc
    kwargs = {"streaming": bool(streaming)}
    if config_name:
        ds = load_dataset(name, config_name, split=split, **kwargs)
    else:
        ds = load_dataset(name, split=split, **kwargs)
    n = 0
    for row in ds:
        text = row[text_field] if text_field in row else row.get("text") or row.get("content", "")
        if isinstance(text, str) and text.strip():
            yield text
            n += 1
            if max_docs is not None and n >= int(max_docs):
                return


def _push_window(rows, ids, L, pad, cap):
    """Append one length-L window as a compact uint32 row. Returns False at cap."""
    if cap and len(rows) >= cap:
        return False
    seq = [int(x) for x in ids]
    if len(seq) < L:
        seq.extend([pad] * (L - len(seq)))
    elif len(seq) > L:
        seq = seq[:L]
    rows.append(array.array("I", seq))
    return True


def _rows_to_tensor(rows, L):
    """Materialize packed rows as an int64 tensor without a giant list of Python ints."""
    if not rows:
        return torch.empty(0, L, dtype=torch.long)
    buf = array.array("I")
    for row in rows:
        buf.extend(row)
    raw = torch.frombuffer(bytearray(buf), dtype=torch.int32).clone()
    return raw.to(dtype=torch.long).view(-1, L)


def pack_documents(
    documents,
    pre_tokenizer,
    context_length,
    min_windows=0,
    repeat=1,
    respect_sentences=False,
    max_windows=0,
    progress_s=30.0,
):
    """Encode documents, append EOS, and pack/pad to length L.

    When respect_sentences is True, a new window starts at a sentence boundary
    whenever the next sentence would overflow the current window. A sentence
    longer than L is packed by tokens as usual.

    Windows are stored as uint32 rows while streaming so a few hundred thousand
    windows stay near 1 GiB instead of a multi-gigabyte list of Python ints.
    The returned tensor is still int64, matching the previous packing result.
    """
    eos = int(pre_tokenizer.eos_token_id)
    pad = int(pre_tokenizer.pad_token_id)
    L = int(context_length)
    n_docs = 0
    n_toks = 0
    rows = []
    cap = int(max_windows or 0)

    def enough():
        return cap > 0 and len(rows) >= cap

    def push(ids):
        return _push_window(rows, ids, L, pad, cap)

    import time
    started = time.monotonic()
    last_log = started
    announced = False

    def note_progress(force=False):
        """Time-based progress. A 5000-doc stride stayed silent for the whole v6 hang."""
        nonlocal last_log, announced
        interval = float(progress_s if progress_s is not None else 30.0)
        # Quiet for tiny unit-test caps unless the caller asked for frequent logs.
        if cap < 65536 and interval >= 30.0:
            return
        now = time.monotonic()
        if announced and not force and (now - last_log) < interval:
            return
        announced = True
        last_log = now
        print(
            f"packing... docs={n_docs} windows={len(rows)} tokens={n_toks} "
            f"elapsed={now - started:.0f}s cap={cap or 'none'}",
            flush=True,
        )

    if respect_sentences:
        from utils.linguistic import split_sentences
        cur = []
        stop = False
        for _ in range(max(int(repeat), 1)):
            for text in documents:
                sentences = split_sentences(text) or ([text] if str(text).strip() else [])
                wrote = False
                for sent in sentences:
                    ids = list(pre_tokenizer.encode(sent) or [])
                    if not ids:
                        continue
                    wrote = True
                    n_toks += len(ids)
                    if len(ids) > L:
                        if cur:
                            push(cur)
                            cur = []
                        for i in range(0, len(ids), L):
                            if enough():
                                break
                            chunk = ids[i:i + L]
                            if len(chunk) == L:
                                push(chunk)
                            else:
                                cur = list(chunk)
                        continue
                    if cur and len(cur) + len(ids) > L:
                        push(cur)
                        cur = []
                    cur.extend(ids)
                    if enough():
                        break
                if wrote:
                    n_docs += 1
                    if len(cur) >= L:
                        push(cur[:L])
                        cur = cur[L:]
                    if not enough():
                        if len(cur) + 1 <= L:
                            cur.append(eos)
                            n_toks += 1
                        else:
                            push(cur)
                            cur = [eos]
                            n_toks += 1
                    note_progress()
                if enough():
                    stop = True
                    break
            if stop or enough():
                break
        if cur and not enough():
            push(cur)
        if not rows:
            raise ValueError("no tokens produced from the given documents")
    else:
        stream = array.array("I")
        for _ in range(max(int(repeat), 1)):
            for text in documents:
                ids = pre_tokenizer.encode(text)
                if not ids:
                    continue
                stream.extend(int(x) for x in ids)
                stream.append(eos)
                n_docs += 1
                note_progress()
                if cap and len(stream) >= cap * L:
                    break
            if cap and len(stream) >= cap * L:
                break
        if not stream:
            raise ValueError("no tokens produced from the given documents")
        n_toks = len(stream)
        for i in range(0, len(stream), L):
            if enough():
                break
            chunk = stream[i:i + L]
            if not chunk:
                continue
            push(chunk)

    if cap and len(rows) > cap:
        del rows[cap:]
    if min_windows and len(rows) < min_windows:
        reps = (min_windows + len(rows) - 1) // len(rows)
        rows = rows * reps
    note_progress(force=True)
    data = _rows_to_tensor(rows, L)
    return data, n_docs, n_toks


def split_heldout_windows(windows: torch.Tensor, n_val: int, min_train: int = 1):
    """Hold out a deterministic tail. The train prefix is not reshuffled.

    Returns the original tensor when the pool is too small to spare `n_val`
    windows and still leave `min_train` for the DistributedSampler.
    """
    n_val = int(n_val or 0)
    min_train = max(int(min_train or 0), 0)
    n = int(windows.shape[0]) if windows is not None and windows.ndim == 2 else 0
    if n_val <= 0 or n <= n_val + min_train:
        empty = windows.new_empty((0, windows.shape[1])) if windows is not None and windows.ndim == 2 else windows
        return windows, empty
    return windows[:-n_val], windows[-n_val:]


def corpus_cache_meta(args, L, max_windows, respect_sentences, pack_repeat, eos, pad):
    """Identity of a packed pool. A mismatch rebuilds artifacts/packed_windows.pt."""
    max_docs = getattr(args, "max_docs", None)
    return {
        "L": int(L),
        "max_windows": int(max_windows or 0),
        "respect_sentences": bool(respect_sentences),
        "pack_repeat": int(pack_repeat),
        "eos": int(eos),
        "pad": int(pad),
        "hf_dataset": str(getattr(args, "hf_dataset", None) or ""),
        "hf_split": str(getattr(args, "hf_split", None) or ""),
        "hf_text_field": str(getattr(args, "hf_text_field", None) or ""),
        "data": str(getattr(args, "data", None) or ""),
        "max_docs": int(max_docs) if max_docs else 0,
    }


def _cache_matches(saved, expect) -> bool:
    if not isinstance(saved, dict) or not isinstance(expect, dict):
        return False
    return all(saved.get(k) == expect.get(k) for k in expect)


def _atomic_torch_save(path: Path, payload):
    """Write via a temp file and os.replace so readers never see a partial cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# Longer than NCCL's default 10-minute watchdog. Sentence-packing 524288
# BookCorpus windows on one CPU exceeded that in Kaggle session v6.
PACK_WAIT_TIMEOUT_S = 6 * 3600.0


def _load_packed_cache(path: Path, expect):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"ignoring packed-window cache {path} ({exc!r})", flush=True)
        return None
    if not _cache_matches(payload.get("meta"), expect):
        return None
    windows = payload.get("windows")
    if not torch.is_tensor(windows) or windows.ndim != 2:
        return None
    return windows.to(dtype=torch.long), int(payload.get("n_docs", 0)), int(payload.get("n_toks", 0))


def _wait_for_packed_cache(cache_path, expect, error_path, timeout_s, poll_s):
    """Poll the cache file. This must not enter an NCCL collective.

    Kaggle v6 died here: rank 1 blocked in an NCCL barrier while rank 0 was
    still sentence-packing, and the watchdog aborted at 600s. Rank 0 had
    not entered that collective. A filesystem poll has no watchdog.
    """
    import time
    cache_path = Path(cache_path)
    timeout_s = float(timeout_s)
    poll_s = max(float(poll_s), 0.01)
    started = time.monotonic()
    start_wall = time.time()
    next_log = started
    while True:
        if error_path is not None and error_path.is_file():
            try:
                fresh = error_path.stat().st_mtime >= start_wall - 1.0
            except OSError:
                fresh = False
            if fresh:
                raise RuntimeError(error_path.read_text(encoding="utf-8") or f"pack failed: {error_path}")
        loaded = _load_packed_cache(cache_path, expect)
        if loaded is not None:
            windows, n_docs, n_toks = loaded
            print(
                f"loaded packed windows from {cache_path} "
                f"({len(windows)} windows, cap={expect.get('max_windows')})",
                flush=True,
            )
            return windows, n_docs, n_toks
        now = time.monotonic()
        if now - started >= timeout_s:
            raise TimeoutError(
                f"timed out after {timeout_s:.0f}s waiting for {cache_path} "
                "(rank 0 is still packing or the cache was not published)"
            )
        if now >= next_log:
            print(
                f"waiting for packed windows at {cache_path} "
                f"({now - started:.0f}s, no NCCL barrier)",
                flush=True,
            )
            next_log = now + 30.0
        time.sleep(poll_s)


def acquire_packed_windows(
    documents,
    pre_tokenizer,
    context_length,
    *,
    min_windows,
    repeat,
    respect_sentences,
    max_windows,
    cache_path,
    cache_meta,
    rank=0,
    world_size=1,
    wait_timeout_s=PACK_WAIT_TIMEOUT_S,
    wait_poll_s=2.0,
    progress_s=30.0,
):
    """Pack on rank 0, or reuse a matching cache, then load the same tensor everywhere.

    Only rank 0 packs, so two Kaggle processes do not each hold the Python
    pack buffer. The result is published with os.replace. Other ranks poll
    that file. They do not enter a collective while the pack runs, because
    a 524288-window sentence pack is longer than the default 10-minute
    watchdog (Kaggle session v6).

    A later session reuses artifacts/packed_windows.pt when `cache_meta`
    matches (cap, sentence packing, tokenizer eos/pad, corpus). A truncated
    or unreadable file is ignored and rebuilt. The sampler still sees one
    identical window tensor on every rank.
    """
    cache_path = Path(cache_path) if cache_path else None
    error_path = cache_path.with_name(cache_path.name + ".error") if cache_path is not None else None
    if cache_path is not None:
        loaded = _load_packed_cache(cache_path, cache_meta)
        if loaded is not None:
            windows, n_docs, n_toks = loaded
            print(
                f"loaded packed windows from {cache_path} "
                f"({len(windows)} windows, cap={cache_meta.get('max_windows')})",
                flush=True,
            )
            return windows, n_docs, n_toks, False

    if int(rank) != 0:
        if cache_path is None:
            raise RuntimeError("multi-rank packing requires a cache_path")
        windows, n_docs, n_toks = _wait_for_packed_cache(
            cache_path, cache_meta, error_path, wait_timeout_s, wait_poll_s
        )
        return windows, n_docs, n_toks, False

    if error_path is not None and error_path.exists():
        error_path.unlink()
    try:
        print(
            f"packing windows on rank 0 cap={int(max_windows or 0)} "
            f"L={int(context_length)} respect_sentences={bool(respect_sentences)} "
            "(other ranks poll the cache file; no NCCL barrier)",
            flush=True,
        )
        windows, n_docs, n_toks = pack_documents(
            documents,
            pre_tokenizer,
            context_length,
            min_windows=min_windows,
            repeat=repeat,
            respect_sentences=respect_sentences,
            max_windows=max_windows,
            progress_s=progress_s,
        )
        if cache_path is not None:
            _atomic_torch_save(
                cache_path,
                {
                    "windows": windows.to(dtype=torch.int32),
                    "n_docs": int(n_docs),
                    "n_toks": int(n_toks),
                    "meta": dict(cache_meta),
                },
            )
            # Refuse to publish a file we cannot read back. Peers only accept a match.
            check = _load_packed_cache(cache_path, cache_meta)
            if check is None:
                raise RuntimeError(f"packed-window cache failed verification: {cache_path}")
        return windows, n_docs, n_toks, True
    except Exception as exc:
        if error_path is not None:
            _atomic_write_text(error_path, repr(exc))
        raise


def default_workers(user_value=None, pin_memory=False):
    import os
    if user_value is not None:
        return int(user_value)
    if os.name == "nt":
        return 0
    return 2 if pin_memory else 0


def make_loader(
    windows: torch.Tensor,
    batch_size: int,
    seed: int = 42,
    drop_last: bool = False,
    pin_memory: bool = False,
    num_workers: int | None = None,
    rank: int = 0,
    world_size: int = 1,
):
    ds = PackedTokenDataset(windows)
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    workers = default_workers(num_workers, pin_memory=pin_memory)
    kwargs = dict(
        batch_size=batch_size,
        pin_memory=bool(pin_memory),
        num_workers=workers,
        persistent_workers=bool(workers > 0),
    )
    # With a small corpus, sample with replacement so every step has a full batch.
    if world_size <= 1 and len(ds) < batch_size:
        sampler = RandomSampler(ds, replacement=True, num_samples=batch_size * 64, generator=gen)
        return DataLoader(ds, sampler=sampler, **kwargs), sampler
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        # drop_last so every rank hits all_reduce the same number of times (VQ EMA).
        sampler = DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=True, seed=int(seed), drop_last=True
        )
        loader = DataLoader(ds, sampler=sampler, **kwargs)
        return loader, sampler
    loader = DataLoader(
        ds,
        shuffle=True,
        drop_last=drop_last and len(ds) >= batch_size,
        generator=gen,
        **kwargs,
    )
    return loader, None


def infinite_loader(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def iter_all_documents(args):
    """Yield documents from --data and/or --hf_dataset, capped by --max_docs."""
    max_docs = getattr(args, "max_docs", None)
    max_docs = int(max_docs) if max_docs else None
    n = 0

    def limited(it):
        nonlocal n
        for text in it:
            if max_docs is not None and n >= max_docs:
                return
            n += 1
            yield text

    if getattr(args, "data", None):
        yield from limited(iter_documents_from_file(args.data))
    if getattr(args, "hf_dataset", None):
        yield from limited(
            iter_documents_from_hf(
                args.hf_dataset,
                split=getattr(args, "hf_split", "train"),
                text_field=getattr(args, "hf_text_field", "text"),
                config_name=getattr(args, "hf_config", None),
                streaming=True,
            )
        )


def collect_documents(args):
    docs = list(iter_all_documents(args))
    if not docs:
        raise SystemExit("provide --data and/or --hf_dataset")
    return docs
