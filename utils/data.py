"""Text loading and packing into fixed-length token windows."""

from __future__ import annotations

import json
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


def iter_documents_from_hf(name, split="train", text_field="text", config_name=None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The optional 'datasets' package is required for --hf_dataset. "
            "Install it with: pip install datasets"
        ) from exc
    kwargs = {}
    if config_name:
        ds = load_dataset(name, config_name, split=split, **kwargs)
    else:
        ds = load_dataset(name, split=split, **kwargs)
    for row in ds:
        text = row[text_field] if text_field in row else row.get("text", "")
        if isinstance(text, str) and text.strip():
            yield text


def pack_documents(documents, pre_tokenizer, context_length, min_windows=0, repeat=1, respect_sentences=False):
    """Encode documents, append EOS, and pack/pad to length L.

    When respect_sentences is True, a new window starts at a sentence boundary
    whenever the next sentence would overflow the current window. A sentence
    longer than L is packed by tokens as usual.
    """
    eos = int(pre_tokenizer.eos_token_id)
    pad = int(pre_tokenizer.pad_token_id)
    L = int(context_length)
    n_docs = 0
    n_toks = 0
    windows = []

    def flush(cur):
        if not cur:
            return []
        if len(cur) < L:
            return cur + [pad] * (L - len(cur))
        return cur[:L]

    if respect_sentences:
        from utils.linguistic import split_sentences
        cur = []
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
                            windows.append(flush(cur))
                            cur = []
                        for i in range(0, len(ids), L):
                            chunk = ids[i:i + L]
                            if len(chunk) == L:
                                windows.append(chunk)
                            else:
                                cur = list(chunk)
                        continue
                    if cur and len(cur) + len(ids) > L:
                        windows.append(flush(cur))
                        cur = []
                    cur.extend(ids)
                if wrote:
                    n_docs += 1
                    if len(cur) >= L:
                        windows.append(cur[:L])
                        cur = cur[L:]
                    if len(cur) + 1 <= L:
                        cur.append(eos)
                        n_toks += 1
                    else:
                        windows.append(flush(cur))
                        cur = [eos]
                        n_toks += 1
        if cur:
            windows.append(flush(cur))
        if not windows:
            raise ValueError("no tokens produced from the given documents")
    else:
        stream = []
        for _ in range(max(int(repeat), 1)):
            for text in documents:
                ids = pre_tokenizer.encode(text)
                if not ids:
                    continue
                stream.extend(ids)
                stream.append(eos)
                n_docs += 1
        if not stream:
            raise ValueError("no tokens produced from the given documents")
        n_toks = len(stream)
        for i in range(0, len(stream), L):
            chunk = stream[i:i + L]
            if not chunk:
                continue
            if len(chunk) < L:
                chunk = chunk + [pad] * (L - len(chunk))
            windows.append(chunk)

    if min_windows and len(windows) < min_windows:
        reps = (min_windows + len(windows) - 1) // len(windows)
        windows = windows * reps
    data = torch.tensor(windows, dtype=torch.long)
    return data, n_docs, n_toks


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
    if len(ds) < batch_size:
        sampler = RandomSampler(ds, replacement=True, num_samples=batch_size * 64, generator=gen)
        return DataLoader(ds, sampler=sampler, **kwargs)
    return DataLoader(
        ds,
        shuffle=True,
        drop_last=drop_last and len(ds) >= batch_size,
        generator=gen,
        **kwargs,
    )


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch
