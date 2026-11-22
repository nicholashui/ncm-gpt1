"""Sentence → phrase → word splitter for concept-aware NCM.

NCM itself still uses fixed square levels. This module splits text first, so
training windows can start at sentence boundaries and generation can expand
cells only where the overlapping sentence, phrase, or word is still uncertain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import torch

# Titles / abbreviations that should not end a sentence at the period.
_ABBREV = frozenset(
    """
    mr mrs ms mz mx dr prof sr jr st vs etc inc ltd co corp univ assn
    jan feb mar apr jun jul aug sep sept oct nov dec
    no vol fig pp est approx dept
    """.split()
)
_ABBREV_DOT = frozenset({"e.g", "i.e", "u.s", "u.k", "u.s.a", "a.m", "p.m"})

_SENT_END = set(".?!。！？…")
_SENT_TRAIL = _SENT_END | set("\"\'”’)」")
_CONJ = re.compile(
    r"\s+(?:and|but|or|so|yet|nor|because|although|though|while|whereas|"
    r"moreover|however|therefore|plus|而且|但是|然後|或者|或是|因為|所以|雖然)\b",
    re.IGNORECASE,
)
_PHRASE_PUNCT = re.compile(r"[,;:，；：]\s+")
_WORD = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*|[^\s]")


@dataclass
class Span:
    kind: str
    start: int  # token index, inclusive
    end: int  # token index, exclusive
    text: str = ""
    char_start: int = 0
    char_end: int = 0
    children: list["Span"] = field(default_factory=list)

    def as_tuple(self) -> tuple[int, int]:
        return (int(self.start), int(self.end))


def _is_abbrev_period(text: str, i: int) -> bool:
    if i == 0 or text[i] != ".":
        return False
    # Decimal: 3.14 or .5 already handled if prev is digit and next is digit.
    if i + 1 < len(text) and text[i - 1].isdigit() and text[i + 1].isdigit():
        return True
    # Initials: "A." or "U.S."
    if text[i - 1].isalpha() and (i == 1 or not text[i - 2].isalpha()):
        if i + 1 < len(text) and (text[i + 1].isalpha() or text[i + 1] == "."):
            return True
        # Single-letter initial at end of a name often followed by space + capital — still a word, not a sentence,
        # unless the next word is clearly a new sentence of length... keep conservative: single letter + period is abbrev
        # if next is space+lowercase or another initial.
        if i + 1 < len(text) and text[i + 1] in " \t" and i + 2 < len(text) and text[i + 2].islower():
            return True
        if text[i - 1].isupper() and (i == 1 or text[i - 2] in " \t("):
            return True
    j = i - 1
    while j >= 0 and text[j].isalpha():
        j -= 1
    word = text[j + 1 : i].lower()
    if word in _ABBREV:
        return True
    # e.g. / i.e. / u.s.
    k = max(0, i - 5)
    tail = re.sub(r"\s+", "", text[k : i + 1]).lower().rstrip(".")
    return tail in _ABBREV_DOT or tail.rstrip(".") in _ABBREV_DOT


def split_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    parts: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\n" and i + 1 < n and text[i + 1] == "\n":
            chunk = text[start:i].strip()
            if chunk:
                parts.append(chunk)
            i += 2
            while i < n and text[i] in " \t\n":
                i += 1
            start = i
            continue
        if ch in _SENT_END:
            if _is_abbrev_period(text, i):
                i += 1
                continue
            j = i + 1
            while j < n and text[j] in _SENT_TRAIL:
                j += 1
            while j < n and text[j] in " \t":
                j += 1
            chunk = text[start:j].strip()
            if chunk:
                parts.append(chunk)
            start = j
            i = j
            continue
        i += 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def split_phrases(sentence: str) -> list[str]:
    sentence = sentence.strip()
    if not sentence:
        return []
    cuts = [0]
    for m in _PHRASE_PUNCT.finditer(sentence):
        pos = m.end()
        if 0 < pos < len(sentence):
            cuts.append(pos)
    for m in _CONJ.finditer(sentence):
        pos = m.start() + 1  # keep the spaces with previous; new phrase starts at conjunction word
        # m.start() is the leading whitespace; phrase starts after it
        pos = m.start() + len(m.group()) - len(m.group().lstrip())
        if 0 < pos < len(sentence):
            cuts.append(pos)
    cuts = sorted(set(cuts))
    cuts.append(len(sentence))
    parts = [sentence[a:b].strip() for a, b in zip(cuts, cuts[1:])]
    return [p for p in parts if p] or [sentence]


def split_words(text: str) -> list[str]:
    return [m.group(0) for m in _WORD.finditer(text)]


def _locate(text: str, pieces: list[str], origin: int = 0) -> list[tuple[int, int, str]]:
    out = []
    cursor = 0
    for p in pieces:
        if not p:
            continue
        i = text.find(p, cursor)
        if i < 0:
            i = cursor
            j = min(len(text), cursor + max(len(p), 1))
        else:
            j = i + len(p)
        out.append((origin + i, origin + j, p))
        cursor = j
    return out


def split_text_tree(text: str) -> list[dict[str, Any]]:
    """Hierarchical split of a raw string (char offsets)."""
    tree = []
    for s_ch in _locate(text, split_sentences(text)):
        sent = {"kind": "sentence", "char_start": s_ch[0], "char_end": s_ch[1], "text": s_ch[2], "phrases": []}
        for p_ch in _locate(text[s_ch[0] : s_ch[1]], split_phrases(s_ch[2]), origin=s_ch[0]):
            phrase = {"kind": "phrase", "char_start": p_ch[0], "char_end": p_ch[1], "text": p_ch[2], "words": []}
            for w_ch in _locate(text[p_ch[0] : p_ch[1]], split_words(p_ch[2]), origin=p_ch[0]):
                phrase["words"].append(
                    {"kind": "word", "char_start": w_ch[0], "char_end": w_ch[1], "text": w_ch[2]}
                )
            sent["phrases"].append(phrase)
        tree.append(sent)
    return tree


def format_tree(tree: list[dict[str, Any]]) -> str:
    lines = []
    for i, sent in enumerate(tree, 1):
        lines.append(f"S{i}: {sent['text']}")
        for j, phrase in enumerate(sent.get("phrases", []), 1):
            words = " | ".join(w["text"] for w in phrase.get("words", []))
            lines.append(f"  P{i}.{j}: {phrase['text']}")
            if words:
                lines.append(f"    W: {words}")
    return "\n".join(lines)


def token_char_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Return (ids, [(char_start, char_end), ...]) aligned to `text`."""
    if hasattr(tokenizer, "hf_tokenizer"):
        hf = tokenizer.hf_tokenizer
        try:
            enc = hf(text, return_offsets_mapping=True, add_special_tokens=False)
            ids = list(enc["input_ids"])
            offs = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
            if ids and offs and len(ids) == len(offs):
                return ids, offs
        except Exception:
            pass
    ids = list(tokenizer.encode(text) or [])
    # Character tokenizer (or any 1-id-per-char encoding).
    if len(ids) == len(text):
        return ids, [(i, i + 1) for i in range(len(text))]
    # Prefix-decode fallback.
    offs = []
    prev = ""
    for i in range(len(ids)):
        try:
            cur = tokenizer.decode(ids[: i + 1], skip_special_tokens=False)
        except TypeError:
            cur = tokenizer.decode(ids[: i + 1])
        start = len(prev)
        # If decode added a leading space not in text, search from start.
        if start > len(text):
            start = len(text)
        end = min(len(text), max(start, len(cur)))
        offs.append((start, end))
        prev = cur
    return ids, offs


def _span_for_chars(char_start: int, char_end: int, offs: list[tuple[int, int]], n_tok: int) -> tuple[int, int]:
    start_t, end_t = None, None
    for i, (a, b) in enumerate(offs):
        if b <= char_start or a >= char_end:
            continue
        if start_t is None:
            start_t = i
        end_t = i + 1
    if start_t is None:
        return (min(n_tok, 0), min(n_tok, 0))
    return (start_t, end_t if end_t is not None else start_t)


def tree_to_token_spans(tree: list[dict[str, Any]], offs: list[tuple[int, int]]) -> list[Span]:
    n_tok = len(offs)
    sentences: list[Span] = []
    for sent in tree:
        s0, s1 = _span_for_chars(sent["char_start"], sent["char_end"], offs, n_tok)
        s_span = Span("sentence", s0, s1, sent["text"], sent["char_start"], sent["char_end"])
        for phrase in sent.get("phrases", []):
            p0, p1 = _span_for_chars(phrase["char_start"], phrase["char_end"], offs, n_tok)
            p_span = Span("phrase", p0, p1, phrase["text"], phrase["char_start"], phrase["char_end"])
            for word in phrase.get("words", []):
                w0, w1 = _span_for_chars(word["char_start"], word["char_end"], offs, n_tok)
                p_span.children.append(Span("word", w0, w1, word["text"], word["char_start"], word["char_end"]))
            s_span.children.append(p_span)
        sentences.append(s_span)
    return sentences


def split_text(text: str, tokenizer=None) -> list[Span]:
    tree = split_text_tree(text or "")
    if tokenizer is None:
        # Char offsets reused as "token" indexes when no tokenizer is given.
        def as_char(nodes, kind):
            return [
                Span(kind, n["char_start"], n["char_end"], n["text"], n["char_start"], n["char_end"])
                for n in nodes
            ]
        # Build a char-index tree
        out = []
        for sent in tree:
            s = Span("sentence", sent["char_start"], sent["char_end"], sent["text"], sent["char_start"], sent["char_end"])
            for phrase in sent.get("phrases", []):
                p = Span("phrase", phrase["char_start"], phrase["char_end"], phrase["text"], phrase["char_start"], phrase["char_end"])
                for word in phrase.get("words", []):
                    p.children.append(
                        Span("word", word["char_start"], word["char_end"], word["text"], word["char_start"], word["char_end"])
                    )
                s.children.append(p)
            out.append(s)
        return out
    _, offs = token_char_offsets(tokenizer, text)
    return tree_to_token_spans(tree, offs)


def decode_ids(tokenizer, ids: Iterable[int], skip_special: bool = False) -> str:
    ids = [int(x) for x in ids]
    try:
        return tokenizer.decode(ids, skip_special_tokens=skip_special)
    except TypeError:
        return tokenizer.decode(ids)


def split_token_ids(ids: list[int], tokenizer, eos_id: Optional[int] = None, pad_id: Optional[int] = None) -> list[Span]:
    """Split a generated / packed token window by sentence → phrase → word."""
    ids = [int(x) for x in ids]
    cut = len(ids)
    if eos_id is not None and eos_id in ids:
        cut = ids.index(eos_id)
    useful = []
    for i, t in enumerate(ids[:cut]):
        if pad_id is not None and t == pad_id:
            continue
        useful.append((i, t))
    if not useful:
        return []
    sub_ids = [t for _, t in useful]
    text = decode_ids(tokenizer, sub_ids, skip_special=False)
    tree = split_text_tree(text)
    # Offsets are relative to `text` / `sub_ids`, remap to original window indices.
    _, offs = token_char_offsets(tokenizer, text)
    # If encode(text) does not match sub_ids (decode/encode not identity), map by char onto sub_ids length.
    if len(offs) != len(sub_ids):
        offs = []
        n = max(len(text), 1)
        for k in range(len(sub_ids)):
            a = int(round(k * n / len(sub_ids)))
            b = int(round((k + 1) * n / len(sub_ids)))
            offs.append((a, max(b, a + 1)))
    spans = tree_to_token_spans(tree, offs)
    index_map = [i for i, _ in useful]

    def remap(span: Span) -> Span:
        if span.start >= len(index_map):
            s = index_map[-1] if index_map else 0
            e = s
        else:
            s = index_map[span.start]
            e = index_map[min(span.end, len(index_map)) - 1] + 1 if span.end > span.start else s
        child = Span(span.kind, s, e, span.text, span.char_start, span.char_end)
        child.children = [remap(c) for c in span.children]
        return child

    return [remap(s) for s in spans]


def collect_spans(sentences: list[Span], kind: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if kind == "sentence":
        out = [s.as_tuple() for s in sentences if s.end > s.start]
    elif kind == "phrase":
        for s in sentences:
            out.extend(p.as_tuple() for p in s.children if p.end > p.start)
    else:
        for s in sentences:
            for p in s.children:
                out.extend(w.as_tuple() for w in p.children if w.end > w.start)
    return out or ([(sentences[0].start, sentences[-1].end)] if sentences else [])


def pick_unit(n_cells: int, context_length: int, unit: str = "auto") -> str:
    if unit and unit != "auto":
        return str(unit)
    r = float(n_cells) / max(int(context_length), 1)
    if r <= 0.08:
        return "sentence"
    if r <= 0.35:
        return "phrase"
    return "word"


def map_span_expand_to_cells(
    span_expand: torch.Tensor,
    spans: list[tuple[int, int]],
    n_cells: int,
    length: int,
) -> torch.Tensor:
    """A cell expands if it overlaps any expanding linguistic span."""
    device = span_expand.device
    out = torch.zeros(int(n_cells), dtype=torch.bool, device=device)
    L = max(int(length), 1)
    N = int(n_cells)
    for k, (s, e) in enumerate(spans):
        if k >= span_expand.numel() or not bool(span_expand.view(-1)[k]):
            continue
        c0 = int(s * N // L)
        c1 = int((max(e, s + 1) * N + L - 1) // L)
        c0 = max(0, min(N - 1, c0))
        c1 = max(c0 + 1, min(N, c1))
        out[c0:c1] = True
    return out


def expand_mask_from_units(
    conf: torch.Tensor,
    token_ids: torch.Tensor,
    tokenizer,
    n_cells: int,
    unit: str,
    conf_threshold: float,
    min_expand_frac: float,
    eos_id: Optional[int] = None,
    pad_id: Optional[int] = None,
) -> tuple[torch.Tensor, list[str]]:
    """Per-sample linguistic expand mask mapped onto NCM cells. Also returns the unit used."""
    from utils.adaptive import spatial_expand_mask

    B, L = conf.shape
    masks = []
    units = []
    kind = pick_unit(int(n_cells), int(L), unit)
    for b in range(B):
        ids = token_ids[b].detach().tolist()
        try:
            sentences = split_token_ids(ids, tokenizer, eos_id=eos_id, pad_id=pad_id)
            spans = collect_spans(sentences, kind)
        except Exception:
            spans = [(0, L)]
        if not spans:
            spans = [(0, L)]
        cell_conf = conf.new_zeros(1, len(spans))
        for i, (s, e) in enumerate(spans):
            s = max(0, min(L, int(s)))
            e = max(s + 1, min(L, int(e)))
            cell_conf[0, i] = conf[b, s:e].mean()
        span_expand = spatial_expand_mask(cell_conf, conf_threshold, min_expand_frac)[0]
        masks.append(map_span_expand_to_cells(span_expand, spans, int(n_cells), L))
        units.append(kind)
    return torch.stack(masks, dim=0), units


def default_splitter_cfg(raw: Any = None) -> dict:
    src = dict(raw) if raw is not None else {}
    return {
        "enabled": bool(src.get("enabled", False)),
        "backend": str(src.get("backend", "rules")),
        "ckpt": src.get("ckpt", "artifacts/splitter/splitter.pt"),
        "pack_by_sentence": bool(src.get("pack_by_sentence", True)),
        "guide_expand": bool(src.get("guide_expand", True)),
        "unit": str(src.get("unit", "auto")),
    }


def pack_kwargs_from_cfg(cfg) -> dict:
    raw = {}
    if cfg is not None:
        try:
            raw = dict(cfg.get("splitter", {}) or {})
        except Exception:
            raw = {}
    sc = default_splitter_cfg(raw)
    return {"respect_sentences": bool(sc["enabled"] and sc["pack_by_sentence"])}


def _splitter_raw(cfg) -> dict:
    raw = {}
    if cfg is None:
        return raw
    try:
        got = cfg.get("splitter", {}) or {}
        raw = dict(got)
    except Exception:
        raw = {}
    return raw


def _env_rank0() -> bool:
    import os
    return os.environ.get("RANK", "0") in ("0", "")


def activate_splitter(cfg, stage_output_dir, pre_tokenizer=None) -> dict:
    """Turn the configured splitter on and leave a record under artifacts/splitter.

    The Kaggle config uses backend=rules. That backend has no learned weights,
    so a previous run wrote no artifacts/splitter/ tree and looked like a
    no-op. Sentence packing still has to run, and guide_expand still has to
    be recorded for generation. A model backend loads and copies its
    checkpoint; a missing file is logged and falls back to the same rules.
    """
    import json
    import shutil
    from pathlib import Path

    sc = default_splitter_cfg(_splitter_raw(cfg))
    dest = Path(stage_output_dir).resolve().parent / "splitter"
    pack = bool(sc["enabled"] and sc["pack_by_sentence"])
    guide = bool(sc["enabled"] and sc["guide_expand"])
    status = {
        "enabled": bool(sc["enabled"]),
        "backend": sc["backend"],
        "pack_by_sentence": pack,
        "guide_expand": guide,
        "unit": sc["unit"],
        "weights": False,
        "ckpt": str(sc["ckpt"]),
    }
    messages = []
    if not sc["enabled"]:
        messages.append("splitter disabled; pack_by_sentence and guide_expand are off")
    elif sc["backend"] == "rules":
        messages.append(
            "splitter backend=rules has no learned weights; "
            f"applying pack_by_sentence={pack} and guide_expand={guide} "
            "(rules are not a no-op)"
        )
    elif sc["backend"] == "model":
        src = Path(str(sc["ckpt"]))
        if not src.is_file():
            status["fallback"] = "rules"
            messages.append(
                f"splitter backend=model checkpoint missing at {src}; "
                f"falling back to rules and still applying pack_by_sentence={pack} "
                f"guide_expand={guide}"
            )
        else:
            try:
                from models.splitter import load_boundary_tagger
                vocab = int(getattr(pre_tokenizer, "vocab_size", 50257) or 50257)
                pad_id = int(getattr(pre_tokenizer, "pad_token_id", 0) or 0)
                tagger = load_boundary_tagger(src, vocab, pad_id=pad_id)
                status["weights"] = True
                status["loaded_ckpt"] = str(src.resolve())
                status["n_params"] = int(sum(p.numel() for p in tagger.parameters()))
                del tagger
                if _env_rank0():
                    dest.mkdir(parents=True, exist_ok=True)
                    target = dest / "splitter.pt"
                    if src.resolve() != target.resolve():
                        shutil.copy2(src, target)
                    status["saved_ckpt"] = str(target)
                messages.append(
                    f"splitter backend=model loaded {status['n_params']} parameters from {src}; "
                    f"pack_by_sentence={pack} guide_expand={guide}"
                )
            except Exception as exc:
                status["weights"] = False
                status["fallback"] = "rules"
                status["error"] = repr(exc)
                messages.append(
                    f"splitter backend=model failed to load {src} ({exc!r}); "
                    f"falling back to rules and still applying pack_by_sentence={pack} "
                    f"guide_expand={guide}"
                )
    else:
        status["fallback"] = "rules"
        messages.append(
            f"splitter backend={sc['backend']!r} is unknown; "
            f"using rules with pack_by_sentence={pack} guide_expand={guide}"
        )
    if _env_rank0():
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        for line in messages:
            print(line, flush=True)
    return {"respect_sentences": pack, "splitter_status": status}
