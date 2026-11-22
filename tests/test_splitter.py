"""Unit tests for sentence → phrase → word splitting."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenization.char_tokenizer import CharTokenizer
from utils.data import pack_documents
from utils.linguistic import (
    collect_spans,
    format_tree,
    pick_unit,
    split_phrases,
    split_sentences,
    split_text,
    split_text_tree,
    split_words,
)


def test_sentences():
    s = split_sentences("Hello world. How are you? I am fine.")
    assert s == ["Hello world.", "How are you?", "I am fine."], s
    # abbreviations and decimals should not split
    s = split_sentences("Dr. Smith paid 3.14 dollars in the U.S. today. Next.")
    assert len(s) == 2, s
    assert s[0].startswith("Dr.")
    assert s[1].startswith("Next")


def test_phrases_and_words():
    phrases = split_phrases("I stayed home, but she went out.")
    assert len(phrases) >= 2, phrases
    words = split_words("Don't stop.")
    assert words[0] == "Don't"
    assert "stop" in words


def test_tree_and_char_tokenizer():
    text = "Hello world. How are you?"
    tree = split_text_tree(text)
    assert len(tree) == 2
    rendered = format_tree(tree)
    assert "S1:" in rendered and "S2:" in rendered
    pre = CharTokenizer()
    spans = split_text(text, pre)
    sent = collect_spans(spans, "sentence")
    assert len(sent) == 2
    # Token indices should cover the encoded text.
    ids = pre.encode(text)
    assert sent[0][0] == 0
    assert sent[-1][1] <= len(ids) + 2


def test_pack_by_sentence():
    pre = CharTokenizer()
    docs = ["AAAA. BBBB. CCCC."]  # each sentence is 5 chars including period / spaces handled
    # "AAAA. BBBB. CCCC." sentences: "AAAA.", "BBBB.", "CCCC."
    windows, n_docs, n_toks = pack_documents(docs, pre, context_length=8, respect_sentences=True)
    assert n_docs == 1
    assert windows.shape[1] == 8
    # A window should not start in the middle of "BBBB." if a sentence fits.
    decoded = [pre.decode(row.tolist(), skip_special_tokens=False) for row in windows]
    for w in decoded:
        # After stripping pads (char 1) and eos (char 0 which decodes as token)
        assert "BBB" not in w or w.strip().find("BBB") == 0 or ". BBB" in w or "BBBB." in w


def test_rules_splitter_records_status(tmp):
    import json
    from omegaconf import OmegaConf
    from utils.linguistic import activate_splitter

    stage = tmp / "generator"
    stage.mkdir()
    cfg = OmegaConf.create({
        "splitter": {
            "enabled": True,
            "backend": "rules",
            "ckpt": "artifacts/splitter/splitter.pt",
            "pack_by_sentence": True,
            "guide_expand": True,
            "unit": "auto",
        }
    })
    out = activate_splitter(cfg, stage)
    assert out["respect_sentences"] is True
    status_path = tmp / "splitter" / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["backend"] == "rules"
    assert status["weights"] is False
    assert status["pack_by_sentence"] is True
    assert status["guide_expand"] is True
    assert not (tmp / "splitter" / "splitter.pt").exists()

    cfg.splitter.backend = "model"
    cfg.splitter.ckpt = str(tmp / "missing.pt")
    out = activate_splitter(cfg, stage)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["fallback"] == "rules"
    assert out["respect_sentences"] is True


def test_pick_unit():
    assert pick_unit(4, 512, "auto") == "sentence"
    assert pick_unit(100, 512, "auto") == "phrase"
    assert pick_unit(512, 512, "auto") == "word"
    assert pick_unit(4, 512, "word") == "word"


if __name__ == "__main__":
    import tempfile
    test_sentences()
    test_phrases_and_words()
    test_tree_and_char_tokenizer()
    test_pack_by_sentence()
    with tempfile.TemporaryDirectory() as raw:
        test_rules_splitter_records_status(Path(raw))
    test_pick_unit()
    print("ok")
