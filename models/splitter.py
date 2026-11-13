"""Tiny token-boundary tagger: predicts sentence / phrase / word starts.

The default pipeline uses rule-based splits (utils.linguistic). This model can
be trained from those labels with train_splitter.py and then used as backend.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryTagger(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden: int = 128, pad_id: int = 0):
        super().__init__()
        self.pad_id = int(pad_id)
        self.emb = nn.Embedding(int(vocab_size), int(emb_dim), padding_idx=self.pad_id)
        self.lstm = nn.LSTM(
            int(emb_dim), int(hidden), num_layers=1, batch_first=True, bidirectional=True
        )
        self.head = nn.Linear(int(hidden) * 2, 3)  # sentence, phrase, word starts

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(token_ids)
        h, _ = self.lstm(x)
        return self.head(h)

    @torch.no_grad()
    def predict_starts(self, token_ids: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Boolean tensor (B, L, 3) for sentence / phrase / word start flags."""
        prob = torch.sigmoid(self.forward(token_ids))
        flags = prob >= float(threshold)
        flags[:, 0, :] = True  # first token starts every unit
        return flags


def load_boundary_tagger(path, vocab_size, pad_id=0, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ckpt.get("config", {})
    model = BoundaryTagger(
        vocab_size=int(cfg.get("vocab_size", vocab_size)),
        emb_dim=int(cfg.get("emb_dim", 64)),
        hidden=int(cfg.get("hidden", 128)),
        pad_id=int(cfg.get("pad_id", pad_id)),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


def maybe_load_splitter_model(cfg, tokenizer, map_location="cpu"):
    """Return a BoundaryTagger if backend=model and the checkpoint exists."""
    raw = {}
    if cfg is not None:
        try:
            raw = dict(cfg.get("splitter", {}) or {})
        except Exception:
            raw = {}
    if str(raw.get("backend", "rules")) != "model":
        return None
    path = Path(str(raw.get("ckpt", "artifacts/splitter/splitter.pt")))
    if not path.is_file():
        return None
    vocab = int(getattr(tokenizer, "vocab_size", 50257))
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    return load_boundary_tagger(path, vocab, pad_id=pad_id, map_location=map_location)
