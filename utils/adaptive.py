"""Expand-only-if-needed helpers for coarse-to-fine NCM.

At generation time there is no ground-truth residual z, so we use the frozen
VQ decoder's confidence (and token stability) as a proxy: a span that is
already sharply decoded is treated like a uniform / "black background" region
and is not refined further. Skipped positions contribute a near-zero residual
(the codebook vector closest to 0).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def default_adaptive_cfg(raw: Any = None) -> dict:
    src = dict(raw) if raw is not None else {}
    return {
        "enabled": bool(src.get("enabled", False)),
        "early_stop": bool(src.get("early_stop", True)),
        "spatial": bool(src.get("spatial", True)),
        "min_levels": int(src.get("min_levels", 12)),
        "spatial_after": int(src.get("spatial_after", 8)),
        "token_conf_threshold": float(src.get("token_conf_threshold", 0.92)),
        "token_stable_threshold": float(src.get("token_stable_threshold", 0.98)),
        "spatial_conf_threshold": float(src.get("spatial_conf_threshold", 0.97)),
        "min_expand_frac": float(src.get("min_expand_frac", 0.25)),
    }


def nearest_zero_indices(vq) -> list[int]:
    """Codebook index whose embedding is closest to 0, one per level."""
    out = []
    for emb in vq.embeddings:
        dist = emb.weight.detach().float().pow(2).sum(dim=-1)
        out.append(int(dist.argmin().item()))
    return out


def pool_to_cells(values: torch.Tensor, n_cells: int) -> torch.Tensor:
    """Pool a length-L map (B, L) onto n_cells with area averaging."""
    if values.shape[-1] == n_cells:
        return values
    return F.interpolate(values.unsqueeze(1).float(), size=int(n_cells), mode="area").squeeze(1)


def decode_confidence(vqvae, z_hat: torch.Tensor):
    """Return (max-prob confidence, argmax tokens) from the VQ decoder."""
    logits = vqvae.decoder(z_hat)
    conf, tokens = logits.softmax(dim=-1).max(dim=-1)
    return conf, tokens


def sample_done_mask(conf: torch.Tensor, tokens: torch.Tensor, prev_tokens, min_conf: float, min_stable: float):
    """Per-sample flag: this sequence is already sharp and stable."""
    sharp = conf.mean(dim=-1) >= float(min_conf)
    if prev_tokens is None or prev_tokens.shape != tokens.shape:
        return sharp & False  # need at least one comparison after min_levels
    stable = (tokens == prev_tokens).float().mean(dim=-1) >= float(min_stable)
    return sharp & stable


def spatial_expand_mask(cell_conf: torch.Tensor, conf_threshold: float, min_expand_frac: float) -> torch.Tensor:
    """True where a cell should still be expanded. Always keeps the most uncertain fraction."""
    B, N = cell_conf.shape
    need = cell_conf < float(conf_threshold)
    keep = max(1, int(round(float(min_expand_frac) * N)))
    keep = min(keep, N)
    # Lowest confidence = most uncertain = must expand
    kth = torch.topk(cell_conf, k=keep, dim=-1, largest=False).values[:, -1]
    forced = cell_conf <= kth.unsqueeze(-1)
    return need | forced


def apply_spatial_skip(codes: torch.Tensor, expand: torch.Tensor, zero_code: int) -> torch.Tensor:
    """Replace codes at positions that should not expand with the near-zero code."""
    if expand.dtype != torch.bool:
        expand = expand.bool()
    skipped = codes.clone()
    skipped[~expand] = int(zero_code)
    return skipped
