"""Shared helpers for NCM two-stage training and generation."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from models.next_concept import NextConceptTransformer
from models.vqvae import VQVAE
from utils.misc import count_params
from utils.registry import tokenizers
import hydra


def set_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_cuda(device: torch.device) -> None:
    """Enable TF32 / cuDNN autotune and print GPU info. No-op on CPU."""
    if device.type != "cuda":
        print(f"device={device}  (CUDA not available; training will use CPU)")
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    idx = device.index if device.index is not None else 0
    props = torch.cuda.get_device_properties(idx)
    vram_gb = props.total_memory / (1024 ** 3)
    cap = torch.cuda.get_device_capability(idx)
    print(
        f"device={device}  gpu={props.name}  cap={cap[0]}.{cap[1]}  "
        f"vram={vram_gb:.1f}GiB  cuda={torch.version.cuda}  torch={torch.__version__}"
    )


def setup_amp(device: torch.device, amp_flag: bool | None):
    """Return (autocast_ctx, grad_scaler_or_None, dtype_name).

    amp_flag is None (auto: on if CUDA), True, or False.
    Prefer bf16; fall back to fp16 + GradScaler when the GPU has no bf16.
    """
    from contextlib import nullcontext

    want = (device.type == "cuda") if amp_flag is None else bool(amp_flag)
    if not want or device.type != "cuda":
        if want and device.type != "cuda":
            print("AMP requested but CUDA is unavailable; running fp32 on CPU")
        return nullcontext(), None, "fp32"
    if torch.cuda.is_bf16_supported():
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return ctx, None, "bf16"
    scaler = torch.amp.GradScaler("cuda")
    ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    return ctx, scaler, "fp16"


def optimize(loss, opt, params, clip, scaler=None):
    """Backward, clip, and step, with or without a GradScaler."""
    if scaler is None:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        return
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(params, clip)
    scaler.step(opt)
    scaler.update()


def resolve_compile(device: torch.device, compile_flag: bool | None) -> bool:
    """None means auto: enable torch.compile on CUDA."""
    return (device.type == "cuda") if compile_flag is None else bool(compile_flag)


def maybe_compile(module, enabled: bool, name: str = "model"):
    if not enabled:
        return module
    try:
        compiled = torch.compile(module)
        print(f"torch.compile enabled for {name}")
        return compiled
    except Exception as exc:
        print(f"torch.compile failed for {name} ({exc!r}); continuing without it")
        return module


def add_device_args(parser):
    """Shared CUDA flags for train / generate scripts."""
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu (default: cuda if available)")
    parser.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        default=None,
        help="mixed precision (default: on when CUDA is available)",
    )
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="disable mixed precision")
    parser.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        default=None,
        help="torch.compile (default: on when CUDA is available)",
    )
    parser.add_argument("--no-compile", dest="compile", action="store_false", help="disable torch.compile")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="DataLoader workers (default: 0 on Windows, 2 on CUDA elsewhere)",
    )
    return parser


def build_pre_tokenizer(cfg):
    """Build the surface (pre-)tokenizer from config. Smoke uses a char vocab (no network)."""
    name = str(cfg.tokenizer.pre_tokenizer.name)
    cls = hydra.utils.get_class(tokenizers[name])
    return cls.from_config(cfg.tokenizer.pre_tokenizer)


def build_vqvae(cfg, device=None) -> VQVAE:
    """Instantiate a VQ-VAE from config without loading a checkpoint."""
    vqvae = VQVAE(**cfg.tokenizer.vqvae)
    if device is not None:
        vqvae = vqvae.to(device)
    return vqvae


def load_vqvae(cfg, checkpoint_path, device, frozen=False) -> VQVAE:
    vqvae = build_vqvae(cfg, device=device)
    sd = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    vqvae.load_state_dict(state, strict=True)
    if frozen:
        vqvae.eval()
        for p in vqvae.parameters():
            p.requires_grad_(False)
    return vqvae


def build_generator(vqvae, cfg, device=None) -> NextConceptTransformer:
    net = NextConceptTransformer(vqvae_local=vqvae, config=cfg.model)
    if device is not None:
        net = net.to(device)
    return net


def actual_prefix_pad_id(cfg, pre_tokenizer):
    """Prefer the real tokenizer pad id; fall back to the config value."""
    tok_id = getattr(pre_tokenizer, "pad_token_id", None)
    if tok_id is not None:
        return int(tok_id)
    raw = getattr(cfg.model, "prefix_pad_token_id", None)
    return int(raw) if raw is not None else None


def lr_at(step: int, warmup: int, peak: float, floor: float, max_steps: int) -> float:
    """Linear warmup to peak, then cosine decay down to floor."""
    step = max(int(step), 0)
    warmup = max(int(warmup), 0)
    if warmup > 0 and step < warmup:
        return float(peak) * (step + 1) / warmup
    if max_steps <= warmup:
        return float(peak)
    t = (step - warmup) / max(max_steps - warmup, 1)
    t = min(max(t, 0.0), 1.0)
    return float(floor) + 0.5 * (1.0 + math.cos(math.pi * t)) * (float(peak) - float(floor))


def set_lr(optimizer, lr: float):
    for group in optimizer.param_groups:
        group["lr"] = lr


def level_weights(levels, alpha: float, device=None, dtype=torch.float32):
    """w_k = ell_k^{-alpha} / sum_j ell_j^{1-alpha} (paper ELBO weighting)."""
    ell = torch.tensor([float(x) for x in levels], dtype=dtype, device=device)
    denom = ell.pow(1.0 - float(alpha)).sum().clamp(min=1e-12)
    return ell.pow(-float(alpha)) / denom


def generator_ce_loss(logits, targets, levels, vocab_sizes, alpha: float = 0.0):
    """Per-position CE, weighted by w_k, sliced to each level's true vocab size.

    When alpha=0 this equals the mean CE over all predicted code positions.
    logits:  (B, sum(levels), max_vocab)
    targets: (B, sum(levels))  -- clean codes for levels[1:] when level 0 is not predicted
    """
    B = targets.shape[0]
    device = logits.device
    w = level_weights(levels, alpha, device=device, dtype=torch.float32)
    total = logits.new_zeros(())
    offset = 0
    for k, (ell, vk) in enumerate(zip(levels, vocab_sizes)):
        ell, vk = int(ell), int(vk)
        lg = logits[:, offset : offset + ell, :vk].reshape(-1, vk).float()
        tg = targets[:, offset : offset + ell].reshape(-1)
        ce = F.cross_entropy(lg, tg, reduction="none")
        total = total + (ce * w[k]).sum()
        offset += ell
    return total / B


def corrupt_codes(idx, levels, vocab_sizes, p_max=0.8, p_min=0.2):
    """Replace a fraction of INPUT codes uniformly from the same level's codebook.

    Rate decreases linearly from p_max at level 0 to p_min at the last level
    (mean 0.50 when p_max=0.8, p_min=0.2). Labels are not modified.
    """
    out = idx.clone()
    K = len(levels)
    offset = 0
    for k, (ell, vk) in enumerate(zip(levels, vocab_sizes)):
        ell, vk = int(ell), int(vk)
        p = float(p_max) if K <= 1 else float(p_max) - (float(p_max) - float(p_min)) * k / (K - 1)
        chunk = out[:, offset : offset + ell]
        mask = torch.rand(chunk.shape, device=idx.device) < p
        noise = torch.randint(0, vk, chunk.shape, device=idx.device)
        out[:, offset : offset + ell] = torch.where(mask, noise, chunk)
        offset += ell
    return out


def build_prefix_ids(batch_size, eos_id, prefix_len=1, device=None):
    ids = torch.full((batch_size, int(prefix_len)), int(eos_id), device=device, dtype=torch.long)
    return ids


def prefix_latents(vqvae, prefix_ids, raw_token_prefix: bool):
    if raw_token_prefix:
        return vqvae.encoder.wte(prefix_ids)
    return vqvae.encode_z(prefix_ids)[:, : prefix_ids.shape[1], :]


def tokenizer_recon_accuracy(logits, tokens, pad_id):
    pred = logits.argmax(dim=-1)
    mask = tokens != int(pad_id)
    if mask.any():
        return (pred[mask] == tokens[mask]).float().mean().item()
    return 0.0


def param_counts(model, exclude=None):
    exclude = exclude or []
    total, non_embd = count_params(model, exclude_params=exclude)
    return int(total), int(non_embd)


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def maybe_copy_config(cfg, dest_dir):
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, dest / "config.yaml")
