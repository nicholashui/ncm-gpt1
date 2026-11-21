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

# Forwarders so maybe_compile can log on the main rank only. Real helpers are at the bottom.
def is_main_process() -> bool:
    try:
        import torch.distributed as dist
        return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        return True


def dist_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)



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
    # T4 (sm_75) reports is_bf16_supported() on some PyTorch builds but has no
    # native bf16 Tensor Cores; emulated bf16 wastes memory. Require sm >= 80.
    cap = torch.cuda.get_device_capability(device)
    if cap[0] >= 8 and torch.cuda.is_bf16_supported():
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return ctx, None, "bf16"
    scaler = torch.amp.GradScaler("cuda")
    ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    return ctx, scaler, "fp16"


def optimize(loss, opt, params, clip, scaler=None):
    """Backward, clip, and step, with or without a GradScaler.

    Returns the total grad norm from before clipping (a Python float).
    """
    if scaler is None:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        return float(grad_norm)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    grad_norm = torch.nn.utils.clip_grad_norm_(params, clip)
    scaler.step(opt)
    scaler.update()
    return float(grad_norm)


def rank0_barrier_call(fn):
    """Run `fn` on rank 0 between collectives so a val pass cannot deadlock DDP.

    Every rank enters and leaves together. A failure on rank 0 is broadcast
    as a flag, then re-raised, so the other ranks do not wait at the next
    training all_reduce.
    """
    world = get_world_size()
    if world > 1:
        import torch.distributed as dist
        dist.barrier()
    result = None
    error = None
    if is_main_process():
        try:
            result = fn()
        except Exception as exc:
            error = exc
    if world > 1:
        import torch.distributed as dist
        if torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device("cpu")
        flag = torch.tensor([0 if error is None else 1], device=device, dtype=torch.int32)
        dist.broadcast(flag, src=0)
        if int(flag.item()) != 0:
            if error is not None:
                raise error
            raise RuntimeError("rank 0 failed inside rank0_barrier_call")
    elif error is not None:
        raise error
    return result


def resolve_compile(device: torch.device, compile_flag: bool | None) -> bool:
    """None means auto: enable torch.compile on CUDA."""
    return (device.type == "cuda") if compile_flag is None else bool(compile_flag)


def maybe_compile(module, enabled: bool, name: str = "model"):
    if not enabled:
        return module
    try:
        compiled = torch.compile(module)
        dist_print(f"torch.compile enabled for {name}")
        return compiled
    except Exception as exc:
        dist_print(f"torch.compile failed for {name} ({exc!r}); continuing without it")
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
    add_runtime_args(parser)
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
    """w_k = ell_k^{-alpha} / sum_j ell_j^{1-alpha}.

    These weights satisfy sum_k ell_k * w_k = 1, so the generator loss is a
    weighted mean of per-position cross-entropy.

    alpha=0 is the paper ELBO: every code position has the same weight, and
    long fine levels dominate because they have more cells. alpha>0 departs
    from that ELBO and moves mass onto shorter levels. On the 22 predicted
    Kaggle levels, alpha=0.5 still leaves the central third (the high-CE
    band) with about the same share as alpha=0; further alpha mostly feeds
    the shortest levels. That central band is handled by mid_aux in
    generator_ce_loss. Other configs keep alpha=0. The value stays
    configurable; it is not hard-coded in the loss.
    """
    ell = torch.tensor([float(x) for x in levels], dtype=dtype, device=device)
    denom = ell.pow(1.0 - float(alpha)).sum().clamp(min=1e-12)
    return ell.pow(-float(alpha)) / denom


def mid_level_slice(n_levels: int) -> tuple[int, int]:
    """Half-open index range of the central third of predicted levels.

    For the 22 Kaggle predicted lengths this is indices [7, 14): lengths
    81 through 225. Official v7 (alpha=0.5) median CE on that band was about
    5.2–6.1, against about 4.5 on the shortest predicted level and 1.5 on
    the 512-cell level. Raising alpha further does not aim at this band.
    """
    n = int(n_levels)
    if n <= 0:
        return 0, 0
    if n < 3:
        return 0, n
    start = n // 3
    end = (2 * n) // 3
    if end <= start:
        end = min(n, start + 1)
    return start, end


def generator_ce_loss(
    logits,
    targets,
    levels,
    vocab_sizes,
    alpha: float = 0.0,
    mid_aux: float = 0.0,
    return_parts=False,
):
    """Per-position CE, weighted by w_k, sliced to each level's true vocab size.

    When alpha=0 and mid_aux=0 this equals the mean CE over all predicted
    code positions (the paper ELBO). alpha!=0 is a documented departure; see
    level_weights. mid_aux>0 adds mid_aux times the mean of the central-third
    level CEs. That term changes the scalar loss, so a run with mid_aux must
    not be compared to alpha-only curves (v5 alpha=0, v7/v8 alpha=0.5).
    Per-level CE parts stay the raw means either way.
    logits:  (B, sum(levels), max_vocab)
    targets: (B, sum(levels))  -- clean codes for levels[1:] when level 0 is not predicted
    """
    B = targets.shape[0]
    device = logits.device
    w = level_weights(levels, alpha, device=device, dtype=torch.float32)
    total = logits.new_zeros(())
    parts = []
    level_means = []
    offset = 0
    for k, (ell, vk) in enumerate(zip(levels, vocab_sizes)):
        ell, vk = int(ell), int(vk)
        lg = logits[:, offset : offset + ell, :vk].reshape(-1, vk).float()
        tg = targets[:, offset : offset + ell].reshape(-1)
        ce = F.cross_entropy(lg, tg, reduction="none")
        total = total + (ce * w[k]).sum()
        level_means.append(ce.mean())
        parts.append(float(level_means[-1].detach()))
        offset += ell
    loss = total / B
    coef = float(mid_aux or 0.0)
    if coef != 0.0 and level_means:
        start, end = mid_level_slice(len(level_means))
        if end > start:
            mid_mean = torch.stack(level_means[start:end]).mean()
            loss = loss + coef * mid_mean
    if return_parts:
        return loss, parts
    return loss


def corrupt_level_rates(n_levels, p_max=0.8, p_min=0.2):
    """Per-level input corruption rates, coarse p_max down to fine p_min."""
    n = int(n_levels)
    if n <= 0:
        return []
    if n == 1:
        return [float(p_max)]
    span = float(p_max) - float(p_min)
    return [float(p_max) - span * k / (n - 1) for k in range(n)]


def corrupt_codes(idx, levels, vocab_sizes, p_max=0.8, p_min=0.2, generator=None):
    """Replace a fraction of INPUT codes uniformly from the same level's codebook.

    Rate decreases linearly from p_max at level 0 to p_min at the last level
    (mean 0.50 when p_max=0.8, p_min=0.2). Labels are not modified.
    Pass a torch.Generator to keep a validation pass from changing the train RNG.
    """
    out = idx.clone()
    rates = corrupt_level_rates(len(levels), p_max=p_max, p_min=p_min)
    offset = 0
    for p, ell, vk in zip(rates, levels, vocab_sizes):
        ell, vk = int(ell), int(vk)
        chunk = out[:, offset : offset + ell]
        mask = _rand_like(chunk.shape, idx.device, generator) < p
        noise = _randint_like(vk, chunk.shape, idx.device, generator)
        out[:, offset : offset + ell] = torch.where(mask, noise, chunk)
        offset += ell
    return out


def _generator_device(generator):
    if generator is None:
        return None
    dev = getattr(generator, "device", None)
    if dev is None:
        return torch.device("cpu")
    return dev if isinstance(dev, torch.device) else torch.device(dev)


def _rand_like(shape, device, generator):
    """Draw Uniform(0, 1). A CPU generator still works when `device` is CUDA."""
    if generator is None or _generator_device(generator).type == torch.device(device).type:
        return torch.rand(shape, device=device, generator=generator)
    return torch.rand(shape, generator=generator).to(device)


def _randint_like(high, shape, device, generator):
    if generator is None or _generator_device(generator).type == torch.device(device).type:
        return torch.randint(0, int(high), shape, device=device, generator=generator)
    return torch.randint(0, int(high), shape, generator=generator).to(device)


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
    """Atomic checkpoint write (tmp + os.replace). Call only on the main rank."""
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def maybe_copy_config(cfg, dest_dir):
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, dest / "config.yaml")


def add_runtime_args(parser):
    """Wall-clock budget and corpus caps, used by Kaggle / long GPU sessions."""
    parser.add_argument(
        "--max_hours",
        type=float,
        default=None,
        help="stop this run after this many hours (saves a checkpoint first)",
    )
    parser.add_argument(
        "--reserve_minutes",
        type=float,
        default=10.0,
        help="minutes reserved at the end of --max_hours for saving",
    )
    parser.add_argument("--max_docs", type=int, default=None, help="cap documents packed from --data / --hf_dataset")
    parser.add_argument("--max_windows", type=int, default=None, help="cap packed training windows")
    return parser


def get_rank() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return 0


def get_world_size() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size())
    except Exception:
        pass
    return 1


def init_distributed(device_name: str | None = None):
    """Bind this process to its GPU when launched with torchrun.

    Returns (device, rank, world_size, local_rank). A plain `python train_*.py`
    launch is unchanged (world_size=1).
    """
    import os
    if os.environ.get("RANK") is None:
        device = resolve_device(device_name)
        return device, 0, 1, 0
    import torch.distributed as dist
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if not torch.cuda.is_available():
        raise RuntimeError("torchrun DDP requires CUDA")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    return device, rank, world_size, local_rank


def cleanup_distributed():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    except Exception:
        pass


def wrap_ddp(model, local_rank: int, world_size: int, find_unused_parameters: bool = False):
    if world_size <= 1:
        return model
    from torch.nn.parallel import DistributedDataParallel as DDP
    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=False,
    )


def unwrap_model(model):
    from torch.nn.parallel import DistributedDataParallel as DDP
    if isinstance(model, DDP):
        model = model.module
    return getattr(model, "_orig_mod", model)


def module_state_dict(model):
    return unwrap_model(model).state_dict()


def load_module_state(model, state, strict=True):
    if state and any(k.startswith("_orig_mod.") for k in state):
        state = {k[10:] if k.startswith("_orig_mod.") else k: v for k, v in state.items()}
    unwrap_model(model).load_state_dict(state, strict=strict)


def resolve_resume(resume, output_dir, preferred=("latest.pt",)):
    if resume is None:
        return None
    if str(resume) != "auto":
        path = Path(resume)
        if not path.is_file():
            raise SystemExit(f"resume checkpoint not found: {resume}")
        return str(path)
    out = Path(output_dir)
    for name in preferred:
        path = out / name
        if path.is_file():
            return str(path)
    return None


def write_run_status(out_dir, step, max_steps, finished, extra=None):
    import json
    payload = {"step": int(step), "max_steps": int(max_steps), "finished": bool(finished)}
    if extra:
        payload.update(extra)
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "status.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def all_reduce_sum(value, device):
    t = torch.tensor([float(value)], device=device, dtype=torch.float64)
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    except Exception:
        pass
    return float(t.item())


class TimeBudget:
    def __init__(self, max_hours=None, reserve_minutes=10.0):
        import time
        self._time = time
        self.start = time.monotonic()
        self.max_seconds = None if max_hours in (None, 0) else float(max_hours) * 3600.0
        self.reserve = float(reserve_minutes or 0.0) * 60.0

    def should_stop(self):
        if self.max_seconds is None:
            return False
        return (self._time.monotonic() - self.start) >= (self.max_seconds - self.reserve)

    def elapsed_hours(self):
        return (self._time.monotonic() - self.start) / 3600.0


class ThroughputMeter:
    def __init__(self):
        import time
        self._time = time
        self.reset()

    def update(self, n_tokens, n_steps=1):
        self.tokens += int(n_tokens)
        self.steps += int(n_steps)

    def rate(self):
        dt = max(self._time.perf_counter() - self.t0, 1e-9)
        return self.steps / dt, self.tokens / dt

    def reset(self):
        self.t0 = self._time.perf_counter()
        self.tokens = 0
        self.steps = 0


def cuda_mem_gb():
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated()) / (1024 ** 3)


def prune_checkpoints(out_dir, pattern, keep=2):
    """Keep the newest `keep` files matching a glob, delete the rest."""
    out = Path(out_dir)
    files = [p for p in out.glob(pattern) if p.is_file() and not p.name.endswith(".tmp")]
    files.sort(key=lambda p: p.stat().st_mtime)
    for path in files[:-int(keep)]:
        try:
            path.unlink()
        except OSError:
            pass


class TrainLogger:
    """Rank-0 stdout + JSONL log for long Kaggle / multi-session runs."""

    def __init__(self, out_dir, name):
        self.enabled = is_main_process()
        self.path = Path(out_dir) / f"{name}_log.jsonl"
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **fields):
        import json
        from datetime import datetime, timezone, timedelta
        rec = {
            "time": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
            **fields,
        }
        if not self.enabled:
            return rec
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return rec
