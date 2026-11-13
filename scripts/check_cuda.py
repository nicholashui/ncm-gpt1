#!/usr/bin/env python3
"""Print the installed PyTorch / CUDA / GPU status."""
import sys

try:
    import torch
except ImportError:
    print("PyTorch is not installed.")
    sys.exit(1)

print(f"python     {sys.version.split()[0]}")
print(f"torch      {torch.__version__}")
print(f"cuda build {torch.version.cuda}")
print(f"cuda avail {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        cap = torch.cuda.get_device_capability(i)
        vram = p.total_memory / (1024 ** 3)
        bf16 = torch.cuda.is_bf16_supported() if i == 0 else ""
        print(f"gpu {i}      {p.name}  cap={cap[0]}.{cap[1]}  vram={vram:.1f}GiB")
    print(f"bf16       {torch.cuda.is_bf16_supported()}")
    x = torch.randn(1024, 1024, device="cuda")
    y = (x @ x.T).sum().item()
    print(f"matmul ok  {y:.4g}")
else:
    print("No CUDA GPU visible. Install a CUDA build of PyTorch (see install_cuda.bat / .sh).")
    sys.exit(2)
