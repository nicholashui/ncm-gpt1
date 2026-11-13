@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo Python not found on PATH. Install Python 3.10+ and retry.
  exit /b 1
)

echo Installing CUDA PyTorch (cu128) and project dependencies...
python -m pip install -U pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
  echo cu128 wheel failed, trying cu126...
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
)
python -m pip install -r requirements.txt
python -m pip install datasets
echo.
python scripts\check_cuda.py
exit /b %ERRORLEVEL%
