@echo off
REM One command to run the ET monitor on this machine (Windows).
REM   run.bat                    auto-detect GPU, find llama-server on :8080
REM   run.bat --gpu-price 0.50   show $ wasted to idle
REM   run.bat --demo             scripted timeline, no GPU/model needed
setlocal
cd /d "%~dp0"

if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv
)

call .venv\Scripts\python.exe -m pip install -q --upgrade pip

call .venv\Scripts\pip.exe install -q -e ".[gpu]"
if errorlevel 1 (
  echo nvidia-ml-py unavailable; installing core only ^(nvidia-smi / mock fallback^).
  call .venv\Scripts\pip.exe install -q -e .
)

call .venv\Scripts\et-monitor.exe %*
endlocal
