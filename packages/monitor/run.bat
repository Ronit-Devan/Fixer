@echo off
REM One command to run the ET monitor on this machine (Windows).
REM   run.bat                    auto-detect GPU, find llama-server on :8080
REM   run.bat --gpu-price 0.50   show $ wasted to idle
REM   run.bat --demo             scripted timeline, no GPU/model needed
REM
REM Override the interpreter:  set PYTHON=C:\path\to\python.exe & run.bat
setlocal
cd /d "%~dp0"

REM Pick a Python launcher: %PYTHON% override, else "python", else the "py" launcher.
set "PY=%PYTHON%"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  where py >nul 2>nul && set "PY=py -3"
)
if not defined PY (
  echo Error: Python not found on PATH. Install Python 3.10+ from python.org, or set PYTHON.
  exit /b 1
)

REM Reuse an existing venv only if its interpreter is present AND is Python 3.10+.
REM Probing the interpreter (not just the .venv folder) recovers from a half-made
REM or stale venv left by an interrupted first run; a too-old venv is rebuilt.
set "VENV_PY=.venv\Scripts\python.exe"
set "NEED_VENV=1"
if exist "%VENV_PY%" (
  "%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && set "NEED_VENV=0"
)
if "%NEED_VENV%"=="1" (
  %PY% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
  if errorlevel 1 (
    echo Error: Python 3.10+ is required. Install a newer Python from python.org, or set PYTHON.
    exit /b 1
  )
  if exist .venv rmdir /s /q .venv
  echo Creating virtual environment...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo Error: failed to create the virtualenv. Ensure Python 3.10+ is installed correctly.
    exit /b 1
  )
)

call "%VENV_PY%" -m pip install -q --upgrade pip
REM Try with the NVIDIA telemetry extra; fall back to core if the wheel won't build.
call "%VENV_PY%" -m pip install -q -e ".[gpu]"
if errorlevel 1 (
  echo nvidia-ml-py unavailable; installing core only ^(nvidia-smi / mock fallback^).
  call "%VENV_PY%" -m pip install -q -e .
  if errorlevel 1 (
    echo Error: installation failed; see the pip output above.
    exit /b 1
  )
)

call ".venv\Scripts\et-monitor.exe" %*
endlocal
