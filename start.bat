@echo off
setlocal
title Koscine 3.0 launcher
cd /d "%~dp0"
set "PYTHONPATH=%cd%\src"

echo ============================================
echo   Koscine 3.0 - Large-Move Options Console
echo ============================================
echo.

REM First-run: install frontend deps if missing
if not exist "frontend\node_modules" (
  echo Installing frontend dependencies ^(first run, one-time^)...
  pushd frontend
  call npm install
  popd
  echo.
)

echo Starting backend API on http://127.0.0.1:8003 ...
start "Koscine API (8003)" cmd /k python -m uvicorn api.main:app --port 8003

echo Starting frontend on http://localhost:5174 ...
start "Koscine Web (5174)" cmd /k "cd frontend && npm run dev"

echo Waiting for servers to come up ...
timeout /t 8 /nobreak >nul

echo Opening browser ...
start "" "http://localhost:5174"

echo.
echo   Frontend : http://localhost:5174
echo   API      : http://127.0.0.1:8003/prod/manifest
echo.
echo   Two server windows opened. Close them (or press Ctrl+C in each) to stop.
echo   You can close THIS window now.
echo.
pause
endlocal
