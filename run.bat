@echo off
rem FlareSim Web - local preview
rem Starts a static server on http://localhost:8080/ and opens the browser.
rem WebGPU requires Chrome / Edge 113+ (HDR output: Chrome / Edge 129+).
setlocal
cd /d "%~dp0"
set PORT=8080
if not "%~1"=="" set PORT=%~1

start "" "http://localhost:%PORT%/"

where node >nul 2>nul
if %errorlevel%==0 (
  node tools\serve.mjs %PORT%
) else (
  echo Node.js not found - using PowerShell server.
  powershell -NoProfile -ExecutionPolicy Bypass -File tools\serve.ps1 -Port %PORT%
)
endlocal
