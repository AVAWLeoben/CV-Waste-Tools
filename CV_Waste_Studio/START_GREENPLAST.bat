@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title CV Waste Tools Launcher

rem Try the active Conda installation first, then common Windows paths.
set "CONDA_BAT="
if defined CONDA_EXE (
    for %%I in ("%CONDA_EXE%") do (
        if exist "%%~dpI..\condabin\conda.bat" set "CONDA_BAT=%%~dpI..\condabin\conda.bat"
    )
)
if not defined CONDA_BAT (
    for /f "delims=" %%I in ('where conda.bat 2^>nul') do (
        if not defined CONDA_BAT set "CONDA_BAT=%%I"
    )
)
if not defined CONDA_BAT if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\miniforge3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\miniforge3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%LOCALAPPDATA%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%LOCALAPPDATA%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%LOCALAPPDATA%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%LOCALAPPDATA%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%ProgramData%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%ProgramData%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%ProgramData%\Anaconda3\condabin\conda.bat" set "CONDA_BAT=%ProgramData%\Anaconda3\condabin\conda.bat"

if not defined CONDA_BAT (
    echo ERROR: Conda was not found.
    echo Install Miniconda, Miniforge or Anaconda first, then double-click this file.
    echo https://docs.conda.io/projects/conda/en/latest/user-guide/install/
    if /i not "%~1"=="--silent" pause
    exit /b 1
)

echo Using Conda launcher: "%CONDA_BAT%"
rem Fast path: start the already-installed Conda base Python directly.
rem This avoids `conda activate base` and environment enumeration on repeat starts.
set "BASE_PYTHON="
for %%I in ("%CONDA_BAT%") do if exist "%%~dpI..\python.exe" set "BASE_PYTHON=%%~dpI..\python.exe"
if defined BASE_PYTHON goto :fast_launch
rem Fallback for nonstandard Conda layouts.
call "%CONDA_BAT%" activate base
if errorlevel 1 (
    echo ERROR: Could not activate the Conda base environment.
    if /i not "%~1"=="--silent" pause
    exit /b 1
)
rem A few minimal Conda base environments omit Tkinter. Bootstrap from the
rem standard library in that case; the dedicated environment installs tk.
python -c "import tkinter" >nul 2>&1
if errorlevel 1 (
    echo Conda base lacks Tkinter. Starting headless environment bootstrap...
    python -u "%~dp0greenplast_bootstrap.py"
) else (
    python -u "%~dp0greenplast_launcher.py"
)
if errorlevel 1 (
    echo.
    echo The GreenPlast launcher exited with an error.
    if /i not "%~1"=="--silent" pause
    exit /b 1
)
exit /b 0

:fast_launch
for %%I in ("%BASE_PYTHON%") do set "CONDA_ROOT=%%~dpI"
if exist "%CONDA_ROOT%Scripts\conda.exe" set "CONDA_EXE=%CONDA_ROOT%Scripts\conda.exe"
set "PATH=%CONDA_ROOT%;%CONDA_ROOT%Library\bin;%CONDA_ROOT%Scripts;%PATH%"
"%BASE_PYTHON%" -c "import tkinter" >nul 2>&1
if errorlevel 1 (
    "%BASE_PYTHON%" -u "%~dp0greenplast_bootstrap.py"
) else (
    "%BASE_PYTHON%" -u "%~dp0greenplast_launcher.py"
)
if errorlevel 1 (
    echo.
    echo The CV Waste Tools launcher exited with an error.
    if /i not "%~1"=="--silent" pause
    exit /b 1
)
exit /b 0
