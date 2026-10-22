@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ============================================================
rem AVAW LiteRT Export Menu
rem ============================================================

set "WINDOWS_WORKDIR=%~dp0"
rem WSL --cd accepts a Windows folder path directly; no wslpath conversion needed.
rem This also works when the Windows username or folder location changes.
set "WSL_ENV=~/yolo_export"
set "PY_SCRIPT=export_to_litert_menu_cli.py"

title AVAW LiteRT Export

:START
cls
set "IMGSZ="
set "MODEL="
set "NMS=none"
set "INT8=false"
set "HALF=false"

echo ============================================================
echo                 AVAW LiteRT Export
echo ============================================================
echo.
echo Folder:
echo   %WINDOWS_WORKDIR%
echo.

if not exist "%WINDOWS_WORKDIR%" (
    echo ERROR: Folder does not exist:
    echo   %WINDOWS_WORKDIR%
    echo.
    pause
    exit /b 2
)

rem ------------------------------------------------------------
rem Find .pt models
rem ------------------------------------------------------------
set "COUNT=0"

for /f "delims=" %%F in ('dir /b /a-d "%WINDOWS_WORKDIR%\*.pt" 2^>nul') do (
    set /a COUNT+=1
    set "MODEL_!COUNT!=%%F"
)

if "%COUNT%"=="0" (
    echo No .pt models found in:
    echo   %WINDOWS_WORKDIR%
    echo.
    echo Copy a YOLO .pt model into that folder and run this BAT again.
    echo.
    pause
    exit /b 3
)

echo Available models:
echo.

for /L %%I in (1,1,%COUNT%) do (
    echo   [%%I] !MODEL_%%I!
)

echo.
echo   [R] Refresh model list
echo   [Q] Quit
echo.

set /p "MODEL_CHOICE=Select model: "

if /I "%MODEL_CHOICE%"=="Q" exit /b 0
if /I "%MODEL_CHOICE%"=="R" goto START

for /f "delims=0123456789" %%A in ("%MODEL_CHOICE%") do (
    echo.
    echo Invalid selection.
    timeout /t 2 >nul
    goto START
)

if "%MODEL_CHOICE%"=="" goto START
if %MODEL_CHOICE% LSS 1 goto START
if %MODEL_CHOICE% GTR %COUNT% goto START

for %%I in (%MODEL_CHOICE%) do set "MODEL=!MODEL_%%I!"

rem ------------------------------------------------------------
rem Image size menu
rem ------------------------------------------------------------
:IMGSZ_MENU
cls
echo ============================================================
echo                 AVAW LiteRT Export
echo ============================================================
echo.
echo Model:
echo   %MODEL%
echo.
echo Select export image size:
echo.
echo   [1] 160
echo   [2] 320
echo   [3] 640
echo   [4] Custom
echo   [B] Back
echo.
set /p "SIZE_CHOICE=Selection: "

if /I "%SIZE_CHOICE%"=="B" goto START

if "%SIZE_CHOICE%"=="1" set "IMGSZ=160"
if "%SIZE_CHOICE%"=="2" set "IMGSZ=320"
if "%SIZE_CHOICE%"=="3" set "IMGSZ=640"

if "%SIZE_CHOICE%"=="4" (
    echo.
    echo Enter a single size such as:
    echo   512
    echo.
    echo or a rectangular size such as:
    echo   640,384
    echo.
    set /p "IMGSZ=Custom imgsz: "
    if "!IMGSZ!"=="" goto IMGSZ_MENU
)

if not defined IMGSZ goto IMGSZ_MENU

rem ------------------------------------------------------------
rem Optional advanced settings
rem ------------------------------------------------------------
cls
echo ============================================================
echo                 AVAW LiteRT Export
echo ============================================================
echo.
echo Selected:
echo   Model : %MODEL%
echo   imgsz : %IMGSZ%
echo.
echo Default export settings:
echo   NMS   : none
echo   INT8  : false
echo   HALF  : false
echo.
set /p "ADVANCED=Change advanced settings? [y/N]: "

if /I "%ADVANCED%"=="Y" (
    echo.
    set /p "NMS_IN=NMS [none/true/false] (default none): "
    if not "!NMS_IN!"=="" set "NMS=!NMS_IN!"

    set /p "INT8_IN=INT8 [true/false] (default false): "
    if not "!INT8_IN!"=="" set "INT8=!INT8_IN!"

    set /p "HALF_IN=HALF [true/false] (default false): "
    if not "!HALF_IN!"=="" set "HALF=!HALF_IN!"
)

rem ------------------------------------------------------------
rem Confirmation
rem ------------------------------------------------------------
cls
echo ============================================================
echo                 AVAW LiteRT Export
echo ============================================================
echo.
echo Model:
echo   %MODEL%
echo.
echo Export parameters:
echo   format = litert
echo   imgsz  = %IMGSZ%
echo   nms    = %NMS%
echo   int8   = %INT8%
echo   half   = %HALF%
echo.
echo WSL environment:
echo   %WSL_ENV%
echo.
echo Working folder:
echo   %WINDOWS_WORKDIR%
echo.
echo ============================================================
echo.
set /p "CONFIRM=Start export? [Y/n]: "

if /I "%CONFIRM%"=="N" goto START

echo.
echo Checking WSL environment and installing dependencies if needed...
echo.
if not exist "%WINDOWS_WORKDIR%setup_yolo_export.sh" (
    echo ERROR: setup_yolo_export.sh is missing from the script folder.
    pause
    exit /b 4
)
wsl.exe --cd "%WINDOWS_WORKDIR%" bash ./setup_yolo_export.sh
if errorlevel 1 (
    echo.
    echo ERROR: WSL environment setup or verification failed.
    pause
    exit /b 4
)

echo.
echo Starting export...
echo.
wsl.exe --cd "%WINDOWS_WORKDIR%" bash -c "exec ~/yolo_export/bin/python export_to_litert_menu_cli.py --model '%MODEL%' --imgsz '%IMGSZ%' --nms %NMS% --int8 %INT8% --half %HALF%"
set "RC=%ERRORLEVEL%"

echo.
echo ============================================================

if "%RC%"=="0" (
    echo EXPORT FINISHED SUCCESSFULLY.
) else (
    echo EXPORT FAILED with exit code %RC%.
)

echo ============================================================
echo.
set /p "AGAIN=Export another model? [y/N]: "

if /I "%AGAIN%"=="Y" goto START

exit /b %RC%
