@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python is required before this project can run.
        echo.
        echo Recommended terminal command:
        echo winget install -e --id Python.Python.3.11
        echo.
        echo If winget is not available, install Python 3.11 or newer from:
        echo https://www.python.org/downloads/
        echo.
        echo During setup, tick "Add python.exe to PATH".
        echo Then close and reopen the terminal, and run this file again.
        pause
        exit /b 1
    )
    set "PYTHON_CMD=python"
) else (
    set "PYTHON_CMD=py -3"
)

set "VENV_DIR=.web_venv"

if exist "%VENV_DIR%\Scripts\python.exe" (
    "%VENV_DIR%\Scripts\python.exe" -c "import sys" >nul 2>nul
    if errorlevel 1 (
        set "BACKUP=%VENV_DIR%_broken_%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
        set "BACKUP=%BACKUP: =0%"
        echo Existing website environment is not usable. Renaming it to %BACKUP%...
        ren "%VENV_DIR%" "%BACKUP%"
    )
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating website environment...
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

call "%VENV_DIR%\Scripts\activate.bat"
if not exist "C:\tmp\credit-card-approval-temp" mkdir "C:\tmp\credit-card-approval-temp"
set "TEMP=C:\tmp\credit-card-approval-temp"
set "TMP=C:\tmp\credit-card-approval-temp"
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Package installation failed.
    echo Please connect to the internet, then run this file again.
    echo You can also run this manually:
    echo python -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo Website is starting...
echo Open this link in your browser:
echo http://127.0.0.1:5000
echo.
python run.py
