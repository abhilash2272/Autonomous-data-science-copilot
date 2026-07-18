@echo off
:: ─────────────────────────────────────────────────────────────────────────────
:: run.bat  —  Activate the virtual environment and launch the app
:: ─────────────────────────────────────────────────────────────────────────────

SETLOCAL

SET PROJECT_DIR=%~dp0
SET VENV_DIR=%PROJECT_DIR%venv
SET VENV_PYTHON=%VENV_DIR%\Scripts\python.exe
SET VENV_STREAMLIT=%VENV_DIR%\Scripts\streamlit.exe

echo.
echo  ╔══════════════════════════════════════════════════════╗
echo  ║   Autonomous Data Science Co-Pilot  —  Launcher     ║
echo  ╚══════════════════════════════════════════════════════╝
echo.

:: ── Check venv exists ─────────────────────────────────────────────────────────
IF NOT EXIST "%VENV_PYTHON%" (
    echo  [SETUP] Virtual environment not found. Creating it now...
    python -m venv "%VENV_DIR%"
    IF ERRORLEVEL 1 (
        echo  [ERROR] Failed to create virtual environment.
        echo         Make sure Python 3.11+ is installed and in PATH.
        pause
        exit /b 1
    )
    echo  [OK]    Virtual environment created.
    echo.
    echo  [SETUP] Installing dependencies from requirements.txt ...
    "%VENV_PYTHON%" -m pip install --upgrade pip --quiet
    "%VENV_PYTHON%" -m pip install -r "%PROJECT_DIR%requirements.txt"
    IF ERRORLEVEL 1 (
        echo  [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
    echo  [OK]    Dependencies installed.
) ELSE (
    echo  [OK]    Virtual environment found at: %VENV_DIR%
)

:: ── Launch the app ────────────────────────────────────────────────────────────
echo.
echo  [LAUNCH] Starting Streamlit on http://localhost:8501 ...
echo.

"%VENV_STREAMLIT%" run "%PROJECT_DIR%app.py" --server.port 8501 --server.headless false

ENDLOCAL
