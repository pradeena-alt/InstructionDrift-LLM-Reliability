@echo off
setlocal
cd /d "%~dp0"
echo InstructionDrift Dashboard
if not exist "app.py" (
  echo ERROR: app.py is missing from this folder.
  pause
  exit /b 1
)
python -m pip install -r requirements-dashboard.txt
if errorlevel 1 (
  echo Dependency installation failed. Check Python and internet connection.
  pause
  exit /b 1
)
python -m streamlit run app.py
pause
