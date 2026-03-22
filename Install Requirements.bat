@echo off
cd /d "%~dp0"
echo Checking Python version...
python --version
echo Checking pip version...
python -m pip --version
echo.
echo Installing requirements please wait patiently...
python -m pip install -r requirements.txt
python -m pip install -r pytorch_fix.txt
pause