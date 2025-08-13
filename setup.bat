@echo off
echo VLM Backend Setup Script
echo ========================
echo.

REM Check if Python is installed
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ✗ Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

python --version
echo ✓ Python is installed
echo.

REM Install requirements
echo Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ✗ Error: Failed to install dependencies
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo ✓ Dependencies installed successfully
echo.

REM Test the backend (without starting the server)
echo Testing backend configuration...
python -c "import flask, flask_cors, requests; print('✓ All imports successful')"
if errorlevel 1 (
    echo ✗ Error: Some dependencies are not working correctly
    pause
    exit /b 1
)

echo ✓ Backend configuration is valid
echo.

echo Setup complete! You can now:
echo 1. Start your llama.cpp VLM server on port 8080
echo 2. Run 'start_backend.bat' to start the Python backend
echo 3. Open 'index.html' in a web browser
echo.
echo For testing, you can run 'python test_backend.py' after starting the backend
echo.

pause
