@echo off
echo Setting up LLM to TTS Demo...

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install --upgrade pip

REM Try multiple approaches for llama-cpp-python
echo Attempting to install llama-cpp-python (this may take a few tries)...

REM Method 1: Try pre-compiled CPU version
echo Method 1: Trying pre-compiled CPU version...
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --no-cache-dir
if %ERRORLEVEL% equ 0 goto :llama_success

REM Method 2: Try force binary only
echo Method 2: Trying binary-only installation...
pip install llama-cpp-python --only-binary=llama-cpp-python --no-cache-dir
if %ERRORLEVEL% equ 0 goto :llama_success

REM Method 3: Try specific version
echo Method 3: Trying specific pre-compiled version...
pip install llama-cpp-python==0.2.11 --only-binary=llama-cpp-python --no-cache-dir
if %ERRORLEVEL% equ 0 goto :llama_success

echo All llama-cpp-python installation methods failed.
echo You can still use TTS-only functionality.
set LLAMA_FAILED=1
goto :continue_install

:llama_success
echo llama-cpp-python installed successfully!
set LLAMA_FAILED=0

:continue_install
REM Install other dependencies
echo Installing other dependencies...
pip install piper-tts
pip install pyaudio

echo.
echo Setup complete!
echo.
echo To run the demo:
echo 1. For TTS test only: python llm_tts.py --test-tts
echo 2. For full LLM+TTS chat: python llm_tts.py --model path\to\your\model.gguf
echo.
echo You'll need to download a compatible .gguf model file for the LLM functionality.
echo Recommended models:
echo - llama-2-7b-chat.Q4_K_M.gguf
echo - mistral-7b-instruct-v0.1.Q4_K_M.gguf
echo.
echo Download from: https://huggingface.co/TheBloke
echo.
pause
