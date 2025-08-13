# LLM to TTS Demo - Updated for llama-server

## Overview

This demo now uses your existing **llama-server** running on localhost:8080 instead of requiring llama-cpp-python installation. This is much cleaner and avoids build tool dependencies!

## Quick Start

### 1. Ensure llama-server is running
Make sure your llama-server is running on localhost:8080. If not, start it with:
```bash
llama-server --host 0.0.0.0 --port 8080 --model /path/to/your/model.gguf
```

### 2. Test the components

```bash
# Test TTS only
python llm_tts.py --test-tts

# Test LLM connection only
python llm_tts.py --test-llm

# Test both LLM and TTS together
python llm_tts.py --test-all
```

### 3. Start interactive chat
```bash
# Full interactive chat with TTS
python llm_tts.py

# Use custom LLM server URL
python llm_tts.py --llm-url http://localhost:8080
```

## What's Working

✅ **HTTP API Integration**: Connects to your llama-server via REST API
✅ **Streaming Responses**: Real-time token streaming from LLM
✅ **TTS Synthesis**: Converts text to speech as sentences complete
✅ **Audio Playback**: Streams audio directly to speakers
✅ **Low Latency**: Parallel processing for smooth experience
✅ **Interactive Chat**: Full conversation with history management

## Architecture

```
User Input → HTTP Request → llama-server → Streaming Response → Text Buffer → TTS → Audio → Speakers
```

## Commands

```bash
# Basic chat
python llm_tts.py

# Test TTS only
python llm_tts.py --test-tts

# Test LLM only  
python llm_tts.py --test-llm

# Test both systems
python llm_tts.py --test-all

# Custom LLM server
python llm_tts.py --llm-url http://localhost:8080

# Custom TTS model
python llm_tts.py --tts-model path/to/model.onnx

# Single text test
python llm_tts.py --test-tts --text "Custom text to speak"
```

## Features

- **No Build Tools Required**: Uses HTTP API instead of native bindings
- **Server Compatibility**: Works with any llama.cpp server
- **Auto-Detection**: Automatically detects if LLM server is available
- **Graceful Degradation**: Falls back to TTS-only if LLM unavailable
- **Conversation History**: Maintains context across chat turns
- **Sentence-Level Streaming**: Speaks complete sentences for natural flow
- **Audio Threading**: Non-blocking audio playback

## Requirements

- **Running llama-server** on localhost:8080 (or custom URL)
- **Python dependencies**: `piper-tts`, `pyaudio`, `requests`
- **TTS model**: `en_US-hfc_female-medium.onnx` (included)

This approach is much more robust and doesn't require complex C++ build environments!
