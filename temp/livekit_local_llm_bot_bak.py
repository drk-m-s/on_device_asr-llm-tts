"""
LiveKit + Local LLM + Vosk ASR + Piper TTS
- ASR converts mic audio to text
- Calls local LLM API (HTTP POST) with recognized text
- Sends LLM response to browser
- Plays TTS using Piper
- Voice interruption cancels ongoing TTS
- Never echoes user's speech
"""

import asyncio
import base64
import json
import os
import shlex
import subprocess
import sys
import tempfile

import aiohttp
import numpy as np
import websockets
from vosk import KaldiRecognizer, Model

# CONFIG
WS_HOST = "0.0.0.0"
WS_PORT = 8765
VOSK_MODEL_PATH = "./vosk_models/vosk-model-small-en-us-0.15"
LLM_API_URL = "http://localhost:8080/completion"
PIPER_CMD_TEMPLATE = "piper --model /Users/shuyuew/Documents/GitHub/on_device_asr-llm-tts/tts_models/en_US-hfc_female-medium.onnx --output-file {outfile}"
SAMPLE_RATE = 16000
VOICE_THRESHOLD = 500  # simple VAD

if not os.path.exists(VOSK_MODEL_PATH):
    print(f"ERROR: Vosk model not found at {VOSK_MODEL_PATH}.")
    sys.exit(1)

vosk_model = Model(VOSK_MODEL_PATH)


class ClientState:
    def __init__(self, ws):
        self.ws = ws
        self.rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
        self.buffer = bytearray()
        self.playing = False
        self.stop_playback_flag = False
        self.llm_task = None

    def feed_audio(self, pcm_bytes: bytes):
        # voice-activated interruption
        if self.playing and self.is_speech(pcm_bytes):
            self.stop_playback_flag = True

        self.buffer.extend(pcm_bytes)
        # process every 0.1s of audio
        if len(self.buffer) >= SAMPLE_RATE * 2 * 0.1:
            chunk = bytes(self.buffer)
            self.buffer.clear()
            if self.rec.AcceptWaveform(chunk):
                res = json.loads(self.rec.Result())
                text = res.get("text", "")
                return text, True
            else:
                partial = json.loads(self.rec.PartialResult()).get("partial", "")
                return partial, False
        else:
            return "", False

    def is_speech(self, pcm_bytes):
        data = np.frombuffer(pcm_bytes, dtype=np.int16)
        return np.max(np.abs(data)) > VOICE_THRESHOLD


async def call_llm(prompt_text):
    payload = {"prompt": prompt_text, "n_predict": 128}
    async with aiohttp.ClientSession() as session:
        async with session.post(LLM_API_URL, json=payload) as resp:
            data = await resp.json()
            return data.get("content", "")


async def play_tts(client, text):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpf:
        wav_path = tmpf.name

    cmd = PIPER_CMD_TEMPLATE.format(outfile=shlex.quote(wav_path))
    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE)
    proc.communicate(input=text.encode("utf-8"))

    with open(wav_path, "rb") as f:
        wav_bytes = f.read()
    b64 = base64.b64encode(wav_bytes).decode("ascii")

    client.playing = True
    await client.ws.send(json.dumps({"type": "tts", "wav_b64": b64}))
    client.playing = False

    try:
        os.remove(wav_path)
    except Exception:
        pass


async def run_llm_and_tts(client, prompt_text):
    try:
        if client.stop_playback_flag:
            client.stop_playback_flag = False
            return

        llm_response = await call_llm(prompt_text)

        # Send the LLM response to browser
        await client.ws.send(json.dumps({"type": "partial", "text": llm_response}))
        await client.ws.send(json.dumps({"type": "final", "text": llm_response}))

        # Play TTS
        await play_tts(client, llm_response)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        print("Error in LLM/TTS task:", e)


async def handle_ws(websocket):
    client = ClientState(websocket)
    print("WS client connected")

    try:
        async for message in websocket:
            try:
                msg = json.loads(message)
            except Exception:
                continue

            if msg.get("type") == "audio":
                pcm_b64 = msg.get("pcm_b64")
                if not pcm_b64:
                    continue
                pcm = base64.b64decode(pcm_b64)
                text, is_final = client.feed_audio(pcm)
                if text and is_final:
                    # Only trigger LLM; do not echo ASR
                    if client.llm_task and not client.llm_task.done():
                        client.stop_playback_flag = True
                        client.llm_task.cancel()
                    client.llm_task = asyncio.create_task(run_llm_and_tts(client, text))

    except websockets.exceptions.ConnectionClosed:
        print("client disconnected")


async def main():
    print(f"Starting bot WebSocket server on ws://{WS_HOST}:{WS_PORT}")
    async with websockets.serve(handle_ws, WS_HOST, WS_PORT):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down")
