import asyncio
import json
import queue
import threading
import numpy as np
import sounddevice as sd
import subprocess
import requests
from pathlib import Path
from livekit_client import Room, LocalAudioTrack, AudioCaptureOptions, TrackPublicationOptions
import vosk

# -----------------------------
# Config
# -----------------------------
LIVEKIT_URL = "ws://localhost:7880"
ROOM_NAME = "chatbot_room"
USER_NAME = "python_bot"
LLM_API_URL = "http://localhost:8080/completion"
VOSK_MODEL_PATH = "vosk_models/vosk-model-small-en-us-0.15"
TTS_MODEL_PATH = "en_US-hfc_female-medium.onnx"

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024

# -----------------------------
# Globals
# -----------------------------
audio_queue = queue.Queue()
interrupt_event = threading.Event()
tts_process = None
local_tts_track: LocalAudioTrack = None

# -----------------------------
# ASR Thread
# -----------------------------
def asr_loop():
    model = vosk.Model(VOSK_MODEL_PATH)
    rec = vosk.KaldiRecognizer(model, SAMPLE_RATE)

    print("ASR ready. Speak into your microphone.")

    while True:
        data = audio_queue.get()
        if data is None:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "")
            if text.strip():
                print(f"[ASR] User said: {text}")
                asyncio.run(handle_user_input(text))

# -----------------------------
# Handle LLM + TTS + LiveKit
# -----------------------------
async def handle_user_input(text):
    global tts_process, interrupt_event, local_tts_track
    # Interrupt any current TTS
    interrupt_event.set()
    interrupt_event.clear()

    # Call LLM
    payload = {"prompt": text, "n_predict": 128}
    resp = requests.post(LLM_API_URL, json=payload)
    resp_json = resp.json()
    llm_text = resp_json.get("content", "Sorry, I have no answer.")

    print(f"[LLM] Response: {llm_text}")

    # Save text to temporary file
    temp_text_file = "temp_response.txt"
    with open(temp_text_file, "w") as f:
        f.write(llm_text)

    # Run Piper TTS
    tts_cmd = [
        "python3", "-m", "piper.cli.synthesize",
        "--voice", TTS_MODEL_PATH,
        "--text-file", temp_text_file,
        "--output", "temp_response.wav"
    ]
    tts_process = subprocess.Popen(tts_cmd)

    # Wait for TTS completion but allow interruption
    while tts_process.poll() is None:
        if interrupt_event.is_set():
            tts_process.terminate()
            print("[TTS] Interrupted!")
            return
        await asyncio.sleep(0.1)

    # Send TTS audio to LiveKit
    if local_tts_track:
        await send_wav_to_livekit("temp_response.wav", local_tts_track)

# -----------------------------
# Send WAV to LiveKit
# -----------------------------
async def send_wav_to_livekit(wav_path, track: LocalAudioTrack):
    import soundfile as sf
    data, sr = sf.read(wav_path, dtype='int16')
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr}")
    print("[TTS] Sending audio to LiveKit...")
    # LiveKit expects float32 normalized audio
    float_data = data.astype(np.float32) / 32768.0
    await track.write(float_data)

# -----------------------------
# Mic audio callback
# -----------------------------
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.tobytes())

def start_mic_stream():
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=CHUNK_SIZE,
        callback=audio_callback
    )
    stream.start()
    return stream

# -----------------------------
# LiveKit connection
# -----------------------------
async def connect_livekit():
    global local_tts_track
    room = Room()
    await room.connect(LIVEKIT_URL, ROOM_NAME, USER_NAME)
    print(f"[LiveKit] Connected as {USER_NAME} to room {ROOM_NAME}")

    # Publish local TTS track
    local_tts_track = LocalAudioTrack()
    await room.local_participant.publish_track(
        local_tts_track,
        TrackPublicationOptions()
    )
    print("[LiveKit] Local TTS track published")

    return room

# -----------------------------
# Main
# -----------------------------
async def main():
    # Start ASR thread
    threading.Thread(target=asr_loop, daemon=True).start()
    # Start mic capture
    mic_stream = start_mic_stream()
    # Connect to LiveKit
    room = await connect_livekit()

    print("Chatbot is running. Press Ctrl+C to exit.")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
        audio_queue.put(None)
        mic_stream.stop()
        mic_stream.close()
        if local_tts_track:
            await local_tts_track.stop()
        await room.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
