import os
import io
import time
import queue
import threading
import subprocess
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import sounddevice as sd
import webrtcvad
import requests

# ========== CONFIG ==========
SAMPLE_RATE = 16000          # 16kHz mono for ASR/VAD
BLOCK_SIZE = 160             # 10 ms frames at 16kHz (160 samples)
CHANNELS = 1
VAD_AGGRESSIVENESS = 2       # 0..3 (higher -> more aggressive speech detection)
UTTERANCE_MIN_MS = 250       # ignore ultra-short blips
ENDPOINT_MS = 400            # silence length to consider utterance ended
INTERRUPT_ON_SPEECH = True

# LLM endpoint (you provided this)
LLM_URL = "http://localhost:8080/completion"
LLM_N_PREDICT = 128

# TTS backend: "pyttsx3" or "piper"
TTS_BACKEND = os.environ.get("TTS_BACKEND", "pyttsx3")  # default offline & easy
# If you choose Piper, set these:
PIPER_BIN = os.environ.get("PIPER_BIN", "piper")
PIPER_VOICE = os.environ.get("PIPER_VOICE", "")  # e.g., "/path/to/en_US-amy-medium.onnx"
PIPER_OUT = "tts_out.wav"

# faster-whisper config (if you select it)
ASR_BACKEND = os.environ.get("ASR_BACKEND", "faster-whisper")  # or "vosk"
# FAST_WHISPER_MODEL_DIR = os.environ.get("FAST_WHISPER_MODEL_DIR", "models/ggml-small")
FAST_WHISPER_MODEL_DIR = "/Users/shuyuew/Documents/GitHub/on_device_asr-llm-tts/temp/ggml-small"
# ============================


# --------- Utility: ring buffer to hold raw PCM -----------
class AudioRing:
    def __init__(self, seconds: float, sr: int):
        self.maxlen = int(seconds * sr)
        self.buf = np.zeros(self.maxlen, dtype=np.float32)
        self.write_pos = 0

    def push(self, chunk: np.ndarray):
        n = len(chunk)
        if n >= self.maxlen:
            self.buf[:] = chunk[-self.maxlen:]
            self.write_pos = 0
            return
        end = self.write_pos + n
        if end <= self.maxlen:
            self.buf[self.write_pos:end] = chunk
        else:
            first = self.maxlen - self.write_pos
            self.buf[self.write_pos:] = chunk[:first]
            self.buf[:n - first] = chunk[first:]
        self.write_pos = (self.write_pos + n) % self.maxlen

    def tail(self, n: int) -> np.ndarray:
        if n >= self.maxlen:
            return self.buf.copy()
        start = (self.write_pos - n) % self.maxlen
        if start + n <= self.maxlen:
            return self.buf[start:start + n].copy()
        first = self.maxlen - start
        return np.concatenate([self.buf[start:], self.buf[:n - first]])


# --------- VAD frame helper ----------
def pcm16_from_float32(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    ints = (x * 32767.0).astype(np.int16)
    return ints.tobytes()

def chunker(arr: np.ndarray, size: int):
    for i in range(0, len(arr), size):
        yield arr[i:i + size]

@dataclass
class Utterance:
    audio: np.ndarray  # float32 mono 16k
    started_at: float
    ended_at: float


# --------- ASR backends ----------
class ASR:
    def transcribe(self, audio_f32_16k: np.ndarray) -> str:
        raise NotImplementedError

class FasterWhisperASR(ASR):
    def __init__(self, model_dir: str):
        from faster_whisper import WhisperModel
        # CPU-friendly; change compute_type if you have GPU
        self.model = WhisperModel(model_dir, device="cpu", compute_type="int8")
    def transcribe(self, audio_f32_16k: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio_f32_16k, language=None, vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments).strip()

class VoskASR(ASR):
    def __init__(self, model_dir: str):
        import vosk
        self.model = vosk.Model(model_dir)
        self.sr = SAMPLE_RATE
    def transcribe(self, audio_f32_16k: np.ndarray) -> str:
        import vosk, json
        rec = vosk.KaldiRecognizer(self.model, self.sr)
        rec.SetWords(True)
        pcm = pcm16_from_float32(audio_f32_16k)
        rec.AcceptWaveform(pcm)
        res = json.loads(rec.FinalResult())
        return (res.get("text") or "").strip()


# --------- TTS backends ----------
class TTSBase:
    def say(self, text: str, interrupt_event: threading.Event):
        raise NotImplementedError

class Pyttsx3TTS(TTSBase):
    def __init__(self):
        import pyttsx3
        self.engine = pyttsx3.init()
        # Optional voice/rate tuning:
        # for v in self.engine.getProperty('voices'): print(v.id)
        # self.engine.setProperty('voice', 'com.apple.speech.synthesis.voice.samantha')
        # self.engine.setProperty('rate', 180)
        # We'll drive with a background loop so we can interrupt.
    def say(self, text: str, interrupt_event: threading.Event):
        # pyttsx3 has no clean "stop mid-utterance" via API across platforms, so we chunk.
        tokens = text.split()
        chunk = []
        for w in tokens:
            if interrupt_event.is_set():
                break
            chunk.append(w)
            if len(chunk) >= 12:  # speak in small bursts
                self.engine.say(" ".join(chunk))
                self.engine.runAndWait()
                chunk = []
        if (not interrupt_event.is_set()) and chunk:
            self.engine.say(" ".join(chunk))
            self.engine.runAndWait()

class PiperTTS(TTSBase):
    def __init__(self, piper_bin: str, voice_path: str, out_path: str):
        self.bin = piper_bin
        self.voice = voice_path
        self.out = out_path
        if not self.voice:
            raise RuntimeError("Piper voice not set. Set PIPER_VOICE to your .onnx voice.")
    def say(self, text: str, interrupt_event: threading.Event):
        # 1) synthesize wav via piper CLI
        proc = subprocess.Popen(
            [self.bin, "--model", self.voice, "--output_file", self.out],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        proc.stdin.write(text.encode("utf-8"))
        proc.stdin.close()
        proc.wait()
        # 2) stream-play the wav so we can stop on interrupt
        import soundfile as sf
        data, sr = sf.read(self.out, dtype="float32", always_2d=False)
        stream = sd.OutputStream(samplerate=sr, channels=1, dtype='float32')
        stream.start()
        idx = 0
        CHUNK = 1024
        while idx < len(data):
            if interrupt_event.is_set():
                break
            end = min(idx + CHUNK, len(data))
            stream.write(data[idx:end])
            idx = end
        stream.stop()
        stream.close()


# --------- LLM client ----------
def llm_complete(prompt: str, n_predict: int = LLM_N_PREDICT) -> str:
    try:
        r = requests.post(LLM_URL, json={"prompt": prompt, "n_predict": n_predict}, timeout=60)
        r.raise_for_status()
        j = r.json()
        # Many local servers return {"content":"..."} or {"text":"..."}; handle common keys:
        for k in ("text", "content", "response"):
            if k in j and isinstance(j[k], str):
                return j[k]
        # Fallback: concatenate any strings
        return " ".join(v for v in j.values() if isinstance(v, str)) or ""
    except Exception as e:
        return f"Sorry, the local LLM endpoint failed: {e}"


# --------- Global control flags ----------
tts_playing_lock = threading.Lock()
tts_playing = False  # gate ASR when robot is speaking
interrupt_event = threading.Event()  # used to interrupt TTS instantly

def set_tts_playing(state: bool):
    global tts_playing
    with tts_playing_lock:
        tts_playing = state

def is_tts_playing() -> bool:
    with tts_playing_lock:
        return tts_playing


# --------- Audio capture + VAD + utterance segmentation ----------
class CaptureWorker(threading.Thread):
    def __init__(self, out_q: "queue.Queue[Utterance]"):
        super().__init__(daemon=True)
        self.out_q = out_q
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.buffer = io.BytesIO()
        self.started_at = None
        self.last_voice_ms = None
        self.temp_float = np.zeros(0, dtype=np.float32)
        self.frame_ms = int(1000 * BLOCK_SIZE / SAMPLE_RATE)
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        def callback(indata, frames, time_info, status):
            if status:
                pass  # ignore over/underflows
            # Gate mic when robot is speaking (prevents self-feedback)
            if is_tts_playing():
                return
            mono = indata[:, 0].astype(np.float32).copy()
            # Resample not needed if stream uses 16k; we set samplerate=16k below.
            # VAD expects 16-bit PCM at 10/20/30ms frames
            for chunk in chunker(mono, BLOCK_SIZE):
                if len(chunk) < BLOCK_SIZE:
                    # pad last partial frame
                    pad = np.zeros(BLOCK_SIZE - len(chunk), dtype=np.float32)
                    chunk = np.concatenate([chunk, pad])
                pcm = pcm16_from_float32(chunk)
                is_speech = self.vad.is_speech(pcm, SAMPLE_RATE)
                now_ms = int(time.time() * 1000)

                if is_speech:
                    if self.started_at is None:
                        self.started_at = now_ms
                        self.buffer = io.BytesIO()
                    self.buffer.write(pcm)
                    self.last_voice_ms = now_ms
                    # If we detect speech while TTS is playing (race condition), raise interrupt
                    if INTERRUPT_ON_SPEECH and is_tts_playing():
                        interrupt_event.set()
                else:
                    # No speech this frame
                    if self.started_at is not None and self.last_voice_ms is not None:
                        if now_ms - self.last_voice_ms >= ENDPOINT_MS:
                            # finalize utterance
                            dur = self.last_voice_ms - self.started_at
                            if dur >= UTTERANCE_MIN_MS:
                                pcm_bytes = self.buffer.getvalue()
                                # Convert PCM16 -> float32
                                audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                                utt = Utterance(audio=audio, started_at=self.started_at/1000.0, ended_at=self.last_voice_ms/1000.0)
                                self.out_q.put(utt)
                            self.started_at = None
                            self.last_voice_ms = None
                            self.buffer = io.BytesIO()

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32', blocksize=BLOCK_SIZE, callback=callback):
            while self.running:
                time.sleep(0.05)


# --------- Orchestrator ----------
class Assistant:
    def __init__(self, asr: ASR, tts: TTSBase):
        self.asr = asr
        self.tts = tts

    def handle_utterance(self, utt: Utterance):
        # Transcribe
        text = self.asr.transcribe(utt.audio)
        if not text:
            return
        print(f"[USER] {text}")

        # Interrupt any ongoing TTS (if any)
        interrupt_event.set()

        # Query LLM
        reply = llm_complete(text, n_predict=LLM_N_PREDICT)
        print(f"[BOT]  {reply}")

        # Speak reply (clear the interrupt flag first)
        interrupt_event.clear()
        set_tts_playing(True)
        try:
            self.tts.say(reply, interrupt_event)
        finally:
            set_tts_playing(False)
            interrupt_event.clear()


def main():
    # Choose ASR
    if ASR_BACKEND == "vosk":
        asr = VoskASR(model_dir=FAST_WHISPER_MODEL_DIR)
    else:
        asr = FasterWhisperASR(model_dir=FAST_WHISPER_MODEL_DIR)

    # Choose TTS
    if TTS_BACKEND.lower() == "piper":
        tts = PiperTTS(PIPER_BIN, PIPER_VOICE, PIPER_OUT)
    else:
        tts = Pyttsx3TTS()

    bot = Assistant(asr=asr, tts=tts)

    utter_q: "queue.Queue[Utterance]" = queue.Queue()
    cap = CaptureWorker(utter_q)
    cap.start()

    print("Ready. Speak to the robot. (Ctrl+C to exit)")
    try:
        while True:
            try:
                utt = utter_q.get(timeout=0.1)
            except queue.Empty:
                # If user starts talking while bot is speaking, pre-empt TTS:
                if INTERRUPT_ON_SPEECH and is_tts_playing() and interrupt_event.is_set():
                    # stop audio immediately
                    sd.stop()  # stops any OutputStream currently playing
                continue

            # As soon as we have a finalized utterance, process it
            bot.handle_utterance(utt)

    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
        time.sleep(0.2)


if __name__ == "__main__":
    main()
