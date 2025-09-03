#!/usr/bin/env python3
"""
Generate a WAV file using the Piper TTS ONNX model en_US-hfc_female-medium.onnx.
The output will read: "Hello, this is a five second test sentence for voiceprint comparison."
"""
from pathlib import Path
import sys
import wave

from piper import PiperVoice

# Text to synthesize
# TEXT = "Hello, this is a five second test sentence for voiceprint comparison."
TEXT = "Hi, this is another five second test sentence for voiceprint and its comparison."


def resolve_model_path() -> Path:
    """Resolve the path to en_US-hfc_female-medium.onnx relative to this file or CWD."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "en_US-hfc_female-medium.onnx",  # repo root
        script_dir / "en_US-hfc_female-medium.onnx",         # alongside this script
        Path.cwd() / "en_US-hfc_female-medium.onnx",         # current working dir
    ]
    for p in candidates:
        if p.is_file():
            return p
    # Fallback: return name in CWD (let Piper raise a helpful error if missing)
    return Path("en_US-hfc_female-medium.onnx")


def generate_wav(text: str = TEXT, out_wav: str | Path | None = None, model_path: str | Path | None = None) -> Path:
    """Synthesize text with Piper and write a 16-bit PCM WAV file incrementally."""
    model_path = Path(model_path) if model_path else resolve_model_path()
    out_wav = Path(out_wav) if out_wav else (Path(__file__).resolve().parent / "tts_ref.wav")

    print(f"Loading Piper voice: {model_path}")
    voice = PiperVoice.load(str(model_path))

    print(f"Synthesizing to: {out_wav}")
    with wave.open(str(out_wav), "wb") as wf:
        initialized = False
        for chunk in voice.synthesize(text):
            if not initialized:
                wf.setnchannels(chunk.sample_channels)
                wf.setsampwidth(chunk.sample_width)  # bytes per sample (e.g., 2 for int16)
                wf.setframerate(chunk.sample_rate)
                initialized = True
            wf.writeframes(chunk.audio_int16_bytes)

    print(f"✅ Saved WAV: {out_wav}")
    return out_wav


if __name__ == "__main__":
    # Optional: allow overriding the text via CLI args
    text = TEXT if len(sys.argv) == 1 else " ".join(sys.argv[1:])
    generate_wav(text=text)
