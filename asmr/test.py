import numpy as np
import torchaudio
import torch
from pathlib import Path
from scipy.io.wavfile import write
from piper.voice import PiperVoice  # <-- official class

# --- CONFIGURATION ---
VOICE_MODEL_PATH = "/Users/shuyuew/Documents/GitHub/on_device_asr-llm-tts/tts_models/en_US-hfc_female-medium.onnx"   # or zh_CN-huayan-high.onnx
OUTPUT_DIR = Path("tts_output")
OUTPUT_DIR.mkdir(exist_ok=True)
TEXT = "Hello, welcome to your relaxation session. Just breathe, and enjoy the calm. Just breathe, and enjoy the calm... Just breathe, and enjoy the calm..."

# Optional ASMR adjustments
SPEED = 1        # 0.7–1.0 → slower feels calmer
# PITCH_SHIFT = -2     # semitones, negative = lower
PITCH_SHIFT = -3
ADD_BREATH = True
# ADD_BREATH = False
# BREATH_WAV_PATH = "soft_breath.wav"  # pre-recorded breath file
BREATH_WAV_PATH = "single-kiss.wav"  
# BREATH_WAV_PATH = "porno-uh.aiff"  
# BREATH_WAV_PATH = "orgasm-scream.mp3"  
# BREATH_WAV_PATH = "horny-woman-moaning.wav"  

# -------------------------------

# Load Piper voice
voice = PiperVoice.load(VOICE_MODEL_PATH)

# Generate raw audio (int16 stream at 22050 Hz mono by default)
audio_chunks = []
for chunk in voice.synthesize(TEXT):
    audio_chunks.append(chunk.audio_int16_bytes)
audio_bytes = b"".join(audio_chunks)

# Convert bytes → numpy waveform
wav = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
# Get sample rate from the first audio chunk
first_chunk = next(voice.synthesize(TEXT[:10]))  # Get a small sample to extract sample rate
sample_rate = first_chunk.sample_rate

# --- ASMR Adjustments ---

# 1. Speed adjustment
def change_speed(wave, sr, speed=1.0):
    if speed == 1.0:
        return wave, sr
    new_sr = int(sr * speed)
    wave_tensor = torchaudio.functional.resample(
        torch.from_numpy(wave).unsqueeze(0), sr, new_sr
    )
    return wave_tensor.squeeze(0).numpy(), new_sr

wav, sample_rate = change_speed(wav, sample_rate, SPEED)

# 2. Pitch shift
def pitch_shift(wave, sr, semitones):
    if semitones == 0:
        return wave
    rate = 2 ** (semitones / 12)
    new_sr = int(sr * rate)
    shifted = torchaudio.functional.resample(
        torch.from_numpy(wave).unsqueeze(0), sr, new_sr
    )
    return shifted.squeeze(0).numpy()

wav = pitch_shift(wav, sample_rate, PITCH_SHIFT)

# 3. Breath overlay (optional)
if ADD_BREATH and Path(BREATH_WAV_PATH).exists():
    breath_wave, breath_rate = torchaudio.load(BREATH_WAV_PATH)
    # resample to match speech
    breath_wave = torchaudio.functional.resample(breath_wave, breath_rate, sample_rate)
    min_len = min(len(wav), breath_wave.shape[1])
    wav[:min_len] += 0.02 * breath_wave[0, :min_len].numpy()  # scale breath volume

# 4. Normalize
wav = wav / np.max(np.abs(wav)) * 0.95

# Save final ASMR-style audio
output_file = OUTPUT_DIR / "relaxation_tts.wav"
write(str(output_file), sample_rate, (wav * 32767).astype(np.int16))

print(f"✅ ASMR TTS generated: {output_file}")
