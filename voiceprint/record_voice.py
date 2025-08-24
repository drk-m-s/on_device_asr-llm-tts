# record_voice.py
import sounddevice as sd
import soundfile as sf

# TEXT = "Hi, this is another five second test sentence for voiceprint."

def record_voice(filename="my_voice.wav", duration=5, samplerate=16000):
    print(f"🎤 Recording {duration} seconds... Speak now!")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    sf.write(filename, audio, samplerate)
    print(f"✅ Saved recording to {filename}")

if __name__ == "__main__":
    record_voice()
