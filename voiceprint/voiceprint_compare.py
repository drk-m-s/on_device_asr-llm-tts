# voiceprint_compare.pyimport warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="webrtcvad")

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
import librosa

encoder = VoiceEncoder()

def extract_voiceprint(audio_path):
    wav, sr = librosa.load(audio_path, sr=16000)
    wav_preprocessed = preprocess_wav(wav, source_sr=sr)
    return encoder.embed_utterance(wav_preprocessed)

def compare_voiceprints(ref_audio, new_audio, threshold=0.75):
    ref_embed = extract_voiceprint(ref_audio)
    new_embed = extract_voiceprint(new_audio)

    similarity = np.dot(ref_embed, new_embed) / (np.linalg.norm(ref_embed) * np.linalg.norm(new_embed))
    print(f"🔎 Cosine similarity: {similarity:.4f}")
    if similarity > threshold:
        print("✅ Likely same voice")
    else:
        print("❌ Different voice")

if __name__ == "__main__":
    import time
    time_start = time.time()
    compare_voiceprints("tts_ref.wav", "my_voice.wav")
    # compare_voiceprints("tts_ref.wav", "tts_ref_0.wav")
    time_end = time.time()
    print(f"Time cost: {time_end - time_start:.4f}")
