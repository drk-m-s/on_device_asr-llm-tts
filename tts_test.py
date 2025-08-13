# from kittentts import KittenTTS
# m = KittenTTS("KittenML/kitten-tts-nano-0.1")

# audio = m.generate("This high quality TTS model works without a GPU", voice='expr-voice-5-f' )

# # available_voices : [  'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',  'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f' ]

# # Save the audio
# import soundfile as sf
# sf.write('output.wav', audio, 24000)

import wave
from piper import PiperVoice

path_to_voice_file = r"D:\code\smolvlm-realtime-webcam-main\en_US-hfc_female-medium.onnx"

voice = PiperVoice.load(path_to_voice_file)
with wave.open("test.wav", "wb") as wav_file:
    voice.synthesize_wav("Welcome to the world of speech synthesis! What is up my man! Shit is about to go down, you know what I mean?", wav_file)