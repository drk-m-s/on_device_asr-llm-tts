"""
Local LLM to TTS to Speaker Demo (Coqui TTS version)
Streams LLM output through Coqui TTS model to computer speakers with low latency.
Based on vanilla_PiperVoice/llm_tts.py but replaces PiperVoice with Coqui TTS.
"""

import threading
import queue
import time
import os

# Audio libraries
import platform
import numpy as np
if platform.system() != 'Darwin':
    import pyaudio
else:
    import sounddevice as sd
    import numpy as np

# HTTP client for llama-server API
import httpx
import json
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from dataclasses import dataclass
from typing import List
import random

# TTS library (Coqui TTS)
from TTS.api import TTS


class VocabularyStyle(Enum):
    CASUAL = "casual"
    PROFESSIONAL = "professional"
    ENTHUSIASTIC = "enthusiastic"
    THOUGHTFUL = "thoughtful"
    CONCISE = "concise"


@dataclass
class ConversationStyle:
    vocabulary: VocabularyStyle = VocabularyStyle.CASUAL
    personality_traits: List[str] = None
    response_length: str = "medium"  # short, medium, long
    use_filler_words: bool = True
    add_natural_hesitations: bool = True

    def __post_init__(self):
        if self.personality_traits is None:
            self.personality_traits = ["helpful", "friendly", "conversational"]


class LLMTTSStreamer:
    def __init__(self, llm_server_url=None, tts_model_path=None, conversation_style=None):
        # Default TTS model name for Coqui
        if tts_model_path is None:
            tts_model_path = "tts_models/en/ljspeech/tacotron2-DDC"
        if llm_server_url is None:
            llm_server_url = "http://localhost:8080"

        self.llm_server_url = llm_server_url.rstrip('/')
        self.conversation_style = conversation_style

        if self.conversation_style:
            self.filler_words = ["um", "uh", "well", "you know"]
            self.conversation_pauses = ["...", ", ", " - "]

        # Initialize audio
        if platform.system() != 'Darwin':
            self.audio = pyaudio.PyAudio()
        else:
            self.audio_output_active = False
        self.audio_stream = None
        self.audio_queue = queue.Queue()

        # Interruption control
        self.interrupt_tts = threading.Event()
        self.user_speaking = threading.Event()
        self.stream_generation = 0

        # Initialize Coqui TTS
        print("Loading Coqui TTS model...")
        # If a path exists, try loading via model_path; otherwise as model_name
        if os.path.exists(tts_model_path):
            # Attempt local model directory (requires config path typically). If failed, fallback to model_name
            try:
                self.tts = TTS(model_path=tts_model_path)
            except Exception:
                self.tts = TTS(model_name=tts_model_path)
        else:
            self.tts = TTS(model_name=tts_model_path)
        print(f"Coqui TTS model loaded: {tts_model_path}")

        # Determine sample rate from model (fallback to 22050)
        self.tts_sample_rate = getattr(getattr(self.tts, 'synthesizer', None), 'output_sample_rate', None)
        if self.tts_sample_rate is None:
            self.tts_sample_rate = getattr(self.tts, 'output_sample_rate', 22050) or 22050

        # Test LLM server connection
        self.llm_available = self._test_llm_server()
        if self.llm_available:
            print(f"LLM server connected: {self.llm_server_url}")
        else:
            print(f"Warning: Could not connect to LLM server at {self.llm_server_url}")
            print("Make sure llama-server is running with: llama-server --host 0.0.0.0 --port 8080")

        # Threading control
        self.stop_audio = threading.Event()
        self.audio_thread = None

        # Audio format
        self.sample_rate = None
        self.sample_width = None
        self.channels = None

        # HTTP session
        self.session = httpx.Client(
            timeout=httpx.Timeout(connect=0.5, read=30.0, write=1.0, pool=0.5),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=3600),
            http2=True,
            verify=False,
        )

        self.completion_url = f"{self.llm_server_url}/completion"
        self.base_payload = {
            "temperature": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "stream": True,
            "stop": ["Human:", "Assistant:", "\n\n"],
        }
        self.sentence_endings = {'.', '!', '?', '\n'}
        self.tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="TTS")
        self._warm_up_connection()

    def _warm_up_connection(self):
        try:
            self.session.get(f"{self.llm_server_url}/health", timeout=1.0)
            print("Connection warmed up successfully")
        except Exception:
            pass

    def _test_llm_server(self):
        try:
            with httpx.Client(timeout=1.0) as client:
                r = client.get(f"{self.llm_server_url}/health")
                return r.status_code == 200
        except Exception:
            try:
                with httpx.Client(timeout=1.0) as client:
                    r = client.get(f"{self.llm_server_url}/v1/models")
                    return r.status_code == 200
            except Exception:
                return False

    def _build_system_prompt(self, base_prompt="You are a helpful AI assistant."):
        if not self.conversation_style:
            return base_prompt
        style = self.conversation_style
        personality = ", ".join(style.personality_traits)
        vocab_instructions = {
            VocabularyStyle.CASUAL: "Use informal, friendly language with contractions. Be conversational and relaxed.",
            VocabularyStyle.PROFESSIONAL: "Use professional but approachable language. Be clear and articulate.",
            VocabularyStyle.ENTHUSIASTIC: "Use energetic, expressive language. Show excitement and positivity.",
            VocabularyStyle.THOUGHTFUL: "Use reflective, considered language. Take time to think through responses.",
            VocabularyStyle.CONCISE: "Be brief and to-the-point. Avoid unnecessary elaboration.",
        }
        length_guidance = {
            "short": "Keep responses brief (1-2 sentences).",
            "medium": "Provide moderate detail (2-4 sentences).",
            "long": "Give comprehensive responses when appropriate.",
        }
        enhanced_prompt = f"""{base_prompt}

Personality: You are {personality} chatter.

Communication Style: {vocab_instructions[style.vocabulary]}

Response Length: {length_guidance[style.response_length]}

Natural Speech: Use natural speech patterns including occasional hesitations and filler words when appropriate.

- Use natural, conversational language
- Spell out numbers, dates, and abbreviations as they should be spoken
- Use complete sentences that flow naturally when spoken
- Avoid complex punctuation that doesn't translate to speech
- Keep responses clear and direct for audio consumption

- Never include asterisks (*), brackets [], parentheses (), or any formatting symbols
- Never include index numbers, citations, or reference markers
- Never include stage directions, annotations, or meta-commentary
- Never include markup like bold, italics, or bullet points
- Avoid spelling out punctuation or formatting cues
- Write everything as if speaking directly to the listener 
"""
        return enhanced_prompt

    def _enhance_response_naturalness(self, response_text):
        if not self.conversation_style:
            return response_text
        style = self.conversation_style
        enhanced_text = response_text
        if style.use_filler_words and random.random() < 0.3:
            sentences = enhanced_text.split('. ')
            if len(sentences) > 1:
                target_sentence = random.randint(0, min(1, len(sentences) - 1))
                filler = random.choice(self.filler_words)
                sentences[target_sentence] = f"{filler}, {sentences[target_sentence]}"
                enhanced_text = '. '.join(sentences)
        if style.add_natural_hesitations and random.random() < 0.4:
            pause = random.choice(self.conversation_pauses)
            enhanced_text = enhanced_text.replace(' and ', f' and{pause}')
            enhanced_text = enhanced_text.replace(' but ', f' but{pause}')
            enhanced_text = enhanced_text.replace(' so ', f' so{pause}')
        return enhanced_text

    def set_audio_format(self, sample_rate, sample_width, channels):
        if platform.system() != 'Darwin':
            if (self.sample_rate != sample_rate or self.sample_width != sample_width or self.channels != channels):
                if self.audio_stream:
                    self.audio_stream.stop_stream()
                    self.audio_stream.close()
                self.sample_rate = sample_rate
                self.sample_width = sample_width
                self.channels = channels
                self.audio_stream = self.audio.open(
                    format=self.audio.get_format_from_width(sample_width),
                    channels=channels,
                    rate=sample_rate,
                    output=True,
                    frames_per_buffer=256,
                )
                print(f"Audio format: {sample_rate}Hz, {sample_width} bytes, {channels} channels")
        else:
            if (self.sample_rate != sample_rate or self.sample_width != sample_width or self.channels != channels):
                if self.audio_stream:
                    self.audio_stream.close()
                self.sample_rate = sample_rate
                self.sample_width = sample_width
                self.channels = channels
                self.audio_stream = sd.OutputStream(
                    samplerate=sample_rate,
                    channels=channels,
                    dtype='int16',
                    blocksize=256,
                    latency='low',
                )
                self.audio_stream.start()
                print(f"Audio format: {sample_rate}Hz, {sample_width} bytes, {channels} channels")

    def write_raw_data(self, audio_data):
        if (hasattr(self, 'interrupt_tts') and self.interrupt_tts.is_set()) or (hasattr(self, 'user_speaking') and self.user_speaking.is_set()):
            return
        self.audio_queue.put(audio_data)

    def audio_playback_worker(self):
        while not self.stop_audio.is_set():
            try:
                if ((hasattr(self, 'interrupt_tts') and self.interrupt_tts.is_set()) or (hasattr(self, 'user_speaking') and self.user_speaking.is_set())):
                    cleared = 0
                    try:
                        while not self.audio_queue.empty():
                            self.audio_queue.get_nowait()
                            self.audio_queue.task_done()
                            cleared += 1
                    except queue.Empty:
                        pass
                    if cleared > 0:
                        print(f"🗑️ Audio worker cleared {cleared} chunks")
                    if platform.system() == 'Darwin' and hasattr(self, 'audio_output_active') and self.audio_output_active:
                        try:
                            sd.stop()
                            self.audio_output_active = False
                        except Exception:
                            pass
                    time.sleep(0.01)
                    continue
                try:
                    audio_data = self.audio_queue.get(timeout=0.01)
                except queue.Empty:
                    continue
                if ((hasattr(self, 'interrupt_tts') and self.interrupt_tts.is_set()) or (hasattr(self, 'user_speaking') and self.user_speaking.is_set())):
                    self.audio_queue.task_done()
                    continue
                if self.audio_stream and audio_data:
                    try:
                        if platform.system() != 'Darwin':
                            self.audio_stream.write(audio_data)
                        else:
                            audio_array = np.frombuffer(audio_data, dtype=np.int16)
                            if hasattr(self, 'channels') and self.channels > 1:
                                audio_array = audio_array.reshape(-1, self.channels)
                            if hasattr(self, 'audio_output_active'):
                                self.audio_output_active = True
                            sd.play(audio_array, samplerate=self.sample_rate, blocking=False)
                            if hasattr(self, 'interrupt_tts'):
                                chunk_duration = len(audio_array) / self.sample_rate
                                sleep_time = min(chunk_duration, 0.1)
                                time.sleep(sleep_time)
                                if not self.interrupt_tts.is_set() and not self.user_speaking.is_set():
                                    sd.wait()
                            else:
                                sd.wait()
                            if hasattr(self, 'audio_output_active'):
                                self.audio_output_active = False
                    except Exception as e:
                        print(f"Audio playback error: {e}")
                        if hasattr(self, 'audio_output_active'):
                            self.audio_output_active = False
                self.audio_queue.task_done()
            except Exception as e:
                print(f"Audio playback worker error: {e}")
                time.sleep(0.01)

    def start_audio_thread(self):
        if self.audio_thread is None or not self.audio_thread.is_alive():
            self.stop_audio.clear()
            self.audio_thread = threading.Thread(target=self.audio_playback_worker, daemon=True)
            self.audio_thread.start()

    def stop_audio_thread(self):
        self.stop_audio.set()
        if platform.system() == 'Darwin' and hasattr(self, 'audio_output_active') and self.audio_output_active:
            try:
                sd.stop()
                self.audio_output_active = False
            except Exception:
                pass
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=0.1)

    def stream_tts_async(self, text, expected_generation=None):
        if not text.strip():
            return
        local_gen = self.stream_generation if expected_generation is None else expected_generation
        if local_gen != self.stream_generation:
            return
        self.start_audio_thread()
        try:
            # Synthesize with Coqui (returns float waveform [-1, 1])
            wav = self.tts.tts(text=text)
            # Ensure numpy array
            if not isinstance(wav, np.ndarray):
                wav = np.array(wav, dtype=np.float32)
            # Convert to int16 bytes
            wav = np.clip(wav, -1.0, 1.0)
            wav_int16 = (wav * 32767.0).astype(np.int16)
            sample_rate = int(self.tts_sample_rate)
            channels = 1 if wav_int16.ndim == 1 else wav_int16.shape[1]
            self.set_audio_format(sample_rate, 2, channels)
            audio_bytes = wav_int16.tobytes()
            # Chunk and enqueue
            samples_per_chunk = 2048
            bytes_per_sample = 2 * channels
            chunk_size = samples_per_chunk * bytes_per_sample
            for i in range(0, len(audio_bytes), chunk_size):
                if ((hasattr(self, 'interrupt_tts') and self.interrupt_tts.is_set()) or (hasattr(self, 'user_speaking') and self.user_speaking.is_set()) or (local_gen != self.stream_generation)):
                    break
                chunk = audio_bytes[i:i+chunk_size]
                if not chunk:
                    break
                self.write_raw_data(chunk)
        except Exception as e:
            print(f"Coqui TTS synthesis error: {e}")

    def stream_tts(self, text):
        if text.strip():
            self.tts_executor.submit(self.stream_tts_async, text, self.stream_generation)

    def stream_llm_response_ultra_optimized(self, prompt, max_tokens=512, expected_generation=None):
        if not self.llm_available:
            print("LLM server not available!")
            return ""
        local_gen = self.stream_generation if expected_generation is None else expected_generation
        enhanced_prompt = self._build_system_prompt(prompt) if self.conversation_style else prompt
        start_time = time.perf_counter()
        first_token_time = None
        token_count = 0
        payload = {
            "prompt": enhanced_prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "stream": True,
            "stop": ["Human:", "Assistant:", "\n\n"],
        }
        response_text = ""
        text_buffer = ""
        data_prefix = 'data: '
        done_marker = '[DONE]'
        aborted = False
        try:
            with self.session.stream("POST", self.completion_url, json=payload) as response:
                if response.status_code != 200:
                    print(f"Error from LLM server: {response.status_code}")
                    return ""
                buffer = b""
                for chunk in response.iter_bytes(chunk_size=4096):
                    if ((hasattr(self, 'interrupt_tts') and self.interrupt_tts.is_set()) or (hasattr(self, 'user_speaking') and self.user_speaking.is_set()) or (local_gen != self.stream_generation)):
                        aborted = True
                        break
                    if not chunk:
                        continue
                    buffer += chunk
                    while b'\n' in buffer:
                        if ((hasattr(self, 'interrupt_tts') and self.interrupt_tts.is_set()) or (hasattr(self, 'user_speaking') and self.user_speaking.is_set()) or (local_gen != self.stream_generation)):
                            aborted = True
                            break
                        line_bytes, buffer = buffer.split(b'\n', 1)
                        try:
                            line = line_bytes.decode('utf-8', errors='ignore').strip()
                        except Exception:
                            continue
                        if not line.startswith(data_prefix):
                            continue
                        data_str = line[6:]
                        if data_str == done_marker:
                            break
                        try:
                            data = json.loads(data_str)
                            token = None
                            if 'content' in data:
                                token = data['content']
                            elif 'choices' in data and data['choices']:
                                choice = data['choices'][0]
                                if 'text' in choice:
                                    token = choice['text']
                                elif 'delta' in choice and choice['delta'] and 'content' in choice['delta']:
                                    token = choice['delta']['content']
                            if not token:
                                continue
                            if first_token_time is None and token.strip():
                                first_token_time = time.perf_counter()
                                latency_ms = (first_token_time - start_time) * 1000
                                print(f"\n[TIMING] First token latency: {latency_ms:.1f}ms")
                                print("LLM Response: ", end="", flush=True)
                            print(token, end="", flush=True)
                            response_text += token
                            text_buffer += token
                            token_count += 1
                            if (('.' in token or '!' in token or '?' in token or '\n' in token) and len(text_buffer.strip()) > 10):
                                if not ((hasattr(self, 'interrupt_tts') and self.interrupt_tts.is_set()) or (hasattr(self, 'user_speaking') and self.user_speaking.is_set()) or (local_gen != self.stream_generation)):
                                    enhanced_text = self._enhance_response_naturalness(text_buffer.strip())
                                    self.tts_executor.submit(self.stream_tts_async, enhanced_text, local_gen)
                                text_buffer = ""
                        except json.JSONDecodeError:
                            continue
                    if aborted:
                        break
            if text_buffer.strip():
                if not ((hasattr(self, 'interrupt_tts') and self.interrupt_tts.is_set()) or (hasattr(self, 'user_speaking') and self.user_speaking.is_set()) or (local_gen != self.stream_generation)):
                    enhanced_text = self._enhance_response_naturalness(text_buffer.strip())
                    self.tts_executor.submit(self.stream_tts_async, enhanced_text, local_gen)
            if aborted:
                print("\n⏹️ LLM response interrupted by user.")
                return response_text
            total_time = time.perf_counter() - start_time
            if first_token_time and token_count > 1:
                generation_time = time.perf_counter() - first_token_time
                tps = (token_count - 1) / generation_time if generation_time > 0 else 0
                print(f"\n[TIMING] Total response time: {total_time:.1f}s")
                print(f"[TIMING] Tokens per second: {tps:.1f}")
            return response_text
        except Exception as e:
            print(f"Error connecting to LLM server: {e}")
            return ""

    def chat_loop(self):
        if not self.llm_available:
            print("LLM server not available!")
            return
        print("\n=== Local LLM to TTS Chat Demo (Coqui) ===")
        print("Commands: 'quit' to exit, 'clear' to reset history")
        if self.conversation_style:
            print(f"Conversation style: {self.conversation_style.vocabulary.value}")
        print("-" * 40)
        conversation_history = []
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'clear':
                    conversation_history = []
                    print("Conversation history cleared.")
                    continue
                elif not user_input:
                    continue
                self.interrupt_tts.set()
                self.stream_generation += 1
                try:
                    while not self.audio_queue.empty():
                        self.audio_queue.get_nowait()
                        self.audio_queue.task_done()
                except queue.Empty:
                    pass
                if platform.system() == 'Darwin':
                    try:
                        sd.stop()
                        if hasattr(self, 'audio_output_active'):
                            self.audio_output_active = False
                    except Exception:
                        pass
                if conversation_history:
                    prompt = "\n".join(conversation_history) + f"\nHuman: {user_input}\nAssistant:"
                else:
                    prompt = f"Human: {user_input}\nAssistant:"
                self.interrupt_tts.clear()
                print("Assistant: ", end="", flush=True)
                response_text = self.stream_llm_response_ultra_optimized(prompt, expected_generation=self.stream_generation)
                if response_text:
                    conversation_history.append(f"Human: {user_input}")
                    conversation_history.append(f"Assistant: {response_text}")
                    if len(conversation_history) > 12:
                        conversation_history = conversation_history[-12:]
            except KeyboardInterrupt:
                print("\nChat interrupted.")
                break
            except Exception as e:
                print(f"Error: {e}")

    def test_llm_connection(self):
        if not self.llm_available:
            print("LLM server not available!")
            return
        test_prompt = "Hello, how are you?"
        print(f"Testing LLM with: {test_prompt}")
        self.stream_llm_response_ultra_optimized(test_prompt, max_tokens=50)

    def test_tts_only(self, text="Hello! This is a test of the text-to-speech system."):
        print(f"Testing TTS with: {text}")
        self.stream_tts_async(text)
        time.sleep(2)
        while not self.audio_queue.empty():
            time.sleep(0.1)

    def cleanup(self):
        if platform.system() != 'Darwin':
            self.stop_audio_thread()
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            if hasattr(self, 'audio'):
                self.audio.terminate()
            if hasattr(self, 'session'):
                self.session.close()
            if hasattr(self, 'tts_executor'):
                self.tts_executor.shutdown(wait=False)
        else:
            self.stop_audio_thread()
            if self.audio_stream:
                self.audio_stream.close()
            if hasattr(self, 'session'):
                self.session.close()
            if hasattr(self, 'tts_executor'):
                self.tts_executor.shutdown(wait=False)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Local LLM to TTS Streamer (Coqui TTS)")
    parser.add_argument("--llm-url", type=str, default="http://localhost:8080", help="URL of the llama-server (default: http://localhost:8080)")
    parser.add_argument("--tts-model", type=str, default="tts_models/en/ljspeech/tacotron2-DDC", help="Coqui TTS model name or local path")
    parser.add_argument("--test-tts", action="store_true", help="Test TTS only")
    parser.add_argument("--test-llm", action="store_true", help="Test LLM connection only")
    parser.add_argument("--test-all", action="store_true", help="Test both LLM and TTS")
    parser.add_argument("--text", type=str, help="Text to synthesize (for TTS test)")
    parser.add_argument("--conversation-style", type=str, choices=[s.value for s in VocabularyStyle], default="casual", help="Set conversation vocabulary style (default: casual)")
    parser.add_argument("--disable-enhancements", action="store_true", help="Disable conversation enhancements")
    args = parser.parse_args()

    try:
        conversation_style = None
        if not args.disable_enhancements:
            vocab_style = VocabularyStyle(args.conversation_style)
            conversation_style = ConversationStyle(vocabulary=vocab_style)
        streamer = LLMTTSStreamer(
            llm_server_url=args.llm_url,
            tts_model_path=args.tts_model,
            conversation_style=conversation_style,
        )
        if conversation_style:
            print(f"Enhanced conversation enabled with {conversation_style.vocabulary.value} style")
        else:
            print("Enhanced conversation disabled (original behavior)")
        if args.test_tts:
            test_text = args.text or "Hello! This is a test of the local text-to-speech system. It should stream audio directly to your speakers with low latency."
            streamer.test_tts_only(test_text)
        elif args.test_llm:
            streamer.test_llm_connection()
        elif args.test_all:
            print("=== Testing LLM Connection ===")
            streamer.test_llm_connection()
            print("\n=== Testing TTS ===")
            streamer.test_tts_only("This is a test of the complete LLM to TTS pipeline!")
        else:
            streamer.chat_loop()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'streamer' in locals():
            streamer.cleanup()


if __name__ == "__main__":
    main()