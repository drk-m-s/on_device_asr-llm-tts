"""
Local LLM to TTS to Speaker Demo
Streams LLM output through TTS model to computer speakers with low latency.
"""

import threading
import queue
import time

# Audio libraries
import platform
if platform.system() != 'Darwin':
    import pyaudio
else:
    ## for mac os, use sounddevice
    import sounddevice as sd
    import numpy as np

# HTTP client for llama-server API
import httpx  # Much faster than requests
import json
from concurrent.futures import ThreadPoolExecutor

# TTS library
from piper import PiperVoice


class LLMTTSStreamer:
    def __init__(self, llm_server_url=None, tts_model_path=None):
        """
        Initialize the LLM-TTS streamer with maximum optimization.
        """
        # Default paths
        if tts_model_path is None:
            tts_model_path = "en_US-hfc_female-medium.onnx"
        
        if llm_server_url is None:
            llm_server_url = "http://localhost:8080"
        
        self.llm_server_url = llm_server_url.rstrip('/')
        
        # Initialize audio
        if platform.system() != 'Darwin':
            self.audio = pyaudio.PyAudio()
        else:
            self.audio_output_active = False
        self.audio_stream = None
        self.audio_queue = queue.Queue()
        
        # Interruption control flags (can be overridden by subclasses)
        self.interrupt_tts = threading.Event()
        self.user_speaking = threading.Event()
        # Stream generation counter for cancellation of stale responses
        self.stream_generation = 0
        
        # Initialize TTS
        print("Loading TTS model...")
        self.voice = PiperVoice.load(tts_model_path)
        print(f"TTS model loaded: {tts_model_path}")
        
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
        
        # Audio format (will be set when TTS starts)
        self.sample_rate = None
        self.sample_width = None
        self.channels = None
        
        # ULTRA-OPTIMIZED HTTP SESSION WITH CONNECTION POOLING
        self.session = httpx.Client(
            timeout=httpx.Timeout(
                connect=0.5,    # Ultra-fast connection timeout
                read=30.0,      # Read timeout for streaming
                write=1.0,      # Ultra-fast write timeout
                pool=0.5        # Ultra-fast pool timeout
            ),
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=50,
                keepalive_expiry=3600  # Keep connections alive longer
            ),
            http2=True,
            verify=False,  # Skip SSL verification for localhost
        )
        
        # Pre-compile everything possible
        self.completion_url = f"{self.llm_server_url}/completion"
        self.base_payload = {
            "temperature": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "stream": True,
            "stop": ["Human:", "Assistant:", "\n\n"]
        }
        
        # Pre-compile sentence endings for faster checking
        self.sentence_endings = {'.', '!', '?', '\n'}
        
        # TTS thread pool for parallel processing
        self.tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="TTS")
        
        # WARM UP CONNECTION - Pre-establish connection to reduce first request latency
        self._warm_up_connection()

    def _warm_up_connection(self):
        """Pre-establish connection to reduce first request latency."""
        try:
            # Make a quick health check to establish connection
            self.session.get(f"{self.llm_server_url}/health", timeout=1.0)
            print("Connection warmed up successfully")
        except:
            pass

    def _test_llm_server(self):
        """Test if the LLM server is accessible with optimized connection."""
        try:
            with httpx.Client(timeout=1.0) as client:
                response = client.get(f"{self.llm_server_url}/health")
                return response.status_code == 200
        except:
            try:
                with httpx.Client(timeout=1.0) as client:
                    response = client.get(f"{self.llm_server_url}/v1/models")
                    return response.status_code == 200
            except:
                return False

    def set_audio_format(self, sample_rate, sample_width, channels):
        if platform.system() != 'Darwin':
            """Set audio format and initialize PyAudio stream."""
            if (self.sample_rate != sample_rate or 
                self.sample_width != sample_width or 
                self.channels != channels):
                
                # Close existing stream
                if self.audio_stream:
                    self.audio_stream.stop_stream()
                    self.audio_stream.close()
                
                # Update format
                self.sample_rate = sample_rate
                self.sample_width = sample_width
                self.channels = channels
                
                # Open new stream with ultra-optimized buffer
                self.audio_stream = self.audio.open(
                    format=self.audio.get_format_from_width(sample_width),
                    channels=channels,
                    rate=sample_rate,
                    output=True,
                    frames_per_buffer=256  # Even smaller buffer for ultra-low latency
                )
                print(f"Audio format: {sample_rate}Hz, {sample_width} bytes, {channels} channels")
        else:
            """Set audio format and initialize sounddevice stream."""
            if (self.sample_rate != sample_rate or 
                self.sample_width != sample_width or 
                self.channels != channels):
                
                # Close existing stream
                if self.audio_stream:
                    self.audio_stream.close()
                
                # Update format
                self.sample_rate = sample_rate
                self.sample_width = sample_width
                self.channels = channels
                
                # Open new stream with low latency
                self.audio_stream = sd.OutputStream(
                    samplerate=sample_rate,
                    channels=channels,
                    dtype='int16',
                    blocksize=256,
                    latency='low'
                )
                self.audio_stream.start()
                print(f"Audio format: {sample_rate}Hz, {sample_width} bytes, {channels} channels")

    def write_raw_data(self, audio_data):
        """Queue audio data for playback - with optional interruption check."""
        # Check for interruption if flags exist (for subclass compatibility)
        if (hasattr(self, 'interrupt_tts') and self.interrupt_tts.is_set()) or \
           (hasattr(self, 'user_speaking') and self.user_speaking.is_set()):
            return
        self.audio_queue.put(audio_data)

    def audio_playback_worker(self):
        """Ultra-optimized worker thread for audio playback with interruption support."""
        while not self.stop_audio.is_set():
            try:
                # Check for interruption before getting audio data
                if (hasattr(self, 'interrupt_tts') and self.interrupt_tts.is_set()) or \
                   (hasattr(self, 'user_speaking') and self.user_speaking.is_set()):
                    # Clear any remaining audio data when interrupted
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
                    
                    # Stop any active audio output
                    if platform.system() == 'Darwin' and hasattr(self, 'audio_output_active') and self.audio_output_active:
                        try:
                            sd.stop()
                            self.audio_output_active = False
                        except:
                            pass
                    
                    time.sleep(0.01)
                    continue
                
                # Get audio data with timeout for responsiveness
                timeout = 0.005 if hasattr(self, 'interrupt_tts') else 0.01
                try:
                    audio_data = self.audio_queue.get(timeout=timeout)
                except queue.Empty:
                    continue
                
                # Check interruption again before playing
                if (hasattr(self, 'interrupt_tts') and self.interrupt_tts.is_set()) or \
                   (hasattr(self, 'user_speaking') and self.user_speaking.is_set()):
                    self.audio_queue.task_done()
                    continue
                
                if self.audio_stream and audio_data:
                    try:
                        if platform.system() != 'Darwin':
                            self.audio_stream.write(audio_data)
                        else:
                            # Convert bytes to numpy array for sounddevice
                            if hasattr(self, 'sample_width') and self.sample_width == 4:
                                audio_array = np.frombuffer(audio_data, dtype=np.int32)
                            else:
                                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                            
                            # Reshape for channels if needed
                            if hasattr(self, 'channels') and self.channels > 1:
                                audio_array = audio_array.reshape(-1, self.channels)
                            
                            # Play audio chunk using sounddevice
                            if hasattr(self, 'audio_output_active'):
                                self.audio_output_active = True
                            sd.play(audio_array, samplerate=self.sample_rate, blocking=False)
                            
                            # Wait for the chunk to finish or be interrupted
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
        """Start the audio playback thread."""
        if self.audio_thread is None or not self.audio_thread.is_alive():
            self.stop_audio.clear()
            self.audio_thread = threading.Thread(target=self.audio_playback_worker, daemon=True)
            self.audio_thread.start()

    def stop_audio_thread(self):
        """Stop the audio playback thread."""
        self.stop_audio.set()
        if platform.system() == 'Darwin' and hasattr(self, 'audio_output_active') and self.audio_output_active:
            try:
                sd.stop()
                self.audio_output_active = False
            except:
                pass
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=0.1)

    def stream_tts_async(self, text, expected_generation=None):
        """Asynchronous TTS processing to avoid blocking. Cancels if generation changes or interrupted."""
        if not text.strip():
            return
        
        # Capture generation snapshot for this TTS task
        local_gen = self.stream_generation if expected_generation is None else expected_generation
        
        # If already stale before starting, abort early
        if local_gen != self.stream_generation:
            return
        
        self.start_audio_thread()
        
        for chunk in self.voice.synthesize(text):
            # Abort if interrupted or generation changed
            if ((hasattr(self, 'interrupt_tts') and self.interrupt_tts.is_set()) or
                (hasattr(self, 'user_speaking') and self.user_speaking.is_set()) or
                (local_gen != self.stream_generation)):
                break
            
            self.set_audio_format(
                chunk.sample_rate, 
                chunk.sample_width, 
                chunk.sample_channels
            )
            
            # Last-moment check before enqueueing audio
            if ((hasattr(self, 'interrupt_tts') and self.interrupt_tts.is_set()) or
                (hasattr(self, 'user_speaking') and self.user_speaking.is_set()) or
                (local_gen != self.stream_generation)):
                break
            
            self.write_raw_data(chunk.audio_int16_bytes)

    def stream_tts(self, text):
        """Submit TTS task to thread pool for parallel processing."""
        if text.strip():
            # Pass current generation so the task can self-cancel if stale
            self.tts_executor.submit(self.stream_tts_async, text, self.stream_generation)

    def stream_llm_response_ultra_optimized(self, prompt, max_tokens=512, expected_generation=None):
        """Ultra-optimized LLM response streaming with cancellation support."""
        if not self.llm_available:
            print("LLM server not available!")
            return ""
        
        # Capture the generation this stream belongs to
        local_gen = self.stream_generation if expected_generation is None else expected_generation
        
        start_time = time.perf_counter()
        first_token_time = None
        token_count = 0
        
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "stream": True,
            "stop": ["Human:", "Assistant:", "\n\n"]
        }
        
        response_text = ""
        text_buffer = ""
        
        data_prefix = 'data: '
        done_marker = '[DONE]'
        
        aborted = False
        
        try:
            # Use raw bytes processing for maximum speed
            with self.session.stream("POST", self.completion_url, json=payload) as response:

                if response.status_code != 200:
                    print(f"Error from LLM server: {response.status_code}")
                    return ""
                
                # Ultra-fast streaming with minimal overhead
                buffer = b""
                for chunk in response.iter_bytes(chunk_size=4096):  # Larger chunks for efficiency
                    # Abort immediately if interrupted, user is speaking, or generation changed
                    if ((hasattr(self, 'interrupt_tts') and self.interrupt_tts.is_set()) or
                        (hasattr(self, 'user_speaking') and self.user_speaking.is_set()) or
                        (local_gen != self.stream_generation)):
                        aborted = True
                        break

                    if not chunk:
                        continue

                    buffer += chunk
                    
                    # Process complete lines only
                    while b'\n' in buffer:
                        # Check interruption and generation inside inner loop as well
                        if ((hasattr(self, 'interrupt_tts') and self.interrupt_tts.is_set()) or
                            (hasattr(self, 'user_speaking') and self.user_speaking.is_set()) or
                            (local_gen != self.stream_generation)):
                            aborted = True
                            break

                        line_bytes, buffer = buffer.split(b'\n', 1)

                        try:
                            line = line_bytes.decode('utf-8', errors='ignore').strip()
                        except:
                            continue

                        if not line.startswith(data_prefix):
                            continue

                        data_str = line[6:]  # Remove 'data: ' prefix
                        if data_str == done_marker:
                            break

                        try:
                            # Ultra-fast JSON parsing with minimal error checking
                            data = json.loads(data_str)

                            # Extract token with absolute minimal checks
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
                            
                            # Record first token time IMMEDIATELY
                            if first_token_time is None and token.strip():
                                first_token_time = time.perf_counter()
                                latency_ms = (first_token_time - start_time) * 1000
                                print(f"\n[TIMING] First token latency: {latency_ms:.1f}ms")
                                print("LLM Response: ", end="", flush=True)

                            print(token, end="", flush=True)
                            response_text += token
                            text_buffer += token
                            token_count += 1
                            
                            # Ultra-fast sentence detection
                            if (('.' in token or '!' in token or '?' in token or '\n' in token) 
                                and len(text_buffer.strip()) > 10):
                                # Submit to TTS immediately without blocking, unless interrupted or generation changed
                                if not ((hasattr(self, 'interrupt_tts') and self.interrupt_tts.is_set()) or
                                        (hasattr(self, 'user_speaking') and self.user_speaking.is_set()) or
                                        (local_gen != self.stream_generation)):
                                    self.tts_executor.submit(self.stream_tts_async, text_buffer.strip(), local_gen)
                                text_buffer = ""

                        except json.JSONDecodeError:
                            continue
                    
                    if aborted:
                        break
            
            # Handle remaining text
            if text_buffer.strip():
                # Only submit if not interrupted and generation unchanged
                if not ((hasattr(self, 'interrupt_tts') and self.interrupt_tts.is_set()) or
                        (hasattr(self, 'user_speaking') and self.user_speaking.is_set()) or
                        (local_gen != self.stream_generation)):
                    self.tts_executor.submit(self.stream_tts_async, text_buffer.strip(), local_gen)
            
            # If aborted due to interruption, stop early
            if aborted:
                print("\n⏹️ LLM response interrupted by user.")
                return response_text
            
            # Calculate performance metrics
            total_time = time.perf_counter() - start_time
            if first_token_time and token_count > 1:
                generation_time = time.perf_counter() - first_token_time
                tokens_per_second = (token_count - 1) / generation_time if generation_time > 0 else 0
                print(f"\n[TIMING] Total response time: {total_time:.1f}s")
                print(f"[TIMING] Tokens per second: {tokens_per_second:.1f}")
            
            return response_text

        except Exception as e:
            print(f"Error connecting to LLM server: {e}")
            return ""

    def chat_loop(self):
        """Interactive chat loop with ultra-optimized response handling."""
        if not self.llm_available:
            print("LLM server not available!")
            return
        
        print("\n=== Local LLM to TTS Chat Demo ===")
        print("Type 'quit' to exit")
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
                
                # Interrupt any ongoing TTS and cancel previous response generation
                self.interrupt_tts.set()
                self.stream_generation += 1
                
                # Flush any remaining audio to prevent overlap with the new response
                try:
                    while not self.audio_queue.empty():
                        self.audio_queue.get_nowait()
                        self.audio_queue.task_done()
                except queue.Empty:
                    pass
                
                # On macOS, ensure any active playback stops immediately
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
                
                # Allow new TTS to play for the upcoming response
                self.interrupt_tts.clear()
                
                print("Assistant: ", end="", flush=True)
                response_text = self.stream_llm_response_ultra_optimized(prompt, expected_generation=self.stream_generation)
                if response_text:
                    conversation_history.append(f"Human: {user_input}")
                    conversation_history.append(f"Assistant: {response_text}")
                    
                    # Keep only last 6 exchanges
                    if len(conversation_history) > 12:
                        conversation_history = conversation_history[-12:]

            except KeyboardInterrupt:
                print("\nChat interrupted.")
                break
            except Exception as e:
                print(f"Error: {e}")

    def test_llm_connection(self):
        """Test LLM server with a simple prompt."""
        if not self.llm_available:
            print("LLM server not available!")
            return

        test_prompt = "Hello, how are you?"
        print(f"Testing LLM with: {test_prompt}")
        self.stream_llm_response_ultra_optimized(test_prompt, max_tokens=50)

    def test_tts_only(self, text="Hello! This is a test of the text-to-speech system."):
        """Test TTS functionality without LLM."""
        print(f"Testing TTS with: {text}")
        self.stream_tts_async(text)  # Use the async version directly
        
        # Wait for audio to finish
        time.sleep(2)
        while not self.audio_queue.empty():
            time.sleep(0.1)

    def cleanup(self):
        if platform.system() != 'Darwin':
            """Clean up resources."""
            self.stop_audio_thread()
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            if self.audio:
                self.audio.terminate()
            if hasattr(self, 'session'):
                self.session.close()
            if hasattr(self, 'tts_executor'):
                self.tts_executor.shutdown(wait=False)
        else:
            """Clean up resources."""
            self.stop_audio_thread()
            if self.audio_stream:
                self.audio_stream.close()
            if hasattr(self, 'session'):
                self.session.close()
            if hasattr(self, 'tts_executor'):
                self.tts_executor.shutdown(wait=False)


def main():
    """Main function to run the demo."""
    import argparse
    parser = argparse.ArgumentParser(description="Local LLM to TTS Streamer")
    parser.add_argument("--llm-url", type=str, default="http://localhost:8080", help="URL of the llama-server (default: http://localhost:8080)")
    parser.add_argument("--tts-model", type=str, default="en_US-hfc_female-medium.onnx", help="Path to Piper TTS model")
    parser.add_argument("--test-tts", action="store_true", help="Test TTS only")
    parser.add_argument("--test-llm", action="store_true", help="Test LLM connection only")
    parser.add_argument("--test-all", action="store_true", help="Test both LLM and TTS")
    parser.add_argument("--text", type=str, help="Text to synthesize (for TTS test)")
    args = parser.parse_args()

    try:
        streamer = LLMTTSStreamer(
            llm_server_url=args.llm_url,
            tts_model_path=args.tts_model
        )

        if args.test_tts: # Test TTS only
            test_text = args.text or "Hello! This is a test of the local text-to-speech system. It should stream audio directly to your speakers with low latency."
            streamer.test_tts_only(test_text)
        elif args.test_llm: # Test LLM only
            streamer.test_llm_connection()
        elif args.test_all: # Test both
            print("=== Testing LLM Connection ===")
            streamer.test_llm_connection()
            print("\n=== Testing TTS ===")
            streamer.test_tts_only("This is a test of the complete LLM to TTS pipeline!")
        else: # Start chat loop
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
