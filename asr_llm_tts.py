"""
Complete ASR-LLM-TTS Voice Conversation System
Listens to user speech, processes it through LLM, and responds with synthesized speech.
Supports interruption: user can speak while AI is talking to interrupt it.
"""

import threading
import time
import json
import hashlib
import queue

# Import only what's not already in parent class
import platform
if platform.system() == 'Darwin':
    import sounddevice as sd
    import numpy as np

# ASR library
from RealtimeSTT import AudioToTextRecorder

from llm_tts import LLMTTSStreamer

class VoiceConversationSystem(LLMTTSStreamer):
    def __init__(self, llm_server_url=None, tts_model_path=None,
                 use_http2=True,
                 use_raw_stream=True,
                 pause_asr_during_prefill=True,
                 low_latency_mode=True,
                 enable_history_summarization=True,
                 summarize_after_turns=10,
                 history_trim_threshold=12):
        """
        Initialize the complete voice conversation system with ultra-optimizations.
        Added client-side optimization toggles:
          use_http2: enable/disable HTTP/2 (can disable if adds latency)  
          use_raw_stream: use raw byte parser instead of iter_lines for earlier token flush
          pause_asr_during_prefill: temporarily pause ASR polling while waiting first token
          low_latency_mode: strip advanced sampling (mirostat/typical_p) to speed first token
          enable_history_summarization: summarize older turns to shrink prompt
        """
        # Initialize parent class first
        super().__init__(llm_server_url=llm_server_url, tts_model_path=tts_model_path)
        
        # ASR-specific optimization toggles
        self.use_http2 = use_http2
        self.use_raw_stream = use_raw_stream
        self.pause_asr_during_prefill = pause_asr_during_prefill
        self.low_latency_mode = low_latency_mode
        self.enable_history_summarization = enable_history_summarization
        self.summarize_after_turns = summarize_after_turns
        self.history_trim_threshold = history_trim_threshold
        self.asr_paused = False  # local pause flag (non-invasive)
        
        # Initialize ASR with proper callback handling
        print("Loading ASR model...")
        self.asr_recorder = AudioToTextRecorder(
            enable_realtime_transcription=True,
            silero_sensitivity=0.3,
            silero_use_onnx = True,  # Use ONNX for faster inference
            post_speech_silence_duration = 0.5, 
            # wakeword_backend = "pvporcupine",
            # wake_words = "alexa",
            language="en",
            # on_recording_start=self._on_recording_start,
            # on_recording_stop=self._on_recording_stop,
            on_vad_start = self._on_recording_start,
            on_vad_stop = self._on_recording_stop,
            on_transcription_start=self._on_transcription_start
        )
        print("ASR model loaded and ready")
        
        # ASR-specific conversation control
        self.conversation_active = False
        
        # TTS interruption control - ENHANCED FOR IMMEDIATE INTERRUPTION
        self.tts_playing = threading.Event()  # Flag to track if TTS is playing
        self.interrupt_tts = threading.Event()  # Flag to signal TTS interruption
        self.user_speaking = threading.Event()  # Flag to track if user is speaking
        self.ai_should_be_quiet = threading.Event()  # Flag to prevent AI from speaking during user input
        
        # Conversation history - separate from parent for ASR features
        self.conversation_history = []
        
        # Override base payload for ASR optimizations
        if not self.low_latency_mode:
            self.base_payload.update({
                "typical_p": 0.95,
                "mirostat": 2,
                "mirostat_tau": 5.0,
                "mirostat_eta": 0.1,
            })
        
        # Add macOS audio support flag
        if platform.system() == 'Darwin':
            self.audio_output_active = False

    def _on_recording_start(self, *args, **kwargs):
        """Callback when user starts speaking - interrupt TTS immediately."""
        print("\n🎤 User started speaking...")
        
        # Set user speaking flag immediately
        self.user_speaking.set()
        self.ai_should_be_quiet.set()
        
        # If TTS is playing, interrupt it immediately
        if self.tts_playing.is_set():
            print("⏹️ Interrupting AI speech immediately!")
            self.interrupt_tts_immediately()

    def _on_recording_stop(self, *args, **kwargs):
        """Callback when user stops speaking."""
        print("🎤 User stopped speaking")
        self.user_speaking.clear()
        # Keep AI quiet for a short moment to ensure clean transition
        threading.Timer(0.5, self.ai_should_be_quiet.clear).start()

    def _on_transcription_start(self, *args, **kwargs):
        """Callback when transcription starts - accepts any arguments."""
        print("📝 Transcription starting...")
        # Ensure TTS is completely stopped
        if self.tts_playing.is_set():
            print("⏹️ Stopping TTS for transcription")
            self.interrupt_tts_immediately()

    def interrupt_tts_immediately(self):
        """Immediately interrupt TTS output and clear audio queue."""
        print("🚨 EMERGENCY STOP: Interrupting TTS output NOW!")
        
        # Signal TTS interruption FIRST
        self.interrupt_tts.set()
        self.tts_playing.clear()

        if platform.system() != 'Darwin':
            # Immediately stop audio stream
            if self.audio_stream:
                try:
                    self.audio_stream.stop_stream()
                    print("🔇 Audio stream stopped")
                except Exception as e:
                    print(f"Error stopping audio stream: {e}")
            
            # Clear the entire audio queue aggressively
            cleared_items = 0
            try:
                while True:
                    try:
                        self.audio_queue.get_nowait()
                        self.audio_queue.task_done()
                        cleared_items += 1
                    except queue.Empty:
                        break
            except Exception as e:
                print(f"Error clearing queue: {e}")
            
            if cleared_items > 0:
                print(f"🗑️ Cleared {cleared_items} audio chunks from queue")
            
            # Restart audio stream for clean state
            if self.audio_stream and self.sample_rate:
                try:
                    self.audio_stream.start_stream()
                    print("🔊 Audio stream restarted")
                except Exception as e:
                    print(f"Error restarting audio stream: {e}")
        else:
            # Stop sounddevice stream
            if self.audio_output_active:
                try:
                    sd.stop()
                    self.audio_output_active = False
                    print("🔇 Audio output stopped")
                except Exception as e:
                    print(f"Error stopping audio output: {e}")
            
            # Clear the entire audio queue aggressively
            cleared_items = 0
            try:
                while True:
                    try:
                        self.audio_queue.get_nowait()
                        self.audio_queue.task_done()
                        cleared_items += 1
                    except queue.Empty:
                        break
            except Exception as e:
                print(f"Error clearing queue: {e}")
            
            if cleared_items > 0:
                print(f"🗑️ Cleared {cleared_items} audio chunks from queue")



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
                    frames_per_buffer=128  # Even smaller buffer for instant interruption
                )
        else:
            """Set audio format for sounddevice."""
            self.sample_rate = sample_rate
            self.sample_width = sample_width
            self.channels = channels
        
        print(f"Audio format: {sample_rate}Hz, {sample_width} bytes, {channels} channels")

    def write_raw_data(self, audio_data):
        """Queue audio data for playback - with interruption check."""
        # Double check interruption before queuing
        if not self.interrupt_tts.is_set() and not self.user_speaking.is_set():
            super().write_raw_data(audio_data)

    def audio_playback_worker(self):
        if platform.system() != 'Darwin':
            """Ultra-optimized worker thread for audio playback with instant interruption support."""
            while not self.stop_audio.is_set():
                try:
                    # Check for interruption before getting audio data
                    if self.interrupt_tts.is_set() or self.user_speaking.is_set():
                        # Aggressively clear any remaining audio data when interrupted
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
                        
                        time.sleep(0.01)
                        continue
                    
                    # Get audio data with very short timeout for responsiveness
                    try:
                        audio_data = self.audio_queue.get(timeout=0.005)  # Even faster timeout
                    except queue.Empty:
                        continue
                    
                    # Triple-check interruption before actually playing
                    if (not self.interrupt_tts.is_set() and 
                        not self.user_speaking.is_set() and 
                        self.audio_stream and 
                        audio_data):
                        try:
                            self.audio_stream.write(audio_data)
                        except Exception as e:
                            print(f"Audio write error: {e}")
                    
                    self.audio_queue.task_done()
                    
                except Exception as e:
                    print(f"Audio playback worker error: {e}")
                    time.sleep(0.01)
        else:
            """Ultra-optimized worker thread for audio playback with instant interruption support using sounddevice."""
            while not self.stop_audio.is_set():
                try:
                    # Check for interruption before getting audio data
                    if self.interrupt_tts.is_set() or self.user_speaking.is_set():
                        # Aggressively clear any remaining audio data when interrupted
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
                        
                        # Stop any active sounddevice playback
                        if self.audio_output_active:
                            try:
                                sd.stop()
                                self.audio_output_active = False
                            except:
                                pass
                        
                        time.sleep(0.01)
                        continue
                    
                    # Get audio data with very short timeout for responsiveness
                    try:
                        audio_data = self.audio_queue.get(timeout=0.005)  # Even faster timeout
                    except queue.Empty:
                        continue
                    
                    # Triple-check interruption before actually playing
                    if (not self.interrupt_tts.is_set() and 
                        not self.user_speaking.is_set() and 
                        audio_data):
                        try:
                            # Convert bytes to numpy array for sounddevice
                            if self.sample_width == 2:  # 16-bit audio
                                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                            elif self.sample_width == 4:  # 32-bit audio
                                audio_array = np.frombuffer(audio_data, dtype=np.int32)
                            else:  # Default to 16-bit
                                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                            
                            # Reshape for channels if needed
                            if self.channels > 1:
                                audio_array = audio_array.reshape(-1, self.channels)
                            
                            # Play audio chunk using sounddevice
                            self.audio_output_active = True
                            sd.play(audio_array, samplerate=self.sample_rate, blocking=False)
                            
                            # Wait for the chunk to finish or be interrupted
                            chunk_duration = len(audio_array) / self.sample_rate
                            sleep_time = min(chunk_duration, 0.1)  # Check for interruption frequently
                            time.sleep(sleep_time)
                            
                            # Wait for playback to complete if not interrupted
                            if not self.interrupt_tts.is_set() and not self.user_speaking.is_set():
                                sd.wait()
                            
                            self.audio_output_active = False
                            
                        except Exception as e:
                            print(f"Audio playback error: {e}")
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
        if platform.system() == 'Darwin':
            if self.audio_output_active:
                try:
                    sd.stop()
                    self.audio_output_active = False
                except:
                    pass
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=0.1)

    def stream_tts_async(self, text):
        """
        Asynchronous TTS processing with enhanced interruption support.
        Extends parent implementation with ASR-specific interruption handling.
        """
        if not text.strip():
            return
        
        # Don't start TTS if user is speaking or should be quiet
        if self.user_speaking.is_set() or self.ai_should_be_quiet.is_set():
            print(f"🔇 Skipping TTS (user active): {text[:50]}...")
            return
            
        print(f"🔊 Synthesizing: {text[:50]}...")
        
        # Clear interruption flag and set TTS playing flag
        self.interrupt_tts.clear()
        self.tts_playing.set()
        
        try:
            # Override parent's synthesis to add interruption checks
            self.start_audio_thread()
            
            chunk_count = 0
            for chunk in self.voice.synthesize(text):
                chunk_count += 1
                
                # Check for interruption before processing each chunk
                if self.interrupt_tts.is_set() or self.user_speaking.is_set():
                    print(f"🔇 TTS interrupted at chunk {chunk_count}")
                    break
                
                # Use parent's audio format setting
                self.set_audio_format(
                    chunk.sample_rate, 
                    chunk.sample_width, 
                    chunk.sample_channels
                )
                
                # Use parent's audio writing method
                self.write_raw_data(chunk.audio_int16_bytes)
                
                # Frequent interruption checks during synthesis
                if chunk_count % 5 == 0:
                    if self.interrupt_tts.is_set() or self.user_speaking.is_set():
                        print(f"🔇 TTS interrupted during synthesis at chunk {chunk_count}")
                        break
                
                time.sleep(0.0005)  # Allow interruption detection
                
        except Exception as e:
            print(f"TTS error: {e}")
        finally:
            self.tts_playing.clear()
            print("🔊 TTS synthesis completed")

    def stream_tts(self, text):
        """Submit TTS task to thread pool for parallel processing with ASR guards."""
        if text.strip() and not self.user_speaking.is_set() and not self.ai_should_be_quiet.is_set():
            # Reuse parent's thread pool and our overridden async method
            self.tts_executor.submit(self.stream_tts_async, text)

    def _pause_asr(self):
        """Lightweight ASR pause (client-side) to free CPU during critical prefill."""
        if not self.asr_paused:
            self.asr_paused = True
            # If library exposes a pause/stop we could call it here.
            # try: self.asr_recorder.pause() except: pass
            # Minimally stop calling .text() loop while paused.

    def _resume_asr(self):
        if self.asr_paused:
            self.asr_paused = False
            # try: self.asr_recorder.resume() except: pass

    def _summarize_history(self):
        """Summarize older history turns into a compact form to shrink prompt.
        Simple heuristic: take all but last 4 entries, truncate each line, join.
        Replace with a single summary line at beginning.
        """
        if len(self.conversation_history) < self.summarize_after_turns:
            return
        older = self.conversation_history[:-4]
        recent = self.conversation_history[-4:]
        # Naive compression (could be replaced with a local summarizer)
        compressed = []
        for line in older:
            if len(line) > 160:
                compressed.append(line[:157] + '...')
            else:
                compressed.append(line)
        summary = "Summary: " + " | ".join(compressed)
        if len(summary) > 1000:  # Hard cap
            summary = summary[:997] + '...'
        self.conversation_history = [summary] + recent
        print("📝 History summarized (length reduced)")

    def process_llm_response_ultra_optimized(self, user_input):
        """Ultra-optimized LLM processing with detailed latency instrumentation & new client optimizations."""
        if not self.llm_available:
            print("❌ LLM server not available!")
            return
        print(f"🤖 Processing: {user_input}")
        # ==== Phase 1: Prompt build timing ===
        t0 = time.perf_counter()
        if self.conversation_history:
            recent_history = self.conversation_history[-8:]
            prompt = "\n".join(recent_history) + f"\nHuman: {user_input}\nAssistant:"
        else:
            prompt = f"Human: {user_input}\nAssistant:"
        import hashlib
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        t_build_done = time.perf_counter()
        first_token_time = None
        first_sse_line_time = None
        first_byte_time = None
        token_count = 0
        # Build payload fresh each time (avoid unintended mutation)
        payload = self.base_payload.copy()
        payload.update({
            "prompt": prompt,
            "max_tokens": 256,
            "top_k": 20,
        })
        # If low-latency mode, ensure advanced sampling removed (double safety)
        if self.low_latency_mode:
            for k in ["typical_p", "mirostat", "mirostat_tau", "mirostat_eta"]:
                payload.pop(k, None)
        else:
            # If not low latency and user wants, can add advanced sampling here
            pass
        if not hasattr(self, 'context_cache'):
            self.context_cache = {}
        if not hasattr(self, 'last_prompt_hash'):
            self.last_prompt_hash = None
        if self.last_prompt_hash and prompt.startswith(self.context_cache.get(self.last_prompt_hash, "")):
            payload["continue"] = True
        response_text = ""
        text_buffer = ""
        data_prefix = 'data: '
        headers = {"Accept": "text/event-stream"}
        print("🤖 Assistant: ", end="", flush=True)
        # Optionally pause ASR to free resources during prefill until first token
        if self.pause_asr_during_prefill:
            self._pause_asr()
        try:
            t_req_start = time.perf_counter()
            with self.session.stream("POST", self.completion_url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    print(f"❌ Error from LLM server: {response.status_code}")
                    return
                # Raw vs line-based streaming
                if self.use_raw_stream:
                    buffer = b""
                    for raw in response.iter_raw():  # quickest access to incoming bytes
                        if self.user_speaking.is_set():
                            print("\n🎤 User speaking - stopping LLM generation...")
                            break
                        if first_byte_time is None:
                            first_byte_time = time.perf_counter()
                        buffer += raw
                        # Process complete lines
                        while b'\n' in buffer:
                            line_bytes, buffer = buffer.split(b'\n', 1)
                            if not line_bytes:
                                continue
                            try:
                                line = line_bytes.decode('utf-8', errors='ignore').strip()
                            except:
                                continue
                            if not line:
                                continue
                            if first_sse_line_time is None:
                                first_sse_line_time = time.perf_counter()
                            if not line.startswith(data_prefix):
                                continue
                            data_str = line[len(data_prefix):].strip()
                            if data_str == '[DONE]':
                                buffer = b""  # drop remaining
                                break
                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                            token = self._extract_token_fast(data)
                            if not token:
                                continue
                            if first_token_time is None and token.strip():
                                first_token_time = time.perf_counter()
                                # Resume ASR now that first token arrived
                                if self.pause_asr_during_prefill:
                                    self._resume_asr()
                                print(f"\n⏱️ First token latency: {(first_token_time - t0)*1000:.1f}ms")
                                print("🤖 Assistant: ", end="", flush=True)
                            print(token, end="", flush=True)
                            response_text += token
                            text_buffer += token
                            token_count += 1
                            if (any(end in token for end in ['.', '!', '?']) and len(text_buffer.strip()) > 5 \
                                and not self.user_speaking.is_set()):
                                self.tts_executor.submit(self.stream_tts_async, text_buffer.strip())
                                text_buffer = ""
                        # Early exit if finished
                        if buffer == b"" and self.user_speaking.is_set():
                            break
                else:
                    # First byte arrival
                    first_byte_time = time.perf_counter()
                    for line in response.iter_lines():
                        if self.user_speaking.is_set():
                            print("\n🎤 User speaking - stopping LLM generation...")
                            break
                        if not line:
                            continue
                        if first_sse_line_time is None:
                            first_sse_line_time = time.perf_counter()
                        if not line.startswith(data_prefix):
                            continue
                        data_str = line[len(data_prefix):].strip()
                        if data_str == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        token = self._extract_token_fast(data)
                        if not token:
                            continue
                        if first_token_time is None and token.strip():
                            first_token_time = time.perf_counter()
                            if self.pause_asr_during_prefill:
                                self._resume_asr()
                            print(f"\n⏱️ First token latency: {(first_token_time - t0)*1000:.1f}ms")
                            print("🤖 Assistant: ", end="", flush=True)
                        print(token, end="", flush=True)
                        response_text += token
                        text_buffer += token
                        token_count += 1
                        if (any(end in token for end in ['.', '!', '?']) and len(text_buffer.strip()) > 5 \
                            and not self.user_speaking.is_set()):
                            self.tts_executor.submit(self.stream_tts_async, text_buffer.strip())
                            text_buffer = ""
        except Exception as e:
            print(f"❌ Error connecting to LLM server: {e}")
            return
        finally:
            # Safety: resume ASR if no token arrived (error or abort)
            if self.pause_asr_during_prefill and first_token_time is None:
                self._resume_asr()
        # ==== Phase 3: Finalize ===
        if text_buffer.strip() and not self.user_speaking.is_set():
            self.tts_executor.submit(self.stream_tts_async, text_buffer.strip())
        self.context_cache[prompt_hash] = prompt
        self.last_prompt_hash = prompt_hash
        if not self.user_speaking.is_set():
            self.conversation_history.append(f"Human: {user_input}")
            self.conversation_history.append(f"Assistant: {response_text}")
            # Summarization / trimming logic
            if self.enable_history_summarization and len(self.conversation_history) > self.summarize_after_turns:
                self._summarize_history()
            # Hard trim to avoid runaway history size
            if len(self.conversation_history) > self.history_trim_threshold:
                self.conversation_history = self.conversation_history[-self.history_trim_threshold:]
        # ==== Phase 4: Timing breakdown ===
        t_end = time.perf_counter()
        def ms(a, b):
            return (b - a) * 1000 if (a and b) else None
        build_ms = ms(t0, t_build_done)
        req_overhead_ms = ms(t_build_done, t_req_start)
        first_byte_ms = ms(t_req_start, first_byte_time) if first_byte_time else None
        sse_line_ms = ms(t_req_start, first_sse_line_time) if first_sse_line_time else None
        first_tok_ms = ms(t0, first_token_time) if first_token_time else None
        total_ms = ms(t0, t_end)
        gen_phase_ms = ms(first_token_time, t_end) if first_token_time else None
        print("\n[TIMING BREAKDOWN]" )
        if build_ms is not None: print(f"  build_prompt: {build_ms:.1f} ms")
        if req_overhead_ms is not None: print(f"  pre_request_gap (python idle): {req_overhead_ms:.1f} ms")
        if first_byte_ms is not None: print(f"  network+server_first_byte: {first_byte_ms:.1f} ms")
        if sse_line_ms is not None: print(f"  first_sse_line_after_send: {sse_line_ms:.1f} ms")
        if first_tok_ms is not None: print(f"  first_token_latency_total: {first_tok_ms:.1f} ms")
        if gen_phase_ms is not None and token_count>1:
            tps = (token_count-1)/(gen_phase_ms/1000)
            print(f"  generation_phase: {gen_phase_ms:.1f} ms | tokens: {token_count} | tps: {tps:.1f}")
        if total_ms is not None: print(f"  end_to_end: {total_ms:.1f} ms")
        if first_tok_ms and first_tok_ms > 500:
            hints = []
            if first_byte_ms and first_byte_ms > 200: hints.append("Check server busy / model size / quantization")
            if (first_sse_line_time and first_byte_time and (first_sse_line_time-first_byte_time)>0.15): hints.append("Server buffering before flush")
            if req_overhead_ms and req_overhead_ms>50: hints.append("Python scheduling delay")
            if build_ms and build_ms>50: hints.append("Trim prompt / summarize earlier")
            if hints:
                print("  HINTS: " + "; ".join(hints))

    def _extract_token_fast(self, data):
        """Fast token extraction with minimal checks."""
        # Try most common format first
        if 'content' in data:
            return data['content']
        
        # Try choices format
        choices = data.get('choices')
        if choices and choices[0]:
            choice = choices[0]
            
            # Try text field
            if 'text' in choice:
                return choice['text']
            
            # Try delta content
            delta = choice.get('delta')
            if delta and 'content' in delta:
                return delta['content']
        
        return None

    def process_speech_input(self, text):
        """
        Callback function for ASR - processes recognized speech.
        
        Args:
            text: Recognized text from speech
        """
        if not text.strip():
            return
            
        # Filter out very short or noise inputs
        if len(text.strip()) < 3:
            return
            
        print(f"\n👤 You said: {text}")
        
        # Check for exit commands
        if text.lower().strip() in ['quit', 'exit', 'stop', 'goodbye']:
            print("👋 Goodbye!")
            self.conversation_active = False
            return
        
        # Check for clear command
        if 'clear' in text.lower() or 'reset' in text.lower():
            self.conversation_history = []
            print("🗑️ Conversation history cleared.")
            # Only respond if AI is not being interrupted
            if not self.user_speaking.is_set():
                self.stream_tts_async("Conversation history cleared.")
            return
        
        # Process through LLM and respond with TTS using ultra-optimized method
        self.process_llm_response_ultra_optimized(text)

    def start_voice_conversation(self):
        """Start the voice conversation loop."""
        if not self.llm_available:
            print("❌ LLM server not available! Please start llama-server on localhost:8080")
            print("Command: llama-server --host 0.0.0.0 --port 8080 --model /path/to/your/model.gguf")
            return
        
        print("\n🎙️ === Voice Conversation System ===")
        print("🔊 Speak naturally to have a conversation with the AI")
        print("🎤 You can interrupt the AI anytime by speaking")
        print("📢 Say 'quit', 'exit', or 'stop' to end the conversation")
        print("🗑️ Say 'clear' or 'reset' to clear conversation history")
        print("-" * 50)
        
        welcome_msg = "Hello! I'm ready to chat with you. You can interrupt me anytime by speaking. Please speak whenever you're ready."
        print(f"🤖 {welcome_msg}")
        self.stream_tts_async(welcome_msg)
        
        self.conversation_active = True
        
        try:
            print("\n🎤 Listening... (speak now)")
            while self.conversation_active:
                try:
                    if not self.asr_paused:  # skip polling when paused
                        self.asr_recorder.text(self.process_speech_input)
                    time.sleep(0.1)
                except Exception as e:
                    print(f"❌ ASR Error: {e}")
                    time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️ Voice conversation interrupted.")
        finally:
            self.conversation_active = False

    def test_components(self):
        """Test all components individually."""
        print("\n🧪 === Testing Components ===")
        
        # Test TTS
        print("1. Testing TTS...")
        test_text = "Hello! This is a test of the text-to-speech system."
        self.stream_tts_async(test_text)
        time.sleep(3)  # Wait for TTS to finish
        
        # Test ASR
        print("2. Testing ASR...")
        print("🎤 Please say something (you have 5 seconds)...")
        
        def test_asr_callback(text):
            print(f"✅ ASR recognized: {text}")
        
        # Capture speech for 5 seconds
        start_time = time.time()
        while time.time() - start_time < 5:
            try:
                self.asr_recorder.text(test_asr_callback)
                time.sleep(0.1)
            except:
                break
        
        # Test LLM
        print("3. Testing LLM...")
        if self.llm_available:
            self.process_llm_response_ultra_optimized("Hello, can you hear me?")
        else:
            print("❌ LLM server not available")
        
        print("✅ Component testing complete!")

    def cleanup(self):
        """Clean up resources including ASR-specific ones."""
        # ASR-specific cleanup
        self.conversation_active = False
        self.interrupt_tts.set()  # Stop any ongoing TTS
        
        # Call parent cleanup for common resources
        super().cleanup()


def main():
    """Main function to run the voice conversation system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Conversation System (ASR-LLM-TTS)")
    parser.add_argument("--llm-url", type=str, default="http://localhost:8080", 
                       help="URL of the llama-server (default: http://localhost:8080)")
    parser.add_argument("--tts-model", type=str, default="en_US-hfc_female-medium.onnx", 
                       help="Path to Piper TTS model")
    parser.add_argument("--test", action="store_true", help="Test all components")
    parser.add_argument("--test-tts", action="store_true", help="Test TTS only")
    
    args = parser.parse_args()
    
    # Initialize the voice conversation system
    try:
        system = VoiceConversationSystem(
            llm_server_url=args.llm_url,
            tts_model_path=args.tts_model
        )
        
        if args.test_tts:
            # Test TTS only
            test_text = "Hello! This is a test of the voice conversation system. The text-to-speech is working correctly. Try speaking to interrupt this message."
            system.stream_tts_async(test_text)
            time.sleep(3)
        elif args.test:
            # Test all components
            system.test_components()
        else:
            # Start voice conversation
            system.start_voice_conversation()
            
    except KeyboardInterrupt:
        print("\n⏹️ Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'system' in locals():
            system.cleanup()


if __name__ == "__main__":
    main()