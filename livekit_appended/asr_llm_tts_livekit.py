"""Complete ASR-LLM-TTS Voice Conversation System
Listens to user speech, processes it through LLM, and responds with synthesized speech.
Supports interruption: user can speak while AI is talking to interrupt it.
"""

import threading
import time
import queue
import asyncio

try:
    import numpy as _np  # used for optional channel downmixing
except Exception:
    _np = None

from collections import deque

# ASR library
from RealtimeSTT import AudioToTextRecorder

# Import parent class
from llm_tts_livekit import LLMTTSStreamer


class SoftwareEchoCanceller:
    """Lightweight software echo canceller using far-end reference.
    - Maintains a far-end buffer (reference) at a fixed sample rate (e.g., 16 kHz, mono)
    - For each near-end frame, finds best delay via cross-correlation within a window
    - Subtracts scaled reference from near-end to suppress echo
    This is a simplified AEC for practicality and does not replace full WebRTC AEC.
    """

    def __init__(self, sample_rate: int = 16000, max_delay_ms: int = 200, frame_ms: int = 20, buffer_seconds: float = 2.0):
        self.enabled = True if _np is not None else False
        self.sample_rate = int(sample_rate)
        self.frame_len = int(self.sample_rate * frame_ms / 1000)
        self.max_delay = int(self.sample_rate * max_delay_ms / 1000)
        self._farend = deque(maxlen=int(self.sample_rate * buffer_seconds))
        self._lock = threading.Lock()
        self._eps = 1e-8

    def set_sample_rate(self, sample_rate: int):
        self.sample_rate = int(sample_rate)
        self.frame_len = max(1, int(self.sample_rate * 20 / 1000))
        self.max_delay = max(self.frame_len, int(self.sample_rate * 200 / 1000))
        with self._lock:
            self._farend = deque(list(self._farend), maxlen=int(self.sample_rate * 2.0))

    def _downmix_mono(self, pcm: _np.ndarray, channels: int) -> _np.ndarray:
        if channels <= 1:
            return pcm
        pcm = pcm.astype(_np.int32)
        pcm = pcm.reshape(-1, channels).mean(axis=1)
        return _np.clip(pcm, -32768, 32767).astype(_np.int16)

    def _resample_int16(self, data: _np.ndarray, in_rate: int, out_rate: int) -> _np.ndarray:
        if in_rate == out_rate or len(data) == 0:
            return data
        # Linear interpolation resampling
        x = _np.arange(len(data), dtype=_np.float32)
        n_out = max(1, int(round(len(data) * out_rate / float(in_rate))))
        x_new = _np.linspace(0, len(data) - 1, n_out, dtype=_np.float32)
        y = _np.interp(x_new, x, data.astype(_np.float32))
        return _np.clip(_np.round(y), -32768, 32767).astype(_np.int16)

    def feed_farend(self, audio_bytes: bytes, in_rate: int, channels: int = 1):
        if not self.enabled or audio_bytes is None:
            return
        try:
            pcm = _np.frombuffer(audio_bytes, dtype=_np.int16)
            if channels and channels > 1:
                pcm = self._downmix_mono(pcm, channels)
            pcm = self._resample_int16(pcm, in_rate, self.sample_rate)
            if pcm.size == 0:
                return
            with self._lock:
                self._farend.extend(pcm.tolist())
        except Exception:
            pass

    def _cancel_frame(self, near_frame: _np.ndarray) -> _np.ndarray:
        # near_frame is int16 mono at self.sample_rate
        if near_frame.size == 0:
            return near_frame
        with self._lock:
            if len(self._farend) < self.frame_len:
                return near_frame
            # Take last (frame_len + max_delay) samples of farend for alignment
            take = min(len(self._farend), self.frame_len + self.max_delay)
            segment = _np.array(list(self._farend)[-take:], dtype=_np.float32)
        near = near_frame.astype(_np.float32)
        if segment.size < self.frame_len:
            return near_frame
        # Cross-correlation to find best alignment
        try:
            c = _np.correlate(segment, near, mode='valid')  # length: take - frame_len + 1
            idx = int(_np.argmax(_np.abs(c)))
            fseg = segment[idx: idx + self.frame_len]
            denom = _np.dot(fseg, fseg) + self._eps
            g = float(_np.dot(near, fseg) / denom)
            enhanced = near - g * fseg
            # Optional residual attenuation when correlation is high
            corr_norm = float(_np.abs(c[idx]) / ((_np.linalg.norm(near) * _np.linalg.norm(fseg)) + self._eps))
            if corr_norm > 0.3:
                enhanced *= 0.9
            return _np.clip(_np.round(enhanced), -32768, 32767).astype(_np.int16)
        except Exception:
            return near_frame

    def process_nearend(self, audio_bytes: bytes) -> bytes:
        if not self.enabled or audio_bytes is None:
            return audio_bytes
        try:
            pcm = _np.frombuffer(audio_bytes, dtype=_np.int16)
            if pcm.size == 0:
                return audio_bytes
            out = _np.empty_like(pcm)
            N = self.frame_len
            # Process in frames
            for i in range(0, len(pcm), N):
                frame = pcm[i:i + N]
                if frame.size < N:
                    # Zero-pad last frame for processing
                    pad = _np.zeros(N, dtype=_np.int16)
                    pad[:frame.size] = frame
                    processed = self._cancel_frame(pad)
                    out[i:i + frame.size] = processed[:frame.size]
                else:
                    out[i:i + N] = self._cancel_frame(frame)
            return out.tobytes()
        except Exception:
            return audio_bytes

    def farend_rms(self, window_ms: int = 100) -> float:
        """Estimate RMS energy of the far-end reference over the recent window."""
        if not self.enabled:
            return 0.0
        try:
            win = max(1, int(self.sample_rate * window_ms / 1000))
            with self._lock:
                if len(self._farend) == 0:
                    return 0.0
                seg = _np.array(list(self._farend)[-win:], dtype=_np.float32)
            if seg.size == 0:
                return 0.0
            return float(_np.sqrt(_np.mean(seg * seg)))
        except Exception:
            return 0.0


class VoiceConversationSystem(LLMTTSStreamer):
    """Voice conversation system that inherits LLM-TTS functionality."""
    
    def __init__(self, llm_server_url=None, tts_model_path=None,
                 asr_model="tiny",
                 enable_history_summarization=True,
                 summarize_after_turns=10,
                 history_trim_threshold=12):
        """
        Initialize the complete voice conversation system.
        
        Args:
            llm_server_url: URL of the LLM server (default: http://localhost:8080)
            tts_model_path: Path to the TTS model file
            asr_model: Fast_Whisper ASR model size ('tiny', 'base', 'small', 'medium', 'large')
            enable_history_summarization: Whether to summarize older turns to shrink prompt
            summarize_after_turns: Number of turns after which to summarize
            history_trim_threshold: Maximum number of history entries to keep
        """
        # Initialize parent class first
        super().__init__(llm_server_url=llm_server_url, tts_model_path=tts_model_path)
        
        # Persist ASR model spec for potential re-initialization when LiveKit ASR is enabled
        self._asr_model_spec = asr_model
        
        # ASR-specific configuration
        self.enable_history_summarization = enable_history_summarization
        self.summarize_after_turns = summarize_after_turns
        self.history_trim_threshold = history_trim_threshold
        
        # Initialize ASR with proper callback handling
        print(f"Loading ASR model: {asr_model}")
        self.asr_recorder = AudioToTextRecorder(
            model=asr_model,
            enable_realtime_transcription=True,
            silero_sensitivity=0.3,
            silero_use_onnx=True,
            post_speech_silence_duration=0.5,
            language="en",
            on_vad_start=self._on_recording_start,
            on_vad_stop=self._on_recording_stop,
            on_transcription_start=self._on_transcription_start
        )
        print("ASR model loaded and ready")
        
        # ASR-specific conversation control
        self.conversation_active = False
        
        # TTS interruption control
        self.interrupt_tts = threading.Event()
        self.user_speaking = threading.Event()
        self.ai_should_be_quiet = threading.Event()
        
        # LiveKit ASR input state
        self._lk_in_enabled = False
        self._lk_in_stream = None
        self._lk_in_thread = None
        self._lk_in_stop = threading.Event()
        self._lk_in_target_rate = 16000
        self._lk_in_target_channels = 1
        self._lk_in_rtc = None
        self._lk_in_resampler = None
        self._lk_in_input_rate = None
        self._lk_in_input_channels = None

        # Software AEC
        self.enable_aec = True
        self._aec = SoftwareEchoCanceller(sample_rate=self._lk_in_target_rate) if _np is not None else None
        
        # Conversation history
        self.conversation_history = []

        # ASR gating related to AI speaking
        self.ai_talking = threading.Event()
        self._asr_tail_mute_until = 0.0  # monotonic seconds
        self._aec_rms_threshold = 900.0  # int16 RMS threshold to drop frames during AI speech
        # Require multiple consecutive frames to consider barge-in voice valid
        self._barge_in_frames_required = 4
        self._barge_in_counter = 0

    def _on_recording_start(self, *args, **kwargs):
        """Callback when user starts speaking - interrupt TTS immediately."""
        print("\n🎤 User started speaking...")
        # Increment generation so any ongoing LLM/TTS becomes stale
        if hasattr(self, 'stream_generation'):
            self.stream_generation += 1
        # Set user speaking flags
        self.user_speaking.set()
        self.ai_should_be_quiet.set()
        # Interrupt any ongoing TTS immediately
        print("⏹️ Interrupting AI speech immediately!")
        self.interrupt_tts_immediately()

    def _on_recording_stop(self, *args, **kwargs):
        """Callback when user stops speaking."""
        print("🎤 User stopped speaking")
        # Allow TTS to resume now that user stopped speaking
        self.user_speaking.clear()
        self.interrupt_tts.clear()
        # Keep AI quiet for a short moment to ensure clean transition and add short ASR tail mute
        try:
            self._asr_tail_mute_until = max(getattr(self, '_asr_tail_mute_until', 0.0), time.monotonic() + 0.2)
        except Exception:
            pass
        try:
            threading.Timer(0.4, self.ai_should_be_quiet.clear).start()
        except Exception:
            pass

    def _on_transcription_start(self, *args, **kwargs):
        """Callback when transcription starts."""
        print("📝 Transcription starting...")
        # Briefly prevent new speech to avoid overlap, then allow
        self.ai_should_be_quiet.set()
        try:
            threading.Timer(0.3, self.ai_should_be_quiet.clear).start()
        except Exception:
            pass

    def enable_livekit_asr_input(self, audio_stream, target_sample_rate: int = 16000, target_channels: int = 1):
        """Enable feeding ASR from a LiveKit AudioStream.
        This re-initializes the ASR recorder to disable microphone and starts a background consumer.
        Args:
            audio_stream: An instance of livekit.rtc.AudioStream constructed from a subscribed track.
            target_sample_rate: Sample rate to feed to ASR (default 16000).
            target_channels: Number of channels to feed to ASR (default 1).
        """
        try:
            from livekit import rtc as _rtc
        except Exception as e:
            raise RuntimeError("livekit package is required for LiveKit ASR input. Please install it with `pip install livekit`.") from e

        # Recreate ASR recorder with microphone disabled
        self.asr_recorder = AudioToTextRecorder(
            model=self._asr_model_spec,
            enable_realtime_transcription=True,
            silero_sensitivity=0.3,
            silero_use_onnx=True,
            post_speech_silence_duration=0.5,
            language="en",
            on_vad_start=self._on_recording_start,
            on_vad_stop=self._on_recording_stop,
            on_transcription_start=self._on_transcription_start,
            use_microphone=False,
        )

        # Store LiveKit pieces
        self._lk_in_rtc = _rtc
        self._lk_in_stream = audio_stream
        self._lk_in_target_rate = int(target_sample_rate)
        self._lk_in_target_channels = int(target_channels)
        self._lk_in_stop.clear()
        self._lk_in_enabled = True

        # Update AEC sample rate
        if self._aec is not None:
            self._aec.set_sample_rate(self._lk_in_target_rate)

        # Reset tail mute on enabling
        self._asr_tail_mute_until = 0.0

        # Start background thread with its own asyncio loop to consume frames
        if self._lk_in_thread is None or not self._lk_in_thread.is_alive():
            self._lk_in_thread = threading.Thread(target=self._run_livekit_asr_consumer, daemon=True)
            self._lk_in_thread.start()
        print(f"LiveKit ASR input enabled @ {self._lk_in_target_rate}Hz, {self._lk_in_target_channels}ch")

    def disable_livekit_asr_input(self):
        """Stop LiveKit ASR input consumer and revert to normal microphone capture."""
        self._lk_in_enabled = False
        self._lk_in_stop.set()
        if self._lk_in_thread and self._lk_in_thread.is_alive():
            self._lk_in_thread.join(timeout=0.5)
        self._lk_in_thread = None
        self._lk_in_stream = None
        self._lk_in_resampler = None

        # Recreate ASR recorder back with microphone enabled
        self.asr_recorder = AudioToTextRecorder(
            model=self._asr_model_spec,
            enable_realtime_transcription=True,
            silero_sensitivity=0.3,
            silero_use_onnx=True,
            post_speech_silence_duration=0.5,
            language="en",
            on_vad_start=self._on_recording_start,
            on_vad_stop=self._on_recording_stop,
            on_transcription_start=self._on_transcription_start,
            use_microphone=True,
        )

    def _run_livekit_asr_consumer(self):
        """Thread target: create an asyncio loop and run the LiveKit ASR consumer."""
        try:
            asyncio.run(self._livekit_asr_consumer_loop())
        except Exception as e:
            print(f"❌ LiveKit ASR consumer error: {e}")

    async def _livekit_asr_consumer_loop(self):
        """Consume audio frames from LiveKit AudioStream, resample, and feed to ASR."""
        if not self._lk_in_stream:
            return
        # Reset resampler state
        self._lk_in_resampler = None
        self._lk_in_input_rate = None
        self._lk_in_input_channels = None

        # Async iterate over LiveKit audio frames
        try:
            async for event in self._lk_in_stream:
                if self._lk_in_stop.is_set() or not self._lk_in_enabled:
                    break
                frame = event.frame if hasattr(event, 'frame') else event
                try:
                    in_rate = getattr(frame, 'sample_rate', 48000)
                    in_channels = getattr(frame, 'num_channels', 1)
                    # Obtain raw int16 bytes
                    raw_mv = frame.data  # memoryview
                    raw_bytes = raw_mv.cast('b') if hasattr(raw_mv, 'cast') else bytes(raw_mv)

                    # If we are in the short tail mute after TTS, drop frames early
                    if time.monotonic() < getattr(self, '_asr_tail_mute_until', 0.0):
                        continue

                    # Reset barge-in counter when AI not talking
                    if not self.ai_talking.is_set():
                        self._barge_in_counter = 0

                    # Optional downmix to mono if needed using numpy if available
                    if in_channels != self._lk_in_target_channels:
                        if _np is not None:
                            pcm = _np.frombuffer(raw_bytes, dtype=_np.int16)
                            if in_channels > 1:
                                pcm = pcm.reshape(-1, in_channels).mean(axis=1).astype(_np.int16)
                            raw_bytes = pcm.tobytes()
                            in_channels = 1
                        else:
                            # Fallback: take first channel by slicing
                            raw_bytes = raw_bytes[0::in_channels]
                            in_channels = 1

                    # Resample if needed
                    if in_rate != self._lk_in_target_rate:
                        if (self._lk_in_resampler is None or
                                self._lk_in_input_rate != in_rate or
                                self._lk_in_input_channels != in_channels):
                            self._lk_in_resampler = self._lk_in_rtc.AudioResampler(
                                input_rate=in_rate,
                                output_rate=self._lk_in_target_rate,
                                num_channels=in_channels,
                            )
                            self._lk_in_input_rate = in_rate
                            self._lk_in_input_channels = in_channels
                        out_bytes_list = []
                        for rframe in self._lk_in_resampler.push(bytearray(raw_bytes)):
                            out_bytes_list.append(rframe.data.cast('b'))
                        if out_bytes_list:
                            try:
                                near_bytes = b''.join(out_bytes_list)
                                # Apply software echo cancellation
                                if getattr(self, 'enable_aec', False) and getattr(self, '_aec', None) is not None:
                                    near_bytes = self._aec.process_nearend(near_bytes)
                                # Energy-based gating during AI speech to avoid loops while allowing barge-in
                                if self.ai_talking.is_set():
                                    if _np is not None and len(near_bytes) > 0:
                                        arr = _np.frombuffer(near_bytes, dtype=_np.int16).astype(_np.float32)
                                        near_rms = float(_np.sqrt(_np.mean(arr * arr)) if arr.size else 0.0)
                                        fe_rms = 0.0
                                        if getattr(self, 'enable_aec', False) and getattr(self, '_aec', None) is not None:
                                            try:
                                                fe_rms = float(self._aec.farend_rms(window_ms=120))
                                            except Exception:
                                                fe_rms = 0.0
                                        dyn_thr = max(self._aec_rms_threshold, fe_rms * 0.85)
                                        if near_rms < dyn_thr:
                                            self._barge_in_counter = 0
                                            # Likely just echo; skip feeding to ASR
                                            continue
                                        else:
                                            self._barge_in_counter += 1
                                            if self._barge_in_counter < self._barge_in_frames_required:
                                                # Require stability over multiple frames to accept barge-in
                                                continue
                                self.asr_recorder.feed_audio(near_bytes)
                            except Exception as e:
                                print(f"ASR feed_audio error: {e}")
                    else:
                        try:
                            near_bytes = raw_bytes
                            # Apply software echo cancellation
                            if getattr(self, 'enable_aec', False) and getattr(self, '_aec', None) is not None:
                                near_bytes = self._aec.process_nearend(near_bytes)
                            # Energy-based gating during AI speech to avoid loops while allowing barge-in
                            if self.ai_talking.is_set():
                                if _np is not None and len(near_bytes) > 0:
                                    arr = _np.frombuffer(near_bytes, dtype=_np.int16).astype(_np.float32)
                                    near_rms = float(_np.sqrt(_np.mean(arr * arr)) if arr.size else 0.0)
                                    fe_rms = 0.0
                                    if getattr(self, 'enable_aec', False) and getattr(self, '_aec', None) is not None:
                                        try:
                                            fe_rms = float(self._aec.farend_rms(window_ms=120))
                                        except Exception:
                                            fe_rms = 0.0
                                    dyn_thr = max(self._aec_rms_threshold, fe_rms * 0.85)
                                    if near_rms < dyn_thr:
                                        self._barge_in_counter = 0
                                        continue
                                    else:
                                        self._barge_in_counter += 1
                                        if self._barge_in_counter < self._barge_in_frames_required:
                                            continue
                            self.asr_recorder.feed_audio(near_bytes)
                        except Exception as e:
                            print(f"ASR feed_audio error: {e}")
                except Exception as e:
                    print(f"LiveKit frame processing error: {e}")
        except Exception as e:
            print(f"LiveKit AudioStream error: {e}")

    def interrupt_tts_immediately(self):
        """Signal immediate interruption; parent audio worker will handle stopping and clearing."""
        print("🚨 EMERGENCY STOP: Interrupting TTS output NOW!")
        self.interrupt_tts.set()
        # Briefly mute ASR tail and mark AI as not talking
        try:
            self._asr_tail_mute_until = time.monotonic() + 0.5
            self.ai_talking.clear()
        except Exception:
            pass

    def stream_tts_async(self, text, expected_generation=None):
        """
        Stream TTS while signaling AI speaking state and tail mute to prevent echo loops.
        """
        if not text.strip():
            return

        # Mark AI as speaking
        self.ai_talking.set()

        # Delegate to parent implementation (queues or publishes audio)
        super().stream_tts_async(text, expected_generation)

        # Start a short background task to clear ai_talking when playback likely finished
        def _clear_ai_flag_after_playback():
            # For local playback, wait until queue drains or interruption, with a safety timeout
            timeout_s = 8.0
            deadline = time.monotonic() + timeout_s
            try:
                if not (getattr(self, 'livekit_enabled', False) and getattr(self, 'livekit_audio_source', None) is not None):
                    # Wait for audio_queue to drain or interruption
                    while time.monotonic() < deadline:
                        if self.interrupt_tts.is_set():
                            break
                        if self.audio_queue.empty():
                            break
                        time.sleep(0.05)
                else:
                    # In LiveKit publishing path, we cannot observe remote playback.
                    # Use a conservative short delay to keep gating active briefly.
                    time.sleep(0.5)
            except Exception:
                pass
            # Tail mute and clear flag
            self._asr_tail_mute_until = time.monotonic() + 0.25
            self.ai_talking.clear()

        threading.Thread(target=_clear_ai_flag_after_playback, daemon=True).start()

    def write_raw_data(self, audio_data):
        """
        Override to feed far-end reference audio to software AEC before playback/publish.
        """
        try:
            if audio_data and getattr(self, 'enable_aec', False) and getattr(self, '_aec', None) is not None:
                in_rate = getattr(self, 'sample_rate', None) or getattr(self, 'livekit_sample_rate', 48000)
                channels = getattr(self, 'channels', 1)
                self._aec.feed_farend(audio_data, in_rate, channels)
            # Ensure AI talking flag is set while chunks are being output
            if audio_data and not self.ai_talking.is_set():
                self.ai_talking.set()
        except Exception:
            pass
        return super().write_raw_data(audio_data)

    def _summarize_history(self):
        """
        Summarize older history turns into a compact form to shrink prompt.
        
        Takes all but last 4 entries, truncates each line, and creates a summary.
        """
        if len(self.conversation_history) < self.summarize_after_turns:
            return
            
        older = self.conversation_history[:-4]
        recent = self.conversation_history[-4:]
        
        # Compress older entries
        compressed = []
        for line in older:
            if len(line) > 160:
                compressed.append(line[:157] + '...')
            else:
                compressed.append(line)
                
        summary = "Summary: " + " | ".join(compressed)
        if len(summary) > 1000:
            summary = summary[:997] + '...'
            
        self.conversation_history = [summary] + recent
        print("📝 History summarized (length reduced)")

    def process_llm_response_ultra_optimized(self, user_input):
        """
        Build prompt with history and delegate streaming to parent implementation.
        
        Args:
            user_input: User's input text to process
        """
        if not self.llm_available:
            print("❌ LLM server not available!")
            return
        
        # Ensure any ongoing TTS/LLM is stopped and previous audio is cleared
        self.interrupt_tts.set()
        try:
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
        except queue.Empty:
            pass
        
        # Build prompt from recent history
        if self.conversation_history:
            recent_history = self.conversation_history[-8:]
            prompt = "\n".join(recent_history) + f"\nHuman: {user_input}\nAssistant:"
        else:
            prompt = f"Human: {user_input}\nAssistant:"
        
        # Increment generation for this new response so previous streams cancel
        if hasattr(self, 'stream_generation'):
            self.stream_generation += 1
        local_gen = self.stream_generation
        
        # Allow new TTS to play for the upcoming response only if user is not speaking
        if not self.user_speaking.is_set():
            self.interrupt_tts.clear()
        
        # Delegate actual streaming to parent
        response_text = super().stream_llm_response_ultra_optimized(prompt, expected_generation=local_gen)
        
        # Update history and optionally summarize/trim
        if response_text:
            self.conversation_history.append(f"Human: {user_input}")
            self.conversation_history.append(f"Assistant: {response_text}")
            
            if self.enable_history_summarization and len(self.conversation_history) > self.summarize_after_turns:
                self._summarize_history()
                
            if len(self.conversation_history) > self.history_trim_threshold:
                self.conversation_history = self.conversation_history[-self.history_trim_threshold:]

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
        
        # Process through LLM and respond with TTS
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
        
        welcome_msg = ("Hello! I'm ready to chat with you. You can interrupt me "
                      "anytime by speaking. Please speak whenever you're ready.")
        print(f"🤖 {welcome_msg}")
        self.stream_tts_async(welcome_msg)
        
        self.conversation_active = True
        
        try:
            print("\n🎤 Listening... (speak now)")
            while self.conversation_active:
                try:
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
        time.sleep(3)
        
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
        # Stop LiveKit ASR if enabled
        try:
            self.disable_livekit_asr_input()
        except Exception:
            pass
        # ASR-specific cleanup
        self.conversation_active = False
        self.interrupt_tts.set()
        
        # Call parent cleanup for common resources
        super().cleanup()


def main():
    """Main function to run the voice conversation system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Conversation System (ASR-LLM-TTS)")
    parser.add_argument("--llm-url", type=str, default="http://localhost:8080",
                       help="URL of the llama-server (default: http://localhost:8080)")
    parser.add_argument("--tts-model", type=str, default="../tts_models/en_US-hfc_female-medium.onnx",
                       help="Path to Piper TTS model")
    parser.add_argument("--asr-model", type=str, default="tiny",
                       help="ASR model size (tiny, base, small, medium, large) or path to local model file")
    parser.add_argument("--test", action="store_true", help="Test all components")
    parser.add_argument("--test-tts", action="store_true", help="Test TTS only")
    
    args = parser.parse_args()
    
    # Initialize the voice conversation system
    try:
        system = VoiceConversationSystem(
            llm_server_url=args.llm_url,
            tts_model_path=args.tts_model,
            asr_model=args.asr_model
        )
        
        if args.test_tts:
            # Test TTS only
            test_text = ("Hello! This is a test of the voice conversation system. "
                        "The text-to-speech is working correctly. Try speaking to interrupt this message.")
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