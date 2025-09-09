"""Complete ASR-LLM-TTS Voice Conversation System with AEC
Listens to user speech, processes it through LLM, and responds with synthesized speech.
Includes Acoustic Echo Cancellation (AEC) to prevent AI speech from interfering with ASR.
Supports interruption: user can speak while AI is talking to interrupt it.
"""

# Suppress warnings from dependencies
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="webrtcvad")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*compute type inferred.*", module="ctranslate2")

import threading
import time
import queue
import numpy as np
import platform

# Audio libraries
import sounddevice as sd
from scipy import signal
from collections import deque
import math

# ASR library
from RealtimeSTT import AudioToTextRecorder

# Import parent class
from llm_tts import LLMTTSStreamer

class AECProcessor:
    """Acoustic Echo Cancellation processor using adaptive filtering."""
    
    def __init__(self, sample_rate=16000, frame_size=512, filter_length=1024):
        """
        Initialize AEC processor.
        
        Args:
            sample_rate: Audio sample rate
            frame_size: Size of audio frames to process
            filter_length: Length of adaptive filter for echo cancellation
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.filter_length = filter_length
        
        # Adaptive filter coefficients (NLMS algorithm)
        self.filter_coeffs = np.zeros(filter_length, dtype=np.float32)
        self.reference_buffer = deque(maxlen=filter_length)
        
        # NLMS parameters
        self.mu = 0.1  # Step size (learning rate)
        self.epsilon = 1e-8  # Small constant to prevent division by zero
        
        # Echo path estimation
        self.reference_history = np.zeros(filter_length, dtype=np.float32)
        
        # Voice activity detection parameters
        self.noise_floor = 0.001
        self.vad_threshold = 3.0  # Threshold above noise floor
        self.speech_prob_threshold = 0.7
        
        # Smoothing for VAD
        self.energy_smooth = 0.0
        self.alpha_energy = 0.1
        
        print(f"AEC initialized: {sample_rate}Hz, {frame_size} frame, {filter_length} filter")

    def update_reference(self, reference_audio):
        """Update reference signal buffer with AI speech output."""
        if reference_audio is not None and len(reference_audio) > 0:
            # Convert to float32 and normalize
            if reference_audio.dtype != np.float32:
                if reference_audio.dtype == np.int16:
                    reference_audio = reference_audio.astype(np.float32) / 32768.0
                elif reference_audio.dtype == np.int32:
                    reference_audio = reference_audio.astype(np.float32) / 2147483648.0
                else:
                    reference_audio = reference_audio.astype(np.float32)
            
            # Add to circular buffer
            for sample in reference_audio:
                self.reference_buffer.append(sample)

    def process_microphone_audio(self, mic_audio):
        """
        Process microphone audio to remove echo.
        
        Args:
            mic_audio: Microphone input audio (numpy array)
            
        Returns:
            Processed audio with echo removed
        """
        if len(mic_audio) == 0:
            return mic_audio
        
        # Convert to float32 if needed
        if mic_audio.dtype != np.float32:
            if mic_audio.dtype == np.int16:
                mic_audio = mic_audio.astype(np.float32) / 32768.0
            elif mic_audio.dtype == np.int32:
                mic_audio = mic_audio.astype(np.float32) / 2147483648.0
            else:
                mic_audio = mic_audio.astype(np.float32)
        
        # If we don't have enough reference data, return original
        if len(self.reference_buffer) < self.filter_length:
            return mic_audio
        
        # Convert reference buffer to array
        reference_array = np.array(list(self.reference_buffer), dtype=np.float32)
        
        # Process each frame
        processed_audio = np.zeros_like(mic_audio)
        
        for i in range(0, len(mic_audio), self.frame_size):
            end_idx = min(i + self.frame_size, len(mic_audio))
            frame = mic_audio[i:end_idx]
            
            # Pad frame if necessary
            if len(frame) < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - len(frame)))
            
            # Apply NLMS adaptive filtering
            processed_frame = self._nlms_filter(frame, reference_array[-self.filter_length:])
            processed_audio[i:end_idx] = processed_frame[:end_idx-i]
        
        return processed_audio

    def _nlms_filter(self, input_frame, reference_signal):
        """
        Apply Normalized Least Mean Squares adaptive filter.
        
        Args:
            input_frame: Current microphone input frame
            reference_signal: Reference signal (AI output)
            
        Returns:
            Filtered audio frame
        """
        output_frame = np.zeros_like(input_frame)
        
        for n, x_n in enumerate(input_frame):
            # Ensure we have enough reference history
            if len(reference_signal) < self.filter_length:
                output_frame[n] = x_n
                continue
            
            # Get reference vector (reversed for convolution)
            x_vec = reference_signal[-self.filter_length:][::-1]
            
            # Predict echo
            echo_estimate = np.dot(self.filter_coeffs, x_vec)
            
            # Calculate error (desired - estimated)
            error = x_n - echo_estimate
            output_frame[n] = error
            
            # Update filter coefficients using NLMS
            norm_factor = np.dot(x_vec, x_vec) + self.epsilon
            self.filter_coeffs += (self.mu * error / norm_factor) * x_vec
            
            # Prevent filter instability
            self.filter_coeffs = np.clip(self.filter_coeffs, -1.0, 1.0)
        
        return output_frame

    def is_speech_present(self, audio):
        """
        Simple voice activity detection to help AEC performance.
        
        Args:
            audio: Audio signal to analyze
            
        Returns:
            Boolean indicating if speech is likely present
        """
        if len(audio) == 0:
            return False
        
        # Calculate energy
        energy = np.mean(audio**2)
        
        # Smooth energy estimate
        self.energy_smooth = (self.alpha_energy * energy + 
                             (1 - self.alpha_energy) * self.energy_smooth)
        
        # Adaptive noise floor estimation
        if energy < self.energy_smooth:
            self.noise_floor = 0.9 * self.noise_floor + 0.1 * energy
        
        # Speech detection
        speech_threshold = self.noise_floor * self.vad_threshold
        return energy > speech_threshold


class VoiceConversationSystemWithAEC(LLMTTSStreamer):
    """Voice conversation system with Acoustic Echo Cancellation."""
    
    def __init__(self, llm_server_url=None, tts_model_path=None,
                 asr_model="tiny",
                 enable_history_summarization=True,
                 summarize_after_turns=10,
                 history_trim_threshold=12,
                 conversation_style=None,
                 enable_aec=True,
                 aec_filter_length=1024):
        """
        Initialize the voice conversation system with AEC.
        
        Args:
            llm_server_url: URL of the LLM server
            tts_model_path: Path to the TTS model file
            asr_model: ASR model size or path
            enable_history_summarization: Whether to summarize older turns
            summarize_after_turns: Number of turns after which to summarize
            history_trim_threshold: Maximum number of history entries
            conversation_style: Optional ConversationStyle for natural conversation
            enable_aec: Whether to enable Acoustic Echo Cancellation
            aec_filter_length: Length of AEC adaptive filter
        """
        # Initialize parent class
        super().__init__(llm_server_url=llm_server_url, 
                        tts_model_path=tts_model_path, 
                        conversation_style=conversation_style)
        
        # AEC configuration
        self.enable_aec = enable_aec
        self.aec_filter_length = aec_filter_length
        
        # ASR configuration
        self.enable_history_summarization = enable_history_summarization
        self.summarize_after_turns = summarize_after_turns
        self.history_trim_threshold = history_trim_threshold
        
        # Audio parameters for AEC
        self.asr_sample_rate = 16000  # RealtimeSTT typically uses 16kHz
        self.asr_frame_size = 512
        
        # Initialize AEC processor if enabled
        if self.enable_aec:
            self.aec_processor = AECProcessor(
                sample_rate=self.asr_sample_rate,
                frame_size=self.asr_frame_size,
                filter_length=self.aec_filter_length
            )
            print("AEC enabled for echo cancellation")
        else:
            self.aec_processor = None
            print("AEC disabled")
        
        # TTS audio monitoring for AEC reference
        self.tts_audio_buffer = deque(maxlen=self.asr_sample_rate * 2)  # 2 seconds buffer
        self.tts_monitoring_lock = threading.Lock()
        
        # Initialize ASR with AEC-aware callback
        print(f"Loading ASR model: {asr_model}")
        
        # Custom audio preprocessing for AEC
        def aec_audio_processor(audio_data):
            """Preprocess audio through AEC before ASR."""
            if self.enable_aec and self.aec_processor and len(audio_data) > 0:
                # Apply AEC to remove echo
                processed_audio = self.aec_processor.process_microphone_audio(audio_data)
                return processed_audio
            return audio_data
        
        self.asr_recorder = AudioToTextRecorder(
            model=asr_model,
            enable_realtime_transcription=True,
            silero_sensitivity=0.3,
            # - 0.0 : Least sensitive (requires very clear, loud speech to trigger detection)
            # - 1.0 : Most sensitive (may trigger on background noise or very quiet sounds)
            # - Default : 0.6
            silero_use_onnx=True,
            post_speech_silence_duration=0.5,
            language="en",
            on_vad_start=self._on_recording_start,
            on_vad_stop=self._on_recording_stop,
            on_transcription_start=self._on_transcription_start
        )
        
        print("ASR model loaded with AEC integration")
        
        # Conversation control
        self.conversation_active = False
        self.interrupt_tts = threading.Event()
        self.user_speaking = threading.Event()
        self.ai_should_be_quiet = threading.Event()
        self.conversation_history = []

    def write_raw_data(self, audio_data):
        """Override parent method to capture TTS audio for AEC reference."""
        # Call parent method for normal audio playback
        super().write_raw_data(audio_data)
        
        # Capture audio for AEC reference if enabled
        if self.enable_aec and self.aec_processor and audio_data:
            try:
                # Convert audio data to numpy array for AEC
                if isinstance(audio_data, bytes):
                    # Assume int16 format from TTS
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    # Resample if needed for AEC (TTS might be different sample rate)
                    if hasattr(self, 'sample_rate') and self.sample_rate != self.asr_sample_rate:
                        # Simple resampling - in production, use proper resampling
                        audio_array = signal.resample(
                            audio_array, 
                            int(len(audio_array) * self.asr_sample_rate / self.sample_rate)
                        ).astype(np.int16)
                    
                    with self.tts_monitoring_lock:
                        self.aec_processor.update_reference(audio_array)
                        
            except Exception as e:
                print(f"Warning: AEC reference update failed: {e}")

    def _on_recording_start(self, *args, **kwargs):
        """Callback when user starts speaking."""
        print("\n🎤 User started speaking...")
        
        # Increment generation for cancellation
        if hasattr(self, 'stream_generation'):
            self.stream_generation += 1
        
        # Set flags for interruption
        self.user_speaking.set()
        self.ai_should_be_quiet.set()
        
        # Interrupt TTS immediately
        print("⏹️ Interrupting AI speech for user input")
        self.interrupt_tts_immediately()

    def _on_recording_stop(self, *args, **kwargs):
        """Callback when user stops speaking."""
        print("🎤 User stopped speaking")
        self.interrupt_tts.clear()
        self.user_speaking.clear()
        threading.Timer(0.5, self.ai_should_be_quiet.clear).start()

    def _on_transcription_start(self, *args, **kwargs):
        """Callback when transcription starts."""
        print("📝 Transcription starting...")
        self.ai_should_be_quiet.set()
        threading.Timer(0.3, self.ai_should_be_quiet.clear).start()

    def interrupt_tts_immediately(self):
        """Signal immediate TTS interruption."""
        print("🚨 EMERGENCY STOP: Interrupting TTS output NOW!")
        self.interrupt_tts.set()

    def _summarize_history(self):
        """Summarize conversation history to manage prompt length."""
        if len(self.conversation_history) < self.summarize_after_turns:
            return
            
        older = self.conversation_history[:-4]
        recent = self.conversation_history[-4:]
        
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
        print("📋 History summarized (length reduced)")

    def process_llm_response_ultra_optimized(self, user_input):
        """Process user input through LLM with conversation history."""
        if not self.llm_available:
            print("❌ LLM server not available!")
            return
        
        # Clear previous audio and interrupt ongoing generation
        self.interrupt_tts.set()
        try:
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
        except queue.Empty:
            pass
        
        # Build prompt with history
        if self.conversation_history:
            recent_history = self.conversation_history[-8:]
            prompt = "\n".join(recent_history) + f"\nHuman: {user_input}\nAssistant:"
        else:
            prompt = f"Human: {user_input}\nAssistant:"
        
        # Increment generation counter
        if hasattr(self, 'stream_generation'):
            self.stream_generation += 1
        local_gen = self.stream_generation
        
        # Allow new TTS if user isn't speaking
        if not self.user_speaking.is_set():
            self.interrupt_tts.clear()
        
        # Process through parent LLM streaming
        response_text = super().stream_llm_response_ultra_optimized(prompt, expected_generation=local_gen)
        
        # Update conversation history
        if response_text:
            self.conversation_history.append(f"Human: {user_input}")
            self.conversation_history.append(f"Assistant: {response_text}")
            
            if self.enable_history_summarization and len(self.conversation_history) > self.summarize_after_turns:
                self._summarize_history()
                
            if len(self.conversation_history) > self.history_trim_threshold:
                self.conversation_history = self.conversation_history[-self.history_trim_threshold:]

    def process_speech_input(self, text):
        """Process recognized speech input."""
        if not text.strip() or len(text.strip()) < 3:
            return
            
        print(f"\n👤 You said: {text}")
        
        # Handle commands
        if text.lower().strip() in ['quit', 'exit', 'stop', 'goodbye']:
            print("👋 Goodbye!")
            self.conversation_active = False
            return
        
        if 'clear' in text.lower() or 'reset' in text.lower():
            self.conversation_history = []
            print("🗑️ Conversation history cleared.")
            if not self.user_speaking.is_set():
                self.stream_tts_async("Conversation history cleared.")
            return
        
        # Process through LLM
        self.process_llm_response_ultra_optimized(text)

    def start_voice_conversation(self):
        """Start the voice conversation loop."""
        if not self.llm_available:
            print("❌ LLM server not available! Please start llama-server")
            return
        
        print("\n🎙️ === Voice Conversation System with AEC ===")
        print("🔊 Speak naturally to chat with the AI")
        print("🎤 You can interrupt the AI anytime by speaking")
        print("🔇 Echo cancellation is active to prevent feedback")
        print("📢 Say 'quit', 'exit', or 'stop' to end")
        print("🗑️ Say 'clear' or 'reset' to clear history")
        print("-" * 50)
        
        welcome_msg = ("Hello! I'm ready to chat with you. I have echo cancellation "
                      "enabled, so you can interrupt me anytime by speaking. What would you like to talk about?")
        print(f"🤖 {welcome_msg}")
        self.stream_tts_async(welcome_msg)
        
        self.conversation_active = True
        
        try:
            print("\n🎤 Listening with AEC... (speak now)")
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

    def test_aec_performance(self):
        """Test AEC performance by playing audio and monitoring microphone."""
        if not self.enable_aec:
            print("❌ AEC is disabled. Enable AEC to run this test.")
            return
        
        print("\n🧪 === AEC Performance Test ===")
        print("This test will play audio while monitoring the microphone")
        print("to demonstrate echo cancellation.")
        
        # Play test audio
        test_audio = "This is a test of acoustic echo cancellation. The system should prevent this speech from being picked up by the microphone and processed again."
        print("🔊 Playing test audio...")
        self.stream_tts_async(test_audio)
        
        # Monitor microphone for a few seconds
        print("🎤 Monitoring microphone for echo (5 seconds)...")
        
        def echo_test_callback(text):
            if text.strip():
                print(f"⚠️  Echo detected: '{text}' (This should be minimal with AEC)")
            else:
                print("✅ No significant echo detected")
        
        start_time = time.time()
        while time.time() - start_time < 5:
            try:
                self.asr_recorder.text(echo_test_callback)
                time.sleep(0.1)
            except:
                break
        
        print("✅ AEC test complete!")

    def cleanup(self):
        """Clean up all resources."""
        self.conversation_active = False
        self.interrupt_tts.set()
        
        # AEC cleanup
        if hasattr(self, 'aec_processor'):
            self.aec_processor = None
        
        # Parent cleanup
        super().cleanup()


def main():
    """Main function for the AEC-enabled voice conversation system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Conversation System with AEC")
    parser.add_argument("--llm-url", type=str, default="http://localhost:8080",
                       help="URL of the llama-server")
    parser.add_argument("--tts-model", type=str, default="../tts_models/en_US-hfc_female-medium.onnx",
                       help="Path to Piper TTS model")
    parser.add_argument("--asr-model", type=str, default="tiny",
                       help="ASR model size")
    parser.add_argument("--disable-aec", action="store_true",
                       help="Disable Acoustic Echo Cancellation")
    parser.add_argument("--aec-filter-length", type=int, default=1024,
                       help="AEC filter length (default: 1024)")
    parser.add_argument("--test-aec", action="store_true",
                       help="Test AEC performance")
    parser.add_argument("--conversation-style", type=str, 
                       choices=["casual", "professional", "enthusiastic", "thoughtful", "concise"],
                       default="casual", help="Conversation style")
    
    args = parser.parse_args()
    
    # Setup conversation style
    conversation_style = None
    if not hasattr(args, 'disable_enhancements') or not args.disable_enhancements:
        from llm_tts import ConversationStyle, VocabularyStyle
        style_map = {
            "casual": VocabularyStyle.CASUAL,
            "professional": VocabularyStyle.PROFESSIONAL,
            "friendly": VocabularyStyle.ENTHUSIASTIC,
            "technical": VocabularyStyle.THOUGHTFUL
        }
        vocab_style = style_map.get(args.conversation_style.lower(), VocabularyStyle.CASUAL)
        conversation_style = ConversationStyle(vocabulary=vocab_style)
    
    try:
        # Initialize system
        system = VoiceConversationSystemWithAEC(
            llm_server_url=args.llm_url,
            tts_model_path=args.tts_model,
            asr_model=args.asr_model,
            enable_aec=not args.disable_aec,
            aec_filter_length=args.aec_filter_length,
            conversation_style=conversation_style
        )
        
        if args.test_aec:
            # Test AEC performance
            system.test_aec_performance()
        else:
            # Start voice conversation
            system.start_voice_conversation()
            
    except KeyboardInterrupt:
        print("\n⏹️ Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'system' in locals():
            system.cleanup()


if __name__ == "__main__":
    main()