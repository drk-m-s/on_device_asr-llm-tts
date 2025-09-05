"""ASR-LLM-TTS Voice Conversation System with Echo Suppression (AEC-like)

Goal:
- Preserve the original vanilla pipeline behavior and latency optimizations
- Prevent the AI's own speaker output from triggering the mic's VAD or becoming a query
- Still allow barge-in: if the user speaks while AI is talking, and the words differ from the AI's speech, interrupt TTS and process the user

Notes:
- This implements a pragmatic echo suppression layer without relying on native WebRTC AEC bindings
- It gates ASR events during TTS and filters transcripts that match the TTS text being spoken
- If a transcript differs sufficiently from the current TTS utterance(s), we treat it as a real user interruption
"""

import threading
import time
import queue
from collections import deque
from difflib import SequenceMatcher

# ASR library
from RealtimeSTT import AudioToTextRecorder

# Import parent class
from llm_tts import LLMTTSStreamer


class VoiceConversationSystemAEC(LLMTTSStreamer):
    """Voice conversation system with echo suppression on top of LLM-TTS."""

    def __init__(self, llm_server_url=None, tts_model_path=None,
                 asr_model="tiny",
                 enable_history_summarization=True,
                 summarize_after_turns=10,
                 history_trim_threshold=12,
                #  aec_similarity_threshold=0.6,
                 aec_similarity_threshold=0.5,
                 aec_tail_seconds=0.6,
                 conversation_style=None):
        """
        Initialize the complete voice conversation system with echo suppression.

        Args:
            llm_server_url: URL of the LLM server (default: http://localhost:8080)
            tts_model_path: Path to the TTS model file
            asr_model: Fast_Whisper ASR model size ('tiny', 'base', 'small', 'medium', 'large')
            enable_history_summarization: Whether to summarize older turns to shrink prompt
            summarize_after_turns: Number of turns after which to summarize
            history_trim_threshold: Maximum number of history entries to keep
            aec_similarity_threshold: If ASR text is this similar to current TTS text, drop it as echo
            aec_tail_seconds: Keep echo-guard active for this many seconds after last TTS chunk enqueue
            conversation_style: Optional ConversationStyle for enhanced natural conversation
        """
        # Initialize parent class first
        super().__init__(llm_server_url=llm_server_url, tts_model_path=tts_model_path, conversation_style=conversation_style)

        # ASR-specific configuration
        self.enable_history_summarization = enable_history_summarization
        self.summarize_after_turns = summarize_after_turns
        self.history_trim_threshold = history_trim_threshold

        # Echo suppression configuration
        self.aec_similarity_threshold = aec_similarity_threshold
        self.aec_tail_seconds = aec_tail_seconds

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

        # Conversation control
        self.conversation_active = False

        # Interruption control
        self.interrupt_tts = threading.Event()
        self.user_speaking = threading.Event()
        self.ai_should_be_quiet = threading.Event()

        # Echo suppression state
        self.aec_guard_active = threading.Event()  # True while TTS is enqueuing/playing
        self.potential_user_speaking = threading.Event()  # Vad fired during TTS; verify with transcript first
        self.last_tts_activity_time = 0.0
        self.recent_tts_chunks = deque(maxlen=5)  # Keep last few chunks sent to TTS

        # Conversation history
        self.conversation_history = []

    # -------------------- Echo-aware callbacks --------------------
    def _on_recording_start(self, *args, **kwargs):
        """Callback when VAD detects speech.
        If TTS might be causing it, don't immediately interrupt; wait for transcript to verify.
        """
        print("\n🎤 VAD detected speech...")

        if self._is_aec_guard_on():
            # Likely echo from our own speaker; mark potential and wait for transcript to decide
            print("🛡️ AEC guard active: deferring interruption until transcript verification")
            self.potential_user_speaking.set()
            # Do NOT set user_speaking, do NOT interrupt yet
            return

        # Otherwise, real user start: interrupt TTS immediately
        if hasattr(self, 'stream_generation'):
            self.stream_generation += 1
        self.user_speaking.set()
        self.ai_should_be_quiet.set()
        print("⏹️ Interrupting AI speech immediately!")
        self.interrupt_tts_immediately()

    def _on_recording_stop(self, *args, **kwargs):
        """Callback when user stops speaking."""
        print("🎤 Speech ended")
        self.interrupt_tts.clear()
        self.user_speaking.clear()
        self.potential_user_speaking.clear()
        threading.Timer(0.5, self.ai_should_be_quiet.clear).start()

    def _on_transcription_start(self, *args, **kwargs):
        """Callback when transcription starts."""
        print("📝 Transcription starting...")
        self.ai_should_be_quiet.set()
        threading.Timer(0.3, self.ai_should_be_quiet.clear).start()

    def interrupt_tts_immediately(self):
        print("🚨 EMERGENCY STOP: Interrupting TTS output NOW!")
        self.interrupt_tts.set()

    # -------------------- TTS with echo guard --------------------
    def _enable_aec_guard(self):
        self.aec_guard_active.set()
        self.last_tts_activity_time = time.time()

    def _disable_aec_guard_if_idle(self):
        # Disable only if no recent TTS activity
        if time.time() - self.last_tts_activity_time >= self.aec_tail_seconds:
            self.aec_guard_active.clear()

    def _is_aec_guard_on(self):
        # Also consider a tail window after last chunk to ignore trailing echo
        if self.aec_guard_active.is_set():
            return True
        return (time.time() - self.last_tts_activity_time) < self.aec_tail_seconds

    def stream_tts_async(self, text, expected_generation=None):
        """Wrap parent TTS with echo guard and remember spoken text for transcript filtering."""
        if not text or not text.strip():
            return

        # Record the current chunk for echo matching
        self.recent_tts_chunks.append(text.strip())

        # Turn on echo guard and keep it on for a short tail after last chunk
        self._enable_aec_guard()
        try:
            super().stream_tts_async(text, expected_generation)
        finally:
            # Update activity time; schedule a delayed guard-off check
            self.last_tts_activity_time = time.time()
            threading.Timer(self.aec_tail_seconds, self._disable_aec_guard_if_idle).start()

    # -------------------- LLM Response --------------------
    def _summarize_history(self):
        if len(self.conversation_history) < self.summarize_after_turns:
            return
        older = self.conversation_history[:-4]
        recent = self.conversation_history[-4:]
        compressed = []
        for line in older:
            compressed.append(line[:157] + '...' if len(line) > 160 else line)
        summary = "Summary: " + " | ".join(compressed)
        if len(summary) > 1000:
            summary = summary[:997] + '...'
        self.conversation_history = [summary] + recent
        print("📝 History summarized (length reduced)")

    def process_llm_response_ultra_optimized(self, user_input):
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

    # -------------------- ASR handling with echo filter --------------------
    def _looks_like_echo(self, text: str) -> bool:
        """Return True if text is similar to any recent TTS chunk (likely echo)."""
        if not text:
            return False
        t = text.strip().lower()
        if not t:
            return False
        # Exact/substring quick checks
        for chunk in self.recent_tts_chunks:
            c = chunk.lower()
            if not c:
                continue
            if t in c or c in t:
                return True
        # Similarity ratio checks
        for chunk in self.recent_tts_chunks:
            c = chunk.strip()
            if not c:
                continue
            ratio = SequenceMatcher(None, t, c.lower()).ratio()
            if ratio >= self.aec_similarity_threshold:
                return True
        return False

    def process_speech_input(self, text):
        """Callback function for ASR - processes recognized speech with echo filtering."""
        if not text or not text.strip():
            return
        if len(text.strip()) < 3:
            return

        # If we're in echo-guard, drop transcripts that look like our own TTS
        if self._is_aec_guard_on():
            if self._looks_like_echo(text):
                print(f"🔇 Dropped echo-like transcript: {text}")
                return
            else:
                # Not echo => this is a real user interruption while AI is speaking
                print("🛑 Dissimilar to TTS: treating as real user interruption")
                if hasattr(self, 'stream_generation'):
                    self.stream_generation += 1
                self.user_speaking.set()
                self.ai_should_be_quiet.set()
                self.interrupt_tts_immediately()

        print(f"\n👤 You said: {text}")

        # Check for exit commands
        if text.lower().strip() in ['quit', 'exit', 'stop', 'goodbye']:
            print("👋 Goodbye!")
            self.conversation_active = False
            return

        # Clear command
        if 'clear' in text.lower() or 'reset' in text.lower():
            self.conversation_history = []
            print("🗑️ Conversation history cleared.")
            if not self.user_speaking.is_set():
                self.stream_tts_async("Conversation history cleared.")
            return

        # Process through LLM and respond with TTS
        self.process_llm_response_ultra_optimized(text)

    # -------------------- Run / Test / Cleanup --------------------
    def start_voice_conversation(self):
        if not self.llm_available:
            print("❌ LLM server not available! Please start llama-server on localhost:8080")
            print("Command: llama-server --host 0.0.0.0 --port 8080 --model /path/to/your/model.gguf")
            return

        print("\n🎙️ === Voice Conversation System (AEC) ===")
        print("🔊 Speak naturally to have a conversation with the AI")
        print("🛡️ AI's own speech won't trigger ASR or be treated as your query")
        print("🗣️ You can still interrupt by speaking something different while AI is talking")
        print("📢 Say 'quit', 'exit', or 'stop' to end the conversation")
        print("🗑️ Say 'clear' or 'reset' to clear conversation history")
        print("-" * 50)

        welcome_msg = ("Hello! I'm ready to chat with you. While I'm speaking, "
                       "my own audio won't trigger the microphone. If you want to interrupt, "
                       "just start speaking and I'll stop.")
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
        print("\n🧪 === Testing Components (AEC) ===")
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
        self.conversation_active = False
        self.interrupt_tts.set()
        super().cleanup()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Voice Conversation System with Echo Suppression (ASR-LLM-TTS)")
    parser.add_argument("--llm-url", type=str, default="http://localhost:8080",
                        help="URL of the llama-server (default: http://localhost:8080)")
    parser.add_argument("--tts-model", type=str, default="../tts_models/en_US-hfc_female-medium.onnx",
                        help="Path to Piper TTS model")
    parser.add_argument("--asr-model", type=str, default="tiny",
                        help="ASR model size (tiny, base, small, medium, large) or path to local model file")
    parser.add_argument("--conversation-style", type=str, choices=["casual", "professional", "friendly", "technical"],
                        default="friendly", help="Conversation style for enhanced natural speech (default: friendly)")
    parser.add_argument("--disable-enhancements", action="store_true",
                        help="Disable conversation enhancements (use original behavior)")
    parser.add_argument("--test", action="store_true", help="Test all components")
    parser.add_argument("--test-tts", action="store_true", help="Test TTS only")
    args = parser.parse_args()

    # Determine conversation style
    conversation_style = None
    if not args.disable_enhancements:
        from llm_tts import ConversationStyle, VocabularyStyle
        style_map = {
            "casual": VocabularyStyle.CASUAL,
            "professional": VocabularyStyle.PROFESSIONAL,
            "friendly": VocabularyStyle.FRIENDLY,
            "technical": VocabularyStyle.TECHNICAL
        }
        conversation_style = ConversationStyle(
            vocabulary_style=style_map[args.conversation_style],
            personality_traits=["helpful", "engaging"]
        )
        print(f"🎭 Enhanced conversation mode: {args.conversation_style}")
    else:
        print("🔧 Using original conversation behavior")

    try:
        system = VoiceConversationSystemAEC(
            llm_server_url=args.llm_url,
            tts_model_path=args.tts_model,
            asr_model=args.asr_model,
            conversation_style=conversation_style
        )
        if args.test_tts:
            test_text = ("Hello! This is a test of the voice conversation system with echo suppression. "
                         "Try speaking to interrupt this message. If you repeat me, I will ignore it.")
            system.stream_tts_async(test_text)
            time.sleep(3)
        elif args.test:
            system.test_components()
        else:
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