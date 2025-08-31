"""Complete ASR-LLM-TTS Voice Conversation System
Listens to user speech, processes it through LLM, and responds with synthesized speech.
Supports interruption: user can speak while AI is talking to interrupt it.
"""

import threading
import time
import queue

# ASR library
from RealtimeSTT import AudioToTextRecorder

# Import parent class
from llm_tts import LLMTTSStreamer

class VoiceConversationSystem(LLMTTSStreamer):
    """Voice conversation system that inherits LLM-TTS functionality."""
    
    def __init__(self, llm_server_url=None, tts_model_path=None,
                 asr_model="tiny",
                 enable_history_summarization=True,
                 summarize_after_turns=10,
                 history_trim_threshold=12,
                 conversation_style=None):
        """
        Initialize the complete voice conversation system.
        
        Args:
            llm_server_url: URL of the LLM server (default: http://localhost:8080)
            tts_model_path: Path to the TTS model file
            asr_model: Fast_Whisper ASR model size ('tiny', 'base', 'small', 'medium', 'large')
            enable_history_summarization: Whether to summarize older turns to shrink prompt
            summarize_after_turns: Number of turns after which to summarize
            history_trim_threshold: Maximum number of history entries to keep
            conversation_style: Optional ConversationStyle for enhanced natural conversation
        """
        # Initialize parent class first
        super().__init__(llm_server_url=llm_server_url, tts_model_path=tts_model_path, conversation_style=conversation_style)
        
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
        
        # Conversation history
        self.conversation_history = []
        

    def _on_recording_start(self, *args, **kwargs):
        """Callback when user starts speaking - interrupt TTS immediately."""
        print("\n🎤 User started speaking...")
        
        # Increment generation so any ongoing LLM/TTS becomes stale
        if hasattr(self, 'stream_generation'):
            self.stream_generation += 1
        
        # Set user speaking flag immediately
        self.user_speaking.set()
        self.ai_should_be_quiet.set()
        
        # Interrupt any ongoing TTS immediately
        print("⏹️ Interrupting AI speech immediately!")
        self.interrupt_tts_immediately()

    def _on_recording_stop(self, *args, **kwargs):
        """Callback when user stops speaking."""
        print("🎤 User stopped speaking")
        
        # Allow TTS to resume now that user stopped speaking
        self.interrupt_tts.clear()
        self.user_speaking.clear()
        
        # Keep AI quiet for a short moment to ensure clean transition
        threading.Timer(0.5, self.ai_should_be_quiet.clear).start()

    def _on_transcription_start(self, *args, **kwargs):
        """Callback when transcription starts."""
        print("📝 Transcription starting...")
        
        # Briefly prevent new speech to avoid overlap, then allow
        self.ai_should_be_quiet.set()
        threading.Timer(0.3, self.ai_should_be_quiet.clear).start()

    def interrupt_tts_immediately(self):
        """Signal immediate interruption; parent audio worker will handle stopping and clearing."""
        print("🚨 EMERGENCY STOP: Interrupting TTS output NOW!")
        self.interrupt_tts.set()

    def stream_tts_async(self, text, expected_generation=None):
        """
        Delegate to parent async TTS for playback.
        
        Args:
            text: Text to synthesize
            expected_generation: Generation number for cancellation
        """
        if not text.strip():
            return
        
        # Delegate to parent implementation
        super().stream_tts_async(text, expected_generation)

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
    
    # Initialize the voice conversation system
    try:
        system = VoiceConversationSystem(
            llm_server_url=args.llm_url,
            tts_model_path=args.tts_model,
            asr_model=args.asr_model,
            conversation_style=conversation_style
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