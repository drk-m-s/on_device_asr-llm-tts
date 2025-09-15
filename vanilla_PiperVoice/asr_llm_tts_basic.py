#!/usr/bin/env python3
"""
Simplified ASR-LLM-TTS voice conversation script (rewritten)
- No conversation-style options
- No test or test-tts options
- Starts the voice conversation loop directly
"""

import threading
import time
import queue
import argparse

# ASR library
from RealtimeSTT import AudioToTextRecorder

# Base LLM+TTS streamer (we reuse the lower-level building block, not the prior voice script)
from llm_tts import LLMTTSStreamer


class BasicVoiceConversation(LLMTTSStreamer):
    """Minimal voice conversation system built on LLMTTSStreamer.
    Handles ASR input, streams response from LLM via TTS, and supports interruption.
    """

    def __init__(
        self,
        llm_server_url: str | None = None,
        tts_model_path: str | None = None,
        asr_model: str = "tiny",
    ) -> None:
        # Initialize base LLM+TTS
        super().__init__(llm_server_url=llm_server_url, tts_model_path=tts_model_path, conversation_style=None)

        # Conversation state
        self.conversation_history: list[str] = []
        self.conversation_active = False

        # Interruption flags (LLMTTSStreamer already provides these)
        # self.interrupt_tts: threading.Event
        # self.user_speaking: threading.Event

        # Initialize ASR
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
            on_transcription_start=self._on_transcription_start,
        )
        print("ASR model loaded and ready")

    # ===== ASR callbacks =====
    def _on_recording_start(self, *_, **__):
        print("\n🎤 User started speaking...")
        # Any ongoing TTS/LLM should be interrupted immediately
        if hasattr(self, "stream_generation"):
            self.stream_generation += 1
        self.user_speaking.set()
        self.interrupt_tts.set()

    def _on_recording_stop(self, *_, **__):
        print("🎤 User stopped speaking")
        # Allow TTS to resume
        self.user_speaking.clear()
        self.interrupt_tts.clear()

    def _on_transcription_start(self, *_, **__):
        print("📝 Transcription starting...")

    # ===== Core processing =====
    def process_speech_input(self, text: str) -> None:
        """Handle recognized speech from ASR."""
        if not text or not text.strip():
            return

        print(f"\n👤 You said: {text}")

        # Basic commands
        lowered = text.lower().strip()
        if lowered in {"quit", "exit", "stop", "goodbye"}:
            print("👋 Goodbye!")
            self.conversation_active = False
            return
        if "clear" in lowered or "reset" in lowered:
            self.conversation_history.clear()
            print("🗑️ Conversation history cleared.")
            # Give a brief spoken confirmation if not currently speaking
            if not self.user_speaking.is_set():
                self.stream_tts_async("Conversation history cleared.")
            return

        # Filter out very short/noisy inputs
        if len(lowered) < 3:
            return

        # Build prompt and stream reply
        self._process_llm_response(text)

    def _process_llm_response(self, user_input: str) -> None:
        if not self.llm_available:
            print("❌ LLM server not available!")
            return

        # Interrupt any ongoing TTS and clear queued audio
        self.interrupt_tts.set()
        try:
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
        except queue.Empty:
            pass

        # Build prompt from short recent history
        if self.conversation_history:
            recent = self.conversation_history[-8:]
            prompt = "\n".join(recent) + f"\nHuman: {user_input}\nAssistant:"
        else:
            prompt = f"Human: {user_input}\nAssistant:"

        # New generation id so any stale streams are canceled downstream
        if hasattr(self, "stream_generation"):
            self.stream_generation += 1
        expected_gen = self.stream_generation

        # Allow new TTS to play (if user isn't speaking)
        if not self.user_speaking.is_set():
            self.interrupt_tts.clear()

        # Stream LLM response via parent implementation
        response_text = super().stream_llm_response_ultra_optimized(
            prompt, expected_generation=expected_gen
        )

        if response_text:
            self.conversation_history.append(f"Human: {user_input}")
            self.conversation_history.append(f"Assistant: {response_text}")

    # ===== Conversation control =====
    def start(self) -> None:
        if not self.llm_available:
            print("❌ LLM server not available! Please start llama-server on localhost:8080")
            print("Command: llama-server --host 0.0.0.0 --port 8080 --model /path/to/your/model.gguf")
            return

        print("\n🎙️ === Voice Conversation (Basic) ===")
        print("🔊 Speak naturally to converse with the AI")
        print("🎤 You can interrupt the AI anytime by speaking")
        print("📢 Say 'quit', 'exit', or 'stop' to end the conversation")
        print("🗑️ Say 'clear' or 'reset' to clear conversation history")
        print("-" * 50)

        welcome = (
            "Hello! I'm ready to chat with you. You can interrupt me anytime by speaking."
        )
        print(f"🤖 {welcome}")
        self.stream_tts_async(welcome)

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
            print("\n⏹️ Conversation interrupted.")
        finally:
            self.conversation_active = False

    def cleanup(self) -> None:
        # Stop any ongoing activity
        self.conversation_active = False
        self.interrupt_tts.set()
        super().cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic Voice Conversation (ASR-LLM-TTS)")
    parser.add_argument(
        "--llm-url",
        type=str,
        default="http://localhost:8080",
        help="URL of the llama-server (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--tts-model",
        type=str,
        default="../tts_models/en_US-hfc_female-medium.onnx",
        help="Path to Piper TTS model",
    )
    parser.add_argument(
        "--asr-model",
        type=str,
        default="tiny",
        help="ASR model size (tiny, base, small, medium, large) or path to local model file",
    )

    args = parser.parse_args()

    try:
        convo = BasicVoiceConversation(
            llm_server_url=args.llm_url,
            tts_model_path=args.tts_model,
            asr_model=args.asr_model,
        )
        convo.start()
    except KeyboardInterrupt:
        print("\n⏹️ Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if "convo" in locals():
            convo.cleanup()


if __name__ == "__main__":
    main()