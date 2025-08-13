# Real-Time Voice Conversation System (ASR + LLM + TTS) : 500ms time-to-first-token

This repository contains a real‑time demo of ASR + LLM + TTS built on top of `llama.cpp`:

1. Voice Conversation Pipeline (ASR → LLM → TTS) with interruption & latency instrumentation
2. Original SmolVLM real-time camera demo (vision + instruction following)

---
## 1. Voice Conversation System
Low-latency, interruption-aware voice assistant pipeline:

Workflow:
```
Mic → RealtimeSTT → Prompt Build → llama.cpp /completion (SSE) → Token Buffer → Piper TTS → Speaker
          ^                                                           |
          | (pause during prefill)                                    v
    User speech interrupts  <----  Immediate TTS stop & queue flush (barge-in)
```

### Features
- Sub-second target first-token latency with detailed timing breakdown
- Raw byte SSE parsing (optional) for earliest token flush vs line iterator
- HTTP/2 toggle (disable if adding head-of-line blocking or delay)
- Optional ASR pause during LLM prefill to reduce CPU contention
- Sentence-chunk incremental TTS playback (speaks while still generating)
- Instant barge-in (user speech interrupts and flushes audio queue)
- Conversation history summarization & trimming (keeps context small)
- Low latency mode (removes mirostat / typical_p; simpler sampling)
- Tokens/sec generation metrics

### Key File
`asr_llm_tts.py` – class `VoiceConversationSystem`

### Installation
```
pip install -r requirements.txt
```
Ensure you have:
- `llama-server` (from llama.cpp) running locally
- Piper ONNX voice model (e.g. `en_US-hfc_female-medium.onnx`) placed in repo root (gitignored)

### Run
```
python asr_llm_tts.py --llm-url http://localhost:8080 --tts-model en_US-hfc_female-medium.onnx
```
Test components:
```
python asr_llm_tts.py --test
```

### llama.cpp Server Example
```
./llama-server --model ./models/your-model.Q4_K_M.gguf --host 0.0.0.0 --port 8080 \
  --ctx-size 4096 --parallel 2 --no-mmap
```
Tune args for your hardware (quant, threads, ctx-size). For fastest first token, prefer smaller / quantized model.

### Latency Metrics Printed
```
build_prompt
pre_request_gap
network+server_first_byte
first_sse_line_after_send
first_token_latency_total
generation_phase (tokens + tps)
end_to_end
```
Interpretation:
- Large `network+server_first_byte`: server busy / model prefill
- Gap between first_byte and first_sse_line: server buffering before flush
- High pre_request_gap: Python scheduling contention (reduce threads/ASR)

### Optimization Toggles (constructor params)
- `use_http2` (default True)
- `use_raw_stream` (default True) – raw byte parser for minimum buffering
- `pause_asr_during_prefill` (default True)
- `low_latency_mode` (default True) – strips advanced sampling
- `enable_history_summarization` (default True)
- `summarize_after_turns` (default 10)
- `history_trim_threshold` (default 12)

### History Summarization
After threshold, older turns are compressed into a single `Summary:` line (simple truncation). Replace with a smarter summarizer if desired.

### Barge-In
ASR callbacks (`on_recording_start/stop/transcription_start`) immediately:
- Set interruption flags
- Stop / flush TTS queue
- Prevent new TTS until user finishes

### Roadmap Ideas
- CLI flags to expose toggles (currently hardcoded defaults)
- Multiprocessing for TTS to avoid GIL influence on first token
- Smarter semantic summarization using a tiny local model
- WebSocket / browser microphone client integration
- Automated model download / verification script
