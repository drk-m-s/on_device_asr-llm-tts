# Contributing

Thanks for your interest! This project currently houses two demos (voice assistant + camera). Feel free to open issues / PRs for:
- Latency improvements
- Cross‑platform audio input/output
- Better history summarization
- Web UI integration (WebRTC, websocket streaming)
- Documentation fixes

## Development Setup
1. Create virtual environment (Python 3.12+)
2. `pip install -r requirements.txt`
3. Acquire models (llama.cpp GGUF + Piper ONNX) and place in repo root or a `models/` folder you reference.
4. Run `python asr_llm_tts.py --test` to verify components.

## Coding Guidelines
- Keep first-token latency instrumentation intact for performance PRs.
- Avoid introducing heavy dependencies without justification.
- Wrap experimental features behind a constructor flag.
- Prefer small, focused PRs.

## Adding Models
Do not commit large model binaries. Reference download commands or scripts.

## Issues
When reporting latency issues include:
```
Hardware (CPU/GPU/RAM)
Model & quantization
Timing breakdown sample
Toggle settings (raw_stream, http2, low_latency_mode, etc.)
```

## License
Ensure any contribution is compatible with the repository license (see LICENSE).
