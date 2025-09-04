# Real-Time Voice Conversation System (ASR + LLM + TTS) 

Voice Conversation Pipeline (ASR → LLM → TTS) 

[ ] ASR：interruption， latency, and echo cancellation
in vanilla_PiperVoice, modify asr_llm_tts.py STEP by STEP:
- can it quasi-immediately detect if the audio is from the same voice of voiceprint/tts_ref_0.wav?
- if so, can it be quickly done?
- if so, can it quit recordding immediately if it knows it actually from the sound of itself?



[ ] LLM: a fine-tuned model with customized vocabulary style
- [x] a substitute gguf
- [x] a safetensors to be distilled into gguf.(https://chatgpt.com/share/68b3c412-52b4-800e-8b44-267495c715df)
- [ ] finetune one (for Mandarin)
    - [ ] collect one set of dataset
    - [ ] use one plain llm
    - [ ] get rid of the nsfw impact
    - [ ] train/finetune
    - [ ] export to gguf form.
[ ] TTS: voice customization 
  - [ ] use openvoice to copy the assigned voice color to a dataset.
  - [ ] use PiperVocie to train one. A gpu-intensive task.
[ ] when heard a voice and starts recording, recognise the source and judge if it is its own. If so, quit the recording.
  - [ ] require fast voice print generation as well as fast comparison
  - [ ] extend the recorded wav test to ongoing  test




temp/ is another attempt.
https://chatgpt.com/share/68b7e7a6-6cb0-800e-9af5-3402967aecb2
where 
go build -v -o livekit-server ./cmd/livekit-server 
is changed to
go build -v -o livekit-server ./cmd/server

https://chatgpt.com/share/68b7effc-2580-800e-bb99-c67afb1ca8f0

better prompt,
just do the asr_llm_tts normally, first;
then add echo cancellation, thing.

The main idea is to use livekit to wrap up all asr-llm-tts stuff and let gpt-5 and claude-4o to establish a system where:
- user gets response from chatbox
- user can interrupt
- the robot never falls into the loops of answering its own output
But it seems that the problems (especially the conversation loop one) has no one fixed that publicly. (We stand to be corrected.)
---
## 1. ASR
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

### llm models - gguf's
- Llama-3.2-3B-Instruct-IQ3_M.gguf (t'was gud.)
- llama_3_Base_adult.Q2_K.gguf
- gemma-3-270m-it-F16.gguf
- LFM2-1.2B-F16.gguf

                         
- TinyLlama-1.1B-Chat-v1.0-finetune.f16.gguf
- TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf
- TinyLlama-1.1B-Chat-v1.0-finetune.Q2_K.gguf     
- TinyLlama-1.1B-Chat-v1.0-finetune.Q6_K.gguf     
- SmallThinker-3B-Preview.Q2_K.gguf               
- SmallThinker-3B-Preview.Q8_0.gguf               
- minichat-3b.q2_k.gguf
- minichat-3b.q8_0.gguf


- Qwen2.5_Uncensored_V2_Sexting.gguf (t'was gud.)
- nsfw-3b-q4_k_m.gguf
- NSFW-Ameba-3.2-1B.f16.gguf
- NSFW_13B_sft.Q2_K.gguf

### llm models - safetensors's
Novaciano/SEX_ROLEPLAY-3.2-1B [gguf'd]
Novaciano/SENTIMENTAL_SEX-3.2-1B [gguf'd]



#### llm adaptation
https://chatgpt.com/s/t_68ab14d326648191991a7a411520d4a7


vocabulary style: via prompt

breath/read style: i don't know how yet. finetune the tts model??



----

how to customize your own voice by recording via vits-piper
https://ssamjh.nz/create-custom-piper-tts-voice/


some collections of piper where you can download the correspoinding onnx-json pair by clicking `download`.
https://rhasspy.github.io/piper-samples/#en_GB-southern_english_female-low


### Installation
```
pip install -r requirements.txt
```
  - for macOS, if python==3.10, please `brew install portaudio` before executing the above so that `pyaudio` can be installed. However, the `pyaudio` **here** serves `RealtimeSTT`, instead of itself, that's why `sounddevice` is introduced.


Ensure you have:
- `llama-server` (from llama.cpp) running locally
  - e.g. 1
    ```bash
    ./llama-server --model ./models/your-model.Q4_K_M.gguf --host 0.0.0.0 --port 8080 \
      --ctx-size 4096 --parallel 2 --no-mmap
    ```
    Tune args for your hardware (quant, threads, ctx-size). For fastest first token, prefer smaller / quantized model.

  - e.g. 2
  ``` bash
  llama-server -m Llama-3.2-3B-Instruct-IQ3_M.gguf 
  ```
  which can be verified by calling
  ```bash
  curl http://localhost:8080/completion -d '{
    "prompt": "Your prompt here",
    "n_predict": 128
  }'
  ```
  if get results like 
  ```bash
  {"index":0,"content":" It's great to be here. I'm so excited ...
  ```
  Then it's okay.

- Piper ONNX voice model (e.g. `en_US-hfc_female-medium.onnx` as well as `en_US-hfc_female-medium.onnx.json`) placed in repo root (gitignored)
  - e.g. download the model from `https://huggingface.co/csukuangfj/vits-piper-en_US-hfc_female-medium`.


#### tts_models
zh_CN-huayan-medium.onnx
zh_CN-huayan-medium.onnx.json
en_US-hfc_female-medium.onnx
en_US-hfc_female-medium.onnx.json


### Run

```
python asr_llm_tts.py --llm-url http://localhost:8080 --tts-model en_US-hfc_female-medium.onnx
```
Test components:
```
python asr_llm_tts.py --test
```

- for macOS --
```bash
pip install "httpx[http2]"
pip install sounddevice
```
Make sure the `numpy` used is of version less than 2.0.



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

--------------------------------------------------



### voiceprint 
the folder of voiceprint/ illustrates the voiceprint registration from human throat via recording  vs. from onnx model, and its comparison.

to run them, you need to do:
```bash
pip install resemblyzer librosa numpy
pip install soundfile
```
- `record_voice.py` is how human read and get wav file.
- `tts_generate.py` is how onnx creates its wav file according to the piece of text.
- `voiceprint_compare.py` is how to compare the voiceprint between wav files.


### echo cancellation
#### approach 0: the vanilla dir --- vanilla_PiperVoice/


#### approach 1: only take the user's sound as THE input --- voiceprint/


user's sound is pre-recorded in voiceprint/ dir.
Not successful by far.
Now， it can know how similar the audio is with the pre-recorded sample wav file.
But the mute-record-speaker interaction is not sorted out...

#### approach 2: LiveKit + WebRTC AEC (recommended) ---  livekit_appended
LiveKit is built on top of WebRTC, and WebRTC has battle-tested Acoustic Echo Cancellation (AEC)
- [x] rewrite the asr_llm_tts.py upon the foundation of WebRTC and LiveKit.
- [ ] apply AEC then.






