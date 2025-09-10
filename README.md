# Real-Time Voice Conversation System (ASR + LLM + TTS) 
Voice Conversation Pipeline (ASR → LLM → TTS) 
Low-latency, interruption-aware voice assistant pipeline:

Workflow:
```
Mic → RealtimeSTT → Prompt Build → llama.cpp /completion (SSE) → Token Buffer → Piper TTS → Speaker
          ^                                                           |
          | (pause during prefill)                                    v
    User speech interrupts  <----  Immediate TTS stop & queue flush (barge-in)
```

[ ] ASR：interruption， latency, and echo cancellation

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


---

## llm models 
### gguf's
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


### safetensors's
Novaciano/SEX_ROLEPLAY-3.2-1B [gguf'd]
Novaciano/SENTIMENTAL_SEX-3.2-1B [gguf'd]


----

## tts 
### models
zh_CN-huayan-medium.onnx
zh_CN-huayan-medium.onnx.json
en_US-hfc_female-medium.onnx
en_US-hfc_female-medium.onnx.json

- Piper ONNX voice model (e.g. `en_US-hfc_female-medium.onnx` as well as `en_US-hfc_female-medium.onnx.json`) placed in repo root (gitignored)
  - e.g. download the model from `https://huggingface.co/csukuangfj/vits-piper-en_US-hfc_female-medium`.

### how to customize your own voice by recording via vits-piper
https://ssamjh.nz/create-custom-piper-tts-voice/

### some collections of piper where you can download the correspoinding onnx-json pair by clicking `download`.
https://rhasspy.github.io/piper-samples/#en_GB-southern_english_female-low


## Installation
### pip packages
```
pip install -r requirements.txt
```
  - for macOS, if python==3.10, please `brew install portaudio` before executing the above so that `pyaudio` can be installed. However, the `pyaudio` **here** serves `RealtimeSTT`, instead of itself, that's why `sounddevice` is introduced.

### llama.cpp
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


## vanilla_PiperVoice/ 
run:
```bash
python asr_llm_tts.py --llm-url http://localhost:8080 --tts-model en_US-hfc_female-medium.onnx
```


- for macOS ---
```bash
pip install "httpx[http2]"
pip install sounddevice
```
Make sure the `numpy` used is of version **less than 2.0**.

sometimes the dialogue stops because the llm part is not functioning properly.

-----------------

Let us disentangle the asr_llm_tts.py's code structure.
- initializes asr, llm, and tts.
- Creates an audio playback worker thread that continuously monitors an audio queue
  - interrupt_tts : Event to signal TTS interruption
  - user_speaking : Event indicating user is currently speaking
  - ai_should_be_quiet : Event to prevent AI from speaking
  - stream_generation : Counter to cancel stale responses
- asr calback registration
  - `_on_recording_start` : Triggered when user starts speaking
  - `_on_recording_stop` : Triggered when user stops speaking
  - `_on_transcription_start` : Triggered when transcription begins

### When the user starts speaking while AI is talking:
#### Voice Activity Detection (VAD)
- 1. Immediate Detection : `_on_recording_start` callback fires instantly
- 2. Generation Increment : Increments stream_generation counter, making any ongoing LLM/TTS responses "stale"
- 3. Flag Setting : Sets multiple interruption flags:
  - user_speaking.set() : Indicates user is actively speaking
  - ai_should_be_quiet.set() : Prevents new AI speech
- 4. Emergency Stop : Calls `interrupt_tts_immediately` which sets interrupt_tts event

#### Audio Interruption
The `audio_playback_worker` continuously monitors interruption flags:
1. Queue Clearing : When interruption is detected, immediately clears all pending audio chunks from the queue
2. Audio Stop : On macOS, calls sd.stop() to halt current audio playback
3. Chunk Skipping : Any remaining audio chunks are discarded without playing

#### Speech Recognition
1. Transcription : User's speech is transcribed in real-time
2. Processing : When user stops speaking, `_on_recording_stop` clears interruption flags
3. Callback : Recognized text is passed to `process_speech_input`


## voiceprint/
user's sound is pre-recorded in voiceprint/ dir.
Now， it can know how similar the audio is with the pre-recorded sample wav file.

## livekit_appended/: LiveKit + WebRTC AEC 
LiveKit is built on top of WebRTC, and WebRTC has battle-tested Acoustic Echo Cancellation (AEC)
- [x] rewrite the asr_llm_tts.py upon the foundation of WebRTC and LiveKit.
- [ ] apply AEC then.

## temp/: liveKit-based 
The main idea is to use livekit to wrap up all asr-llm-tts stuff and let gpt-5 and claude-4o to establish a system where:
- user gets response from chatbox
- user can interrupt
- the robot never falls into the loops of answering its own output
But it seems that the problems (especially the conversation loop one) has no one fixed that publicly. (We stand to be corrected.)

- https://chatgpt.com/share/68b7e7a6-6cb0-800e-9af5-3402967aecb2
- where: 
  ```
  go build -v -o livekit-server ./cmd/livekit-server 
  ```
is changed to
  ```
  go build -v -o livekit-server ./cmd/server
  ```

https://chatgpt.com/share/68b7effc-2580-800e-bb99-c67afb1ca8f0

