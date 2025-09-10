# Real-Time Voice Conversation System (ASR + LLM + TTS) 
Voice Conversation Pipeline (ASR → LLM → TTS) 

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
[ ] when heard a voice and starts recording, recognise the source and judge if it is its own. If so, quit the recording.
  - [ ] require fast voice print generation as well as fast comparison
  - [ ] extend the recorded wav test to ongoing  test



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


------------------------------------------------------------
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


### acoustic echo cancellation （AEC）
The technique that makes the chatting robot never answers its own output on speaker the approach of acoustic is echo canclellation.
The problem of a chat robot "answering its own output" when it plays audio through a speaker and simultaneously listens with a microphone is essentially acoustic echo. 
The solution is Acoustic Echo Cancellation (AEC), which is a DSP (digital signal processing) technique that removes the echo of the loudspeaker’s signal from the microphone’s input.

There are several mature, open-sourced implementations:
#### WebRTC AEC
The WebRTC project (by Google) has a state-of-the-art echo canceller (AEC, AEC2, and AEC3). It’s widely regarded as one of the best open-source AEC solutions.

Libraries:
- webrtcvad (Python, though mainly for VAD)
- py-webrtc-audio-processing (Python wrapper around WebRTC’s AEC, NS, AGC, etc.)
- C++ version is built into the WebRTC stack.

#### SpeexDSP
Part of the Speex project. Provides a simpler echo canceller (less advanced than WebRTC AEC). Easier to integrate in lightweight systems (e.g., embedded).

#### Pulseaudio / PipeWire echo cancellation
Linux audio servers like PulseAudio and PipeWire have built-in echo cancellation modules. For example:
```bash
pactl load-module module-echo-cancel
```
They use either Speex or WebRTC AEC internally.

#### Deep learning based AEC (research)
Some newer approaches use neural nets, e.g., AdenoNet, NSNet2, etc. These are not as mature for real-time embedded use, but available in research repos.

#### approach 0: the vanilla dir --- vanilla_PiperVoice/ 
in vanilla_PiperVoice, modify asr_llm_tts.py :
GPT-5 says ----

Since you’re in Python, you can wrap an AEC engine:
- py-webrtc-audio-processing
Provides bindings for WebRTC AEC, noise suppression, gain control, etc.
Best quality for real-time robot applications.
- pyaudio + WebRTC AEC (lower-level integration).
- If running Linux desktop → you can rely on PulseAudio/ PipeWire to do AEC system-wide (simpler, but less portable).

Keep a short buffer of TTS samples being played. Or an external wav file of audio sample.

Since the output sound is changing via selecting different tts model, the way to contrl 'echo cancellation' should be within software realm instead of hardware realm, as far as I am concerned.

The problem is relevant with asr, that's fo' sho'.


my prompt given to chatgpt:
```text
suppose i have a wav file (/Users/shuyuew/Documents/GitHub/on_device_asr-llm-tts/voiceprint/tts_ref_0.wav) which is a 5-second sound sample of the output. Please modify the script so that, when it listens to any audio , it does not immediately send it to subsequent llm and tts models, but detect if it is similar to the wav file: if so, just ignore that like nothing happens; if not, do the interrupttion.for every detection/judement, print out if it is 'simillar' or 'not similar', so i can see if it works. I need a full complete updated scripts: 
(asr_llm_tts.py) ``; and (llm_tts.py) `` 
```

```text
suppose i have a wav file (/Users/shuyuew/Documents/GitHub/on_device_asr-llm-tts/voiceprint/tts_ref_0.wav) which is a 5-second sound sample of the output. Please modify my following script so that, when it listens to any audio, it does not immediately send it to subsequent llm and tts models, but detect every chunk if it is similar to the wav file: if yes, just ignore ; if not, do the interrupttion.for every detection/judement, print out if it is 'simillar' or 'not similar', so i can see if it works. Speed is prior to accuracy, and make the similarity check robust. I need a full complete update of my script: `` 
```

```text
suppose i have a wav file (/Users/shuyuew/Documents/GitHub/on_device_asr-llm-tts/voiceprint/tts_ref_0.wav) which is a 5-second sound sample of the output. Please write a new script so that, when it listens to any audio, 

it detects every chunk if it is similar to the wav file: if yes, just ignore ; if not,  print out if it is 'simillar' or 'not similar', send to my following script: `` 
```


```text

What shoul




suppose I have a wav file (/Users/shuyuew/Documents/GitHub/on_device_asr-llm-tts/voiceprint/tts_ref_0.wav) which is a 5-second sound sample of the output. Please write a new script so that, when it listens to any audio, 

it detects every chunk if it is similar to the wav file: if yes, just ignore ; if not,  print out if it is 'simillar' or 'not similar', send to my following script: `` 
```


```zsh
brew install portaudio
pip install sounddevice numpy py-webrtc-audio-processing
pip install pyobjc
pip install pyannote.audio torchaudio librosa
pip install python_speech_features
```
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

##### When the user starts speaking while AI is talking:

Phase 1 Voice Activity Detection (VAD)
- 1. Immediate Detection : `_on_recording_start` callback fires instantly
- 2. Generation Increment : Increments stream_generation counter, making any ongoing LLM/TTS responses "stale"
- 3. Flag Setting : Sets multiple interruption flags:
  - user_speaking.set() : Indicates user is actively speaking
  - ai_should_be_quiet.set() : Prevents new AI speech
- 4. Emergency Stop : Calls `interrupt_tts_immediately` which sets interrupt_tts event

Phase 2 Audio Interruption

The `audio_playback_worker` continuously monitors interruption flags:
1. Queue Clearing : When interruption is detected, immediately clears all pending audio chunks from the queue
2. Audio Stop : On macOS, calls sd.stop() to halt current audio playback
3. Chunk Skipping : Any remaining audio chunks are discarded without playing

### Phase 3: Speech Recognition
1. Transcription : User's speech is transcribed in real-time
2. Processing : When user stops speaking, `_on_recording_stop` clears interruption flags
3. Callback : Recognized text is passed to `process_speech_input`



```text
In the following script, i need you to add a function that for each chunk, its similarity against a given wav file (/Users/shuyuew/Documents/GitHub/on_device_asr-llm-tts/voiceprint/tts_ref_0.wav)  is tested and print out the result (yes/no, value): ``


```

supoose we have known the similarity of each chunk that is input from the user.
If the similarity is above a threshold, we just ignore it.
If the similarity is below the threshold, we send it to the llm.
But the recording is already on and the output is instantly stale...
Thus the IMMEDIATE audio source recognition is required.

[x] i should save all the temporarry wav file **from the user** of each chunk in input audio.
[ ] i should test and find out the fastest way to get the similarity check on every chunk.
[ ] I will add IMMEDIATE similarity comparison to EACH CHUNK.



[ ] If that does not affect the main pipeline, i go on to advance the comparison earlier.
The comparison should be done within several milliseconds, because 
- for audio output buffer, there are 256 samples:  At 16 kHz sample rate: 256 ÷ 16,000 = 16 milliseconds; At 22.05 kHz sample rate (TTS): 256 ÷ 22,050 = 11.6 milliseconds.
- for asr input butter, there are 512 sampls: - At 16 kHz sample rate: 512 ÷ 16,000 = 32 milliseconds; This is the duration of each audio chunk processed by the speech recognition system.




sometimes the dialogue stops because the llm part is not functioning properly.





#### approach 1: only take the user's sound as THE input --- voiceprint/
user's sound is pre-recorded in voiceprint/ dir.
Not successful by far.
Now， it can know how similar the audio is with the pre-recorded sample wav file.
But the mute-record-speaker interaction is not sorted out...

#### approach 2: LiveKit + WebRTC AEC  ---  livekit_appended/
LiveKit is built on top of WebRTC, and WebRTC has battle-tested Acoustic Echo Cancellation (AEC)
- [x] rewrite the asr_llm_tts.py upon the foundation of WebRTC and LiveKit.
- [ ] apply AEC then.

#### approach 3: LiveKit-based ---  temp/
temp/ is another attempt.
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

better prompt,
just do the asr_llm_tts normally, first;
then add echo cancellation, thing.

The main idea is to use livekit to wrap up all asr-llm-tts stuff and let gpt-5 and claude-4o to establish a system where:
- user gets response from chatbox
- user can interrupt
- the robot never falls into the loops of answering its own output
But it seems that the problems (especially the conversation loop one) has no one fixed that publicly. (We stand to be corrected.)




