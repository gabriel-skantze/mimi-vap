# Voice Activity Projection

Voice Activity Projection (VAP) is a model of conversational turn-taking. It
listens to the audio of a conversation and continuously predicts the future
speech activity of both participants. These predictions help distinguish a
pause within someone's turn from an opportunity for the other person to speak,
and can be used to anticipate turn shifts and backchannels. VAP learns from
speech-activity patterns in recorded conversations, without requiring explicit
turn-taking event labels or transcripts.

This repository extends the
[original VoiceActivityProjection implementation](https://github.com/ErikEkstedt/VoiceActivityProjection)
by Erik Ekstedt and Gabriel Skantze. The approach was introduced in
[Voice Activity Projection: Self-supervised Learning of Turn-taking Events](https://www.isca-archive.org/interspeech_2022/ekstedt22_interspeech.html)
(Interspeech 2022). This version predicts current voice activity (VAD) and the
relative future activity of each speaker (`p_now` and `p_fut`) every 80 ms,
with a forecast horizon of 2.4 seconds.

## How this version extends VAP

- **Stereo model:** this release focuses on two-channel conversations, with
  one speaker per channel. Separate channel representations interact through
  cross-attention. The original project supported both mono and stereo models;
  this release builds on its stereo architecture. A single speaker's mono
  recording can also be used by adding a silent second channel.
- **Mimi encoder:** the original CPC speech encoder is replaced with a frozen
  [Mimi encoder](https://arxiv.org/abs/2410.00037). The predictor uses its
  continuous features before quantization, combining acoustic and linguistic
  information at a frame rate of 12.5 Hz.
- **24 kHz audio:** Mimi processes 24 kHz waveforms, extending the 16 kHz input
  used by the upstream CPC encoder. WAV files at other sample rates are
  resampled automatically; live streaming accepts 24 kHz audio.
- **Different frame rate and longer horizon:** predictions are produced at
  **12.5 Hz** (one frame every 80 ms), compared with the upstream model's
  50 Hz (20 ms). The future activity window extends from 2.0 to **2.4 seconds**,
  using four bins with widths of 0.24, 0.48, 0.72, and 0.96 seconds.
- **Bigger model:** the VAP predictor and heads have approximately 18.3 million
  trainable parameters. The transformer has width 384, two channel
  self-attention layers, six stereo cross-attention layers, and six attention
  heads, increasing capacity over the smaller upstream architecture. Mimi
  remains frozen during training.
- **More training data:** the model was trained on approximately **6,630 hours
  of conversational recordings** from Fisher parts 1 and 2, Open Yap 1K, and
  Seamless Interaction. Together these cover telephone conversations, remote
  conversations between familiar speakers, and face-to-face interactions.
- **Offline and streaming inference:** process a WAV of any length into a CSV
  of frame-level predictions, or use a WebSocket server for live conversations.
  Both modes preserve Mimi's streaming state and use bounded transformer
  key/value (KV) caches, processing each new audio frame once. The default VAP
  cache retains 20 seconds of context, keeping memory use bounded for long
  recordings and conversations.

### Training data

The training manifest contains **1,507,616 overlapping 20-second windows** drawn
from the following recordings:

| Corpus | Conversation type | Training audio (hours) |
| --- | --- | ---: |
| [Fisher part 1](https://aclanthology.org/L04-1500/) | Two-speaker telephone calls | 830.65 |
| [Fisher part 2](https://aclanthology.org/L04-1500/) | Two-speaker telephone calls | 942.54 |
| [Open Yap 1K](https://theagenticdatacompany.com/open-yap-1k) | Natural remote conversations between people who know each other | 964.24 |
| [Seamless Interaction](https://arxiv.org/abs/2506.22554) | Face-to-face interactions in naturalistic and prompted settings | 3,892.55 |
| **Total** | | **6,629.99** |

Hours are the durations of training recordings that contributed at least one
window, counting each conversation once across both channels. They exclude
validation and test recordings and do not sum overlapping window durations;
the windows select portions of these recordings. Fisher and OpenYap were split
by speaker, and Seamless retains its supplied splits. See the
[model card](MODEL_CARD.md) for the training objective and label preparation.

The default weights are the epoch-05 checkpoint, selected by joint validation
shift/hold balanced accuracy (0.7980). Its full identity is in the model card.

## Install

Python 3.10–3.12, Linux or macOS. CPU and CUDA are supported. PyTorch and
TorchAudio must have matching versions; for CUDA, install the appropriate pair
from [PyTorch](https://pytorch.org/get-started/locally/) first.

```bash
git clone https://github.com/gabriel-skantze/mimi-vap.git
cd mimi-vap
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The default model downloads automatically on first use; no GitHub login is
required. Alternatively, download `mimi-vap-epoch05.pt` from the
[v0.1.0 release](https://github.com/gabriel-skantze/mimi-vap/releases/tag/v0.1.0)
and pass `--checkpoint /path/to/mimi-vap-epoch05.pt`.
The compact checkpoint includes Mimi encoder weights; no separate Hugging Face
download is needed. Downloads are cached and checked against a pinned SHA-256.

## WAV → CSV

```bash
mimi-vap predict conversation.wav -o predictions.csv
# Explicit local weights / device:
mimi-vap predict conversation.wav -o predictions.csv --checkpoint models/mimi-vap-epoch05.pt --device cpu
```

Mono and stereo WAVs at any sample rate are resampled to 24 kHz with a stateful
resampler. Stereo should contain one speaker per channel: left = speaker 0,
right = speaker 1. Mono is assigned to speaker 0, with a silent speaker 1.
Mixed mono recordings aren't separated or diarized.

There is one row per model frame, including a final partial frame padded with
silence. Empty audio produces a header with no rows.

| CSV field | Meaning |
| --- | --- |
| `frame` | Zero-based model frame index |
| `time_s`, `end_s` | Start and end of this frame in the input audio |
| `vad_0`, `vad_1` | Predicted current voice-activity probabilities; threshold at 0.5 for binary VAD |
| `p_now_0`, `p_now_1` | Relative projected activity for each speaker over 0–0.72 s |
| `p_fut_0`, `p_fut_1` | Relative projected activity for each speaker over 0.72–2.40 s |

Each `p_*` pair sums to one. These scores compare projected speaker activity;
they aren't calibrated probabilities that speech will occur.
`--aggregation neutral` moves silence or balanced evidence toward 0.5;
`relative` preserves the original model's normalized speaker balance.

Timestamps describe audio time, not wall-clock delivery time. A prediction
becomes available after that frame's audio arrives (plus computation and any
input resampling delay).

## Streaming WebSocket server

```bash
mimi-vap serve --host 127.0.0.1 --port 8765
# Optional: --checkpoint models/mimi-vap-epoch05.pt --device cuda
```

Run the example client to stream a WAV and save its responses:

```bash
python examples/stream_wav.py conversation.wav -o streamed.csv
```

Protocol:

1. Connect to `ws://127.0.0.1:8765`. The server sends `{"type":"ready",...}`
   with format, sample rate, frame rate, and cache context.
2. Send binary messages containing **24 kHz, two-channel, little-endian float32**
   PCM, normalized to roughly `[-1, 1]`, interleaved `[left0,right0,left1,right1,...]`.
   Message sizes may vary; 1,920 sample pairs = 80 ms is a useful default.
   Messages must contain complete sample pairs and be at most 1 MiB.
3. Each binary message receives `{"type":"predictions","frames":[...]}`.
   Each frame uses the CSV field names above. The list is empty while fewer
   than 1,920 samples are buffered. Only new frames are returned.
4. Send `{"type":"flush"}` at the end. The server pads the final incomplete
   frame, sends its predictions, then `{"type":"done","frames":N}`.
   Further audio requires a reset or a new connection.
5. Send `{"type":"reset"}` to clear encoder state, KV caches, pending audio,
   and timestamps, and receive `{"type":"reset"}`. Closing the socket also
   releases all stream state. Disconnecting without flushing drops partial audio.

Example response:

```json
{"type":"predictions","frames":[{"frame":0,"time_s":0.0,"end_s":0.08,"vad_0":0.8,"vad_1":0.1,"p_now_0":0.7,"p_now_1":0.3,"p_fut_0":0.6,"p_fut_1":0.4}]}
```

One server process supports **one active client**. Additional clients receive a
busy error; this prevents sharing Mimi's mutable encoder state across sessions.
Use multiple processes/ports for independent conversations. Inference runs on
a dedicated worker thread so network handling remains responsive. The server
binds to localhost by default; use your own authentication/TLS proxy if exposing
it to a network.

## Python API

```python
import numpy as np
from mimi_vap import Predictor, predict_file

model = Predictor("models/mimi-vap-epoch05.pt", device="cpu")
predict_file(model, "conversation.wav", "predictions.csv")

with model.stream(context_seconds=20) as stream:
    rows = stream.push(np.zeros((1920, 2), dtype=np.float32))
    last_rows = stream.flush()
```

Set `context_seconds` in Python or `--context-seconds` on the command line to
change the VAP cache length. Input chunks may have arbitrary lengths; incomplete
frames are buffered until more samples arrive or `flush()` is called.

Cached higher-layer representations retain information from earlier context,
so predictions can differ from re-encoding a fresh sliding window after cache
eviction. This is intentional. The streaming model isn't retrained or modified.
CPU is the default when CUDA is unavailable; `--device mps` is available for
experimentation but isn't part of the tested compatibility target.

## Development

```bash
python -m pip install -e '.[dev]'
pytest
# Optional tests using real released weights:
MIMI_VAP_CHECKPOINT=models/mimi-vap-epoch05.pt pytest
```

The upstream MIT notice is retained in `LICENSE`. Checkpoint and encoder
provenance are documented in the [model card](MODEL_CARD.md).

## Citation

If you use this model in a paper, please cite the original VAP paper:

Erik Ekstedt and Gabriel Skantze (2022).
[Voice Activity Projection: Self-supervised Learning of Turn-taking Events](https://www.isca-archive.org/interspeech_2022/ekstedt22_interspeech.html).
*Interspeech 2022*, pp. 5190–5194.

```bibtex
@inproceedings{ekstedt22_interspeech,
  author    = {Erik Ekstedt and Gabriel Skantze},
  title     = {{Voice Activity Projection: Self-supervised Learning of Turn-taking Events}},
  booktitle = {Proc. Interspeech 2022},
  year      = {2022},
  pages     = {5190--5194},
  doi       = {10.21437/Interspeech.2022-10955}
}
```

Please also cite the Mimi/Moshi paper for the pretrained encoder (see below),
and identify this repository, release, and checkpoint so readers can reproduce
which variant you used. The following software reference supplements the VAP
paper citation:

```bibtex
@misc{skantze2026mimivap,
  author       = {Gabriel Skantze},
  title        = {{Mimi VAP}: Voice Activity Projection with a Mimi Encoder},
  year         = {2026},
  howpublished = {Software, version 0.1.0},
  url          = {https://github.com/gabriel-skantze/mimi-vap},
  note         = {Epoch-05 checkpoint selected by joint validation shift/hold
                  balanced accuracy (0.7980)}
}
```

For your methods section, describe the changes summarized in
[How this version extends VAP](#how-this-version-extends-vap), and report the
checkpoint, cache context length, and probability aggregation setting used in
your experiments. Technical details of the soft-target objective and label
preparation are in the [model card](MODEL_CARD.md).

### Related papers and resources

- **Stereo VAP:** Koji Inoue, Bing'er Jiang, Erik Ekstedt, Tatsuya Kawahara,
  and Gabriel Skantze (2024).
  [Real-time and Continuous Turn-taking Prediction Using Voice Activity Projection](https://arxiv.org/abs/2401.04868).
- **Mimi encoder:** Alexandre Défossez, Laurent Mazaré, Manu Orsini,
  Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave, and Neil Zeghidour
  (2024). [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037).
- **ALiBi:** Ofir Press, Noah A. Smith, and Mike Lewis (2022).
  [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409).
  *ICLR 2022*. ALiBi was already used in the upstream stereo VAP code.
- **Attention scaling:** Ashish Vaswani et al. (2017).
  [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
  The predictor scales attention scores by `1 / sqrt(head_dim)`.
- **Training-label representation:** Sanyuan Chen et al. (2022).
  [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900).
  WavLM was used in the training-label pipeline; Mimi is the inference encoder.

Dataset references are linked in [Training data](#training-data); the
[Open Yap 1K release](https://github.com/The-Agentic-Data-Co/open-yap-1k#citation)
provides its recommended dataset citation.
