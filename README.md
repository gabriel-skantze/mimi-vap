# Mimi VAP

Voice Activity Projection for two-speaker audio. Run a WAV of any duration to get
frame-level CSV predictions, or start a WebSocket server for live audio.
Both use the frozen Mimi encoder's persistent streaming state and bounded
transformer key/value caches. Audio and CSV are processed incrementally.

The default weights are the requested four-corpus Mimi / ALiBi 18M model:

```
vap-fisher-openyap-seamless-mimi-alibi18m-softkl-30f-sd0-v2/
  threshold-eval/checkpoints/
  best-joint-shift-hold-epoch=05-val_joint_shift_hold_balanced_accuracy=0.7980.ckpt
```

## Install

Python 3.10–3.12, Linux or macOS. CPU and CUDA are supported. PyTorch and
TorchAudio must have matching versions; for CUDA, install the appropriate pair
from [PyTorch](https://pytorch.org/get-started/locally/) first.

```bash
git clone git@github.com:gabriel-skantze/mimi-vap.git
cd mimi-vap
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The repository is private. To download its model release automatically, install
the [GitHub CLI](https://cli.github.com/) and run `gh auth login` with an account
that has access. Alternatively, download `mimi-vap-epoch05.pt` from this repository's
**v0.1.0 release** and pass `--checkpoint /path/to/mimi-vap-epoch05.pt`.
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

There is one row per 80 ms model frame (12.5 Hz), including a final partial frame
that is padded with silence. Empty audio produces a header with no rows.

| CSV field | Meaning |
| --- | --- |
| `frame` | Zero-based model frame index |
| `time_s`, `end_s` | Start and end of this frame in the input audio |
| `vad_0`, `vad_1` | Predicted current voice-activity probabilities; threshold at 0.5 for binary VAD |
| `p_now_0`, `p_now_1` | Relative projected activity for each speaker in the first two bins |
| `p_fut_0`, `p_fut_1` | Relative projected activity for each speaker in the last two bins |

For this 12.5 Hz model, bin widths are `[3, 6, 9, 12]` frames, so the near-future
window is approximately 0–0.72 s and the future window 0.72–2.40 s after the
current frame. The `30f` run name denotes the training data's VAD envelope,
not a 30 Hz prediction rate. Each `p_*` pair sums to one. These scores compare
projected speaker activity; they aren't calibrated probabilities that speech
will occur. `--aggregation neutral` moves silence or balanced evidence toward
0.5; `relative` preserves the original model's normalized speaker balance.

Timestamps describe audio time, not wall-clock delivery time. A prediction
becomes available after that frame's audio arrives (plus computation and any
input resampling delay). A partial final frame uses zero padding.

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

## Cache behavior and Python API

```python
import numpy as np
from mimi_vap import Predictor, predict_file

model = Predictor("models/mimi-vap-epoch05.pt", device="cpu")
predict_file(model, "conversation.wav", "predictions.csv")

with model.stream(context_seconds=20) as stream:
    rows = stream.push(np.zeros((1920, 2), dtype=np.float32))
    last_rows = stream.flush()
```

The default VAP transformer cache retains 250 frames (20 seconds) per layer;
change it with `--context-seconds`. Mimi maintains its own native bounded
convolution and attention state. Each input frame is encoded once, with new
attention queries using cached projected keys and values. WAV inference and
WebSocket inference both advance one frame at a time, independently of input
message boundaries.

Cached higher-layer representations retain information from earlier context,
so predictions can differ from re-encoding a fresh sliding window after cache
eviction. This is intentional. The streaming model isn't retrained or modified.
CPU is the default when CUDA is unavailable; `--device mps` is available for
experimentation but isn't part of the tested compatibility target.

## Development and provenance

```bash
python -m pip install -e '.[dev]'
pytest
# Optional tests using real released weights:
MIMI_VAP_CHECKPOINT=models/mimi-vap-epoch05.pt pytest
```

The inference core is adapted from Erik Ekstedt's
[VoiceActivityProjection](https://github.com/ErikEkstedt/VoiceActivityProjection),
including Gabriel Skantze's Mimi, ALiBi, normalization, and incremental cache
changes. The upstream MIT notice is retained in `LICENSE`.
[Mimi/Moshi](https://github.com/kyutai-labs/moshi) supplies the encoder runtime;
the pretrained encoder originates from `kyutai/moshiko-pytorch-bf16`.
See `MODEL_CARD.md` for checkpoint provenance and interpretation limits.
