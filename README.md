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

### How this model differs

This release builds on the waveform-only stereo VAP implementation, whose
self-attention and cross-attention architecture is described by
[Inoue et al. (2024)](https://arxiv.org/abs/2401.04868).
The changes below describe this repository's variant; they should be reported
alongside the original VAP citation.

- **Speech representation:** the CPC encoder is replaced with a frozen
  [Mimi encoder from Moshi](https://arxiv.org/abs/2410.00037). It supplies
  continuous 512-dimensional features before quantization, using 24 kHz audio
  and a 12.5 Hz frame rate. The model does not use discrete Mimi tokens.
- **Predictor:** approximately 18.3M trainable parameters, with model width 384,
  two channel self-attention layers, six stereo cross-attention layers, and six
  attention heads. It retains the upstream stereo implementation's
  [ALiBi positional biases](https://arxiv.org/abs/2108.12409), uses standard
  per-head attention scaling, and adds normalization of the cross-attention
  source and VAD-head input.
- **Forecast targets:** four bins per speaker have widths `[3, 6, 9, 12]`
  frames, covering 2.4 seconds in total. Instead of thresholding each bin into
  a hard activity bit, the training target uses its speech-occupancy fraction
  as a Bernoulli probability. Products over the eight bits define a soft
  distribution over the 256 VAP classes. Training minimizes
  `KL(soft target || predicted distribution)` together with binary
  cross-entropy for current VAD. This soft-target construction is a change
  in this implementation, not a method attributed to the original VAP paper.
- **Training:** a fresh predictor and VAP/VAD heads were trained over frozen
  Mimi features on Fisher parts 1 and 2, Open Yap 1K, and Seamless Interaction.
  Fisher and OpenYap splits are speaker-disjoint; Seamless uses its supplied
  splits. Fisher part 1 uses its original activity regions, while the other
  sources use labels produced by a WavLM-based VAD model. Same-channel speech
  regions separated by less than 200 ms were merged. The released model was
  selected by pooled validation shift/hold balanced accuracy.
- **Inference:** persistent Mimi state and bounded per-layer KV caches enable
  streaming without re-encoding the entire context. The default VAP cache is
  20 seconds. This is an inference implementation change; it does not alter
  the released weights. Report the context length and `relative` or `neutral`
  aggregation setting used in your experiments.

A short methods description you can adapt:

> We used a modified stereo Voice Activity Projection model (Ekstedt and
> Skantze, 2022; Inoue et al., 2024), replacing its CPC encoder with frozen,
> continuous Mimi features (Défossez et al., 2024). The model operates at
> 12.5 Hz and uses an approximately 18.3M-parameter ALiBi transformer to predict
> a joint distribution over two speakers' activity in four future bins spanning
> 2.4 s, together with current voice activity. It was trained on Fisher,
> Open Yap 1K, and Seamless Interaction using occupancy-based soft VAP targets
> and a VAD objective. We used the epoch-05 checkpoint from Mimi VAP v0.1.0,
> selected by validation shift/hold balanced accuracy, with a 20 s KV cache and
> relative speaker-probability aggregation.

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

For training-data provenance, also see the
[Fisher corpus paper (Cieri, Miller, and Walker, 2004)](https://aclanthology.org/L04-1500/),
the [Open Yap 1K dataset release and citation](https://github.com/The-Agentic-Data-Co/open-yap-1k#citation),
and the [Seamless Interaction dataset paper (Agrawal et al., 2025)](https://arxiv.org/abs/2506.22554).
The soft-target objective, normalization changes, and cache implementation
are documented here as modifications to this model; these references do not
constitute a separate publication evaluating this exact checkpoint.
