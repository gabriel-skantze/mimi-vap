# Default model

The release contains the exact model selected by Gabriel Skantze:

- Run: `vap-fisher-openyap-seamless-mimi-alibi18m-softkl-30f-sd0-v2`
- Selection: `best-joint-shift-hold-epoch=05-val_joint_shift_hold_balanced_accuracy=0.7980.ckpt`
- Epoch: 5 (zero-based), global step: 282678.
- Selection metric: joint validation shift/hold balanced accuracy, 0.7980.
- Sources: Fisher part 1, Fisher part 2, OpenYap, and Seamless.
- Architecture: frozen Mimi encoder, stereo ALiBi transformer, head attention
  scaling, cross-source and VAD normalization, approximately 18M trainable parameters.
- Input: 24,000 Hz, two aligned speaker channels.
- Output: 12.5 Hz, two VAD probabilities and a 256-class projection distribution,
  aggregated into `p_now` and `p_fut` for each speaker by this package.
- Projection widths: `[0.24, 0.48, 0.72, 0.96]` seconds = `[3, 6, 9, 12]` frames.
- Training objective: occupancy-aware soft-target VAP KL plus VAD BCE.

## Training details

Mimi provides continuous 512-dimensional features before quantization; discrete
codec tokens are not used. The 384-dimensional predictor has two channel
self-attention layers, six stereo cross-attention layers, and six attention
heads. It retains ALiBi, uses `1 / sqrt(head_dim)` attention scaling, and
normalizes the cross-attention source and VAD-head input.

For each of the eight speaker/future-bin bits, the target probability is the
fraction of active VAD frames in that bin. Products of these Bernoulli
probabilities define a soft distribution over the 256 VAP classes. Training
minimizes `KL(soft target || predicted distribution)` plus binary cross-entropy
for current VAD. This occupancy-based objective replaces hard bin labels in
this variant; it is not attributed to the original VAP paper.

The predictor and VAP/VAD heads were trained from scratch over frozen Mimi
features. Fisher part 1 uses its original activity regions; Fisher part 2,
OpenYap, and Seamless use labels produced by a WavLM-based VAD model. Same-channel
speech regions separated by less than 200 ms were merged. Fisher and OpenYap
have speaker-disjoint splits; Seamless retains its supplied splits. Corpus
sizes are reported in the [README](README.md#training-data).

The `30f` run name refers to the 30-frame future horizon: `[3, 6, 9, 12]` frames
sum to 2.4 seconds at 12.5 Hz.

## Checkpoint export and interpretation

`mimi-vap-epoch05.pt` is a compact state dictionary containing only the original
Lightning checkpoint's `model.*` tensors with that prefix removed. It includes
the frozen Mimi encoder. Optimizer, scheduler, and training data aren't included.
No quantization, fine-tuning, or weight changes were applied during export.
`src/mimi_vap/model.json` records the download location and exact SHA-256.

The selected checkpoint's 0.7980 is its internal validation selection metric.
This release does not claim held-out TurnBench results for this checkpoint;
results from other checkpoints or fine-tuned models aren't interchangeable.

The model expects separate speaker channels. It does not perform diarization or
source separation. Relative speaker projection scores aren't calibrated
speech-occurrence or turn-end probabilities. Predictions on a silent second
channel or out-of-domain audio should be interpreted accordingly.

The inference core derives from the MIT-licensed VoiceActivityProjection
project. The Mimi architecture/runtime is provided by Kyutai's Moshi package;
encoder initialization used `kyutai/moshiko-pytorch-bf16`. The retained upstream
code license is in `LICENSE`. That code license does not establish an independent
license for the model weights or training corpora; this release makes no
additional grant for redistribution of those materials.
