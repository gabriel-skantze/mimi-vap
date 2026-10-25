"""The same frame-by-frame inference path for files and live streams."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch

from ._core.modules.VAP import load_model_from_state_dict

SAMPLE_RATE = 24000
FRAME_SAMPLES = 1920
FRAME_HZ = 12.5
CSV_FIELDS = ["frame", "time_s", "end_s", "vad_0", "vad_1",
              "p_now_0", "p_now_1", "p_fut_0", "p_fut_1"]


class Predictor:
    """One loaded model. Only one Stream may be active on it at a time."""

    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = load_model_from_state_dict(str(checkpoint)).eval().to(self.device)
        if self.model.sample_rate != SAMPLE_RATE or self.model.frame_hz != FRAME_HZ:
            raise ValueError("Expected a 24 kHz / 12.5 Hz Mimi VAP checkpoint")
        self._active = False

    def stream(self, context_seconds: float = 20.0,
               aggregation: str = "relative") -> Stream:
        return Stream(self, context_seconds, aggregation)


class Stream:
    """Accept float32 [samples, 2] at 24 kHz; buffer incomplete 80 ms frames."""

    def __init__(self, predictor: Predictor, context_seconds: float = 20.0,
                 aggregation: str = "relative"):
        if not math.isfinite(context_seconds) or not 0.08 <= context_seconds <= 600:
            raise ValueError("context_seconds must be between 0.08 and 600")
        if aggregation not in {"relative", "neutral"}:
            raise ValueError("aggregation must be relative or neutral")
        if predictor._active:
            raise RuntimeError("This predictor already has an active stream")
        self.predictor = predictor
        self.aggregation = aggregation
        self.pending = np.empty((0, 2), dtype=np.float32)
        self.samples = 0
        self.frame = 0
        self.closed = False
        self.finished = False
        self.state = predictor.model.start_streaming(1, context_seconds)
        predictor._active = True

    @torch.inference_mode()
    def _frame(self, audio: np.ndarray, end_s: float) -> dict:
        waveform = torch.from_numpy(np.ascontiguousarray(audio.T)).unsqueeze(0)
        out = self.predictor.model.stream_step(waveform.to(self.predictor.device), self.state)
        probs = out["logits"].softmax(-1)
        objective = self.predictor.model.objective
        now_0 = objective.probs_next_speaker_aggregate(
            probs, 0, 1, aggregation=self.aggregation)[0, 0].item()
        fut_0 = objective.probs_next_speaker_aggregate(
            probs, 2, 3, aggregation=self.aggregation)[0, 0].item()
        vad = out["vad"].sigmoid()[0, 0].cpu().tolist()
        row = dict(zip(CSV_FIELDS, [self.frame, self.frame / FRAME_HZ, end_s,
                                   *vad, now_0, 1 - now_0, fut_0, 1 - fut_0]))
        self.frame += 1
        return row

    def push(self, audio: np.ndarray) -> list[dict]:
        if self.closed or self.finished:
            raise RuntimeError("Stream is closed or already flushed")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 2 or audio.shape[1] != 2:
            raise ValueError("Expected audio shaped [samples, 2]")
        if not np.isfinite(audio).all():
            raise ValueError("Audio must contain finite values")
        self.samples += len(audio)
        audio = np.concatenate((self.pending, audio))
        complete = len(audio) // FRAME_SAMPLES * FRAME_SAMPLES
        rows = []
        for start in range(0, complete, FRAME_SAMPLES):
            rows.append(self._frame(audio[start:start + FRAME_SAMPLES],
                                    (self.frame + 1) / FRAME_HZ))
        self.pending = audio[complete:].copy()
        return rows

    def flush(self) -> list[dict]:
        if self.closed or self.finished:
            raise RuntimeError("Stream is closed or already flushed")
        rows = []
        if len(self.pending):
            padded = np.zeros((FRAME_SAMPLES, 2), dtype=np.float32)
            padded[:len(self.pending)] = self.pending
            rows.append(self._frame(padded, self.samples / SAMPLE_RATE))
        self.pending = np.empty((0, 2), dtype=np.float32)
        self.finished = True
        return rows

    @torch.inference_mode()
    def reset(self) -> None:
        if self.closed:
            raise RuntimeError("Stream is closed")
        self.predictor.model.reset_streaming(self.state)
        self.pending = np.empty((0, 2), dtype=np.float32)
        self.samples = self.frame = 0
        self.finished = False

    def close(self) -> None:
        if not self.closed:
            self.predictor.model.stop_streaming(self.state)
            self.predictor._active = False
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def audio_blocks(path: str | Path):
    """Read and resample in bounded blocks, including the resampler tail."""
    with sf.SoundFile(path) as source:
        if source.channels not in (1, 2):
            raise ValueError("WAV must have one or two channels, one speaker per channel")
        resampler = None if source.samplerate == SAMPLE_RATE else soxr.ResampleStream(
            source.samplerate, SAMPLE_RATE, 2, dtype="float32")
        emitted = 0
        for block in source.blocks(blocksize=source.samplerate, dtype="float32", always_2d=True):
            if source.channels == 1:
                block = np.concatenate((block, np.zeros_like(block)), axis=1)
            if resampler is not None:
                block = resampler.resample_chunk(block)
            emitted += len(block)
            yield block
        if resampler is not None:
            block = resampler.resample_chunk(np.empty((0, 2), dtype=np.float32), last=True)
            emitted += len(block)
            yield block
        # Preserve even a fractional target sample at the end of very short input.
        expected = (source.frames * SAMPLE_RATE + source.samplerate - 1) // source.samplerate
        if expected > emitted:
            yield np.zeros((expected - emitted, 2), dtype=np.float32)


def predict_file(predictor: Predictor, audio: str | Path, output: str | Path,
                 context_seconds: float = 20.0, aggregation: str = "relative") -> int:
    """Write one CSV row per model frame, with memory independent of WAV length."""
    if Path(audio).resolve() == Path(output).resolve():
        raise ValueError("Input audio and output CSV must be different files")
    with sf.SoundFile(audio) as source:
        duration = source.frames / source.samplerate
        if source.channels not in (1, 2):
            raise ValueError("WAV must have one or two channels")
    with predictor.stream(context_seconds, aggregation) as stream:
        with Path(output).open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for block in audio_blocks(audio):
                rows = stream.push(block)
                for row in rows:
                    row["end_s"] = min(row["end_s"], duration)
                writer.writerows(rows)
            rows = stream.flush()
            for row in rows:
                row["end_s"] = min(row["end_s"], duration)
            writer.writerows(rows)
        return stream.frame
