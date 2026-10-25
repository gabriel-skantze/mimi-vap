"""Frozen Kyutai Mimi encoder for VAP."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn


class EncoderMimi(nn.Module):
    """Waveform-to-latent Mimi encoder without decoder or quantizer weights.

    Mimi is loaded through Kyutai's native PyTorch implementation, then only
    the modules required by ``encode_to_latent(..., quantize=False)`` are kept.
    This keeps VAP checkpoints substantially smaller than registering the full
    audio codec.
    """

    DEFAULT_REPO_ID = "kyutai/moshiko-pytorch-bf16"
    DEFAULT_FILENAME = "tokenizer-e351c8d8-checkpoint125.safetensors"

    def __init__(
        self,
        load_pretrained: bool = True,
        freeze: bool = True,
        repo_id: str = DEFAULT_REPO_ID,
        filename: str = DEFAULT_FILENAME,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.sample_rate = 24_000
        self.frame_hz = 12.5
        self.dim = 512
        self.output_dim = self.dim
        self.requires_exact_frame_alignment = True
        self.frame_size = int(self.sample_rate / self.frame_hz)
        self.repo_id = repo_id
        self.filename = filename
        self._streaming_stack: Optional[ExitStack] = None
        self._streaming_batch_size: Optional[int] = None

        # A stable state key lets compact-checkpoint loading distinguish Mimi
        # from CPC without depending on third-party internal parameter names.
        self.register_buffer("_mimi_version_marker", torch.tensor(1), persistent=True)

        try:
            from moshi.models.loaders import get_mimi
        except ImportError as error:  # pragma: no cover - exercised on Berzelius
            raise ImportError(
                "EncoderMimi requires moshi==0.2.13; install this package with its declared dependencies"
            ) from error

        weights: Optional[Path] = None
        if load_pretrained:
            if checkpoint_path:
                weights = Path(checkpoint_path).expanduser()
                if not weights.is_file():
                    raise FileNotFoundError(weights)
            else:
                from huggingface_hub import hf_hub_download

                weights = Path(hf_hub_download(repo_id=repo_id, filename=filename))

        mimi = get_mimi(weights, device="cpu")
        self.encoder = mimi.encoder
        self.encoder_transformer = mimi.encoder_transformer
        self.downsample = mimi.downsample
        del mimi

        if freeze:
            self.freeze()

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def unfreeze(self) -> None:
        raise RuntimeError("EncoderMimi is intentionally frozen for this VAP variant")

    def train(self, mode: bool = True):
        """Keep frozen Mimi layers deterministic when the parent model trains."""
        super().train(False)
        return self

    def forward(self, waveform: Tensor) -> Tensor:
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError(
                f"EncoderMimi expects [batch, 1, samples], got {tuple(waveform.shape)}"
            )
        if waveform.shape[-1] <= 0:
            raise ValueError("EncoderMimi requires non-empty audio")

        # Match Mimi's non-streaming padding contract. Candor's 20-second,
        # 24-kHz windows are already exact multiples of the 1,920-sample frame.
        from moshi.modules.conv import pad_for_conv1d

        expected_frames = (waveform.shape[-1] + self.frame_size - 1) // self.frame_size
        with torch.no_grad():
            latent = self.encoder(
                pad_for_conv1d(waveform, self.frame_size, self.frame_size)
            )
            if self.encoder_transformer is not None:
                (latent,) = self.encoder_transformer(latent)
            latent = self.downsample(latent)

        if latent.shape[-1] != expected_frames:
            raise RuntimeError(
                "Mimi frame alignment failed: "
                f"expected {expected_frames}, got {latent.shape[-1]}"
            )
        return latent.transpose(1, 2).contiguous()

    def start_streaming(self, batch_size: int) -> None:
        """Start persistent native Mimi state for exact 1,920-sample frames."""
        if self._streaming_stack is not None:
            raise RuntimeError("EncoderMimi is already streaming")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        stack = ExitStack()
        try:
            stack.enter_context(self.encoder.streaming(batch_size))
            if self.encoder_transformer is not None:
                stack.enter_context(self.encoder_transformer.streaming(batch_size))
            stack.enter_context(self.downsample.conv.streaming(batch_size))
        except Exception:
            stack.close()
            raise
        self._streaming_stack = stack
        self._streaming_batch_size = batch_size

    def reset_streaming(self) -> None:
        if self._streaming_stack is None:
            raise RuntimeError("EncoderMimi is not streaming")
        self.encoder.reset_streaming()
        if self.encoder_transformer is not None:
            self.encoder_transformer.reset_streaming()
        self.downsample.conv.reset_streaming()

    def stop_streaming(self) -> None:
        if self._streaming_stack is None:
            return
        self._streaming_stack.close()
        self._streaming_stack = None
        self._streaming_batch_size = None

    def stream_step(self, waveform: Tensor) -> Tensor:
        if self._streaming_stack is None or self._streaming_batch_size is None:
            raise RuntimeError("Call start_streaming() before stream_step()")
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError(
                f"EncoderMimi expects [batch, 1, samples], got {tuple(waveform.shape)}"
            )
        if waveform.shape[0] != self._streaming_batch_size:
            raise ValueError(
                f"Streaming batch size {self._streaming_batch_size} != {waveform.shape[0]}"
            )
        samples = waveform.shape[-1]
        if samples <= 0 or samples % self.frame_size:
            raise ValueError(
                f"Streaming Mimi input must be a positive multiple of {self.frame_size}, "
                f"got {samples}"
            )

        with torch.no_grad():
            latent = self.encoder(waveform)
            if self.encoder_transformer is not None:
                (latent,) = self.encoder_transformer(latent)
            latent = self.downsample(latent)
        expected_frames = samples // self.frame_size
        if latent.shape[-1] != expected_frames:
            raise RuntimeError(
                "Streaming Mimi frame alignment failed: "
                f"expected {expected_frames}, got {latent.shape[-1]}"
            )
        return latent.transpose(1, 2).contiguous()
