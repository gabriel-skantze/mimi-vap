import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mimi_vap import Predictor
from mimi_vap._core.modules.VAP import VAP
from mimi_vap._core.modules.modules import TransformerStereo


class TinyEncoder(nn.Module):
    sample_rate = 24000
    frame_hz = 12.5
    frame_size = 1920
    dim = 16

    def start_streaming(self, batch_size):
        pass

    def stop_streaming(self):
        pass

    def reset_streaming(self):
        pass

    def stream_step(self, waveform):
        x = waveform.unfold(-1, 1920, 1920).mean(-1).transpose(1, 2)
        return x.expand(-1, -1, self.dim).contiguous()


@pytest.fixture
def predictor():
    torch.set_num_threads(1)
    torch.manual_seed(13)
    result = Predictor.__new__(Predictor)
    result.device = torch.device("cpu")
    result._active = False
    result.model = VAP(TinyEncoder(), TransformerStereo(
        dim=16, self_layers=2, cross_layers=2, num_heads=4, dff_k=2,
        dropout=0.0, positional_encoding="alibi", attention_scale="head",
        normalize_cross_source=True), frame_hz=12.5,
        bin_times=[0.24, 0.48, 0.72, 0.96], normalize_vad=True).eval()
    return result


@pytest.fixture(scope="session")
def real_predictor():
    checkpoint = os.environ.get("MIMI_VAP_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set MIMI_VAP_CHECKPOINT to run real-weight integration tests")
    torch.set_num_threads(2)
    return Predictor(checkpoint, "cpu")
