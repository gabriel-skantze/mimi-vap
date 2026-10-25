import csv

import numpy as np
import pytest
import soundfile as sf
import soxr
import torch

from mimi_vap import predict_file
from mimi_vap.inference import audio_blocks, FRAME_SAMPLES


@pytest.mark.parametrize("samples", [0, 1, 1919, 1920, 1921, 3840, 4001])
def test_partial_frames_and_timestamps(predictor, tmp_path, samples):
    audio = tmp_path / "input.wav"
    output = tmp_path / "output.csv"
    sf.write(audio, np.zeros((samples, 2)), 24000, subtype="FLOAT")
    count = predict_file(predictor, audio, output)
    rows = list(csv.DictReader(output.open()))
    assert count == len(rows) == (samples + 1919) // 1920
    for i, row in enumerate(rows):
        assert int(row["frame"]) == i
        assert float(row["time_s"]) == i * 0.08
        assert float(row["end_s"]) == pytest.approx(min((i + 1) * 0.08, samples / 24000))
        for key in ("vad_0", "vad_1", "p_now_0", "p_now_1", "p_fut_0", "p_fut_1"):
            assert 0 <= float(row[key]) <= 1


def test_chunk_invariance_reset_and_bounded_cache(predictor):
    audio = np.random.default_rng(4).normal(0, 0.01, (FRAME_SAMPLES * 12 + 7, 2)).astype("float32")
    with predictor.stream(context_seconds=0.32) as stream:
        expected = stream.push(audio) + stream.flush()
        layers = stream.state.transformer.channel_layers + stream.state.transformer.stereo_layers
        for layer in layers:
            assert layer.self_attention.length <= 4
            assert layer.self_attention.next_position == 13
            if layer.cross_attention:
                assert layer.cross_attention.length <= 4
        with pytest.raises(RuntimeError, match="active"):
            predictor.stream()
        stream.reset()
        actual = []
        for block in np.array_split(audio, 37):
            actual += stream.push(block)
        actual += stream.flush()
        assert expected == actual
        with pytest.raises(RuntimeError, match="flushed"):
            stream.push(audio[:1])
    assert not predictor._active


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 44100, 48000])
def test_stateful_resampling_and_mono_mapping(tmp_path, rate):
    audio = np.random.default_rng(2).normal(0, 0.05, rate * 2 + 17).astype("float32")
    path = tmp_path / "mono.wav"
    sf.write(path, audio, rate, subtype="FLOAT")
    actual = np.concatenate(list(audio_blocks(path)))
    expected = soxr.resample(audio, rate, 24000)
    np.testing.assert_allclose(actual[:len(expected), 0], expected, atol=1e-6)
    assert np.count_nonzero(actual[:, 1]) == 0
    assert len(actual) == (len(audio) * 24000 + rate - 1) // rate


def test_reject_bad_audio_and_preserve_input(predictor, tmp_path):
    with predictor.stream() as stream:
        with pytest.raises(ValueError):
            stream.push(np.zeros((5, 1)))
        with pytest.raises(ValueError):
            stream.push(np.full((5, 2), np.nan))
    path = tmp_path / "input.wav"
    sf.write(path, np.zeros((20, 2)), 24000)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="different"):
        predict_file(predictor, path, path)
    assert path.read_bytes() == original


def test_real_checkpoint_streaming_and_reset(real_predictor):
    assert real_predictor.model.objective.bin_frames == [3, 6, 9, 12]
    audio = np.random.default_rng(8).normal(0, 0.005, (1920 * 5 + 71, 2)).astype("float32")
    with real_predictor.stream(context_seconds=0.24) as stream:
        expected = stream.push(audio) + stream.flush()
        stream.reset()
        actual = []
        for chunk in np.array_split(audio, 19):
            actual += stream.push(chunk)
        actual += stream.flush()
    assert actual == expected


@pytest.mark.parametrize("positional_encoding", ["alibi", "rope"])
def test_cached_transformer_matches_causal_batch(positional_encoding):
    from mimi_vap._core.modules.modules import TransformerStereo
    torch.manual_seed(5)
    model = TransformerStereo(dim=16, self_layers=2, cross_layers=2,
        num_heads=4, dff_k=2, dropout=0.0, positional_encoding=positional_encoding).eval()
    left, right = torch.randn(2, 17, 16), torch.randn(2, 17, 16)
    with torch.inference_mode():
        expected = model(left, right)["x"]
        cache = model.init_streaming_cache(2, 32)
        outputs = [model.forward_incremental(left[:, a:b], right[:, a:b], cache)["x"]
                   for a, b in [(0, 1), (1, 5), (5, 9), (9, 17)]]
    torch.testing.assert_close(torch.cat(outputs, 1), expected, rtol=1e-5, atol=1e-6)
