import asyncio
import csv
import json

import numpy as np
import pytest
import soundfile as sf
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from mimi_vap import predict_file
from mimi_vap.server import StreamingServer


async def receive(socket, kind):
    result = json.loads(await asyncio.wait_for(socket.recv(), 10))
    assert result["type"] == kind, result
    return result


def test_protocol_parity_busy_reset_disconnect(predictor, tmp_path):
    audio = np.random.default_rng(11).normal(0, 0.02, (1920 * 8 + 13, 2)).astype("float32")
    path, target = tmp_path / "input.wav", tmp_path / "output.csv"
    sf.write(path, audio, 24000, subtype="FLOAT")
    predict_file(predictor, path, target, context_seconds=0.24)
    expected = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(target.open())]

    async def scenario():
        service = StreamingServer(predictor, context_seconds=0.24)
        try:
            async with serve(service.handler, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                url = f"ws://127.0.0.1:{port}"
                async with connect(url) as socket:
                    ready = await receive(socket, "ready")
                    assert ready["format"] == "float32_le"
                    async with connect(url) as second:
                        await receive(second, "error")
                    for _ in range(2):
                        actual = []
                        for block in np.array_split(audio, 23):
                            await socket.send(block.astype("<f4").tobytes())
                            actual += (await receive(socket, "predictions"))["frames"]
                        await socket.send('{"type":"flush"}')
                        actual += (await receive(socket, "predictions"))["frames"]
                        assert (await receive(socket, "done"))["frames"] == len(expected)
                        assert actual == expected
                        await socket.send('{"type":"reset"}')
                        await receive(socket, "reset")
                    await socket.send(b"invalid")
                    await receive(socket, "error")
                    await socket.wait_closed()
                # Wait for worker cleanup before the next conversation.
                for _ in range(100):
                    if not service.busy:
                        break
                    await asyncio.sleep(0.01)
                async with connect(url) as socket:
                    await receive(socket, "ready")
                    await socket.send(audio[:1].tobytes())
                    assert (await receive(socket, "predictions"))["frames"] == []
                for _ in range(100):
                    if not service.busy:
                        break
                    await asyncio.sleep(0.01)
                assert not predictor._active
        finally:
            service.executor.shutdown(wait=True)
    asyncio.run(scenario())


def test_real_websocket_matches_file(real_predictor, tmp_path):
    # Repeat the complete protocol test with native Mimi and the selected weights.
    test_protocol_parity_busy_reset_disconnect(real_predictor, tmp_path)
