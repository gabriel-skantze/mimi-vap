"""Binary PCM WebSocket protocol with one exclusive model stream per process."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging

import numpy as np
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from .inference import FRAME_HZ, FRAME_SAMPLES, SAMPLE_RATE

logger = logging.getLogger(__name__)


class StreamingServer:
    def __init__(self, predictor, context_seconds=20.0, aggregation="relative"):
        self.predictor = predictor
        self.context_seconds = context_seconds
        self.aggregation = aggregation
        self.busy = False
        # Native Mimi streaming state is entered and used on this same thread.
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def _call(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, fn, *args)

    async def handler(self, socket):
        if self.busy:
            await socket.send(json.dumps({"type": "error", "message": "Server busy: one stream per process"}))
            await socket.close(code=1013, reason="One active stream per process")
            return
        self.busy = True
        stream = None
        try:
            stream = await self._call(self.predictor.stream, self.context_seconds, self.aggregation)
            await socket.send(json.dumps({"type": "ready", "sample_rate": SAMPLE_RATE,
                "channels": 2, "format": "float32_le", "frame_hz": FRAME_HZ,
                "frame_samples": FRAME_SAMPLES, "aggregation": self.aggregation,
                "context_seconds": stream.state.context_frames / FRAME_HZ}))
            async for message in socket:
                if isinstance(message, bytes):
                    if len(message) % 8:
                        raise ValueError("Binary audio must be interleaved stereo float32_le (8 bytes/sample pair)")
                    audio = np.frombuffer(message, dtype="<f4").reshape(-1, 2).copy()
                    rows = await self._call(stream.push, audio)
                    await socket.send(json.dumps({"type": "predictions", "frames": rows}))
                else:
                    control = json.loads(message)
                    if not isinstance(control, dict):
                        raise ValueError("Control message must be a JSON object")
                    if control.get("type") == "flush":
                        rows = await self._call(stream.flush)
                        await socket.send(json.dumps({"type": "predictions", "frames": rows}))
                        await socket.send(json.dumps({"type": "done", "frames": stream.frame}))
                    elif control.get("type") == "reset":
                        await self._call(stream.reset)
                        await socket.send(json.dumps({"type": "reset"}))
                    else:
                        raise ValueError("Unknown control type; use flush or reset")
        except ConnectionClosed:
            pass
        except (ValueError, RuntimeError) as error:
            try:
                await socket.send(json.dumps({"type": "error", "message": str(error)}))
                await socket.close(code=1008, reason="Invalid input or inference failure")
            except ConnectionClosed:
                pass
        except Exception:
            logger.exception("WebSocket inference failed")
            await socket.close(code=1011, reason="Inference failed")
        finally:
            if stream is not None:
                await self._call(stream.close)
            self.busy = False

    async def run(self, host="127.0.0.1", port=8765):
        try:
            async with serve(self.handler, host, port, max_size=1024 * 1024,
                             max_queue=4, compression=None) as server:
                print(f"Listening on ws://{host}:{port}", flush=True)
                await server.serve_forever()
        finally:
            self.executor.shutdown(wait=True)
