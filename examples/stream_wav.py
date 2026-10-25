"""Stream a WAV to the server and save its predictions to CSV.

Usage: python examples/stream_wav.py conversation.wav -o predictions.csv
"""

import argparse
import asyncio
import csv
import json

from websockets.asyncio.client import connect

from mimi_vap.inference import CSV_FIELDS, audio_blocks


async def run(args):
    async with connect(args.url, max_size=4 * 1024 * 1024) as socket:
        ready = json.loads(await socket.recv())
        if ready["type"] != "ready":
            raise RuntimeError(ready)
        with open(args.output, "w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for block in audio_blocks(args.audio):
                # A request/response exchange supplies backpressure for long WAVs.
                await socket.send(block.astype("<f4").tobytes())
                message = json.loads(await socket.recv())
                if message["type"] != "predictions":
                    raise RuntimeError(message)
                writer.writerows(message["frames"])
            await socket.send(json.dumps({"type": "flush"}))
            message = json.loads(await socket.recv())
            if message["type"] != "predictions":
                raise RuntimeError(message)
            writer.writerows(message["frames"])
            done = json.loads(await socket.recv())
            if done["type"] != "done":
                raise RuntimeError(done)
            print(f"Wrote {done['frames']} frames to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("-o", "--output", default="predictions.csv")
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    asyncio.run(run(parser.parse_args()))
