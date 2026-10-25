import argparse
import asyncio
import os
from pathlib import Path

import torch

from .inference import Predictor, predict_file
from .weights import resolve_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Mimi VAP: per-frame predictions with bounded KV caching")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="Download and verify the default checkpoint")
    for name in ("predict", "serve"):
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", type=Path, help="Use local weights instead of downloading")
        p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
        p.add_argument("--context-seconds", type=float, default=20.0)
        p.add_argument("--aggregation", choices=["relative", "neutral"], default="relative")
        p.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
        if name == "predict":
            p.add_argument("audio", type=Path)
            p.add_argument("-o", "--output", type=Path, help="Default: input filename with .csv extension")
        else:
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        if args.command == "download":
            print(resolve_checkpoint())
            return
        if args.threads < 1:
            parser.error("--threads must be positive")
        torch.set_num_threads(args.threads)
        predictor = Predictor(resolve_checkpoint(args.checkpoint), args.device)
        if args.command == "predict":
            output = args.output or args.audio.with_suffix(".csv")
            count = predict_file(predictor, args.audio, output, args.context_seconds, args.aggregation)
            print(f"Wrote {count} frames to {output}")
        else:
            from .server import StreamingServer
            asyncio.run(StreamingServer(predictor, args.context_seconds, args.aggregation).run(args.host, args.port))
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    except KeyboardInterrupt:
        pass
