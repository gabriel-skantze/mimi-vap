"""Download a pinned GitHub release asset and verify its digest."""

import hashlib
import json
import os
from importlib.resources import files
from pathlib import Path
import subprocess
import tempfile
import urllib.error
import urllib.request


def manifest():
    return json.loads(files("mimi_vap").joinpath("model.json").read_text())


def checksum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_checkpoint(path=None):
    if path is not None:
        result = Path(path).expanduser()
        if not result.is_file():
            raise FileNotFoundError(result)
        return result
    config = manifest()
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "mimi-vap"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / config["filename"]
    if target.exists() and checksum(target) == config["sha256"]:
        return target
    with tempfile.TemporaryDirectory(dir=cache) as directory:
        part = Path(directory) / config["filename"]
        try:
            # Also works without GitHub credentials when the repository is public.
            with urllib.request.urlopen(config["url"], timeout=60) as response, part.open("wb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
        except urllib.error.HTTPError as error:
            if error.code not in (401, 403, 404):
                raise
            try:
                subprocess.run(["gh", "release", "download", config["release"], "--repo",
                                config["repository"], "--pattern", config["filename"],
                                "--dir", directory, "--clobber"], check=True)
            except (FileNotFoundError, subprocess.CalledProcessError) as cause:
                raise RuntimeError("Model download requires access to the GitHub repository. "
                                   "Run `gh auth login`, or supply --checkpoint /path/to/model.pt.") from cause
        if checksum(part) != config["sha256"]:
            raise RuntimeError("Checkpoint SHA-256 mismatch; download was not installed")
        os.replace(part, target)
    return target
