"""Verify frozen source bundles and unpack safely without running any code."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from dataset_reader import safe_relative


def unpack(output):
    here = Path(__file__).resolve().parent
    destination = Path(output)
    if destination.exists():
        raise ValueError("Output must be a new directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".source-", dir=str(destination.parent)))
    try:
        for bundle in json.loads((here / "SOURCE_BUNDLES.json").read_text())["bundles"]:
            archive = here / bundle["file"]
            digest = hashlib.sha256()
            with archive.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if archive.stat().st_size != bundle["bytes"] or digest.hexdigest() != bundle["sha256"]:
                raise ValueError("Source bundle integrity mismatch")
            with tarfile.open(archive, "r|gz") as source:
                for member in source:
                    relative = safe_relative(member.name)
                    if not member.isfile() or member.size > 16 * 1024 * 1024:
                        raise ValueError("Unexpected source member")
                    target = stage.joinpath(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as src, target.open("xb") as dst:
                        shutil.copyfileobj(src, dst, 1024 * 1024)
                    target.chmod(0o755 if member.mode & 0o111 else 0o644)
        stage.rename(destination)
        stage = None
    finally:
        if stage:
            shutil.rmtree(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    unpack(args.output)
    print("Verified source extracted; no source code executed.")
