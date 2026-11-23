"""Portable CPU-only reader and bounded-memory archive integrity checker."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile


def safe_relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or any(p in ("", ".", "..") for p in name.split("/")):
        raise ValueError("Unsafe archive path")
    if "\\" in name or "\x00" in name or ":" in name:
        raise ValueError("Unsafe archive path")
    return path


def validate_episode(episode):
    if episode["format"] != "robofate-prefix-success-episode-v1":
        raise ValueError("Unsupported episode format")
    actions, observations = episode["actions"], episode["observations"]
    t1 = episode["teacher_start"]
    if not 0 < t1 < len(actions) or len(observations) != len(actions) + 1:
        raise ValueError("Invalid trajectory/boundary length")
    for i, action in enumerate(actions):
        role = "prefix" if i < t1 else "teacher"
        if action["index"] != i or action["role"] != role or action["supervise"] != (i >= t1):
            raise ValueError("Invalid action index or supervision mask")
        values = action["applied_joint_command"]
        if len(values) != 14 or not all(math.isfinite(v) for v in values):
            raise ValueError("Invalid action vector")
        if not all(0 <= values[j] <= 1 for j in (6, 13)):
            raise ValueError("Invalid gripper command")
    rows = observations + [episode["original_prefix_terminal_observation"]]
    for i, obs in enumerate(observations):
        if obs["applied_actions"] != i:
            raise ValueError("Invalid observation sequence")
    for obs in rows:
        if len(obs["state"]) != 14 or not all(math.isfinite(v) for v in obs["state"]):
            raise ValueError("Invalid state vector")
        if set(obs["images"]) != {"head", "left_wrist", "right_wrist"}:
            raise ValueError("Missing camera")
        for name in list(obs["images"].values()) + [obs["native_observation"]]:
            safe_relative(name)
    return episode


def verify_archive(archive, expected_sha256=None, extract_to=None):
    """Hash every member in one pass; extract only regular files to a new directory."""
    archive = Path(archive)
    if expected_sha256:
        digest = hashlib.sha256()
        with archive.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("Archive SHA256 mismatch")
    destination = Path(extract_to) if extract_to else None
    if destination and destination.exists():
        raise ValueError("Extraction destination must not exist")
    stage = None
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".robofate-", dir=str(destination.parent)))
    actual, captured, root = {}, {}, None
    try:
        with tarfile.open(archive, "r|gz") as stream:
            for member in stream:
                path = safe_relative(member.name)
                if not member.isfile() or len(path.parts) < 2:
                    raise ValueError("Only nested regular files are permitted")
                if root is None:
                    root = path.parts[0]
                if path.parts[0] != root or member.size > 64 * 1024 * 1024:
                    raise ValueError("Unexpected root or oversized member")
                relative = "/".join(path.parts[1:])
                if relative in actual:
                    raise ValueError("Duplicate member")
                output = None
                if stage:
                    target = stage.joinpath(*path.parts[1:])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    output = target.open("xb")
                digest, size = hashlib.sha256(), 0
                keep = relative in ("episode.json", "checksums.json", "provenance.json")
                buffer = io.BytesIO() if keep else None
                try:
                    with stream.extractfile(member) as source:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            digest.update(chunk)
                            size += len(chunk)
                            if output:
                                output.write(chunk)
                            if buffer:
                                buffer.write(chunk)
                finally:
                    if output:
                        output.close()
                if size != member.size:
                    raise ValueError("Truncated member")
                actual[relative] = {"sha256": digest.hexdigest(), "bytes": size}
                if keep:
                    captured[relative] = json.loads(buffer.getvalue())
        checksums = captured["checksums.json"]
        if checksums["algorithm"] != "sha256":
            raise ValueError("Unsupported checksum algorithm")
        del actual["checksums.json"]
        if actual != checksums["files"]:
            raise ValueError("Archive member inventory/checksum mismatch")
        episode = validate_episode(captured["episode.json"])
        if episode["episode_key"] != root:
            raise ValueError("Episode/root mismatch")
        for obs in episode["observations"] + [episode["original_prefix_terminal_observation"]]:
            for name in list(obs["images"].values()) + [obs["native_observation"]]:
                if name not in actual:
                    raise ValueError("Observation refers to a missing member")
        if stage:
            stage.rename(destination)
            stage = None
        return {"episode_key": root, "members": len(actual) + 1,
                "actions": len(episode["actions"]),
                "teacher_actions": sum(a["supervise"] for a in episode["actions"])}
    finally:
        if stage:
            shutil.rmtree(stage)


def load_episode(directory):
    return validate_episode(json.loads((Path(directory) / "episode.json").read_text()))


def iter_transitions(directory, teacher_only=True):
    """Yield RGB paths, state and real held-action labels; no model execution."""
    root = Path(directory)
    episode = load_episode(root)
    for action in episode["actions"]:
        if teacher_only and not action["supervise"]:
            continue
        index = action["index"]
        obs = episode["observations"][index]
        yield {
            "instruction": episode["instruction"], "state": obs["state"],
            "images": {name: root / value for name, value in obs["images"].items()},
            "action": action["applied_joint_command"], "supervise": action["supervise"],
            "index": index, "teacher_start": episode["teacher_start"],
            "next_state": episode["observations"][index + 1]["state"],
        }


def action_chunk(episode, start, horizon):
    """Return causal actions with an explicit valid/teacher loss mask; no fake padding labels."""
    if start < 0 or horizon < 1 or start >= len(episode["actions"]):
        raise ValueError("Invalid chunk bounds")
    rows = episode["actions"][start:start + horizon]
    return [r["applied_joint_command"] for r in rows], [r["supervise"] for r in rows]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--sha256")
    parser.add_argument("--extract-to")
    args = parser.parse_args()
    print(json.dumps(verify_archive(args.archive, args.sha256, args.extract_to)))
