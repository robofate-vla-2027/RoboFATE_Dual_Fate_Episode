"""Synthetic CPU fixtures only: no network, robot, model, simulator, or GPU."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from dataset_reader import action_chunk, iter_transitions, safe_relative, validate_episode, verify_archive


def fixture():
    obs = lambda i: {"applied_actions": i, "state": [0.0] * 14,
                     "images": {"head": "head.png", "left_wrist": "left.png",
                                "right_wrist": "right.png"},
                     "native_observation": "observation.json"}
    return {
        "format": "robofate-prefix-success-episode-v1", "episode_key": "synthetic__abc",
        "instruction": "Synthetic unit-test fixture; not a scientific result.",
        "teacher_start": 1,
        "actions": [{"index": i, "role": "prefix" if i == 0 else "teacher",
                     "supervise": i != 0, "applied_joint_command": [0.0] * 14}
                    for i in range(3)],
        "observations": [obs(i) for i in range(4)],
        "original_prefix_terminal_observation": obs(1),
    }


def make_archive(path, corrupt=False, extra=None):
    members = {"episode.json": json.dumps(fixture()).encode(),
               "provenance.json": b'{"synthetic":true}',
               "head.png": b"synthetic", "left.png": b"synthetic", "right.png": b"synthetic",
               "observation.json": b"{}"}
    checksums = {k: {"sha256": hashlib.sha256(v).hexdigest(), "bytes": len(v)}
                 for k, v in members.items()}
    members["checksums.json"] = json.dumps({"algorithm": "sha256", "files": checksums}).encode()
    if corrupt:
        members["head.png"] = b"corrupted"
    with tarfile.open(path, "w:gz") as output:
        for key, value in members.items():
            entry = tarfile.TarInfo("synthetic__abc/" + key)
            entry.size = len(value)
            output.addfile(entry, io.BytesIO(value))
        if extra:
            output.addfile(extra)


class ReaderTests(unittest.TestCase):
    def test_valid_episode(self):
        self.assertEqual(validate_episode(fixture())["teacher_start"], 1)

    def test_bad_paths(self):
        for path in ("", "/a", "../a", "a/../b", "./a", "a//b", "a\\b", "a/", "C:/a", "a:stream"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                safe_relative(path)

    def test_supervision(self):
        e = fixture()
        e["actions"][0]["supervise"] = True
        with self.assertRaises(ValueError):
            validate_episode(e)

    def test_finite_state(self):
        e = fixture()
        e["observations"][0]["state"][0] = float("nan")
        with self.assertRaises(ValueError):
            validate_episode(e)

    def test_length(self):
        e = fixture()
        e["observations"].pop()
        with self.assertRaises(ValueError):
            validate_episode(e)

    def test_gripper(self):
        e = fixture()
        e["actions"][0]["applied_joint_command"][6] = 2
        with self.assertRaises(ValueError):
            validate_episode(e)

    def test_chunk_mask_and_no_fabricated_padding(self):
        values, mask = action_chunk(fixture(), 0, 10)
        self.assertEqual(len(values), 3)
        self.assertEqual(mask, [False, True, True])

    def test_verify_extract_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.tar.gz"
            make_archive(path)
            target = Path(tmp) / "extracted"
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            result = verify_archive(path, sha, target)
            self.assertEqual(result["teacher_actions"], 2)
            self.assertEqual(len(list(iter_transitions(target))), 2)
            self.assertEqual(len(list(iter_transitions(target, False))), 3)
            with self.assertRaises(ValueError):
                verify_archive(path, sha, target)

    def test_integrity_failure_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.tar.gz"
            target = Path(tmp) / "extracted"
            make_archive(path, corrupt=True)
            with self.assertRaises(ValueError):
                verify_archive(path, extract_to=target)
            self.assertFalse(target.exists())
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)

    def test_archive_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.tar.gz"
            make_archive(path)
            with self.assertRaises(ValueError):
                verify_archive(path, "0" * 64)

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.tar.gz"
            link = tarfile.TarInfo("synthetic__abc/link")
            link.type, link.linkname = tarfile.SYMTYPE, "/tmp"
            make_archive(path, extra=link)
            with self.assertRaises(ValueError):
                verify_archive(path)

    def test_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.tar.gz"
            make_archive(path, extra=tarfile.TarInfo("synthetic__abc/head.png"))
            with self.assertRaises(ValueError):
                verify_archive(path)


if __name__ == "__main__":
    unittest.main()
