"""Synthetic recording fixtures, never scientific outcomes or real approvals."""
import hashlib
import io
import json
from pathlib import Path
import socket
import struct
import tarfile
import tempfile
import unittest
import zipfile
import zlib

import dataset_reader
import export_training
import inventory_episodes
from tools.make_export_plan import make_plan


def synthetic_png():
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 640, 480, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\x00" + b"\x00" * 640 * 3) * 480))
            + chunk(b"IEND", b""))


def write_run(root, steps, teacher=False):
    root.mkdir()
    (root / "dense").mkdir()
    command = {"left_arm_joint_state": [0.0] * 6, "right_arm_joint_state": [0.0] * 6,
               "left_ee_joint_state": [0.5], "right_ee_joint_state": [0.5]}
    native, controls = [], []
    for index in range(steps):
        native.append({"index": index, "actual_control_steps": (index + 1) * 10,
                       "action": command})
        for tick in range(10):
            controls.append({
                "index": index * 10 + tick + 1, "macro_index": index,
                "native_action_count": index + 1,
                "executed_actuator_targets": [
                    {"method": "set_joint_position_target", "body_index": body,
                     "kwargs": {"joint_ids": list(range(6))}, "values": [[0.0] * 6]}
                    for body in (0, 1)],
            })
    for name, rows in (("native_actions.jsonl", native), ("dense/control_steps.jsonl", controls)):
        (root / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "result.json").write_text(json.dumps({"native_success": teacher,
                                                 "synthetic_fixture_only": True}))
    image = synthetic_png()
    for index in range(steps + 1):
        directory = root / ("initial" if index == 0 else "macro_obs_%06d" % index)
        directory.mkdir()
        state = dict(command)
        state["left_arm_joint_state"] = [9.0 if teacher else 1.0] * 6
        (directory / "observation.json").write_text(json.dumps({
            "state": state, "instruction": "SYNTHETIC CPU FIXTURE; not a collected task",
        }))
        with zipfile.ZipFile(directory / "arrays.npz", "w"):
            pass
        for camera in export_training.SOURCE_CAMERAS:
            (directory / (camera + ".png")).write_bytes(image)


class ExportRoundtripTests(unittest.TestCase):
    def test_inventory_export_verify_and_causal_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_run(root / "baseline", 3)
            write_run(root / "teacher", 2, teacher=True)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"synthetic_fixture_only": True}))
            request = {
                "expected_hostname": socket.gethostname(), "release_root": str(root / "inventory"),
                "episodes": [{
                    "episode_key": "synthetic__roundtrip", "source_id": "synthetic-only",
                    "task": "synthetic", "t1_native": 1, "native_total_budget": 3,
                    "source_manifest": str(manifest),
                    "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    "runs": {"dm_failure": str(root / "baseline"),
                             "teacher_success": str(root / "teacher")},
                }],
            }
            inventory = inventory_episodes.main(request)
            self.assertEqual(inventory["errors"], 0)
            plan = make_plan(inventory, root / "export")
            output = Path(plan["output"])
            output.mkdir()
            receipt = export_training.export_episode(plan["episodes"][0], output)
            self.assertEqual(receipt["actions"], 2)
            self.assertEqual(receipt["teacher_actions"], 1)
            self.assertEqual(receipt["image_count"], 12)
            archive = Path(receipt["path"])
            checked = dataset_reader.verify_archive(archive, receipt["sha256"], root / "decoded")
            self.assertEqual(checked["actions"], 2)
            episode = dataset_reader.load_episode(root / "decoded")
            self.assertEqual(episode["original_prefix_terminal_observation"]["state"][0], 1.0)
            self.assertEqual(episode["observations"][1]["state"][0], 9.0)
            self.assertEqual([a["supervise"] for a in episode["actions"]], [False, True])
            with tarfile.open(archive) as stream:
                names = stream.getnames()
            self.assertFalse(any("teacher/observations/000000" in name for name in names))
            self.assertFalse(any("prefix/observations/000002" in name for name in names))
            self.assertEqual(export_training.export_episode(plan["episodes"][0], output), receipt)

    def test_failed_inventory_cannot_make_export_plan(self):
        with self.assertRaisesRegex(ValueError, "successful"):
            make_plan({"errors": 1, "episodes": [{}]}, "/synthetic/export")

    def test_relative_export_directory_rejected(self):
        with self.assertRaisesRegex(ValueError, "absolute"):
            make_plan({"errors": 0, "episodes": [{}]}, "relative/export")


if __name__ == "__main__":
    unittest.main()
