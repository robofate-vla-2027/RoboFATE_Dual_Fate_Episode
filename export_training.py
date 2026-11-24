"""Export immutable prefix/success records without simulator or model imports."""
import ast
import base64
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import struct
import tarfile
import time
import zipfile
import zlib

CAMERAS = ("head", "left_wrist", "right_wrist")
SOURCE_CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def json_bytes(value):
    return (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()


def vector(meta, archive):
    if isinstance(meta, list):
        values = meta
    else:
        with archive.open(meta["npz_key"] + ".npy") as stream:
            prefix = stream.read(8)
            if prefix[:6] != b"\x93NUMPY":
                raise ValueError("Invalid NPY header")
            width = 2 if prefix[6] == 1 else 4
            length = int.from_bytes(stream.read(width), "little")
            if length > 65536:
                raise ValueError("Oversized NPY header")
            header = ast.literal_eval(stream.read(length).decode("latin1").strip())
            if len(header["shape"]) != 1 or header["descr"] not in ("<f4", "<f8"):
                raise ValueError("Expected a one-dimensional floating-point vector")
            count = header["shape"][0]
            data = stream.read()
            fmt = "<" + ("f" if header["descr"] == "<f4" else "d") * count
            if len(data) != struct.calcsize(fmt):
                raise ValueError("NPY length mismatch")
            values = list(struct.unpack(fmt, data))
    if not all(isinstance(x, (float, int)) and math.isfinite(x) for x in values):
        raise ValueError("Invalid vector values")
    return values


def validate_png(data):
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Invalid PNG signature")
    offset, compressed, header, ended = 8, [], None, False
    while offset < len(data):
        size = struct.unpack(">I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + size]
        end = offset + 8 + size
        if len(payload) != size or end + 4 > len(data):
            raise ValueError("Truncated PNG chunk")
        crc = struct.unpack(">I", data[end:end + 4])[0]
        if zlib.crc32(kind + payload) & 0xffffffff != crc:
            raise ValueError("PNG chunk CRC mismatch")
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", payload)
        elif kind == b"IDAT":
            compressed.append(payload)
        elif kind == b"IEND":
            ended = True
        offset = end + 4
    if not ended or header is None or header[:3] != (640, 480, 8):
        raise ValueError("Unexpected PNG dimensions or bit depth")
    if header[3] not in (2, 6) or header[4:] != (0, 0, 0):
        raise ValueError("Unexpected PNG color/compression/interlace")
    channels = 3 if header[3] == 2 else 4
    scanlines = zlib.decompress(b"".join(compressed))
    stride = 1 + header[0] * channels
    if len(scanlines) != stride * header[1]:
        raise ValueError("PNG decompressed length mismatch")
    if any(scanlines[row * stride] > 4 for row in range(header[1])):
        raise ValueError("Invalid PNG scanline filter")


def arm_targets(row):
    arms = {}
    for target in row["executed_actuator_targets"]:
        if target["method"] == "set_joint_position_target" and target.get(
                "kwargs", {}).get("joint_ids") == list(range(6)):
            values = target["values"]
            if len(values) != 1 or len(values[0]) != 6:
                raise ValueError("Invalid executed arm-target shape")
            if target["body_index"] in arms:
                raise ValueError("Repeated arm target")
            arms[target["body_index"]] = values[0]
    if set(arms) != {0, 1}:
        raise ValueError("Missing executed arm target")
    return arms[0], arms[1]


class HashWriter:
    def __init__(self, stream):
        self.stream, self.digest, self.bytes = stream, hashlib.sha256(), 0

    def write(self, data):
        self.digest.update(data)
        self.bytes += len(data)
        return self.stream.write(data)

    def flush(self):
        self.stream.flush()


def export_episode(episode, output):
    key, t1 = episode["episode_key"], episode["t1_native"]
    if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", key):
        raise ValueError("Unsafe episode key")
    final = output / (key + ".tar.gz")
    ready = output / (key + ".ready.json")
    if ready.exists():
        receipt = json.loads(ready.read_bytes())
        if not final.is_file() or final.stat().st_size != receipt["bytes"]:
            raise ValueError("Previously exported archive changed")
        digest = hashlib.sha256()
        with final.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != receipt["sha256"]:
            raise ValueError("Previously exported archive checksum mismatch")
        for field in ("episode_key", "source_id", "task", "source_manifest_sha256", "t1_native"):
            if receipt[field] != episode[field]:
                raise ValueError("Previously exported episode identity changed")
        if hashlib.sha256(Path(episode["source_manifest"]).read_bytes()).hexdigest() != episode["source_manifest_sha256"]:
            raise ValueError("Source manifest changed")
        return receipt
    if final.exists():
        raise ValueError("Archive exists without a completion receipt")
    inventory = json.loads(Path(episode["inventory_path"]).read_bytes())
    files = {f["source"]: f for f in inventory["files"]}
    source_manifest = Path(episode["source_manifest"]).read_bytes()
    if hashlib.sha256(source_manifest).hexdigest() != episode["source_manifest_sha256"]:
        raise ValueError("Source manifest changed")
    manifest = json.loads(source_manifest)
    partial = output / (key + ".partial." + str(os.getpid()))
    checksums = {}
    observations, actions, segments = [], [], []
    image_count, source_bytes = 0, 0
    control_buffer = io.BytesIO()
    controls_writer = gzip.GzipFile(fileobj=control_buffer, mode="wb", compresslevel=6, mtime=0)
    instruction = None
    prefix_boundary = None
    clipped = 0

    with partial.open("xb") as raw:
        hashed = HashWriter(raw)
        with gzip.GzipFile(fileobj=hashed, mode="wb", compresslevel=1, mtime=0, filename="") as gz:
            with tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                def add_bytes(relative, data):
                    info = tarfile.TarInfo(key + "/" + relative)
                    info.size, info.mode, info.mtime = len(data), 0o644, 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    tar.addfile(info, io.BytesIO(data))
                    checksums[relative] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

                def add_source(relative, path):
                    nonlocal source_bytes, image_count
                    meta = files[str(path)]
                    stat = path.stat()
                    if stat.st_size != meta["bytes"] or stat.st_mtime_ns != meta["mtime_ns"]:
                        raise ValueError("Inventoried source file changed: " + str(path))
                    data = path.read_bytes()
                    if path.suffix == ".png":
                        validate_png(data)
                        image_count += 1
                    source_bytes += len(data)
                    add_bytes(relative, data)
                    return data

                for source_role, role in (("dm_failure", "prefix"), ("teacher_success", "teacher")):
                    root = Path(episode["branches"][source_role]["resolved_path"])
                    with (root / "native_actions.jsonl").open() as stream:
                        native = [json.loads(line) for line in stream if line.strip()]
                    if [r["index"] for r in native] != list(range(len(native))):
                        raise ValueError("Noncontiguous native command records")
                    start, stop = (0, t1) if role == "prefix" else (t1, len(native))
                    if role == "teacher":
                        result = json.loads((root / "result.json").read_bytes())
                        if result.get("native_success") is not True or stop <= start:
                            raise ValueError("Successful teacher suffix is required")
                    segments.append({"role": role, "action_start": start, "action_stop": stop,
                                     "observation_start": start, "observation_stop_inclusive": stop})
                    for n in range(start, stop + 1):
                        if n == 0:
                            directory = root / "initial"
                        else:
                            directory = root / ("macro_obs_%06d" % n)
                        if not directory.is_dir():
                            candidates = [root / ("%s_%06d" % (name, n))
                                          for name in ("native_obs", "observation")]
                            directory = next((p for p in candidates if p.is_dir()), directory)
                        relative = "%s/observations/%06d" % (role, n)
                        obs_data = add_source(relative + "/observation.json", directory / "observation.json")
                        obs = json.loads(obs_data)
                        npz_data = add_source(relative + "/arrays.npz", directory / "arrays.npz")
                        with zipfile.ZipFile(io.BytesIO(npz_data)) as arrays:
                            left = vector(obs["state"]["left_arm_joint_state"], arrays)
                            right = vector(obs["state"]["right_arm_joint_state"], arrays)
                            left_grip = vector(obs["state"]["left_ee_joint_state"], arrays)
                            right_grip = vector(obs["state"]["right_ee_joint_state"], arrays)
                        state = left + left_grip + right + right_grip
                        if len(state) != 14 or len(left) != 6 or len(right) != 6:
                            raise ValueError("Invalid native robot-state shape")
                        current_instruction = obs.get("instruction")
                        if not current_instruction or (instruction is not None and instruction != current_instruction):
                            raise ValueError("Task instruction missing or changes across the selected trajectory")
                        instruction = current_instruction
                        images = {}
                        for camera, source_camera in zip(CAMERAS, SOURCE_CAMERAS):
                            filename = source_camera + ".png"
                            add_source(relative + "/" + filename, directory / filename)
                            images[camera] = relative + "/" + filename
                        row = {"applied_actions": n, "role": role, "state": state, "images": images,
                               "native_observation": relative + "/observation.json"}
                        if role == "prefix" and n == t1:
                            prefix_boundary = row
                        else:
                            observations.append(row)

                    selected_native = {r["index"]: r for r in native if start <= r["index"] < stop}
                    current, group = None, []

                    def finish():
                        nonlocal clipped
                        if current is None:
                            return
                        row = selected_native[current]
                        if len(group) != 10 or group[-1]["index"] != row["actual_control_steps"]:
                            raise ValueError("Native/control boundary mismatch")
                        indices = [r["index"] for r in group]
                        if indices != list(range(indices[0], indices[-1] + 1)):
                            raise ValueError("Control sequence gap")
                        if any(r["native_action_count"] != current + 1 for r in group):
                            raise ValueError("Control/native action count mismatch")
                        left, right = arm_targets(group[-1])
                        prior_left, prior_right = arm_targets(group[-2])
                        if max(abs(a - b) for a, b in zip(left + right, prior_left + prior_right)) > 1e-6:
                            raise ValueError("Recorded endpoint is not a held joint target")
                        requested = row["action"]
                        joint_mode = "left_arm_joint_state" in requested
                        if joint_mode:
                            original = requested["left_arm_joint_state"] + requested["right_arm_joint_state"]
                            if max(abs(a - b) for a, b in zip(left + right, original)) > 1e-5:
                                raise ValueError("Native joint request differs from executed endpoint")
                        else:
                            if not {"left_ee_pose", "right_ee_pose"} <= set(requested):
                                raise ValueError("Unknown native action interface")
                        grips = [requested[side + "_ee_joint_state"][0] for side in ("left", "right")]
                        if not all(math.isfinite(x) for x in grips):
                            raise ValueError("Nonfinite gripper request")
                        clipped += sum(not 0 <= x <= 1 for x in grips)
                        grips = [max(0.0, min(1.0, x)) for x in grips]
                        command = left + [grips[0]] + right + [grips[1]]
                        if len(command) != 14 or not all(math.isfinite(x) for x in command):
                            raise ValueError("Invalid projected command")
                        actions.append({
                            "index": current, "role": role, "supervise": role == "teacher",
                            "applied_joint_command": command,
                            "native_requested_action": requested,
                            "arm_target_origin": "native_joint_command" if joint_mode else "recorded_IK_actuator_hold",
                            "control_first": group[0]["index"], "control_last": group[-1]["index"],
                            "simulation_time_before": group[0].get("simulation_time_before"),
                            "simulation_time_after": group[-1].get("simulation_time_after"),
                        })

                    with (root / "dense/control_steps.jsonl").open() as stream:
                        for line in stream:
                            control = json.loads(line)
                            n = control["macro_index"]
                            if n < start:
                                continue
                            if n >= stop:
                                break
                            if n != current:
                                finish()
                                if current is not None and n != current + 1:
                                    raise ValueError("Control group order gap")
                                current, group = n, []
                            group.append(control)
                            projected = {k: control[k] for k in (
                                "index", "macro_index", "native_action_count", "executed_actuator_targets",
                                "robot_feedback", "simulation_time_before", "simulation_time_after",
                                "physical_dt_s", "decimation") if k in control}
                            projected["role"] = role
                            projected["robot_feedback_recorded"] = bool(control.get("robot_feedback"))
                            controls_writer.write(json_bytes(projected))
                    finish()
                controls_writer.close()
                add_bytes("controls.jsonl.gz", control_buffer.getvalue())
                if [r["index"] for r in actions] != list(range(len(actions))):
                    raise ValueError("Selected action trajectory has a gap")
                if [r["applied_actions"] for r in observations] != list(range(len(actions) + 1)):
                    raise ValueError("Selected observation trajectory has a gap")
                payload = {
                    "format": "robofate-prefix-success-episode-v1",
                    "source_id": episode["source_id"], "episode_key": key, "task": episode["task"],
                    "instruction": instruction, "native_success": True, "teacher_start": t1,
                    "action_budget": episode["native_total_budget"],
                    "action_interface": "dm05_absolute_joint_command",
                    "action_order": ["left_arm_0:6", "left_gripper", "right_arm_0:6", "right_gripper"],
                    "arm_units": "radians", "gripper_command_range": [0, 1], "native_hz": 25,
                    "state_gripper_semantics": "native recorded normalized previous control, not an independent sensor",
                    "gripper_request_clipping_count": clipped,
                    "actions": actions, "observations": observations, "segments": segments,
                    "original_prefix_terminal_observation": prefix_boundary,
                    "boundary_note": "The original DM t1 observation is retained separately; the flattened trajectory uses the fresh restored teacher t1 observation.",
                    "controls": "controls.jsonl.gz",
                }
                add_bytes("episode.json", json_bytes(payload))
                provenance = {
                    "original_manifest_sha256": episode["source_manifest_sha256"],
                    "source_id": episode["source_id"],
                    "group_id": manifest.get("group_id"), "student_split": manifest.get("student_split"),
                    "exposure_status": manifest.get("exposure_status"),
                    "original_manifest_qualified": manifest.get("qualified"),
                    "original_qualification_scope": manifest.get("qualification_scope"),
                    "original_teacher_result_flags": {
                        k: result.get(k) for k in ("native_success", "qualified", "counted_in_quota")},
                    "source_initialization": {
                        k: manifest.get("source_initialization", {}).get(k)
                        for k in ("canonical_layout_sha256", "encoded_seed", "layout_ordinal")},
                    "export_scope": "common prefix and teacher success suffix only",
                    "excluded": ["DM failure suffix", "teacher replayed prefix", "extra dense RGB",
                                 "full-state restore snapshots", "renderer diagnostics"],
                }
                add_bytes("provenance.json", json_bytes(provenance))
                checksum_data = json_bytes({"algorithm": "sha256", "files": checksums})
                add_bytes("checksums.json", checksum_data)
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(str(partial), str(final))
    receipt = {
        "episode_key": key, "source_id": episode["source_id"], "task": episode["task"],
        "path": str(final), "bytes": hashed.bytes, "sha256": hashed.digest.hexdigest(),
        "source_manifest_sha256": episode["source_manifest_sha256"],
        "t1_native": t1, "actions": len(actions), "teacher_actions": len(actions) - t1,
        "observations": len(observations), "image_count": image_count,
        "member_count": len(checksums), "selected_original_observation_bytes": source_bytes,
        "validation": "All PNG chunks/CRC/deflate scanlines, state vectors, command/control alignment and held targets checked",
    }
    temporary_receipt = ready.with_suffix(".json.tmp")
    temporary_receipt.write_bytes(json_bytes(receipt))
    os.replace(str(temporary_receipt), str(ready))
    return receipt


def main(request):
    if socket.gethostname() != request["expected_hostname"]:
        raise RuntimeError("Hostname mismatch")
    if not isinstance(request.get("episodes"), list) or not request["episodes"]:
        raise ValueError("A nonempty episodes list is required")
    keys = [episode["episode_key"] for episode in request["episodes"]]
    if len(keys) != len(set(keys)) or any(
            not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", key)
            for key in keys):
        raise ValueError("Episode keys must be unique safe names")
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=True)
    for episode in request["episodes"]:
        receipt = export_episode(episode, output)
        print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    if "REQUEST_B64" in globals():
        main(json.loads(base64.b64decode(REQUEST_B64)))
    else:
        import argparse
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--plan", required=True,
                            help="JSON with expected_hostname, output, and inventoried episodes")
        args = parser.parse_args()
        main(json.loads(Path(args.plan).read_bytes()))
