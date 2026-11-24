"""Inventory immutable episode payloads; never import or run the collector."""
import base64
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import time

def main(request):
    host = socket.gethostname()
    if host != request["expected_hostname"]:
        raise ValueError("Hostname mismatch; no files written")
    if not isinstance(request.get("episodes"), list) or not request["episodes"]:
        raise ValueError("A nonempty episodes list is required")
    keys = [item["episode_key"] for item in request["episodes"]]
    if len(keys) != len(set(keys)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", key) for key in keys):
        raise ValueError("Episode keys must be unique safe names")
    root = Path(request["release_root"])
    if not root.is_absolute():
        raise ValueError("release_root must be absolute")
    root.mkdir(parents=True, exist_ok=False)
    allowed_names = {
        "camera_profile.json", "source_identity.json", "initialized_layout.json",
        "initial_capture.json", "initial_physics_state.json", "initial_runtime_evidence.json",
        "initial_task_auxiliary.json", "native_actions.jsonl", "native_observations.jsonl",
        "renderer_raw_observations.jsonl", "result.json", "recording_contract.json",
        "recording_progress.json", "writer_close.json", "episode_data_committed.json",
        "native_restore_config.json", "actual_t1_verified.json", "state_scope.json",
        "initial_boundary_saved.json", "actual_initial_binding.json", "timing_summary.json",
        "restore_auxiliary_render_comparison.json", "restore_boundary.json",
        "restore_observation_comparison.json", "restore_request.json",
        "restore_state_comparison.json", "restore_profile.json",
    }
    frame_metadata = re.compile(r"(camera|physics|task_auxiliary|runtime_evidence)_\d+\.json$")
    frame_directory = re.compile(r"(macro_obs|native_obs|observation)_\d+$")
    summary = {"host": host, "episodes": [], "started_at": time.time()}
    for episode in request["episodes"]:
        item = dict(episode)
        item["branches"] = {}
        inventory = []
        errors = []
        for role, run in episode["runs"].items():
            source = Path(run)
            physical = episode.get("physical_runs", {}).get(role)
            if physical:
                candidate = Path(physical)
                if not candidate.is_absolute() or not candidate.is_dir():
                    raise ValueError("Explicit relocation must be an existing absolute directory")
                if candidate.is_dir():
                    manifest_raw = Path(episode["source_manifest"]).read_bytes()
                    assert hashlib.sha256(manifest_raw).hexdigest() == episode["source_manifest_sha256"]
                    manifest = json.loads(manifest_raw)
                    identity = json.loads((candidate / "source_identity.json").read_bytes())
                    assert episode["source_id"] in json.dumps(identity), "Relocated source identity mismatch"
                    reference = (manifest["baseline"].get("source_result_ref")
                                 if role == "dm_failure" else
                                 manifest["recording"].get("native_actions_ref"))
                    if reference:
                        local = candidate / ("result.json" if role == "dm_failure" else "native_actions.jsonl")
                        assert hashlib.sha256(local.read_bytes()).hexdigest() == reference["sha256"]
                    source = candidate
            source = source.resolve()
            resolved_source = str(source)
            info = {"original_path": run, "resolved_path": str(source.resolve()),
                    "files": 0, "bytes": 0, "root_excluded": [], "suffixes": {}}
            item["branches"][role] = info
            if not source.is_dir():
                errors.append(role + ": missing run directory")
                continue
            selected = []
            for entry in sorted(source.iterdir()):
                if entry.is_file() and (entry.name in allowed_names or frame_metadata.fullmatch(entry.name)):
                    selected.append(entry)
                elif entry.is_dir() and (entry.name in {"dense", "initial"} or frame_directory.fullmatch(entry.name)):
                    if entry.is_symlink():
                        errors.append(role + ": symlinked payload directory " + entry.name)
                        continue
                    for base, directories, files in os.walk(entry, followlinks=False):
                        for name in sorted(files):
                            path = Path(base) / name
                            if path.suffix in {".json", ".jsonl", ".npz", ".npy", ".png", ".jpg", ".jpeg"}:
                                selected.append(path)
                else:
                    info["root_excluded"].append(entry.name)
            for path in selected:
                if path.is_symlink() and os.path.commonpath(
                        (str(path.resolve()), resolved_source)) != resolved_source:
                    errors.append(role + ": external symlink " + str(path))
                    continue
                stat = path.stat()
                relative = str(path.relative_to(source))
                inventory.append({"source": str(path), "archive_path": role + "/" + relative,
                                  "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
                info["files"] += 1
                info["bytes"] += stat.st_size
                info["suffixes"][path.suffix] = info["suffixes"].get(path.suffix, 0) + 1
            for required in ("native_actions.jsonl", "dense/control_steps.jsonl", "result.json"):
                if not (source / required).is_file():
                    errors.append(role + ": missing " + required)
            for name in ("result.json", "timing_summary.json", "native_wait.json"):
                path = source / name
                if path.is_file() and path.stat().st_size < 100000:
                    info[name] = json.loads(path.read_bytes())
        item["errors"] = errors
        item["files"] = len(inventory)
        item["bytes"] = sum(x["bytes"] for x in inventory)
        path = root / "inventories" / (episode["episode_key"] + ".json")
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"episode": item, "files": inventory}, indent=2) + "\n")
        item["inventory_path"] = str(path)
        summary["episodes"].append(item)
    summary["finished_at"] = time.time()
    summary["file_count"] = sum(e["files"] for e in summary["episodes"])
    summary["bytes"] = sum(e["bytes"] for e in summary["episodes"])
    summary["errors"] = sum(len(e["errors"]) for e in summary["episodes"])
    summary["network_probe_performed"] = False
    (root / "inventory_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    if "REQUEST_B64" in globals():
        plan = json.loads(base64.b64decode(REQUEST_B64))
    else:
        import argparse
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--plan", required=True, help="Inventory request JSON; see configs/inventory.example.json")
        args = parser.parse_args()
        plan = json.loads(Path(args.plan).read_bytes())
    result = main(plan)
    print(json.dumps(result))
    raise SystemExit(1 if result["errors"] else 0)
