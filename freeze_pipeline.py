"""Freeze a private, credential-scanned source snapshot. No runtime execution."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import tarfile
import time

assert socket.gethostname() == "20250103-instance"
release = Path("/vla/mxk/projects/ReFATE/releases/dual_fate_episode_20261122_01a201f9")
destination = release / "pipeline"
destination.mkdir(parents=True, exist_ok=True)
roots = {
    "runtime": Path("/data/refate_chat2_20261106_01a19072/runtime"),
    "source_tree": Path("/vla/mxk/projects/ReFATE/coordination/20261106_chat2_h20_01a19072/source_tree"),
}
extensions = {".py", ".sh", ".json", ".jsonl", ".toml", ".yaml", ".yml", ".md",
              ".txt", ".kit", ".ini", ".cfg", ".xml", ".jinja", ".j2"}
excluded = {"__pycache__", ".git", ".cache", "node_modules", "runs", "run_data",
            "published", "sessions", "agent", "home", "logs", "teacher_package_0.151.0-alpha.7.1"}
secret = re.compile(
    rb"(?:ms-[a-f0-9]{8}-[a-f0-9-]{27,}|sk-[A-Za-z0-9_-]{20,}|"
    rb"github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
)
files = []
blocked = []
for label, root in roots.items():
    for base, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in excluded and not d.startswith("."))
        for name in sorted(names):
            path = Path(base) / name
            if (path.suffix not in extensions and not name.startswith(("LICENSE", "NOTICE", "Dockerfile"))) \
                    or path.is_symlink() or name.startswith(".") or any(
                        x in name.lower() for x in ("credential", "access_token", "password", "auth.json")):
                continue
            if path.stat().st_size > 16000000:
                blocked.append({"path": str(path), "reason": "oversized source/config file"})
                continue
            raw = path.read_bytes()
            if secret.search(raw):
                blocked.append({"path": str(path), "reason": "credential pattern"})
                continue
            files.append({
                "path": str(path), "archive_path": label + "/" + str(path.relative_to(root)),
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            })
manifest = {
    "schema": "robofate.private_source_snapshot.v1",
    "host": socket.gethostname(), "created_at_unix": time.time(),
    "files": files, "excluded_paths": blocked,
    "runtime_executed": False, "gpu_used": False, "credentials_included": False,
    "deployment_ready_claimed": False,
    "notes": [
        "Source snapshot, not a new benchmark or a qualification of every task.",
        "External simulator, frozen model weights, private teacher executable, and API credentials are not redistributed.",
        "Historical absolute paths and hash pins are preserved. Deployment requires explicit path bindings and resource authorization.",
        "Local orchestration helpers are not part of this remote snapshot and must be added to the final repository.",
    ],
}
(destination / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
archive = destination / "chat2_production_source.tar.gz"
assert not archive.exists(), "Never overwrite an existing frozen snapshot"
with tarfile.open(archive, "w:gz", compresslevel=4) as output:
    for entry in files:
        path = Path(entry["path"])
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"], "Source changed during snapshot"
        info = tarfile.TarInfo(entry["archive_path"])
        info.size = len(raw)
        info.mode = 0o755 if path.suffix == ".sh" else 0o644
        info.mtime = 0
        output.addfile(info, io.BytesIO(raw))
    raw = json.dumps(manifest, indent=2).encode()
    info = tarfile.TarInfo("SOURCE_MANIFEST.json")
    info.size = len(raw)
    output.addfile(info, io.BytesIO(raw))
summary = {
    "archive": str(archive), "bytes": archive.stat().st_size,
    "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    "source_files": len(files), "source_bytes": sum(x["bytes"] for x in files),
    "excluded": blocked, "local_orchestration_still_required": True,
    "published": False,
}
(destination / "SNAPSHOT_RECEIPT.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary))
