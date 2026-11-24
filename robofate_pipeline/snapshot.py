"""Freeze explicitly selected source roots without executing project code."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import tarfile
import tempfile

EXCLUDED_DIRS = {".git", ".cache", ".ssh", ".venv", "venv", "__pycache__",
                 "node_modules", "runs", "run_data", "sessions", "logs"}
SOURCE_SUFFIXES = {
    ".py", ".sh", ".json", ".jsonl", ".toml", ".yaml", ".yml", ".md", ".txt",
    ".kit", ".ini", ".cfg", ".xml", ".jinja", ".j2", ".cu", ".cuh", ".cpp",
    ".cc", ".c", ".h", ".hpp", ".cmake", ".in", ".urdf", ".xacro", ".usda",
    ".js", ".mjs", ".ts", ".rst", ".pyi", ".typed", ".proto", ".patch", ".diff",
    ".mdx", ".launch", ".msg", ".cff", ".Dockerfile", ".mdl", ".example",
}
SOURCE_NAMES = {"CMakeLists.txt", "Makefile", "Dockerfile", ".gitignore",
                ".gitattributes", ".gitmodules", ".dockerignore", "uv.lock",
                "poetry.lock", "requirements.lock"}
SECRET = re.compile(
    rb"(?:ms-[a-f0-9]{8}-[a-f0-9-]{27,}|sk-[A-Za-z0-9_-]{20,}|"
    rb"github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
)


def collect(roots, max_bytes=256 * 1024 * 1024):
    entries, omitted, total = [], [], 0
    if not roots or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in roots):
        raise ValueError("At least one safe root label is required")
    for label, root in sorted(roots.items()):
        root = Path(root)
        if not root.is_dir() or root.is_symlink():
            raise ValueError("Source root must be a real directory: " + str(root))
        for base, dirs, names in os.walk(root, followlinks=False):
            kept = []
            for name in sorted(dirs):
                path = Path(base) / name
                if name in EXCLUDED_DIRS or path.is_symlink():
                    omitted.append({"path": label + "/" + path.relative_to(root).as_posix(),
                                    "reason": "excluded directory or symlink"})
                else:
                    kept.append(name)
            dirs[:] = kept
            for name in sorted(names):
                path = Path(base) / name
                relative = label + "/" + path.relative_to(root).as_posix()
                if path.is_symlink():
                    omitted.append({"path": relative, "reason": "symlink"})
                    continue
                if (name.startswith(".env") or any(x in name.lower() for x in (
                        "credential", "access_token", "password", "auth.json"))):
                    omitted.append({"path": relative, "reason": "secret-bearing filename"})
                    continue
                if (path.suffix not in SOURCE_SUFFIXES and name not in SOURCE_NAMES
                        and not name.startswith(("LICENSE", "NOTICE", "Dockerfile"))):
                    omitted.append({"path": relative, "reason": "non-source extension"})
                    continue
                if path.stat().st_size > 16 * 1024 * 1024:
                    raise ValueError("Oversized source file: " + relative)
                raw = path.read_bytes()
                if SECRET.search(raw):
                    raise ValueError("Credential pattern; manual review required: " + relative)
                total += len(raw)
                if total > max_bytes:
                    raise ValueError("Source snapshot byte budget exceeded")
                entries.append((path, {"path": relative, "bytes": len(raw),
                                       "sha256": hashlib.sha256(raw).hexdigest(),
                                       "mode": 0o755 if path.stat().st_mode & 0o111 else 0o644}))
    if not entries:
        raise ValueError("No source files selected")
    return entries, omitted


def freeze(roots, output, expected_hostname):
    if socket.gethostname() != expected_hostname:
        raise ValueError("Hostname mismatch; no files written")
    output = Path(output)
    if output.exists():
        raise ValueError("Output already exists")
    entries, omitted = collect(roots)
    manifest = {
        "schema": "robofate.source-snapshot.v2",
        "files": [entry for _, entry in entries], "omitted": omitted,
        "runtime_executed": False, "deployment_ready_claimed": False,
        "note": "Source selection and credential patterns are not a dependency-closure proof or a full secret audit.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".snapshot-", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as raw_output:
            with tarfile.open(fileobj=raw_output, mode="w:gz") as archive:
                for path, entry in entries:
                    raw = path.read_bytes()
                    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                        raise ValueError("Source changed during freeze: " + entry["path"])
                    member = tarfile.TarInfo(entry["path"])
                    member.size, member.mode = len(raw), entry["mode"]
                    archive.addfile(member, io.BytesIO(raw))
                raw = (json.dumps(manifest, indent=2) + "\n").encode()
                member = tarfile.TarInfo("SOURCE_MANIFEST.json")
                member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))
        # Publish without replacing a concurrently created destination.
        os.link(temporary, output)
    finally:
        os.unlink(temporary)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-hostname", required=True)
    args = parser.parse_args()
    roots = {}
    for value in args.root:
        label, separator, path = value.partition("=")
        if not separator or label in roots:
            parser.error("--root requires unique LABEL=PATH values")
        roots[label] = Path(path)
    manifest = freeze(roots, args.output, args.expected_hostname)
    print(json.dumps({"files": len(manifest["files"]), "omitted": len(manifest["omitted"]),
                      "output": str(args.output)}))
