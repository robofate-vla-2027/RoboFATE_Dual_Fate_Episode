"""Fail-closed release inventory. Reports rule names, never matched secret text."""
import hashlib
import json
from pathlib import Path
import re
import stat

RULES = {
    "credential": re.compile(
        rb"(?:ms-[a-f0-9]{8}-[a-f0-9-]{27,}|sk-[A-Za-z0-9_-]{20,}|"
        rb"github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
        rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"),
    "user_home": re.compile(rb"(?<![\w/])/(?:Users|home)/(?!agent/)[A-Za-z0-9_.-]+/"),
    "machine_id": re.compile(rb"(?:GPU-[a-fA-F0-9-]{20,}|nb-[a-z0-9]{8,}-[0-9]+)"),
    "historical_collection": re.compile(
        rb"(?:refate[-_]20[0-9]{6}[-_]|chat[12]_01[a-f0-9]{6}|"
        rb"[a-z_]+_b[0-9]+_l[0-9]+_[a-f0-9]{16}|"
        rb"01[a-f0-9]{6}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})",
        re.I),
}
FORBIDDEN_PARTS = {".git", ".ssh", ".venv", "__pycache__", "node_modules",
                   "build", "dist", "logs", "checkpoints", "run_data"}
FORBIDDEN_SUFFIXES = {".gz", ".zip", ".tar", ".xz", ".bz2", ".7z", ".whl", ".pyc",
                      ".npz", ".npy", ".mp4", ".png", ".jpg", ".jpeg", ".pt", ".pth",
                      ".safetensors", ".jsonl", ".log", ".sqlite", ".db"}
FORBIDDEN_NAMES = {"CHECKPOINT.json", "episode.json", "result.json", "source_identity.json",
                   "PUBLICATION_VALIDATION.json", "LOCAL_REPAIR_STATUS.json", ".DS_Store"}
INDEX = "SOURCE_INDEX.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def audit_tree(root, *, denylist=(), verify_index=True):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Release root must be a real directory")
    if not isinstance(denylist, (tuple, list)) or any(not isinstance(x, str) or not x for x in denylist):
        raise ValueError("Private denylist must contain nonempty literal strings")
    denied = [value.encode().lower() for value in denylist]
    findings, inventory = [], []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        reasons = []
        if any(part in FORBIDDEN_PARTS or part.endswith(".egg-info") for part in path.relative_to(root).parts):
            reasons.append("generated_or_private_path")
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            reasons.append("link_or_special_file")
        if path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name.startswith(".env"):
            reasons.append("runtime_payload_or_archive")
        if stat.S_ISREG(mode):
            raw = path.read_bytes()
            combined = relative.encode() + b"\n" + raw
            for name, pattern in RULES.items():
                if pattern.search(combined):
                    reasons.append(name)
            folded = combined.lower() if denied else b""
            if any(value in folded for value in denied):
                reasons.append("private_literal")
            if relative != INDEX:
                inventory.append({"path": relative, "bytes": len(raw), "sha256": digest(raw)})
            # The only binary source resources permitted are upstream robot meshes.
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError:
                if not (relative.startswith("source/native/third_party/curobo/curobo/content/assets/")
                        and path.suffix.lower() == ".stl"):
                    reasons.append("unreviewed_binary")
        if reasons:
            # A private path must not be echoed into a shareable report.
            findings.append({"file_id": digest(relative.encode()), "rules": sorted(set(reasons))})
    if not inventory:
        findings.append({"file_id": "tree", "rules": ["empty_release"]})
    if verify_index:
        try:
            expected = json.loads((root / INDEX).read_text())
            if expected.get("schema") != "robofate.clean-source.v1" or expected.get("files") != inventory:
                raise ValueError("Inventory mismatch")
        except (OSError, ValueError, TypeError):
            findings.append({"file_id": "inventory", "rules": ["source_index_mismatch"]})
    return {"passed": not findings, "files_checked": len(inventory) + int((root / INDEX).is_file()),
            "bytes_checked": sum(row["bytes"] for row in inventory),
            "tree_sha256": digest(json.dumps(inventory, sort_keys=True).encode()),
            "findings": findings, "historical_literal_scan_enabled": bool(denied),
            "scope": "Distributed paths, file bytes, payload exclusion and exact inventory; not a universal privacy proof."}


def write_index(root):
    root = Path(root)
    result = audit_tree(root, verify_index=False)
    if not result["passed"]:
        raise ValueError("Cannot index an unclean tree: " + json.dumps(result["findings"]))
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != INDEX:
            raw = path.read_bytes()
            rows.append({"path": path.relative_to(root).as_posix(), "bytes": len(raw), "sha256": digest(raw)})
    (root / INDEX).write_text(json.dumps({"schema": "robofate.clean-source.v1", "files": rows}, indent=2) + "\n")
