"""Explicit, read-only host/owner/dependency checks. Never select or launch GPUs."""
import hashlib
from pathlib import Path
import socket


def inspect(config, hostname=None):
    errors = []
    actual = socket.gethostname() if hostname is None else hostname
    if not config.get("expected_hostname") or config["expected_hostname"] != actual:
        errors.append("Explicit expected_hostname must match")
    if not isinstance(config.get("execution_owner"), str) or not config["execution_owner"].strip():
        errors.append("Explicit execution_owner is required")
    if config.get("data_kind") != "actual":
        errors.append("Synthetic metadata cannot authorize real collection")
    dependencies = config.get("dependencies", {})
    required = ("native_root", "simulator_python", "policy_python", "policy_weights",
                "teacher_runner", "task_assets", "deployment_manifest")
    for name in required:
        value = dependencies.get(name)
        if not isinstance(value, str) or not value or not Path(value).is_absolute() or not Path(value).exists():
            errors.append("Missing explicit dependency: " + name)
    try:
        import json
        manifest = json.loads(Path(dependencies["deployment_manifest"]).read_text())
        if not isinstance(manifest["files"], list) or not manifest["files"]:
            raise ValueError("Nonempty dependency pins required")
        for entry in manifest["files"]:
            path = Path(entry["path"])
            if not path.is_absolute() or path.resolve(strict=True) != path or not path.is_file():
                raise ValueError("Invalid pinned dependency")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != entry["sha256"]:
                raise ValueError("Dependency SHA256 mismatch")
    except (KeyError, TypeError, OSError, ValueError) as error:
        errors.append("Invalid deployment_manifest: " + str(error))
    return {"static_checks_passed": not errors, "errors": errors,
            "collection_ready": False, "live_collection_validated": False,
            "note": "Presence and identity are not resource authorization, restore qualification or simulator readiness."}
