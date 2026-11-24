"""Entry for the original owned supervisor. Never run during offline tests."""
from pathlib import Path
import sys

from ..evidence import exclusive, need
from ..runtime_policy import RuntimePolicy


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    need(len(args) == 5, "Run, OCI SHA, parent PID, policy path and policy SHA required")
    run, digest, parent, policy_path, policy_sha = args
    policy = RuntimePolicy({"path": policy_path, "sha256": policy_sha})
    policy.host_guard()
    policy.verify_release()
    gs = policy.load("teacher/runsc_impl.py", dict(
        RUNSC=Path(policy.value["runsc"]), ID="refate-teacher",
        sessions=policy.root / "sessions", host_guard=policy.host_guard))
    if policy.value["private_runsc_tmpfs"]:
        from .private_tmp import bind_private_tmp
        exclusive(Path(run) / "oci/RUNSC_TMP_BINDING_001.json", bind_private_tmp())
    supervisor = policy.load("teacher/supervisor_impl.py", dict(gs=gs))
    return supervisor.supervise(run, digest, int(parent))


if __name__ == "__main__":
    raise SystemExit(main())
