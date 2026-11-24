"""Use the outer container's private tmpfs for the runsc chroot only."""
import os
from pathlib import Path
import stat


def bind_private_tmp():
    path = Path("/tmp")
    value = path.lstat()
    assert stat.S_ISDIR(value.st_mode) and not path.is_symlink()
    assert value.st_uid == 0 and stat.S_IMODE(value.st_mode) == 0o1777
    mounts = [row.split() for row in Path("/proc/self/mountinfo").read_text().splitlines()]
    matches = [row for row in mounts if row[4] == "/tmp"]
    assert len(matches) == 1 and matches[0][matches[0].index("-") + 1] == "tmpfs"
    assert "rw" in matches[0][5].split(",")
    old = os.environ.get("TMPDIR")
    os.environ["TMPDIR"] = str(path)
    return {"previous_TMPDIR": old, "runsc_TMPDIR": str(path), "private_tmpfs": True,
            "native_environment_modified": False, "permissions_modified": False}
