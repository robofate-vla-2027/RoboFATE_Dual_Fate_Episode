"""Canonical bounded evidence reads and create-without-overwrite JSON receipts."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat


def need(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "Duplicate evidence key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: need(False, "Nonfinite evidence"))


def read_bytes(path, *, roots=(), limit=256 * 1024 * 1024, allow_empty=False):
    path = Path(path)
    need(path.is_absolute() and path.resolve(strict=True) == path, "Canonical evidence path required")
    need(not roots or any(path.is_relative_to(root) for root in roots), "Evidence outside allowed roots")
    signature = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and (0 if allow_empty else 1) <= before.st_size <= limit,
             "Bounded regular evidence required")
        raw = stream.read(before.st_size + 1)
        after = os.fstat(stream.fileno())
    need(len(raw) == before.st_size and signature(before) == signature(after) == signature(path.stat()),
         "Evidence changed while reading")
    return raw


def file_ref(path, *, roots=(), allow_empty=False):
    path = Path(path)
    raw = read_bytes(path, roots=roots, allow_empty=allow_empty)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def read_pin(reference, *, roots=(), allow_empty=False):
    need(type(reference) is dict and set(reference) == {"path", "sha256"}
         and type(reference["sha256"]) is str and re.fullmatch("[0-9a-f]{64}", reference["sha256"]),
         "Exact SHA256 file reference required")
    raw = read_bytes(reference["path"], roots=roots, allow_empty=allow_empty)
    need(hashlib.sha256(raw).hexdigest() == reference["sha256"], "Pinned evidence changed")
    return raw


def document(path):
    return decode(read_bytes(path))


def verify_blob(reference):
    """Hash large external weights in bounded memory; never deserialize them."""
    need(type(reference) is dict and set(reference) == {"path", "sha256"}
         and type(reference["sha256"]) is str and re.fullmatch("[0-9a-f]{64}", reference["sha256"]),
         "Exact blob SHA256 reference required")
    path = Path(reference["path"])
    need(path.is_absolute() and path.resolve(strict=True) == path, "Canonical blob path required")
    signature = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    h = hashlib.sha256()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_size > 0, "Nonempty regular blob required")
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
        after = os.fstat(stream.fileno())
    need(signature(before) == signature(after) == signature(path.stat())
         and h.hexdigest() == reference["sha256"], "Blob changed or checksum differs")
    return before.st_size


def exclusive(path, value):
    path = Path(path)
    need(path.is_absolute() and path.parent.resolve(strict=True) == path.parent
         and path.resolve() == path, "Canonical existing receipt parent required")
    raw = (canonical(value) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        need(stream.write(raw) == len(raw), "Short receipt write")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
