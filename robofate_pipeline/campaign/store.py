"""Durable locked CAS journal. Interrupted metadata commits require reconciliation."""
import copy
from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import re
import stat
import tempfile

from .. import evidence
from ..evidence import canonical, need
from ..runtime_policy import PACKAGE


def token(value):
    need(type(value) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", value),
         "Bounded path token required")
    return value


def digest(value):
    return hashlib.sha256((canonical(value) + "\n").encode()).hexdigest()


class Store:
    def __init__(self, root, *, data_kind, campaign, owner, host):
        self.root = Path(root)
        need(self.root.is_absolute() and self.root.resolve(strict=True) == self.root
             and self.root.is_dir() and not self.root.is_relative_to(PACKAGE.parent)
             and not PACKAGE.parent.is_relative_to(self.root), "Private external ledger directory required")
        need(data_kind in ("actual", "synthetic") and all(type(v) is str and v for v in (campaign, owner, host)),
             "Explicit ledger scope required")
        self.scope = dict(data_kind=data_kind, campaign_id=campaign, owner=owner, host=host)
        self.state = self.root / "state.json"
        self.journal = self.root / "transactions"
        self.lock = self.root / "ledger.lock"

    @contextmanager
    def locked(self):
        fd = os.open(self.lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        try:
            info = os.fstat(fd)
            need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "Private regular ledger lock required")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            current = self.lock.lstat()
            need((info.st_dev, info.st_ino) == (current.st_dev, current.st_ino), "Ledger lock replaced")
            binding = self.root / "binding.json"
            expected = dict(self.scope, lock_device=info.st_dev, lock_inode=info.st_ino)
            if binding.exists():
                need(evidence.document(binding) == expected, "Persistent lock or ledger scope changed")
            else:
                need(not self.state.exists() and not self.journal.exists(), "Missing established ledger binding")
                evidence.exclusive(binding, expected)
            yield info.st_ino
            need((self.lock.lstat().st_dev, self.lock.lstat().st_ino) == (info.st_dev, info.st_ino),
                 "Ledger lock changed during transaction")
        finally:
            os.close(fd)

    def validate(self, value):
        need(value["data_kind"] == self.scope["data_kind"] and value["campaign_id"] == self.scope["campaign_id"]
             and value["nodes"][self.scope["host"]]["active_chat_id"] == self.scope["owner"],
             "Ledger kind/campaign/owner changed")
        need(type(value["revision"]) is int and value["revision"] >= 0, "Invalid ledger revision")

    def initialize(self, value):
        self.validate(value)
        with self.locked():
            need(not self.state.exists() and not self.journal.exists(), "Ledger already exists or initialization interrupted")
            self.journal.mkdir(mode=0o700)
            evidence.exclusive(self.state, value)
            evidence.exclusive(self.root / "INITIAL.json", dict(
                state_sha256=evidence.file_ref(self.state)["sha256"], revision=value["revision"]))
        return evidence.file_ref(self.state)

    def _pending(self):
        need(self.journal.resolve(strict=True) == self.journal, "Linked journal directory")
        pending = []
        for path in self.journal.iterdir():
            need(path.is_dir() and path.resolve() == path, "Unexpected journal entry")
            if not (path / "COMMITTED.json").exists():
                pending.append(path.name)
        return pending

    def snapshot(self):
        with self.locked():
            need(not self._pending(), "Pending ledger transaction requires explicit reconciliation")
            value = self._verified_state()
            return value, evidence.file_ref(self.state)

    def _verified_state(self):
        value = evidence.document(self.state)
        self.validate(value)
        records = [evidence.document(self.root / "INITIAL.json")]
        for path in self.journal.iterdir():
            if (path / "COMMITTED.json").exists():
                record = evidence.document(path / "COMMITTED.json")
                need(record["intent"] == evidence.file_ref(path / "INTENT.json"), "Committed intent changed")
                records.append(record)
        latest = max(record["revision"] for record in records)
        hashes = {r["state_sha256"] for r in records if r["revision"] == latest}
        need(value["revision"] == latest and evidence.file_ref(self.state)["sha256"] in hashes,
             "Ledger changed outside its committed journal")
        return value

    def _replace(self, before_sha, after):
        need(evidence.file_ref(self.state)["sha256"] == before_sha, "CAS conflict")
        fd, name = tempfile.mkstemp(prefix=".state-", dir=self.root)
        try:
            raw = (canonical(after) + "\n").encode()
            with os.fdopen(fd, "wb") as stream:
                need(stream.write(raw) == len(raw), "Short ledger write")
                stream.flush()
                os.fsync(stream.fileno())
            need(evidence.file_ref(self.state)["sha256"] == before_sha, "CAS changed during publication")
            os.replace(name, self.state)
            directory = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def _foreign_unchanged(self, before, after):
        host = self.scope["host"]
        for key in ("campaign_id", "data_kind", "started_at", "target_at", "task_order",
                    "fixed_assignment_sha256", "quota_history", "quota_revision"):
            need(before.get(key) == after.get(key), "Campaign clock/assignment reset: " + key)
        for key, value in before["nodes"].items():
            if key != host:
                need(after["nodes"][key] == value, "Foreign node changed")
        for key, value in before["tasks"].items():
            if value["assigned_host"] != host:
                need(after["tasks"][key] == value, "Foreign task changed")
        for field in ("source_index", "reservations", "split_manifests"):
            for key, value in before[field].items():
                if value.get("host", host) != host:
                    need(after[field][key] == value, "Foreign source/split changed")

    def transact(self, operation_id, request, compute):
        token(operation_id)
        with self.locked() as inode:
            path = self.journal / operation_id
            if (path / "COMMITTED.json").exists():
                intent = evidence.document(path / "INTENT.json")
                receipt = evidence.document(path / "COMMITTED.json")
                need(intent["request"] == request and receipt["intent"] == evidence.file_ref(path / "INTENT.json"),
                     "Operation reused with different request or altered journal")
                need(not self._pending(), "Another transaction needs reconciliation")
                self._verified_state()
                return copy.deepcopy(intent["result"])
            need(not self._pending() and not path.exists(), "Interrupted transaction; do not blindly repeat")
            raw = evidence.read_bytes(self.state)
            before = evidence.decode(raw)
            need(before == self._verified_state(), "Ledger changed during snapshot")
            after, result = compute(copy.deepcopy(before), inode)
            self.validate(after)
            self._foreign_unchanged(before, after)
            need(after["revision"] == before["revision"] + 1 or after == before,
                 "Exactly one revision per mutation required")
            path.mkdir(mode=0o700)
            intent = dict(schema="robofate.ledger-intent.v1", scope=self.scope,
                request=copy.deepcopy(request), before_sha256=hashlib.sha256(raw).hexdigest(),
                before_revision=before["revision"], after=after, after_sha256=digest(after), result=result)
            reference = evidence.exclusive(path / "INTENT.json", intent)
            if after != before:
                self._replace(intent["before_sha256"], after)
            evidence.exclusive(path / "COMMITTED.json", dict(
                intent=reference, state_sha256=evidence.file_ref(self.state)["sha256"], revision=after["revision"]))
            return copy.deepcopy(result)

    def reconcile(self, operation_id):
        """Finish only a saved metadata intent; never rerun collection or compute."""
        token(operation_id)
        with self.locked():
            need(self._pending() == [operation_id], "Exactly the named uncertain transaction required")
            path = self.journal / operation_id
            intent = evidence.document(path / "INTENT.json")
            need(intent["scope"] == self.scope and digest(intent["after"]) == intent["after_sha256"],
                 "Intent binding changed")
            current = evidence.file_ref(self.state)["sha256"]
            if current == intent["before_sha256"]:
                self._replace(current, intent["after"])
            else:
                need(current == intent["after_sha256"], "Ledger diverged; manual audit required")
            evidence.exclusive(path / "COMMITTED.json", dict(
                intent=evidence.file_ref(path / "INTENT.json"), state_sha256=intent["after_sha256"],
                revision=intent["after"]["revision"], reconciled_metadata_only=True))
            return intent["result"]
