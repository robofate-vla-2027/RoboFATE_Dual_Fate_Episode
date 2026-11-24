"""One owned foreground teacher, with a finite wait and no automatic retry.

This is deployment-neutral composition of the original parent lifecycle, not
source allocation, baseline collection, independent readback or ledger commit.
"""
from contextlib import contextmanager
import ctypes
import fcntl
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import time

from . import evidence
from .evidence import need
from .execution import load_context
from .native import deployment
from .native.resources import inspect as inspect_resources
from .process_identity import identity

SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


@contextmanager
def simulator_lock(policy):
    path = policy.root / "simulator.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        info = os.fstat(fd)
        need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "Private regular simulator lock required")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield fd
    finally:
        # Closing, rather than LOCK_UN, preserves the inherited lock if a child survives.
        os.close(fd)


def lease(policy, context_ref, run, fd):
    info = os.fstat(fd)
    return evidence.exclusive(run / "NATIVE_LEASE.json", dict(
        schema="robofate.native-lease.v1", context_ref=context_ref,
        execution_policy_ref=policy.reference, execution_owner=policy.value["execution_owner"],
        parent=identity(os.getpid()), lock_path=str(policy.root / "simulator.lock"),
        descriptor=fd, device=info.st_dev, inode=info.st_ino, automatic_repeat=False))


def validate_lease(policy, context_ref, lease_ref, fd):
    """Check the inherited open-file description; never acquire a second lock."""
    run = Path(context_ref["path"]).parent
    need(Path(lease_ref["path"]) == run / "NATIVE_LEASE.json", "Wrong Native lease path")
    value = evidence.decode(evidence.read_pin(lease_ref, roots=(policy.root,)))
    need(value["schema"] == "robofate.native-lease.v1" and value["context_ref"] == context_ref
         and value["execution_policy_ref"] == policy.reference
         and value["execution_owner"] == policy.value["execution_owner"]
         and type(fd) is int and fd >= 3 and value["descriptor"] == fd,
         "Native lease binding mismatch")
    need(os.getppid() == value["parent"]["pid"] and identity(os.getppid()) == value["parent"],
         "Original foreground parent is no longer alive")
    path = policy.root / "simulator.lock"
    info, actual = os.fstat(fd), path.lstat()
    need(value["lock_path"] == str(path) and stat.S_ISREG(info.st_mode)
         and stat.S_ISREG(actual.st_mode) and actual.st_nlink == 1
         and (info.st_dev, info.st_ino) == (actual.st_dev, actual.st_ino) == (value["device"], value["inode"]),
         "Inherited simulator lock identity changed")
    probe = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise ValueError("Foreground simulator lock is not held")
    finally:
        os.close(probe)
    os.set_inheritable(fd, False)
    return value


def parent_death_guard(parent):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG")
    need(os.getppid() == parent["pid"] and identity(os.getppid()) == parent,
         "Foreground parent changed during child startup")


def stop_owned(process, owned):
    """Original TERM/20s/KILL sequence, rechecking identity before each signal."""
    if process.poll() is not None:
        return process.wait(timeout=5)
    for sig, timeout in ((signal.SIGTERM, 20), (signal.SIGKILL, 5)):
        need(owned is not None and owned["pid"] == process.pid
             and owned["process_group"] == process.pid and identity(process.pid) == owned,
             "Changed or unknown owned process; refusing signal")
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return process.wait(timeout=5)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if sig == signal.SIGKILL:
                raise
    raise RuntimeError("Owned process cleanup did not settle")


def original_absent(owned):
    if owned is None:
        return False
    try:
        return identity(owned["pid"]) != owned
    except FileNotFoundError:
        return True


def group_members(owned):
    """Read-only residual check; a reaped leader is not proof its group is empty."""
    need(owned is not None and
         Path("/proc/sys/kernel/random/boot_id").read_text().strip() == owned["boot_id"],
         "Cannot verify an unknown process group or changed boot")
    members = []
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            found = identity(int(path.name))
        except FileNotFoundError:
            continue
        if found["process_group"] == owned["process_group"]:
            members.append(found)
    return sorted(members, key=lambda item: item["pid"])


def cleanup(process, owned, log, run):
    errors = []
    if process is not None:
        try:
            rc = stop_owned(process, owned)
            absent = original_absent(owned)
            remaining = group_members(owned)
            complete = absent and not remaining
            evidence.exclusive(run / "process_cleanup.json", dict(
                native=dict(identity=owned, returncode=rc, original_process_absent=absent,
                            process_group_remaining=remaining),
                owned_cleanup_complete=complete, cleanup_scope="Native process group only",
                isolated_worker_lifecycle_verified=False, data_status_not_decided_by_exit=True))
            need(complete, "Original owned process or group still present after wait")
        except BaseException as error:
            errors.append(error)
            try:
                evidence.exclusive(run / "process_cleanup_failed.json", dict(
                    type=type(error).__name__, message=str(error), owned_cleanup_complete=False,
                    qualified=False, automatic_repeat=False))
            except BaseException as secondary:
                error.add_note("Cleanup failure receipt also failed: " + repr(secondary))
    if log is not None:
        try:
            log.close()
        except BaseException as error:
            errors.append(error)
    return errors


def foreground(context_ref, *, on_started=None):
    context, _, _, policy = load_context(context_ref)
    policy.host_guard()
    run, session = Path(context["run_dir"]), Path(context["runtime_binding"]["run_dir"])
    need(not any((run / name).exists() or (run / name).is_symlink() for name in
                 ("foreground_started.json", "native_started.json", "NATIVE_LEASE.json", "result.json", "failure.json")),
         "Started or uncertain context requires reconciliation, not relaunch")
    need((session / "inputs").is_dir() and not any((session / name).exists() for name in ("agent", "runtime")),
         "Original teacher inputs must be prepared and never executed")
    need(context["native_wall_seconds"] == 3600 and type(context["native_wall_seconds"]) is int,
         "Original Native wall budget required")
    need(os.environ.get("MODEL_UPSTREAM") == policy.value["model_upstream"],
         "Authorized model-forwarder environment differs")
    with simulator_lock(policy) as fd:
        return _foreground(context_ref, context, policy, run, fd, on_started=on_started)


def _foreground(context_ref, context, policy, run, fd, *, on_started=None):
    process = owned = log = None
    previous = {}
    try:
        evidence.exclusive(run / "foreground_started.json", dict(
            identity=identity(os.getpid()), context_ref=context_ref, wall_ns=time.time_ns(),
            teacher_session=context["runtime_binding"]["session"], automatic_repeat=False))
        def interrupted(sig, frame):
            raise KeyboardInterrupt("User/foreground signal " + str(sig))
        for sig in SIGNALS:
            previous[sig] = signal.signal(sig, interrupted)
        native_python, env = deployment.prepare(policy, context)
        # Listener presence only. No request, provider call or route fallback.
        with socket.create_connection(deployment.route(policy), timeout=3):
            pass
        policy.host_guard()
        evidence.exclusive(run / "resource_selection.json", inspect_resources(policy, run))
        reference = lease(policy, context_ref, run, fd)
        command = [native_python, "-B", "-m", "robofate_pipeline.native_child",
                   "--context", context_ref["path"], "--sha256", context_ref["sha256"],
                   "--lease", reference["path"], "--lease-sha256", reference["sha256"],
                   "--lock-fd", str(fd)]
        log = (run / "stdout.log").open("x")
        process = subprocess.Popen(command, cwd=run, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(fd,))
        owned = identity(process.pid)
        need(owned["pid"] == process.pid and owned["process_group"] == process.pid,
             "Native child did not create its own process group")
        evidence.exclusive(run / "owned_native.json", dict(
            identity=owned, command=command, parent_pid=os.getpid(), role="native_teacher"))
        deadline = time.monotonic() + context["native_wall_seconds"]
        if on_started is not None:
            from .baseline.foreground import wait_file
            started = wait_file(run / "native_started.json", [process], deadline)
            need(started["pid"] == process.pid and started["context_ref"] == context_ref,
                 "Native start evidence differs")
            on_started(context_ref, owned, evidence.file_ref(run / "native_started.json"))
        remaining = deadline - time.monotonic()
        need(remaining > 0, "Native wall budget exhausted")
        rc = process.wait(timeout=remaining)
        evidence.exclusive(run / "native_wait.json", dict(
            pid=process.pid, returncode=rc, original_process_absent=original_absent(owned),
            wall_ns=time.time_ns(), data_status_not_decided_by_exit=True))
        return rc
    except BaseException as primary:
        try:
            evidence.exclusive(run / "foreground_error.json", dict(
                type=type(primary).__name__, message=str(primary), wall_ns=time.time_ns(),
                data_status_not_decided_by_exit=True, qualified=False, automatic_repeat=False))
        except BaseException as secondary:
            primary.add_note("Foreground error receipt also failed: " + repr(secondary))
        raise
    finally:
        primary = sys.exception()
        # Repeated foreground signals must not interrupt bounded owned cleanup.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        errors = cleanup(process, owned, log, run)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if errors:
            if primary is None:
                raise errors[0]
            for error in errors:
                primary.add_note("Foreground cleanup also failed: " + repr(error))
