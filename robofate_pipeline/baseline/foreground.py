"""Owned Native-first baseline and DM service, one lease and bounded cleanup."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .. import evidence
from ..evidence import need
from .. import foreground as owned
from ..native import deployment
from ..native.resources import inspect as inspect_resources
from ..process_identity import identity
from . import service
from .runtime import load_context, validate_ready


def wait_file(path, children, deadline):
    while not path.exists():
        need(all(process.poll() is None for process in children), "Owned process exited before required boundary")
        if time.monotonic() >= deadline:
            raise TimeoutError("Original Native/policy wall allowance exhausted")
        time.sleep(.25)
    need(all(process.poll() is None for process in children), "Owned process stopped at boundary")
    return evidence.document(path)


def cleanup(children, identities, logs, run):
    errors, records = [], {}
    for role in ("native", "policy"):
        if role not in children:
            continue
        try:
            process, owner = children[role], identities.get(role)
            rc = owned.stop_owned(process, owner)
            absent, members = owned.original_absent(owner), owned.group_members(owner)
            records[role] = dict(identity=owner, returncode=rc, original_process_absent=absent,
                                 process_group_remaining=members)
            need(absent and not members, "Owned baseline process group not clean: " + role)
        except BaseException as error:
            errors.append(error)
    for log in logs:
        try:
            log.close()
        except BaseException as error:
            errors.append(error)
    try:
        evidence.exclusive(run / "process_cleanup.json", dict(
            records, owned_cleanup_complete=not errors and "native" in records,
            cleanup_scope="Native and policy process groups", isolated_worker_lifecycle_verified=True,
            data_status_not_decided_by_exit=True, errors=[repr(e) for e in errors]))
    except BaseException as error:
        errors.append(error)
    return errors


def foreground(context_ref, *, on_started=None):
    c, policy = load_context(context_ref)
    policy.host_guard()
    run = Path(c["run_dir"])
    need(not any((run / name).exists() or (run / name).is_symlink() for name in (
        "foreground_started.json", "native_started.json", "NATIVE_LEASE.json", "result.json", "failure.json")),
        "Started/uncertain baseline requires reconciliation, not relaunch")
    with owned.simulator_lock(policy) as fd:
        return _foreground(context_ref, c, policy, run, fd, on_started=on_started)


def _foreground(context_ref, c, policy, run, fd, *, on_started=None):
    children, identities, logs, previous = {}, {}, [], {}
    try:
        evidence.exclusive(run / "foreground_started.json", dict(
            identity=identity(os.getpid()), context_ref=context_ref, wall_ns=time.time_ns(), automatic_repeat=False))
        def interrupted(sig, frame):
            raise KeyboardInterrupt("Owned baseline signal " + str(sig))
        for sig in owned.SIGNALS:
            previous[sig] = signal.signal(sig, interrupted)
        native_python, native_env = deployment.prepare(policy, dict(
            c, source_context_ref=context_ref, native_limit=c["native_total_steps"]))
        policy_python, policy_env = service.environment(policy, native_env)
        reference = owned.lease(policy, context_ref, run, fd)
        def launch(role, python, env):
            policy.host_guard()
            evidence.exclusive(run / (role + "_resource_selection.json"), inspect_resources(policy, run))
            log = (run / (role + ".log")).open("x")
            logs.append(log)
            command = [python, "-B", "-m", "robofate_pipeline.baseline.child", role,
                "--context", context_ref["path"], "--sha256", context_ref["sha256"],
                "--lease", reference["path"], "--lease-sha256", reference["sha256"], "--lock-fd", str(fd)]
            process = subprocess.Popen(command, env=env, cwd=run, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(fd,))
            children[role] = process
            owner = identity(process.pid)
            identities[role] = owner
            need(owner["process_group"] == process.pid, "Owned child process group differs")
            evidence.exclusive(run / ("owned_" + role + ".json"), dict(
                identity=owner, command=command, role=role, parent_pid=os.getpid()))
            return process
        native = launch("native", native_python, native_env)
        deadline = time.monotonic() + c["native_wall_seconds"]
        started = wait_file(run / "native_started.json", [native], deadline)
        need(started["pid"] == native.pid and started["context_ref"] == context_ref, "Native start evidence differs")
        if on_started is not None:
            on_started(context_ref, identities["native"], evidence.file_ref(run / "native_started.json"))
        boundary = wait_file(run / "native_boundary_ready.json", [native], deadline)
        need(boundary["native_used"] == boundary["control_ticks"] == 0
             and boundary["phase"] == "actual_initial_boundary_saved_before_policy_load",
             "DM may load only after the actual saved initial boundary")
        dm = launch("policy", policy_python, policy_env)
        ready = wait_file(run / "server/model_ready.json", [native, dm], min(deadline, time.monotonic() + 900))
        validate_ready(ready, c, policy, dm.pid)
        remaining = deadline - time.monotonic()
        need(remaining > 0, "Native wall budget exhausted")
        rc = native.wait(timeout=remaining)
        evidence.exclusive(run / "native_wait.json", dict(pid=native.pid, returncode=rc,
            original_process_absent=owned.original_absent(identities["native"]), data_status_not_decided_by_exit=True))
        return rc
    except BaseException as primary:
        try:
            evidence.exclusive(run / "foreground_error.json", dict(type=type(primary).__name__,
                message=str(primary), qualified=False, automatic_repeat=False, data_status_not_decided_by_exit=True))
        except BaseException as secondary:
            primary.add_note("Foreground receipt also failed: " + repr(secondary))
        raise
    finally:
        primary = sys.exception()
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        errors = cleanup(children, identities, logs, run)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if errors:
            if primary is None:
                raise errors[0]
            for error in errors:
                primary.add_note("Baseline foreground cleanup also failed: " + repr(error))
