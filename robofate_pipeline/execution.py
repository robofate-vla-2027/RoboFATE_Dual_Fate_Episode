"""One reserved source to original Native/teacher/result, without source allocation.

External admission is mandatory. This is not the campaign scheduler or baseline
collector. Saved-data acceptance and the ledger remain separate from this entry.
"""
import datetime
import os
from pathlib import Path
import re
import signal
import sys
import time
import traceback

from . import collection, evidence
from .continuation import SourceContinuation
from .evidence import canonical, need
from .native import NativeRuntime
from .native.snapshot import runtime_snapshot
from .native.resources import inspect as inspect_resources
from .native import deployment
from .recording import common, core
from .restoration import RestoreBinding
from .runtime_policy import RuntimePolicy
from .teacher import install


def _dispatch(reference, policy):
    value = evidence.decode(evidence.read_pin(reference, roots=(policy.root,)))
    p = policy.value
    need(value.get("schema") == "robofate.teacher-dispatch.v1" and value.get("data_kind") == "actual"
         and p["data_kind"] == "actual" and value["execution_owner"] == p["execution_owner"]
         and value["campaign"] == p["campaign"], "Actual source-owner dispatch required")
    need(value["source_reserved"] is True and type(value["positive_slots"]) is int
         and value["positive_slots"] == 1 and type(value["negative_slots"]) is int
         and value["negative_slots"] == 0, "One independently reserved positive slot required")
    for key in ("run_id", "session_id"):
        need(type(value[key]) is str and re.fullmatch(r"[0-9]{12}_[A-Za-z0-9_-]{1,60}", value[key]),
             "Fresh bounded run/session token required")
    expiry = datetime.datetime.fromisoformat(value["valid_until"])
    need(expiry.tzinfo is not None and datetime.datetime.now(datetime.timezone.utc) < expiry,
         "Source dispatch expired or lacks timezone")
    need(type(value["ledger_revision"]) is int and value["ledger_revision"] > 0, "Invalid ledger revision")
    need(value["task"] in p["task_totals"] and type(value["native_total_steps"]) is int
         and value["native_total_steps"] == p["task_totals"][value["task"]]
         and type(value["t1"]) is int and 0 < value["t1"] < value["native_total_steps"]
         and value["source_id"] not in p["excluded_source_ids"], "Source dispatch budget/exclusion mismatch")
    return value


def prepare_teacher(dispatch_ref, restore_policy_ref, execution_policy_ref):
    """Stage original inputs and pinned context only. No Native or worker launch."""
    policy = RuntimePolicy(execution_policy_ref)
    dispatch = _dispatch(dispatch_ref, policy)
    restore = RestoreBinding(restore_policy_ref)
    need(restore.root == policy.root and restore.policy["data_kind"] == "actual"
         and restore.policy["execution_owner"] == policy.value["execution_owner"]
         and restore.policy["partition"] == policy.value["chat"],
         "Restore and runtime scopes differ")
    material = restore.source_material(dispatch["source_context_ref"], dispatch["source_manifest_ref"],
                                       dispatch["source_readback_ref"], t1=dispatch["t1"])
    need(material["source_id"] == dispatch["source_id"] and material["task"] == dispatch["task"]
         and material["native_limit"] == dispatch["native_total_steps"], "Dispatch/source mismatch")
    NativeRuntime(execution_policy_ref).layout(material["source_preparation"])
    run = policy.root / "runs" / dispatch["run_id"]
    session = policy.root / "sessions" / dispatch["session_id"]
    need(run.is_dir() and run.resolve() == run and not any(run.iterdir()) and not session.exists(),
         "Run must be newly reserved and empty; session must not exist")
    evidence.exclusive(run / "PREPARATION_ARMED.json", dict(
        dispatch_ref=dispatch_ref, restore_policy_ref=restore_policy_ref,
        execution_policy_ref=execution_policy_ref, automatic_repeat=False))
    identity = dict(campaign=dispatch["campaign"], chat=policy.value["chat"],
        active_chat_id=dispatch["execution_owner"], task=material["task"], source_id=material["source_id"],
        owner=material["source_owner"], session=dispatch["session_id"], epoch=dispatch["ledger_revision"])
    prep = restore.prepare_context(material, run=run, owner=identity,
                                   core_modules=collection.recording_modules(), enabled=True)
    runtime_binding = dict(schema="robofate.runtime.v1", data_kind="actual",
        execution_policy_ref=execution_policy_ref, execution_owner=dispatch["execution_owner"],
        campaign=dispatch["campaign"], task=material["task"], native_limit=material["native_limit"],
        t1=material["t1"], actor_wall_s=3600, source_id=material["source_id"],
        session=identity["session"], run_dir=str(session), identity=identity,
        source_context_ref=material["source_context"], runtime_files=policy.manifest["files"],
        source_commit=dispatch["source_commit"])
    runtime = install(runtime_binding)
    from orchestrator.config import config_from_dict
    from orchestrator.session_runner import prepare_inputs
    cfg = config_from_dict(runtime.task_config())
    for path in cfg.workspace.rglob("*"):
        if path.is_file():
            policy.pin(path)
    session.mkdir()
    prepare_inputs(cfg, session)
    main_ref = evidence.exclusive(run / "MAIN_BINDING_001.json", dict(schema="robofate.main-binding.v1",
        identity=identity, source_context_ref=material["source_context"], native_used=material["t1"],
        native_limit=material["native_limit"], remaining_steps=material["native_limit"] - material["t1"]))
    context = dict(schema="robofate.teacher-context.v1", data_kind="actual",
        execution_owner=dispatch["execution_owner"], campaign=dispatch["campaign"],
        run_id=run.name, run_dir=str(run), source_root=material["source"], source_id=material["source_id"],
        source_owner=material["source_owner"], task=material["task"], native_limit=material["native_limit"],
        t1=material["t1"], remaining_steps=material["native_limit"] - material["t1"],
        actor_wall_s=3600, native_wall_seconds=3600, identity=identity, dispatch_ref=dispatch_ref,
        execution_policy_ref=execution_policy_ref, restore_policy_ref=restore_policy_ref,
        runtime_binding=runtime_binding, restore_preparation=prep,
        source_context_ref=material["source_context"], source_manifest_ref=material["source_manifest"],
        source_readback_ref=material["source_readback"], main_binding_ref=main_ref,
        outline_sha256=dispatch["outline_sha256"], directive_sha256=dispatch["directive_sha256"])
    reference = evidence.exclusive(run / "CONTEXT_001.json", context)
    load_context(reference)
    evidence.exclusive(run / "STATIC_PRELAUNCH_001.json", dict(context_ref=reference,
        original_inputs_prepared=True, actual_Native_started=False, actual_teacher_started=False,
        qualified=False, ledger_commit_pending=True))
    return reference


def load_context(reference):
    c = evidence.decode(evidence.read_pin(reference))
    need(c.get("schema") == "robofate.teacher-context.v1" and c.get("data_kind") == "actual",
         "Actual prepared teacher context required")
    policy = RuntimePolicy(c["execution_policy_ref"])
    dispatch = _dispatch(c["dispatch_ref"], policy)
    run = Path(c["run_dir"])
    need(run == policy.root / "runs" / c["run_id"] and run.resolve(strict=True) == run
         and Path(reference["path"]) == run / "CONTEXT_001.json", "Wrong teacher context path")
    for key, expected in (("run_id", dispatch["run_id"]), ("source_id", dispatch["source_id"]),
                          ("task", dispatch["task"]), ("t1", dispatch["t1"]),
                          ("native_limit", dispatch["native_total_steps"]),
                          ("execution_owner", dispatch["execution_owner"]), ("campaign", dispatch["campaign"])):
        need(type(c[key]) is type(expected) and c[key] == expected, "Prepared dispatch drift: " + key)
    need(c["identity"] == c["runtime_binding"]["identity"] == c["restore_preparation"]["owner"]
         and c["runtime_binding"]["execution_policy_ref"] == c["execution_policy_ref"]
         and c["actor_wall_s"] == c["native_wall_seconds"] == 3600
         and c["remaining_steps"] == c["native_limit"] - c["t1"], "Teacher context budget/identity drift")
    restore = RestoreBinding(c["restore_policy_ref"])
    need(restore.root == policy.root and restore.policy["execution_owner"] == c["execution_owner"],
         "Restore/runtime root or owner differs")
    material = restore.source_material(c["source_context_ref"], c["source_manifest_ref"],
                                       c["source_readback_ref"], t1=c["t1"])
    need(material["source_id"] == c["source_id"] and material["task"] == c["task"]
         and material["native_limit"] == c["native_limit"]
         and material["source"] == c["source_root"], "Prepared source changed")
    NativeRuntime(c["execution_policy_ref"]).layout(material["source_preparation"])
    return c, material, restore, policy


def run_teacher(reference, *, lease_ref=None, lock_fd=None):
    """Run once in the externally guarded Native child, preserving failures."""
    context, material, restore, policy = load_context(reference)
    policy.host_guard()
    need(policy.value.get("execution_enabled") is True, "Explicit execution enable required")
    run = Path(context["run_dir"])
    from .foreground import validate_lease, parent_death_guard
    need(lease_ref is not None and lock_fd is not None,
         "Native child requires an explicit owned foreground lease")
    inherited = validate_lease(policy, reference, lease_ref, lock_fd)
    parent_death_guard(inherited["parent"])
    try:
        deployment.verify_environment(policy, context)
        return _run_teacher(reference, context, material, restore, policy, run)
    finally:
        os.close(lock_fd)


def _run_teacher(reference, context, material, restore, policy, run):
    need(not any((run / name).exists() or (run / name).is_symlink()
                 for name in ("native_started.json", "result.json", "failure.json", "dense")),
         "Executed or uncertain context requires reconciliation, not rerun")
    selection = inspect_resources(policy, run)
    NativeRuntime(context["execution_policy_ref"]).device_guard()
    evidence.exclusive(run / "native_resource_selection.json", selection)
    runtime = install(context["runtime_binding"])
    runtime.execution_guard()
    task = recorder = model = continuation = None
    timing, started = {}, time.monotonic()
    last_stage = "before_native"
    def mark(name):
        nonlocal last_stage
        last_stage = name
        value = dict(stage=name, wall_ns=time.time_ns(), elapsed_seconds=time.monotonic() - started)
        timing[name] = value
        evidence.exclusive(run / (name + ".json"), value)
    def interrupted(sig, frame):
        raise KeyboardInterrupt("Foreground signal " + str(sig))
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        evidence.exclusive(run / "native_started.json", dict(pid=os.getpid(), wall_ns=time.time_ns(), context_ref=reference))
        evidence.exclusive(run / "source_identity.json", material["source_identity"])
        task = NativeRuntime(context["execution_policy_ref"]).create(
            material["source_preparation"], run, source_context_ref=context["source_context_ref"])
        recorder, publish = common.initialize_recording(
            task, run, core_modules=collection.recording_modules(), t1=context["t1"],
            runtime_capture=runtime_snapshot, renderer_device=policy.value["native_devices"]["renderer_index"])
        continuation = SourceContinuation(restore, material, context["restore_preparation"],
            task=task, recorder=recorder, publish=publish, identity=context["identity"])
        mark("initial_boundary_saved")
        continuation.restore()
        mark("actual_t1_verified")
        teacher = continuation.create_teacher(runtime, main_binding_ref=context["main_binding_ref"])
        model = teacher.model
        mark("teacher_ready")
        continuation.run()
        result = terminal_result(task, recorder, context)
        evidence.exclusive(run / "result.json", result)
        mark("episode_data_committed")
        recorder.close()
        return result
    except BaseException as primary:
        if recorder is not None:
            try:
                recorder.writer.fail(primary)
            except BaseException as secondary:
                primary.add_note("Writer latch also failed: " + repr(secondary))
        try:
            evidence.exclusive(run / "failure.json", dict(type=type(primary).__name__, message=str(primary),
                traceback=traceback.format_exc(), source_id=context["source_id"], run_id=context["run_id"],
                stage=last_stage, native_used=task.control_steps if task else None,
                actual_control_steps=recorder.control_index if recorder else None,
                provider_retry_allowed=False, qualified=False))
        except BaseException as secondary:
            primary.add_note("Failure receipt also failed: " + repr(secondary))
        raise
    finally:
        primary = sys.exception()
        errors = cleanup(recorder, model, task, run, timing, started)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if errors:
            if primary is None:
                raise errors[0]
            for error in errors:
                primary.add_note("Cleanup also failed: " + repr(error))


def terminal_result(task, recorder, context):
    core.wait_for_native_publication(recorder, task.control_steps)
    need(recorder.control_index == 10 * task.control_steps
         and task.native.step_lim == context["native_limit"], "Terminal count/budget mismatch")
    return dict(status="native_terminal" if task.done else "teacher_finished_without_native_terminal",
        native_success=bool(task.success), terminated=bool(task.terminated), truncated=bool(task.truncated),
        native_steps=task.control_steps, native_step_limit=task.native.step_lim,
        actual_control_steps=recorder.control_index, rtx_profile="pt_spatial4_collection_candidate_v1",
        mode="EmbodiedEvo_teacher_suffix", source_id=context["source_id"], task=context["task"],
        teacher_actions=task.control_steps - context["t1"], qualified=False, counted_in_quota=False)


def cleanup(recorder, model, task, run, timing, started):
    """Only caller-owned objects; cleanup failure never changes a data outcome."""
    errors = []
    if recorder is not None and not recorder.closed_attempted:
        try:
            recorder.close()
        except BaseException as error:
            errors.append(error)
    if model is not None:
        try:
            worker = model.worker
            model.close()
            if worker is not None:
                worker.join(timeout=35)
                need(not worker.is_alive(), "Owned worker cleanup incomplete")
                evidence.exclusive(run / "teacher_worker_wait.json", dict(
                    pid=worker.pid, exitcode=worker.exitcode,
                    original_process_absent=not Path(f"/proc/{worker.pid}").exists(),
                    data_status_not_decided_by_exit=True))
        except BaseException as error:
            errors.append(error)
    try:
        evidence.exclusive(run / "timing_summary.json", dict(stages=timing,
            total_seconds=time.monotonic() - started,
            writer_metrics=recorder.writer.metrics if recorder else None,
            main_phase_seconds=recorder.phases if recorder else None,
            overlap_note="Overlapped phases are not summed", cleanup_errors=[repr(e) for e in errors]))
    except BaseException as error:
        errors.append(error)
    if task is not None:
        try:
            task.close()
        except BaseException as error:
            errors.append(error)
    return errors
