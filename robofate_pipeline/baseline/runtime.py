"""Fresh reserved baseline through the original Native, recorder and OpenDM loop."""
import datetime
import importlib
import os
from pathlib import Path
import signal
import sys
import time
from types import ModuleType
from urllib.parse import urlsplit

from .. import collection, evidence
from ..campaign.sources import SourceCatalog, PROFILE
from ..evidence import need
from ..execution import cleanup
from ..native import NativeRuntime, deployment, transport_rng
from ..native.resources import inspect as inspect_resources
from ..native.snapshot import runtime_snapshot
from ..recording import common, core
from ..runtime_policy import RuntimePolicy


def load_context(reference):
    c = evidence.decode(evidence.read_pin(reference))
    policy = RuntimePolicy(c["execution_policy_ref"])
    p = policy.value
    need(c["schema"] == "robofate.baseline-context.v1" and c["data_kind"] == p["data_kind"]
         and c["execution_owner"] == p["execution_owner"] and c["campaign"] == p["campaign"],
         "Bound baseline context required")
    run = policy.root / "runs" / c["run_id"]
    need(c["run_dir"] == str(run) and run.resolve(strict=True) == run
         and reference["path"] == str(run / "CONTEXT_001.json"), "Wrong baseline context path")
    d = evidence.decode(evidence.read_pin(c["dispatch_ref"], roots=(policy.root,)))
    need(d["source_reserved"] is True and d["negative_slots"] == 1 and d["positive_slots"] == 0
         and d["source_started"] is False and d["execution_owner"] == p["execution_owner"]
         and d["campaign"] == p["campaign"], "Fresh baseline reservation required")
    for key in ("source_id", "run_id", "task", "native_total_steps", "native_wall_seconds"):
        need(c[key] == d[key] and type(c[key]) is type(d[key]), "Baseline dispatch drift")
    expiry = datetime.datetime.fromisoformat(d["valid_until"])
    need(expiry.tzinfo is not None and datetime.datetime.now(datetime.timezone.utc) < expiry, "Expired baseline dispatch")
    need(type(c["native_wall_seconds"]) is int and 900 < c["native_wall_seconds"] <= 3600,
         "Explicit original Native wall allowance required")
    prepared = SourceCatalog(policy).validate(c["preparation"])
    s = prepared["source"]
    need(s["source_id"] == c["source_id"] and s["task"] == c["task"]
         and prepared["native_step_limit"] == c["native_total_steps"], "Baseline source/budget mismatch")
    c["self_ref"] = reference
    return c, policy


def validate_ready(value, context, policy, owned_pid=None):
    need(value["execution_policy_ref"] == policy.reference and value["context_ref"] == context["self_ref"]
         and value["alias"] == policy.value["ssh_alias"] and value["hostname"] == policy.value["expected_hostname"]
         and value["physical_gpu"] == policy.value["native_devices"]["physical_gpu_ids"][0]
         and value["trainable_parameters"] == 0 and value["all_parameters_frozen"] is True
         and value["all_parameters_device"] == "cuda:0" and value["all_parameters_dtype"] == "torch.float32"
         and value["action_shape"] == [25, 14] and value["prediction_shape"] == [50, 14]
         and value["rng_reset_after_load"] is False, "Wrong frozen policy readiness")
    if owned_pid is not None:
        need(value["pid"] == owned_pid, "Readiness belongs to another policy process")
    endpoint = urlsplit(value["url"])
    need(endpoint.scheme == "ws" and endpoint.hostname == "127.0.0.1"
         and endpoint.port and not endpoint.username and not endpoint.password
         and endpoint.path in ("", "/") and not endpoint.query and not endpoint.fragment,
         "Policy must use its owned loopback websocket")


def evaluator(policy):
    path = policy.native / "XPolicyLab/policy/OpenDM/deploy.py"
    module = ModuleType("_robofate_original_dm_deploy")
    exec(compile(policy.pin(path), str(path), "exec"), module.__dict__)
    return module.eval_one_episode


def run(reference, *, lease_ref, lock_fd):
    from ..foreground import validate_lease, parent_death_guard
    c, policy = load_context(reference)
    policy.host_guard()
    inherited = validate_lease(policy, reference, lease_ref, lock_fd)
    parent_death_guard(inherited["parent"])
    try:
        deployment.verify_environment(policy, dict(c, source_context_ref=reference, native_limit=c["native_total_steps"]))
        return _run(reference, c, policy)
    finally:
        os.close(lock_fd)


def _run(reference, c, policy):
    run_dir = Path(c["run_dir"])
    need(not any((run_dir / n).exists() for n in ("native_started.json", "result.json", "failure.json", "dense")),
         "Executed/uncertain baseline cannot repeat")
    task = recorder = client = None
    started = time.monotonic()
    previous = {}
    try:
        def interrupted(sig, frame):
            raise KeyboardInterrupt("Owned baseline signal " + str(sig))
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[sig] = signal.signal(sig, interrupted)
        evidence.exclusive(run_dir / "native_started.json", dict(
            pid=os.getpid(), wall_ns=time.time_ns(), context_ref=reference))
        evidence.exclusive(run_dir / "source_identity.json", c["preparation"]["source"])
        evidence.exclusive(run_dir / "native_resource_selection.json", inspect_resources(policy, run_dir))
        connection = importlib.import_module("websockets.sync.connection")
        pin = policy.value["frozen_dm"]["transport_ref"]
        need(Path(connection.__file__).resolve() == Path(pin["path"]), "Transport module path differs")
        policy.pin(pin["path"])
        evidence.exclusive(run_dir / "transport_rng_isolation.json", transport_rng.install(connection, pin["sha256"]))
        task = NativeRuntime(policy.reference).create(c["preparation"], run_dir, source_context_ref=reference)
        recorder, publish = common.initialize_recording(task, run_dir, core_modules=collection.recording_modules(),
            runtime_capture=runtime_snapshot, renderer_device=policy.value["native_devices"]["renderer_index"])
        evidence.exclusive(run_dir / "native_boundary_ready.json", dict(native_used=task.control_steps,
            control_ticks=recorder.control_index, wall_ns=time.time_ns(),
            phase="actual_initial_boundary_saved_before_policy_load"))
        ready_path = run_dir / "server/model_ready.json"
        deadline = time.monotonic() + 900
        while not ready_path.exists():
            recorder.writer.check()
            if time.monotonic() >= deadline:
                raise TimeoutError("Original 900-second DM loading allowance exhausted")
            time.sleep(.25)
        ready = evidence.document(ready_path)
        validate_ready(ready, c, policy)
        from client_server.ws.model_client import WsModelClient
        client = WsModelClient(url=ready["url"], evaluation_id=c["run_id"], trial_id=c["run_id"],
            ws_ping_interval_s=120, ws_ping_timeout_s=120, max_connect_attempts=1)
        evidence.exclusive(run_dir / "ready.json", dict(native_used=0, native_limit=task.native.step_lim,
            native_step_limit=task.native.step_lim, rtx_profile=PROFILE, actual_shared_recorder=True, qualified=False))
        collection.run_baseline(task, client, evaluator(policy), recorder, publish, run_id=c["run_id"],
                                admission=policy.host_guard)
        core.wait_for_native_publication(recorder, task.control_steps)
        result = dict(status="native_terminal", native_success=bool(task.success),
            terminated=bool(task.terminated), truncated=bool(task.truncated), native_steps=task.control_steps,
            native_step_limit=task.native.step_lim, actual_control_steps=recorder.control_index,
            rtx_profile=PROFILE, mode="independent_baseline", source_id=c["source_id"], task=c["task"],
            qualified=False, counted_in_quota=False)
        evidence.exclusive(run_dir / "result.json", result)
        recorder.close()
        return result
    except BaseException as primary:
        if recorder is not None:
            recorder.writer.fail(primary)
        try:
            evidence.exclusive(run_dir / "failure.json", dict(type=type(primary).__name__, message=str(primary),
                source_id=c["source_id"], run_id=c["run_id"], native_failure_proven=False,
                automatic_repeat=False, qualified=False))
        except BaseException as secondary:
            primary.add_note("Baseline failure receipt also failed: " + repr(secondary))
        raise
    finally:
        primary = sys.exception()
        errors = []
        if client is not None:
            try:
                client.close()
            except BaseException as error:
                errors.append(error)
        errors += cleanup(recorder, None, task, run_dir, {}, started)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if errors:
            if primary is None:
                raise errors[0]
            for error in errors:
                primary.add_note("Baseline cleanup also failed: " + repr(error))
