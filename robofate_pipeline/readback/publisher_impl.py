"""Original exclusive manifest transaction; metadata supplied by the binder."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import re

class PublishError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise PublishError(message)


def absolute(value):
    require(isinstance(value, (str, Path)) and Path(value).is_absolute(),
            "expected an absolute path")
    return Path(value).resolve()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_ref(value):
    require(isinstance(value, dict), "expected FileRef")
    path = absolute(value.get("path"))
    require(path.is_file(), "missing evidence: " + str(path))
    require(isinstance(value.get("sha256"), str) and
            re.fullmatch(r"[0-9a-f]{64}", value["sha256"]), "invalid FileRef hash")
    return dict(value)


def checked_bytes(ref):
    ref = file_ref(ref)
    data = Path(ref["path"]).read_bytes()
    require(hashlib.sha256(data).hexdigest() == ref["sha256"],
            "FileRef hash mismatch: " + ref["path"])
    return data


def object_from(ref):
    value = json.loads(checked_bytes(ref))
    require(isinstance(value, dict), "expected metadata object: " + ref["path"])
    return value


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def check_identity(identity, source_id, task):
    require(isinstance(identity, dict), "missing source identity")
    for key in ("source_id", "candidate_source_id"):
        if key in identity:
            require(identity[key] == source_id, "source identity mismatch: " + key)
    if "task" in identity:
        require(identity["task"] == task, "source identity task mismatch")


def check_data_binding(report, context, summary, failure, teacher,
                       source_id, task, run, failure_run):
    """Check the same 1.2/1.3 claims against the pinned small raw metadata."""
    require(failure.get("native_success") is False,
            "DM result contradicts native failure")
    require(failure.get("terminated") is True or failure.get("truncated") is True,
            "DM result is not terminal")
    require(teacher.get("native_success") is True and teacher.get("terminated") is True,
            "teacher result is not terminated native success")
    steps, actions = report.get("teacher_steps"), report.get("teacher_actions")
    t1, limit = report["t1"], report["native_limit"]
    require(type(steps) is int and type(teacher.get("native_steps")) is int and
            steps == teacher["native_steps"], "teacher_steps mismatch")
    require(t1 <= steps <= limit, "teacher steps outside takeover/native limit")
    require(type(actions) is int and actions == steps - t1,
            "teacher_actions mismatch: expected steps-t1")
    identity = context.get("identity")
    require(isinstance(identity, dict) and identity.get("source_id") == source_id,
            "context source mismatch")
    require(identity.get("task") == task, "context task mismatch")
    require(absolute(context.get("run_dir")) == run, "context run_dir mismatch")
    for key, expected in (("t1", t1), ("native_limit", limit),
                          ("remaining_steps", limit - t1)):
        require(type(context.get(key)) is int and context[key] == expected,
                "context " + key + " mismatch")
    require(summary.get("source_id") == source_id, "failure_summary source mismatch")
    run_keys = [key for key in ("run", "run_dir", "failure_run") if key in summary]
    require(run_keys, "failure_summary missing run")
    for key in run_keys:
        require(absolute(summary[key]) == failure_run, "failure_summary run mismatch")
    require(encoded(summary.get("native_result")) == encoded(failure),
            "failure_summary result mismatch")


def prepare(report_ref, *, source_id, task, run, failure_run):
    """Validate the actual pinned report and bind evidence to original roles."""
    report_ref = file_ref(report_ref)
    report = object_from(report_ref)
    require(report.get("schema") == "refate.minimum_data_readback.v1",
            "not a minimum data readback report")
    require(report.get("data_status") == "qualified", "data is not qualified")
    conditions = report.get("conditions")
    require(isinstance(conditions, dict), "missing data conditions")
    for key in ("1.1", "1.2", "1.3"):
        require(isinstance(conditions.get(key), dict) and
                conditions[key].get("status") == "passed", "data condition " + key)
    require(report.get("source_id") == source_id, "report source mismatch")
    require(report.get("task") == task, "report task mismatch")
    run, failure_run = absolute(run), absolute(failure_run)
    require(absolute(report.get("run")) == run, "teacher run mismatch")
    require(absolute(report.get("failure_run")) == failure_run, "failure run mismatch")
    require(run != failure_run, "teacher and failure roles must be distinct")
    identity = report.get("source_identity")
    check_identity(identity, source_id, task)
    refs = report.get("evidence_refs")
    require(isinstance(refs, list) and refs, "missing evidence refs")
    by_path = {}
    for value in refs:
        ref = file_ref(value)
        path = absolute(ref["path"])
        require(path not in by_path or by_path[path]["sha256"] == ref["sha256"],
                "conflicting evidence refs")
        by_path[path] = ref

    def role(path):
        require(path in by_path, "missing evidence role: " + str(path))
        return by_path[path]

    evidence = {
        "recording_readback": report_ref,
        "context": file_ref(report.get("context_ref")),
        "failure_summary": file_ref(report.get("failure_summary_ref")),
        "DM_source_result": role(failure_run / "result.json"),
        "teacher_native_result": role(run / "result.json"),
    }
    # Only small role metadata is rehashed/parsed here, never raw trajectories.
    for name in ("context", "failure_summary"):
        ref = evidence[name]
        require(role(absolute(ref["path"]))["sha256"] == ref["sha256"],
                name + " evidence role hash mismatch")
    context = object_from(evidence["context"])
    failure_summary = object_from(evidence["failure_summary"])
    failure_result = object_from(evidence["DM_source_result"])
    teacher_result = object_from(evidence["teacher_native_result"])
    for folder, name in ((run, "source_identity"),
                         (failure_run, "DM_source_identity")):
        path = folder / "source_identity.json"
        if path in by_path:
            original = object_from(by_path[path])
            check_identity(original, source_id, task)
            for key in ("group_id", "student_split"):
                if key in original and key in identity:
                    require(original[key] == identity[key], "source " + key + " mismatch")
            evidence[name] = by_path[path]
    t1, limit = report.get("t1"), report.get("native_limit")
    require(type(t1) is int and type(limit) is int and 0 <= t1 <= limit,
            "invalid shared-prefix/native-limit metadata")
    check_data_binding(report, context, failure_summary, failure_result,
                       teacher_result, source_id, task, run, failure_run)
    teacher = dict(report.get("teacher") or {})
    teacher.update({"run": report["run"], "native_result_ref": evidence["teacher_native_result"],
                    "native_success": teacher_result.get("native_success"),
                    "teacher_steps": report["teacher_steps"],
                    "teacher_actions": report["teacher_actions"]})
    if "identity" in teacher:
        teacher["release_identity"] = teacher["identity"]
    recording = {"integrity_passed": True, "reader_result_retained_unchanged": True,
                 "trajectory_redecoded_by_publisher": False}
    for relative, name in (("native_actions.jsonl", "native_actions_ref"),
                           ("dense/control_steps.jsonl", "dense_actions_ref")):
        if run / relative in by_path:
            recording[name] = by_path[run / relative]
    return {
        "schema": "refate-dual-fate-episode-manifest-v1",
        "source_id": source_id, "task": task, "episode_id": run.name,
        "host": _NODE["host"], "hostname": _NODE["hostname"], "owner_chat": _NODE["owner_chat"],
        "campaign": _NODE["campaign"],
        "qualification_status": "qualified", "qualified": True,
        "qualification_scope": "episode_data_only", "data_status": "qualified",
        "conditions": conditions, "counted_in_quota": False, "source_count_credit": 0,
        "ledger_registration_done": False, "admission": False,
        "clean_eval_eligible": False, "complete_teacher_API_session_qualified": False,
        "source_identity": identity, "group_id": identity.get("group_id"),
        "student_split": identity.get("student_split"),
        "baseline": {"source_run": report["failure_run"],
                     "source_native_success": failure_result.get("native_success"),
                     "source_result_ref": evidence["DM_source_result"]},
        "teacher": teacher,
        "takeover": {"t1_native": t1, "causal_history_native_range": [0, t1],
                     "shared_prefix_source_run": report["failure_run"],
                     "native_total_budget": limit, "context_ref": evidence["context"]},
        "recording": recording, "evidence": evidence, "evidence_refs": refs,
        "diagnostic_limits_preserved": {
            "diagnostics_original": report.get("diagnostics", {}),
            "source_identity_original": identity,
            "DM_result_original": failure_result, "result_original": teacher_result},
        "derived_labels": {"basis": report_ref, "conditions": conditions,
                           "raw_actions_modified": False},
        "original_records_unchanged": True,
        "synthetic_fixture": report.get("synthetic_fixture", False),
        "publication_authority": "minimum data conditions 1.1/1.2/1.3; no quota commit",
        "limitations": ["No API completion, process-exit or bridge-close qualification.",
                        "Raw trajectory refs are carried from the hash-bound reader report; "
                        "the publisher does not rehash or decode full trajectories.",
                        "Core/spare selection and ledger commit remain with main."],
    }


def publish(report_ref, *, manifest_root, source_id, task, run, failure_run):
    require(source_id == _NODE["source_id"], "unexpected source id")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", task), "invalid task path")
    root = absolute(manifest_root)
    target = root / task / _NODE["source_id"]
    require(target.resolve() == target, "symlinked publication destination")
    manifest = prepare(report_ref, source_id=source_id, task=task,
                       run=run, failure_run=failure_run)
    mpath, rpath = target / "manifest.json", target / "publication_receipt.json"
    # Reserve one source slot; interrupted/partial publications fail closed.
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.mkdir()
        reused = False
    except FileExistsError:
        reused = True
    if reused:
        require(mpath.is_file() and rpath.is_file(), "incomplete existing publication")
        existing = json.loads(mpath.read_bytes())
        require(isinstance(existing, dict) and isinstance(existing.get("published_at"), str),
                "conflicting existing manifest")
        manifest["published_at"] = existing["published_at"]
        require(existing == manifest, "conflicting existing manifest; refusing overwrite")
    else:
        manifest["published_at"] = datetime.now(timezone.utc).isoformat()
        with mpath.open("xb") as stream:
            stream.write(encoded(manifest))
            stream.flush()
            os.fsync(stream.fileno())
    receipt = {
        "schema": "refate-dual-fate-publication-receipt-v1",
        "at": manifest["published_at"], "source_id": source_id, "task": task,
        "host": _NODE["host"], "hostname": _NODE["hostname"],
        "campaign": _NODE["campaign"], "episode_id": manifest["episode_id"],
        "owner_chat": manifest["owner_chat"],
        "qualification_status": "qualified", "qualification_scope": "episode_data_only",
        "counted_in_quota": False, "source_count_credit": 0,
        "manifest": {"path": str(mpath), "sha256": digest(mpath)},
        "recording_readback": manifest["evidence"]["recording_readback"],
        "complete_teacher_API_session_qualified": False,
        "remote_readback_verified": True, "ledger_registration_done": False,
        "original_records_unchanged": True,
        "synthetic_fixture": manifest["synthetic_fixture"],
    }
    if reused:
        require(json.loads(rpath.read_bytes()) == receipt,
                "conflicting existing receipt; refusing overwrite")
    else:
        with rpath.open("xb") as stream:
            stream.write(encoded(receipt))
            stream.flush()
            os.fsync(stream.fileno())
    if reused:
        for path in (mpath, rpath):
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    directory_fd = os.open(str(target), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    require(json.loads(mpath.read_bytes()) == manifest and
            json.loads(rpath.read_bytes()) == receipt, "publication readback mismatch")
    return {"reused": reused, "manifest": receipt["manifest"],
            "publication_receipt": {"path": str(rpath), "sha256": digest(rpath)},
            "counted_in_quota": False, "ledger_registration_done": False}
