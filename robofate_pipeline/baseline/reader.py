"""Once-only baseline readback; data status is independent of process exit."""
import datetime
import fcntl
import os
from pathlib import Path
import stat
import time

from .. import evidence
from ..evidence import need
from ..readback.reader import sig
from ..recording import observation_io, pt_capture


def native_label(result):
    need(result.get("status") == "native_terminal"
         and all(type(result.get(k)) is bool for k in ("native_success", "terminated", "truncated")),
         "Real native terminal flags required")
    need(result["terminated"] or result["truncated"], "No native terminal result")
    need(not result["native_success"] or result["terminated"], "Success contradicts native termination")
    return 0 if result["native_success"] else None


def inventory(run):
    out = {}
    for path in sorted(run.rglob("*")):
        name = path.relative_to(run)
        if name.parts[0] == "baseline_readback_001":
            continue
        need(not path.is_symlink(), "Linked baseline evidence")
        if path.is_file():
            out[str(name)] = sig(path)
    return out


def audit(policy, context):
    run = Path(context["run_dir"])
    capture_ref = evidence.file_ref(pt_capture.__file__)
    policy.pin(capture_ref["path"])
    api = policy.load("baseline/reader_impl.py", dict(
        S=policy.root, _NODE={"source_id": context["source_id"], "task": context["task"]},
        _COMMON_OBSERVATION_IO=observation_io, _CAPTURE_REF=capture_ref,
        _CAPTURE_EXPECTED_SHA=policy.pins[capture_ref["path"]]["sha256"],
        _optional_diagnostic=lambda path: dict(path=str(path), present=path.exists(), admission_gate=False)))
    def boundary(folder, observations):
        derived = [dict(native_index=r["index"], observation_dir=r["observation_dir"],
                        terminal=r["terminal"]) for r in observations]
        legacy = folder / "policy/history_updates.jsonl"
        present = legacy.exists() or legacy.is_symlink()
        return (api.rows(api.native25_owned(legacy)) if present else derived), dict(
            derived_from_saved_native_boundaries=True, record_count=len(derived),
            claims_actual_DM_API_calls=False, terminal_policy_receipt_synthesized=False,
            legacy_history_sha256=api.digest(legacy) if present else None)
    api._boundary_index = boundary
    return api.audit_native25(run, api.native25_doc(run / "recording_contract.json"))


def settle(context_ref, policy):
    c = evidence.decode(evidence.read_pin(context_ref, roots=(policy.root,)))
    need(c["data_kind"] == policy.value["data_kind"] and c["campaign"] == policy.value["campaign"]
         and c["execution_owner"] == policy.value["execution_owner"], "Baseline readback scope differs")
    run = Path(c["run_dir"])
    need(run == policy.root / "runs" / c["run_id"] and run.resolve(strict=True) == run, "Wrong baseline run")
    once = run / "baseline_readback_001"
    once.mkdir(exist_ok=True)
    need(once.resolve() == once, "Linked readback directory")
    lock = os.open(once / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock, "a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = dict(context_ref=context_ref, execution_policy_ref=policy.reference)
        if (once / "SUMMARY.json").exists():
            saved = evidence.document(once / "SUMMARY.json")
            need(saved["settlement_binding"] == binding and not (once / "FAILED.json").exists(),
                 "Conflicting prior settlement")
            for ref in (saved["manifest"], saved["files"]["readback.json"],
                        saved["files"]["source_inventory_before.json"]):
                evidence.read_pin(ref)
            old = evidence.decode(evidence.read_pin(saved["files"]["source_inventory_before.json"]))
            for name, signature in old.items():
                if stat.S_ISREG(signature[2]):
                    need(sig(run / name) == signature, "Previously reviewed baseline changed")
            return saved
        need(not (once / "ARMED.json").exists(), "Uncertain readback requires reconciliation, not re-read")
        evidence.exclusive(once / "ARMED.json", dict(binding, automatic_repeat=False))
        start = time.monotonic()
        try:
            result = evidence.document(run / "result.json")
            label = native_label(result)
            identity = evidence.document(run / "source_identity.json")
            need(identity == c["preparation"]["source"] and result["source_id"] == c["source_id"]
                 and result["task"] == c["task"] and result["native_step_limit"] == c["native_total_steps"],
                 "Baseline result/context/source mismatch")
            before = inventory(run)
            report = audit(policy, c)
            need(report["file_recording_readback_passed"] is True and report["result"] == result,
                 "Baseline readback/result mismatch")
            synthetic = c["data_kind"] == "synthetic"
            need(bool(result.get("synthetic_fixture", False)) == synthetic, "Synthetic/actual evidence mixed")
            need(before == inventory(run), "Baseline changed during independent readback")
            report.update(data_kind=c["data_kind"], synthetic_fixture=synthetic)
            report_ref = evidence.exclusive(once / "readback.json", report)
            inventory_ref = evidence.exclusive(once / "source_inventory_before.json", before)
            manifest = dict(schema="refate.common_baseline.source_manifest.v1",
                data_kind=c["data_kind"], source_id=c["source_id"], run_id=c["run_id"],
                task=c["task"], source_root=str(run), campaign=c["campaign"],
                source_owner=c["source_owner"], execution_owner=c["execution_owner"],
                source_group_id=identity["group_id"], split=identity["split"],
                layout_ref=identity["layout_ref"],
                initialization_content_sha256=identity["initialization_content_sha256"],
                context=context_ref, readback_report=report_ref, result=evidence.file_ref(run / "result.json"),
                native_success=result["native_success"], native_steps=result["native_steps"],
                native_total_steps=c["native_total_steps"], actual_controls=report["actual_controls"],
                required_data_readable=True, qualified_positive=False, qualified_negative=label == 0 and not synthetic,
                controller_episode_label=label if not synthetic else None, counted_in_quota=False)
            manifest_ref = evidence.exclusive(run / "SOURCE_MANIFEST_001.json", manifest)
            summary = dict(data_kind=c["data_kind"], source_id=c["source_id"], run_id=c["run_id"], run=str(run),
                settlement_binding=binding, file_recording_readback_passed=True, native_result=result,
                native_success=result["native_success"], native_steps=result["native_steps"],
                actual_controls=report["actual_controls"], qualified_positive=False,
                qualified_negative=label == 0 and not synthetic, counted_in_quota=False,
                decoded_observation_count=report["decoded_observation_count"],
                decoded_RGB_images=report["decoded_RGB_images"], manifest=manifest_ref,
                files={"readback.json": report_ref, "source_inventory_before.json": inventory_ref},
                readback_wall_seconds=time.monotonic() - start,
                finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
            evidence.exclusive(once / "SUMMARY.json", summary)
            return summary
        except BaseException as error:
            try:
                evidence.exclusive(once / "FAILED.json", dict(type=type(error).__name__, message=str(error),
                    qualified=False, automatic_repeat=False, settlement_binding=binding))
            except BaseException as secondary:
                error.add_note("Readback failure receipt also failed: " + repr(secondary))
            raise
