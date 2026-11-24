"""One-shot original readback/publisher composition; no collection or quota grant."""
import datetime
import fcntl
import hashlib
import os
from pathlib import Path
import re
from types import ModuleType

from .. import evidence
from ..evidence import need
from . import reader


def publisher(*, context, config):
    root = Path(config["data_root"])
    published = Path(config["manifest_root"])
    need(published.is_relative_to(root), "Publication must remain in the private data root")
    for name in ("source_id", "task"):
        need(type(context[name]) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,254}", context[name]),
             "Unsafe publication identity component")
    node = dict(config["execution"], source_id=context["source_id"])
    need(set(node) == {"host", "hostname", "owner_chat", "campaign", "source_id"}
         and all(type(v) is str and v for v in node.values()), "Explicit execution metadata required")
    need(node["owner_chat"] == context["execution_owner"] and node["campaign"] == context["campaign"],
         "Execution owner/campaign mismatch")
    path = Path(__file__).with_name("publisher_impl.py")
    pin = config["code_pins"]["publisher_impl.py"]
    need(pin["path"] == str(path), "Wrong publisher implementation")
    raw = evidence.read_pin(pin)
    module = ModuleType(__package__ + "._publisher")
    module.__file__ = str(path)
    module._NODE = node
    exec(compile(raw, str(path), "exec"), module.__dict__)
    original_absolute, original_file_ref = module.absolute, module.file_ref
    def absolute(value):
        path = original_absolute(value)
        module.require(Path(value) == path and path.is_relative_to(root), "Noncanonical or unbound publication path")
        return path
    def file_ref(value):
        module.require(isinstance(value, dict), "Expected evidence reference")
        absolute(value.get("path"))
        return original_file_ref(value)
    module.absolute, module.file_ref = absolute, file_ref
    args = dict(source_id=context["source_id"], task=context["task"],
                run=Path(context["run_dir"]), failure_run=Path(context["source_root"]),
                manifest_root=published)
    return lambda report_ref: module.publish(report_ref, **args)


def settle(context_ref, config_ref):
    """Review exactly once, retaining uncertain/failed attempts without retry.

    Synthetic fixtures can produce test reports, but never publication manifests
    or positive credits. The real reader, not process exit, owns data acceptance.
    """
    config = evidence.decode(evidence.read_pin(config_ref))
    need(config.get("schema") == "robofate.readback-policy.v1"
         and config.get("data_kind") in ("actual", "synthetic"), "Explicit readback policy required")
    root = Path(config["data_root"])
    need(root.is_absolute() and root.resolve(strict=True) == root and root.is_dir(), "Canonical data root required")
    context = evidence.decode(evidence.read_pin(context_ref, roots=(root,)))
    need(context.get("data_kind") == config["data_kind"] and context["source_id"] == context["identity"]["source_id"]
         and context["task"] == context["identity"]["task"], "Readback data kind or identity differs")
    need(context["execution_owner"] == context["identity"]["active_chat_id"]
         == config["execution"]["owner_chat"] and context["campaign"] == config["execution"]["campaign"],
         "Readback execution owner/campaign mismatch")
    code_files = ("reader.py", "decoder.py", "contracts.py", "content.py", "publisher_impl.py", "settlement.py")
    need(set(config["code_pins"]) == set(code_files), "Pin every readback implementation")
    for name in code_files:
        pin = config["code_pins"][name]
        need(pin["path"] == str(Path(__file__).with_name(name)), "Unexpected readback code path")
        evidence.read_pin(pin)
    need(set(config["protocol_refs"]) == {"outline", "directive"}, "Exact protocol document pins required")
    for ref in config["protocol_refs"].values():
        evidence.read_pin(ref)
    run, source = Path(context["run_dir"]), Path(context["source_root"])
    need(run.parent == source.parent == root / "runs" and run != source
         and run.resolve(strict=True) == run and source.resolve(strict=True) == source,
         "Canonical distinct paired runs required")
    once = run / "teacher_readback_001"
    once.mkdir(exist_ok=True)
    need(once.resolve() == once, "Linked settlement folder")
    binding = dict(context_ref=context_ref, config_ref=config_ref)
    lock_fd = os.open(once / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock_fd, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = once / "SUMMARY.json"
        if saved.exists():
            value = evidence.document(saved)
            need(value["settlement_binding"] == binding, "Previous settlement has another binding")
            if value.get("readback_ref"):
                report = evidence.decode(evidence.read_pin(value["readback_ref"], roots=(root,)))
                for path, signature in report["stable_files"].items():
                    need(reader.sig(path) == signature, "Previously reviewed input changed")
                for ref in (value.get("publication") or {}).values():
                    if isinstance(ref, dict) and set(ref) == {"path", "sha256"}:
                        evidence.read_pin(ref, roots=(root,))
            else:
                current = evidence.file_ref(run / "result.json") if (run / "result.json").exists() else None
                need(current == value["native_result_ref"], "Terminal evidence changed after settlement")
            return value
        need(not (once / "ARMED.json").exists(),
             "Uncompleted readback needs reconciliation; no blind retry")
        evidence.exclusive(once / "ARMED.json", binding)
        result_path = run / "result.json"
        result = evidence.document(result_path) if result_path.exists() else None
        base = dict(schema="robofate.teacher-settlement.v1", data_kind=context["data_kind"],
                    source_id=context["source_id"], run_id=run.name,
                    settlement_binding=binding, native_result=result,
                    native_result_ref=evidence.file_ref(result_path) if result else None,
                    ledger_commit_pending=True, counted_in_quota=False,
                    at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        try:
            if not result or result.get("native_success") is not True:
                summary = dict(base, data_status="teacher_not_native_success" if result else "unknown_native_outcome",
                               qualified_positive=False, new_positive=0, new_negative=0,
                               complete_positive_reader_run=False)
            else:
                report = reader.review(reader.Files(root), context_ref, context["source_readback_ref"],
                                       data_root=root,
                                       outline_sha256=config["protocol_refs"]["outline"]["sha256"],
                                       directive_sha256=config["protocol_refs"]["directive"]["sha256"])
                report["synthetic_fixture"] = context["data_kind"] == "synthetic"
                report["data_kind"] = context["data_kind"]
                report["transport"] = "in_process_original_Model_with_original_native_persistence"
                report_ref = evidence.exclusive(once / "REPORT.json", report)
                synthetic = report["synthetic_fixture"]
                published = None if synthetic else publisher(context=context, config=config)(report_ref)
                summary = dict(base, data_status="synthetic_validation_only" if synthetic else "qualified",
                               qualified_positive=not synthetic, new_positive=0 if synthetic else 1, new_negative=0,
                               readback_ref=report_ref, publication=published, complete_positive_reader_run=True)
            evidence.exclusive(saved, summary)
            return summary
        except BaseException as error:
            try:
                evidence.exclusive(once / "FAILED.json", dict(
                    error_type=type(error).__name__, error=str(error), qualified=False,
                    data_status="pending_review", automatic_repeat=False, settlement_binding=binding))
            except BaseException as secondary:
                error.add_note("Failure receipt also failed: " + repr(secondary))
            raise
