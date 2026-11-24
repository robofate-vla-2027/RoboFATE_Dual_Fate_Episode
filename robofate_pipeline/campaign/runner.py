"""One explicit foreground plan; no background scheduler, fallback or actor retry."""
from pathlib import Path

from .. import evidence, execution, foreground
from ..baseline import foreground as baseline_parent, reader as baseline_reader
from ..evidence import need
from ..native.resources import inspect as inspect_resources
from ..readback.settlement import settle as settle_teacher
from ..runtime_policy import RuntimePolicy
from .ledger import Ledger, now, run_name


class Campaign:
    def __init__(self, execution_policy_ref):
        self.policy = RuntimePolicy(execution_policy_ref)
        self.ledger = Ledger(self.policy)

    def cleanup_complete(self, context):
        """Resource evidence gates another launch, never saved-data qualification."""
        self.policy.host_guard()
        run = Path(context["run_dir"])
        path = run / "process_cleanup.json"
        if not path.exists():
            return False
        receipt = evidence.document(path)
        if receipt.get("owned_cleanup_complete") is not True:
            return False
        for role in ("native", "policy"):
            if role in receipt:
                row = receipt[role]
                if row["original_process_absent"] is not True or row.get("process_group_remaining") != []:
                    return False
                if not foreground.original_absent(row["identity"]) or foreground.group_members(row["identity"]):
                    return False
        if context["schema"] == "robofate.teacher-context.v1":
            session = Path(context["runtime_binding"]["run_dir"])
            if (session / "oci").exists() or (session / "agent").exists():
                lifecycle = session / "oci/LIFECYCLE.json"
                worker = run / "teacher_worker_wait.json"
                if not lifecycle.exists() or not worker.exists():
                    return False
                life = evidence.document(lifecycle)
                waited = evidence.document(worker)
                if (life["owned_cleanup_complete"] is not True or life["run_root"] != str(session)
                        or waited["original_process_absent"] is not True):
                    return False
        return True

    def prepare_baseline(self, entry):
        """Source reservation and context creation, before any simulator starts."""
        self.policy.host_guard()
        run_id = entry["baseline_run_id"]
        need(run_name(run_id), "Explicit baseline run token required")
        run = self.policy.root / "runs" / run_id
        run.mkdir(mode=0o700)
        self.ledger.select_task(entry["task"], run_id + "_task")
        prepared = self.ledger.catalog.prepare(entry["task"], entry["layout_id"], entry["split_ref"])
        prep_ref = evidence.exclusive(run / "PREPARATION.json", prepared)
        observation = dict(data_kind=self.policy.value["data_kind"], execution_owner=self.policy.value["execution_owner"],
            execution_policy_ref=self.policy.reference, at=now(), run_id=run_id,
            inspection=inspect_resources(self.policy, run))
        resource_ref = evidence.exclusive(run / "RESOURCES.json", observation)
        dispatch = self.ledger.reserve_baseline(prep_ref, run_id, resource_ref, run_id + "_reserved")
        dispatch_ref = evidence.exclusive(run / "DISPATCH.json", dispatch)
        context = dict(schema="robofate.baseline-context.v1", data_kind="actual",
            campaign=dispatch["campaign"], execution_owner=dispatch["execution_owner"],
            source_owner=prepared["source"]["source_owner"], source_id=dispatch["source_id"], task=entry["task"],
            run_id=run_id, run_dir=str(run), native_total_steps=dispatch["native_total_steps"],
            native_wall_seconds=dispatch["native_wall_seconds"], preparation=prepared, preparation_ref=prep_ref,
            execution_policy_ref=self.policy.reference, dispatch_ref=dispatch_ref,
            native_kit_extensions=["isaacsim.asset.gen.conveyor"] if entry["task"] == "pick_from_conveyor_by_image" else [])
        if context["native_kit_extensions"]:
            context["native_extension_pins"] = self.policy.value["deployment"]["conveyor"]["pins_ref"]
        reference = evidence.exclusive(run / "CONTEXT_001.json", context)
        self.ledger.attach_context(reference, teacher=False, operation_id=run_id + "_armed")
        return reference

    def prepare_teacher(self, baseline_ref, entry, *, restore_template_ref, readback_policy_ref):
        self.policy.host_guard()
        base = evidence.decode(evidence.read_pin(baseline_ref))
        config = evidence.decode(evidence.read_pin(readback_policy_ref))
        need(config["data_kind"] == "actual" and config["data_root"] == str(self.policy.root)
             and config["execution"]["owner_chat"] == self.policy.value["execution_owner"]
             and config["execution"]["campaign"] == self.policy.value["campaign"], "Readback policy scope differs")
        protocol = dict(source_commit=self.ledger.config["source_commit"],
            outline_sha256=config["protocol_refs"]["outline"]["sha256"],
            directive_sha256=config["protocol_refs"]["directive"]["sha256"])
        for ref in config["protocol_refs"].values():
            evidence.read_pin(ref)
        dispatch = self.ledger.reserve_teacher(base["source_id"], entry["teacher_run_id"], entry["session_id"],
            entry["t1"], entry["teacher_run_id"] + "_reserved", protocol=protocol)
        config_root = self.policy.directory(self.ledger.config["config_root"])
        private = config_root / entry["teacher_run_id"]
        private.mkdir(mode=0o700)
        admission = dict(schema="robofate.source-admission.v1", data_kind="actual",
            execution_owner=self.policy.value["execution_owner"], sources={base["source_id"]: dict(
                source_id=base["source_id"], task=base["task"], status="reserved", consumed=False, refused=False,
                context_filename=Path(baseline_ref["path"]).name, context_ref=baseline_ref,
                manifest_ref=dispatch["source_manifest_ref"], readback_ref=dispatch["source_readback_ref"])})
        admission_ref = evidence.exclusive(private / "ADMISSION.json", admission)
        template = evidence.decode(evidence.read_pin(restore_template_ref))
        need(template["data_kind"] == "actual" and template["execution_owner"] == self.policy.value["execution_owner"]
             and template["data_root"] == str(self.policy.root) and template["config_root"] == str(config_root),
             "Restore template scope differs")
        template["admission_ref"] = admission_ref
        restore_ref = evidence.exclusive(private / "RESTORE_POLICY.json", template)
        # prepare_teacher requires an empty run. Keep source-owner dispatch outside it.
        dispatch_ref = evidence.exclusive(private / "DISPATCH.json", dispatch)
        run = self.policy.root / "runs" / entry["teacher_run_id"]
        run.mkdir(mode=0o700)
        # The dispatch is evidence scoped to data_root by the execution contract.
        need(private.is_relative_to(self.policy.root), "Campaign config_root must be inside the private data root")
        reference = execution.prepare_teacher(dispatch_ref, restore_ref, self.policy.reference)
        self.ledger.attach_context(reference, teacher=True, operation_id=entry["teacher_run_id"] + "_armed")
        return reference

    def baseline(self, context_ref):
        context = evidence.decode(evidence.read_pin(context_ref))
        primary, committed = None, False
        try:
            baseline_parent.foreground(context_ref, on_started=self.ledger.started)
        except BaseException as error:
            primary = error
        try:
            summary = baseline_reader.settle(context_ref, self.policy)
            summary_ref = evidence.file_ref(Path(context["run_dir"]) / "baseline_readback_001/SUMMARY.json")
            safe = self.cleanup_complete(context)
            receipt = self.ledger.settle_baseline(context_ref, summary_ref, safe=safe)
            committed = True
            need(safe, "Baseline data settled but owned resources are not confirmed clean")
        except BaseException as secondary:
            run = Path(context["run_dir"])
            for path in (run / "baseline_readback_001/FAILED.json", run / "failure.json", run / "foreground_error.json"):
                if not committed and path.exists():
                    try:
                        self.ledger.close_failed_baseline(context_ref, evidence.file_ref(path),
                            safe=self.cleanup_complete(context))
                    except BaseException as closure:
                        secondary.add_note("Failed-source quarantine also failed: " + repr(closure))
                    break
            if primary is not None:
                primary.add_note("Independent baseline settlement also failed: " + repr(secondary))
                raise primary
            raise
        if primary is not None:
            raise primary
        return summary, receipt

    def confirm_cleanup(self, context_ref):
        c = evidence.decode(evidence.read_pin(context_ref))
        need(self.cleanup_complete(c), "Owned resource cleanup is not currently verified")
        return self.ledger.confirm_cleanup(context_ref, evidence.file_ref(Path(c["run_dir"]) / "process_cleanup.json"))

    def teacher(self, context_ref, readback_policy_ref):
        context = evidence.decode(evidence.read_pin(context_ref))
        primary = None
        try:
            foreground.foreground(context_ref, on_started=self.ledger.started)
        except BaseException as error:
            primary = error
        try:
            summary = settle_teacher(context_ref, readback_policy_ref)
            summary_ref = evidence.file_ref(Path(context["run_dir"]) / "teacher_readback_001/SUMMARY.json")
            safe = self.cleanup_complete(context)
            receipt = self.ledger.settle_teacher(context_ref, summary_ref, safe=safe)
            need(safe, "Teacher data settled but owned resources are not confirmed clean")
        except BaseException as secondary:
            if primary is not None:
                primary.add_note("Independent teacher settlement also failed: " + repr(secondary))
                raise primary
            raise
        if primary is not None:
            raise primary
        return summary, receipt

    def run_plan(self, reference):
        """Explicit finite sequence; stop on any uncertainty, never auto-resume."""
        plan = evidence.decode(evidence.read_pin(reference))
        need(plan["schema"] == "robofate.campaign-plan.v1" and plan["data_kind"] == "actual"
             and plan["execution_policy_ref"] == self.policy.reference, "Actual pinned plan required")
        self.policy.host_guard()
        need(type(plan["entries"]) is list and plan["entries"], "Nonempty finite source plan required")
        results = []
        for entry in plan["entries"]:
            base_ref = self.prepare_baseline(entry)
            baseline, commit = self.baseline(base_ref)
            result = dict(baseline_context=base_ref, baseline_commit=commit, teacher=None)
            if baseline["native_success"] is False and baseline["native_steps"] > entry["t1"]:
                teacher_ref = self.prepare_teacher(base_ref, entry,
                    restore_template_ref=plan["restore_template_ref"], readback_policy_ref=plan["readback_policy_ref"])
                summary, receipt = self.teacher(teacher_ref, plan["readback_policy_ref"])
                result["teacher"] = dict(context=teacher_ref, commit=receipt, data_status=summary["data_status"])
            results.append(result)
        return dict(data_kind="actual", sources=results, automatic_repeat=False)
