"""Private temporary synthetic source catalog and ledger, never a live campaign."""
import copy
import hashlib
from pathlib import Path

from robofate_pipeline import evidence
from robofate_pipeline.campaign.family import family
from robofate_pipeline.campaign.ledger import Ledger, now
from robofate_pipeline.campaign.sources import SourceCatalog
from robofate_pipeline.runtime_policy import RuntimePolicy
from synthetic_runtime import RuntimeFixture


class CampaignFixture(RuntimeFixture):
    def __init__(self, *, mixed=False):
        super().__init__()
        self.native = self.root / "native"
        task_dir = self.native / "task/RoboDojo/tasks"
        task_dir.mkdir(parents=True)
        task = task_dir / "synthetic_task.py"
        task.write_text("class SyntheticTask:\n    def __init__(self):\n        self.step_lim = 4\n")
        layouts = self.native / "Assets/Eval_Layout/RoboDojo/arx_x5/0"
        layouts.mkdir(parents=True)
        rows, refs = {}, []
        for i in range(3):
            value = {"Rigid": {"synthetic": [{"label": "SYNTHETIC_OBJECT",
                "default_pos": [i, 0, 0], "default_ori": [1, 0, 0, 0], "SYNTHETIC_unknown_field": 7}]},
                "Geometry": {}}
            ref = evidence.exclusive(layouts / f"synthetic_task_{i}.json", value)
            refs.append(ref)
            gid = "synthetic_task/task_content_family/" + family(value)["content_sha256"]
            rows[str(i)] = dict(layout_id=i, native_bank=0, layout_ref=ref,
                initialization_content_sha256=hashlib.sha256(evidence.canonical(value).encode()).hexdigest(),
                initialization_distinct_from_preceding=True, group_id=gid,
                split="student_dev" if mixed and i == 1 else "student_train")
        self.layout_refs = refs
        groups = {gid: dict(group_id=gid,
            split="conflict_preserved_not_admitted" if mixed else "student_train",
            layouts=[dict(layout_id=i, bank=0, split=rows[str(i)]["split"], **refs[i]) for i in range(2)])}
        self.metadata = self.data / "config"
        self.metadata.mkdir()
        self.split = dict(schema="refate.split_manifest.v1", task="synthetic_task",
            owner_chat="SYNTHETIC_EXECUTOR", campaign_id="SYNTHETIC_CAMPAIGN", initial_source_sequence=[0, 1],
            original_partition=dict(train=[0] if mixed else [0, 1], development=[1] if mixed else [], reserved_final=[3]),
            layouts={k: v for k, v in rows.items() if k != "2"}, families=groups,
            layout_to_group={"0": gid, "1": gid}, preserved_all_original_memberships=True, sealed_content_not_read=True)
        self.split_ref = evidence.exclusive(self.metadata / "SPLIT_MANIFEST_001.json", self.split)
        catalog_ref = evidence.exclusive(self.metadata / "catalog.json", dict(
            schema="robofate.source-catalog.v1", data_kind="synthetic", execution_owner="SYNTHETIC_EXECUTOR",
            campaign="SYNTHETIC_CAMPAIGN", tasks={"synthetic_task": dict(
                directory=str(self.metadata), initial_count=2, reserved_final=[3], base_ref=self.split_ref,
                task_ref=evidence.file_ref(task), mixed_collection_base=mixed)}))
        (self.data / "ledger").mkdir()
        campaign_ref = evidence.exclusive(self.metadata / "campaign.json", dict(
            schema="robofate.campaign-config.v1", data_kind="synthetic", execution_owner="SYNTHETIC_EXECUTOR",
            campaign="SYNTHETIC_CAMPAIGN", ledger_root=str(self.data / "ledger"),
            config_root=str(self.metadata), source_commit="SYNTHETIC_ONLY",
            tasks={"synthetic_task": dict(positive_quota=2, negative_quota=2, native_wall_seconds=2400)}))
        old = evidence.decode(evidence.read_pin(self.manifest_ref))
        old["files"] += [evidence.file_ref(task), *refs]
        self.policy.update(native_root=str(self.native), source_catalog_ref=catalog_ref,
                           campaign_config_ref=campaign_ref)
        self.policy["manifest_ref"] = evidence.exclusive(self.metadata / "runtime-files.json", old)
        self.policy_ref = evidence.exclusive(self.metadata / "execution.json", self.policy)
        self.runtime = RuntimePolicy(self.policy_ref)
        self.catalog = SourceCatalog(self.runtime)
        self.ledger = Ledger(self.runtime)
        self.ledger.initialize()

    def reserve(self, ordinal=0, run_id="200001010001_SYNTHETIC_BASELINE"):
        self.ledger.select_task("synthetic_task", run_id + "_task")
        run = self.data / "runs" / run_id
        run.mkdir()
        prepared = self.catalog.prepare("synthetic_task", ordinal, self.split_ref)
        prep_ref = evidence.exclusive(run / "PREPARATION.json", prepared)
        resource = evidence.exclusive(run / "RESOURCES.json", dict(data_kind="synthetic",
            execution_owner="SYNTHETIC_EXECUTOR", execution_policy_ref=self.policy_ref, at=now(), run_id=run_id))
        dispatch = self.ledger.reserve_baseline(prep_ref, run_id, resource, run_id + "_reserved")
        ref = evidence.exclusive(run / "DISPATCH.json", dispatch)
        context = dict(schema="robofate.baseline-context.v1", data_kind="synthetic",
            execution_owner="SYNTHETIC_EXECUTOR", campaign="SYNTHETIC_CAMPAIGN",
            source_owner=prepared["source"]["source_owner"], source_id=dispatch["source_id"], task="synthetic_task",
            run_id=run_id, run_dir=str(run), native_total_steps=4, native_wall_seconds=2400,
            preparation=prepared, preparation_ref=prep_ref, dispatch_ref=ref, execution_policy_ref=self.policy_ref)
        context_ref = evidence.exclusive(run / "CONTEXT_001.json", context)
        self.ledger.attach_context(context_ref, teacher=False, operation_id=run_id + "_armed")
        started_ref = evidence.exclusive(run / "native_started.json", dict(
            pid=731, context_ref=context_ref, synthetic_fixture=True))
        self.ledger.started(context_ref, {"pid": 731}, started_ref)
        return context_ref, context

    def pure_settlement(self, context, *, success=False, safe=True):
        """In-memory original transaction simulation, not production readback."""
        d, _ = self.ledger.store.snapshot()
        api = self.ledger._bind("baseline_ledger_impl.py", LOCK_INODE=17)
        ref = evidence.exclusive(self.metadata / "SYNTHETIC_SUMMARY.json", dict(synthetic_fixture=True))
        q = dict(data_kind="synthetic", source_id=context["source_id"], run_id=context["run_id"],
            file_recording_readback_passed=True, manifest=ref, native_success=success, qualified_negative=success,
            qualified_positive=False, native_steps=3, actual_controls=30)
        e = dict(data_kind="synthetic", task=context["task"], summary=q, execution_owner="SYNTHETIC_EXECUTOR",
            source_id=context["source_id"], run_id=context["run_id"], first_readback_passed=True,
            manifest=ref, new_negative=int(success), new_positive=0, settlement_ref=ref, summary_ref=ref,
            foreground_to_readback_seconds=1, safe_for_next_simulator=safe)
        return api, d, e, api.settle(d, e, now())
