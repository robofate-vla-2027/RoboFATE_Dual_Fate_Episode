"""Campaign composition of original reservation and fate-counting transactions."""
import copy
import datetime
from pathlib import Path
import re

from .. import evidence
from ..evidence import need
from .sources import SourceCatalog
from .store import Store


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run_name(value):
    return type(value) is str and re.fullmatch(r"[0-9]{12}_[A-Za-z0-9_-]{1,60}", value) is not None


class Ledger:
    def __init__(self, policy):
        self.policy, self.p = policy, policy.value
        self.catalog = SourceCatalog(policy)
        self.config_ref = self.p["campaign_config_ref"]
        self.config = evidence.decode(evidence.read_pin(self.config_ref))
        c = self.config
        need(c["schema"] == "robofate.campaign-config.v1" and c["data_kind"] == self.p["data_kind"]
             and c["campaign"] == self.p["campaign"] and c["execution_owner"] == self.p["execution_owner"]
             and set(c["tasks"]) == set(self.p["task_totals"]), "Campaign policy binding differs")
        self.host, self.owner = self.p["ssh_alias"], self.p["execution_owner"]
        root = Path(c["ledger_root"])
        need(root.parent == policy.root and root.name == "ledger", "Ledger must be under private data root")
        self.store = Store(root, data_kind=self.p["data_kind"], campaign=self.p["campaign"],
                           owner=self.owner, host=self.host)

    def _bind(self, name, **values):
        return self.policy.load("campaign/" + name, dict(
            OWNER=self.owner, HOST=self.host, HOSTNAME=self.p["expected_hostname"],
            CAMPAIGN=self.p["campaign"], DATA_KIND=self.p["data_kind"],
            EXCLUDED=self.p["excluded_source_ids"], _run_name=run_name, **values))

    def initialize(self):
        """New empty campaign only. No fabricated predecessor or imported history."""
        at = now()
        tasks = {}
        for task, spec in self.config["tasks"].items():
            pos, neg = spec["positive_quota"], spec["negative_quota"]
            need(type(pos) is int and type(neg) is int and pos >= 0 and neg >= 0 and pos + neg > 0,
                 "Explicit nonnegative quotas required")
            need(type(spec["native_wall_seconds"]) is int and 900 < spec["native_wall_seconds"] <= 3600,
                 "Original finite baseline wall scope required")
            tasks[task] = dict(positive_quota=pos, negative_quota=neg, core_quota=pos+neg,
                qualified_count=0, negative_count=0, core_count=0, remaining_count=pos,
                negative_remaining_count=neg, core_remaining_count=pos+neg, reserved_count=0,
                negative_reserved_count=0, assigned_host=self.host, active_chat_id=self.owner,
                active_run_id=None, latest_segment_started_at=None)
        pq, nq = sum(t["positive_quota"] for t in tasks.values()), sum(t["negative_quota"] for t in tasks.values())
        counts = dict(positive_quota=pq, negative_quota=nq, core_quota=pq+nq,
            qualified_count=0, negative_count=0, core_count=0, remaining_count=pq,
            negative_remaining_count=nq, core_remaining_count=pq+nq, reserved_count=0, negative_reserved_count=0)
        node = dict(counts, active_chat_id=self.owner, fixed_tasks=list(tasks), user_pause_active=False,
            user_stop=False, active_real_simulators=0, active_run_id=None, launch_inflight=False,
            completed_pending_sources=[], common_pending_readback_run_ids=[], current_task=None,
            current_dispatch={}, current_common_foreground={"actual_entry": "robofate.campaign"},
            current_runtime_adoption={"runtime_release": self.p["manifest_ref"]},
            baseline_accounting={key: 0 for key in (
                "baseline_attempts_started", "baseline_attempts_running", "baseline_attempts_native_terminal",
                "baseline_attempts_rgb_readback_passed", "baseline_attempts_native_success", "baseline_attempts_native_failure")})
        value = dict(counts, schema="robofate.campaign-ledger.v1", data_kind=self.p["data_kind"],
            campaign_id=self.p["campaign"], revision=0, started_at=at, updated_at=at,
            target_at=self.config.get("target_at"), task_order=list(tasks),
            fixed_assignment_sha256=self.config_ref["sha256"], quota_history=[], quota_revision=0,
            tasks=tasks, nodes={self.host: node}, source_index={}, reservations={}, split_manifests={},
            negative_task_coverage=0, config_ref=self.config_ref)
        return self.store.initialize(value)

    def _guard(self, d):
        need(evidence.decode(evidence.read_pin(self.config_ref)) == self.config, "Campaign configuration changed")
        n = d["nodes"][self.host]
        need(n["active_chat_id"] == self.owner and not n["user_pause_active"] and not n["user_stop"],
             "Owner/pause/stop changed")
        return n

    def _idle(self, d):
        n = self._guard(d)
        need(n["active_real_simulators"] == 0 and n["active_run_id"] is None
             and not n["launch_inflight"] and n["baseline_accounting"]["baseline_attempts_running"] == 0,
             "Previous run is active, uncertain or awaiting cleanup")
        return n

    def select_task(self, task, operation_id):
        need(task in self.config["tasks"], "Unsupported scheduled task")
        def compute(d, inode):
            n = self._idle(d)
            at = now()
            if n["current_task"] != task:
                n.setdefault("task_segment_history", []).append(dict(task=n["current_task"], ended_at=at))
                d["tasks"][task]["latest_segment_started_at"] = at
            n["current_task"] = task
            n["current_dispatch"] = dict(native_wall_seconds=self.config["tasks"][task]["native_wall_seconds"],
                                          teacher_wall_seconds=3600)
            n["current_runtime_adoption"]["runtime_release"] = self.p["manifest_ref"]
            d.update(revision=d["revision"]+1, updated_at=at)
            return d, {"task": task, "revision": d["revision"]}
        return self.store.transact(operation_id, dict(action="select_task", task=task), compute)

    def _predecessor(self, ledger, boundary, source):
        prior = boundary["settlement"]
        own = [s for s in ledger["source_index"].values() if s.get("host") == self.host]
        if not own:
            need(prior == {"bootstrap": True} and not source["required_accounted_prefix"]
                 and not ledger["reservations"], "Only a genuinely empty campaign can bootstrap")
            return
        need(prior.get("bootstrap") is not True, "Bootstrap cannot skip existing sources")
        old = ledger["source_index"][prior["source_id"]]
        need(prior["safe_for_next_simulator"] is True and old["settlement"] == prior["settlement_ref"],
             "Predecessor evidence changed")
        evidence.read_pin(prior["settlement_ref"])
        api = self._bind("reservation_impl.py")
        need(api._settled_predecessor(ledger, prior, boundary), "Real predecessor not settled")

    def reserve_baseline(self, preparation_ref, run_id, resource_ref, operation_id):
        need(run_name(run_id), "Fresh baseline run token required")
        prepared = evidence.decode(evidence.read_pin(preparation_ref))
        self.catalog.validate(prepared)
        observation = evidence.decode(evidence.read_pin(resource_ref))
        need(observation["data_kind"] == self.p["data_kind"]
             and observation["execution_owner"] == self.owner
             and observation["execution_policy_ref"] == self.policy.reference
             and observation["run_id"] == run_id, "Resource observation scope differs")
        def compute(d, inode):
            n = self._idle(d)
            prior = n.get("last_source_settlement", {"bootstrap": True})
            boundary = dict(at=observation["at"], next_preparation={"prepared": prepared, "ref": preparation_ref},
                resource_precheck={"summary": {"run_id": run_id}, "ref": resource_ref},
                settlement=prior, runtime_release=self.p["manifest_ref"], teacher_supplement_present=True)
            if "last_teacher_settlement" in n:
                boundary["last_teacher_settlement"] = n["last_teacher_settlement"]
            api = self._bind("reservation_impl.py", _predecessor=self._predecessor)
            out, dispatch = api.reserve(d, boundary, now())
            sid = dispatch["source_id"]
            s = out["source_index"][sid]
            s.update(group_id=s["source_group_id"], source_owner=prepared["source"]["source_owner"],
                     required_accounted_prefix=prepared["source"]["required_accounted_prefix"],
                     source_sequence=prepared["source"]["source_sequence"])
            if "collection_append_provenance" in prepared:
                s["collection_append_provenance"] = copy.deepcopy(prepared["collection_append_provenance"])
            r = out["reservations"][sid]
            r.update(quota_reserved=False, negative_quota_reserved=True, source_owner=s["source_owner"])
            return out, dispatch
        return self.store.transact(operation_id, dict(action="reserve_baseline", preparation_ref=preparation_ref,
            run_id=run_id, resource_ref=resource_ref), compute)

    def attach_context(self, context_ref, *, teacher, operation_id):
        c = evidence.decode(evidence.read_pin(context_ref, roots=(self.policy.root,)))
        def compute(d, inode):
            n = self._guard(d)
            r, s = d["reservations"][c["source_id"]], d["source_index"][c["source_id"]]
            key = "current_attempt_context" if teacher else "context"
            run_key = "current_attempt_run_id" if teacher else "run_id"
            need(r[run_key] == c["run_id"] == n["active_run_id"] and not n["launch_inflight"]
                 and n["active_real_simulators"] == 0 and key not in r, "Context already attached or active")
            r[key] = context_ref
            if not teacher:
                s["context"] = context_ref
            n["launch_inflight"] = True
            d.update(revision=d["revision"]+1, updated_at=now())
            return d, dict(context_ref=context_ref, launch_armed=True)
        return self.store.transact(operation_id, dict(action="attach_context", context_ref=context_ref, teacher=teacher), compute)

    def started(self, context_ref, identity, start_ref):
        c = evidence.decode(evidence.read_pin(context_ref))
        start = evidence.decode(evidence.read_pin(start_ref))
        need(start["pid"] == identity["pid"] and start["context_ref"] == context_ref, "Actual child start differs")
        teacher = c["schema"] == "robofate.teacher-context.v1"
        def compute(d, inode):
            n = self._guard(d)
            s, r = d["source_index"][c["source_id"]], d["reservations"][c["source_id"]]
            key = "current_attempt_context" if teacher else "context"
            need(n["active_run_id"] == c["run_id"] and n["active_real_simulators"] == 0
                 and n["launch_inflight"] is True and r[key] == context_ref, "Start not current armed context")
            n.update(active_real_simulators=1, launch_inflight=False)
            n["current_runtime_adoption"].update(actual_native_started=True, actual_start=start_ref)
            if teacher:
                need(r["positive_slots"] == 1 and not s.get("active_teacher_attempt"), "Teacher already active")
                s["active_teacher_attempt"] = dict(run_id=c["run_id"], context=context_ref,
                                                   actual_start=start_ref, identity=identity)
            else:
                need(not s["source_started"] and r["status"] == "reserved" and r["negative_slots"] == 1,
                     "Baseline already started")
                s.update(source_started=True, actual_start=start_ref, native_identity=identity)
                r["status"] = "running"
                n["baseline_accounting"]["baseline_attempts_started"] += 1
                n["baseline_accounting"]["baseline_attempts_running"] += 1
                n["current_common_foreground"]["stage"] = "actual_native_started"
            d.update(revision=d["revision"]+1, updated_at=now())
            return d, dict(actual_start=start_ref, positive_delta=0, negative_delta=0)
        return self.store.transact(c["run_id"] + "_started",
            dict(context_ref=context_ref, identity=identity, start_ref=start_ref), compute)

    def settle_baseline(self, context_ref, summary_ref, *, safe):
        c, q = (evidence.decode(evidence.read_pin(ref)) for ref in (context_ref, summary_ref))
        need(c["data_kind"] == q["data_kind"] == "actual", "Synthetic readback cannot change production quotas")
        need(q["settlement_binding"]["context_ref"] == context_ref, "Baseline settlement binding differs")
        evidence.read_pin(q["manifest"])
        def compute(d, inode):
            api = self._bind("baseline_ledger_impl.py", LOCK_INODE=inode)
            event = dict(data_kind="actual", task=c["task"], source_id=c["source_id"], run_id=c["run_id"],
                execution_owner=self.owner, summary=q, first_readback_passed=True, manifest=q["manifest"],
                new_negative=int(q["native_success"]), new_positive=0, settlement_ref=summary_ref,
                summary_ref=summary_ref, foreground_to_readback_seconds=q["readback_wall_seconds"],
                safe_for_next_simulator=safe)
            out, receipt = api.settle(d, event, now())
            if out != d:
                s = out["source_index"][c["source_id"]]
                s.update(complete_readback=True, preparation=c["preparation_ref"])
                out["nodes"][self.host]["last_source_settlement"] = dict(source_id=c["source_id"],
                    settlement_ref=summary_ref, safe_for_next_simulator=safe)
            return out, receipt
        return self.store.transact(c["run_id"] + "_settled",
            dict(context_ref=context_ref, summary_ref=summary_ref, safe=safe), compute)

    def close_failed_baseline(self, context_ref, failure_ref, *, safe):
        """Quarantine an infrastructure-failed source; never call it a DM failure."""
        c = evidence.decode(evidence.read_pin(context_ref))
        failure = evidence.decode(evidence.read_pin(failure_ref))
        run = Path(c["run_dir"])
        need(c["schema"] == "robofate.baseline-context.v1" and c["data_kind"] == self.p["data_kind"]
             and Path(failure_ref["path"]).parent in (run, run / "baseline_readback_001")
             and failure.get("qualified") is False, "Explicit failed baseline evidence required")
        event = dict(schema="robofate.baseline-quarantine.v1", data_kind=self.p["data_kind"],
            source_id=c["source_id"], run_id=c["run_id"], context_ref=context_ref, failure_ref=failure_ref,
            safe_for_next_simulator=safe, native_failure_proven=False, new_positive=0, new_negative=0)
        def compute(d, inode):
            n = self._guard(d)
            s, r, t = d["source_index"][c["source_id"]], d["reservations"][c["source_id"]], d["tasks"][c["task"]]
            need(r["context"] == context_ref and r["negative_slots"] == 1 and r["positive_slots"] == 0
                 and s.get("settlement") is None, "Not an unsettled baseline reservation")
            if s["source_started"]:
                need(n["baseline_accounting"]["baseline_attempts_running"] == 1, "Baseline count drift")
                n["baseline_accounting"]["baseline_attempts_running"] -= 1
            for scope in (n, t):
                need(scope["negative_reserved_count"] > 0, "Missing negative reservation")
                scope["negative_reserved_count"] -= 1
            s.update(status="quarantined_infrastructure_failure", reusable_DM_failure=False,
                complete_readback=False, settlement=failure_ref)
            r.update(status="settled_quarantined_no_quota", positive_slots=0, negative_slots=0,
                     quota_reserved=False, negative_quota_reserved=False, settlement=failure_ref)
            n.update(active_run_id=None if safe else c["run_id"],
                active_real_simulators=0 if safe else n["active_real_simulators"], launch_inflight=False,
                last_source_settlement=dict(source_id=c["source_id"], settlement_ref=failure_ref, safe_for_next_simulator=safe))
            t["active_run_id"] = None if safe else c["run_id"]
            s.setdefault("history", []).append(event)
            d.update(revision=d["revision"]+1, updated_at=now())
            return d, event
        return self.store.transact(c["run_id"] + "_quarantined", event, compute)

    def confirm_cleanup(self, context_ref, cleanup_ref):
        """Record independently rechecked owned cleanup after data settlement."""
        c = evidence.decode(evidence.read_pin(context_ref))
        need(cleanup_ref == evidence.file_ref(Path(c["run_dir"]) / "process_cleanup.json"),
             "Original cleanup evidence required")
        need(evidence.decode(evidence.read_pin(cleanup_ref))["owned_cleanup_complete"] is True,
             "Cleanup is still unverified")
        def compute(d, inode):
            n = self._guard(d)
            s, r = d["source_index"][c["source_id"]], d["reservations"][c["source_id"]]
            need(n["active_run_id"] == c["run_id"] and r["positive_slots"] == r["negative_slots"] == 0
                 and (s.get("settlement") or r.get("current_attempt_settlement")),
                 "Only the current settled source may release the resource gate")
            n.update(active_run_id=None, active_real_simulators=0, launch_inflight=False)
            n["last_source_settlement"]["safe_for_next_simulator"] = True
            d["tasks"][c["task"]]["active_run_id"] = None
            d.update(revision=d["revision"]+1, updated_at=now())
            return d, dict(cleanup_ref=cleanup_ref, positive_delta=0, negative_delta=0)
        return self.store.transact(c["run_id"] + "_cleanup_confirmed",
            dict(context_ref=context_ref, cleanup_ref=cleanup_ref), compute)

    def reserve_teacher(self, source_id, run_id, session_id, t1, operation_id, *, protocol):
        need(run_name(run_id) and run_name(session_id), "Fresh teacher run/session tokens required")
        def compute(d, inode):
            n = self._idle(d)
            s, r = d["source_index"][source_id], d["reservations"][source_id]
            t = d["tasks"][s["task"]]
            need(source_id not in self.p["excluded_source_ids"] and s.get("reusable_DM_failure") is True
                 and s.get("complete_readback") is True and s.get("native_success") is False
                 and not s.get("provider_safety_refusal") and not s.get("counted_in_quota")
                 and not s.get("counted_in_negative_quota") and not s.get("active_teacher_attempt")
                 and r["negative_slots"] == r["positive_slots"] == 0, "Source cannot admit a new teacher")
            need("current_attempt_run_id" not in r, "No automatic second teacher attempt")
            need(type(t1) is int and 0 < t1 < s["native_steps"] <= s["native_budget"], "Invalid causal cutoff")
            need(all(scope["qualified_count"] + scope.get("reserved_count", 0) < scope["positive_quota"]
                     for scope in (d, n, t)), "No positive quota headroom")
            need(not any(run_id in (x.get("run_id"), x.get("current_attempt_run_id"))
                         for x in d["reservations"].values()), "Used run ID")
            source = evidence.decode(evidence.read_pin(s["context"]))
            prepared = self.catalog.validate(source["preparation"])
            at = now()
            dispatch = dict(schema="robofate.teacher-dispatch.v1", data_kind=self.p["data_kind"],
                campaign=self.p["campaign"], execution_owner=self.owner, source_owner=s["source_owner"],
                source_id=source_id, task=s["task"], run_id=run_id, session_id=session_id,
                issued_at=at, valid_until=(datetime.datetime.fromisoformat(at)+datetime.timedelta(hours=2)).isoformat(),
                ledger_revision=d["revision"]+1, source_reserved=True, positive_slots=1, negative_slots=0,
                native_total_steps=s["native_budget"], t1=t1, source_context_ref=s["context"],
                source_manifest_ref=s["manifest"], source_readback_ref=s["readback"],
                split_manifest=prepared["provenance"]["split"], **protocol)
            r.update(status="reserved_shared_Evo_teacher", quota_reserved=True, negative_quota_reserved=False,
                positive_slots=1, negative_slots=0, current_attempt_run_id=run_id,
                active_attempt_owner=self.owner, source_owner=s["source_owner"], execution_owner=self.owner,
                split=s["split"], split_manifest=dispatch["split_manifest"], dispatch=copy.deepcopy(dispatch))
            for scope in (d, n, t):
                scope["reserved_count"] = scope.get("reserved_count", 0)+1
            n.update(active_run_id=run_id, launch_inflight=False)
            t["active_run_id"] = run_id
            d.update(revision=d["revision"]+1, updated_at=at)
            return d, dispatch
        return self.store.transact(operation_id, dict(action="reserve_teacher", source_id=source_id,
            run_id=run_id, session_id=session_id, t1=t1, protocol=protocol), compute)

    def settle_teacher(self, context_ref, summary_ref, *, safe):
        c, q = (evidence.decode(evidence.read_pin(ref)) for ref in (context_ref, summary_ref))
        need(c["data_kind"] == q["data_kind"] == "actual"
             and q["settlement_binding"]["context_ref"] == context_ref
             and q["source_id"] == c["source_id"] and q["run_id"] == c["run_id"],
             "Actual teacher settlement binding differs")
        need(type(c["native_limit"]) is int and c["native_limit"] == self.p["task_totals"][c["task"]]
             and type(c["t1"]) is int and 0 < c["t1"] < c["native_limit"]
             and type(c["remaining_steps"]) is int and c["remaining_steps"] == c["native_limit"]-c["t1"],
             "Teacher settlement changed the original task/native budget")
        if q["native_result_ref"]:
            need(evidence.decode(evidence.read_pin(q["native_result_ref"])) == q["native_result"], "Changed terminal result")
        def compute(d, inode):
            n = self._guard(d)
            sid = c["source_id"]
            s, r = d["source_index"][sid], d["reservations"][sid]
            need(r["current_attempt_context"] == context_ref and r["current_attempt_run_id"] == c["run_id"]
                 and r["positive_slots"] == 1 and r["quota_reserved"] is True, "Not current positive reservation")
            at = now()
            if q["qualified_positive"]:
                need(q["new_positive"] == 1 and q["complete_positive_reader_run"] is True, "Unqualified positive")
                pub = q["publication"]
                manifest = evidence.decode(evidence.read_pin(pub["manifest"]))
                receipt = evidence.decode(evidence.read_pin(pub["publication_receipt"]))
                need(manifest["source_id"] == receipt["source_id"] == sid
                     and receipt["manifest"] == pub["manifest"] and manifest["episode_id"] == c["run_id"]
                     and manifest["synthetic_fixture"] is False, "Publication mismatch")
                report = evidence.decode(evidence.read_pin(q["readback_ref"]))
                need(report["context_ref"] == context_ref and report["failure_summary_ref"] == c["source_readback_ref"]
                     and report["source_id"] == sid and report["task"] == c["task"]
                     and report["run"] == c["run_dir"] and report["t1"] == c["t1"]
                     and report["native_limit"] == c["native_limit"] and report["synthetic_fixture"] is False
                     and manifest["evidence"]["recording_readback"] == q["readback_ref"]
                     and manifest["evidence"]["context"] == context_ref,
                     "Positive report/publication/context linkage differs")
                from ..readback.reader import sig
                for path, signature in report["stable_files"].items():
                    need(sig(path) == signature, "Reviewed teacher data changed before commit")
                request = dict(schema="refate.positive.ledger_commit_input.v1", data_kind="actual",
                    campaign_id=c["campaign"], owner=self.owner, source_owner=c["source_owner"], host=self.host,
                    source_id=sid, task=c["task"], run_id=c["run_id"],
                    teacher="EmbodiedEvo public-recipe primitive baseline", split=s["split"], group_id=s["group_id"],
                    native_dm_success=False, native_teacher_success=q["native_result"]["native_success"],
                    conditions={k: report["conditions"][k]["status"] == "passed" for k in ("1.1", "1.2", "1.3")},
                    refs=dict(publication=pub["publication_receipt"], manifest=pub["manifest"], acceptance=q["readback_ref"],
                        context=context_ref, source_report=c["source_readback_ref"], teacher_report=q["readback_ref"],
                        split_manifest=r["split_manifest"]))
                api = self._bind("positive_impl.py", LOCK_INODE=inode, SOURCE=sid, TASK=c["task"],
                    SOURCE_OWNER=c["source_owner"], _evidence_path=lambda p: Path(p).is_relative_to(self.policy.root),
                    _collection_split_compat=self.catalog.collection_guard(d, request))
                out, result = api.apply_commit(d, request, now=at, lock_inode=inode)
                out["source_index"][sid]["qualified_count"] = 1
            else:
                need(q["new_positive"] == q["new_negative"] == 0, "Unqualified result cannot receive credit")
                out, result = copy.deepcopy(d), dict(positive_delta=0, negative_delta=0, idempotent_reuse=False)
                out["revision"] += 1
            s, r, n, t = out["source_index"][sid], out["reservations"][sid], out["nodes"][self.host], out["tasks"][c["task"]]
            s.pop("active_teacher_attempt", None)
            s.setdefault("teacher_attempt_history", []).append(dict(at=at, summary=summary_ref, **result))
            refusal = Path(c["runtime_binding"]["run_dir"]) / "agent/provider_safety.json"
            if refusal.exists():
                s["provider_safety_refusal"] = evidence.file_ref(refusal)
            r.update(status="shared_Evo_teacher_settled", quota_reserved=False, negative_quota_reserved=False,
                positive_slots=0, negative_slots=0, current_attempt_settlement=summary_ref, settled_at=at)
            for scope, rows in ((out, list(out["reservations"].values())),
                                (n, [v for v in out["reservations"].values() if v["host"] == self.host]),
                                (t, [v for v in out["reservations"].values() if v["task"] == c["task"]])):
                scope["reserved_count"] = sum(x.get("quota_reserved") is True for x in rows)
            n.update(active_real_simulators=0 if safe else n["active_real_simulators"],
                active_run_id=None if safe else c["run_id"], launch_inflight=False, last_teacher_settlement=summary_ref)
            n["last_source_settlement"] = dict(source_id=sid, settlement_ref=s["settlement"], safe_for_next_simulator=safe)
            t["active_run_id"] = None if safe else c["run_id"]
            out["updated_at"] = at
            return out, dict(result, summary_ref=summary_ref, revision=out["revision"])
        return self.store.transact(c["run_id"] + "_settled", dict(context_ref=context_ref, summary_ref=summary_ref, safe=safe), compute)
