"""Bind the original restore code to pinned external source/admission metadata.

No simulator, model, resource grant or tolerance defaults are created here.
The source/profile/owner checks are retained; equality never closes hidden state.
"""
import copy
import hashlib
import os
from pathlib import Path
import stat
import sys
from types import ModuleType, SimpleNamespace
import uuid

from .. import evidence
from ..evidence import canonical, decode, need

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
IMPLEMENTATIONS = ("context_impl.py", "task_impl.py", "comparator_impl.py",
                   "support_impl.py", "material_impl.py")


class RestoreBinding:
    def __init__(self, policy_ref):
        self.policy_ref = copy.deepcopy(policy_ref)
        self.policy = decode(evidence.read_pin(self.policy_ref))
        p = self.policy
        need(p.get("schema") == "robofate.restore-policy.v1"
             and p.get("data_kind") in ("actual", "synthetic"), "Explicit restore policy required")
        self.root = Path(p["data_root"])
        self.config_root = Path(p["config_root"])
        for root in (self.root, self.config_root):
            need(root.is_absolute() and root.resolve(strict=True) == root and root.is_dir(),
                 "Canonical existing policy roots required")
        need(not self.root.is_relative_to(PACKAGE_ROOT) and not PACKAGE_ROOT.is_relative_to(self.root),
             "Source recordings must be separate from installed code")
        self.roots = (self.root, self.config_root, PACKAGE_ROOT)
        need(Path(policy_ref["path"]).is_relative_to(self.config_root), "Policy must be in config_root")
        need(all(type(p.get(k)) is str and p[k] for k in ("execution_owner", "partition")),
             "Explicit execution owner and partition required")
        need(type(p["task_totals"]) is dict and p["task_totals"]
             and all(type(k) is str and k and type(v) is int and v > 0 for k, v in p["task_totals"].items()),
             "Explicit task budget table required")
        need(type(p["excluded_source_ids"]) is list
             and all(type(v) is str and v for v in p["excluded_source_ids"])
             and len(set(p["excluded_source_ids"])) == len(p["excluded_source_ids"]),
             "Explicit source exclusions required")
        need(set(p["rules"]) == {"profile_template", "invariants"}, "Exact external restore rule pins required")
        need(set(p["code_pins"]) == set(IMPLEMENTATIONS), "Pin all original restore implementations")
        for name in IMPLEMENTATIONS:
            need(p["code_pins"][name]["path"] == str(HERE / name), "Unexpected restore implementation path")
            self.read_pin(p["code_pins"][name])
        for ref in p["rules"].values():
            self.read_pin(ref)
        self.read_pin(p["admission_ref"])
        self._modules = []
        self.material_api = self._load("material_impl.py", {
            "need": need, "ref": self.ref, "document": self.document,
            "runtime": SimpleNamespace(read_pin=self.read_pin),
            "stage": SimpleNamespace(SHARED=self.root, OWNER=p["execution_owner"], write_new=evidence.exclusive),
            "_policy": copy.deepcopy(p), "_bound_budget": self.budget,
            "_rule_document": self.rule_document, "_source_admission": self.source_admission,
        })

    def read_pin(self, ref):
        return evidence.read_pin(ref, roots=self.roots)

    def ref(self, path):
        return evidence.file_ref(path, roots=self.roots)

    def document(self, ref):
        return decode(self.read_pin(ref))

    def rule_document(self, name):
        ref = self.policy["rules"][name]
        return self.document(ref), copy.deepcopy(ref)

    def budget(self, task, total, t1):
        need(type(task) is str and task in self.policy["task_totals"]
             and type(total) is int and total == self.policy["task_totals"][task]
             and type(t1) is int and 0 < t1 < total, "Unbound task/native budget/cutoff")
        return total

    def source_admission(self, context, manifest, summary):
        need(self.policy == decode(evidence.read_pin(self.policy_ref)), "Restore policy changed")
        ledger = self.document(self.policy["admission_ref"])
        need(ledger.get("schema") == "robofate.source-admission.v1"
             and ledger.get("data_kind") == self.policy["data_kind"]
             and ledger.get("execution_owner") == self.policy["execution_owner"],
             "Wrong independently pinned admission ledger")
        sid = context["source_id"]
        row = ledger["sources"][sid]
        need(sid not in self.policy["excluded_source_ids"] and row.get("status") == "reserved"
             and row.get("consumed") is False and row.get("refused") is False
             and row.get("source_id") == sid and row.get("task") == context["task"],
             "Excluded/consumed/refused/unreserved source")
        need(row["context_ref"] == self.ref(Path(context["run_dir"]) / row["context_filename"]),
             "Admission must independently pin the source context")
        need(self.document(row["context_ref"]) == context
             and self.document(row["manifest_ref"]) == manifest
             and self.document(row["readback_ref"]) == summary, "Admission/source evidence differs")

    def _load(self, filename, injected):
        ref = self.policy["code_pins"][filename]
        raw = self.read_pin(ref)
        name = __package__ + "._bound_" + uuid.uuid4().hex
        module = ModuleType(name)
        module.__file__ = str(HERE / filename)
        module.__package__ = __package__
        module.__dict__.update(injected)
        sys.modules[name] = module
        try:
            exec(compile(raw, module.__file__, "exec"), module.__dict__)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        self._modules.append(module)
        return module

    def source_material(self, context_ref, manifest_ref, readback_ref, *, t1):
        return self.material_api.source_material(context_ref, manifest_ref, readback_ref, t1=t1)

    def load(self, material, *, core_modules):
        task, total, t1 = material["task"], material["native_limit"], material["t1"]
        self.budget(task, total, t1)
        checked = self.source_material(material["source_context"], material["source_manifest"],
                                       material["source_readback"], t1=t1)
        need(canonical(checked) == canonical(material), "Source material changed")
        profile = material["profile"]
        sid = material["source_id"]
        source = Path(material["source"])
        def identity_id(value):
            need(canonical(value) == canonical(material["source_identity"]), "Full source identity differs")
            need(value.get("source_id") == sid and value.get("task") == task
                 and ("candidate_source_id" not in value or value["candidate_source_id"] == sid),
                 "Conflicting source identity aliases")
            return sid
        projections = {key: hashlib.sha256(canonical(profile[key]).encode()).hexdigest()
                       for key in ("schema_contract", "thresholds", "joint_units", "required_discrete_exact")}
        context = self._load("context_impl.py", dict(
            _TASK=task, _TOTAL=total, _T1=t1, _SOURCE_ID=sid, _PROJECTIONS=projections, _identity_id=identity_id))
        refs = {"source": self.ref(source / "source_identity.json"), "readback": material["source_readback"],
                "physics": material["boundary_ref"], "auxiliary": self.ref(source / f"task_auxiliary_{t1:06d}.json"),
                "camera": self.ref(source / f"camera_{t1:06d}.json"),
                "observation": self.ref(source / f"macro_obs_{t1:06d}/observation.json"),
                "result": self.ref(source / "result.json")}
        contract = dict(profile, source_evidence=refs)
        task_api = self._load("task_impl.py", dict(
            TASK=task, TOTAL=total, T1=t1, REMAINING=total-t1, SOURCE_ID=sid,
            _SOURCE_SEED=material["source_identity"]["encoded_seed"],
            _SOURCE_ORDINAL=material["source_identity"]["native_ordinal"],
            _identity_id=identity_id, contract=lambda: copy.deepcopy(contract)))
        comparator = self._load("comparator_impl.py", dict(
            _TASK=task, _SOURCE_ID=sid, check_profile=context.check_profile,
            require_context=context.require_context, profile_total=context.profile_total,
            verify_bound=task_api.verify_bound, validate_profile=task_api.validate_profile))
        support = self._load("support_impl.py", dict(
            ROOT=self.root, READ_ROOTS=self.roots, _TOTAL=total, _T1=t1, _COMPARATOR=comparator,
            **{name: getattr(context, name) for name in
               ("check_native", "check_profile", "check_restore", "native_snapshot", "canonical", "_decode")},
            **{name: getattr(task_api, name) for name in
               ("verify_bound", "verify_restore_config", "verify_source", "validate_profile")},
            contract=lambda: copy.deepcopy(contract)))

        def recorded_prefix(reference, config, *, context=None):
            context_api.check_restore(context, config)
            task_api.verify_bound(context)
            task_api.verify_restore_config(config)
            need(str(reference) == str(source), "Different original prefix source")
            current = self.source_material(material["source_context"], material["source_manifest"],
                                           material["source_readback"], t1=t1)
            need(current["profile"] == profile, "Source metadata changed since preparation")
            support.native25_recording(source)
            filename = source / "native_actions.jsonl"
            with os.fdopen(os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb", buffering=0) as stream:
                before = os.fstat(stream.fileno())
                need(stat.S_ISREG(before.st_mode), "Regular original action stream required")
                rows = support.prefix_rows(stream, t1, total=total)
                after = os.fstat(stream.fileno())
            signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            need(signature(before) == signature(after) == signature(filename.stat()), "Original causal prefix changed")
            return rows
        context_api = context
        support.recorded_prefix = recorded_prefix
        return SimpleNamespace(context=context, task=task_api, comparator=comparator,
                               support=support, material=copy.deepcopy(material), profile=copy.deepcopy(profile))

    def prepare_context(self, material, **kwargs):
        self.budget(material["task"], material["native_limit"], material["t1"])
        run = Path(kwargs["run"])
        root = Path(kwargs.get("metadata_root") or run)
        need(root.is_absolute() and root.resolve() == root
             and (root == run or root.is_relative_to(self.config_root))
             and not root.is_relative_to(Path(material["source"])),
             "Restore metadata must use the new run or private configuration root")
        current = self.source_material(material["source_context"], material["source_manifest"],
                                       material["source_readback"], t1=material["t1"])
        need(canonical(current) == canonical(material), "Source changed before restore preparation")
        return self.material_api.prepare_context(material, **kwargs)

    def bind_actual(self, api, preparation, task):
        return self.material_api.bind_actual(api, preparation, task)
