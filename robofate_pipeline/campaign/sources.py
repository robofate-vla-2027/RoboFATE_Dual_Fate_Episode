"""Pinned original source assembly and contiguous, outcome-blind registration."""
import copy
import hashlib
from pathlib import Path
import re
from types import ModuleType, SimpleNamespace

from .. import evidence
from ..evidence import canonical, need
from .family import family

PROFILE = "pt_spatial4_collection_candidate_v1"
SENTINEL = "conflict_preserved_not_admitted"


class SourceCatalog:
    def __init__(self, policy):
        self.policy = policy
        self.p = policy.value
        self.reference = self.p["source_catalog_ref"]
        self.value = evidence.decode(evidence.read_pin(self.reference))
        need(self.value["schema"] == "robofate.source-catalog.v1"
             and self.value["data_kind"] == self.p["data_kind"]
             and self.value["execution_owner"] == self.p["execution_owner"]
             and self.value["campaign"] == self.p["campaign"], "Source catalog scope differs")
        self.tasks = self.value["tasks"]
        need(set(self.tasks) == set(self.p["task_totals"]), "Task catalog/budgets differ")
        self.mixed = {task: spec["base_ref"] for task, spec in self.tasks.items()
                      if spec.get("mixed_collection_base") is True}
        self.membership = policy.load("campaign/membership_impl.py", {"MIXED_REFS": self.mixed})
        self.registration = policy.load("campaign/registration_impl.py", dict(
            OWNER=self.p["execution_owner"], CAMPAIGN=self.p["campaign"], NATIVE=policy.native,
            profile=self.profile, document=self.document, ref=evidence.file_ref))

    def document(self, ref):
        return evidence.decode(evidence.read_pin(ref))

    def profile(self, task):
        need(task in self.tasks, "Unregistered task")
        s = self.tasks[task]
        root = self.policy.directory(s["directory"])
        count, sealed = s["initial_count"], s["reserved_final"]
        need(type(count) is int and count > 0 and type(sealed) is list and sealed
             and all(type(i) is int and i >= count for i in sealed)
             and sealed == sorted(set(sealed)), "Explicit frozen initial/sealed partition required")
        need(Path(s["base_ref"]["path"]).parent == root, "Base manifest outside task metadata root")
        return dict(root=root, base=s["base_ref"], initial_count=count, reserved_final=sealed)

    def lineage(self, ref, task):
        need(evidence.decode(evidence.read_pin(self.reference)) == self.value, "Source catalog changed")
        if task in self.mixed:
            return self._mixed_registration(task).lineage(ref)
        return self.registration.lineage(ref, task)

    def _mixed_registration(self, task):
        p = self.profile(task)
        original = SimpleNamespace(same=lambda a, b: canonical(a) == canonical(b))
        return self.policy.load("campaign/mixed_registration_impl.py", dict(
            TASK=task, BASE_REF=p["base"], INITIAL_COUNT=p["initial_count"], SEALED=p["reserved_final"],
            metadata=SimpleNamespace(NATIVE=self.policy.native, OWNER=self.p["execution_owner"],
                sequence=SimpleNamespace(CAMPAIGN=self.p["campaign"]), original=original),
            previous=SimpleNamespace(SENTINEL=SENTINEL), document=self.document,
            path_for=lambda i: p["root"] / f"SPLIT_APPEND_L{i}_001.json"))

    def _read(self, ref, *, expected_path=None):
        if expected_path is not None:
            need(ref["path"] == str(expected_path), "Unexpected task/layout file")
            need(self.policy.pins[str(expected_path)] == ref, "Task/layout not pinned by release")
        return evidence.read_pin(ref)

    def _module(self, task):
        need(task in self.tasks, "Unsupported task")
        spec = self.tasks[task]
        return self.policy.load("campaign/source_impl.py", dict(
            SPECS={task: (self.p["task_totals"][task], spec["task_ref"]["sha256"])},
            NATIVE=self.policy.native, OWNER=self.p["execution_owner"], PROFILE=PROFILE,
            sequence=SimpleNamespace(CAMPAIGN=self.p["campaign"]),
            original=SimpleNamespace(canonical=canonical), read_ref=self._read,
            _membership=self._membership))

    def _membership(self, split, ordinal, ref, raw):
        task = split["task"]
        row = split["layouts"][str(ordinal)]
        need(row["group_id"] == task + "/task_content_family/" + family(raw)["content_sha256"],
             "Full non-pose task family mismatch")
        if task in self.mixed and ref != self.mixed[task]:
            need(split == self.lineage(ref, task) and ordinal == split["collection_append"]["added_layout"],
                 "Different current mixed append or ancestry")
            return True
        return self.membership.membership(split, ordinal, ref, raw)

    def prepare(self, task, ordinal, split_ref):
        split = self.lineage(split_ref, task)
        api = self._module(task)
        total, task_ref = api.task_budget(task)
        need(task_ref == self.tasks[task]["task_ref"], "Task AST budget binding differs")
        row = split["layouts"][str(ordinal)]
        path = self.policy.native / f"Assets/Eval_Layout/RoboDojo/arx_x5/0/{task}_{ordinal}.json"
        raw = evidence.decode(self._read(row["layout_ref"], expected_path=path))
        # Execute only the original pure filename/ordinal decoder, never Native.
        decoder_path = self.policy.teacher / "simulators/robodojo_layouts.py"
        import ast
        text = self.policy.pin(decoder_path).decode()
        tree = ast.parse(text)
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("layout_files", "decode_scene")]
        need(len(nodes) == 2, "Original layout decoder seam missing")
        decoder = ModuleType("_bound_layout_decoder")
        decoder.__dict__.update(ROOT=self.policy.native, STRIDE=1_000_000, re=re)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(decoder_path), "exec"), decoder.__dict__)
        need(decoder.decode_scene(task, ordinal) == (0, ordinal, path), "Original numeric ordinal differs")
        prepared = api.assemble(task, ordinal, split_ref, split, total, task_ref, raw)
        group = split["families"][row["group_id"]]
        if task in self.mixed and split_ref != self.mixed[task]:
            prepared["collection_append_provenance"] = dict(
                original_manifest=copy.deepcopy(self.mixed[task]), append_manifest=copy.deepcopy(split_ref),
                original_memberships_modified=False, collection_scope=True,
                independent_development_claimed=False, independent_evaluation_claimed=False)
        elif group["split"] == SENTINEL:
            prepared["collection_group_compatibility"] = dict(
                schema="refate.collection_only_mixed_legacy_labels.v1",
                original_manifest=copy.deepcopy(split_ref), original_group=copy.deepcopy(group),
                original_row=copy.deepcopy(row), collection_scope=True,
                independent_development_claimed=False, independent_evaluation_claimed=False,
                original_memberships_modified=False,
                limitation="Mixed legacy student labels remain; not independent development/evaluation groups")
        return prepared

    def validate(self, preparation):
        s = preparation["source"]
        expected = self.prepare(s["task"], s["layout_id"], preparation["provenance"]["split"])
        need(canonical(preparation) == canonical(expected), "Full source preparation changed")
        return expected

    def append(self, task, prior_ref, layout_ref, at):
        """Metadata only. No source issuance and no inspection of sealed layouts."""
        prior = self.lineage(prior_ref, task)
        i = len(prior["initial_source_sequence"])
        spec = self.profile(task)
        need(i < spec["reserved_final"][0], "Sealed or exhausted source sequence")
        path = self.policy.native / f"Assets/Eval_Layout/RoboDojo/arx_x5/0/{task}_{i}.json"
        raw = evidence.decode(self._read(layout_ref, expected_path=path))
        gid = task + "/task_content_family/" + family(raw)["content_sha256"]
        group = prior["families"].get(gid)
        mixed = task in self.mixed
        need((mixed and group is not None and group["split"] == SENTINEL)
             or (not mixed and (group is None or group["split"] in ("student_train", "student_dev"))),
             "Append family is outside its registered lineage mode")
        row = dict(layout_id=i, native_bank=0, layout_ref=layout_ref,
            initialization_content_sha256=hashlib.sha256(canonical(raw).encode()).hexdigest(),
            initialization_distinct_from_preceding=True, group_id=gid,
            split=group["split"] if group and not mixed else "student_train")
        helper = evidence.file_ref(Path(self.registration.__file__))
        value = (self._mixed_registration(task).merge(prior, prior_ref, row) if mixed
                 else self.registration.merge(prior, prior_ref, row, at, helper))
        return evidence.exclusive(spec["root"] / f"SPLIT_APPEND_L{i}_001.json", value)

    def collection_guard(self, ledger, request):
        """Allow only independently revalidated original mixed-base members."""
        def allow(other):
            task = request["task"]
            need(task in self.mixed, "Cross-split compatibility requires a registered mixed base")
            current = request["refs"]["split_manifest"]
            self.lineage(current, task)
            ancestors = []
            while True:
                need(len(ancestors) <= self.profile(task)["reserved_final"][0], "Excessive append ancestry")
                ancestors.append(current)
                if current == self.mixed[task]:
                    break
                current = self.document(current)["collection_append"]["prior_split"]
            for row, expected_split in ((ledger["source_index"][request["source_id"]], request["split"]),
                                        (other, other.get("split"))):
                prep = self.document(row["source_preparation_ref"])
                self.validate(prep)
                source = prep["source"]
                need(source["task"] == task and source["group_id"] == request["group_id"]
                     and source["split"] == expected_split and source["source_id"] == row["source_id"]
                     and row["host"] == self.p["ssh_alias"] and prep["provenance"]["split"] in ancestors,
                     "Changed collection-only family member")
                if "collection_append_provenance" in prep:
                    need(row["collection_append_provenance"] == prep["collection_append_provenance"],
                         "Changed append limitation")
                else:
                    need(row["collection_group_limitation"] == prep["collection_group_compatibility"],
                         "Changed original collection limitation")
            return True
        return allow
