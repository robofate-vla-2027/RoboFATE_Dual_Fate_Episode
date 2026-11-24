"""Temporary, explicitly synthetic configuration; never a real source or grant."""
from pathlib import Path
import tempfile

from robofate_pipeline import evidence
from robofate_pipeline.runtime_policy import PACKAGE

PROJECT = PACKAGE.parent


class RuntimeFixture:
    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-runtime-")
        self.root = Path(self.temporary.name).resolve()
        self.data = self.root / "data"
        self.data.mkdir()
        (self.data / "sessions").mkdir()
        (self.data / "runs").mkdir()
        self.config = self.root / "config"
        self.config.mkdir()
        self.source = self.data / "synthetic-source"
        self.source.mkdir()
        self.session = "200001010000_SYNTHETIC"
        self.run = self.data / "sessions" / self.session
        self.run.mkdir()
        package = self.root / "synthetic-package"
        (package / "bin").mkdir(parents=True)
        (package / "codex-resources").mkdir()
        for name in ("codex", "codex-code-mode-host"):
            (package / "bin" / name).write_text("SYNTHETIC NOT EXECUTABLE\n")
        (package / "codex-resources" / "fixture.txt").write_text("SYNTHETIC\n")
        self.runsc = self.root / "synthetic-runsc"
        self.runsc.write_text("SYNTHETIC NOT EXECUTABLE\n")
        original = PROJECT / "source/teacher/tasks_baseline/RoboDojo/task.toml"
        text = original.read_text().replace('model = "gpt-6-astra"', 'model = "gt-6-as-a"')
        text = text.replace('source = "Workspace"', 'source = ' + repr(str(original.parent / "Workspace")))
        (self.config / "task.toml").write_text(text)
        self.source_ref = evidence.exclusive(self.source / "context.json", dict(
            data_kind="synthetic", task="synthetic_task", source_id="SYNTHETIC_SOURCE",
            native_total_steps=4))
        files = list(PACKAGE.rglob("*.py")) + list((PROJECT / "source/teacher").rglob("*.py"))
        files += [PROJECT / "source/native/XPolicyLab/model_template.py", self.runsc, self.config / "task.toml"]
        files += [p for p in package.rglob("*") if p.is_file()]
        self.manifest_ref = evidence.exclusive(self.config / "manifest.json", {
            "schema": "robofate.runtime-files.v1",
            "files": [evidence.file_ref(p, allow_empty=True) for p in sorted(files)]})
        self.policy = dict(
            schema="robofate.execution-policy.v1", data_kind="synthetic", data_root=str(self.data),
            teacher_root=str(PROJECT / "source/teacher"), native_root=str(PROJECT / "source/native"),
            manifest_ref=self.manifest_ref, execution_owner="SYNTHETIC_EXECUTOR",
            campaign="SYNTHETIC_CAMPAIGN", chat="SYNTHETIC_CHAT",
            expected_hostname="synthetic.invalid", ssh_alias="SYNTHETIC_ALIAS",
            excluded_source_ids=[], task_totals={"synthetic_task": 4},
            teacher_config_ref=evidence.file_ref(self.config / "task.toml"), agent_package=str(package),
            runsc=str(self.runsc), model_upstream="http://127.0.0.1:1", private_runsc_tmpfs=False,
            native_devices=dict(cuda_visible_devices="0,1", nvidia_visible_devices="SYNTHETIC_DEVICES",
                                renderer_index=1), planner_refs={})
        self.policy_ref = evidence.exclusive(self.config / "policy.json", self.policy)
        self.identity = dict(campaign="SYNTHETIC_CAMPAIGN", chat="SYNTHETIC_CHAT",
            active_chat_id="SYNTHETIC_EXECUTOR", session=self.session, task="synthetic_task",
            source_id="SYNTHETIC_SOURCE", owner="SYNTHETIC_OWNER", epoch=1)
        self.binding = dict(schema="robofate.runtime.v1", data_kind="synthetic",
            execution_policy_ref=self.policy_ref, execution_owner="SYNTHETIC_EXECUTOR",
            campaign="SYNTHETIC_CAMPAIGN", task="synthetic_task", native_limit=4, t1=1, actor_wall_s=3600,
            source_id="SYNTHETIC_SOURCE", session=self.session, run_dir=str(self.run), identity=self.identity,
            source_context_ref=self.source_ref, runtime_files=[], source_commit="SYNTHETIC_REVISION")
        self.main_ref = evidence.exclusive(self.data / "main.json", dict(
            schema="robofate.main-binding.v1", identity=self.identity, source_context_ref=self.source_ref,
            native_used=1, native_limit=4, remaining_steps=3))
        self.context = dict(self.identity, native_used=1, native_limit=4, remaining_steps=3,
            dm_submissions_disabled=True, dm_inflight=0, dm_queue_discarded=True, native_terminal=False,
            boundary_receipt_ref="SYNTHETIC_BOUNDARY", budget_mapping_ref="SYNTHETIC_BUDGET")
        self.observation = dict(instruction="SYNTHETIC task", state={"robot_arm_joint_state": [0.]})
        self.history = [dict(native_step=i, observation=self.observation,
            executed_action=None if i == 0 else {"robot_arm_joint_state": [0.]},
            feedback=dict(success=False, terminated=False, truncated=False)) for i in (0, 1)]

    def close(self):
        self.temporary.cleanup()
