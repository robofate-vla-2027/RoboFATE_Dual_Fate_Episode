"""Bind original teacher implementations without historical machine constants."""
import copy
import os
from pathlib import Path
import re
from types import SimpleNamespace
from urllib.parse import urlsplit

from .. import evidence
from ..evidence import canonical, need
from ..runtime_policy import RuntimePolicy, PACKAGE
from ..recording.predict import ExistingEnvTeacher

IDENTITY = {"campaign", "chat", "active_chat_id", "session", "task", "source_id", "owner", "epoch"}


class TeacherRuntime:
    need = staticmethod(need)
    ExistingEnvTeacher = ExistingEnvTeacher

    def __init__(self, binding):
        self.binding = copy.deepcopy(binding)
        self.policy = RuntimePolicy(binding["execution_policy_ref"])
        self.policy.install_import_paths()
        self.validate_binding()
        self.policy.verify_release()
        from integrations.robodojo_policy.bridge import BridgeTask, policy_observation, validate_action
        from XPolicyLab.model_template import ModelTemplate
        from orchestrator.files import append_jsonl, write_json
        from .worker_entry import run_worker

        p, b = self.policy.value, self.binding
        self.main_binding = SimpleNamespace(load_binding=self.load_binding, derive_task_config=self.derive_task_config)
        self.takeover = self.policy.load("teacher/takeover_impl.py", dict(
            BridgeTask=BridgeTask, policy_observation=policy_observation, validate_action=validate_action,
            load_binding=self.load_binding, derive_task_config=self.derive_task_config,
            verify_release=self.policy.verify_release, task_config=self.task_config,
            OUTPUT=self.policy.root, CAMPAIGN=p["campaign"], CHAT=p["chat"],
            TASK=b["task"], ACTIVE_CHAT_ID=p["execution_owner"], NATIVE_LIMIT=b["native_limit"]))
        self.stage = self.make_stage()
        resources = self.policy.load("teacher/resources_impl.py", dict(stage=self.stage))
        self.original_worker = self.policy.load("teacher/worker_impl.py", dict(
            agent_command=resources.agent_command, AgentResources=resources.AgentResources,
            **{n: getattr(self.takeover, n) for n in ("TakeoverBridgeTask", "validate_worker_spec",
                                                    "append_context_instruction", "require")}))
        model = self.policy.load("teacher/model_impl.py", dict(
            ModelTemplate=ModelTemplate, append_jsonl=append_jsonl, write_json=write_json,
            run_worker=run_worker, load_binding=self.load_binding, _execution_guard=self.execution_guard,
            **{n: getattr(self.takeover, n) for n in ("bind_model", "accept_model_transition", "require", "PUBLIC_KEYS")}))
        self.Model = model.Model

    def validate_binding(self):
        self.policy.unchanged()
        b, p = self.binding, self.policy.value
        need(b.get("schema") == "robofate.runtime.v1" and b.get("data_kind") == p["data_kind"],
             "Wrong teacher runtime schema/scope")
        ident = b["identity"]
        need(type(ident) is dict and set(ident) == IDENTITY
             and all(type(ident[k]) is str and ident[k] for k in IDENTITY - {"epoch"})
             and type(ident["epoch"]) is int and ident["epoch"] > 0, "Complete teacher identity required")
        need(b["campaign"] == ident["campaign"] == p["campaign"]
             and ident["chat"] == p["chat"] and b["execution_owner"] == ident["active_chat_id"] == p["execution_owner"]
             and b["source_id"] == ident["source_id"] and b["task"] == ident["task"]
             and b["session"] == ident["session"], "Teacher runtime identity drift")
        need(re.fullmatch(r"[0-9]{12}_[A-Za-z0-9_-]{1,60}", b["session"]), "Fresh session token required")
        need(type(b["native_limit"]) is int and p["task_totals"].get(b["task"]) == b["native_limit"]
             and type(b["t1"]) is int and 0 < b["t1"] < b["native_limit"]
             and type(b["actor_wall_s"]) is int and b["actor_wall_s"] == 3600, "Original teacher budget required")
        need(b["source_id"] not in p["excluded_source_ids"], "Excluded source")
        need(type(b.get("source_commit")) is str and b["source_commit"], "External release identity required")
        run = Path(b["run_dir"])
        need(run == self.policy.root / "sessions" / b["session"] and run.resolve() == run,
             "Canonical source-private session required")
        source = evidence.decode(evidence.read_pin(b["source_context_ref"], roots=(self.policy.root,)))
        need(source["data_kind"] == b["data_kind"] and source["task"] == b["task"]
             and source["source_id"] == b["source_id"] and source["native_total_steps"] == b["native_limit"],
             "Pinned source identity/budget differs")
        for ref in b["runtime_files"]:
            self.policy.pin(ref["path"])
            need(self.policy.pins[ref["path"]] == ref, "Runtime file differs from release pin")
        self.task_config()
        return copy.deepcopy(b)

    def task_config(self):
        from orchestrator.config import config_dict, load_task
        reference = self.policy.value["teacher_config_ref"]
        self.policy.pin(reference["path"])
        need(self.policy.pins[reference["path"]] == reference and Path(reference["path"]).name == "task.toml",
             "Exact external teacher task.toml pin required")
        return self.derive_task_config(config_dict(load_task(Path(reference["path"]).parent, env_task=self.binding["task"])))

    def derive_task_config(self, original):
        value = copy.deepcopy(original)
        need(value.get("baseline") is True and value["simulator"]["env_task"] == self.binding["task"],
             "Task must be selected through the original configuration loader")
        value["test"]["shared_plan"] = None
        return value

    def load_binding(self, reference, *, session=None, context=None):
        b = self.binding
        actual = evidence.decode(evidence.read_pin(reference, roots=(self.policy.root,)))
        need(actual.get("schema") == "robofate.main-binding.v1" and actual.get("identity") == b["identity"],
             "Wrong actual run identity")
        need(actual["source_context_ref"] == b["source_context_ref"]
             and type(actual["native_used"]) is int and actual["native_used"] == b["t1"]
             and type(actual["native_limit"]) is int and actual["native_limit"] == b["native_limit"]
             and type(actual["remaining_steps"]) is int and actual["remaining_steps"] == b["native_limit"] - b["t1"],
             "Original source/budget drift")
        if session is not None:
            need(session == b["session"], "Wrong worker session")
        if context is not None:
            for key, value in b["identity"].items():
                need(type(context.get(key)) is type(value) and context[key] == value,
                     "Actual takeover identity drift: " + key)
            need(type(context.get("native_used")) is int and context["native_used"] == b["t1"]
                 and type(context.get("native_limit")) is int and context["native_limit"] == b["native_limit"],
                 "Actual takeover budget drift")
        return actual

    def execution_guard(self):
        self.validate_binding()
        self.policy.host_guard()
        marker = Path(self.binding["run_dir"]) / "agent/provider_safety.json"
        need(not marker.exists() and not marker.is_symlink(), "Provider stop retained; no restart")
        need(not os.environ.get("EMBODIEDEVO_AGENT_RECOVERY", "").strip(" {}"),
             "Automatic teacher recovery is not allowed")

    def make_stage(self):
        p, policy = self.policy.value, self.policy
        endpoint = urlsplit(p["model_upstream"])
        need(endpoint.scheme == "http" and endpoint.hostname == "127.0.0.1" and endpoint.port
             and not endpoint.username and not endpoint.password and endpoint.path in ("", "/")
             and not endpoint.query and not endpoint.fragment,
             "Explicit existing loopback model forwarding route required")
        package = policy.directory(p["agent_package"])
        for path in (Path(p["runsc"]), package / "bin/codex", package / "bin/codex-code-mode-host"):
            policy.pin(path)
        resource = policy.directory(package / "codex-resources")
        for path in resource.rglob("*"):
            need(not path.is_symlink(), "Linked agent resource")
            if path.is_file():
                policy.pin(path)
        stage = policy.load("teacher/stage_impl.py", dict(
            need=need, write_new=evidence.exclusive, host_guard=policy.host_guard,
            HERE=PACKAGE / "teacher", SHARED=policy.root, OWNER=p["execution_owner"],
            CHAT=p["chat"], PACKAGE=package, RUNSC=Path(p["runsc"])))
        stage.policy_ref = copy.deepcopy(policy.reference)
        stage.MODEL_UPSTREAM = p["model_upstream"]
        stage.transport = lambda: policy.load("teacher/transport_impl.py")
        return stage


def install(binding):
    """No Native, process, network listener or model request is created here."""
    policy = RuntimePolicy(binding["execution_policy_ref"])
    return TeacherRuntime(binding)
