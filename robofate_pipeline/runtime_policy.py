"""Pinned external runtime configuration, without implicit launch authority."""
import copy
import hashlib
import os
from pathlib import Path
import re
import socket
import sys
from types import ModuleType
import uuid

from . import evidence
from .evidence import need

PACKAGE = Path(__file__).resolve().parent


class RuntimePolicy:
    def __init__(self, reference):
        self.reference = copy.deepcopy(reference)
        self.value = evidence.decode(evidence.read_pin(reference))
        p = self.value
        need(p.get("schema") == "robofate.execution-policy.v1"
             and p.get("data_kind") in ("actual", "synthetic"), "Explicit execution policy required")
        self.root = self.directory(p["data_root"])
        self.teacher = self.directory(p["teacher_root"])
        self.native = self.directory(p["native_root"])
        need(not self.root.is_relative_to(PACKAGE.parent) and not PACKAGE.parent.is_relative_to(self.root),
             "Runtime output must stay outside the source checkout")
        need(not self.root.is_relative_to(self.teacher) and not self.teacher.is_relative_to(self.root)
             and not self.root.is_relative_to(self.native) and not self.native.is_relative_to(self.root),
             "Runtime data and dependency sources must be separate")
        self.manifest = evidence.decode(evidence.read_pin(p["manifest_ref"]))
        need(self.manifest.get("schema") == "robofate.runtime-files.v1"
             and type(self.manifest.get("files")) is list and self.manifest["files"],
             "Explicit runtime dependency inventory required")
        self.pins = {}
        for ref in self.manifest["files"]:
            path = ref["path"]
            need(path not in self.pins, "Duplicate runtime file")
            self.pins[path] = copy.deepcopy(ref)
        need(all(str(path) in self.pins for path in PACKAGE.rglob("*.py")),
             "Pin every installed pipeline Python source")
        need(all(type(p.get(k)) is str and p[k] for k in
                 ("execution_owner", "campaign", "chat", "expected_hostname", "ssh_alias")),
             "Missing explicit runtime identity")
        need(type(p.get("excluded_source_ids")) is list, "Explicit excluded-source list required")
        need(type(p.get("task_totals")) is dict and p["task_totals"]
             and all(type(k) is str and re.fullmatch(r"[A-Za-z0-9_]+", k)
                     and type(v) is int and v > 0 for k, v in p["task_totals"].items()),
             "Explicit task budget table required")

    @staticmethod
    def directory(value):
        path = Path(value)
        need(path.is_absolute() and path.resolve(strict=True) == path and path.is_dir(),
             "Canonical existing directory required")
        return path

    def unchanged(self):
        need(evidence.decode(evidence.read_pin(self.reference)) == self.value
             and evidence.decode(evidence.read_pin(self.value["manifest_ref"])) == self.manifest,
             "Runtime configuration changed")

    def pin(self, path):
        path = str(path)
        need(path in self.pins, "Runtime dependency is not pinned: " + path)
        return evidence.read_pin(self.pins[path], allow_empty=True)

    def verify_release(self):
        self.unchanged()
        for ref in self.pins.values():
            evidence.read_pin(ref, allow_empty=True)
        return {name: ref["sha256"] for name, ref in self.pins.items()}

    def host_guard(self):
        self.unchanged()
        p = self.value
        need(p["data_kind"] == "actual", "Synthetic policy cannot start an actual actor")
        need(p.get("execution_enabled") is True, "Explicit execution enable required")
        need(socket.gethostname() == p["expected_hostname"]
             and os.environ.get("ROBOFATE_SSH_ALIAS") == p["ssh_alias"]
             and os.environ.get("ROBOFATE_EXECUTION_OWNER") == p["execution_owner"],
             "Actual host/alias/execution-owner mismatch")

    def install_import_paths(self):
        # Pin the entire small teacher Python tree, not only top-level imports.
        for path in self.teacher.rglob("*.py"):
            self.pin(path)
        self.pin(self.native / "XPolicyLab/model_template.py")
        for prefix, root in (("orchestrator", self.teacher), ("simulators", self.teacher),
                             ("integrations", self.teacher), ("environments", self.teacher),
                             ("XPolicyLab", self.native)):
            for name, module in tuple(sys.modules.items()):
                if name == prefix or name.startswith(prefix + "."):
                    path = getattr(module, "__file__", None)
                    need(path is not None and Path(path).resolve().is_relative_to(root),
                         "Conflicting upstream module: " + name)
        for path in reversed((self.teacher, self.native)):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

    def load(self, relative, namespace=None):
        path = PACKAGE / relative
        raw = self.pin(path)
        name = "robofate_pipeline." + path.parent.name + "._bound_" + uuid.uuid4().hex
        module = ModuleType(name)
        module.__file__ = str(path)
        module.__package__ = "robofate_pipeline." + path.parent.name
        module.__dict__.update(namespace or {})
        sys.modules[name] = module
        try:
            exec(compile(raw, str(path), "exec"), module.__dict__)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        return module
