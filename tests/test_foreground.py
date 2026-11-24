"""Synthetic CPU lifecycle/deployment cases, with no real launches or signals.

Only temporary files/locks are real. Process, listener and signal operations are
mocked; the outer audit guard forbids actual peers, models, GPUs and subprocesses.
These fixtures are not source admissions, production receipts or qualifications.
"""
import copy
import fcntl
import os
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from robofate_pipeline import evidence, foreground as fg
from robofate_pipeline.native import deployment
from robofate_pipeline.runtime_policy import PACKAGE, RuntimePolicy
from synthetic_runtime import RuntimeFixture


class ForegroundTests(unittest.TestCase):
    def setUp(self):
        self.f = RuntimeFixture()
        self.addCleanup(self.f.close)
        self.run = self.f.data / "runs" / "200001010000_SYNTHETIC"
        self.run.mkdir()
        (self.f.run / "inputs").mkdir()
        self.context = dict(data_kind="synthetic", run_dir=str(self.run), source_id="SYNTHETIC_SOURCE",
            task="synthetic_task", native_limit=4, native_wall_seconds=3600,
            runtime_binding={"session": self.f.session, "run_dir": str(self.f.run)},
            source_context_ref=self.f.source_ref)
        self.reference = evidence.exclusive(self.run / "CONTEXT_001.json", self.context)
        self.owner = dict(pid=731, start_ticks=11, boot_id="SYNTHETIC_BOOT", process_group=731)
        self.policy = RuntimePolicy(self.f.policy_ref)

    def deployment_fixture(self):
        root = self.f.config / "synthetic-deployment"
        root.mkdir()
        library = root / "lib"
        library.mkdir()
        cuda = root / "cuda"
        (cuda / "bin").mkdir(parents=True)
        files = []
        def file(path, text, executable=False):
            path.write_bytes(text)
            if executable:
                path.chmod(0o700)
            files.append(path)
            return evidence.file_ref(path)
        def document(path, value):
            ref = evidence.exclusive(path, value)
            files.append(path)
            return ref
        python = file(root / "synthetic-python", b"SYNTHETIC NOT EXECUTABLE\n", True)
        ffmpeg = file(root / "ffmpeg", b"SYNTHETIC NOT EXECUTABLE\n", True)
        elf = file(library / "synthetic-icd.so", b"\x7fELF_SYNTHETIC_NOT_LOADABLE")
        icd = document(root / "icd.json", {"ICD": {"library_path": elf["path"]}})
        optix = file(root / "synthetic-optix", b"SYNTHETIC_OPTIX")
        optix_manifest = document(root / "optix.json", dict(optix, bytes=len(b"SYNTHETIC_OPTIX")))
        layer = root / "vulkan"
        layer.mkdir()
        (layer / "build-evidence").mkdir()
        manifest = file(layer / "VkLayer_REFATE_image_copy_staging.json",
                       (PACKAGE / "native/vulkan/VkLayer_REFATE_image_copy_staging.json").read_bytes())
        layer_ref = file(layer / "libVkLayer_REFATE_image_copy_staging.so", b"\x7fELF_SYNTHETIC_NOT_LOADABLE")
        build_ref = document(layer / "build-evidence/BUILD.json", dict(
            schema="robofate.vulkan-build.v1", expected_hostname="synthetic.invalid",
            ssh_alias="SYNTHETIC_ALIAS", execution_owner="SYNTHETIC_EXECUTOR", library_ref=layer_ref,
            sources={p.name: evidence.file_ref(p) for p in (PACKAGE / "native/vulkan").iterdir()}))
        extension = root / "isaacsim.asset.gen.conveyor"
        (extension / "config").mkdir(parents=True)
        (extension / "bin").mkdir()
        toml = file(extension / "config/extension.toml", b"""[package]
version = "1.0.23"
[[python.module]]
name = "isaacsim.asset.gen.conveyor"
[[native.plugin]]
path = "bin/*.plugin"
recursive = false
""")
        plugin = file(extension / "bin/libisaacsim.asset.gen.conveyor.plugin.so", b"\x7fELF_SYNTHETIC")
        pins = document(root / "extension.json", dict(
            schema="robofate.native-extension.v1", extension="isaacsim.asset.gen.conveyor",
            version="1.0.23", task="pick_from_conveyor_by_image", native_total_steps=700, files=[toml, plugin]))
        policy = copy.deepcopy(self.f.policy)
        policy["deployment"] = dict(
            native_python=dict(path=python["path"], target_ref=python),
            ffmpeg=dict(path=ffmpeg["path"], target_ref=ffmpeg), path_dirs=[str(root)],
            native_library_dirs=[str(library)], native_preload_refs=[], cuda_home=str(cuda),
            icd_ref=icd, icd_library_ref=elf, optix_manifest_ref=optix_manifest, accept_simulator_eula=True,
            vulkan=dict(manifest_ref=manifest, library_ref=layer_ref, build_ref=build_ref),
            conveyor=dict(root=str(extension), pins_ref=pins))
        policy["native_devices"].update(physical_gpu_ids=[4, 5], visible_gpu_indices=[0, 1],
            gpu_uuids=["SYNTHETIC_A", "SYNTHETIC_B"], nvidia_visible_devices="SYNTHETIC_A,SYNTHETIC_B")
        old = evidence.decode(evidence.read_pin(self.f.manifest_ref))
        old["files"] += [evidence.file_ref(p) for p in files]
        policy["manifest_ref"] = evidence.exclusive(root / "all-pins.json", old)
        reference = evidence.exclusive(root / "policy.json", policy)
        return RuntimePolicy(reference)

    def parent_patches(self):
        fake = SimpleNamespace(root=self.policy.root, reference=self.policy.reference,
                               value=self.policy.value, host_guard=Mock())
        self.patches = [
            patch.object(fg, "load_context", return_value=(self.context, None, None, fake)),
            patch.object(fg, "identity", return_value=self.owner),
            patch.object(fg, "group_members", return_value=[]),
            patch.object(fg, "inspect_resources", return_value={"synthetic": True}),
            patch.object(fg.deployment, "prepare", return_value=("SYNTHETIC_PYTHON", {"SYNTHETIC": "true"})),
            patch.object(fg.deployment, "route", return_value=("127.0.0.1", 1)),
            patch.object(fg.socket, "create_connection"),
            patch.object(fg.signal, "signal", return_value=signal.SIG_DFL),
            patch.dict(os.environ, MODEL_UPSTREAM=self.policy.value["model_upstream"]),
        ]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        return fake

    def test_synthetic_policy_refuses_before_listener_or_process(self):
        with patch.object(fg, "load_context", return_value=(self.context, None, None, self.policy)):
            with self.assertRaisesRegex(ValueError, "Synthetic"):
                fg.foreground(self.reference)
        self.assertFalse((self.run / "foreground_started.json").exists())

    def test_parent_inherits_lock_and_keeps_exit_unqualified(self):
        self.parent_patches()
        process = Mock(pid=731)
        process.wait.return_value = 0
        process.poll.return_value = 0
        with patch.object(fg.subprocess, "Popen", return_value=process) as launch, \
             patch.object(fg, "original_absent", return_value=True):
            self.assertEqual(fg.foreground(self.reference), 0)
        kwargs = launch.call_args.kwargs
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(len(kwargs["pass_fds"]), 1)
        self.assertGreater(process.wait.call_args_list[0].kwargs["timeout"], 3590)
        self.assertLessEqual(process.wait.call_args_list[0].kwargs["timeout"], 3600)
        receipt = evidence.document(self.run / "native_wait.json")
        self.assertTrue(receipt["data_status_not_decided_by_exit"])
        self.assertNotIn("qualified_positive", receipt)
        self.assertTrue(evidence.document(self.run / "process_cleanup.json")["owned_cleanup_complete"])
        with self.assertRaisesRegex(ValueError, "reconciliation"):
            fg.foreground(self.reference)

    def test_timeout_preserves_original_and_bounded_owned_cleanup(self):
        self.parent_patches()
        primary = subprocess.TimeoutExpired("SYNTHETIC", 3600)
        process = Mock(pid=731)
        process.wait.side_effect = [primary, -15]
        process.poll.return_value = None
        with patch.object(fg.subprocess, "Popen", return_value=process), \
             patch.object(fg.os, "killpg") as kill, \
             patch.object(fg, "original_absent", return_value=True):
            with self.assertRaises(subprocess.TimeoutExpired) as raised:
                fg.foreground(self.reference)
        self.assertIs(raised.exception, primary)
        kill.assert_called_once_with(731, signal.SIGTERM)
        self.assertFalse(evidence.document(self.run / "foreground_error.json")["automatic_repeat"])

    def test_cleanup_error_never_masks_parent_error(self):
        self.parent_patches()
        process = Mock(pid=731)
        primary = RuntimeError("SYNTHETIC wait failure")
        process.wait.side_effect = primary
        process.poll.return_value = None
        identities = [self.owner, self.owner, self.owner, dict(self.owner, start_ticks=22)]
        with patch.object(fg, "identity", side_effect=identities), \
             patch.object(fg.subprocess, "Popen", return_value=process), \
             patch.object(fg.os, "killpg") as kill:
            with self.assertRaises(RuntimeError) as raised:
                fg.foreground(self.reference)
        self.assertIs(raised.exception, primary)
        self.assertTrue(any("cleanup" in note for note in primary.__notes__))
        kill.assert_not_called()
        self.assertFalse(evidence.document(self.run / "process_cleanup_failed.json")["owned_cleanup_complete"])

    def test_spawn_failure_restores_handlers_and_records_no_success(self):
        self.parent_patches()
        with patch.object(fg.subprocess, "Popen", side_effect=OSError("SYNTHETIC spawn")), \
             patch.object(fg.signal, "signal", return_value=signal.SIG_DFL) as handlers:
            with self.assertRaises(OSError):
                fg.foreground(self.reference)
        for sig in fg.SIGNALS:
            self.assertIn(((sig, signal.SIG_DFL),), [tuple(x)[:1] for x in handlers.call_args_list])
        self.assertFalse((self.run / "process_cleanup.json").exists())
        self.assertFalse(evidence.document(self.run / "foreground_error.json")["qualified"])

    def test_cleanup_rechecks_identity_before_kill(self):
        process = Mock(pid=731)
        process.poll.return_value = None
        process.wait.side_effect = subprocess.TimeoutExpired("SYNTHETIC", 20)
        with patch.object(fg, "identity", side_effect=[self.owner, dict(self.owner, start_ticks=22)]), \
             patch.object(fg.os, "killpg") as kill:
            with self.assertRaisesRegex(ValueError, "refusing signal"):
                fg.stop_owned(process, self.owner)
        kill.assert_called_once_with(731, signal.SIGTERM)

    def test_cleanup_escalates_only_owned_group_with_finite_kill_wait(self):
        process = Mock(pid=731)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("SYNTHETIC", 20), -9]
        with patch.object(fg, "identity", return_value=self.owner), patch.object(fg.os, "killpg") as kill:
            self.assertEqual(fg.stop_owned(process, self.owner), -9)
        self.assertEqual([call.args for call in kill.call_args_list],
                         [(731, signal.SIGTERM), (731, signal.SIGKILL)])
        self.assertEqual([call.kwargs["timeout"] for call in process.wait.call_args_list], [20, 5])

    def test_unknown_or_shared_group_refuses_cleanup(self):
        for owned in (None, dict(self.owner, process_group=730), dict(self.owner, pid=730)):
            process = Mock(pid=731)
            process.poll.return_value = None
            with patch.object(fg, "identity", return_value=self.owner), patch.object(fg.os, "killpg") as kill:
                with self.assertRaises(ValueError):
                    fg.stop_owned(process, owned)
                kill.assert_not_called()

    def test_successful_exit_with_failed_cleanup_is_still_failure(self):
        self.parent_patches()
        process = Mock(pid=731)
        process.wait.return_value = 0
        process.poll.return_value = 0
        with patch.object(fg.subprocess, "Popen", return_value=process), \
             patch.object(fg, "original_absent", return_value=False):
            with self.assertRaisesRegex(ValueError, "still present"):
                fg.foreground(self.reference)
        self.assertFalse(evidence.document(self.run / "process_cleanup.json")["owned_cleanup_complete"])

    def test_reaped_leader_with_remaining_group_is_not_cleanup_success(self):
        process = Mock(pid=731)
        process.poll.return_value = 0
        process.wait.return_value = 0
        residual = dict(self.owner, pid=732)
        with patch.object(fg, "original_absent", return_value=True), \
             patch.object(fg, "group_members", return_value=[residual]), patch.object(fg.os, "killpg") as kill:
            errors = fg.cleanup(process, self.owner, None, self.run)
        self.assertEqual(len(errors), 1)
        kill.assert_not_called()
        result = evidence.document(self.run / "process_cleanup.json")
        self.assertTrue(result["native"]["original_process_absent"])
        self.assertFalse(result["owned_cleanup_complete"])
        self.assertFalse(result["isolated_worker_lifecycle_verified"])

    def test_group_snapshot_filters_peers_without_signaling(self):
        with patch.object(Path, "read_text", return_value="SYNTHETIC_BOOT\n"), \
             patch.object(Path, "iterdir", return_value=iter([Path("731"), Path("732"), Path("733"), Path("self")])), \
             patch.object(fg, "identity", side_effect=[
                 dict(self.owner, process_group=700), dict(self.owner, pid=732), FileNotFoundError()]), \
             patch.object(fg.os, "killpg") as kill:
            self.assertEqual(fg.group_members(self.owner), [dict(self.owner, pid=732)])
        kill.assert_not_called()

    def test_lock_conflict_and_inherited_descriptor_do_not_deadlock(self):
        with fg.simulator_lock(self.policy) as fd:
            with self.assertRaises(BlockingIOError):
                with fg.simulator_lock(self.policy):
                    self.fail("Second open description acquired the same simulator lock")
            with patch.object(fg, "identity", return_value=self.owner):
                reference = fg.lease(self.policy, self.reference, self.run, fd)
                with patch.object(fg.os, "getppid", return_value=731):
                    value = fg.validate_lease(self.policy, self.reference, reference, fd)
                    self.assertEqual(value["descriptor"], fd)
                    self.assertFalse(os.get_inheritable(fd))
        with fg.simulator_lock(self.policy):
            pass

    def test_lease_rejects_unlocked_descriptor(self):
        with fg.simulator_lock(self.policy) as fd, patch.object(fg, "identity", return_value=self.owner):
            reference = fg.lease(self.policy, self.reference, self.run, fd)
            fcntl.flock(fd, fcntl.LOCK_UN)
            with patch.object(fg.os, "getppid", return_value=731):
                with self.assertRaisesRegex(ValueError, "not held"):
                    fg.validate_lease(self.policy, self.reference, reference, fd)

    def test_lease_rejects_parent_reuse_and_wrong_descriptor(self):
        with fg.simulator_lock(self.policy) as fd, patch.object(fg, "identity", return_value=self.owner):
            reference = fg.lease(self.policy, self.reference, self.run, fd)
            with patch.object(fg.os, "getppid", return_value=732):
                with self.assertRaisesRegex(ValueError, "parent"):
                    fg.validate_lease(self.policy, self.reference, reference, fd)
            with self.assertRaisesRegex(ValueError, "binding"):
                fg.validate_lease(self.policy, self.reference, reference, fd + 1)

    def test_lock_replacement_is_rejected(self):
        with fg.simulator_lock(self.policy) as fd, patch.object(fg, "identity", return_value=self.owner):
            reference = fg.lease(self.policy, self.reference, self.run, fd)
            path = self.policy.root / "simulator.lock"
            path.rename(path.with_suffix(".synthetic-old"))
            path.touch()
            with patch.object(fg.os, "getppid", return_value=731):
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    fg.validate_lease(self.policy, self.reference, reference, fd)

    def test_parent_death_setup_detects_race(self):
        libc = Mock()
        libc.prctl.return_value = 0
        with patch.object(fg.ctypes, "CDLL", return_value=libc), patch.object(fg.os, "getppid", return_value=732):
            with self.assertRaisesRegex(ValueError, "changed"):
                fg.parent_death_guard(self.owner)
        libc.prctl.assert_called_once_with(1, signal.SIGTERM, 0, 0, 0)

    def test_deployment_drops_inherited_secrets_and_keeps_original_flags(self):
        policy = self.deployment_fixture()
        with patch.dict(os.environ, {"SYNTHETIC_SECRET": "not-for-child", "PYTHONHOME": "wrong",
                                     "LD_PRELOAD": "wrong", "HTTP_PROXY": "wrong"}):
            python, env = deployment.plan(policy, self.context)
        self.assertTrue(python.endswith("synthetic-python"))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(env["OMNI_VULKAN_ENV_SKIP"], "0")
        self.assertEqual(env["REFATE_VK_COPY_MODE"], "stage")
        self.assertIn("--/persistent/rtx/modes/pt/enabled=true", env[deployment.KIT_KEY])
        self.assertNotIn("conveyor", env[deployment.KIT_KEY])
        for name in ("SYNTHETIC_SECRET", "PYTHONHOME", "LD_PRELOAD", "HTTP_PROXY"):
            self.assertNotIn(name, env)
        self.assertFalse((self.run / "runtime").exists())
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            deployment.prepare(policy, self.context)

    def test_native_environment_drift_is_rejected(self):
        policy = self.deployment_fixture()
        _, env = deployment.plan(policy, self.context)
        with patch.dict(os.environ, env, clear=True):
            deployment.verify_environment(policy, self.context)
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            with self.assertRaisesRegex(ValueError, "environment drift"):
                deployment.verify_environment(policy, self.context)

    def test_forwarder_requires_exact_local_address(self):
        policy = SimpleNamespace(value={})
        for url in ("https://127.0.0.1:1", "http://external.invalid:1", "http://127.0.0.1:1/models",
                    "http://user:secret@127.0.0.1:1", "http://127.0.0.1", "http://127.0.0.1:1/?token=x"):
            policy.value["model_upstream"] = url
            with self.assertRaises(ValueError):
                deployment.route(policy)

    def test_conveyor_requires_original_pinned_budget_and_declaration(self):
        policy = self.deployment_fixture()
        pin = policy.value["deployment"]["conveyor"]["pins_ref"]
        source = dict(task="pick_from_conveyor_by_image", native_total_steps=700,
                      native_kit_extensions=["isaacsim.asset.gen.conveyor"], native_extension_pins=pin)
        self.assertEqual(deployment.extension(policy, source), pin)
        for changes in (dict(native_total_steps=699), dict(native_total_steps=True),
                        dict(native_kit_extensions=[]),
                        dict(native_extension_pins=dict(pin, sha256="0" * 64))):
            with self.assertRaises(ValueError):
                deployment.extension(policy, dict(source, **changes))
        path = Path(policy.value["deployment"]["conveyor"]["root"]) / "config/extension.toml"
        path.write_text(path.read_text().replace("1.0.23", "1.0.24"))
        with self.assertRaisesRegex(ValueError, "changed"):
            deployment.extension(policy, source)

    def test_wrong_source_identity_cannot_enable_conveyor(self):
        policy = self.deployment_fixture()
        with self.assertRaisesRegex(ValueError, "scope"):
            deployment.source_extensions(policy, dict(self.context, source_id="SYNTHETIC_OTHER"))
        with self.assertRaisesRegex(ValueError, "scope"):
            deployment.source_extensions(policy, dict(self.context, native_limit=700))

    def test_only_validated_conveyor_gets_native_enable_flag(self):
        policy = self.deployment_fixture()
        source = evidence.exclusive(self.f.source / "synthetic-conveyor.json", dict(
            data_kind="synthetic", task="pick_from_conveyor_by_image", source_id="SYNTHETIC_SOURCE",
            native_total_steps=700, native_kit_extensions=["isaacsim.asset.gen.conveyor"],
            native_extension_pins=policy.value["deployment"]["conveyor"]["pins_ref"]))
        context = dict(self.context, task="pick_from_conveyor_by_image", native_limit=700,
                       source_context_ref=source)
        _, env = deployment.plan(policy, context)
        self.assertEqual(env[deployment.KIT_KEY].count("--enable isaacsim.asset.gen.conveyor"), 1)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1")

    def test_vulkan_manifest_extra_layer_or_changed_bytes_rejected(self):
        policy = self.deployment_fixture()
        root = deployment.vulkan(policy)
        (root / "extra.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "additional"):
            deployment.vulkan(policy)
        (root / "extra.json").unlink()
        Path(policy.value["deployment"]["vulkan"]["library_ref"]["path"]).write_bytes(b"\x7fELF_CHANGED")
        with self.assertRaisesRegex(ValueError, "changed"):
            deployment.vulkan(policy)

    def test_child_cannot_run_without_foreground_lease(self):
        from robofate_pipeline import execution
        fake = SimpleNamespace(value={"execution_enabled": True}, host_guard=Mock())
        with patch.object(execution, "load_context", return_value=(self.context, None, None, fake)):
            with self.assertRaisesRegex(ValueError, "foreground lease"):
                execution.run_teacher(self.reference)

    def test_vulkan_guard_is_compile_time_and_copy_plan_keeps_caps(self):
        source = PACKAGE / "native/vulkan"
        layer = (source / "copy_layer.cpp").read_text()
        self.assertIn('#include "deployment_binding.hpp"', layer)
        self.assertIn('std::strcmp(host,ROBOFATE_EXPECTED_HOST)', layer)
        self.assertIn('std::strcmp(alias,ROBOFATE_EXPECTED_ALIAS)', layer)
        self.assertIn('ROBOFATE_EXPECTED_OWNER', layer)
        self.assertIn("2ull*1024*1024*1024-b.bytes", layer)
        self.assertIn("512ull*1024*1024", (source / "copy_plan.hpp").read_text())


if __name__ == "__main__":
    unittest.main()
