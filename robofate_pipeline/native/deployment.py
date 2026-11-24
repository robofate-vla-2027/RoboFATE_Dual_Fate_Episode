"""Pinned deployment environment; inspection alone never grants execution."""
import os
from pathlib import Path
import shlex
from urllib.parse import urlsplit

from .. import evidence
from ..evidence import need
from ..runtime_policy import PACKAGE

KIT_KEY = "EMBODIEDEVO_ROBODOJO_KIT_ARGS"
LAYER = "VK_LAYER_REFATE_image_copy_staging"


def pinned(policy, reference):
    raw = policy.pin(reference["path"])
    need(policy.pins[reference["path"]] == reference, "Deployment/release pin mismatch")
    return raw


def executable(policy, entry):
    path = Path(entry["path"])
    need(path.is_absolute() and path.parent.resolve(strict=True) == path.parent,
         "Canonical executable parent required")
    need(str(path.resolve(strict=True)) == entry["target_ref"]["path"] and os.access(path, os.X_OK),
         "Executable target or permissions changed")
    pinned(policy, entry["target_ref"])
    return str(path)


def route(policy):
    endpoint = urlsplit(policy.value["model_upstream"])
    need(endpoint.scheme == "http" and endpoint.hostname == "127.0.0.1"
         and endpoint.port is not None and 0 < endpoint.port < 65536
         and not endpoint.username and not endpoint.password
         and endpoint.path in ("", "/") and not endpoint.query and not endpoint.fragment,
         "Explicit local model-forwarder address required")
    return endpoint.hostname, endpoint.port


def extension(policy, source):
    expected = ["isaacsim.asset.gen.conveyor"] if source["task"] == "pick_from_conveyor_by_image" else []
    need(source.get("native_kit_extensions", []) == expected, "Exact source extension set required")
    if not expected:
        return None
    configured = policy.value["deployment"]["conveyor"]
    root = policy.directory(configured["root"])
    need(root.name == expected[0], "Installed Conveyor root differs")
    def read_ref(reference, expected_path=None):
        if expected_path is not None:
            need(reference["path"] == str(expected_path), "Installed extension path differs")
        return pinned(policy, reference)
    api = policy.load("native/extension_impl.py", dict(
        ROOT=root, PIN=configured["pins_ref"], need=need, decode=evidence.decode, read_ref=read_ref))
    return api.verify(source)


def source_extensions(policy, context):
    source = evidence.decode(evidence.read_pin(context["source_context_ref"], roots=(policy.root,)))
    need(source["task"] == context["task"] and source["source_id"] == context["source_id"]
         and type(source["native_total_steps"]) is int
         and source["native_total_steps"] == context["native_limit"], "Teacher/source extension scope differs")
    return extension(policy, source)


def vulkan(policy):
    binding = policy.value["deployment"]["vulkan"]
    manifest = evidence.decode(pinned(policy, binding["manifest_ref"]))
    expected = evidence.decode((PACKAGE / "native/vulkan/VkLayer_REFATE_image_copy_staging.json").read_bytes())
    need(manifest == expected, "Original Vulkan layer descriptor differs")
    root = Path(binding["manifest_ref"]["path"]).parent
    library = root / expected["layer"]["library_path"]
    need(str(library.resolve(strict=True)) == binding["library_ref"]["path"]
         and pinned(policy, binding["library_ref"])[:4] == b"\x7fELF",
         "Pinned Linux Vulkan library required")
    need(sorted(p.name for p in root.glob("*.json")) == ["VkLayer_REFATE_image_copy_staging.json"],
         "Unexpected additional Vulkan layer manifest")
    receipt = evidence.decode(pinned(policy, binding["build_ref"]))
    need(Path(binding["build_ref"]["path"]) == root / "build-evidence/BUILD.json"
         and receipt["schema"] == "robofate.vulkan-build.v1"
         and receipt["library_ref"] == binding["library_ref"]
         and receipt["expected_hostname"] == policy.value["expected_hostname"]
         and receipt["ssh_alias"] == policy.value["ssh_alias"]
         and receipt["execution_owner"] == policy.value["execution_owner"],
         "Vulkan compile-time deployment guard differs")
    for name in ("copy_layer.cpp", "copy_plan.hpp", "VkLayer_REFATE_image_copy_staging.json"):
        local = PACKAGE / "native/vulkan" / name
        need(receipt["sources"][name] == evidence.file_ref(local), "Vulkan build/source binding differs")
    return root


def plan(policy, context):
    """Validate pins and return a fresh deterministic environment, without writing."""
    policy.unchanged()
    p, d = policy.value, policy.value["deployment"]
    run = Path(context["run_dir"])
    need(run.resolve(strict=True) == run and run.parent == policy.root / "runs",
         "Canonical deployment run required")
    native_python = executable(policy, d["native_python"])
    ffmpeg = executable(policy, d["ffmpeg"])
    need(Path(ffmpeg).name == "ffmpeg", "Original video executable name required")
    runtime = run / "runtime"
    paths = [str(policy.directory(v)) for v in d["path_dirs"]]
    libraries = [str(policy.directory(v)) for v in d["native_library_dirs"]]
    need(paths and libraries, "Explicit native executable/library search paths required")
    preloads = [str(Path(ref["path"])) for ref in d["native_preload_refs"]]
    for ref in d["native_preload_refs"]:
        pinned(policy, ref)
    cuda = policy.directory(d["cuda_home"])
    icd = evidence.decode(pinned(policy, d["icd_ref"]))
    need(icd["ICD"]["library_path"] == d["icd_library_ref"]["path"]
         and pinned(policy, d["icd_library_ref"])[:4] == b"\x7fELF",
         "Pinned ICD library path required; implicit linker lookup is not accepted")
    optix = evidence.decode(pinned(policy, d["optix_manifest_ref"]))
    raw = pinned(policy, {"path": optix["path"], "sha256": optix["sha256"]})
    need(type(optix["bytes"]) is int and optix["bytes"] == len(raw), "OptiX asset size differs")
    layer = vulkan(policy)
    ext_ref = source_extensions(policy, context)
    devices = p["native_devices"]
    need(devices["cuda_visible_devices"] == "0,1" and devices["renderer_index"] == 1,
         "Original container logical planner/renderer pair required")
    need(devices["visible_gpu_indices"] == [0, 1] and len(devices["gpu_uuids"]) == 2
         and devices["nvidia_visible_devices"] == ",".join(devices["gpu_uuids"]),
         "Explicit container UUID enumeration required")
    need(d["accept_simulator_eula"] is True, "Deployment must explicitly accept simulator EULA")
    import_paths = [PACKAGE.parent, policy.teacher, policy.native, policy.native / "XPolicyLab",
                    policy.native / "third_party/IsaacLab/source/isaaclab",
                    policy.native / "third_party/IsaacLab/source/isaaclab_tasks",
                    policy.native / "third_party/curobo"]
    for path in import_paths:
        policy.directory(str(path))
    # Do not inherit credentials, Python hooks, proxies or arbitrary renderer flags.
    env = {k: v for k, v in os.environ.items()
           if k in ("LANG", "LC_ALL", "LC_CTYPE", "TZ") or k.startswith(("SLURM_", "PBS_", "LSB_"))}
    env.update(
        ROBOFATE_SSH_ALIAS=p["ssh_alias"], ROBOFATE_EXECUTION_OWNER=p["execution_owner"],
        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1",
        OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        HOME=str(runtime / "home"), TMPDIR=str(runtime / "tmp"), XDG_CACHE_HOME=str(runtime / "cache"),
        XDG_RUNTIME_DIR=str(runtime / "xdg"), CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=devices["cuda_visible_devices"],
        NVIDIA_VISIBLE_DEVICES=devices["nvidia_visible_devices"],
        CUDA_HOME=str(cuda), PATH=":".join([str(Path(ffmpeg).parent), str(cuda / "bin")] + paths),
        LD_LIBRARY_PATH=":".join(libraries), TORCH_EXTENSIONS_DIR=str(runtime / "torch_extensions"),
        VK_ICD_FILENAMES=d["icd_ref"]["path"], OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y",
        PYTHONNOUSERSITE="1", VK_LAYER_PATH=str(layer), VK_INSTANCE_LAYERS=LAYER,
        REFATE_VK_COPY_MODE="stage", REFATE_VK_COPY_LOG=str(run / "copy_layer.log"),
        OMNI_VULKAN_ENV_SKIP="0", ROBOFATE_OPTIX_MANIFEST=d["optix_manifest_ref"]["path"],
        PYTHONPATH=":".join(map(str, import_paths)), MODEL_UPSTREAM=p["model_upstream"],
        ROBODOJO_PHYSICAL_GPU=str(devices["physical_gpu_ids"][1]))
    if preloads:
        env["LD_PRELOAD"] = ":".join(preloads)
    tokens = ["--portable-root", str(runtime / "kit"), "--enable", "isaacsim.sensors.camera",
              "--enable", "isaacsim.replicator.behavior",
              "--/exts/omni.kit.renderer.core/autostartImGuiRenderer=false",
              "--/persistent/rtx/modes/rt/enabled=true", "--/persistent/rtx/modes/pt/enabled=true"]
    if ext_ref is not None:
        tokens += ["--enable", "isaacsim.asset.gen.conveyor"]
    env[KIT_KEY] = shlex.join(tokens)
    route(policy)
    return native_python, env


def prepare(policy, context):
    policy.host_guard()
    policy.verify_release()
    command, env = plan(policy, context)
    runtime = Path(context["run_dir"]) / "runtime"
    runtime.mkdir(mode=0o700)
    for name in ("home", "tmp", "cache", "torch_extensions", "kit", "xdg"):
        (runtime / name).mkdir(mode=0o700)
    return command, env


def verify_environment(policy, context):
    _, expected = plan(policy, context)
    for key, value in expected.items():
        need(os.environ.get(key) == value, "Native environment drift: " + key)
    if "LD_PRELOAD" not in expected:
        need(not os.environ.get("LD_PRELOAD"), "Unexpected Native preload")
    need(not os.environ.get("PYTHONHOME") and not os.environ.get("PYTHONSTARTUP"),
         "Unexpected Python runtime override")
