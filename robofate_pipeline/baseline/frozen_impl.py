"""Original frozen DM service algorithms; external binding supplies paths."""
from contextlib import contextmanager
from enum import Enum
import hashlib
import json
from pathlib import Path
import time
from types import MethodType
STATE_DIMS = {'left_arm_joint_state': 6, 'left_ee_joint_state': 1, 'right_arm_joint_state': 6, 'right_ee_joint_state': 1}
FIXED_CONFIG = {'policy_name': 'OpenDM', 'protocol': 'ws', 'host': '127.0.0.1', 'bench_name': 'RoboDojo', 'env_cfg_type': 'arx_x5', 'seed': 0, 'action_type': 'joint', 'action_dim': 14, 'eval_batch': False, 'model_action_mode': 'absolute', 'action_chunk_size': 50, 'action_steps': 25, 'history_enabled': True, 'history_image_key': 'images_1', 'history_slots': 20, 'history_fps': 1.0, 'runtime_fps': 25.0, 'history_action_interval': 25, 'history_tokens_per_slot': 16, 'diffusion_steps': 10, 'diffusion_integration_dtype': 'model', 'diffusion_noise_seed': None, 'add_state': True, 'n_bins': 256, 'model_max_length': 1536, 'bf16': False, 'enable_bf16_compute': True, 'force_fp32_action_path': True, 'llm_attn_implementation': 'sdpa', 'vision_attn_implementation': 'sdpa', 'action_attn_implementation': 'sdpa', 'liger_kernel': True, 'control_mode': None, 'speed': '0.5'}
LOADING_STRATEGY = 'native_from_pretrained_direct_cuda_fp32'
CAMERAS = dict(zip(("images_1", "images_2", "images_3"), ("cam_head", "cam_left_wrist", "cam_right_wrist")))

def validate_config(config, port, *, construct_on_device=False):
    if construct_on_device:
        raise ValueError("--construct-on-device is unverified for DM0.5 and is forbidden")
    if not isinstance(config, dict) or config.get("policy_name") != "OpenDM":
        raise ValueError("DM0.5/OpenDM only; XR1 and QwenPI configurations are rejected")
    allowed = set(FIXED_CONFIG) | {
        "port", "model_path", "norm_stats_path", "experiment_path", "task_name",
        "prompt", "robot_type", "ckpt_name", "gpu_id",
    }
    if set(config) - allowed:
        raise ValueError(f"Unsupported DM0.5 config keys: {sorted(set(config) - allowed)}")
    for key, expected in FIXED_CONFIG.items():
        value = config.get(key, expected)
        if value != expected or (isinstance(expected, bool) and type(value) is not bool):
            raise ValueError(f"Validated DM0.5 configuration requires {key}={expected!r}")
    if config.get("ckpt_name") is not None or config.get("gpu_id") is not None:
        raise ValueError("No checkpoint/GPU fallback; use model_path and CUDA_VISIBLE_DEVICES")
    if Path(config.get("model_path") or "").resolve() != MODEL_PATH.resolve():
        raise ValueError(f"Only the validated DM0.5 model_path is allowed: {MODEL_PATH}")
    norm = Path(config.get("norm_stats_path") or MODEL_PATH / "norm_stats.json").resolve()
    if norm != (MODEL_PATH / "norm_stats.json").resolve():
        raise ValueError("DM0.5 requires its original normalization statistics")
    experiment = Path(config.get("experiment_path") or "scripts/robodojo_dm05_history.py")
    if not experiment.is_absolute():
        experiment = NATIVE_ROOT / "XPolicyLab/policy/OpenDM" / experiment
    if experiment.resolve() != EXPERIMENT.resolve():
        raise ValueError("DM0.5 requires its validated history experiment")
    if config.get("robot_type") not in (None, "Dual ARX5"):
        raise ValueError("DM0.5 requires Dual ARX5")
    if type(port) is not int or not 0 <= port < 65536:
        raise ValueError("port must be in [0, 65535]; 0 selects an ephemeral loopback port")
    return {**config, **FIXED_CONFIG, "model_path": str(MODEL_PATH),
            "norm_stats_path": str(norm), "experiment_path": str(EXPERIMENT), "port": port}


def plain(value):
    if isinstance(value, Enum):
        return plain(value.value)
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value.tolist() if hasattr(value, "tolist") else value


def emit(audit, kind, **fields):
    audit.write(json.dumps({"kind": kind, "monotonic_ns": time.monotonic_ns(),
                            **plain(fields)}, allow_nan=False) + "\n")
    audit.flush()


def image_record(image):
    import numpy as np
    array = np.ascontiguousarray(image)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def native14(state):
    import numpy as np
    result = {}
    for key, size in STATE_DIMS.items():
        value = np.asarray(state[key])
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"Expected finite native {key}[{size}], got {value.shape}")
        result[key] = value
    return result


def scope_id(scope):
    value = scope.get("evaluation_id") if isinstance(scope, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Use one explicit evaluation_id for the entire episode, including preparation")
    return value


@contextmanager
def direct_gpu_loading(native):
    """Override only this construction's native config loader, then restore it."""
    original_experiment_loader = native._load_experiment
    overrides = []
    calls = 0
    missing = object()

    def load_experiment(model_config):
        experiment, path = original_experiment_loader(model_config)
        model_cfg = experiment.model_config
        original = model_cfg._load_base_checkpoint_model
        native_types = original.__func__.__globals__
        config_type = native_types["DM05Config"]
        model_type = native_types["DM05ForConditionalGeneration"]

        def load_checkpoint(config_self):
            nonlocal calls
            calls += 1
            if calls != 1:
                raise RuntimeError("Expected exactly one DM0.5 checkpoint load")
            # Same native config construction and dtype; only weight placement changes.
            config = config_type.from_pretrained(config_self.model_name_or_path)
            for attr, value in config_self._config_overrides().items():
                setattr(config, attr, value)
            config_self._materialize_checkpoint_runtime_config(config)
            return model_type.from_pretrained(
                config_self.model_name_or_path, config=config,
                dtype=config_self._torch_dtype(), device_map={"": "cuda:0"},
            )

        previous = vars(model_cfg).get("_load_base_checkpoint_model", missing)
        overrides.append((model_cfg, previous))
        model_cfg._load_base_checkpoint_model = MethodType(load_checkpoint, model_cfg)
        return experiment, path

    native._load_experiment = load_experiment
    try:
        yield
        if calls != 1:
            raise RuntimeError("Native construction did not use the sole direct-GPU loader")
    finally:
        native._load_experiment = original_experiment_loader
        for model_cfg, previous in reversed(overrides):
            if previous is missing:
                del model_cfg._load_base_checkpoint_model
            else:
                model_cfg._load_base_checkpoint_model = previous


def freeze_model(model):
    network = model._inference.model
    network.requires_grad_(False)
    network.eval()
    count = trainable = tensors = 0
    for parameter in network.parameters():
        tensors += 1
        count += parameter.numel()
        trainable += parameter.numel() if parameter.requires_grad else 0
        if str(parameter.device) != "cuda:0" or str(parameter.dtype) != "torch.float32":
            raise RuntimeError(
                f"DM0.5 parameter {tensors} must be cuda:0/torch.float32, "
                f"got {parameter.device}/{parameter.dtype}"
            )
    if trainable or count <= 0 or network.training:
        raise RuntimeError("DM0.5 must contain parameters and remain completely frozen")
    return {"parameter_count": count, "trainable_parameters": trainable,
            "parameter_tensor_count": tensors, "all_parameters_frozen": True,
            "all_parameters_device": "cuda:0", "all_parameters_dtype": "torch.float32"}


def audited_model_class(native, audit, cuda_metrics=None):
    """Compose the installed adapter; do not reimplement its history or inference."""
    import numpy as np

    class AuditedModel(native.Model):
        def __init__(self, config):
            super().__init__(config)
            self.freeze_receipt = freeze_model(self)
            self._episode_scope_seen = False
            self._initial_reset_done = False
            self._query_id = 0
            self._query_scope = None
            self._native_predict = self._inference._predict
            self._inference._predict = self._audited_predict

        def update_obs(self, obs):
            evaluation_id = scope_id(obs)
            if obs.get("env_idx", 0) != 0:
                raise ValueError("This ReFATE service supports one environment, env_idx=0")
            # Only public RGB, native joint measurements and the instruction enter DM0.5.
            clean = {"evaluation_id": evaluation_id, "env_idx": 0,
                     "state": native14(obs["state"]), "vision": {}}
            for image_key, camera in CAMERAS.items():
                clean["vision"][camera] = {"color": native._extract_rgb_image(obs, image_key)}
            clean["instruction"] = native._normalize_prompt(obs, self.default_prompt)
            super().update_obs(clean)
            self._episode_scope_seen = True
            sequence = self._history_sequence_by_env[(evaluation_id, 0)]
            emit(audit, "observation", evaluation_id=evaluation_id, observation_sequence=sequence,
                 native_completed_actions=sequence - 1, state=clean["state"],
                 instruction=obs.get("instruction"),
                 rgb_sha256={name: image_record(camera["color"])["sha256"]
                             for name, camera in obs.get("vision", {}).items()
                             if isinstance(camera, dict) and camera.get("color") is not None},
                 rgb={camera: image_record(data["color"]) for camera, data in clean["vision"].items()})

        def _audited_predict(self, payload):
            query_id = self._query_id
            emit(audit, "prediction_input", query_id=query_id, evaluation_id=self._query_scope,
                 state=payload["state"], instruction=payload["prompt"], meta_data=payload["meta_data"],
                 rgb={key: image_record(payload[key]) for key in CAMERAS},
                 history=[None if image is None else image_record(image)
                          for image in payload["history_images"]])
            started = time.monotonic()
            if query_id == 1:
                import faulthandler
                with Path(audit.name).with_name("first_prediction_stack.txt").open("x") as diagnostic:
                    faulthandler.dump_traceback_later(30, repeat=False, file=diagnostic)
                    try:
                        actions = self._native_predict(payload)
                    finally:
                        faulthandler.cancel_dump_traceback_later()
            else:
                actions = self._native_predict(payload)
            array = np.asarray(actions)
            if array.shape != (50, 14) or not np.isfinite(array).all():
                raise ValueError(f"Expected finite native DM0.5 prediction (50, 14), got {array.shape}")
            emit(audit, "prediction_output", query_id=query_id, evaluation_id=self._query_scope,
                 seconds=time.monotonic() - started, actions=actions)
            return actions

        def get_action(self, scope=None):
            evaluation_id = scope_id(scope)
            started = time.monotonic()
            self._query_id += 1
            self._query_scope = evaluation_id
            try:
                actions = super().get_action(scope)
                if len(actions) != 25:
                    raise ValueError("DM0.5 must return exactly 25 native joint actions")
                for action in actions:
                    if set(action) != set(STATE_DIMS):
                        raise ValueError("Unexpected native DM0.5 action keys")
                    native14(action)
                emit(audit, "action_chunk", evaluation_id=evaluation_id, query_id=self._query_id,
                     observation_sequence=self._history_sequence_by_env[(evaluation_id, 0)],
                     actions=actions, executed_actions_claimed=0,
                     seconds=time.monotonic() - started,
                     **(cuda_metrics() if cuda_metrics is not None else {}))
                return actions
            except Exception as exc:
                emit(audit, "inference_failed", evaluation_id=evaluation_id,
                     query_id=self._query_id, error_type=type(exc).__name__, error=str(exc))
                raise
            finally:
                self._query_scope = None

        def reset_evaluation(self, scope):
            evaluation_id = scope_id(scope)
            super().reset_evaluation(scope)
            self._episode_scope_seen = True
            emit(audit, "reset_evaluation", evaluation_id=evaluation_id,
                 meaning="episode_boundary_only_not_preparation_handback")

        def reset(self):
            # Native EvalEnv.reset calls this once before opening an episode scope.
            if self._initial_reset_done or self._episode_scope_seen or self._query_id:
                raise ValueError("Global reset is initialization-only; never clear history at handback")
            if any((self._observations, self._history_by_env, self._history_sequence_by_env,
                    self._latest_env_idx_by_evaluation)):
                raise ValueError("Initialization reset cannot discard existing observations")
            super().reset()
            self._initial_reset_done = True
            emit(audit, "initial_reset", meaning="native_environment_initialization_before_episode")

        def update_obs_batch(self, obs_list):
            raise ValueError("ReFATE DM0.5 service is single-environment; use update_obs")

        def get_action_batch(self, env_idx_list=None):
            raise ValueError("ReFATE DM0.5 service is single-environment; use get_action")

    return AuditedModel
