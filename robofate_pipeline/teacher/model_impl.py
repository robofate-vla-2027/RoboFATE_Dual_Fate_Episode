"""Original production Model; dependencies injected by runtime."""
import copy
import multiprocessing
from pathlib import Path
import time
import numpy as np

class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.cfg = model_cfg
        self.run = Path(model_cfg["run_dir"])
        self.episode = -1
        self.worker = self.connection = None
        self.obs = None
        self.pending = self.finished = False
        self.step_limit = None
        self.reset_requested = False

    def takeover(self, observation, causal_history, remaining_steps):
        return bind_model(self, observation, causal_history, remaining_steps)

    def takeover_rpc(self, payload):
        # XPolicyLab call() decodes vision at the top level, then calls method(obs).
        require(set(payload) == PUBLIC_KEYS.intersection(payload) |
                {'causal_history', 'remaining_steps'}, 'Invalid takeover payload')
        observation = {k: v for k, v in payload.items() if k in PUBLIC_KEYS}
        return self.takeover(observation, payload['causal_history'], payload['remaining_steps'])

    def accept_transition_rpc(self, payload):
        observation = {k: v for k, v in payload.items() if k in PUBLIC_KEYS}
        self.accept_transition({'kind': 'transition', 'observation': observation,
            **{k: payload[k] for k in ('success', 'terminated', 'truncated', 'refate')}})

    def reset(self):
        raise PermissionError("Takeover candidate never resets; construct a fresh Model")

    def prepare_case(self, case_meta=None):
        raise PermissionError("Use takeover with the actual native cumulative budget")

    def accept_transition(self, transition):
        accept_model_transition(self, transition)

    def get_takeover_request(self):
        require(self.pending and self.takeover_request is not None, "No pending action")
        return copy.deepcopy(self.takeover_request)

    def update_obs(self, obs):
        # Images have already been decoded by the official policy server.
        self.obs = copy.deepcopy(obs)

    def _transition(self, **_kwargs):
        raise RuntimeError("Explicit native execution receipt required; no implicit acknowledgement")

    def get_action(self):
        require(getattr(self, "_takeover_bound", False), "Call takeover first")
        if self.obs is None or self.step_limit is None:
            raise RuntimeError("prepare_case and update_obs must precede get_action")
        if self.worker is None:
            _execution_guard()
            load_binding(self.cfg.get("refate_main_binding_ref"),
                         context=self.cfg.get("refate_takeover"))
            self.episode += 1
            context = multiprocessing.get_context("spawn")
            self.connection, child = context.Pipe()
            target = run_worker
            if self.cfg.get("diagnostic"):
                from integrations.robodojo_policy.probe import run_probe
                target = run_probe
            self.worker = context.Process(target=target,
                args=(child, self.cfg, self.obs, self.step_limit, self.episode))
            self.worker.start()
            child.close()
        elif self.pending:
            self._transition()
        if self.finished:
            return []  # No synthetic post-agent hold actions.
        if self.reset_requested:
            return []
        # Empty chunk means still computing. The deploy loop polls without
        # advancing physics, so long coding turns do not hit RPC timeouts.
        if not self.connection.poll(0.5):
            if not self.worker.is_alive():
                raise RuntimeError("Agent worker died without a completion message")
            return []
        message = self.connection.recv()
        if message["kind"] == "error":
            raise RuntimeError(message["error"])
        if message["kind"] == "finished":
            self.finished = True
            return []
        if message["kind"] == "reset":
            if self.cfg.get("phase") != "playground":
                raise PermissionError("Native reset is forbidden in Test")
            self.reset_requested = True
            return []
        if message["kind"] != "action":
            raise RuntimeError("Unexpected worker message")
        require("refate" in message, "Worker action lacks owner/counter identity")
        self.takeover_request = copy.deepcopy(message)
        self.pending = True
        self._record(message["action"], source="agent_code")
        return [message["action"]]

    def _record(self, action, *, source):
        append_jsonl(self.run / "actions.jsonl", {
            "episode": self.episode, "source": source, "timestamp": time.time(),
            "action": {key: np.asarray(value).tolist() for key, value in action.items()},
        })

    def get_control(self):
        return {"reset": self.reset_requested, "finished": self.finished}

    def complete_reset(self, observation):
        if self.cfg.get("phase") != "playground" or not self.reset_requested:
            raise PermissionError("No Playground reset is pending")
        self.update_obs(observation)
        self.connection.send({"kind": "reset", "observation": self.obs})
        self.reset_requested = False

    def on_trial_end(self, result=None):
        require(not self.pending, "Deliver the final actual native transition first")
        result = result or {}
        if self.pending:
            self._transition(success=bool(result.get("success")),
                             terminated=bool(result.get("terminated")),
                             truncated=bool(result.get("truncated")))
        if self.cfg.get("phase") == "playground":
            return
        if self.worker is not None:
            self.worker.join(timeout=60)
            if self.worker.is_alive():
                raise RuntimeError("Worker did not close after official termination")
            if not self.finished and self.connection.poll():
                message = self.connection.recv()
                if message.get("kind") == "error":
                    raise RuntimeError(message["error"])
        write_json(self.run / f"trial_{self.episode:04d}.json", result)

    def update_obs_batch(self, obs_list):
        if len(obs_list) != 1:
            raise ValueError("This adapter uses one independent policy server per environment")
        self.update_obs(obs_list[0])

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list not in (None, [0]):
            raise ValueError("Only a single environment is supported")
        return [self.get_action()]

    def close(self):
        if self.worker is not None:
            if self.worker.is_alive():
                self.worker.terminate()
            self.worker.join(timeout=10)
            self.worker = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
