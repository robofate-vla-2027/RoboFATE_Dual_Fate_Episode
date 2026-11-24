"""Collection seams for a caller-owned Native task, policy and qualified restore.

This module never selects machines/devices, loads a model, constructs a task,
grants source admission, supplies restore tolerances or retries failed actions.
"""
import copy
from pathlib import Path
import time


def _gate(callback):
    if not callable(callback) or callback() is not None:
        raise ValueError("An explicit gate must return None or raise on rejection")


def recording_modules():
    from .recording import core, observation_io, pt_capture, rtx_profile
    return {"collect_source": core, "observation_io": observation_io,
            "pt_capture": pt_capture, "rtx_profile": rtx_profile}


class PolicyView:
    """The standard OpenDM eval loop sees only public observations and actions."""
    def __init__(self, task, recorder, publish, run_id, admission):
        self.task, self.recorder, self.publish = task, recorder, publish
        self.admission = admission
        self.deploy_cfg = {"evaluation_id": run_id}

    def get_obs(self):
        self.recorder.writer.check()
        return self.task.public_observation()

    def take_action(self, action):
        from .recording.common import execute_action
        _gate(self.admission)
        return execute_action(self.task, self.recorder, self.publish, action, role="DM0.5")

    def is_episode_end(self):
        self.recorder.writer.check()
        return self.task.done


def run_baseline(task, policy, eval_one_episode, recorder, publish, *, run_id, admission):
    """Run the supplied standard frozen-DM loop; no success inferred from exit."""
    from .recording import core
    _gate(admission)
    if not callable(eval_one_episode) or recorder.task is not task or task.control_steps != 0:
        raise ValueError("Explicit original evaluator and fresh task/recorder required")
    try:
        eval_one_episode(PolicyView(task, recorder, publish, run_id, admission), policy)
        core.wait_for_native_publication(recorder, task.control_steps)
        if not task.done or task.control_steps <= 0:
            raise ValueError("Baseline returned without a native terminal outcome")
        return {"native_success": bool(task.success), "native_steps": task.control_steps,
                "qualified": False}
    except BaseException as error:
        recorder.writer.fail(error)
        raise


def replay_prefix(task, recorder, publish, *, source_run, t1, budget,
                  admission, owner_guard, verify_initial, verify_restored):
    """Replay only original actions before t1, then require an external restore proof.

    Gates must validate original source/profile pins, physical/RNG state,
    auxiliary state, rendering and coverage. This function does not fabricate
    a proof from matching pixels or numeric agreement alone.
    """
    from .recording import common, core, teacher
    if not all(callable(value) for value in
               (admission, owner_guard, verify_initial, verify_restored)):
        raise ValueError("Explicit owner, admission and restore gates required")
    if type(t1) is not int or type(budget) is not int or not 0 < t1 < budget:
        raise ValueError("Invalid original boundary or budget")
    if task.control_steps != 0 or task.max_steps != budget or recorder.task is not task:
        raise ValueError("Fresh recorder with unchanged original budget required")
    source = Path(source_run)
    if not source.is_absolute() or source.resolve(strict=True) != source:
        raise ValueError("Canonical source required")
    run = Path(recorder.run).resolve()
    if source.is_relative_to(run) or run.is_relative_to(source):
        raise ValueError("Replay destination must be separate from original source")
    try:
        with owner_guard():
            _gate(admission)
            _gate(verify_initial)
            actions = teacher._read_prefix(source / "native_actions.jsonl", t1)
            for index, event in enumerate(actions):
                if (type(event.get("index")) is not int or event["index"] != index
                        or event.get("role") != "DM0.5"
                        or type(event.get("native_count_after")) is not int
                        or event.get("native_count_after") != index + 1
                        or type(event.get("actual_control_steps")) is not int
                        or event.get("actual_control_steps") != 10 * (index + 1)
                        or not isinstance(event.get("action"), dict)
                        or any(event.get(key) is not False for key in
                               ("native_success", "terminated", "truncated"))):
                    raise ValueError("Invalid, future or terminal prefix row")
            for index, event in enumerate(actions):
                _gate(admission)
                common.execute_action(task, recorder, publish, copy.deepcopy(event["action"]),
                                      role="recorded_DM0.5_prefix_replay",
                                      source_event="native_actions.jsonl:" + str(index))
                core.wait_for_native_publication(recorder, index + 1)
                if task.control_steps != index + 1 or task.max_steps != budget:
                    raise ValueError("Replay changed the original budget or control count")
            _gate(verify_restored)
            if task.control_steps != t1 or task.done or task.max_steps != budget:
                raise ValueError("Restore gate changed the nonterminal boundary or budget")
    except BaseException as error:
        recorder.writer.fail(error)
        raise


def run_teacher(teacher, *, wall_seconds):
    """Advance a previously bound teacher under the existing fail-closed gates."""
    from .recording.teacher import advance_teacher
    if type(wall_seconds) is not int or wall_seconds <= 0:
        raise ValueError("Explicit positive wall budget required")
    deadline = time.monotonic() + wall_seconds
    state = teacher._common_teacher
    try:
        while not state["task"].done:
            if time.monotonic() >= deadline:
                raise TimeoutError("Teacher wall budget exhausted; no automatic retry")
            if teacher.model.finished:
                break
            advance_teacher(teacher)
        task = state["task"]
        state["core_modules"]["collect_source"].wait_for_native_publication(
            state["recorder"], task.control_steps)
        return {"native_success": bool(task.success), "native_steps": task.control_steps,
                "native_terminal": bool(task.done), "qualified": False}
    except BaseException as error:
        from .recording.teacher import _latch
        _latch(teacher, state["recorder"], error)
        raise
