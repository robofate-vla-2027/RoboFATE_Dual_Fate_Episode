"""Original ExistingEnvTeacher on main's existing common native25 recorder.

No task/model/session/recorder construction, reset, render, process, retry,
source admission, restore proof or ledger policy is implemented here. Main
supplies real admission/restore/provider gates and the sole-owner lock. Shared
publication is committed, not claimed fsync-durable before the normal close.
"""
from __future__ import annotations

import copy
import json
from numbers import Integral
import os
from pathlib import Path
import stat

import numpy as np
from . import common as recording

CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
PUBLIC_KEYS = {'instruction', 'vision', 'state', 'action', 'additional_info'}


def need(value, message):
    if not value:
        raise ValueError(message)


def equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b))
    if isinstance(a, np.generic):
        a = a.item()
    if isinstance(b, np.generic):
        b = b.item()
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def integer(value, name):
    need(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and value >= 0,
         'Invalid native25 ' + name)
    return int(value)


def public_observation(observation, policy_observation):
    public = policy_observation(observation)
    need(isinstance(public, dict) and set(public) <= PUBLIC_KEYS
         and isinstance(public.get('instruction'), str) and public['instruction']
         and isinstance(public.get('state'), dict) and public['state'], 'Public native observation required')
    need(isinstance(public.get('vision'), dict) and set(public['vision']) == set(CAMERAS), 'Three common cameras required')
    for camera in CAMERAS:
        frame = public['vision'][camera]['color']
        need(isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and frame.shape == (480, 640, 3),
             'Common RGB shape/dtype differs; adapter never renders or resizes')
    return copy.deepcopy(public)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'Duplicate prefix JSON key')
        result[key] = value
    return result


def prefix_rows(stream, count):
    """Read exactly count complete rows, not even the first future row."""
    need(type(count) is int and count >= 0, 'Explicit prefix count required')
    rows = []
    for _ in range(count):
        raw = stream.readline(1024 * 1024 + 1)
        need(0 < len(raw) <= 1024 * 1024 and raw.endswith(b'\n'), 'Missing/incomplete/oversized prefix row')
        row = json.loads(raw, object_pairs_hook=_unique,
                         parse_constant=lambda _: need(False, 'Nonfinite prefix JSON'))
        need(type(row) is dict, 'Prefix row must be an object')
        rows.append(row)
    return rows


def _read_prefix(path, count):
    need(path.is_absolute() and path.resolve(strict=True) == path, 'Canonical original prefix path required')
    signature = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    # Unbuffered reads also avoid read-ahead into the failure suffix.
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb', buffering=0) as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode), 'Original prefix must be a regular file')
        rows = prefix_rows(stream, count)
        after = os.fstat(stream.fileno())
    need(signature(before) == signature(after) == signature(path.stat()), 'Original prefix changed during read')
    return rows


def load_causal_history(*, source_run, t1, budget, core_modules, policy_observation):
    """Decode original observations 0..t1 and exactly t1 original DM action rows.

    Does not read result/future suffix, establish source admission, replay a
    prefix or infer that source pixels equal main's restored current pixels.
    """
    source = Path(source_run)
    need(source.is_absolute() and source.resolve(strict=True) == source, 'Canonical original source run required')
    need(type(t1) is int and type(budget) is int and 0 < t1 < budget, 'Explicit cutoff/original budget required')
    actions = _read_prefix(source/'native_actions.jsonl', t1)
    observations = _read_prefix(source/'native_observations.jsonl', t1 + 1)
    history = []
    for index, row in enumerate(observations):
        folder = core_modules['collect_source'].recorded_observation_path(source, index)
        need(type(row.get('index')) is int and row['index'] == index
             and type(row.get('control_index')) is int and row['control_index'] == 10 * index
             and row.get('observation_dir') == str(folder) and row.get('terminal') is False,
             'Noncontiguous/future/terminal original observation prefix')
        action, feedback = None, dict(success=False, terminated=False, truncated=False)
        if index:
            event = actions[index - 1]
            need(type(event.get('index')) is int and event['index'] == index - 1
                 and type(event.get('native_count_after')) is int and event['native_count_after'] == index
                 and type(event.get('actual_control_steps')) is int and event['actual_control_steps'] == 10 * index
                 and event.get('role') == 'DM0.5' and isinstance(event.get('action'), dict) and event['action'],
                 'Original completed DM native25 action required')
            need(all(event.get(key) is False for key in ('native_success', 'terminated', 'truncated')),
                 'Terminal/future failure feedback cannot enter causal prefix')
            action = copy.deepcopy(event['action'])
            feedback = {key: event['native_success' if key == 'success' else key] for key in feedback}
        observation = public_observation(core_modules['observation_io'].decode_observation(folder), policy_observation)
        history.append({'native_step': index, 'observation': observation,
                        'executed_action': action, 'feedback': feedback})
    return history


def check_recording(task, recorder, *, core_modules, task_name, budget, committed=False):
    core = core_modules['collect_source']
    writer = recorder.writer
    need(type(recorder) is core.DenseRecorder and type(writer) is core_modules['observation_io'].OrderedObservationWriter,
         'Exact shared DenseRecorder and OrderedObservationWriter required')
    writer.check()
    used = integer(task.control_steps, 'control_steps')
    need(recorder.task is task and getattr(task, '_failure', None) is None
         and task.task_name == task_name and task.native.task_name == task_name,
         'Actual task/recorder changed or failed')
    need(task.official_action_limit is True
         and all(type(value) is int and value == budget for value in (task.max_steps, task.native_step_limit, task.native.step_lim))
         and used == integer(task.native.take_action_cnt[0], 'take_action_cnt') and used <= budget,
         'Original native budget/counter changed')
    need(all(type(getattr(task, key)) is bool for key in ('done', 'success', 'terminated', 'truncated'))
         and task.done == (task.terminated or task.truncated)
         and not (task.terminated and task.truncated) and (not task.success or task.terminated)
         and (used < budget or task.done), 'Native terminal flags/budget differ')
    need(float(task.native.dt) == .004 and integer(task.native.obs_manager.collect_interval, 'collect_interval') == 10
         and recorder.control_index == 10 * used and recorder.capture_count == used + 1
         and recorder.last_observation_time == task.native.sim.sim.current_time
         and recorder.controls == [] and not recorder.closed_attempted,
         'Common control/capture/publication boundary differs')
    with writer.condition:
        writer.check()
        need(writer.submitted == used + 1 and not writer.closed, 'Current native boundary not submitted')
        if committed:
            need(writer.last_committed == used and writer.committed == used + 1
                 and writer.outstanding == 0 and writer.publication_in_flight is None,
                 'Complete ordered publication required before feedback')
    return {'native_used': used, 'native_limit': budget, 'remaining_steps': budget - used,
            'control_index': recorder.control_index, 'capture_count': recorder.capture_count,
            'simulation_time': recorder.last_observation_time, 'terminal': task.done,
            'success': task.success, 'terminated': task.terminated, 'truncated': task.truncated}


def _gate(callback):
    need(callable(callback) and callback() is None, 'Main gate must return None or raise on stop')


def _latch(teacher, recorder, error):
    teacher.failed = True
    try:
        recorder.writer.fail(error)
    except BaseException as secondary:
        if secondary is not error:
            error.add_note('Writer latch also failed: ' + repr(secondary))


def _current(teacher, state, observation):
    need(equal(state['runtime'].binding, state['config'])
         and equal(teacher.model.cfg['runtime_binding'], state['config']), 'Original runtime/model binding changed')
    snapshot = teacher._snapshot(observation)
    for key, value in state['config']['identity'].items():
        need(equal(snapshot.get(key), value), 'Main current identity changed: ' + key)
    return snapshot


def bind_existing_teacher(runtime, model, task, recorder, *, source_run, task_name, t1, budget,
                          config, causal_history, core_modules, owner_guard, owner_snapshot,
                          check_source_admission, check_restored_t1, check_provider_safety,
                          publish, committed_dense_ref):
    """Bind once under main's lock. All three main gates are zero-argument/raise.

    committed_dense_ref(recorder, *, before, after, event) -> nonempty str is
    invoked only after the original writer has completely committed publication.
    Main's restore gate must establish the original-prefix/restored-t1 relation.
    """
    source = Path(source_run)
    need(source.is_absolute() and source.resolve() == source, 'Canonical original source run required')
    need(type(t1) is int and type(budget) is int and 0 < t1 < budget, 'Fixed t1/original budget required')
    need(config.get('schema') == 'robofate.runtime.v1' and config.get('task') == task_name
         and config.get('native_limit') == budget and config.get('t1') == t1
         and config.get('actor_wall_s') == 3600, 'Explicit original common runtime binding required')
    need(equal(runtime.binding, config) and equal(model.cfg['runtime_binding'], config), 'Model/runtime binding differs')
    need(config['identity']['source_id'] == config['source_id'] and config['identity']['task'] == task_name
        , 'Original source identity differs')
    # Source exclusions and consumed/refused IDs belong to the external admission gate.
    need(not Path(config['run_dir']).is_relative_to(source), 'Never write teacher session into original source')
    for callback in (owner_guard, owner_snapshot, publish, committed_dense_ref,
                     check_source_admission, check_restored_t1, check_provider_safety):
        need(callable(callback), 'Explicit main-owned callback required')
    need(isinstance(causal_history, list) and len(causal_history) == t1 + 1, 'Only original history 0..t1 required')
    for index, item in enumerate(causal_history):
        need(set(item) == {'native_step', 'observation', 'executed_action', 'feedback'}
             and type(item['native_step']) is int and item['native_step'] == index
             and item['feedback'] == dict(success=False, terminated=False, truncated=False)
             and all(type(value) is bool for value in item['feedback'].values()), 'Causal prefix schema/terminal mismatch')
        need(item['executed_action'] is None if index == 0 else isinstance(item['executed_action'], dict),
             'Executed original prefix action missing')
    state = dict(runtime=runtime, recorder=recorder, task=task, source_run=source,
                 config=copy.deepcopy(config), core_modules=core_modules, task_name=task_name, budget=budget,
                 owner_guard=owner_guard, check_provider_safety=check_provider_safety,
                 publish=publish, committed_dense_ref=committed_dense_ref)
    teacher = runtime.ExistingEnvTeacher(model, task, owner_snapshot=owner_snapshot)
    teacher._common_teacher = state
    try:
        with owner_guard():
            _gate(check_source_admission)
            _gate(check_restored_t1)
            _gate(check_provider_safety)
            core_modules['collect_source'].wait_for_native_publication(recorder, t1)
            current = check_recording(task, recorder, core_modules=core_modules, task_name=task_name, budget=budget, committed=True)
            need(current['native_used'] == t1 and not task.done, 'Actually restored nonterminal fixed t1 required')
            observation = public_observation(task.public_observation(), runtime.takeover.policy_observation)
            _current(teacher, state, observation)
            teacher.takeover(observation, copy.deepcopy(causal_history), budget - t1)
            need(check_recording(task, recorder, core_modules=core_modules, task_name=task_name,
                                 budget=budget, committed=True) == current, 'Takeover binding advanced native state')
    except BaseException as error:
        _latch(teacher, recorder, error)
        raise
    return teacher


def advance_teacher(teacher):
    """One original prediction/action/committed feedback, or None without control.

    The original teacher/model owns sequence, persistent Session and transitions.
    No new native budget, retry, capture, writer or teacher decision is created.
    """
    state = teacher._common_teacher
    recorder, task, runtime = state['recorder'], state['task'], state['runtime']
    need(teacher.task is task and teacher.bound and not teacher.failed and teacher.pending is None,
         'Failed/unbound/pending teacher cannot advance')
    check = lambda: check_recording(task, recorder, core_modules=state['core_modules'],
                                    task_name=state['task_name'], budget=state['budget'], committed=True)
    try:
        with state['owner_guard']():
            _gate(state['check_provider_safety'])
            before = check()
            observation = public_observation(task.public_observation(), runtime.takeover.policy_observation)
            _current(teacher, state, observation)
            action = teacher.predict(observation)
            if action is None:
                need(check() == before, 'Thinking must not advance native state')
                return None
            _gate(state['check_provider_safety'])
            need(check() == before, 'Native boundary changed before dispatch')
            fresh = public_observation(task.public_observation(), runtime.takeover.policy_observation)
            need(equal(fresh, observation), 'Public observation changed while teacher waited')
            checked = runtime.takeover.check_dispatch(teacher.pending, _current(teacher, state, fresh))
            need(equal(checked, action), 'Original dispatch action differs from prediction')
            teacher.failed = True
            event = recording.execute_action(task, recorder, state['publish'], action, role='EmbodiedEvo_teacher')
            state['core_modules']['collect_source'].wait_for_native_publication(recorder, before['native_used'] + 1)
            after = check()
            need(after['native_used'] == before['native_used'] + 1, 'One original native action required')
            reference = state['committed_dense_ref'](recorder, before=copy.deepcopy(before),
                                                     after=copy.deepcopy(after), event=copy.deepcopy(event))
            need(type(reference) is str and bool(reference.strip()), 'Actual committed dense reference required')
            need(check() == after, 'Dense-reference callback advanced state or failed publication')
            obs_after = public_observation(task.public_observation(), runtime.takeover.policy_observation)
            _current(teacher, state, obs_after)
            teacher.failed = False
            message = teacher.after_action(obs_after, dense_trace_ref=reference)
            need(check() == after, 'Original feedback changed native state/publication')
            return message
    except BaseException as error:
        _latch(teacher, recorder, error)
        raise
