"""Explicit synthetic CPU tests of the thin common native25 teacher seam.

Original teacher/dispatch/transition and common execute_action code are reused.
Task actions, model responses, owner/provider gates and asynchronous publication
completion are synthetic. Exact recorder/writer classes are allocated WITHOUT
constructors, so no Native, model/API, worker, GPU, gVisor or writer thread runs.
Only small labeled codec fixtures and JSON receipts are written under our check/.
"""
import ast
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from robofate_pipeline.recording import teacher as adapter

import tempfile
from robofate_pipeline.recording import core as CORE, observation_io as OBS, takeover
from robofate_pipeline.recording.predict import ExistingEnvTeacher as TEACHER
TEMP = tempfile.TemporaryDirectory(prefix="robofate-synthetic-")
CHECK = Path(TEMP.name).resolve()
SOURCE = CHECK / 'SYNTHETIC_PREFIX'

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
FRAME[:, 320:] = 127


def observation(index):
    return {'instruction': 'SYNTHETIC CPU ONLY', 'state': {'synthetic_step': np.array([index], dtype=np.int64)},
            'vision': {name: {'color': FRAME} for name in adapter.CAMERAS}}


def policy_observation(value):
    return {key: item for key, item in value.items() if key in adapter.PUBLIC_KEYS}


TAKEOVER = vars(takeover)
REAL_WAIT = CORE.wait_for_native_publication


def setUpModule():
    write_synthetic_fixtures()
    global ACTION_PATCH
    ACTION_PATCH = patch.object(takeover, 'validate_action', side_effect=lambda action, state: copy.deepcopy(action))
    ACTION_PATCH.start()


def tearDownModule():
    ACTION_PATCH.stop()
    TEMP.cleanup()


def write_synthetic_fixtures():
    CHECK.mkdir(exist_ok=True)
    if SOURCE.exists():
        marker = json.loads((SOURCE/'SYNTHETIC_ONLY.json').read_text())
        if marker != {'data_kind': 'synthetic', 'actual_source': False, 'source_issued': False}:
            raise RuntimeError('Not this test suite synthetic fixture')
        return
    SOURCE.mkdir()
    for index in range(3):
        folder = CORE.recorded_observation_path(SOURCE, index)
        folder.mkdir()
        structure = {'instruction': 'SYNTHETIC CPU ONLY',
                     'state': {'synthetic_step': {'npz_key': 'array_0000', 'shape': [1], 'dtype': 'int64'}},
                     'vision': {}, '_private_synthetic_field': 'must not enter history'}
        arrays = {'array_0000': np.array([index], dtype=np.int64)}
        for offset, camera in enumerate(adapter.CAMERAS, 1):
            key = 'array_%04d' % offset
            tag = {'npz_key': key} if index == 0 else {'png_key': camera + '.png'}
            tag.update(shape=[480, 640, 3], dtype='uint8')
            structure['vision'][camera] = {'color': tag}
            if index == 0:
                arrays[key] = FRAME
            else:
                Image.fromarray(FRAME).save(folder/(camera + '.png'), format='PNG')
        np.savez_compressed(folder/'arrays.npz', **arrays)
        (folder/'observation.json').write_text(json.dumps(structure))
    actions = [{'index': i, 'native_count_after': i + 1, 'actual_control_steps': 10 * (i + 1),
                'role': 'DM0.5', 'action': {'synthetic_delta': 1},
                'native_success': False, 'terminated': False, 'truncated': False} for i in range(2)]
    boundaries = [{'index': i, 'control_index': 10 * i, 'terminal': False,
                   'observation_dir': str(CORE.recorded_observation_path(SOURCE, i))} for i in range(3)]
    for name, rows in (('native_actions.jsonl', actions), ('native_observations.jsonl', boundaries)):
        (SOURCE/name).write_bytes(b''.join(json.dumps(row).encode() + b'\n' for row in rows)
                                 + b'UNREADABLE_FUTURE_SUFFIX_MUST_NOT_BE_READ\n')
    (SOURCE/'SYNTHETIC_ONLY.json').write_text(json.dumps(
        {'data_kind': 'synthetic', 'actual_source': False, 'source_issued': False}))


class Fixture:
    def __init__(self):
        self.log, self.depth, self.mode = [], 0, None
        self.provider_calls = 0
        self.stop_provider_at = None
        self.task = types.SimpleNamespace(task_name='insert_tubes', control_steps=2, max_steps=5,
            native_step_limit=5, official_action_limit=True, _failure=None,
            done=False, success=False, terminated=False, truncated=False)
        self.task.native = types.SimpleNamespace(task_name='insert_tubes', step_lim=5, dt=.004,
            take_action_cnt=[2], obs_manager=types.SimpleNamespace(collect_interval=10),
            sim=types.SimpleNamespace(sim=types.SimpleNamespace(current_time=.08)))
        self.task.public_observation = lambda: observation(self.task.control_steps)
        self.task.control_step = self.control
        self.recorder = CORE.DenseRecorder.__new__(CORE.DenseRecorder)
        self.recorder.__dict__.update(task=self.task, control_index=20, capture_count=3, controls=[],
            last_observation_time=.08, last_observation=observation(2), closed_attempted=False, owner='DM0.5', macro=1)
        writer = OBS.OrderedObservationWriter.__new__(OBS.OrderedObservationWriter)
        writer.__dict__.update(condition=threading.Condition(threading.RLock()), first_error=None,
            cleanup_errors=[], closing=False, closed=False, submitted=3, committed=3, last_committed=2,
            outstanding=0, publication_in_flight=None, publication_in_flight_at_failure=None)
        self.recorder.writer = writer
        self.identity = dict(campaign='SYNTHETIC_CPU_CAMPAIGN', chat='synthetic-partition', active_chat_id='synthetic-owner',
            task='insert_tubes', source_id='SYNTHETIC_SOURCE', session='SYNTHETIC_SESSION', owner='synthetic', epoch=1)
        self.config = dict(schema='robofate.runtime.v1', data_kind='synthetic', task='insert_tubes',
            source_id=self.identity['source_id'], identity=self.identity, native_limit=5, t1=2, actor_wall_s=3600,
            run_dir=str(CHECK/'SYNTHETIC_TEACHER_NOT_CREATED'))
        context = dict(self.identity, native_used=2, native_limit=5, native_terminal=False,
                       dm_submissions_disabled=True, dm_inflight=0, dm_queue_discarded=True)
        self.model = types.SimpleNamespace(cfg={'runtime_binding': copy.deepcopy(self.config),
            'refate_takeover': context}, used=2, sequence=0, persistent_session=object(), transitions=[])
        self.model.takeover = self.takeover
        self.model.update_obs = lambda obs: self.log.append('update_obs')
        self.model.get_action = self.get_action
        self.model.get_takeover_request = lambda: self.request
        self.model.accept_transition = self.accept
        self.runtime = types.SimpleNamespace(binding=copy.deepcopy(self.config), ExistingEnvTeacher=TEACHER,
            takeover=types.SimpleNamespace(policy_observation=policy_observation, check_dispatch=TAKEOVER['check_dispatch']))
        # The common classes are exact; only asynchronous completion is synthetic.
        self.core_modules = {'collect_source': types.SimpleNamespace(DenseRecorder=CORE.DenseRecorder,
            recorded_observation_path=CORE.recorded_observation_path, wait_for_native_publication=self.wait),
            'observation_io': OBS}
        self.history = adapter.load_causal_history(source_run=SOURCE, t1=2, budget=5,
            core_modules=self.core_modules, policy_observation=policy_observation)

    @contextmanager
    def guard(self):
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1

    def gate(self, name):
        if self.depth != 1:
            raise AssertionError('Main gate not under sole-owner guard')
        self.log.append(name)

    def provider(self):
        self.gate('provider')
        self.provider_calls += 1
        if self.provider_calls == self.stop_provider_at:
            raise RuntimeError('SYNTHETIC structured provider stop')

    def owner(self):
        return dict(self.identity, dm_submissions_disabled=True, dm_inflight=0, dm_queue_discarded=True)

    def takeover(self, obs, history, remaining):
        self.log.append('takeover')
        self.remaining_at_takeover = remaining
        self.received_history = history

    def get_action(self):
        self.log.append('predict')
        if self.mode == 'think':
            return []
        if self.mode == 'predict_error':
            raise RuntimeError('SYNTHETIC model error')
        self.model.sequence += 1
        meta = TAKEOVER['request_metadata'](self.model.cfg['refate_takeover'], self.model.used, self.model.sequence)
        if self.mode == 'stale':
            meta['native_before'] -= 1
        self.request = {'kind': 'action', 'action': {'synthetic_delta': 1}, 'refate': meta}
        return [self.request['action']]

    def control(self, action):
        if self.depth != 1:
            raise AssertionError('Synthetic control outside main guard')
        self.log.append('control')
        if self.mode == 'uncertain':
            raise RuntimeError('SYNTHETIC unknown action outcome')
        self.task.control_steps += 1
        used = self.task.control_steps
        self.task.native.take_action_cnt[0] = used
        self.task.native.sim.sim.current_time = used * .04
        self.recorder.control_index = 10 * used
        self.recorder.capture_count = used + 1
        self.recorder.last_observation_time = used * .04
        self.recorder.last_observation = observation(used)
        self.recorder.controls = [{'synthetic_control_records': 10}]
        if used == 5 or self.mode == 'terminal':
            self.task.done = True
            self.task.success = self.task.terminated = self.mode == 'terminal'
            self.task.truncated = not self.task.success
        if self.mode == 'post_control_error':
            raise RuntimeError('SYNTHETIC error after physical counter advance')

    def publish(self, event):
        self.log.append('publish')
        self.event = event
        writer = self.recorder.writer
        writer.submitted += 1
        writer.outstanding += 1
        self.recorder.controls = []
        if self.mode == 'publish_error':
            raise RuntimeError('SYNTHETIC publish error')

    def wait(self, recorder, index):
        self.log.append('wait')
        if self.mode == 'wait_error':
            raise RuntimeError('SYNTHETIC publication wait error')
        if self.mode == 'uncommitted':
            return
        writer = recorder.writer
        writer.last_committed, writer.committed, writer.outstanding = index, index + 1, 0
        REAL_WAIT(recorder, index)
        self.log.append('committed')

    def trace(self, recorder, *, before, after, event):
        self.log.append('dense_ref')
        if self.depth != 1 or not self.teacher.failed or recorder.writer.last_committed != after['native_used']:
            raise AssertionError('Trace callback before committed/latching/guard boundary')
        if event['native_count_after'] != after['native_used'] or event['role'] != 'EmbodiedEvo_teacher':
            raise AssertionError('Not actual shared execute_action event')
        if self.mode == 'trace_error':
            raise RuntimeError('SYNTHETIC committed reference error')
        if self.mode == 'blank_trace':
            return ''
        if self.mode == 'trace_changes_budget':
            self.task.native.step_lim = 6
        return 'SYNTHETIC_COMMITTED_DENSE_REF_' + str(after['native_used'])

    def accept(self, message):
        self.log.append('after_action')
        if 'dense_ref' not in self.log or self.log.index('dense_ref') > self.log.index('after_action'):
            raise AssertionError('Feedback before dense reference')
        TAKEOVER['validate_transition'](message, self.request)
        if self.mode == 'feedback_error':
            raise RuntimeError('SYNTHETIC transition delivery uncertainty')
        self.model.used = message['refate']['native_after']
        self.model.transitions.append(message)

    def bind(self, **overrides):
        args = dict(source_run=SOURCE, task_name='insert_tubes', t1=2, budget=5, config=self.config,
            causal_history=self.history, core_modules=self.core_modules, owner_guard=self.guard,
            owner_snapshot=self.owner, check_source_admission=lambda: self.gate('admission'),
            check_restored_t1=lambda: self.gate('restore'), check_provider_safety=self.provider,
            publish=self.publish, committed_dense_ref=self.trace)
        args.update(overrides)
        self.teacher = adapter.bind_existing_teacher(self.runtime, self.model, self.task, self.recorder, **args)
        self.log.clear()
        return self.teacher


class TeacherTests(unittest.TestCase):
    def fixture(self):
        value = Fixture()
        value.bind()
        return value

    def test_original_teacher_class_and_takeover_remaining(self):
        f = self.fixture()
        self.assertIs(type(f.teacher), TEACHER)
        self.assertEqual(f.remaining_at_takeover, 3)
        self.assertEqual(len(f.received_history), 3)

    def test_actual_decoder_handles_initial_npz_and_dense_png(self):
        f = Fixture()
        self.assertEqual([x['native_step'] for x in f.history], [0, 1, 2])
        self.assertTrue(adapter.equal(f.history[-1]['observation'], observation(2)))
        self.assertNotIn('_private_synthetic_field', f.history[0]['observation'])

    def test_prefix_never_reads_poison_future_row(self):
        stream = io.BytesIO(b'{"a":0}\n{"a":1}\nPOISON_FUTURE\n')
        self.assertEqual(adapter.prefix_rows(stream, 2), [{'a': 0}, {'a': 1}])
        self.assertEqual(stream.readline(), b'POISON_FUTURE\n')

    def test_prefix_duplicate_nonfinite_or_partial_rows_rejected(self):
        for raw in (b'{"a":1,"a":2}\n', b'{"a":NaN}\n', b'{}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                adapter.prefix_rows(io.BytesIO(raw), 1)

    def test_exact_decoder_calls_end_at_t1(self):
        with patch.object(OBS, 'decode_observation', wraps=OBS.decode_observation) as decoder:
            Fixture()
        self.assertEqual([call.args[0] for call in decoder.call_args_list],
                         [CORE.recorded_observation_path(SOURCE, i) for i in range(3)])

    def test_thinking_advances_nothing(self):
        f = self.fixture(); f.mode = 'think'
        self.assertIsNone(adapter.advance_teacher(f.teacher))
        self.assertEqual(f.task.control_steps, 2)
        self.assertNotIn('control', f.log)
        self.assertFalse(f.teacher.failed)

    def test_shared_execute_publish_commit_reference_before_feedback(self):
        f = self.fixture()
        message = adapter.advance_teacher(f.teacher)
        self.assertEqual([name for name in f.log if name in ('control', 'publish', 'wait', 'committed', 'dense_ref', 'after_action')],
                         ['control', 'publish', 'wait', 'committed', 'dense_ref', 'after_action'])
        self.assertEqual(message['refate']['native_after'], 3)
        self.assertEqual(f.recorder.owner, 'EmbodiedEvo_teacher')
        self.assertFalse(f.teacher.failed)
        self.assertIsNone(f.teacher.pending)

    def test_original_budget_and_persistent_session_across_actions(self):
        f = self.fixture(); session = f.model.persistent_session
        for after in (3, 4, 5):
            f.log.clear()
            message = adapter.advance_teacher(f.teacher)
            self.assertEqual(message['refate']['native_after'], after)
            self.assertEqual(f.task.max_steps, 5)
            self.assertEqual(f.task.native_step_limit - f.task.control_steps, 5 - after)
            self.assertIs(f.model.persistent_session, session)
        self.assertTrue(message['truncated'])
        self.assertEqual(f.teacher.last_sequence, 3)
        self.assertIsNone(adapter.advance_teacher(f.teacher))

    def test_terminal_action_still_delivers_feedback(self):
        f = self.fixture(); f.mode = 'terminal'
        message = adapter.advance_teacher(f.teacher)
        self.assertTrue(message['success'] and message['terminated'])
        self.assertIn('after_action', f.log)

    def assert_latched(self, mode, *, after_action=False):
        f = self.fixture(); f.mode = mode
        expected = {
            "uncertain": "unknown action outcome",
            "post_control_error": "error after physical counter advance",
            "publish_error": "publish error",
            "wait_error": "publication wait error",
            "uncommitted": "Complete ordered publication required",
            "trace_error": "committed reference error",
            "blank_trace": "Actual committed dense reference required",
            "trace_changes_budget": "Original native budget/counter changed",
            "feedback_error": "transition delivery uncertainty",
            "stale": "Stale action or exhausted budget",
            "predict_error": "model error",
        }[mode]
        with self.assertRaisesRegex(Exception, expected):
            adapter.advance_teacher(f.teacher)
        self.assertTrue(f.teacher.failed)
        self.assertIsNotNone(f.recorder.writer.first_error)
        controls = f.log.count('control')
        with self.assertRaises(Exception):
            adapter.advance_teacher(f.teacher)
        self.assertEqual(f.log.count('control'), controls)
        if not after_action:
            self.assertNotIn('after_action', f.log)
        return f

    def test_unknown_control_outcome_latches_without_retry(self):
        self.assert_latched('uncertain')

    def test_post_counter_error_latches_without_replay(self):
        f = self.assert_latched('post_control_error')
        self.assertEqual(f.task.control_steps, 3)

    def test_publication_failure_latches(self):
        self.assert_latched('publish_error')

    def test_wait_failure_latches(self):
        self.assert_latched('wait_error')

    def test_incomplete_commit_cannot_deliver_feedback(self):
        f = self.assert_latched('uncommitted')
        self.assertNotIn('dense_ref', f.log)

    def test_dense_reference_failure_latches(self):
        self.assert_latched('trace_error')

    def test_blank_reference_rejected(self):
        self.assert_latched('blank_trace')

    def test_reference_callback_cannot_change_budget(self):
        self.assert_latched('trace_changes_budget')

    def test_feedback_delivery_uncertainty_latches(self):
        self.assert_latched('feedback_error', after_action=True)

    def test_provider_checked_again_immediately_before_control(self):
        f = self.fixture(); f.stop_provider_at = f.provider_calls + 2
        with self.assertRaisesRegex(RuntimeError, 'provider stop'):
            adapter.advance_teacher(f.teacher)
        self.assertNotIn('control', f.log)
        self.assertTrue(f.teacher.failed)

    def test_provider_stop_before_predict(self):
        f = self.fixture(); f.stop_provider_at = f.provider_calls + 1
        with self.assertRaises(RuntimeError):
            adapter.advance_teacher(f.teacher)
        self.assertNotIn('predict', f.log)

    def test_stale_original_dispatch_rejected(self):
        f = self.assert_latched('stale')
        self.assertNotIn('control', f.log)

    def test_model_failure_is_not_a_hold_action(self):
        f = self.assert_latched('predict_error')
        self.assertNotIn('control', f.log)

    def test_admission_and_restore_are_required_main_gates(self):
        for gate in ('check_source_admission', 'check_restored_t1'):
            f = Fixture()
            def stop():
                raise RuntimeError('SYNTHETIC main gate denied')
            with self.subTest(gate=gate), self.assertRaises(RuntimeError):
                f.bind(**{gate: stop})
            self.assertNotIn('takeover', f.log)
            self.assertIsNotNone(f.recorder.writer.first_error)

    def test_wrong_count_budget_or_recorder_rejected(self):
        for change in ('count', 'budget', 'writer'):
            f = Fixture()
            if change == 'count': f.recorder.capture_count = 99
            if change == 'budget': f.task.native.step_lim = 6
            if change == 'writer': f.recorder.writer = types.SimpleNamespace(check=lambda: None, fail=lambda error: None)
            with self.subTest(change=change), self.assertRaises((ValueError, RuntimeError)):
                f.bind()

    def test_future_history_entry_rejected(self):
        f = Fixture(); f.history.append(copy.deepcopy(f.history[-1]))
        with self.assertRaises(ValueError):
            f.bind()

    def test_restore_gate_not_replaced_by_pixel_equivalence_claim(self):
        f = Fixture()
        f.history[-1]['observation']['state']['synthetic_step'][0] = -1
        f.bind()
        self.assertEqual(f.received_history[-1]['observation']['state']['synthetic_step'][0], -1)
        self.assertEqual(f.task.control_steps, 2)


