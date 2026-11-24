"""Original prefix replay and physical/auxiliary/camera comparison."""
import copy
import hashlib
import json
import os
from pathlib import Path
import stat

def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_bytes(filename):
    path = Path(filename)
    require(path.is_absolute() and path.resolve() == path and not path.is_symlink()
            and any(path.is_relative_to(root) for root in READ_ROOTS), 'Noncanonical/out-of-scope metadata')
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= 256 * 1024 * 1024,
                'Invalid/oversized metadata')
        raw = stream.read(before.st_size + 1)
        after = os.fstat(stream.fileno())
    sig = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(sig(before) == sig(after) == sig(path.stat()) and len(raw) == before.st_size,
            'Metadata changed during read')
    return raw


def read_doc(filename):
    return _decode(read_bytes(filename))


def read_ref(ref):
    raw = read_bytes(ref['path'])
    require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Pinned evidence changed')
    return _decode(raw)


def bound_run(run, config, context):
    context = check_restore(context, config)
    verify_bound(context)
    verify_restore_config(config)
    run = Path(run)
    require(str(run) == context.run and run.is_absolute() and run.resolve() == run
            and run.is_relative_to(ROOT / 'runs')
            and run != Path(config['reference_run']), 'Wrong new teacher run or source overwrite')
    require(not (run / 'policy').exists()
            and not any((run / 'ipc').glob('request_*.json')), 'DM restore worker/request is forbidden')
    require(read_ref(context.reference('native_config'))['enabled'] is True,
            'The independently pinned actual Native config is not enabled')
    return context


def native25_recording(run):
    recording = read_doc(Path(run) / 'recording_contract.json')
    require(recording['schema'] == 'refate-native25-v1'
            and type(recording['visual_hz']) is int and recording['visual_hz'] == 25
            and type(recording['low_level_ticks_per_native']) is int
            and recording['low_level_ticks_per_native'] == 10
            and recording['physical_dt_s'] == .004
            and recording['camera_rgb_shape'] == [480, 640, 3]
            and recording['camera_aliases'] == ['cam_head', 'cam_left_wrist', 'cam_right_wrist'],
            'Only the original three-camera native25 recording contract is supported')
    return recording


def checked_recording_modules(context):
    from ..recording import core as collect_source, observation_io
    for name, module in (('collect_source.py', collect_source), ('observation_io.py', observation_io)):
        ref = context.evo_ref('native25_runtime_refs')[name]
        require(Path(module.__file__) == Path(ref['path'])
                and hashlib.sha256(read_bytes(ref['path'])).hexdigest() == ref['sha256'],
                'Actual Native25 module differs from bound Evo config')
    return collect_source, observation_io


def compare_boundary(run, config, write, *, context=None):
    context = bound_run(run, config, context)
    run = Path(run)
    checked_recording_modules(context)
    native25_recording(run)
    from ..recording.core import scalar, recorded_observation_path
    from ..recording.observation_io import decode_observation
    import numpy as np
    reference = Path(config['reference_run']); t1 = config['t1']
    comparator_path = Path(__file__).with_name('comparator_impl.py')
    comparator_bytes = read_bytes(comparator_path)
    require(hashlib.sha256(comparator_bytes).hexdigest() == config['comparator_sha256'], 'Comparator drift')
    profile_bytes = read_bytes(config['profile_path'])
    require(hashlib.sha256(profile_bytes).hexdigest() == config['profile_sha256'], 'Profile drift at boundary')
    check_profile(context, _decode(profile_bytes))
    validate_profile(_decode(profile_bytes))
    gaps = _decode(context.source_gaps_json)
    for index in (0, t1):
        source_state = read_doc(reference / ('initial_physics_state.json' if index == 0 else f'physics_{index:06d}.json'))
        require(canonical(source_state['unexposed']) == canonical(gaps['physics_boundary' + str(index)]),
                'Actual source missing-state declarations differ from its bound source scope')
    comparator = _COMPARATOR
    result = comparator.compare_files(reference / f'physics_{t1:06d}.json',
                                     run / f'physics_{t1:06d}.json', config['profile_path'], context=context)
    baseline = comparator.compare_files(reference / f'physics_{t1:06d}.json',
                                        reference / f'physics_{t1:06d}.json', config['profile_path'], context=context)
    expected_missing = baseline['coverage']['missing_or_unhandled_paths']
    result['source_scope_missing_paths'] = expected_missing
    # Compare the exact missing-check multiset, including legacy placeholder sides.
    # Source-unexposed bytes are not exported state; any new omission still fails.
    missing_signature = lambda value: sorted(json.dumps(row, sort_keys=True, allow_nan=False)
                                             for row in value['coverage']['missing_or_unhandled_checks'])
    result['source_scope_missing_checks'] = baseline['coverage']['missing_or_unhandled_checks']
    result['no_new_missing_fields'] = (result['coverage']['missing_or_unhandled_paths'] == expected_missing
                                      and missing_signature(result) == missing_signature(baseline))
    write(run / 'restore_state_comparison.json', result)
    require(result['all_compared_fields_within_candidate_rules'], 'Restored physical/discrete state mismatch')
    require(result['no_new_missing_fields'], 'Restored exposed physical fields disappeared')
    expected = decode_observation(recorded_observation_path(reference, t1))
    actual = decode_observation(recorded_observation_path(run, t1))
    pixels = {}
    require(set(expected['vision']) == set(actual['vision'])
            and len(expected['vision']) == 3, 'Restored three-camera set differs')
    for camera in expected['vision']:
        a = expected['vision'][camera]['color']; b = actual['vision'][camera]['color']
        require(a.shape == b.shape == (480, 640, 3) and a.dtype == b.dtype == np.uint8,
                'Restored camera format mismatch')
        delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
        pixels[camera] = {'identical': bool(np.array_equal(a, b)), 'mean_abs_u8': float(delta.mean()),
                          'max_abs_u8': int(delta.max()), 'nonzero_fraction': float(np.count_nonzero(delta) / delta.size)}
    write(run / 'restore_observation_comparison.json', {
        't1': t1, 'images': pixels, 'instruction_exact': actual['instruction'] == expected['instruction'],
        'reference_state': scalar(expected['state']), 'restored_state': scalar(actual['state']),
        'restore_qualified': False, 'interpretation': 'Measured differences, not an inferred sensor equivalence pass.'})
    require(actual['instruction'] == expected['instruction'], 'Instruction changed on restoration')
    write(run / 'restore_boundary.json', {
        'native_count': t1, 'remaining_steps': context.remaining(t1), 'method': 'complete_recorded_prefix_replay',
        'numeric_state_agreement': True, 'restore_qualified': False,
        'unresolved_coverage': result['coverage']['unresolved_requirements'],
        'protocol': 'dual-fate-direct-teacher-v1',
        'purpose': 'Actual direct-teacher prefix boundary; source admission is not teacher success.'})


def replay_prefix(task, recorder, run, publish_boundary, config, write, *, context=None, main_execute=False):
    require(main_execute is True, 'Main must explicitly authorize this one restored prefix after actual gates')
    context = bound_run(run, config, context)
    run = Path(run)
    collector, observation_io = checked_recording_modules(context)
    require(type(recorder) is collector.DenseRecorder
            and type(recorder.writer) is observation_io.OrderedObservationWriter
            and recorder.task is task and Path(recorder.run) == run
            and callable(write) and callable(publish_boundary), 'Existing sole native25 recorder required')
    native25_recording(run)
    check_native(context, native_snapshot(task), expected_count=0)
    reference = Path(config['reference_run']); t1 = config['t1']
    rows = recorded_prefix(reference, config, context=context)
    recorder.owner = 'recorded_DM0.5_prefix_replay'
    for index, row in enumerate(rows[:t1]):
        require(row['index'] == index and row['native_count_after'] == index + 1, 'Original action prefix gap')
        require(task.control_steps == index and not task.done, 'Unexpected replay boundary/termination')
        recorder.macro = index
        check_native(context, native_snapshot(task), expected_count=index)
        recorder.writer.check()
        task.control_step(row['action'])
        require(task.control_steps == index + 1, 'Replay accounting changed')
        require(recorder.control_index == row['actual_control_steps'], 'Replay physical-control count mismatch')
        event = {**row, 'role': 'recorded_DM0.5_prefix_replay', 'original_native_event': str(reference / 'native_actions.jsonl'),
                 'terminated': task.terminated, 'truncated': task.truncated, 'native_success': task.success,
                 'actual_control_steps': recorder.control_index}
        task._frames.clear()
        publish_boundary(event)
    from ..recording.core import wait_for_native_publication
    wait_for_native_publication(recorder, t1)
    require(not task.done, 'Restored boundary already terminal')
    check_native(context, native_snapshot(task), expected_count=t1)
    compare_auxiliary_render(run, config, write, context=context)
    compare_boundary(run, config, write, context=context)
    recorder.owner = 'teacher_awaiting_verified_dispatch'


def compare_auxiliary_render(run, config, write, *, context=None):
    """Additional exact saved-state checks; no state repair or qualification."""
    context = bound_run(run, config, context)
    run = Path(run)
    reference = Path(config['reference_run'])
    t1 = config['t1']
    pairs = []
    for filename in ('initial_task_auxiliary.json', f'task_auxiliary_{t1:06d}.json'):
        paths = [reference / filename, run / filename]
        require(all(p.is_file() and not p.is_symlink() for p in paths), 'Missing saved task auxiliary state')
        raws = [read_bytes(p) for p in paths]
        values = [_decode(raw) for raw in raws]
        exact = len({json.dumps(value, sort_keys=True, allow_nan=False) for value in values}) == 1
        pairs.append({'file': filename, 'typed_json_exact': exact,
                      'sha256': [hashlib.sha256(raw).hexdigest() for raw in raws]})
    render = [read_doc(folder / 'camera_profile.json')['rtx_profile']
              for folder in (reference, run)]
    render_exact = render[0] == render[1]
    requested = config['rtx_profile_id']
    requested_exact = all(value['configuration']['profile'] == requested for value in render)
    result = {'task_auxiliary': pairs, 'rtx_profile_exact': render_exact,
              'requested_rtx_profile_exact': requested_exact, 'rtx_profile_id': requested,
              'reference_rtx_profile': render[0], 'restored_rtx_profile': render[1],
              'qualified': False, 'old_suffix_reuse_allowed': False}
    write(run / 'restore_auxiliary_render_comparison.json', result)
    require(all(pair['typed_json_exact'] for pair in pairs), 'Task auxiliary state mismatch')
    require(render_exact and requested_exact, 'RTX profile mismatch')
    result['rich_runtime_identity'] = 'not_recorded_in_source_not_a_fourth_data_condition'
    return result


def prefix_rows(stream, t1, *, total):
    """Pure bounded stream parser; never reads the first post-t1 row."""
    require(type(total) is int and total == _TOTAL and type(t1) is int and t1 == _T1,
            'Only the explicitly bound source-native total and causal cutoff are supported')
    prefix = []
    for _ in range(t1):
        raw = stream.readline(1024 * 1024 + 1)
        require(0 < len(raw) <= 1024 * 1024 and raw.endswith(b'\n'),
                'Missing/oversized/incomplete native prefix row')
        prefix.append(_decode(raw))
    for index, row in enumerate(prefix):
        require(type(row['index']) is int and row['index'] == index
                and type(row['native_count_after']) is int and row['native_count_after'] == index + 1
                and type(row['actual_control_steps']) is int and row['actual_control_steps'] == (index + 1) * 10,
                'Original DM prefix count/role mismatch')
        require(row.get('role', 'DM0.5') == row.get('owner', 'DM0.5') == 'DM0.5'
                and type(row['action']) is dict and row['action']
                and row['native_success'] is False and row['terminated'] is False
                and row['truncated'] is False, 'Prefix is not a completed nonterminal DM action')
    return prefix
