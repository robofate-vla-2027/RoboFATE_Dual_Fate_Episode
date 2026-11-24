"""Original native25/PT artifact reader with deployed identity/content edits."""
import argparse

from concurrent.futures import ThreadPoolExecutor

import datetime as dt

import hashlib

import json

import math

from pathlib import Path

import statistics

import sys

import time

def need(condition, message):
    if not condition: raise ValueError(message)

def doc(path):
    return json.loads(path.read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rows(path):
    raw=path.read_text();need(raw.endswith('\n'),'Unpublished final line: '+str(path))
    return [json.loads(x) for x in raw.splitlines()]

def read_observation(folder):
    decode_observation = _COMMON_OBSERVATION_IO.decode_observation
    obs=decode_observation(folder);stats=doc(folder/'image_stats.json')
    need(set(obs['vision'])==set(stats),'Image-stat camera mismatch')
    hashes={};content_diagnostics={}
    for name,data in obs['vision'].items():
        image=data['color'];h=hashlib.sha256(image.tobytes()).hexdigest()
        need(h==stats[name]['sha256'],'Decoded original RGB bytes mismatch: '+str(folder))
        content_diagnostics[name]=_content_policy.validate_rgb(image,stats[name])
        hashes[name]=h
    return {'folder':str(folder),'manifest_sha256':digest(folder/'observation.json'),
            'arrays_sha256':digest(folder/'arrays.npz'),'rgb_sha256':hashes,'observation_content_diagnostics':content_diagnostics}

PT_COLLECTION = 'pt_spatial32_collection_v1'

PT4_COLLECTION = 'pt_spatial4_collection_candidate_v1'

PT_COLLECTION_PROFILES = (PT_COLLECTION, PT4_COLLECTION)

PT_COUNTER = '/rtx/externalFrameCounter'

NATIVE25_SCHEMA = 'refate-native25-v1'

NATIVE25_CAMERAS = {'cam_head', 'cam_left_wrist', 'cam_right_wrist'}

PT_FIXED = {
    '/rtx/rendermode': 'PathTracing', '/rtx/pathtracing/spp': 32,
    '/rtx/pathtracing/totalSpp': 32, '/rtx/pathtracing/adaptiveSampling/enabled': False,
    '/rtx/pathtracing/optixDenoiser/enabled': True,
    '/rtx/pathtracing/optixDenoiser/temporalMode/enabled': False,
    '/rtx/pathtracing/temporalAccumulation': False,
    '/rtx-transient/dldenoiser/enabled': False,
    '/rtx-transient/resetPtAccumOnlyWhenExternalFrameCounterChanges': True,
    '/rtx/pathtracing/cached/enabled': False,
    '/rtx/pathtracing/lightcache/cached/enabled': False,
}

def exact(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)

def at_counter(profile, counter):
    expected = json.loads(json.dumps(profile, allow_nan=False))
    expected['actual_settings'][PT_COUNTER] = counter
    return expected

def need_rtx_engines(engine):
    need(isinstance(engine, dict) and engine.get('status') == 'observed', 'PT Hydra identity unverified')
    names, uids, descriptions = (engine.get(k) for k in ('names', 'uids', 'descriptions'))
    need(isinstance(names, list) and names and all(name == 'rtx' for name in names), 'PT Hydra not exclusively RTX')
    need(isinstance(uids, list) and uids and all(type(uid) is int for uid in uids), 'PT Hydra UIDs missing')
    need(len(set(uids)) == len(uids), 'PT Hydra duplicate UID')
    need(isinstance(descriptions, list) and len(descriptions) == len(uids), 'PT Hydra descriptions missing')
    need(all(isinstance(d, dict) and type(d.get('uid')) is int and d['uid'] == uid
             and d.get('engine_type_name') == 'rtx' for uid, d in zip(uids, descriptions)), 'PT Hydra description mismatch')

def audit_pt_stdout(run, first_fresh_ns):
    optional_suffix = ('[Error] [rtx.postprocessing.plugin] createDLSSContext error: '
                       'unable to initialize context. Optional DLSS feature is disabled.')
    optional, fatal = [], []
    for number, line in enumerate((run / 'stdout.log').read_text().splitlines(), 1):
        lower = line.lower()
        record = {'line': number, 'text': line}
        if line.endswith(optional_suffix):
            # Only a timestamp interval wholly before fresh capture can be startup-only.
            stamp = line.split(' ', 1)[0]
            try:
                need(stamp.endswith('Z'), 'Missing UTC startup timestamp')
                parsed = dt.datetime.fromisoformat(stamp[:-1] + '+00:00')
                precision = 10 ** (-len(stamp[:-1].split('.', 1)[1])) if '.' in stamp else 1
                definitely_before = (parsed.timestamp() + precision) * 1e9 <= first_fresh_ns
            except (ValueError, OverflowError):
                definitely_before = False
            if definitely_before:
                optional.append({**record, 'scope': 'exact optional DLSS startup message before counter 1; PT_FIXED disables DLSS-RR'})
                continue
        renderer_error = ('[error]' in lower or '[fatal]' in lower) and any(token in lower for token in
                          ('rtx.', 'hydra', 'graphics-vulkan', 'gpu.foundation', 'renderer', 'usd_mdl', 'sdrshadernode', 'optix'))
        explicit_fatal = any(token in lower for token in
                             ('skipping denoising', 'optix_error_', 'unable to load denoiser weights',
                              'could not open optix denoiser weights', 'vk_error_device_lost', 'error_device_lost'))
        if renderer_error or explicit_fatal or line.endswith(optional_suffix):
            fatal.append(record)
    need(not fatal, 'PT fatal renderer/OptiX stdout records: ' + json.dumps(fatal))
    return {'stdout_sha256': digest(run / 'stdout.log'), 'fatal_records': fatal,
            'allowed_optional_startup_records': optional, 'first_counter_one_wall_time_ns': first_fresh_ns}

def need_capture_readiness(value):
    need(isinstance(value, dict), 'PT initial loading record missing')
    status = value.get('stage_loading_status')
    need(isinstance(status, list) and len(status) == 3, 'PT initial loading shape')
    need(type(status[1]) is int and type(status[2]) is int and 0 <= status[1] == status[2], 'PT initial loading incomplete')
    need(value.get('stage_streaming_busy') is False, 'PT initial streaming busy/unverified')

def audit_capture_boundary(boundary, index, *, event=None):
    need(isinstance(boundary, dict), 'PT capture boundary missing')
    core = {'previous_external_counter': index, 'requested_external_counter': index + 1,
            'physical_reset': False, 'rtx_rng_reset_claimed': False}
    need(all(key in boundary and exact(boundary[key], value) for key, value in core.items()), 'PT dense/initial accumulation core mismatch')
    need(boundary.get('actual_mode') == 'PathTracing', 'PT capture actual mode mismatch')
    stamps = []
    snapshots = []
    for side in ('before', 'after'):
        snapshot = boundary.get(side)
        need(isinstance(snapshot, dict) and snapshot.get('schema') == 'v007.read_only_capture.1', 'PT capture snapshot missing/schema')
        need(snapshot.get('phase') == side + '_existing_capture', 'PT capture snapshot phase')
        need(type(snapshot.get('capture_index')) is int and snapshot['capture_index'] == index, 'PT capture snapshot index')
        native = snapshot.get('native_count')
        physical = snapshot.get('physics_time')
        need(type(native) is int and native >= 0, 'PT capture native count invalid')
        need(type(physical) in (int, float) and math.isfinite(physical), 'PT capture physical time invalid')
        if event is None:
            need(index == 0 and native == 0, 'PT initial native count nonzero')
        else:
            need(native == event['macro_index'], 'PT capture native count must equal pre-macro index')
            need(physical == event['simulation_time_after'], 'PT capture time must equal post-control event time')
        settings = snapshot.get('carb_settings')
        need(isinstance(settings, dict), 'PT capture settings missing')
        for key, value in ((PT_COUNTER, index + 1), ('/rtx/rendermode', 'PathTracing'),
                           ('/rtx-transient/resetPtAccumOnlyWhenExternalFrameCounterChanges', True)):
            need(exact(settings.get(key), {'status': 'observed', 'value': value}), 'PT capture setting/counter mismatch: ' + key)
        clock = snapshot.get('dispatcher_reference_time')
        need(isinstance(clock, dict) and clock.get('status') == 'stable_read', 'PT capture dispatcher unavailable/unstable')
        first, second = clock.get('first'), clock.get('second')
        need(all(isinstance(pair, list) and len(pair) == 2 and all(type(v) is int for v in pair) and pair[1] > 0
                 for pair in (first, second)), 'PT capture dispatcher pair invalid')
        need(first == second, 'PT capture dispatcher changed within read')
        stamps.append(first)
        snapshots.append(snapshot)
    need(stamps[1][0] == stamps[0][0] + 1 and stamps[1][1] == stamps[0][1], 'PT capture dispatcher not exactly one frame')
    need(snapshots[0]['native_count'] == snapshots[1]['native_count']
         and snapshots[0]['physics_time'] == snapshots[1]['physics_time'], 'PT capture advanced native/physics')
    need(exact(snapshots[0]['carb_settings'], snapshots[1]['carb_settings']), 'PT capture settings changed within capture')
    for side in ('loading_before', 'loading_after'):
        need(side in boundary, 'PT capture loading field missing')
        if event is None:
            need_capture_readiness(boundary[side])
        else:
            need(boundary[side] is None, 'PT dense loading must be null')

def audit_initial_capture(run):
    initial = doc(run/'initial_capture.json')
    need(initial.get('physics_rng_exact') is True, 'PT initial physics/RNG equality missing/false')
    need(type(initial.get('native_count')) is int and initial['native_count'] == 0, 'PT initial native count invalid')
    audit_capture_boundary(initial.get('boundary'), 0)
    return {'sha256': digest(run/'initial_capture.json'), 'physics_rng_exact_recorded': True,
            'native_count': 0, 'first_boundary_counter': 1,
            'loading_ready_before_and_after_recorded': True}

def native25_pt_zero_evidence(run, covered):
    capture_path = Path(_CAPTURE_REF['path'])
    capture_sha = _CAPTURE_REF['sha256']
    need(capture_sha == _CAPTURE_EXPECTED_SHA,
         'Native25 preincrement observation requires the pinned capture implementation')
    initial_path = run / 'initial_capture.json'
    initial = doc(initial_path)
    boundary = initial['boundary']
    need(type(initial.get('native_count')) is int and initial['native_count'] == 0
         and initial.get('physics_rng_exact') is True, 'Native25 initial preparation state missing')
    for key, expected in (('previous_external_counter', 0), ('requested_external_counter', 1),
                          ('visual_index', 0), ('low_level_control_index', 0)):
        need(type(boundary.get(key)) is int and boundary[key] == expected,
             'Native25 preparation counter identity: ' + key)
    # capture() observes zero before set_int(1); its snapshots/raw observer run after that increment.
    for side in ('before', 'after'):
        need(exact(boundary[side]['carb_settings'].get(PT_COUNTER),
                   {'status': 'observed', 'value': 1}), 'Native25 first capture counter observation')
    need(1 in covered, 'Native25 first fresh counter missing')
    return {'basis': 'pinned_capture_preincrement_observation',
            'initial_capture_ref': {'path': str(initial_path), 'sha256': digest(initial_path)},
            'capture_implementation_ref': {'path': str(capture_path), 'sha256': capture_sha},
            'observed_preincrement_counter': 0, 'observed_first_capture_counter': 1,
            'raw_counter_zero_observed': 0 in covered}

def audit_pt_collection(run, camera_docs, dense, count, result, *, recording_schema=None):
    need(recording_schema in (None, NATIVE25_SCHEMA), 'Unknown PT recording schema')
    native25 = recording_schema == NATIVE25_SCHEMA
    counter_stride = 1 if native25 else 10
    capture_events = dense[9::10] if native25 else dense
    need(len(dense) == 10 * count, 'PT actual/native control count mismatch')
    baseline = camera_docs[0]['rtx_profile']
    profile_name = baseline['configuration']['profile']
    need(profile_name in PT_COLLECTION_PROFILES, 'PT collection profile mismatch')
    fixed = dict(PT_FIXED)
    if profile_name == PT4_COLLECTION:
        fixed.update({'/rtx/pathtracing/spp': 4, '/rtx/pathtracing/totalSpp': 4})
        requested = baseline['configuration'].get('requested_carb_settings', {})
        for key, value in {**fixed, PT_COUNTER: 0}.items():
            need(key in requested and exact(requested[key], value), 'PT4 requested setting mismatch: ' + key)
        need(baseline['configuration'].get('qualified') is False and baseline.get('qualification') == 'unverified', 'PT4 renderer must remain unverified')
        need(result.get('qualified') is False and result.get('counted_in_quota') is False, 'PT4 result must remain unqualified and uncounted')
    need(baseline.get('attached_hydra_engine_types') == ['rtx'], 'PT camera Hydra identity missing/non-RTX')
    for key, value in fixed.items():
        need(key in baseline['actual_settings'] and exact(baseline['actual_settings'][key], value), 'PT fixed setting mismatch: ' + key)
    for i, camera in enumerate(camera_docs):
        profile = camera['rtx_profile']
        value = profile['actual_settings'].get(PT_COUNTER)
        need(type(value) is int and value == 1 + counter_stride * i, 'PT macro external counter mismatch')
        need(exact(profile, at_counter(baseline, 1 + counter_stride * i)), 'PT camera profile/settings drift beyond prescribed counter')
    initial_capture = audit_initial_capture(run)
    for i, event in enumerate(capture_events, 1):
        audit_capture_boundary(event.get('rtx_accumulation_boundary'), i, event=event)
    raw = rows(run / 'renderer_raw_observations.jsonl')
    need(raw, 'PT raw observations missing')
    last = None
    covered = set()
    first_fresh_ns = None
    for index, record in enumerate(raw):
        need(type(record.get('observation_index')) is int and record['observation_index'] == index, 'PT raw record index gap')
        need(record.get('profile') == profile_name and record.get('read_errors') == {}, 'PT raw profile/read error')
        need(exact(record['configuration'], baseline['configuration']), 'PT raw configuration drift')
        settings = record['actual_settings']
        counter = settings.get(PT_COUNTER)
        need(type(counter) is int and 0 <= counter <= 1 + len(capture_events), 'PT raw counter out of range/type')
        need((last is None and counter in (0, 1)) or (last is not None and last <= counter <= last + 1), 'PT raw counter decreases/skips')
        need(exact(settings, at_counter(baseline, counter)['actual_settings']), 'PT raw actual settings drift')
        need_rtx_engines(record.get('attached_hydra_engines'))
        if counter == 1 and first_fresh_ns is None:
            first_fresh_ns = record.get('wall_time_ns')
            need(type(first_fresh_ns) is int and first_fresh_ns > 0, 'PT first fresh timestamp missing')
        covered.add(counter)
        last = counter
    need(covered - {0} == set(range(1, 2 + len(capture_events))) and last == 1 + len(capture_events), 'PT raw counter coverage incomplete')
    preparation_zero = native25_pt_zero_evidence(run, covered) if native25 else None
    stdout = _optional_diagnostic(run/'stdout.log')
    outcome = _optional_diagnostic(run/'collector_outcome.json')
    return {'profile': profile_name, 'initial_capture': initial_capture, 'validated_capture_boundaries': len(capture_events)+1,
            'raw_observations': len(raw), 'counter_first_fresh': 1,
            'counter_final': last, 'all_required_counters_covered': True,
            'leading_counter_zero_preparation_reads': sum(row['actual_settings'][PT_COUNTER] == 0 for row in raw),
            'preparation_counter_zero_evidence': preparation_zero,
            'stdout': stdout, 'collector_outcome_sha256': outcome.get('sha256'),
            'qualified': False, 'limitation': 'Recorded initial USD readiness and capture metadata only; not live asset/renderer qualification or a pixel shortcut.'}

def native25_number(value):
    return type(value) in (int, float) and math.isfinite(value)

def native25_finite_tree(value):
    if isinstance(value, dict):
        for child in value.values():
            native25_finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            native25_finite_tree(child)
    else:
        need(value is None or type(value) in (str, bool, int)
             or type(value) is float and math.isfinite(value), 'Native25 nonfinite/unsupported state')

def native25_folder(run, index):
    return run / ('initial' if index == 0 else f'macro_obs_{index:06d}')

def native25_contract(value):
    need(isinstance(value, dict) and value.get('schema') == NATIVE25_SCHEMA,
         'Native25 recording contract schema')
    need(type(value.get('visual_hz')) is int and value['visual_hz'] == 25, 'Native25 visual rate')
    need(native25_number(value.get('physical_dt_s')) and value['physical_dt_s'] == .004,
         'Native25 physical timestep')
    need(type(value.get('low_level_ticks_per_native')) is int
         and value['low_level_ticks_per_native'] == 10, 'Native25 control decimation')
    native25_finite_tree(value)

def native25_writer_close(value, count):
    need(isinstance(value, dict) and value.get('status') == 'closed'
         and value.get('failed') is False, 'Native25 writer not successfully closed')
    for key, expected in (('submitted_boundaries', count + 1), ('committed_boundaries', count + 1),
                          ('last_committed_boundary', count)):
        need(type(value.get(key)) is int and value[key] == expected, 'Native25 writer counter: ' + key)
    need(value.get('primary_error') is None and value.get('cleanup_errors', []) == [],
         'Native25 writer retains failure details')
    need(value.get('uncommitted_control_records', []) == [], 'Native25 uncommitted physical controls')
    for key, expected in (('last_actual_executed_step', 10 * count), ('visual_captures', count + 1)):
        if key in value:
            need(type(value[key]) is int and value[key] == expected, 'Native25 writer terminal count: ' + key)
    native25_finite_tree(value)

def native25_indices(run, result, actions, dense, observations, history, initial):
    """Metadata only: interval endpoints are not images at a noncapture tick."""
    count = result.get('native_steps')
    need(type(count) is int and count > 0, 'Native25 native count')
    need(type(result.get('actual_control_steps')) is int
         and result['actual_control_steps'] == len(dense) == 10 * count, 'Native25 10:1 physical controls')
    need(result.get('status') == 'native_terminal', 'Native25 missing terminal result')
    for key in ('terminated', 'truncated', 'native_success'):
        need(type(result.get(key)) is bool, 'Native25 result flag: ' + key)
    need(result['terminated'] or result['truncated'], 'Native25 result is not terminal')
    need(len(actions) == count and len(observations) == len(history) == count + 1,
         'Native25 action/visual/history count')
    prevtime = prevwall = None
    owner = None
    for index, event in enumerate(dense, 1):
        macro = (index - 1) // 10
        need(type(event.get('index')) is int and event['index'] == index, 'Native25 control index gap')
        need(type(event.get('macro_index')) is int and event['macro_index'] == macro, 'Native25 macro binding')
        need(type(event.get('native_action_count')) is int and event['native_action_count'] == macro + 1,
             'Native25 native action count')
        need(isinstance(event.get('owner'), str) and event['owner'], 'Native25 missing control owner')
        if owner is None:
            owner = event['owner']
        need(event['owner'] == owner, 'Native25 mixed control owners')
        need(exact(event.get('visual_interval'), [macro, macro + 1]), 'Native25 visual interval')
        need(event.get('obs_before') == str(native25_folder(run, macro))
             and event.get('obs_after') == str(native25_folder(run, macro + 1)), 'Native25 interval endpoint path')
        fresh = index % 10 == 0
        need(type(event.get('fresh_rgb_capture')) is bool and event['fresh_rgb_capture'] == fresh,
             'Native25 fresh RGB tick')
        need(('rtx_accumulation_boundary' in event) == fresh, 'Native25 capture evidence at wrong tick')
        if fresh:
            need(isinstance(event['rtx_accumulation_boundary'], dict), 'Native25 missing PT boundary')
        need(isinstance(event.get('executed_actuator_targets'), list) and event['executed_actuator_targets'],
             'Native25 missing actuator targets')
        need(isinstance(event.get('robot_feedback'), (dict, list)) and event['robot_feedback'],
             'Native25 missing robot feedback')
        need(type(event.get('decimation')) is int and event['decimation'] == 1, 'Native25 physical decimation')
        need(native25_number(event.get('physical_dt_s')) and event['physical_dt_s'] == .004,
             'Native25 physical dt changed')
        t0, t1 = event.get('simulation_time_before'), event.get('simulation_time_after')
        need(native25_number(t0) and native25_number(t1) and abs(t1 - t0 - .004) < 1e-8,
             'Native25 physical time increment')
        need(prevtime is None or abs(t0 - prevtime) < 1e-8, 'Native25 physical trajectory gap')
        w0, w1 = event.get('wall_before_ns'), event.get('wall_after_ns')
        need(type(w0) is int and type(w1) is int and 0 < w0 <= w1, 'Native25 wall timestamps')
        need(prevwall is None or prevwall <= w0, 'Native25 wall sequence overlap')
        prevtime, prevwall = t1, w1
        native25_finite_tree(event)
    for index, action in enumerate(actions):
        for key, expected in (('index', index), ('native_count_after', index + 1),
                              ('actual_control_steps', 10 * (index + 1))):
            need(type(action.get(key)) is int and action[key] == expected, 'Native25 action mapping: ' + key)
        need(isinstance(action.get('action'), (dict, list)) and action['action'], 'Native25 missing model action')
        for key in ('terminated', 'truncated', 'native_success'):
            need(type(action.get(key)) is bool, 'Native25 action flag: ' + key)
        if index < count - 1:
            need(not action['terminated'] and not action['truncated'], 'Native25 action after terminal')
        else:
            need(all(action[key] == result[key] for key in ('terminated', 'truncated', 'native_success')),
                 'Native25 final action/result mismatch')
        native25_finite_tree(action)
    for index, (observation, update) in enumerate(zip(observations, history)):
        need(type(observation.get('index')) is int and observation['index'] == index, 'Native25 visual index gap')
        need(type(observation.get('control_index')) is int and observation['control_index'] == 10 * index,
             'Native25 visual/control binding')
        path = str(native25_folder(run, index))
        need(observation.get('observation_dir') == path, 'Native25 visual path')
        need(type(observation.get('terminal')) is bool and observation['terminal'] == (index == count),
             'Native25 terminal visual boundary')
        need(type(update.get('native_index')) is int and update['native_index'] == index
             and update.get('observation_dir') == path and type(update.get('terminal')) is bool
             and update['terminal'] == observation['terminal'], 'Native25 causal model history')
        expected_time = dense[0]['simulation_time_before'] if index == 0 else dense[10 * index - 1]['simulation_time_after']
        need(native25_number(observation.get('simulation_time'))
             and abs(observation['simulation_time'] - expected_time) < 1e-8, 'Native25 visual physics time')
        need('rtx_accumulation_boundary' in observation, 'Native25 visual PT binding missing')
        boundary = initial.get('boundary') if index == 0 else dense[10 * index - 1]['rtx_accumulation_boundary']
        # Native zero may refer implicitly to the separately retained initial capture.
        if index != 0 or observation['rtx_accumulation_boundary'] is not None:
            need(exact(observation['rtx_accumulation_boundary'], boundary), 'Native25 visual PT evidence mismatch')
        if index == 0:
            need(isinstance(boundary, dict), 'Native25 initial capture missing')
            for side in ('before', 'after'):
                need(native25_number(boundary.get(side, {}).get('physics_time'))
                     and abs(boundary[side]['physics_time'] - expected_time) < 1e-8,
                     'Native25 initial physics/control mismatch')
        native25_finite_tree(observation)
        native25_finite_tree(update)
    return [native25_folder(run, i) for i in range(count + 1)]

def native25_owned(path, *, directory=False):
    need(not path.is_symlink() and path.resolve() == path
         and (path.is_dir() if directory else path.is_file()), 'Native25 noncanonical/missing artifact: ' + str(path))
    return path

def native25_doc(path):
    value = doc(native25_owned(path))
    need(isinstance(value, dict), 'Native25 expected JSON object: ' + str(path))
    native25_finite_tree(value)
    return value

def native25_observation(folder):
    native25_owned(folder, directory=True)
    stats = native25_doc(folder / 'image_stats.json')
    need(set(stats) == NATIVE25_CAMERAS, 'Native25 exactly three named RGB cameras')
    for value in stats.values():
        need(isinstance(value, dict) and exact(value.get('shape'), [480, 640, 3]),
             'Native25 RGB must be 640x480x3')
    return read_observation(folder)

def native25_state(physics, auxiliary, camera, index, limit, action):
    required = {'robots', 'objects', 'control', 'task', 'reward', 'rng', 'unexposed'}
    need(required <= set(physics), 'Native25 incomplete physical state')
    need(all(isinstance(physics[key], dict) for key in required - {'unexposed'})
         and isinstance(physics['unexposed'], list), 'Native25 physical state shape')
    task = physics['task']
    need(exact(task.get('native_count'), [index]) and type(task.get('step_limit')) is int
         and task['step_limit'] == limit and native25_number(task.get('dt')) and task['dt'] == .004,
         'Native25 physical state native budget/count/dt')
    for key in ('end_flag', 'success_flag'):
        need(isinstance(task.get(key), list) and len(task[key]) == 1 and type(task[key][0]) is bool,
             'Native25 physical task flag')
    if action is not None:
        ended = action['terminated'] or action['truncated']
        need(task['end_flag'][0] == ended, 'Native25 state/action terminal mismatch')
        # Native success_flag means "still viable" until end_flag is set.
        if ended:
            need(task['success_flag'][0] == action['native_success'], 'Native25 terminal success mismatch')
        else:
            need(action['native_success'] is False, 'Native25 nonterminal task success claim')
    need({'pre_state', 'robot_origin_endpose', 'joint_ratio_transition_state'} <= set(auxiliary),
         'Native25 missing task auxiliary state')
    need({'prev_control', 'queues'} <= set(physics['control'])
         and {'python', 'numpy', 'torch_cpu', 'torch_cuda'} <= set(physics['rng']), 'Native25 control/RNG state missing')
    need(isinstance(camera.get('cameras'), dict) and set(camera['cameras']) == NATIVE25_CAMERAS,
         'Native25 camera-state identities')
    for value in camera['cameras'].values():
        need(exact(value.get('shape'), [480, 640, 3]), 'Native25 camera-state resolution')
        need(all(key in value for key in ('intrinsic', 'extrinsic', 'actual_usd_cameras')),
             'Native25 camera geometry missing')
    for value in (physics, auxiliary, camera):
        native25_finite_tree(value)

def audit_native25(run, contract):
    started = time.monotonic()
    native25_contract(contract)
    native25_owned(run, directory=True)
    identity = native25_doc(run/'source_identity.json')
    source_id = _bound_source_identity(identity, _NODE['source_id'], _NODE['task'])
    result = native25_doc(run/'result.json')
    ready = native25_doc(run/'ready.json')
    count, limit = result.get('native_steps'), ready.get('native_step_limit')
    need(type(count) is int and type(limit) is int and 0 < count <= limit, 'Bad native count')
    data = [rows(native25_owned(run/name)) for name in
            ('native_actions.jsonl', 'dense/control_steps.jsonl', 'native_observations.jsonl')]
    actions, dense, native_observations = data
    history, history_provenance = _boundary_index(run, native_observations)
    initial = native25_doc(run/'initial_capture.json')
    folders = native25_indices(run, result, actions, dense, native_observations, history, initial)
    need(not list((run/'dense').glob('obs_*')) and not (run/'macro_obs_000000').exists(), 'Native25 duplicated legacy visual directories')
    need(set(run.glob('macro_obs_*')) == set(folders[1:]), 'Native25 extra/missing macro observations')
    for folder in folders:
        native25_owned(folder, directory=True)
    writer = _optional_diagnostic(run/'writer_close.json')
    timing = _optional_diagnostic(run/'timing_summary.json')
    camera_docs, camera, states = [], [], []
    for prefix in ('camera_', 'physics_', 'task_auxiliary_'):
        expected = {run/f'{prefix}{i:06d}.json' for i in range(count + 1)}
        if prefix == 'camera_':
            expected.add(run/'camera_profile.json')
        need(set(run.glob(prefix + '*.json')) == expected, 'Native25 indexed state inventory: ' + prefix)
    for i in range(count + 1):
        camera_doc = native25_doc(run/f'camera_{i:06d}.json')
        physics, auxiliary = (native25_doc(run/f'{prefix}{i:06d}.json') for prefix in ('physics_', 'task_auxiliary_'))
        native25_state(physics, auxiliary, camera_doc, i, limit, actions[i - 1] if i else None)
        profile = (camera_doc.get('rtx_profile') or {}).get('configuration', {}).get('profile')
        need(profile in PT_COLLECTION_PROFILES and camera_doc.get('renderer_settings', {}).get('/rtx/rendermode') == 'PathTracing',
             'Native25 requires the original collection PT gate')
        camera_docs.append(camera_doc)
        camera.append({'index': i, 'sha256': digest(run/f'camera_{i:06d}.json'), 'rtx_profile': camera_doc['rtx_profile']})
        states.append({'index': i, **{prefix: digest(run/f'{prefix}{i:06d}.json') for prefix in ('physics_', 'task_auxiliary_')}})

    pt_contract = audit_pt_collection(run, camera_docs, dense, count, result, recording_schema=NATIVE25_SCHEMA)
    # Decode each committed native boundary once; never retry with the legacy reader.
    with ThreadPoolExecutor(max_workers=4) as pool:
        observations = list(pool.map(native25_observation, folders))
    return {'schema': 'completed-run-readback-v1', 'recording_schema': NATIVE25_SCHEMA, 'run': str(run),
            'source_id': source_id, 'finished_at': dt.datetime.now(dt.timezone.utc).isoformat(),
            'file_recording_readback_passed': True, 'result': result, 'native_actions': count,
            'actual_controls': len(dense), 'policy_observations': None,
            'saved_boundary_observations': len(native_observations),
            'boundary_index_provenance': history_provenance,
            'decoded_observation_count': len(observations), 'decoded_RGB_images': 3 * len(observations),
            'readback_wall_seconds': time.monotonic() - started,
            'control_wall_median_seconds': statistics.median((x['wall_after_ns']-x['wall_before_ns'])/1e9 for x in dense),
            'observations': observations, 'camera_records': camera, 'state_records': states,
            'recording_contract_sha256': digest(run/'recording_contract.json'),
            'native_observations_sha256': digest(run/'native_observations.jsonl'),
            'control_steps_sha256': digest(run/'dense/control_steps.jsonl'),
            'native_actions_sha256': digest(run/'native_actions.jsonl'),
            'policy_history_sha256': history_provenance['legacy_history_sha256'],
            'writer_close_sha256': writer.get('sha256'), 'writer_close': writer,
            'timing_summary_sha256': timing.get('sha256'), 'timing_summary': timing,
            'pt_collection_contract': pt_contract, 'qualified': False, 'counted_in_quota': False,
            'limitations': ['Integrity/accounting only; no calibration, same-state recovery or episode qualification.',
                           'Noncapture control ticks reference visual interval endpoints, not fresh tick images.',
                           'Once-only invocation must be reserved by the existing source guard; this reader does not retry.']}

from ..readback import content as _content_policy

def _bound_source_identity(identity, expected_source, expected_task):
    if type(identity) is not dict or identity.get("task") != expected_task:
        raise ValueError("Bound baseline task mismatch")
    present = [key for key in ("source_id", "candidate_source_id") if key in identity]
    if not present:
        raise ValueError("Baseline source identity missing")
    for key in present:
        value = identity[key]
        if type(value) is not str or not value or value != expected_source:
            raise ValueError("Baseline source identity invalid or conflicting: " + key)
    return identity[present[0]]
