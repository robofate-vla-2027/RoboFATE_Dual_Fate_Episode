"""Original physical/RNG comparator; agreement never means qualification."""
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
MISSING = object()

def _reject_constant(value):
    raise ValueError('Non-finite JSON constant: ' + value)


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _parse_json(raw):
    return json.loads(raw, parse_constant=_reject_constant,
                      object_pairs_hook=_unique_keys)


def load_json(path):
    return _parse_json(Path(path).read_bytes())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _finite_json(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('Non-finite numeric state')
        return
    if type(value) is list:
        for item in value: _finite_json(item)
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for item in value.values(): _finite_json(item)
        return
    raise ValueError('State must contain JSON types only')


def _at(obj, parts):
    for part in parts:
        if not isinstance(obj, dict) or part not in obj:
            return MISSING
        obj = obj[part]
    return obj


def _short(value):
    if value is MISSING: return {'missing': True}
    if type(value) in (list, dict): return {'type': type(value).__name__, 'length': len(value)}
    return value[:160] if isinstance(value, str) else value


def _exact_differences(left, right, limit=24):
    details = []; count = 0
    def walk(a, b, path):
        nonlocal count
        if type(a) is type(b) and isinstance(a, dict):
            for key in sorted(set(a) | set(b)):
                walk(a.get(key, MISSING), b.get(key, MISSING), path + '/' + key)
        elif type(a) is type(b) and isinstance(a, list):
            for i in range(max(len(a), len(b))):
                walk(a[i] if i < len(a) else MISSING, b[i] if i < len(b) else MISSING, path + '/' + str(i))
        elif type(a) is not type(b) or a != b:
            count += 1
            if len(details) < limit:
                details.append({'path': path or '/', 'reference': _short(a), 'candidate': _short(b)})
    walk(left, right, '')
    return count, details


def _vector(value, size):
    if type(value) is not list or len(value) != size:
        raise ValueError('Expected numeric vector length ' + str(size))
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
        raise ValueError('Invalid/non-finite numeric vector')
    return [float(v) for v in value]


def quaternion_angle_rad(left, right, norm_tolerance=1e-6):
    """wxyz geodesic angle; q and -q agree, identical inputs have exact zero."""
    a, b = _vector(left, 4), _vector(right, 4)
    na, nb = math.hypot(*a), math.hypot(*b)
    if na == 0 or nb == 0 or max(abs(na - 1), abs(nb - 1)) > norm_tolerance:
        raise ValueError('Zero or non-unit quaternion beyond candidate normalization allowance')
    a, b = [x / na for x in a], [x / nb for x in b]
    if sum(x*y for x, y in zip(a, b)) < 0:
        b = [-x for x in b]
    angle = 4 * math.atan2(math.hypot(*(x-y for x, y in zip(a, b))),
                           math.hypot(*(x+y for x, y in zip(a, b))))
    return angle, abs(na - 1), abs(nb - 1)


def compare_states(reference, candidate, profile=None, *, context=None):
    """Compare serialized states without modifying inputs or writing any files."""
    if profile is None:
        raise ValueError('Explicit bound-task source-scoped profile required; no inherited profile')
    p = copy.deepcopy(profile)
    context = check_profile(context, p)
    verify_bound(context)
    validate_profile(p)
    total = context.total
    if profile_total(p) != total:
        raise ValueError('Task-scoped original native budget differs from verified context')
    if not isinstance(p, dict) or p.get('task') != _TASK or p.get('status') != 'qualified_for_source_prefix_restore':
        raise ValueError('Only a scoped direct-teacher bound-task profile is supported')
    if p.get('schema') != 'refate-direct-teacher-restore-profile-v1':
        raise ValueError('Unsupported task-specific profile schema')
    if not isinstance(p.get('source_id'), str) or re.fullmatch(re.escape(_SOURCE_ID), p['source_id']) is None:
        raise ValueError('Explicit source identity required in the task-scoped profile')
    if type(p.get('t1')) is not int or not 1 <= p['t1'] < total or p['t1'] != context.t1:
        raise ValueError('Source-scoped nonterminal t1 differs from verified context')
    if not isinstance(reference, dict) or not isinstance(candidate, dict):
        raise ValueError('Both states must be JSON objects')
    _finite_json(reference); _finite_json(candidate); _finite_json(p)
    t = p['thresholds']; schema = p['schema_contract']; checks = []; entity_max = {}
    for value in [t[k] for k in ('position_tol_m', 'orientation_tol_rad', 'linear_velocity_tol_mps', 'angular_velocity_tol_radps', 'quaternion_norm_error_max')] + list(t['joint_position_tol'].values()) + list(t['joint_velocity_tol'].values()):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Candidate continuous tolerances must be finite positive numbers')

    def add(path, status, **fields):
        checks.append(dict(path=path, status=status, **fields))

    def present(path, a, b):
        if a is MISSING or b is MISSING:
            add(path, 'missing_both' if a is MISSING and b is MISSING else 'missing_one',
                reference_present=a is not MISSING, candidate_present=b is not MISSING)
            return False
        return True

    def exact(parts, expected=MISSING):
        path = '/'.join(parts); a, b = _at(reference, parts), _at(candidate, parts)
        if not present(path, a, b): return
        count, differences = _exact_differences(a, b)
        expected_mismatch = expected is not MISSING and (canonical(a) != canonical(expected) or canonical(b) != canonical(expected))
        add(path, 'mismatch' if count or expected_mismatch else 'exact_match',
            criterion='strict_typed_JSON', difference_count=count, first_differences=differences,
            profile_identity_mismatch=expected_mismatch)

    def keys(parts, expected, entity=False):
        path = '/'.join(parts) or '$'
        for side, root in [('reference', reference), ('candidate', candidate)]:
            obj = _at(root, parts)
            if obj is MISSING: continue
            if not isinstance(obj, dict):
                add(path, 'invalid', side=side, reason='Expected object'); continue
            for key in sorted(set(obj) - set(expected)):
                add(path + '/' + key, 'mismatch' if entity else 'unhandled_field',
                    side=side, reason='Unexpected entity identity' if entity else 'No comparison contract; not silently ignored')
            for key in expected:
                if key not in obj:
                    add(path + '/' + key, 'missing_one', side=side, reason='Required schema key absent')

    def metric(path, a, b, size, unit, tolerance, entity, label, l2=False):
        if not present(path, a, b): return
        try: a, b = _vector(a, size), _vector(b, size)
        except ValueError as exc:
            add(path, 'invalid', reason=str(exc)); return
        errors = [abs(x-y) for x, y in zip(a, b)]
        maximum = max(errors, default=0.0); norm = math.hypot(*errors)
        measured = norm if l2 else maximum
        add(path, 'within_candidate_tolerance' if measured <= tolerance else 'mismatch',
            unit=unit, max_abs_error=maximum, l2_error=norm, threshold=tolerance,
            criterion='L2' if l2 else 'max_abs', component_errors=errors)
        e = entity_max.setdefault(entity, {})
        e[label + '_max_abs'] = maximum; e[label + '_l2'] = norm

    def rotation(path, a, b, entity, label):
        if not present(path, a, b): return
        try: angle, na, nb = quaternion_angle_rad(a, b, t['quaternion_norm_error_max'])
        except ValueError as exc:
            add(path, 'invalid', reason=str(exc)); return
        add(path, 'within_candidate_tolerance' if angle <= t['orientation_tol_rad'] else 'mismatch',
            unit='rad', angle_error=angle, threshold=t['orientation_tol_rad'],
            reference_quaternion_norm_error=na, candidate_quaternion_norm_error=nb,
            criterion='sign_invariant_geodesic')
        entity_max.setdefault(entity, {})[label + '_rad'] = angle

    def parts_of(parts, width, kind):
        values = []
        for root in (reference, candidate):
            value = _at(root, parts)
            if value is MISSING: values.append(MISSING); continue
            try:
                if kind == 'row':
                    if type(value) is not list or len(value) != 1: raise ValueError('Expected one-environment row')
                    value = _vector(value[0], width)
                elif kind == 'pose':
                    if type(value) is not list or len(value) != 2: raise ValueError('Expected [position3, quaternion4]')
                    value = [_vector(value[0], 3), _vector(value[1], 4)]
                values.append(value)
            except ValueError as exc:
                add('/'.join(parts), 'invalid', reason=str(exc)); values.append(MISSING)
        return values

    keys([], schema['top_keys'])
    keys(['robots'], schema['robots'], entity=True)
    keys(['objects'], schema['objects'], entity=True)
    for rid, info in schema['robots'].items():
        base = ['robots', rid]; entity = '/'.join(base)
        keys(base, info['fields'])
        for field in ('joint_names', 'arm_joint_indices', 'gripper_joint_indices'):
            exact(base + [field], info[field])
        a, b = parts_of(base + ['root_state_w'], 13, 'row')
        def segment(value, start, end): return value[start:end] if value is not MISSING else MISSING
        for name, start, end, unit, tol in [('position', 0, 3, 'm', t['position_tol_m']),
                                         ('linear_velocity', 7, 10, 'm/s', t['linear_velocity_tol_mps']),
                                         ('angular_velocity', 10, 13, 'rad/s', t['angular_velocity_tol_radps'])]:
            metric(entity + '/root_state_w/' + name, segment(a, start, end), segment(b, start, end), 3, unit, tol, entity, 'root_' + name + '_' + unit.replace('/', 'p'), l2=True)
        rotation(entity + '/root_state_w/orientation', segment(a, 3, 7), segment(b, 3, 7), entity, 'root_orientation')
        for field in ('joint_pos', 'joint_vel', 'joint_pos_target', 'joint_vel_target', 'joint_effort_target'):
            a, b = parts_of(base + [field], len(info['joint_names']), 'row')
            for role, indices in [('revolute', info['arm_joint_indices']), ('prismatic', info['gripper_joint_indices'])]:
                velocity = 'vel' in field
                if 'effort' in field:
                    unit = 'N*m' if role == 'revolute' else 'N'; tol = 0.0
                else:
                    unit = ('rad' if role == 'revolute' else 'm') + ('/s' if velocity else '')
                    key = role + ('_radps' if role == 'revolute' and velocity else '_rad' if role == 'revolute' else '_mps' if velocity else '_m')
                    tol = t['joint_velocity_tol' if velocity else 'joint_position_tol'][key]
                aa = [a[i] for i in indices] if a is not MISSING else MISSING
                bb = [b[i] for i in indices] if b is not MISSING else MISSING
                metric(entity + '/' + field + '/' + role, aa, bb, len(indices), unit, tol, entity, field + '_' + role)

    for oid, info in schema['objects'].items():
        base = ['objects', oid]; entity = '/'.join(base)
        keys(base, info['fields'])
        for field in ('class', 'prim_path'): exact(base + [field], info[field])
        a, b = parts_of(base + ['world_pose'], None, 'pose')
        metric(entity + '/position', a[0] if a is not MISSING else MISSING, b[0] if b is not MISSING else MISSING,
               3, 'm', t['position_tol_m'], entity, 'position_m', l2=True)
        rotation(entity + '/orientation', a[1] if a is not MISSING else MISSING, b[1] if b is not MISSING else MISSING, entity, 'orientation')
        for field, unit, tol in [('linear_velocity', 'm/s', t['linear_velocity_tol_mps']), ('angular_velocity', 'rad/s', t['angular_velocity_tol_radps'])]:
            metric(entity + '/' + field, _at(reference, base + [field]), _at(candidate, base + [field]), 3, unit, tol, entity, field, l2=True)
        exact(base + ['sleeping'])
        for root in (reference, candidate):
            value = _at(root, base + ['sleeping'])
            if value is not MISSING and type(value) is not bool:
                add(entity + '/sleeping', 'invalid', reason='Sleep state must be a boolean')

    for section in ('task', 'control', 'reward', 'rng'):
        keys([section], schema[section + '_keys'])
        for field in schema[section + '_keys']:
            exact([section, field], total if section == 'task' and field == 'step_limit'
                  else [context.t1] if section == 'task' and field == 'native_count'
                  else p['threshold_basis']['observed_native_dt_s'] if section == 'task' and field == 'dt' else MISSING)
    for side, root in [('reference', reference), ('candidate', candidate)]:
        for field in ('native_count', 'end_flag', 'success_flag'):
            value = _at(root, ['task', field])
            if value is MISSING: continue
            valid = type(value) is list and len(value) == 1
            valid = valid and (type(value[0]) is int and 0 <= value[0] <= total if field == 'native_count' else type(value[0]) is bool)
            if not valid: add('task/' + field, 'invalid', side=side, reason='Invalid discrete task shape/type/range')
        for field, length in schema['rng_top_lengths'].items():
            value = _at(root, ['rng', field])
            if value is not MISSING and (type(value) is not list or len(value) != length):
                add('rng/' + field, 'invalid', side=side, reason='Recorded RNG container length/type changed')
        for field in ('queues', 'prev_control'):
            value = _at(root, ['control', field])
            if value is not MISSING and type(value) is not list:
                add('control/' + field, 'invalid', side=side, reason='Expected ordered control list')
    exact(['unexposed'])
    declarations = []
    for side, root in [('reference', reference), ('candidate', candidate)]:
        value = root.get('unexposed', MISSING)
        if value is MISSING: continue
        if type(value) is not list or any(type(v) is not str for v in value):
            add('unexposed', 'invalid', side=side, reason='Expected list of missing-state declarations')
        else: declarations.extend(value)
    counts = Counter(c['status'] for c in checks)
    mismatch = counts['mismatch']; invalid = counts['invalid']
    status = 'invalid' if invalid else 'mismatch' if mismatch else 'indeterminate'
    evaluated = counts['exact_match'] + counts['within_candidate_tolerance'] + mismatch
    unresolved = copy.deepcopy(p['coverage_requirements'])
    missing_checks = [c for c in checks if c['status'].startswith('missing') or c['status'] == 'unhandled_field']
    missing_paths = sorted({c['path'] for c in missing_checks})
    return {
        'schema': 'refate-xylophone-objects-by-language-state-diff-v1', 'task': _TASK,
        'profile_id': p['profile_id'], 'profile_status': p['status'],
        'profile_content_sha256': hashlib.sha256(canonical(p).encode()).hexdigest(),
        'context_budget': {'total': total, 't1': context.t1, 'remaining': context.remaining(context.t1)},
        'comparison_status': status, 'restore_qualified': False,
        'old_failure_suffix_reuse_allowed': False,
        'all_compared_fields_within_candidate_rules': evaluated > 0 and not mismatch and not invalid,
        'per_entity_max_errors': entity_max, 'checks': checks,
        'coverage': {'complete': False, 'check_count': len(checks), 'evaluated_check_count': evaluated,
                     'status_counts': dict(counts),
                     'required_serialized_fields_present_and_handled': not missing_checks and not invalid,
                     'missing_or_unhandled_path_count': len(missing_paths),
                     'missing_or_unhandled_paths': missing_paths,
                     'missing_or_unhandled_checks': missing_checks,
                     'declared_unexposed_union': sorted(set(declarations)),
                     'unresolved_requirements': unresolved,
                     'joint_unit_mapping_status': (p['joint_units']['mapping_status'] if isinstance(p['joint_units'], dict)
                                                   else 'see_main_source_scope_unit_records'),
                     'joint_unit_records': p['joint_units']},
        'interpretation': 'Agreement only on covered serialized fields under provisional task-specific thresholds. No initial-state equality or missing-field equality establishes t1 restoration, native continuation, renderer compatibility or quota.',
    }


def compare_files(reference_path, candidate_path, profile_path=None, *, context=None):
    context = require_context(context)
    if profile_path is None:
        raise ValueError('Explicit bound-task source-scoped profile path required')
    source = Path(context.reference('source_identity')['path']).parent / f'physics_{context.t1:06d}.json'
    current = Path(context.run) / f'physics_{context.t1:06d}.json'
    if Path(reference_path) != source or Path(candidate_path) not in (source, current):
        raise ValueError('Only the bound actual t1 pair or original self-comparison is permitted')
    # Compare and hash the same byte snapshots, not a later reread of changing files.
    snapshots = [(role, Path(path), read_snapshot(path)) for role, path in
                 [('reference', reference_path), ('candidate', candidate_path), ('profile', profile_path)]]
    if {'path': str(snapshots[2][1]), 'sha256': hashlib.sha256(snapshots[2][2]).hexdigest()} != context.reference('profile'):
        raise ValueError('Compared profile is not the bound profile reference')
    result = compare_states(*[_parse_json(raw) for _, _, raw in snapshots], context=context)
    result['input_files'] = [{'role': role, 'path': str(path),
                              'sha256': hashlib.sha256(raw).hexdigest()}
                             for role, path, raw in snapshots]
    return result


def read_snapshot(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or path.is_symlink():
        raise ValueError('Noncanonical/linked comparison input')
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 256 * 1024 * 1024:
            raise ValueError('Invalid/oversized comparison input')
        raw = stream.read(before.st_size + 1)
        after = os.fstat(stream.fileno())
    sig = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if sig(before) != sig(after) or sig(after) != sig(path.stat()) or len(raw) != before.st_size:
        raise ValueError('Comparison input changed during capture')
    return raw
