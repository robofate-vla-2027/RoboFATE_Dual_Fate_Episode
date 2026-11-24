"""Original metadata binder; instantiated by restoration.binding, not imported."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import PurePosixPath
import re
EVO_SCHEMA = 'refate-evo-native25-public-config-v1'
ROLES = ('profile', 'source_identity', 'source_ready', 'native_config', 'restore_config', 'state_scope')
IDENTITY = {'campaign', 'chat', 'active_chat_id', 'session', 'task', 'source_id', 'owner', 'epoch'}
EVO_REF_KEYS = {'source_admission', 'source_readback_ref', 'native25_runtime_refs'}
TASK_ADAPTERS = {_TASK: 'refate-direct-teacher-restore-profile-v1'}

def need(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def profile_total(profile):
    """The verified old bound schema stores the budget in threshold_basis."""
    total = profile['threshold_basis']['native_step_limit']
    need(type(total) is int and total == _TOTAL, 'Unsupported bound native budget')
    if 'native_step_limit' in profile:
        need(type(profile['native_step_limit']) is int and profile['native_step_limit'] == total,
             'Conflicting optional top-level profile budget')
    return total


def check_profile_definition(profile):
    need(profile['task'] == _TASK
         and profile['schema'] == TASK_ADAPTERS[_TASK]
         and profile['status'] == 'qualified_for_source_prefix_restore', 'Wrong bound profile')
    for key, digest in _PROJECTIONS.items():
        need(hashlib.sha256(canonical(profile[key]).encode()).hexdigest() == digest,
             'Unreviewed bound state schema, units or numeric rules: ' + key)
    profile_total(profile)
    need(profile['threshold_basis']['observed_native_dt_s'] == .004
         and profile['protocol'] == 'dual-fate-direct-teacher-v1'
         and profile['restore_method'] == 'complete_recorded_prefix_replay'
         and profile['renderer_profile_id'] == 'pt_spatial4_collection_candidate_v1',
         'Original bound physics/restore/PT4 contract differs')
    need(profile['DM_continuation_required'] is False
         and profile['DM_private_inference_state_required'] is False
         and profile['actual_teacher_boundary_verified'] is False,
         'Profile cannot authorize DM continuation or claim the new boundary')
    # Original deployed binding compares saved cameras during replay.



def valid_ref(ref):
    need(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'Exact FileRef required')
    path = PurePosixPath(ref['path'])
    need(path.is_absolute() and str(path) == ref['path'] and '..' not in path.parts
         and isinstance(ref['sha256'], str) and re.fullmatch('[0-9a-f]{64}', ref['sha256']),
         'Invalid canonical FileRef')
    return ref


def budget(total, used):
    """Arithmetic only; accepting a number does not admit a task/schema."""
    need(type(total) is int and total > 0, 'Invalid total budget')
    need(type(used) is int and 0 <= used <= total, 'Invalid used budget')
    return {'total': total, 'used': used, 'remaining': total - used}


def _decode(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            need(key not in out, 'Duplicate context JSON key')
            out[key] = value
        return out
    def invalid(value):
        raise ValueError('Nonfinite context JSON: ' + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


@dataclass(frozen=True)
class RestoreContext:
    task: str
    total: int
    t1: int
    source_id: str
    owner: tuple
    refs: tuple
    profile_canonical_sha256: str
    restore_canonical_sha256: str
    run: str
    evo_refs_json: str
    source_gaps_json: str

    def remaining(self, used):
        return budget(self.total, used)['remaining']

    def reference(self, role):
        return next({'path': path, 'sha256': digest}
                    for name, path, digest in self.refs if name == role)

    def evo_ref(self, key):
        return _decode(self.evo_refs_json)[key]


def bind_context(snapshots, *, expected_refs, expected_owner, expected_evo_refs, native,
                 expected_native_schema=EVO_SCHEMA):
    """Caller supplies independently trusted pins and an initialized Native snapshot.

    snapshots[role] = {'path': exact_path, 'raw': retained_bytes}.
    Never derive expected_refs from the untrusted snapshots being checked.
    """
    need(set(snapshots) == set(expected_refs) == set(ROLES), 'Exact context roles required')
    need(set(expected_owner) == IDENTITY
         and type(expected_owner['epoch']) is int and expected_owner['epoch'] > 0
         and all(type(v) is str and v for k, v in expected_owner.items() if k != 'epoch'),
         'Exact full independently trusted Evo identity required')
    need(set(expected_evo_refs) == EVO_REF_KEYS, 'Exact independently trusted Evo ref roles required')
    for key in EVO_REF_KEYS - {'native25_runtime_refs'}:
        valid_ref(expected_evo_refs[key])
    need(set(expected_evo_refs['native25_runtime_refs']) == {'collect_source.py', 'observation_io.py'},
         'Exact live native25 module refs required')
    for ref in expected_evo_refs['native25_runtime_refs'].values():
        valid_ref(ref)
    values = {}; refs = []
    for role in ROLES:
        ref = expected_refs[role]; snapshot = snapshots[role]
        need(set(ref) == {'path', 'sha256'} and set(snapshot) == {'path', 'raw'},
             'Exact context reference/snapshot required')
        path = PurePosixPath(ref['path'])
        need(path.is_absolute() and str(path) == ref['path'] and '..' not in path.parts
             and re.fullmatch('[0-9a-f]{64}', ref['sha256']) is not None,
             'Noncanonical or invalid context reference')
        need(snapshot['path'] == ref['path'] and type(snapshot['raw']) is bytes
             and hashlib.sha256(snapshot['raw']).hexdigest() == ref['sha256'],
             'Context reference/hash mismatch: ' + role)
        values[role] = _decode(snapshot['raw'])
        need(type(values[role]) is dict, 'Context document must be an object')
        refs.append((role, ref['path'], ref['sha256']))
    p, source, ready, cfg, restore, scope = [values[role] for role in ROLES]
    check_profile_definition(p)
    task = p['task']
    need(task in TASK_ADAPTERS and p['schema'] == TASK_ADAPTERS[task]
         and p['status'] == 'qualified_for_source_prefix_restore',
         'No reviewed task-specific state adapter')
    need(cfg['schema'] == expected_native_schema == EVO_SCHEMA, 'Native config schema mismatch')
    identity = cfg['identity']; sid = p['source_id']
    need(canonical(identity) == canonical(expected_owner), 'Native owner/epoch/session mismatch')
    need(sid == _SOURCE_ID
         and source['task'] == identity['task'] == restore['task'] == task
         and _identity_id(source) == identity['source_id'] == restore['source_id'] == sid,
         'Task/source context mismatch')
    need(cfg['restore_config'] == expected_refs['restore_config']
         and cfg['profile_ref'] == expected_refs['profile']
         and {'path': restore['profile_path'], 'sha256': restore['profile_sha256']} == expected_refs['profile'],
         'Native/restore/profile reference chain mismatch')
    need(all(canonical(cfg[key]) == canonical(expected_evo_refs[key]) for key in EVO_REF_KEYS),
         'Evo external identity/ref pins differ from independent context')
    need(restore['source_readback_ref'] == cfg['source_readback_ref']
         and restore['rtx_profile_id'] == cfg['rtx_profile'] == p['renderer_profile_id'],
         'Restore/readback/calibration/RTX linkage mismatch')
    need(p['state_scope_ref'] == restore['state_scope_ref'] == expected_refs['state_scope']
         and scope['schema'] == 'refate-direct-teacher-state-scope-v1'
         and scope['status'] == p['status'] and scope['source_id'] == sid
         and type(scope['t1']) is int and scope['t1'] == p['t1']
         and scope['reference_run'] == restore['reference_run']
         and canonical(scope['joint_units']) == canonical(p['joint_units']),
         'Actual source-specific state scope missing or mismatched')
    need(scope['private_cache_exported'] is False and scope['DM_continuation_required'] is False
         and scope['current_teacher_boundary_verified'] is False and scope['counted_in_quota'] is False,
         'Scope cannot claim hidden export/DM continuation/current teacher success')
    for role, filename in (('source_identity', 'source_identity.json'), ('source_ready', 'ready.json')):
        need(expected_refs[role]['path'] == restore['reference_run'] + '/' + filename,
             'Source context is not from the referenced original run')
    total = budget(profile_total(p), p['t1'])['total']
    need(0 < p['t1'] < total and p['t1'] == _T1, 'Completed nonterminal chunk boundary required')
    need(type(restore['native_step_limit']) is int and type(ready['native_step_limit']) is int
         and restore['native_step_limit'] == ready['native_step_limit'] == total
         and type(restore['t1']) is int and restore['t1'] == p['t1'],
         'Profile/source/restore budget mismatch')
    need(type(cfg['native_step_limit']) is int and cfg['native_step_limit'] == total
         and type(cfg['t1']) is int and cfg['t1'] == p['t1'], 'Evo config budget/t1 mismatch')
    run = PurePosixPath(cfg['run_dir']); reference = PurePosixPath(restore['reference_run'])
    need(run.is_absolute() and str(run) == cfg['run_dir'] and '..' not in run.parts
         and run.parent == reference.parent and run != reference,
         'New teacher run cannot replace the original source')
    need(ready.get('actual_shared_recorder') is True and type(ready['native_used']) is int
         and ready['native_used'] == 0, 'Original source must begin independently at zero')
    need('task' not in ready or ready['task'] == task, 'Source ready task mismatch')
    for value in (p, restore):
        if 'remaining_native_steps' in value:
            need(type(value['remaining_native_steps']) is int
                 and value['remaining_native_steps'] == total - p['t1'],
                 'Declared remaining budget mismatch')
    if 'remaining_native_steps' in ready:
        need(type(ready.get('native_count')) is int
             and type(ready['remaining_native_steps']) is int
             and ready['remaining_native_steps'] == budget(total, ready['native_count'])['remaining'],
             'Source ready remaining budget mismatch')
    gaps = scope['raw_source_gaps']
    for key in ('physics_boundary0', 'physics_boundary' + str(p['t1'])):
        need(type(gaps[key]) is list and all(type(v) is str for v in gaps[key]),
             'Missing actual original missing-state declarations')
    context = RestoreContext(task, total, p['t1'], sid, tuple(sorted(expected_owner.items())),
                             tuple(refs), hashlib.sha256(canonical(p).encode()).hexdigest(),
                             hashlib.sha256(canonical(restore).encode()).hexdigest(),
                             str(run), canonical(expected_evo_refs), canonical(gaps))
    need(native['native_count'] in (0, context.t1), 'Context must bind initial or selected boundary')
    check_native(context, native)
    return context


def require_context(context):
    need(type(context) is RestoreContext and context.task in TASK_ADAPTERS,
         'Explicit verified task-specific restore context required')
    need(0 < context.t1 < budget(context.total, context.t1)['total'], 'Invalid context boundary')
    need(context.total == _TOTAL and context.t1 == _T1, 'Unreviewed bound context boundary')
    return context


def check_profile(context, profile):
    context = require_context(context)
    need(hashlib.sha256(canonical(profile).encode()).hexdigest() == context.profile_canonical_sha256,
         'Profile/schema/threshold content differs from bound snapshot')
    check_profile_definition(profile)
    return context


def check_restore(context, restore):
    context = require_context(context)
    need(hashlib.sha256(canonical(restore).encode()).hexdigest() == context.restore_canonical_sha256
         and restore['source_id'] == context.source_id
         and type(restore['t1']) is int and restore['t1'] == context.t1
         and type(restore['native_step_limit']) is int and restore['native_step_limit'] == context.total
         and {'path': restore['profile_path'], 'sha256': restore['profile_sha256']} == context.reference('profile')
         and context.reference('source_identity')['path'] == restore['reference_run'] + '/source_identity.json',
         'Restore request differs from bound context')
    return context


def native_snapshot(task):
    counts = task.native.take_action_cnt
    counts = list(counts) if type(counts) is list else counts.tolist()
    return {'task': task.task_name, 'native_step_limit': task.native_step_limit,
            'max_steps': task.max_steps, 'step_limit': task.native.step_lim,
            'native_count': task.control_steps, 'dt': task.native.dt,
            'native_count_state': counts,
            'official_action_limit': task.official_action_limit}


def check_native(context, native, expected_count=None):
    context = require_context(context)
    need(native['task'] == context.task, 'Actual Native task mismatch')
    for key in ('native_step_limit', 'max_steps', 'step_limit'):
        need(type(native[key]) is int and native[key] == context.total, 'Actual Native budget mismatch')
    budget(context.total, native['native_count'])
    need(native['official_action_limit'] is True
         and type(native['native_count_state']) is list and len(native['native_count_state']) == 1
         and type(native['native_count_state'][0]) is int
         and native['native_count_state'] == [native['native_count']],
         'Wrapper/native actual counter or official-limit mismatch')
    need(expected_count is None or native['native_count'] == expected_count, 'Actual Native boundary mismatch')
    need(type(native['dt']) in (int, float) and native['dt'] == .004, 'Original physics dt changed')
    return context
