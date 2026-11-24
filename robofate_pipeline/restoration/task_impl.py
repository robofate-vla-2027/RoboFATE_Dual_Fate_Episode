"""Original source-specific checks, bound to external metadata by the factory."""
import hashlib
import json
from pathlib import Path
PROFILE = 'pt_spatial4_collection_candidate_v1'

def need(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def read_ref(ref):
    need(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'Exact FileRef required')
    path = Path(ref['path'])
    need(path.is_absolute() and path.resolve() == path and (not path.is_symlink()), 'Canonical regular input required')
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Pinned Native input changed: ' + str(path))
    return raw


def verify_source(source):
    c = contract()
    need(canonical(source) == canonical(json.loads(read_ref(c['source_evidence']['source']))), 'Exact original bound-source identity required; relabeling is forbidden')
    need(source['task'] == TASK and _identity_id(source) == SOURCE_ID and (type(source['encoded_seed']) is int) and (source['encoded_seed'] == _SOURCE_SEED) and (source['native_ordinal'] == _SOURCE_ORDINAL), 'Wrong bound-source task/seed/sequence')
    return source


def verify_bound(context):
    need(context.task == TASK and context.source_id == SOURCE_ID and (type(context.total) is int) and (context.total == TOTAL) and (type(context.t1) is int) and (context.t1 == T1) and (context.remaining(T1) == REMAINING), 'Wrong bound-source/bound-source/T25 Native context')
    return context


def validate_profile(profile):
    c = contract()
    need(profile['schema'] == 'refate-direct-teacher-restore-profile-v1' and profile['task'] == TASK and (profile['source_id'] == SOURCE_ID) and (profile['reference_run'] == c['reference_run']) and (type(profile['native_step_limit']) is int) and (profile['native_step_limit'] == TOTAL) and (type(profile['t1']) is int) and (profile['t1'] == T1), 'Profile is not original bound-source bound-source/T25')
    need(profile['status'] == 'qualified_for_source_prefix_restore' and profile['restore_method'] == 'complete_recorded_prefix_replay' and (profile['protocol'] == 'dual-fate-direct-teacher-v1') and (profile['renderer_profile_id'] == PROFILE), 'Wrong scoped restore method/RTX profile')
    for key in ('schema_contract', 'thresholds', 'joint_units'):
        need(canonical(profile[key]) == canonical(c[key]), 'bound-source source schema/threshold/units changed: ' + key)
    need(profile['threshold_basis']['observed_native_dt_s'] == 0.004, 'Original timestep changed')
    for key in ('DM_continuation_required', 'DM_private_inference_state_required', 'actual_teacher_boundary_verified', 'serialized_coverage_complete', 'qualified', 'counted_in_quota'):
        need(profile[key] is False, 'Profile may not claim DM continuation/hidden or actual restoration: ' + key)
    need(profile['source_readback_ref'] == c['source_evidence']['readback'], 'Wrong original complete failure readback')
    return profile


def verify_restore_config(config):
    c = contract()
    need(config['task'] == TASK and config['source_id'] == SOURCE_ID and (config['reference_run'] == c['reference_run']) and (type(config['native_step_limit']) is int) and (config['native_step_limit'] == TOTAL) and (type(config['t1']) is int) and (config['t1'] == T1) and (config['rtx_profile_id'] == PROFILE), 'Wrong fixed bound-source source/T25 restore request')
    need(config['source_readback_ref'] == c['source_evidence']['readback'], 'Wrong source readback binding')
    if 'remaining_native_steps' in config:
        need(type(config['remaining_native_steps']) is int and config['remaining_native_steps'] == REMAINING, 'Original remaining475 changed')
    validate_profile(json.loads(read_ref({'path': config['profile_path'], 'sha256': config['profile_sha256']})))
    for role in ('physics', 'auxiliary', 'camera', 'observation', 'result'):
        read_ref(c['source_evidence'][role])
    return config
