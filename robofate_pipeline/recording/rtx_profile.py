"""Task-local native RenderCfg selection; no renderer, physics, or asset fallback."""
import hashlib
import json
import os
from pathlib import Path
import time

PROFILES = ('original', 'fxaa_v1', 'rtpt_cache_off_v1', 'rtpt_cache_off_v2', 'pt_spatial32_probe_v1', 'pt_spatial32_cacheoff_assets_v2', 'pt_spatial32_collection_v1', 'pt_spatial4_cost_candidate_v1', 'pt_spatial4_collection_candidate_v1')
RTPT_PROFILES = ('rtpt_cache_off_v1', 'rtpt_cache_off_v2')
SETTING_PATHS = (
    '/rtx/rendermode', '/rtx/post/aa/op', '/rtx-transient/post/aa/limitedOps',
    '/rtx-transient/dldenoiser/enabled', '/rtx/directLighting/enabled',
    '/rtx/directLighting/sampledLighting/samplesPerPixel',
    '/rtx/reflections/enabled', '/rtx/indirectDiffuse/enabled',
    '/rtx/shadows/enabled', '/rtx/ambientOcclusion/enabled',
    '/rtx/rtpt/cached/enabled', '/rtx/rtpt/lightcache/cached/enabled',
    '/persistent/rtx/modes/rt/enabled', '/persistent/rtx/modes/pt/enabled',
    '/persistent/rtx/modes/rt2/enabled', '/renderer/activeGpu',
    '/renderer/multiGpu/enabled', '/renderer/renderer',
)


PT_PROBE_PROFILE = 'pt_spatial32_probe_v1'
PT_ASSET_PROFILE = 'pt_spatial32_cacheoff_assets_v2'
PT_COST_CANDIDATE_PROFILE = 'pt_spatial4_cost_candidate_v1'
PT_PROBE_PROFILES = (PT_PROBE_PROFILE, PT_ASSET_PROFILE, PT_COST_CANDIDATE_PROFILE)
PT_COLLECTION_PROFILE = 'pt_spatial32_collection_v1'
PT4_COLLECTION_PROFILE = 'pt_spatial4_collection_candidate_v1'
PT_COLLECTION_PROFILES = (PT_COLLECTION_PROFILE, PT4_COLLECTION_PROFILE)
PT_PROFILES = PT_PROBE_PROFILES + PT_COLLECTION_PROFILES
PT_ASSET_PROFILES = (PT_ASSET_PROFILE, PT_COLLECTION_PROFILE, PT_COST_CANDIDATE_PROFILE, PT4_COLLECTION_PROFILE)
PT_ASSET_SETTINGS = {'/rtx/pathtracing/cached/enabled': False, '/rtx/pathtracing/lightcache/cached/enabled': False}
PT_SETTINGS = {
    '/rtx/rendermode': 'PathTracing',
    '/rtx/pathtracing/spp': 32,
    '/rtx/pathtracing/totalSpp': 32,
    '/rtx/pathtracing/adaptiveSampling/enabled': False,
    '/rtx/pathtracing/optixDenoiser/enabled': True,
    '/rtx/pathtracing/optixDenoiser/temporalMode/enabled': False,
    '/rtx/pathtracing/temporalAccumulation': False,
    '/rtx-transient/dldenoiser/enabled': False,
    '/rtx-transient/resetPtAccumOnlyWhenExternalFrameCounterChanges': True,
    '/rtx/externalFrameCounter': 0,
}
SETTING_PATHS += tuple(key for key in PT_SETTINGS if key not in SETTING_PATHS)
SETTING_PATHS += tuple(PT_ASSET_SETTINGS) + ('/renderer/active', '/rtx/post/histogram/enabled', '/rtx/post/histogram/tau', '/rtx/post/motionblur/enabled', '/rtx/pathtracing/clampSpp', '/rtx/pathtracing/optixDenoiser/blendFactor')


def configure(task, profile, *, evidence_path):
    if profile not in PROFILES:
        raise ValueError('Unsupported task-local RTX profile')
    if hasattr(task, '_refate_rtx_profile'):
        raise RuntimeError('RTX profile is already bound')
    task._refate_rtx_evidence_path = Path(evidence_path)
    task._refate_rtx_observation_index = 0
    task._refate_rtx_profile = profile
    task._refate_pt_capture_index = 0
    task._refate_rtx_configuration = None
    original = task.native.setup_sim_cfg

    def setup(config):
        if task._refate_rtx_configuration is not None:
            raise RuntimeError('Repeated native RTX configuration is not authorized')
        original(config)
        render = task.native.sim_cfg.sim.render
        before = render.antialiasing_mode
        if profile == 'fxaa_v1':
            render.antialiasing_mode = 'FXAA'
        modified = ['antialiasing_mode'] if profile == 'fxaa_v1' else []
        if profile in RTPT_PROFILES:
            overrides = dict(render.carb_settings or {})
            overrides.update({'/rtx/rendermode': 'RealTimePathTracing',
                              '/rtx/rtpt/cached/enabled': False,
                              '/rtx/rtpt/lightcache/cached/enabled': False})
            render.carb_settings = overrides
            render.antialiasing_mode = None
            modified = ['antialiasing_mode=None to avoid native RaytracedLighting override',
                        '/rtx/rendermode', '/rtx/rtpt/cached/enabled',
                        '/rtx/rtpt/lightcache/cached/enabled']
        if profile in PT_PROFILES:
            overrides = dict(render.carb_settings or {})
            overrides.update(pt_settings(profile))
            render.carb_settings = overrides
            render.antialiasing_mode = None
            modified = ['antialiasing_mode=None'] + list(pt_settings(profile))
        task._refate_rtx_configuration = {
            'profile': profile, 'original_antialiasing_mode': before,
            'requested_antialiasing_mode': render.antialiasing_mode,
            'rendering_mode': render.rendering_mode,
            'requested_carb_settings': dict(render.carb_settings or {}),
            'modified_fields': modified,
            'qualified': False,
        }

    task.native.setup_sim_cfg = setup


def _persist_raw(task, config, values, read_errors, engine_observation=None):
    path = task._refate_rtx_evidence_path
    index = task._refate_rtx_observation_index
    record = {'schema': 'align_blocks_renderer_raw_v1', 'observation_index': index,
              'wall_time_ns': time.time_ns(), 'profile': task._refate_rtx_profile,
              'configuration': config, 'actual_settings': values, 'read_errors': read_errors,
              'gate_status': 'not_evaluated', 'qualification': 'unverified',
              'attached_hydra_engines': engine_observation}
    payload = json.dumps(record, allow_nan=False) + '\n'
    # The synchronous append and directory fsync must finish before any gate.
    with path.open('a', encoding='utf-8') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    task._refate_rtx_observation_index = index + 1


def _mismatch(task, message, key, expected, actual):
    return RuntimeError(f'{message}; profile={task._refate_rtx_profile!r}; '
                        f'setting={key!r}; expected={expected!r}; actual={actual!r}; '
                        f'raw_artifact={str(task._refate_rtx_evidence_path)!r}; '
                        f'observation_index={task._refate_rtx_observation_index - 1}')


def observe(task, settings=None, engine_reader=None):
    if settings is None:
        import carb
        settings = carb.settings.get_settings()
    profile = task._refate_rtx_profile
    config = task._refate_rtx_configuration
    values, read_errors = {}, {}
    for key in SETTING_PATHS:
        try:
            values[key] = settings.get(key)
        except Exception as error:
            read_errors[key] = {'type': type(error).__name__, 'message': str(error)}
    engines = None
    if profile in PT_ASSET_PROFILES:
        engines = (engine_reader or read_attached_engines)()
    _persist_raw(task, config, values, read_errors, engines)
    if read_errors:
        raise _mismatch(task, 'RTX settings could not be read', 'settings.get', 'all readable', read_errors)
    if config is None:
        raise _mismatch(task, 'Native RTX profile hook was not executed', 'configuration', 'configured', None)
    if profile in (PT_COST_CANDIDATE_PROFILE, PT4_COLLECTION_PROFILE):
        if config.get('profile') != profile:
            raise _mismatch(task, 'PT4 configuration profile mismatch', 'configuration.profile', profile, config.get('profile'))
        requested = config.get('requested_carb_settings', {})
        for key, expected_value in pt_settings(profile).items():
            if type(requested.get(key)) is not type(expected_value) or requested[key] != expected_value:
                raise _mismatch(task, 'PT4 requested setting mismatch', key, expected_value, requested.get(key))
    expected_mode = expected_render_mode(profile)
    if values['/rtx/rendermode'] != expected_mode:
        raise _mismatch(task, 'RTX rendering mode changed', '/rtx/rendermode', expected_mode, values['/rtx/rendermode'])
    expected = {'original': 4, 'fxaa_v1': 2}.get(profile)
    if expected is not None and values['/rtx/post/aa/op'] != expected:
        raise _mismatch(task, 'Requested antialiasing mode was overwritten', '/rtx/post/aa/op', expected, values['/rtx/post/aa/op'])
    if profile == 'fxaa_v1' and values['/rtx-transient/post/aa/limitedOps'] is not False:
        raise _mismatch(task, 'Native FXAA did not disable AA limited operations', '/rtx-transient/post/aa/limitedOps', False, values['/rtx-transient/post/aa/limitedOps'])
    if profile in RTPT_PROFILES:
        for key in ('/rtx/rtpt/cached/enabled', '/rtx/rtpt/lightcache/cached/enabled'):
            if values[key] is not False:
                raise _mismatch(task, 'RTPT cache override missing or overwritten', key, False, values[key])
    if profile in PT_PROFILES:
        for key, expected_value in pt_settings(profile).items():
            if key == '/rtx/externalFrameCounter':
                continue
            if type(values[key]) is not type(expected_value) or values[key] != expected_value:
                raise _mismatch(task, 'PT diagnostic setting mismatch', key, expected_value, values[key])
    if profile in PT_ASSET_PROFILES and not exclusively_rtx(engines):
        raise _mismatch(task, 'Attached Hydra engine identity unverified or non-RTX',
                        'UsdContext.attached_hydra_engines', 'nonempty, exclusively rtx', engines)
    if profile in PT_COLLECTION_PROFILES:
        expected_counter = task._refate_pt_capture_index
        if type(values['/rtx/externalFrameCounter']) is not int or values['/rtx/externalFrameCounter'] != expected_counter:
            raise _mismatch(task, 'PT collection counter changed', '/rtx/externalFrameCounter', expected_counter, values['/rtx/externalFrameCounter'])
    result = {'configuration': dict(config), 'actual_settings': values,
              'qualification': 'unverified'}
    if engines is not None:
        result['attached_hydra_engine_types'] = sorted(set(engines['names']))
    return result


def startup_kit_args(profile):
    if profile not in PROFILES:
        raise ValueError('Unsupported task-local RTX profile')
    if profile in PT_PROFILES:
        return ' --/persistent/rtx/modes/rt/enabled=true --/persistent/rtx/modes/pt/enabled=true'
    if profile != 'rtpt_cache_off_v2':
        return ''
    # Explicitly enable Real-Time 2.0 while preserving legacy initialization.
    return ' --/persistent/rtx/modes/rt/enabled=true --/persistent/rtx/modes/rt2/enabled=true'


def expected_render_mode(profile):
    if profile not in PROFILES:
        raise ValueError('Unsupported task-local RTX profile')
    if profile in PT_PROFILES:
        return 'PathTracing'
    return 'RealTimePathTracing' if profile in RTPT_PROFILES else 'RaytracedLighting'


def prepare_static_capture(task, settings, index):
    if task._refate_rtx_profile not in PT_PROBE_PROFILES:
        return None
    if type(index) is not int or not 0 <= index < 8:
        raise ValueError('PT diagnostic capture outside frozen eight-capture sequence')
    key = '/rtx/externalFrameCounter'
    previous = settings.get(key)
    if type(previous) is not int or previous != index:
        raise RuntimeError('PT diagnostic accumulation boundary is out of sequence')
    # Installed Replicator uses counter changes to restart PT accumulation.
    settings.set_int(key, index + 1)
    return {'previous_external_counter': previous, 'requested_external_counter': index + 1,
            'physical_reset': False, 'rtx_rng_reset_claimed': False}


def pt_settings(profile):
    values = dict(PT_SETTINGS)
    if profile in (PT_COST_CANDIDATE_PROFILE, PT4_COLLECTION_PROFILE):
        values.update({'/rtx/pathtracing/spp': 4, '/rtx/pathtracing/totalSpp': 4})
    if profile in PT_ASSET_PROFILES:
        values.update(PT_ASSET_SETTINGS)
    return values


def nvoptix_asset_metadata():
    """Verify an explicit external asset pin, never inherit a driver installation."""
    manifest = json.loads(Path(os.environ["ROBOFATE_OPTIX_MANIFEST"]).read_text())
    path = Path(manifest["path"])
    if not path.is_absolute() or path.is_symlink():
        raise RuntimeError("Explicit canonical OptiX asset required")
    size = path.stat().st_size
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if size != manifest["bytes"] or digest != manifest["sha256"]:
        raise RuntimeError("OptiX asset identity mismatch")
    return {"path": str(path), "size": size, "sha256": digest, "binding": "readonly",
            "execution_proven_by_asset_presence": False}


def read_attached_engines(context=None):
    """Read existing Hydra identities; never create, attach, switch or render."""
    try:
        if context is None:
            import omni.usd
            context = omni.usd.get_context()
        names = list(context.get_attached_hydra_engine_names())
        uids = list(context.get_attached_hydra_engine_uids())
        if not all(type(name) is str for name in names) or not all(type(uid) is int for uid in uids):
            raise TypeError('Unexpected Hydra identity types')
        descriptions = []
        for uid in uids:
            desc = context.get_attached_hydra_engine_description(uid)
            if type(desc.engine_type_name) is not str or type(desc.uid) is not int or desc.uid != uid:
                raise TypeError('Hydra description identity mismatch')
            descriptions.append({'uid': uid, 'engine_type_name': desc.engine_type_name,
                                 'device_mask': int(desc.config.device_mask),
                                 'creation_index': int(desc.config.creation_index)})
        return {'status': 'observed', 'names': names, 'uids': uids, 'descriptions': descriptions,
                'context_name': str(context.get_name()), 'scope': 'already_attached_to_default_UsdContext'}
    except Exception as error:
        return {'status': 'error', 'error_type': type(error).__name__, 'message': str(error)}


def exclusively_rtx(engines):
    if not isinstance(engines, dict) or engines.get('status') != 'observed':
        return False
    names, uids, descriptions = (engines.get(key) for key in ('names', 'uids', 'descriptions'))
    if not isinstance(names, list) or not names or any(name != 'rtx' for name in names):
        return False
    if not isinstance(uids, list) or not uids or not all(type(uid) is int for uid in uids):
        return False
    if not isinstance(descriptions, list) or len(descriptions) != len(uids):
        return False
    return all(isinstance(desc, dict) and desc.get('uid') == uid and desc.get('engine_type_name') == 'rtx'
               for uid, desc in zip(uids, descriptions))
