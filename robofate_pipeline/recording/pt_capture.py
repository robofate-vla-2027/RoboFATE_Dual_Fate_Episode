"""Actual PT capture boundaries for the selected native task; no physics or RNG rewind."""
from .rtx_profile import PT_COLLECTION_PROFILES, observe
from .render_sampling_observer import snapshot


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def readiness(context):
    loaded = context.get_stage_loading_status()
    streaming = context.get_stage_streaming_status()
    require(isinstance(loaded, (tuple, list)) and len(loaded) == 3,
            'Unexpected stage-loading status')
    require(type(loaded[1]) is int and type(loaded[2]) is int and 0 <= loaded[1] == loaded[2],
            'Initial PT stage files are not ready')
    require(streaming is False, 'Initial PT stage streaming is busy or unverified')
    return {'stage_loading_status': list(loaded), 'stage_streaming_busy': streaming,
            'scope': 'USD streaming only; not a complete material-visual qualification'}


def capture(task, index, *, initial=False, settings=None, graph=None, context=None, observer=None):
    require(task._refate_rtx_profile in PT_COLLECTION_PROFILES, 'Wrong PT capture profile')
    limit = int(task.native.step_lim)
    interval = int(task.native.obs_manager.collect_interval)
    require(limit > 0 and task.max_steps == limit and interval == 10,
            'PT native budget or frozen control mapping differs')
    require(type(index) is int and 0 <= index <= limit,
            'PT visual capture outside native action budget')
    require(initial == (index == 0), 'PT initial/dense capture identity mismatch')
    require(task._refate_pt_capture_index == index, 'PT capture sequence mismatch')
    if settings is None:
        import carb
        settings = carb.settings.get_settings()
    if graph is None:
        import omni.graph.core as graph
    if initial and context is None:
        import omni.usd
        context = omni.usd.get_context()
    key = '/rtx/externalFrameCounter'
    previous = settings.get(key)
    require(type(previous) is int and previous == index, 'PT external counter drift')
    n = task.native
    physical_time = n.sim.sim.current_time
    native_count = task.control_steps
    loading = readiness(context) if initial else None
    settings.set_int(key, index + 1)
    task._refate_pt_capture_index = index + 1
    before = snapshot(settings, graph, phase='before_existing_capture', native_count=native_count,
                      physics_time=physical_time, capture_index=index)
    n.obs_manager.render_for_capture()
    profile = (observer or observe)(task, settings)
    after = snapshot(settings, graph, phase='after_existing_capture', native_count=task.control_steps,
                     physics_time=n.sim.sim.current_time, capture_index=index)
    require(physical_time == n.sim.sim.current_time and native_count == task.control_steps,
            'PT capture advanced native count or physical time')
    pre, post = before['dispatcher_reference_time'], after['dispatcher_reference_time']
    require(pre.get('status') == post.get('status') == 'stable_read', 'PT dispatcher clock unavailable')
    require(pre['first'] == pre['second'] and post['first'] == post['second']
            and post['first'][1] == pre['first'][1]
            and post['first'][0] == pre['first'][0] + 1, 'PT capture did not deliver exactly one frame')
    final_loading = readiness(context) if initial else None
    obs = n.obs_manager.get_obs(env_idx_list=[0])[0]
    obs['env_idx'] = 0
    evidence = {'visual_index':index, 'low_level_control_index':index*interval,
                'previous_external_counter': previous, 'requested_external_counter': index + 1,
                'physical_reset': False, 'rtx_rng_reset_claimed': False,
                'before': before, 'after': after, 'loading_before': loading,
                'loading_after': final_loading, 'actual_mode': profile['actual_settings']['/rtx/rendermode']}
    return obs, evidence
