"""Shared native recorder entrypoints; no machine or source binding."""
import copy
import json
from pathlib import Path
import time
from . import core as core_module


def publisher(core, task, recorder, run, *, t1, started, runtime_capture):
    need(type(t1) is int and 0 <= t1 < task.native.step_lim, 'Invalid boundary')
    need(callable(runtime_capture), 'Explicit runtime capture required')
    run, run_started = Path(run), started
    restore_config = {'t1': t1}
    capture_runtime_evidence = runtime_capture
    physics_state = core.physics_state
    auxiliary_task_state = core.auxiliary_task_state
    camera_profile = core.camera_profile
    def publish_boundary(action_event=None):
        recorder.writer.check()
        i=task.control_steps
        if recorder.control_index!=10*i or recorder.capture_count!=i+1:
            raise RuntimeError('Native25 actual capture/control counts diverged')
        started=time.perf_counter()
        obsdir=Path(recorder.observation_path(i))
        documents={f'physics_{i:06d}.json':physics_state(task),
            f'task_auxiliary_{i:06d}.json':auxiliary_task_state(task),
            f'camera_{i:06d}.json':camera_profile(task)}
        if i==25 or i==restore_config['t1']:documents[f'runtime_evidence_{i:06d}.json']=capture_runtime_evidence(task)
        boundary={'index':i,'observation_dir':str(obsdir),'terminal':bool(task.done),
            'control_index':recorder.control_index,'simulation_time':task.native.sim.sim.current_time,
            'rtx_accumulation_boundary':recorder.last_capture}
        appends={'dense/control_steps.jsonl':list(recorder.controls)}
        if action_event is not None:appends['native_actions.jsonl']=[action_event]
        appends['native_observations.jsonl']=[boundary]
        appends['ipc/observations.jsonl']=[{key:boundary[key] for key in ('index','observation_dir','terminal')}]
        recorder.phases['state_capture_recording']+=time.perf_counter()-started
        recorder.writer.submit(i,obsdir,None if i==0 else recorder.last_observation,documents,appends)
        recorder.controls=[]
        if i<=3 or i%10==0:
            progress={'native_actions':i,'low_level_controls':recorder.control_index,
                'actual_rgb_capture_sets':recorder.capture_count,'phase_seconds':recorder.phases,
                'writer':recorder.writer.receipt(),'wall_seconds':time.perf_counter()-run_started,
                'scope':'partial_normal_collection_not_end_to_end_validation'}
            temp=run/'recording_progress.tmp'
            temp.write_text(json.dumps(progress,allow_nan=False)+'\n');temp.replace(run/'recording_progress.json')
        return obsdir
    return publish_boundary


def need(ok, message):
    if not ok:
        raise ValueError(message)


def initialize_recording(task, run, *, core_modules, t1=0, runtime_capture, renderer_device):
    core = core_modules['collect_source']
    run = Path(run)
    need(task.control_steps == 0 and not task.done and not (run/'dense').exists(), 'Fresh initial boundary required')
    core.write(run/'initial_physics_state.json', core.physics_state(task))
    core.write(run/'initial_task_auxiliary.json', core.auxiliary_task_state(task))
    before = core.physics_state(task)
    fresh, boundary = core_modules['pt_capture'].capture(task, 0, initial=True)
    unchanged = before == core.physics_state(task)
    core.write(run/'initial_capture.json', {'boundary':boundary, 'physics_rng_exact':unchanged,
                                          'native_count':task.control_steps, 'qualified':False})
    need(unchanged, 'Fresh initial PT capture changed physical/RNG state')
    task.observe(fresh)
    core.save_observation(run/'initial', task.public_observation())
    camera = core.camera_profile(task)
    core.write(run/'camera_profile.json', camera)
    need(camera['renderer_settings']['/renderer/activeGpu'] == renderer_device
         and camera['renderer_settings']['/renderer/multiGpu/enabled'] is False,
         'Actual RTX device differs from the explicit deployment selection')
    need(set(camera['cameras']) == {'cam_head', 'cam_left_wrist', 'cam_right_wrist'}
         and all(c['shape'] == [480,640,3] for c in camera['cameras'].values()),
         'Native camera set/resolution differs')
    (run/'ipc').mkdir()
    recorder = core.DenseRecorder(task, run/'dense')
    try:
        publish = publisher(core, task, recorder, run, t1=t1, started=time.perf_counter(), runtime_capture=runtime_capture)
        publish()
        core.wait_for_native_publication(recorder, 0)
    except BaseException as error:
        recorder.writer.fail(error)
        try:
            recorder.close()
        except BaseException as cleanup:
            if cleanup is not error:
                error.add_note('Initial recorder cleanup: '+repr(cleanup))
        raise
    return recorder, publish


def execute_action(task, recorder, publish, action, *, role, source_event=None):
    """One actual upstream control call and shared publication; unknown outcomes never replay."""
    need(role in ('DM0.5', 'recorded_DM0.5_prefix_replay', 'EmbodiedEvo_teacher'), 'Explicit branch role required')
    recorder.writer.check()
    need(not task.done and getattr(task, '_failure', None) is None and task.control_steps < task.max_steps,
         'Task terminal, failed or budget exhausted')
    before = task.control_steps
    recorder.macro = before
    recorder.owner = role
    begin = time.time_ns()
    try:
        task.control_step(copy.deepcopy(action))
        need(task.control_steps == before+1 and recorder.control_index == 10*(before+1),
             'Actual native/control counts diverged')
        core = core_module
        event = {'index':before, 'role':role, 'action':core.scalar(action), 'native_count_after':task.control_steps,
                 'actual_control_steps':recorder.control_index, 'native_success':bool(task.success),
                 'terminated':bool(task.terminated), 'truncated':bool(task.truncated),
                 'wall_before_ns':begin, 'wall_after_ns':time.time_ns()}
        if role == 'recorded_DM0.5_prefix_replay':
            need(isinstance(source_event, str) and source_event, 'Original action reference required')
            event['original_native_event'] = source_event
        publish(event)
        return event
    except BaseException as error:
        recorder.writer.fail(error)
        raise
