"""Native-rate recorder; deployment and source admission are caller-owned."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np


def scalar(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if hasattr(value, 'detach'): return value.detach().cpu().tolist()
    if isinstance(value, dict): return {str(k):scalar(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [scalar(x) for x in value]
    if value is None or isinstance(value,(str,int,float,bool)): return value
    raise TypeError(type(value).__name__)


def write(path, data):
    payload = json.dumps(scalar(data), indent=2, allow_nan=False)+'\n'
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        f.write(payload); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def save_observation(folder, obs):
    from .observation_io import save_observation as save
    return save(folder, obs)


def auxiliary_task_state(task):
    parser=task.native.reward_manager.func_parser
    return {key:scalar(getattr(parser,key)) for key in ('pre_state','robot_origin_endpose','joint_ratio_transition_state')}


def physics_state(task):
    n=task.native; out={'robots':{},'objects':{},'unexposed':[]}
    for i,(robot,body) in enumerate(zip(n.robot_manager.robot_list,n.robot_manager.robot_key)):
        if robot.robot_type!='arm':continue
        record={'joint_names':list(body.joint_names),'arm_joint_indices':list(robot.arm_joint_indices),'gripper_joint_indices':list(robot.gripper_joint_indices)}
        for name in ['root_state_w','joint_pos','joint_vel','joint_pos_target','joint_vel_target','joint_effort_target']:
            try: record[name]=scalar(getattr(body.data,name))
            except (AttributeError,RuntimeError) as e:out['unexposed'].append(f'robot[{i}].{name}:{type(e).__name__}')
        out['robots'][str(i)]=record
    for collection in ['_rigid_and_dynamic_objects','_articulation_objects','_geometry_objects']:
        for key,obj in getattr(n.scene_manager,collection)[0].items():
            record={'class':type(obj).__name__,'prim_path':str(getattr(obj,'prim_path',getattr(obj,'_prim_path','')))}
            for field,method in [('world_pose','get_world_pose'),('linear_velocity','get_linear_velocity'),('angular_velocity','get_angular_velocity'),('sleeping','is_sleeping')]:
                try:record[field]=scalar(getattr(obj,method)())
                except (AttributeError,RuntimeError,TypeError) as e:out['unexposed'].append(f'{key}.{field}:{type(e).__name__}')
            out['objects'][str(key)]=record
    out['control']={'prev_control':scalar(n.robot_manager.control_manager.prev_control),
        'queues':[[scalar(x.control_info_dict) for x in list(q.control_seq.queue)] for q in n.robot_manager.control_manager.control_queue]}
    out['task']={'native_count':list(n.take_action_cnt),'end_flag':list(n.end_flag),'success_flag':list(n.success),'step_limit':int(n.step_lim),'dt':float(n.dt)}
    out['reward']={}
    for k,v in vars(n.reward_manager).items():
        if k in ['env','func_parser']:continue
        try:out['reward'][k]=scalar(v)
        except TypeError:out['unexposed'].append('reward.'+k)
    out['rng']={'python':scalar(random.getstate()),'numpy':scalar(np.random.get_state())}
    import torch
    out['rng']['torch_cpu']=scalar(torch.get_rng_state())
    out['rng']['torch_cuda']=scalar(torch.cuda.get_rng_state_all()) if torch.cuda.is_initialized() else []
    out['unexposed'] += ['PhysX contact/solver caches: prefix reconstruction not yet verified','constraint/grasp relation completeness not yet verified','simulator RNG completeness not yet verified']
    return out


class DenseRecorder:
    def __init__(self, task, root):
        from .observation_io import OrderedObservationWriter
        if float(task.native.dt)!=0.004 or int(task.native.obs_manager.collect_interval)!=10:
            raise RuntimeError('Frozen physics/native observation interval changed')
        self.task=task; self.root=root; self.run=root.parent
        self.control_index=0; self.owner='DM0.5'; self.targets=[]; self.macro=None
        self.controls=[]; self.last_observation=task.public_observation()
        self.last_observation_time=task.native.sim.sim.current_time
        self.last_capture=json.loads((self.run/'initial_capture.json').read_text())['boundary']
        self.capture_count=1; self.closed_attempted=False
        self.phases={key:0.0 for key in ('state_capture_recording','physics_advance','render_sync_camera_readback','policy_requests')}
        root.mkdir()
        write(self.run/'recording_contract.json',{'schema':'refate-native25-v1','visual_hz':25,
              'physical_dt_s':0.004,'low_level_ticks_per_native':10,'camera_rgb_shape':[480,640,3],
              'camera_aliases':['cam_head','cam_left_wrist','cam_right_wrist'],
              'visual_interval_references_are_not_per_tick_observations':True,
              'raw_executed_commands_are_not_derived_training_labels':True})
        self.writer=OrderedObservationWriter(self.run)
        try:
            self.old_step=task.native.step
            for i,body in enumerate(task.native.robot_manager.robot_key):
                for method in ['set_joint_position_target','set_joint_velocity_target']:
                    if not hasattr(body,method):continue
                    original=getattr(body,method)
                    def wrapped(values,*args,_orig=original,_body=i,_method=method,**kwargs):
                        target={'body_index':_body,'method':_method,'values':scalar(values),'args':scalar(args),'kwargs':scalar(kwargs)}
                        result=self.writer.admit(_orig,values,*args,**kwargs)
                        self.targets.append(target)
                        return result
                    setattr(body,method,wrapped)
            self.original_get_obs=task.native.get_obs
            task.native.get_obs=self.boundary_observation
            task.native.step=self.step
        except BaseException as error:
            self.writer.fail(error)
            try:
                self.close()
            except BaseException as cleanup:
                if cleanup is not error:
                    error.add_note('Recorder initialization cleanup failed: ' + repr(cleanup))
            raise


    def observation_path(self, index):
        return str(self.run/'initial' if index==0 else self.run/f'macro_obs_{index:06d}')

    def boundary_observation(self):
        self.writer.check()
        n=self.task.native
        now=n.sim.sim.current_time
        if self.last_observation is None or now!=self.last_observation_time or self.control_index%10:
            raise RuntimeError('Native requested RGB outside its fresh 25Hz boundary')
        if n.physx_monitor_enabled:
            n._check_physx_broken_envs();n._check_endpose_finite([0])
        if not n.end_flag[0]:n._stream_vision(0,self.last_observation)
        return copy.deepcopy(self.last_observation)

    def feedback(self):
        result={}
        for i,body in enumerate(self.task.native.robot_manager.robot_key):
            result[str(i)]={}
            for name in ('root_state_w','joint_pos','joint_vel','joint_pos_target','joint_vel_target','joint_effort_target'):
                try:result[str(i)][name]=scalar(getattr(body.data,name))
                except (AttributeError,RuntimeError) as error:result[str(i)][name]={'unavailable':type(error).__name__}
        return result

    def step(self,*args,**kwargs):
        self.writer.check()
        self.targets=[]
        begin=time.time_ns();timer=time.perf_counter()
        n=self.task.native;simulation_context=n.sim.sim
        before=simulation_context.current_time
        try:
            result=self.writer.admit(self.old_step,*args,**kwargs)
            self.phases['physics_advance']+=time.perf_counter()-timer
            self.control_index+=1
            if not self.targets:raise RuntimeError('Native step did not expose actual actuator targets')
            timer=time.perf_counter()
            feedback=self.feedback()
            event={'index':self.control_index,'owner':self.owner,'macro_index':self.macro,
                'obs_before':self.observation_path(self.macro),'obs_after':self.observation_path(self.macro+1),
                'visual_interval':[self.macro,self.macro+1],'fresh_rgb_capture':self.control_index%10==0,
                'executed_actuator_targets':copy.deepcopy(self.targets),'robot_feedback':feedback,
                'wall_before_ns':begin,'wall_after_ns':time.time_ns(),'simulation_time_before':before,
                'simulation_time_after':simulation_context.current_time,'physical_dt_s':float(n.dt),
                'decimation':kwargs.get('decimation',args[1] if len(args)>1 else 1),
                'native_action_count':int(n.take_action_cnt[0]),'native_end_flag':bool(n.end_flag[0]),
                'native_success_flag':bool(n.success[0]),'terminal_evaluation_at_this_substep':False}
            self.phases['state_capture_recording']+=time.perf_counter()-timer
            if self.control_index%10==0:
                from .pt_capture import capture
                timer=time.perf_counter()
                obs,boundary=capture(self.task,self.control_index//10)
                self.phases['render_sync_camera_readback']+=time.perf_counter()-timer
                self.last_observation=obs;self.last_observation_time=simulation_context.current_time
                self.last_capture=boundary;self.capture_count+=1
                event['rtx_accumulation_boundary']=boundary
            self.controls.append(event)
            self.writer.check()
            return result
        except BaseException as error:
            self.writer.fail(error)
            raise

    def close(self):
        self.closed_attempted=True
        primary=None
        try:self.writer.close()
        except BaseException as error:primary=error
        try:position=self.feedback()
        except BaseException as error:
            position={'unavailable':repr(error)}
            self.writer.fail(error)
            if primary is None:primary=error
            else:primary.add_note('Stop-position read also failed: '+repr(error))
        receipt=self.writer.receipt()
        receipt.update({'last_actual_executed_step':self.control_index,'actual_stop_position':position,
                        'last_submitted_actuator_targets':self.targets,'uncommitted_control_records':self.controls,
                        'visual_captures':self.capture_count})
        try:write(self.run/'writer_close.json',receipt)
        except BaseException as error:
            if primary is None:primary=error
            else:primary.add_note('Writer-close receipt also failed: '+repr(error))
        if primary is not None:raise primary


def recorded_observation_path(run, index):
    if type(index) is not int or index < 0:
        raise ValueError('Invalid native observation index')
    return Path(run) / ('initial' if index == 0 else f'macro_obs_{index:06d}')


def wait_for_native_publication(recorder, index):
    from .observation_io import OrderedObservationWriter
    if type(recorder) is not DenseRecorder or type(recorder.writer) is not OrderedObservationWriter:
        raise RuntimeError('Existing sole native25 recorder/writer required')
    if type(index) is not int or index < 0:
        raise ValueError('Invalid native publication index')
    writer = recorder.writer
    deadline = time.perf_counter() + 120.0
    with writer.condition:
        writer.check()
        if index >= writer.submitted:
            raise ValueError('Native boundary has not been submitted')
        while writer.last_committed < index:
            writer.check()
            if time.perf_counter() >= deadline:
                writer.fail(TimeoutError('Native publication wait exceeded 120s; infrastructure interruption'))
                writer.check()
            writer.condition.wait(.05)
        # A publication completing concurrently with failure is not a success.
        writer.check()


def camera_profile(task):
    import carb
    settings=carb.settings.get_settings()
    n=task.native
    from .rtx_profile import observe as observe_rtx
    rtx_observation=observe_rtx(task,settings)
    cams={}
    for i in range(n.camera_manager.num_cams):
        name=n.camera_manager.camera_names[0][i]
        cams[name]={'intrinsic':scalar(n.camera_manager.get_camera_intrinsics(i,0)),
            'extrinsic':scalar(n.camera_manager.get_camera_extrinsics(i,0)),
            'shape':list(task.public_observation()['vision'][name]['color'].shape),
            'crop':None,'resize':None,'projection_validation_status':'unverified'}
        from pxr import Usd, UsdGeom
        import omni.usd
        camera_xform=n.camera_manager.cameras_xform[0][i]
        stage=omni.usd.get_context().get_stage()
        camera_root=stage.GetPrimAtPath(camera_xform.prim_path)
        actual_cameras=[]
        for prim in Usd.PrimRange(camera_root):
            if not prim.IsA(UsdGeom.Camera):continue
            camera=UsdGeom.Camera(prim)
            focal=float(camera.GetFocalLengthAttr().Get())
            horizontal=float(camera.GetHorizontalApertureAttr().Get())
            vertical=float(camera.GetVerticalApertureAttr().Get())
            height,width=cams[name]['shape'][:2]
            actual_cameras.append({'prim_path':str(prim.GetPath()),'focal_length':focal,'horizontal_aperture':horizontal,'vertical_aperture':vertical,
                'intrinsic_from_actual_usd':[[width*focal/horizontal,0,width/2],[0,height*focal/vertical,height/2],[0,0,1]],
                'camera_to_world_usd':np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T.tolist(),
                'camera_axes':'USD camera +X right, +Y up, -Z forward; pixel +Y down'})
        cams[name]['actual_usd_cameras']=actual_cameras
    return {'cameras':cams,'rtx_profile':rtx_observation,'renderer_settings':{k:settings.get(k) for k in ['/renderer/activeGpu','/renderer/multiGpu/enabled','/rtx/rendermode','/renderer/renderer']},
        'actual_gpu_uuid':os.environ['NVIDIA_VISIBLE_DEVICES'],'vk_icd':os.environ['VK_ICD_FILENAMES'],
        'calibration_status':'measured_intrinsics_extrinsics; reprojection validation still required'}
