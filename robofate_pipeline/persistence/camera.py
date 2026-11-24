"""Expose native camera metadata after t1, without capture, control or hidden state."""
import copy
import json
import math
from pathlib import Path

CAMERAS=('cam_head','cam_left_wrist','cam_right_wrist')
CONVENTIONS={
    'schema':'refate-native-camera-observation-v1',
    'transform':'camera_to_world_USD_column_vectors',
    'world_units':'metres', 'world_up':'+Z',
    'camera_axes':'+X right,+Y up,-Z forward',
    'image_axes':'u right,v down',
    'pixel_coordinates':'array_index_plus_0.5',
    'shape_order':'height,width,channels',
    'crop':None,'resize':None,'distortion':'native_pinhole_no_added_distortion',
    'projection':'q=R.T@(world_point-t); u=fx*q[0]/(-q[2])+cx; v=fy*(-q[1])/(-q[2])+cy',
    'ray':'origin=t; direction=solve(R.T,[(u-cx)/fx,-(v-cy)/fy,-1])',
    'profile':'pt_spatial4_collection_candidate_v1',
    'scope':'Current native camera K/T accepted by this profile calibration; no object geometry or task ground truth.'
}

def need(ok,message):
    if not ok:raise ValueError(message)

def matrix(value,n):
    if hasattr(value,'tolist'):value=value.tolist()
    need(isinstance(value,list) and len(value)==n,'Camera matrix shape differs')
    need(all(isinstance(row,list) and len(row)==n and all(type(x) in (int,float) and math.isfinite(x) for x in row) for row in value),'Camera matrix is not finite numeric')
    return copy.deepcopy(value)

def enrich(observation,cameras,native_index,*,budget):
    """Pure counterpart lets readback verify only declared metadata additions."""
    budget.index(native_index)
    need(set(cameras)==set(CAMERAS) and set(observation['vision'])==set(CAMERAS),'Camera name set differs')
    result=copy.deepcopy(observation)
    for name in CAMERAS:
        camera=cameras[name];image=result['vision'][name]['color'];shape=list(image.shape)
        need(shape==camera['shape']==[480,640,3] and camera['crop'] is None and camera['resize'] is None,'Calibrated original RGB shape/crop differs')
        k=matrix(camera['intrinsic'],3);t=matrix(camera['extrinsic'],4)
        need(k[0][0]>0 and k[1][1]>0 and k[2]==[0,0,1] and t[3]==[0,0,0,1],'Unsupported pinhole transform')
        item=result['vision'][name]
        for key,value in (('intrinsic_matrix',k),('extrinsic_matrix',t)):
            if key in item:need(matrix(item[key],len(value))==value,'Existing native camera field differs')
            item[key]=value
        if 'shape' in item:
            existing=item['shape'].tolist() if hasattr(item['shape'],'tolist') else item['shape']
            need(list(existing)==shape,'Existing native shape differs')
        item['shape']=shape
    info=result.setdefault('additional_info',{})
    need(isinstance(info,dict),'Invalid additional_info')
    convention={**copy.deepcopy(CONVENTIONS),'native_action_index':native_index}
    if 'camera_coordinates' in info:need(info['camera_coordinates']==convention,'Existing camera convention differs')
    info['camera_coordinates']=convention
    return result

def install(task,run,t1,*,budget):
    """Install once after exact raw replay/calibration checks, leaving the raw boundary intact."""
    need(type(t1) is int and t1==budget.t1 and task.control_steps==t1 and not task.done,'Not the qualified early boundary')
    budget.actual_task(task,t1)
    need(not hasattr(task,'_refate_public_camera_original'),'Camera observer already installed')
    run=Path(run);need(run.is_absolute() and run.resolve()==run,'Noncanonical actual run')
    recorded=json.loads((run/f'camera_{t1:06d}.json').read_text())['cameras']
    original=task.public_observation
    def public_observation():
        used=task.control_steps;budget.actual_task(task,used)
        observation=original();manager=task.native.camera_manager
        need(manager.num_cams==3,'Exactly three native cameras required')
        cameras={}
        for i in range(manager.num_cams):
            name=manager.camera_names[0][i]
            need(name in CAMERAS and name not in cameras,'Native camera identity differs')
            cameras[name]={'intrinsic':matrix(manager.get_camera_intrinsics(i,0),3),
                           'extrinsic':matrix(manager.get_camera_extrinsics(i,0),4),
                           'shape':list(observation['vision'][name]['color'].shape),'crop':None,'resize':None}
        need(task.control_steps==used,'Camera read advanced native state')
        return enrich(observation,cameras,used,budget=budget)
    initial=public_observation();expected=enrich(original(),recorded,t1,budget=budget)
    for name in CAMERAS:
        need(initial['vision'][name]['intrinsic_matrix']==expected['vision'][name]['intrinsic_matrix']
             and initial['vision'][name]['extrinsic_matrix']==expected['vision'][name]['extrinsic_matrix'],
             'Initial native camera metadata differs from exact restored snapshot')
    need(task.control_steps==t1 and not task.done,"Initial camera validation left the qualified boundary")
    task._refate_public_camera_original=original
    task.public_observation=public_observation
