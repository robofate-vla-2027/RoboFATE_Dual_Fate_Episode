"""Original independent three-condition saved-data reader."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time
from . import content as _content_policy, decoder as archive_decoder
CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')

class DataIssue(ValueError):
    def __init__(self, condition, message, status='rejected'):
        super().__init__(message)
        self.condition, self.status = condition, status


def need(ok, condition, message):
    if not ok:
        raise DataIssue(condition, message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def sig(p):
    s = Path(p).stat()
    return [s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


class Files:
    def __init__(self, root):
        self.root, self.refs, self.signatures = Path(root), {}, {}

    def raw(self, value):
        p = Path(value)
        need(p.is_absolute() and p.resolve() == p and p.is_relative_to(self.root), '1.1', 'Data path escapes its bound root: '+str(p))
        before = sig(p)
        need(stat.S_ISREG(before[2]), '1.1', 'Not a regular saved file: '+str(p))
        with p.open('rb') as f:
            raw = f.read()
        need(before == sig(p) and len(raw) == before[3], '1.1', 'File changed during read: '+str(p))
        need(str(p) not in self.signatures or self.signatures[str(p)] == before, '1.1', 'File changed between reads: '+str(p))
        self.signatures[str(p)] = before
        self.refs[str(p)] = {'path':str(p), 'sha256':sha(raw)}
        return raw

    def doc(self, p):
        return json.loads(self.raw(p))

    def ref(self, ref):
        raw = self.raw(ref['path'])
        need(sha(raw) == ref['sha256'], '1.2', 'Bound reference content differs: '+ref['path'])
        return json.loads(raw)

    def reference(self, p):
        self.raw(p)
        return self.refs[str(p)]

    def rows(self, p):
        raw = self.raw(p)
        need(raw.endswith(b'\n'), '1.1', 'Incomplete final JSONL row: '+str(p))
        return [json.loads(x) for x in raw.splitlines()]

    def stable(self):
        for p, before in self.signatures.items():
            need(sig(p) == before, '1.1', 'Required saved data changed during review: '+p)


def finite_tree(value):
    if isinstance(value, float):
        need(math.isfinite(value), '1.3', 'Nonfinite recorded numeric value')
    elif isinstance(value, dict):
        for v in value.values(): finite_tree(v)
    elif isinstance(value, (list, tuple)):
        for v in value: finite_tree(v)


def rows_check(ctx, result, actions, dense, observations, prefix, source):
    n, t, budget = result['native_steps'], ctx['t1'], ctx['native_limit']
    run = Path(ctx['run_dir'])
    folder = lambda i: str(run/('initial' if i == 0 else f'macro_obs_{i:06d}'))
    need(type(n) is int and 0 < t < n <= budget and ctx['remaining_steps'] == budget-t, '1.2', 'Original/remaining budget differs')
    need(result['native_success'] is True and result['terminated'] is True and not result['truncated'], '1.3', 'No genuine teacher native success')
    need(len(actions)==n and len(dense)==10*n and len(observations)==n+1, '1.3', 'Action/control/observation range incomplete')
    need(len(prefix)==t, '1.2', 'Incomplete saved source prefix')
    for i, row in enumerate(actions):
        need(row['index']==i and row['native_count_after']==i+1 and row['actual_control_steps']==10*(i+1), '1.3', 'Native action index/control count differs')
        need(all(type(row[k]) is bool for k in ('native_success','terminated','truncated')), '1.3', 'Unknown native flags')
        need((row['terminated'] or row['truncated']) == (i==n-1), '1.3', 'Action after terminal or missing terminal')
        finite_tree(row['action'])
        if i<t:
            need(row['role']=='recorded_DM0.5_prefix_replay' and row['original_native_event']==str(source/'native_actions.jsonl') and row['action']==prefix[i]['action'], '1.2', 'Prefix action/source mismatch')
            need(not any(prefix[i][k] for k in ('native_success','terminated','truncated')), '1.2', 'Source prefix already terminal')
        else:
            need(row['role']=='EmbodiedEvo_teacher', '1.2', 'Actual suffix is not a teacher branch')
    need(all(actions[-1][k]==result[k] for k in ('native_success','terminated','truncated')), '1.3', 'Last action and native result disagree')
    last_sim = last_wall = None
    for idx,row in enumerate(dense,1):
        macro=(idx-1)//10
        need(row['index']==idx and row['macro_index']==macro and row['native_action_count']==macro+1, '1.3', 'Actual control index gap/reset')
        need(row['obs_before']==folder(macro) and row['obs_after']==folder(macro+1) and row['visual_interval']==[macro,macro+1], '1.3', 'Control/observation association differs')
        need(bool(row['executed_actuator_targets']) and bool(row['robot_feedback']), '1.3', 'Actual command or feedback missing')
        finite_tree(row['executed_actuator_targets']); finite_tree(row['robot_feedback'])
        b,a=row['simulation_time_before'],row['simulation_time_after'];wb,wa=row['wall_before_ns'],row['wall_after_ns']
        need(row['physical_dt_s']==.004 and abs(a-b-.004)<1e-8 and (last_sim is None or abs(b-last_sim)<1e-8), '1.3', 'Physics time gap/reset')
        need(wb<=wa and (last_wall is None or wb>=last_wall), '1.3', 'Actual wall time sequence differs')
        need(row['fresh_rgb_capture'] is (idx%10==0), '1.3', 'Native RGB/control boundary differs')
        last_sim,last_wall=a,wa
    for i,row in enumerate(observations):
        expected_time=dense[0]['simulation_time_before'] if i==0 else dense[10*i-1]['simulation_time_after']
        need(row['index']==i and row['control_index']==10*i and row['observation_dir']==folder(i) and row['terminal'] is (i==n), '1.3', 'Observation index/terminal differs')
        need(abs(row['simulation_time']-expected_time)<1e-8, '1.3', 'Observation/physical time mismatch')


def reuse_failure(e, summary_ref, source, sid):
    summary=e.ref(summary_ref)
    need(summary['source_id']==sid and summary['run']==str(source), '1.2', 'Failure audit bound to another source')
    need(summary['file_recording_readback_passed'] is True and summary['native_result']['native_success'] is False, '1.1', 'No valid saved DM-failure readback')
    result=e.doc(source/'result.json')
    need(result==summary['native_result'] and (result['terminated'] or result['truncated']), '1.2', 'Original DM result changed/not terminal')
    rb=e.ref(summary['files']['readback.json'])
    need(rb['file_recording_readback_passed'] is True and rb['source_id']==sid and rb['result']==result, '1.1', 'Failure readback does not cover actual source')
    inventory=e.ref(summary['files']['source_inventory_before.json'])
    reused=0
    for name,old in inventory.items():
        p=Path(name); required=(p.suffix in ('.npz','.png') or p.name in ('native_actions.jsonl','native_observations.jsonl','control_steps.jsonl','history_updates.jsonl','observation.json','image_stats.json') or p.name.startswith(('physics_','task_auxiliary_','camera_')))
        if required and stat.S_ISREG(old[2]):
            actual=source/p
            if sig(actual)!=old:
                raise DataIssue('1.1','Saved DM file changed since valid audit, needs targeted comparison: '+str(actual),'pending_review')
            e.signatures[str(actual)]=old;reused+=1
    return summary,rb,reused


def restored_observation(observation, t, stats0, stats1):
    need(observation["t1"] == t and observation["instruction_exact"] is True and
         observation["reference_state"] == observation["restored_state"],
         "1.2", "Restored task/robot observation differs")
    return all(stats0[c]["sha256"] == stats1[c]["sha256"] for c in CAMERAS)


def review(e, ctx_ref, summary_ref, *, data_root, outline_sha256, directive_sha256):
    S = Path(data_root)
    OUTLINE_SHA, DIRECTIVE_SHA = outline_sha256, directive_sha256
    start=time.monotonic();ctx=e.ref(ctx_ref);run=Path(ctx['run_dir']);ident=ctx['identity'];sid=ident['source_id'];t=ctx['t1'];budget=ctx['native_limit']
    source=Path(e.doc(run/'restore_request.json')['reference_run'])
    need(run.parent==source.parent==S/'runs' and run!=source, '1.2', 'Invalid paired branch paths')
    source_id=e.doc(source/'source_identity.json');teacher_id=e.doc(run/'source_identity.json')
    _paired_source_identity(source_id, teacher_id, sid, ident['task'])
    summary,rb,reused=reuse_failure(e,summary_ref,source,sid)
    request=e.doc(run/'restore_request.json');boundary=e.doc(run/'restore_boundary.json');comparison=e.doc(run/'restore_state_comparison.json');observation=e.doc(run/'restore_observation_comparison.json')
    need(boundary['native_count']==t and boundary['remaining_steps']==budget-t and comparison['context_budget']=={'total':budget,'t1':t,'remaining':budget-t}, '1.2', 'Restored boundary budget mismatch')
    need(comparison['all_compared_fields_within_candidate_rules'] is True and comparison.get('no_new_missing_fields') is True, '1.2', 'Actual t1 state comparison is not matched')
    for ref in comparison['input_files']:
        e.ref(ref)
    stats0=e.doc(source/f'macro_obs_{t:06d}/image_stats.json');stats1=e.doc(run/f'macro_obs_{t:06d}/image_stats.json')
    t1_rgb_exact_match = restored_observation(observation, t, stats0, stats1)
    result=e.doc(run/'result.json');actions=e.rows(run/'native_actions.jsonl');dense=e.rows(run/'dense/control_steps.jsonl');obs=e.rows(run/'native_observations.jsonl')
    with (source/'native_actions.jsonl').open('rb') as stream:
        prefix=[json.loads(next(stream)) for _ in range(t)]
    rows_check(ctx,result,actions,dense,obs,prefix,source)
    ipc=run/'teacher_ipc';bootstrap=e.doc(ipc/'observation_000000.json')
    need(bootstrap['source_id']==sid and bootstrap['native_used']==t and bootstrap['remaining_steps']==budget-t, '1.2', 'Teacher bootstrap does not match verified t1')
    history=e.doc(ipc/'artifacts/causal_history.json')
    need(len(history)==t+1 and all(h['native_step']==i for i,h in enumerate(history)), '1.2', 'Causal history range does not end at t1')
    for i in range(1,t+1):
        need(history[i]['executed_action']==prefix[i-1]['action'], '1.2', 'Causal history action differs from original DM')
    for seq in range(1,result['native_steps']-t+1):
        rsp=e.doc(ipc/f'response_{seq:06d}.json');b=e.doc(ipc/f'observation_{seq:06d}.json')
        request=e.doc(ipc/f'request_{seq:06d}.json');prior=e.reference(ipc/f'observation_{seq-1:06d}.json')
        need(rsp['observation_ref']==request['observation_ref']=={'path':f'observation_{seq-1:06d}.json','sha256':prior['sha256']} and rsp['request_sha256']==e.reference(ipc/f'request_{seq:06d}.json')['sha256'], '1.3', 'Teacher action/actual observation link differs')
        need(rsp['sequence']==seq and rsp['refate']['source_id']==sid and rsp['refate']['native_before']==t+seq-1 and rsp['action']==actions[t+seq-1]['action'], '1.2', 'Actual teacher proposal/action/source mismatch')
        need(b['native_used']==t+seq and b['remaining_steps']==budget-t-seq and b['source_id']==sid, '1.2', 'Teacher budget/source reset')
    for seq in range(result['native_steps']-t+1):
        b=e.doc(ipc/f'observation_{seq:06d}.json');mapping=e.doc(ipc/f'artifacts/public_{seq:06d}/public_camera_mapping.json')
        need(mapping['native_used']==t+seq and b['observation']==mapping['public_descriptor'], '1.3', 'Teacher public observation linked to another boundary')
        for role in ('raw_recorded_descriptor','public_descriptor'):
            desc=mapping[role]
            refs=[desc['json'],desc['arrays']]+list(desc['pngs'].values())
            for item in refs:
                file=ipc/item['path'];need(e.reference(file)['sha256']==item['sha256'], '1.1', 'Teacher saved observation reference differs: '+str(file))
    public=archive_decoder;decoded=0;constant_rgb_diagnostics=[]
    for i in range(result['native_steps']+1):
        folder=run/('initial' if i==0 else f'macro_obs_{i:06d}')
        def ref(n):return {'path':n,'sha256':e.reference(folder/n)['sha256']}
        desc={'format':'refate-observation-native25-v1','json':ref('observation.json'),'arrays':ref('arrays.npz'),'pngs':{c+'.png':ref(c+'.png') for c in CAMERAS}}
        view=public.decode_archive(desc,lambda n:e.raw(folder/n),archive_folder=str(folder));stats=e.doc(folder/'image_stats.json')
        need(bool(view.get('state')) and bool(view.get('instruction')), '1.3', 'Saved observation state/instruction missing')
        for c in CAMERAS:
            array=view['vision'][c]['color'];need(list(array.shape)==[480,640,3] and str(array.dtype)=='uint8' and sha(array.tobytes())==stats[c]['sha256'], '1.3', 'Decoded camera does not match saved boundary: '+str(folder)+'/'+c)
            diagnostic=_content_policy.validate_rgb(array,stats[c])
            if diagnostic['constant_pixel_values']:
                constant_rgb_diagnostics.append(dict(native_index=i,camera=c,**diagnostic))
        physics=e.doc(run/f'physics_{i:06d}.json');aux=e.doc(run/f'task_auxiliary_{i:06d}.json');camera=e.doc(run/f'camera_{i:06d}.json')
        need(bool(physics['robots']) and bool(physics['objects']) and bool(physics['control']) and bool(aux), '1.3', 'Recorded physical/robot/control state missing')
        pt=physics['task'];need(pt['native_count']==[i] and pt['step_limit']==budget, '1.2', 'Recorded physical budget/count reset')
        if i==result['native_steps']:
            need(pt['end_flag']==[True] and pt['success_flag']==[True], '1.3', 'Physical terminal not native success')
        finite_tree(physics);finite_tree(aux);decoded+=1
    diagnostics={}
    for name in ['teacher_finish.json','writer_close.json','container_exit.json','failure.json']:
        if (run/name).is_file():
            v=e.doc(run/name);diagnostics[name]={'reference':e.refs[str(run/name)],'status':v.get('status'),'exit_code':v.get('state',{}).get('ExitCode'),'failed':v.get('failed')}
    diagnostics['bridge_status']='not_assessed_by_data_reader'
    diagnostics['t1_rgb_exact_match']=t1_rgb_exact_match
    diagnostics['observation_content_policy']=_content_policy.policy_reference()
    diagnostics['constant_rgb_frames']=constant_rgb_diagnostics
    e.stable()
    return {'schema':'refate.minimum_data_readback.v1','data_status':'qualified','conditions':{c:{'status':'passed'} for c in ['1.1','1.2','1.3']},'source_id':sid,'task':ident['task'],'run':str(run),'failure_run':str(source),'context_ref':ctx_ref,'failure_summary_ref':summary_ref,'t1':t,'native_limit':budget,'teacher_steps':result['native_steps'],'teacher_actions':result['native_steps']-t,'source_identity':source_id,'teacher':{'name':'EmbodiedEvo','identity':'public-recipe primitive baseline'},'diagnostics':diagnostics,'decoded_teacher_observations':decoded,'decoded_teacher_RGB_images':3*decoded,'original_DM_RGB_redecodes':0,'unchanged_DM_files_reused':reused,'evidence_refs':list(e.refs.values()),'stable_files':e.signatures,'wall_seconds':time.monotonic()-start,'outline_sha256':OUTLINE_SHA,'directive_sha256':DIRECTIVE_SHA,'historical_errors_preserved':True,'counted_in_quota':False,'reviewed_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}


def exclusive(path,value):
    raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
    return {'path':str(path),'sha256':sha(raw)}

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

def _paired_source_identity(source, teacher, expected_source, expected_task):
    # Execution provenance is not immutable source identity. Both original
    # documents remain independently referenced by Files; no field is edited.
    try:
        _bound_source_identity(source, expected_source, expected_task)
        _bound_source_identity(teacher, expected_source, expected_task)
    except ValueError as error:
        raise DataIssue('1.2', str(error)) from error
