"""Original source material and pinned restore preparation, externally bound."""
import copy
from pathlib import Path
PROFILE_ID = 'pt_spatial4_collection_candidate_v1'

def shape(value):
    result=[]
    while isinstance(value,list):
        result.append(len(value))
        if not value:break
        value=value[0]
    return result


def source_material(source_context_ref, source_manifest_ref, source_readback_ref, *, t1):
    context=document(source_context_ref);manifest=document(source_manifest_ref);summary=document(source_readback_ref)
    task=context['task'];total=_bound_budget(task,context['native_total_steps'],t1)
    need(context['data_kind']==_policy['data_kind'],'Source data kind differs')
    sid=context['source_id'];source=Path(context['run_dir'])
    need(source==stage.SHARED/'runs'/context['run_id'],'Wrong original source root')
    need(sid not in _policy['excluded_source_ids'], 'Excluded source cannot be replayed')
    _source_admission(context, manifest, summary)
    need(type(t1) is int and 0<t1<total,'Source-native cutoff required')
    need(manifest['source_id']==summary['source_id']==sid
         and manifest['run_id']==context['run_id'] and manifest['native_success'] is False
         and summary['file_recording_readback_passed'] is True,'Completed original DM failure required')
    result=document(ref(source/'result.json'))
    need(summary['native_result']==result and result['native_success'] is False
         and result['mode']=='independent_baseline' and result['native_steps']>t1
         and result['native_step_limit']==total and (result['terminated'] or result['truncated']),
         'Unknown or non-independent native outcome')
    preparation=context['preparation']
    need(result['rtx_profile']==preparation['renderer_profile']
         ==preparation['prospective_teacher_renderer_profile']==PROFILE_ID,
         'Original and teacher renderer profiles differ')
    initial=document(ref(source/'initial_physics_state.json'));boundary=document(ref(source/f'physics_{t1:06d}.json'))
    need(initial['task']['native_count']==[0] and boundary['task']['native_count']==[t1]
         and boundary['task']['end_flag']==[False],'Missing original nonterminal boundary')
    template,template_ref=_rule_document('profile_template')
    invariants,invariants_ref=_rule_document('invariants')
    schema=copy.deepcopy(template['schema_contract'])
    need(set(initial)==set(schema['top_keys']) and set(initial['robots'])==set(schema['robots']),
         'Different shared physical representation or robot set')
    for robot,layout in schema['robots'].items():
        actual=initial['robots'][robot]
        for key in ('joint_names','arm_joint_indices','gripper_joint_indices'):
            need(actual[key]==layout[key],'Original joint meaning changed: '+key)
        need(shape(actual['joint_pos'])==layout['joint_array_shape']
             and shape(actual['root_state_w'])==layout['root_state_shape'],'Original robot array layout changed')
    object_fields=['class','prim_path','world_pose','linear_velocity','angular_velocity','sleeping']
    schema['objects']={name:{'class':value['class'],'prim_path':value['prim_path'],
        'fields':list(dict.fromkeys(object_fields+list(value)))} for name,value in initial['objects'].items()}
    for name in ('control','task','reward','rng'):
        schema[name+'_keys']=list(initial[name])
    schema['rng_top_lengths']={name:len(value) for name,value in initial['rng'].items()}
    need(set(initial['objects'])==set(boundary['objects'])
         and initial['unexposed']==boundary['unexposed'],'Source structural/missing-state scope changed before t1')
    identity=document(ref(source/'source_identity.json'))
    need(identity==context['preparation']['source'] and identity['source_id']==sid,'Original source metadata differs')
    profile=dict(schema='refate-direct-teacher-restore-profile-v1',profile_id=sid+'_native25_t'+str(t1),
        status='qualified_for_source_prefix_restore',task=task,source_id=sid,
        native_step_limit=total,t1=t1,remaining_native_steps=total-t1,reference_run=str(source),
        protocol='dual-fate-direct-teacher-v1',restore_method='complete_recorded_prefix_replay',
        renderer_profile_id=PROFILE_ID,schema_contract=schema,
        thresholds=copy.deepcopy(template['thresholds']),joint_units=copy.deepcopy(template['joint_units']),
        required_discrete_exact=copy.deepcopy(invariants['required_discrete_exact']),
        coverage_requirements=copy.deepcopy(invariants['coverage_requirements']),
        threshold_basis={'native_step_limit':total,'observed_native_dt_s':.004},
        threshold_provenance={'original_profile':template_ref,'original_invariants':invariants_ref},
        DM_continuation_required=False,DM_private_inference_state_required=False,
        actual_teacher_boundary_verified=False,serialized_coverage_complete=False,
        qualified=False,counted_in_quota=False,source_readback_ref=source_readback_ref)
    return dict(source_context=source_context_ref,source_manifest=source_manifest_ref,
        source_readback=source_readback_ref,source=str(source),source_id=sid,task=task,
        source_owner=identity['source_owner'],native_limit=total,t1=t1,
        profile=profile,source_identity=identity,initial_ref=ref(source/'initial_physics_state.json'),
        boundary_ref=ref(source/f'physics_{t1:06d}.json'),
        raw_source_gaps={'physics_boundary0':initial['unexposed'],'physics_boundary'+str(t1):boundary['unexposed']},
        source_preparation=context['preparation'],data_kind=_policy['data_kind'],scope='source_metadata_not_restore_admission')


def prepare_context(material, *, run, owner, core_modules, metadata_root=None, enabled=False):
    """Save task/source metadata; never claims an actual restored boundary."""
    need(type(enabled) is bool,'Explicit execution-enabled value required')
    need(not enabled or _policy['data_kind']=='actual', 'Synthetic preparation cannot enable execution')
    run=Path(run);source=Path(material['source']);root=Path(metadata_root or run)
    need(run.parent==source.parent==stage.SHARED/'runs' and run!=source,'Distinct same-node teacher run required')
    need(owner['source_id']==material['source_id'] and owner['task']==material['task']
         and owner['chat']==_policy['partition'] and owner['active_chat_id']==stage.OWNER
         and owner['owner']==material['source_owner'],'Source/execution ownership differs')
    root.mkdir(parents=True,exist_ok=True)
    def write(name,value):
        path=root/name
        stage.write_new(path,value)
        return ref(path)
    status='qualified_for_source_prefix_restore'
    scope=dict(schema='refate-direct-teacher-state-scope-v1',status=status,
        source_id=material['source_id'],t1=material['t1'],reference_run=str(source),
        joint_units=material['profile']['joint_units'],raw_source_gaps=material['raw_source_gaps'],
        private_cache_exported=False,DM_continuation_required=False,current_teacher_boundary_verified=False,
        counted_in_quota=False)
    scope_ref=write('state_scope.json',scope)
    profile=dict(material['profile'],state_scope_ref=scope_ref)
    profile_ref=write('restore_profile.json',profile)
    comparator_path=Path(__file__).with_name('comparator_impl.py')
    config=dict(task=material['task'],source_id=material['source_id'],reference_run=str(source),
        native_step_limit=material['native_limit'],t1=material['t1'],
        remaining_native_steps=material['native_limit']-material['t1'],rtx_profile_id=PROFILE_ID,
        profile_path=profile_ref['path'],profile_sha256=profile_ref['sha256'],
        comparator_sha256=ref(comparator_path)['sha256'],source_readback_ref=material['source_readback'],
        state_scope_ref=scope_ref)
    restore_ref=write('restore_request.json',config)
    evo_refs={'source_admission':material['source_manifest'],'source_readback_ref':material['source_readback'],
        'native25_runtime_refs':{name+'.py':ref(module.__file__) for name,module in core_modules.items()
                                 if name in ('collect_source','observation_io')}}
    native=dict(schema='refate-evo-native25-public-config-v1',enabled=enabled,identity=owner,
        run_dir=str(run),native_step_limit=material['native_limit'],t1=material['t1'],
        restore_config=restore_ref,profile_ref=profile_ref,rtx_profile=PROFILE_ID,**evo_refs)
    native_ref=write('native_restore_config.json',native)
    refs={'profile':profile_ref,'source_identity':ref(source/'source_identity.json'),
          'source_ready':ref(source/'ready.json'),'native_config':native_ref,
          'restore_config':restore_ref,'state_scope':scope_ref}
    return {'refs':refs,'owner':copy.deepcopy(owner),'evo_refs':evo_refs,'config':config,
            'profile':profile,'native_config':native,'run':str(run),'metadata_root':str(root),
            'actual_restore_verified':False}


def bind_actual(api, preparation, task):
    snapshots={role:{'path':reference['path'],'raw':runtime.read_pin(reference)}
               for role,reference in preparation['refs'].items()}
    return api.context.bind_context(snapshots,expected_refs=preparation['refs'],
        expected_owner=preparation['owner'],expected_evo_refs=preparation['evo_refs'],
        native=api.context.native_snapshot(task))
