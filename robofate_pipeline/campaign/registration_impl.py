"""Original contiguous append/ancestry rules with external frozen base."""
import copy
import re
from pathlib import Path
from ..evidence import need
ROW_KEYS = {'layout_id','native_bank','layout_ref','initialization_content_sha256','initialization_distinct_from_preceding','group_id','split'}

def merge(prior, prior_ref, row, at, helper_ref):
    task = prior['task']
    p = profile(task)
    need(prior['owner_chat']==OWNER and prior['campaign_id']==CAMPAIGN, 'Wrong campaign/owner')
    seq, i = prior['initial_source_sequence'], row['layout_id']
    need(type(seq) is list and all(type(v) is int for v in seq) and seq==list(range(len(seq))), 'Noncontiguous sequence')
    need(type(i) is int and i==len(seq) and p['initial_count']<=i<p['reserved_final'][0], 'Only next nonsealed original source')
    need(prior['original_partition']['reserved_final']==p['reserved_final'], 'Frozen evaluation boundary changed')
    need(set(row)==ROW_KEYS and type(row['native_bank']) is int and row['native_bank']==0 and row['split'] in ('student_train','student_dev'), 'Wrong source row')
    need(type(row['layout_ref']) is dict and set(row['layout_ref'])=={'path','sha256'}, 'Exact layout reference required')
    need(row['layout_ref']['path']==str(NATIVE/f'Assets/Eval_Layout/RoboDojo/arx_x5/0/{task}_{i}.json'), 'Wrong original layout path')
    need(all(type(v) is str and re.fullmatch('[0-9a-f]{64}',v) for v in [row['layout_ref']['sha256'],row['initialization_content_sha256']]), 'Malformed hashes')
    need(row['initialization_distinct_from_preceding'] is True and row['initialization_content_sha256'] not in {v['initialization_content_sha256'] for v in prior['layouts'].values()}, 'Duplicate initialization')
    gid=row['group_id']
    need(type(gid) is str and re.fullmatch(re.escape(task+'/task_content_family/')+'[0-9a-f]{64}',gid), 'Wrong family identity')
    family=prior['families'].get(gid)
    need(row['split']==(family['split'] if family else 'student_train'), 'Family split changed')
    out=copy.deepcopy(prior)
    out['initial_source_sequence']=seq+[i]
    out['layouts'][str(i)]=copy.deepcopy(row)
    out['layout_to_group'][str(i)]=gid
    if not family:
        out['families'][gid]=dict(group_id=gid,split='student_train',layouts=[])
    out['families'][gid]['layouts'].append(dict(layout_id=i,bank=0,split=row['split'],**row['layout_ref']))
    out['collection_append']=dict(schema='refate.configured_source_registration.v1',at=at,prior_split=copy.deepcopy(prior_ref),added_layout=i,prior_sequence=copy.deepcopy(seq),source_partition=row['split'],new_family_registered=family is None,original_memberships_modified=False,independent_development_claimed=False,independent_evaluation_claimed=False,sealed_content_read=False,outcome_used=False,metadata_helper=copy.deepcopy(helper_ref))
    return out


def validate_append(prior, prior_ref, current):
    record=current['collection_append']
    expected=merge(prior,prior_ref,current['layouts'][str(record['added_layout'])],record['at'],record['metadata_helper'])
    need(current==expected, 'Historical or unrelated metadata changed')
    return current


def lineage(pin, task, seen=None):
    p=profile(task)
    seen=set() if seen is None else seen
    key=(pin['path'],pin['sha256'])
    need(key not in seen and len(seen)<=p['reserved_final'][0]-p['initial_count'], 'Cyclic or excessive registration')
    seen.add(key)
    current=document(pin)
    need(current['task']==task and current['owner_chat']==OWNER and current['campaign_id']==CAMPAIGN, 'Wrong lineage identity')
    if pin==p['base']:
        need(current['initial_source_sequence']==list(range(p['initial_count'])) and current['original_partition']['reserved_final']==p['reserved_final'], 'Initial scope mismatch')
        return current
    record=current['collection_append']
    need(pin['path']==str(p['root']/f"SPLIT_APPEND_L{record['added_layout']}_001.json"), 'Wrong append path')
    need(record['metadata_helper']==ref(__file__), 'Registration helper changed')
    prior=lineage(record['prior_split'],task,seen)
    return validate_append(prior,record['prior_split'],current)
