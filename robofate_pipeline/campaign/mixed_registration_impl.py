"""Original pinned mixed-family append algorithm; external task/base/partition."""
from copy import deepcopy
import re
from ..evidence import need
APPEND_SCHEMA = 'refate.collection_source_append.v1'
ROW_KEYS = {'layout_id','native_bank','layout_ref','initialization_content_sha256','initialization_distinct_from_preceding','group_id','split'}

def merge(prior, prior_ref, row):
    need(prior['task'] == TASK and prior['campaign_id'] == metadata.sequence.CAMPAIGN
         and prior['owner_chat'] == metadata.OWNER, 'Original task/campaign/owner required')
    seq = prior['initial_source_sequence']
    i = row['layout_id']
    need(seq == list(range(len(seq))) and INITIAL_COUNT <= len(seq) < SEALED[0] and type(i) is int
         and i == len(seq), 'Only the next contiguous unused source may be appended')
    need(set(row) == ROW_KEYS and row['native_bank'] == 0 and row['split'] == 'student_train'
         and i not in prior['original_partition']['reserved_final'], 'Nonsealed collection row required')
    expected_path = metadata.NATIVE/f'Assets/Eval_Layout/RoboDojo/arx_x5/0/{TASK}_{i}.json'
    need(row['layout_ref']['path'] == str(expected_path)
         and set(row['layout_ref']) == {'path','sha256'}, 'Different source path')
    need(all(type(v) is str and re.fullmatch('[0-9a-f]{64}', v) for v in
             [row['layout_ref']['sha256'], row['initialization_content_sha256']]), 'Invalid source hashes')
    need(row['initialization_distinct_from_preceding'] is True
         and row['initialization_content_sha256'] not in
             {v['initialization_content_sha256'] for v in prior['layouts'].values()},
         'Duplicate initialization cannot become a new source')
    gid = row['group_id']
    need(gid in prior['families'] and prior['families'][gid]['split'] == previous.SENTINEL,
         'Only the existing pinned mixed collection family; new family requires separate registration')
    out = deepcopy(prior)
    out['initial_source_sequence'] = seq + [i]
    out['layouts'][str(i)] = deepcopy(row)
    out['layout_to_group'][str(i)] = gid
    out['families'][gid]['layouts'].append(dict(layout_id=i, bank=0, split=row['split'], **row['layout_ref']))
    out['collection_append'] = dict(schema=APPEND_SCHEMA, prior_split=deepcopy(prior_ref),
        added_layout=i, prior_sequence=deepcopy(seq), source_partition='student_train',
        original_memberships_modified=False, independent_development_claimed=False,
        independent_evaluation_claimed=False, sealed_content_read=False, outcome_used=False)
    return out


def lineage(ref, seen=None):
    seen = set() if seen is None else seen
    key = (ref['path'], ref['sha256'])
    need(key not in seen and len(seen) <= SEALED[0] - INITIAL_COUNT, 'Cyclic or excessive append chain')
    seen.add(key)
    value = document(ref)
    if ref == BASE_REF:
        need(value['initial_source_sequence'] == list(range(INITIAL_COUNT))
             and value['original_partition']['reserved_final'] == SEALED, 'Pinned initial scope differs')
        return value
    record = value['collection_append']
    i = record['added_layout']
    need(type(i) is int and ref['path'] == str(path_for(i)), 'Versioned append path required')
    prior = lineage(record['prior_split'], seen)
    expected = merge(prior, record['prior_split'], value['layouts'][str(i)])
    need(metadata.original.same(value, expected), 'Historical fields, labels or append evidence changed')
    return value
