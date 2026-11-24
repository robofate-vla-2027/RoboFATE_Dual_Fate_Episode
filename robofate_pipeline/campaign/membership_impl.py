"""Original exact mixed-label membership checks, with external manifest pins."""
from ..evidence import need
from . import family as families
SENTINEL = 'conflict_preserved_not_admitted'

def membership(split, layout_id, split_ref, raw):
    row = split['layouts'][str(layout_id)]
    gid = row['group_id']
    need(split['layout_to_group'][str(layout_id)] == gid, 'Different family mapping')
    group = split['families'][gid]
    if group['split'] == row['split']:
        return True
    task = split['task']
    need(task in MIXED_REFS and split_ref == MIXED_REFS[task], 'Mixed-label compatibility is manifest/version bound')
    need(group['split'] == SENTINEL and group['group_id'] == gid,
         'Unknown or sealed group is not legacy collection overlap')
    need(split['preserved_all_original_memberships'] is True
         and split['sealed_content_not_read'] is True, 'Original partition provenance required')
    partition = split['original_partition']
    train, dev, sealed = (set(partition[k]) for k in ('train', 'development', 'reserved_final'))
    need(not train & dev and not (train | dev) & sealed, 'Original source partitions overlap')
    members = group['layouts']
    ids = [m['layout_id'] for m in members]
    need(len(ids) == len(set(ids)) and layout_id in ids, 'Ambiguous family member identity')
    labels = set()
    for m in members:
        i = m['layout_id']
        need(type(i) is int and i in split['initial_source_sequence'] and i not in sealed,
             'Noncollection family member')
        expected = 'student_train' if i in train else 'student_dev' if i in dev else None
        r = split['layouts'][str(i)]
        need(expected is not None and m['split'] == r['split'] == expected
             and m['bank'] == r['native_bank'] == 0 and r['layout_id'] == i
             and r['group_id'] == split['layout_to_group'][str(i)] == gid,
             'Different collection membership')
        need(r['layout_ref'] == {'path': m['path'], 'sha256': m['sha256']},
             'Different family source reference')
        labels.add(expected)
    need(labels == {'student_train', 'student_dev'}, 'Sentinel is not mixed collection labels')
    need(gid == task + '/task_content_family/' + families.family(raw)['content_sha256'],
         'Actual layout belongs to a different normalized family')
    return True
