"""Original positive transaction with deployed three-condition/counter edits."""
from copy import deepcopy
from collections import Counter
from datetime import datetime
import re
from ..evidence import need
TEACHER = 'EmbodiedEvo public-recipe primitive baseline'
REF_KEYS = {'publication','manifest','acceptance','context','source_report','teacher_report','split_manifest'}

def positive(row):
    return row.get('counted_in_quota') is True and row.get('qualification_status') == 'qualified'


def negative(row):
    return row.get('counted_in_negative_quota') is True and row.get('negative_qualification_status') == 'qualified'


def totals(target, pos, neg):
    pq = target.get('positive_quota', target.get('quota'))
    nq = target['negative_quota']
    need(type(pq) is int and type(nq) is int and 0 <= pos <= pq and 0 <= neg <= nq, 'quota overflow')
    target.update(qualified_count=pos, negative_count=neg, remaining_count=pq-pos,
                  negative_remaining_count=nq-neg, core_count=pos+neg, core_remaining_count=pq+nq-pos-neg)


def apply_commit(ledger, request, *, now, lock_inode, synthetic=False):
    need(lock_inode == LOCK_INODE, 'original stable lock inode required')
    need(type(now) is str and datetime.fromisoformat(now).tzinfo is not None, 'aware commit time required')
    need(request.get('schema') == 'refate.positive.ledger_commit_input.v1'
         and request.get('data_kind') == ('synthetic' if synthetic else 'actual'), 'explicit verified commit input required')
    need(ledger.get('campaign_id') == request.get('campaign_id') == CAMPAIGN, 'campaign mismatch')
    need(request.get('owner') == OWNER and request.get('source_owner') == SOURCE_OWNER
         and request.get('host') == HOST and request.get('source_id') == SOURCE and request.get('task') == TASK,
         'bound source/execution scope required')
    run = request.get('run_id')
    need(type(run) is str and _run_name(run), 'Bound owned run required')
    need(request.get('teacher') == TEACHER and request.get('split') in ('student_train', 'student_dev')
         and type(request.get('group_id')) is str and bool(request['group_id']), 'reviewed teacher/split/group required')
    need(request.get('native_dm_success') is False and request.get('native_teacher_success') is True
         and request.get('conditions') == {'1.1':True,'1.2':True,'1.3':True},
         'all separate scientific/data/resource gates required')
    refs = request.get('refs')
    need(type(refs) is dict and set(refs) == REF_KEYS, 'complete explicit evidence bindings required')
    for ref in refs.values():
        need(type(ref) is dict and set(ref) == {'path', 'sha256'} and type(ref['path']) is str
             and _evidence_path(ref['path'])
             and '/..' not in ref['path'] and type(ref['sha256']) is str
             and re.fullmatch('[0-9a-f]{64}', ref['sha256']), 'typed scoped evidence reference required')
    node = ledger['nodes'][HOST]
    need(node['active_chat_id'] == OWNER and node.get('user_stop') is False, 'execution owner conflict or user pause')
    task = ledger['tasks'][TASK]
    need(task['assigned_host'] == HOST and task['active_chat_id'] == OWNER, 'fixed task owner conflict')
    source = ledger['source_index'][SOURCE]
    need(source['owner_chat'] == SOURCE_OWNER and source['host'] == HOST and source['task'] == TASK,
         'historical source lineage conflict')
    if positive(source):
        need(source.get('positive_publication') == refs['publication'] and source.get('core_episode_id') == run
             and source.get('positive_commit_bindings') == refs and source.get('group_id') == request['group_id'],
             'source already counted with a different publication')
        return deepcopy(ledger), dict(idempotent_reuse=True, positive_delta=0, negative_delta=0)
    need(not negative(source), 'an independent success cannot become a takeover positive')
    for key, row in ledger['source_index'].items():
        need(key == SOURCE or row.get('source_id') != SOURCE or not (positive(row) or negative(row)), 'duplicate source identity')
        if row.get('group_id') == request['group_id']:
            need(row.get('split') in (None, request['split']) or _collection_split_compat(row), 'source group crossed split')
        need(key == SOURCE or row.get('core_episode_id') != run, 'run already belongs to another source')
    reservation = ledger['reservations'][SOURCE]
    need(reservation.get('quota_reserved') is True and reservation.get('negative_quota_reserved') is False
         and reservation.get('current_attempt_run_id') == run and reservation.get('active_attempt_owner') == OWNER
         and reservation.get('execution_owner') == OWNER and reservation.get('source_owner') == SOURCE_OWNER
         and reservation.get('current_attempt_context') == refs['context'] and reservation.get('split') == request['split'],
         'actual current positive reservation/context required')
    before = [r for r in ledger['source_index'].values() if r.get('task') == TASK]
    need(sum(positive(r) for r in before) < task['positive_quota'], 'positive quota is full')
    need(reservation.get('split_manifest') == refs['split_manifest'], 'frozen source split changed')
    out = deepcopy(ledger)
    target = out['source_index'][SOURCE]
    target.setdefault('qualification_history', []).append(dict(at=now, qualification_status=target.get('qualification_status'),
        counted_in_quota=target.get('counted_in_quota'), reason='new successful teacher branch; historical failures retained'))
    target.update(data_type='dual_fate', controller_episode_label=1, qualification_status='qualified',
        counted_in_quota=True, counted_in_negative_quota=False, source_owner=SOURCE_OWNER, execution_owner=OWNER,
        core_episode_id=run, teacher=request['teacher'], group_id=request['group_id'], split=request['split'],
        positive_publication=refs['publication'], positive_commit_bindings=deepcopy(refs), main_qualification_ref=refs['acceptance']['path'],
        main_qualification_sha256=refs['acceptance']['sha256'], remote_manifest=refs['manifest']['path'],
        remote_manifest_sha256=refs['manifest']['sha256'], remote_receipt=refs['publication']['path'],
        remote_receipt_sha256=refs['publication']['sha256'], qualified_at=now)
    r = out['reservations'][SOURCE]
    r.update(quota_reserved=False, negative_quota_reserved=False, status='settled_qualified_dual_fate',
             settled_at=now, current_attempt_settlement=refs['publication'])
    t = out['tasks'][TASK]
    rows = [row for row in out['source_index'].values() if row.get('task') == TASK]
    totals(t, task['qualified_count']+1, task['negative_count'])
    active = [row for row in out['reservations'].values() if row.get('task') == TASK]
    t['reserved_count'] = sum(row.get('quota_reserved') is True for row in active)
    t['negative_reserved_count'] = task['negative_reserved_count']
    need(t['qualified_count'] + t['reserved_count'] <= t['positive_quota'], 'positive reservations overflow')
    need(t['negative_count'] + t['negative_reserved_count'] <= t['negative_quota'], 'negative reservations overflow')
    t['teacher_contributions'] = dict(Counter(row['teacher'] for row in rows if positive(row)))
    t['split_counts'] = dict(Counter(row['split'] for row in rows if positive(row)))
    t.setdefault('remote_manifest_refs', []).append(refs['manifest']['path'])
    t['deficits'] = dict(positive=t['remaining_count'], negative=t['negative_remaining_count'])
    t.update(updated_at=now, last_core_gain_at=now, last_core_gain_category='positive')
    n = out['nodes'][HOST]
    own = [row for row in out['source_index'].values() if row.get('host') == HOST]
    totals(n, node['qualified_count']+1, node['negative_count'])
    allrows = list(out['source_index'].values())
    totals(out, ledger['qualified_count']+1, ledger['negative_count'])
    contribution = sum(positive(row) and row.get('teacher') == request['teacher'] for row in own)
    teacher = out.setdefault('teachers', {}).setdefault(request['teacher'], {}).setdefault('nodes', {}).setdefault(HOST, {})
    teacher.update(qualified_count=contribution, updated_at=now)
    out['revision'] += 1
    out['updated_at'] = now
    n['updated_at'] = now
    return out, dict(idempotent_reuse=False, positive_delta=1, negative_delta=0, revision=out['revision'])
