"""Original canonical recount and baseline transaction with external identity."""
from copy import deepcopy
from ..evidence import need

def export_current_counts(d, task):
    rows = list(d['source_index'].values())
    if not all(type(row) is dict for row in rows):
        raise ValueError('Unknown source index schema')
    for row in rows:
        if row.get('counted_in_quota') is True and row.get('counted_in_negative_quota') is True:
            raise ValueError('One source cannot count both native fates')
    def assign(scope, selected):
        positive = sum(row.get('counted_in_quota') is True for row in selected)
        negative = sum(row.get('counted_in_negative_quota') is True for row in selected)
        if positive > scope['positive_quota'] or negative > scope['negative_quota']:
            raise ValueError('Core source index exceeds quota')
        scope.update(qualified_count=positive, negative_count=negative, core_count=positive+negative,
            remaining_count=scope['positive_quota']-positive,
            negative_remaining_count=scope['negative_quota']-negative,
            core_remaining_count=scope['core_quota']-positive-negative)
    assign(d, rows)
    n = d['nodes'][HOST]
    assign(n, [row for row in rows if row.get('host') == HOST])
    assign(d['tasks'][task], [row for row in rows if row.get('task') == task])
    d['negative_task_coverage'] = len({row['task'] for row in rows if row.get('counted_in_negative_quota') is True})
    if 'negative_accounting' in n:
        n['negative_accounting']['current_node_negative_count'] = n['negative_count']


def settle(ledger, evidence, at):
    original_ledger = ledger
    ledger = deepcopy(ledger)
    export_current_counts(ledger, evidence["task"])
    e = evidence
    q = e['summary']
    need(ledger['campaign_id'] == CAMPAIGN and e['data_kind'] == q['data_kind'] == DATA_KIND, 'Wrong actual campaign/evidence')
    n = ledger['nodes'][HOST]
    need(n['active_chat_id'] == e['execution_owner'] == OWNER and n['user_pause_active'] is False and n['user_stop'] is False, 'Execution owner or pause changed')
    sid, run = e['source_id'], e['run_id']
    s = ledger['source_index'][sid]
    need(s['source_id'] == q['source_id'] == sid and s['run_id'] == q['run_id'] == run
         and s['execution_owner'] == OWNER and s['task'] == e['task'] and s['host'] == HOST, 'Different source or execution scope')
    need(q['file_recording_readback_passed'] is True and e['first_readback_passed'] is True
         and e['manifest'] == q['manifest'], 'Existing full data readback required')
    need(type(q['native_success']) is bool and type(q['qualified_negative']) is bool
         and q['qualified_negative'] is q['native_success'] and q['qualified_positive'] is False,
         'Native/data qualification contradiction')
    negative = int(q['native_success'])
    need(type(e['new_negative']) is int and e['new_negative'] == negative and e['new_positive'] == 0,
         'Wrong independent-baseline quota delta')
    need(type(q['native_steps']) is int and 0 <= q['native_steps'] <= s['native_budget'], 'Native budget/count mismatch')
    if s.get('settlement') is not None:
        need(s['settlement'] == e['settlement_ref'] and s['manifest'] == e['manifest']
             and s['negative_count'] == negative, 'Different existing settlement')
        return deepcopy(original_ledger), {'already_settled': True, 'source_id': sid, 'negative_delta': 0, 'positive_delta': 0}
    t = ledger['tasks'][s['task']]
    r = ledger['reservations'][sid]
    need(n['active_run_id'] == run and n['current_common_foreground']['run_id'] == run and s['source_started'] is True,
         'Not the current started source')
    need(r['run_id'] == run and r['status'] == 'running' and r['negative_slots'] == 1
         and r['positive_slots'] == 0 and s['negative_count'] == s['qualified_count'] == 0, 'Reservation already consumed or wrong')
    need(t['assigned_host'] == HOST and t['negative_reserved_count'] >= 1
         and n['negative_reserved_count'] >= 1 and n['baseline_accounting']['baseline_attempts_running'] == 1,
         'Current reservation/accounting mismatch')
    if negative:
        for scope in (ledger, n, t):
            need(scope['negative_count'] < scope['negative_quota'], 'No negative quota deficit')
        need(not any(row.get('source_id') == sid for row in t.get('negative_remote_manifest_refs', [])), 'Source already in negative manifests')
    d = deepcopy(ledger)
    n, s, r, t = d['nodes'][HOST], d['source_index'][sid], d['reservations'][sid], d['tasks'][e['task']]
    status = 'qualified_native_success_negative' if negative else 'complete_native_failure_readback_passed'
    s.update(status=status, native_success=q['native_success'], native_steps=q['native_steps'],
        actual_controls=q['actual_controls'], readback=e['summary_ref'], manifest=e['manifest'],
        settlement=e['settlement_ref'], reusable_DM_failure=not bool(negative), negative_count=negative, updated_at=at)
    s.update(counted_in_quota=False, counted_in_negative_quota=bool(negative),
        negative_qualification_status="qualified" if negative else "not_negative_native_failure")
    if negative:
        s.update(negative_manifest=e["manifest"], negative_qualified_at=at,
                 controller_episode_label=0, data_type="dm_native_success_negative")
    s.setdefault('history', []).append(dict(at=at, event='settled_existing_complete_baseline_readback',
        manifest=e['manifest'], readback=e['summary_ref'], settlement=e['settlement_ref'],
        foreground_to_readback_seconds=e['foreground_to_readback_seconds'], positive_delta=0,
        negative_delta=negative, negative_slot_released=1))
    r.update(status='settled_qualified_negative' if negative else 'settled_native_failure_no_quota',
        negative_slots=0, positive_slots=0, settlement=e['settlement_ref'], updated_at=at)
    t['negative_reserved_count'] -= 1
    n['negative_reserved_count'] -= 1
    account = n['baseline_accounting']
    account['baseline_attempts_running'] -= 1
    for key in ('baseline_attempts_native_terminal', 'baseline_attempts_rgb_readback_passed',
                'baseline_attempts_native_success' if negative else 'baseline_attempts_native_failure'):
        account[key] += 1
    account.setdefault('closed_source_deltas', {})[sid] = dict(native_success=negative,
        native_failure=1-negative, complete_readback=1, positive_delta=0, negative_delta=negative,
        ref=e['settlement_ref'], end_to_end_seconds=e['foreground_to_readback_seconds'])
    if negative:
        if t['negative_count'] == 0:
            d['negative_task_coverage'] += 1
        for scope in (d, n, t):
            scope['negative_count'] += 1
            scope['negative_remaining_count'] -= 1
            scope['core_count'] += 1
            scope['core_remaining_count'] -= 1
        t.setdefault('negative_split_counts', {}).setdefault(s['split'], 0)
        t['negative_split_counts'][s['split']] += 1
        t.setdefault('negative_remote_manifest_refs', []).append(dict(e['manifest'], source_id=sid))
        negatives = n.setdefault('negative_accounting', {})
        negatives['current_node_negative_count'] = n['negative_count']
        negatives.setdefault('post_legacy_source_deltas', {})[sid] = dict(negative_delta=1,
            manifest=e['manifest'], settlement=e['settlement_ref'], at=at)
    closed = deepcopy(n['current_common_foreground'])
    closed.update(stage=status, settlement=e['settlement_ref'], positive_delta=0, negative_delta=negative)
    n.setdefault('common_foreground_history', []).append(closed)
    n['current_common_foreground'] = closed
    safe = e['safe_for_next_simulator'] is True
    n.update(active_real_simulators=0 if safe else n['active_real_simulators'],
        active_run_id=None if safe else run, launch_inflight=False,
        phase='source_data_settled' if safe else 'source_data_settled_resource_cleanup_pending', updated_at=at)
    t.update(active_run_id=None if safe else run, phase='source_data_settled', updated_at=at)
    adoption = n.get('current_runtime_adoption', {})
    if adoption.get('run_id') == run:
        adoption.update(actual_full_readback_verified=True, native_success=q['native_success'],
                        native_success_negative_publication_verified=bool(negative), settlement=e['settlement_ref'])
    n['latest_baseline_ledger_commit'] = dict(source_id=sid, manifest=e['manifest'],
        settlement=e['settlement_ref'], at=at, positive_delta=0, negative_delta=negative)
    export_current_counts(d, e["task"])
    d['revision'] += 1
    d['updated_at'] = at
    commit = dict(schema='refate.common_baseline.ledger_commit.v1', data_kind=DATA_KIND, at=at,
        revision=d['revision'], source_id=sid, run_id=run, source_manifest=e['manifest'],
        settlement=e['settlement_ref'], positive_delta=0, negative_delta=negative,
        node_positive=n['qualified_count'], node_negative=n['negative_count'], lock_inode=LOCK_INODE)
    r.update(quota_reserved=False, negative_quota_reserved=False)
    return d, commit
