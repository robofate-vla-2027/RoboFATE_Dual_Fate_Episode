"""Original pure owner reservation; identities and bootstrap guard are external."""
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from ..evidence import need

def _settled_predecessor(ledger, prior, boundary):
    row = ledger['reservations'][prior['source_id']]
    status = row['status']
    if status.startswith('settled_'):
        return True
    if status != 'shared_Evo_teacher_settled':
        return False
    expected = boundary.get('last_teacher_settlement')
    run_id = row.get('current_attempt_run_id')
    return (row.get('positive_slots') == 0 and row.get('negative_slots') == 0
            and row.get('quota_reserved') is False
            and row.get('negative_quota_reserved') is False
            and isinstance(expected, dict) and set(expected) == {'path', 'sha256'}
            and isinstance(run_id, str) and bool(run_id)
            and Path(expected['path']).parent.parent.name == run_id
            and row.get('current_attempt_settlement') == expected
            and isinstance(row.get('settled_at'), str))


def _inherited_prefix(ledger, boundary, source, ordinal):
    inherited = boundary.get('inherited_prefix')
    if not isinstance(inherited, dict):
        return False
    prior = boundary['settlement']
    old = ledger['source_index'].get(prior['source_id'], {})
    prepared = inherited.get('prepared_source', {})
    previous_prefix = prepared.get('required_accounted_prefix')
    previous_ordinal = prepared.get('layout_id')
    return (inherited.get('source_id') == prior['source_id']
            and inherited.get('preparation_ref') == old.get('preparation')
            and prepared.get('source_id') == prior['source_id']
            and prepared.get('task') == source['task'] == old.get('task')
            and old.get('source_started') is True
            and old.get('settlement') == prior['settlement_ref']
            and isinstance(previous_prefix, list) and type(previous_ordinal) is int
            and old.get('required_accounted_prefix') == previous_prefix
            and prepared.get('source_sequence') == source.get('source_sequence') == old.get('source_sequence')
            and source['required_accounted_prefix'] == previous_prefix + [previous_ordinal]
            and ordinal in previous_prefix + [previous_ordinal])

def reserve(ledger, boundary, at):
    now = datetime.fromisoformat(at)
    b = boundary
    need(now.tzinfo is not None and 0 <= (now - datetime.fromisoformat(b['at'])).total_seconds() < 600,
         'Fresh resource observation required')
    prepared = b['next_preparation']['prepared']
    source = prepared['source']
    task, sid = source['task'], source['source_id']
    run = b['resource_precheck']['summary']['run_id']
    n, t = ledger['nodes'][HOST], ledger['tasks'][task]
    need(ledger['campaign_id'] == CAMPAIGN and n['active_chat_id'] == OWNER
         and source['execution_owner'] == OWNER and n['user_pause_active'] is False
         and n['user_stop'] is False, 'Owner, campaign or pause changed')
    need(t['assigned_host'] == HOST and task in n['fixed_tasks'], 'Foreign assigned task')
    need(n['active_real_simulators'] == 0 and n['active_run_id'] is None
         and not n['launch_inflight'] and n['baseline_accounting']['baseline_attempts_running'] == 0,
         'Previous source remains active')
    need(len(n['completed_pending_sources']) <= 2 and len(n['common_pending_readback_run_ids']) <= 2,
         'Existing readback backlog limit exceeded')
    need(sid not in ledger['source_index'] and sid not in ledger['reservations'], 'Used or reserved source')
    need(sid not in EXCLUDED,
         'Protected source')
    need(t['negative_count'] + t['negative_reserved_count'] < t['negative_quota']
         and n['negative_count'] + n['negative_reserved_count'] < n['negative_quota'],
         'No independent-negative quota')
    need(type(prepared['native_step_limit']) is int and prepared['native_step_limit'] > 0
         and prepared['mode'] == 'independent_baseline'
         and prepared['requested_initial_native_count'] == 0, 'Wrong native baseline budget or mode')
    need(prepared['renderer_profile'] == prepared['prospective_teacher_renderer_profile'],
         'Renderer differs between prospective branches')
    need(_run_name(run), 'Wrong owned task run')
    prior = b['settlement']
    _predecessor(ledger, b, source)
    rows = [s for s in ledger['source_index'].values() if s.get('host') == HOST and s.get('task') == task]
    need(not any(s.get('initialization_content_sha256') == source['initialization_content_sha256'] for s in rows),
         'Duplicate initialization')
    for i in source['required_accounted_prefix']:
        matches = [s for s in rows if s['source_id'].startswith(f'{task}_b0_l{i}_')]
        need((len(matches) == 1 and matches[0].get('settlement') is not None) or _inherited_prefix(ledger, b, source, i), 'Unsettled original prefix')
    need(t.get('latest_segment_started_at') is not None and n['current_task'] == task,
         'Task switch needs its own scheduling boundary')
    need(b['runtime_release'] == n['current_runtime_adoption']['runtime_release'],
         'Runtime switch needs its own adoption boundary')
    need(n['current_dispatch']['teacher_wall_seconds'] == 3600, 'Teacher wall scope changed')
    need(not any(s.get('run_id') == run for s in ledger['source_index'].values()), 'Used run identity')
    d = deepcopy(ledger)
    n, t = d['nodes'][HOST], d['tasks'][task]
    revision = d['revision'] + 1
    dispatch = dict(schema='refate.foreground_dispatch.v1', data_kind=DATA_KIND, host=HOST,
        expected_hostname=HOSTNAME, owner_chat=OWNER, source_owner=source['source_owner'],
        execution_owner=OWNER, campaign=CAMPAIGN, task=task, source_id=sid, run_id=run,
        issued_at=at, valid_until=(now + timedelta(hours=2)).isoformat(), ledger_revision=revision,
        source_reserved=True, source_started=False, negative_slots=1, positive_slots=0,
        native_total_steps=prepared['native_step_limit'], native_wall_seconds=n['current_dispatch']['native_wall_seconds'], teacher_wall_seconds=3600,
        rtx_profile=prepared['renderer_profile'], original_segment_started_at=t['latest_segment_started_at'],
        campaign_start_preserved=d['started_at'], runtime_release=deepcopy(b['runtime_release']),
        resource_precheck=deepcopy(b['resource_precheck']['ref']), source_preparation=deepcopy(b['next_preparation']['ref']),
        split_manifest=deepcopy(prepared['provenance']['split']), predecessor_disposition=deepcopy(prior.get('settlement_ref')),
        native_kit_extensions=deepcopy(b.get('native_kit_extensions', [])), teacher_ready=False,
        reason='Next unused original source after actual predecessor settlement; same shared baseline, budget and segment clock.')
    if dispatch['native_kit_extensions']:
        dispatch['native_extension_pins'] = deepcopy(b['native_extension_pins'])
    row = dict(source_id=sid, task=task, host=HOST, owner_chat=OWNER,
        source_owner=source['source_owner'], execution_owner=OWNER, run_id=run,
        status='reserved_independent_baseline', created_at=at, updated_at=at, source_started=False,
        source_group_id=source['group_id'], split=source['split'], native_budget=prepared['native_step_limit'],
        layout_sha256=source['layout_sha256'], initialization_content_sha256=source['initialization_content_sha256'],
        source_preparation_ref=deepcopy(b['next_preparation']['ref']), qualified_count=0, negative_count=0,
        counted_in_quota=False, counted_in_negative_quota=False, history=[])
    if 'collection_group_compatibility' in prepared:
        row['collection_group_limitation'] = deepcopy(prepared['collection_group_compatibility'])
    d['source_index'][sid] = row
    d['reservations'][sid] = dict(source_id=sid, task=task, host=HOST, owner_chat=OWNER,
        execution_owner=OWNER, run_id=run, status='reserved', negative_slots=1, positive_slots=0,
        updated_at=at, dispatch=deepcopy(dispatch))
    n['negative_reserved_count'] += 1
    t['negative_reserved_count'] += 1
    previous_adoption = deepcopy(n['current_runtime_adoption'])
    n.setdefault('runtime_adoption_history', []).append(previous_adoption)
    n.setdefault('dispatch_history', []).append(deepcopy(n['current_dispatch']))
    n.update(current_dispatch=deepcopy(dispatch), current_dispatch_reuse_allowed=False,
        active_run_id=run, status='collecting', phase='common_core_source_reserved', launch_inflight=False, updated_at=at)
    n['current_common_foreground'] = dict(at=at, run_id=run, source_id=sid, stage='reserved',
        actual_entry=n['current_common_foreground']['actual_entry'], positive_delta=0, negative_delta=0)
    n['current_runtime_adoption'] = dict(at=at, runtime_release=deepcopy(b['runtime_release']), run_id=run,
        source_id=sid, inherited_actual_adoption=previous_adoption.get('inherited_actual_adoption'),
        actual_native_started=False, actual_teacher_verified=False, actual_full_readback_verified=False,
        scope='Same selected baseline; new source not started, teacher/positive paths not certified')
    if 'shared_teacher_dependency_delivery' in n and not b['teacher_supplement_present']:
        n['shared_teacher_dependency_delivery'].update(at=at, status='required_original_files_absent_at_source_boundary')
    t.update(active_run_id=run, status='collecting', phase='common_core_source_reserved', updated_at=at)
    d.update(revision=revision, updated_at=at)
    return d, dispatch
