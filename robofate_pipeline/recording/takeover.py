"""Owner, budget and transition validation without live deployment state."""
import copy
PUBLIC_KEYS = {'instruction', 'vision', 'state', 'action', 'additional_info'}

def validate_action(action, state):
    from integrations.robodojo_policy.bridge import validate_action as original
    return original(action, state)


class ContractError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ContractError(message)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f'Invalid {name}')
    return value


def request_metadata(context, native_used, seq):
    return {key: context[key] for key in ('campaign', 'chat', 'active_chat_id', 'task',
                                         'session', 'source_id', 'owner', 'epoch')} | {
        'sequence': seq, 'native_before': native_used,
        'native_limit': context['native_limit'],
    }


def validate_transition(message, pending):
    require(isinstance(message, dict) and message.get('kind') == 'transition',
            'Missing native transition; never retry an uncertain action')
    meta = message.get('refate')
    require(isinstance(meta, dict), 'Missing native execution receipt')
    for key, value in pending['refate'].items():
        require(type(meta.get(key)) is type(value) and meta[key] == value,
                f'Mismatched/stale transition {key}')
    after = integer(meta.get('native_after'), 'native_after')
    require(after == meta['native_before'] + 1 and after <= meta['native_limit'],
            'Expected exactly one native action; budget/counter mapping changed')
    require(isinstance(meta.get('dense_trace_ref'), str) and meta['dense_trace_ref'],
            'Missing executor-owned dense trace reference')
    for key in ('success', 'terminated', 'truncated'):
        require(type(message.get(key)) is bool, f'Missing native {key}')
    require(not message['success'] or message['terminated'], 'Success must be terminal')
    require(not (message['terminated'] and message['truncated']), 'Inconsistent terminal flags')
    require(after < meta['native_limit'] or message['terminated'] or message['truncated'],
            'Executor did not terminate at the native limit')
    obs = message.get('observation')
    require(isinstance(obs, dict) and PUBLIC_KEYS.issuperset(obs)
            and isinstance(obs.get('state'), dict) and bool(obs['state'])
            and isinstance(obs.get('instruction'), str), 'Missing public transition observation')
    return meta


def check_dispatch(request, current):
    """Call inside main's existing single-writer critical section, before take_action.

    `current` must be freshly read execution-owner state, never copied from a
    request. The caller retains this lock through action, counter read and trace
    publication. This validator does not itself revoke DM or establish ownership.
    """
    require(isinstance(request, dict) and request.get('kind') == 'action'
            and isinstance(request.get('refate'), dict), 'Missing action envelope')
    meta = request['refate']
    for key in ('campaign', 'chat', 'active_chat_id', 'task', 'session', 'source_id',
                'owner', 'epoch', 'native_limit'):
        require(key in current and type(meta.get(key)) is type(current[key])
                and meta[key] == current[key], f'Wrong current execution owner: {key}')
    require(current.get('dm_submissions_disabled') is True
            and type(current.get('dm_inflight')) is int and current['dm_inflight'] == 0
            and current.get('dm_queue_discarded') is True and current.get('native_terminal') is False,
            'DM fence/native boundary is not ready')
    used = integer(current.get('native_used'), 'current native_used')
    total = integer(current.get('native_limit'), 'current native_limit', 1)
    seq = integer(meta.get('sequence'), 'sequence', 1)
    require(type(meta.get('native_before')) is int and meta['native_before'] == used and used < total,
            'Stale action or exhausted budget')
    require(seq == integer(current.get('last_sequence'), 'last_sequence') + 1,
            'Replayed or out-of-order action')
    return validate_action(request['action'], current['public_state'])
