"""Original takeover, worker validation and unknown-action latch; factory template."""
import copy
import json
from pathlib import Path
PUBLIC_KEYS = {'instruction', 'vision', 'state', 'action', 'additional_info'}

class ContractError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ContractError(message)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f'Invalid {name}')
    return value


def validate_context(context, observation, causal_history, remaining_steps):
    require(isinstance(context, dict), 'Missing execution-owner context')
    c = copy.deepcopy(context)
    require(c.get('campaign') == CAMPAIGN and c.get('chat') == CHAT, 'Wrong campaign/chat')
    require(c.get('task') == TASK and c.get('active_chat_id') == ACTIVE_CHAT_ID,
            'Current task/owner required')
    for key in ('session', 'owner', 'source_id', 'boundary_receipt_ref', 'budget_mapping_ref'):
        require(isinstance(c.get(key), str) and bool(c[key].strip()), f'Missing {key}')
    integer(c.get('epoch'), 'epoch', 1)
    total = integer(c.get('native_limit'), 'native_limit', 1)
    used = integer(c.get('native_used'), 'native_used')
    integer(remaining_steps, 'remaining_steps', 1)
    require(total == NATIVE_LIMIT and used > 0, 'Bound native limit required; use actual t1')
    require(total - used == remaining_steps, 'Remaining budget must equal native limit minus used')
    require(c.get('dm_submissions_disabled') is True and c.get('dm_inflight') == 0
            and type(c.get('dm_inflight')) is int and c.get('dm_queue_discarded') is True,
            'Execution owner has not acknowledged the DM stop boundary')
    require(c.get('native_terminal') is False, 'Cannot take over a terminal environment')
    require(isinstance(observation, dict) and PUBLIC_KEYS.issuperset(observation),
            'Observation contains non-public fields')
    require(isinstance(observation.get('instruction'), str) and observation['instruction']
            and isinstance(observation.get('state'), dict) and observation['state'],
            'Instruction and public state are required')
    require(isinstance(causal_history, list) and len(causal_history) == used + 1,
            'Provide one causal boundary per native index 0..t1, not a future failure suffix')
    history = copy.deepcopy(causal_history)
    for index, entry in enumerate(history):
        require(isinstance(entry, dict) and set(entry) ==
                {'native_step', 'observation', 'executed_action', 'feedback'},
                'History must contain only observable prefix fields')
        require(type(entry['native_step']) is int and entry['native_step'] == index,
                'History must be contiguous and end at t1')
        obs = entry['observation']
        require(isinstance(obs, dict) and PUBLIC_KEYS.issuperset(obs)
                and obs.get('instruction') == observation['instruction'], 'Invalid causal observation')
        require((entry['executed_action'] is None) if index == 0
                else isinstance(entry['executed_action'], dict), 'Missing executed prefix action')
        require(isinstance(entry['feedback'], dict)
                and set(entry['feedback']) == {'success', 'terminated', 'truncated'},
                'History requires all three actual public native flags')
        require(all(type(v) is bool for v in entry['feedback'].values()), 'Non-boolean feedback')
        require(not any(entry['feedback'].values()), 'Prefix contains native termination')
    # Numerical/state equality and restoration qualification remain execution-owner checks.
    c.update(remaining_steps=remaining_steps, causal_history=history)
    return c


def bind_model(model, observation, causal_history, remaining_steps):
    verify_release()
    require(model.worker is None and model.connection is None and model.episode == -1,
            'Use a fresh teacher model; never reset an active worker')
    require(not getattr(model, '_takeover_bound', False), 'Takeover may only be bound once')
    require(model.cfg.get('phase', 'test') == 'test' and not model.cfg.get('diagnostic'),
            'Only original Test tools are allowed')
    c = validate_context(model.cfg.get('refate_takeover'), observation, causal_history, remaining_steps)
    model.cfg = {**model.cfg, 'refate_takeover': c, 'phase': 'test'}
    model.step_limit = c['native_limit']
    model.update_obs(observation)
    model._takeover_bound = True
    model.takeover_request = None
    return {'bound': True, 'native_steps_used': c['native_used'],
            'native_steps_remaining': remaining_steps, 'takeover_verified': False}


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


class TakeoverBridgeTask(BridgeTask):
    """Preserves the four primitives and arbitrary native action targets."""
    def __init__(self, connection, observation, step_limit, *, context, phase='test'):
        require(phase == 'test', 'Playground/reset is not an existing-env takeover')
        self.context = validate_context(context, observation, context['causal_history'],
                                        context['remaining_steps'])
        require(step_limit == self.context['native_limit'], 'Worker received a different native limit')
        self._bound = self.invalid = False
        self._sequence = 0
        super().__init__(connection, observation, step_limit, phase='test')

    def reset(self, *, seed=None, options=None):
        # Session.__init__ calls reset once. This initializes only BridgeTask's
        # Python namespace from supplied data: no pipe/native reset is emitted.
        require(not self._bound and options is None, 'Native/policy rebinding is forbidden')
        self._bound = True
        self._task_prompt = self.initial['instruction']
        self.start_episode(seed, self.initial)
        self.control_steps = self.context['native_used']
        self.truncated = False
        return self.public_observation(), {'task_prompt': self._task_prompt}

    def public_observation(self):
        obs = super().public_observation()
        obs['causal_history'] = copy.deepcopy(self.context['causal_history'])
        obs['remaining_steps'] = max(0, self.max_steps - self.control_steps)
        obs['takeover_native_index'] = self.context['native_used']
        return obs

    def control_step(self, action):
        require(self._bound and not self.invalid, 'Takeover is unbound or invalid')
        if self.done:
            return self.transition(float(self.success))
        require(self.control_steps < self.max_steps, 'Native budget exhausted')
        action = validate_action(action, self._observation['state'])
        self._sequence += 1
        request = {'kind': 'action', 'action': action,
                   'refate': request_metadata(self.context, self.control_steps, self._sequence)}
        # Latch before dispatch: any unknown outcome blocks further submissions.
        self.invalid = True
        self.connection.send(request)
        message = self.connection.recv()
        meta = validate_transition(message, request)
        require(message['observation']['instruction'] == self._task_prompt, 'Task instruction changed')
        self.control_steps = meta['native_after']
        self.success = message['success']
        self.terminated = message['terminated']
        self.truncated = message['truncated']
        self.observe(policy_observation(message['observation']))
        self.invalid = False
        return self.transition(float(self.success))


def accept_model_transition(model, message):
    require(model.pending and model.takeover_request is not None, 'No action awaiting native receipt')
    validate_transition(message, model.takeover_request)
    require(message['observation']['instruction'] == model.obs['instruction'], 'Task changed')
    model.update_obs(message['observation'])
    model.connection.send(copy.deepcopy(message))
    model.pending = False
    model.takeover_request = None


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


def validate_worker_spec(specification, episode):
    load_binding(specification.get('refate_main_binding_ref'),
                 context=specification.get('refate_takeover'))
    verify_release()
    require(episode == 0 and specification.get('phase') == 'test'
            and not specification.get('diagnostic'), 'Exactly one Test takeover per fresh worker')
    run = Path(specification['run_dir'])
    require(run.is_absolute() and run.resolve() == run
            and run.is_relative_to(OUTPUT / 'sessions'), 'Worker run must be session-private')
    actual_run = run if specification.get('flat_episode', False) else run / 'episodes/episode_0000'
    require(actual_run.resolve() == actual_run, 'Linked episode output path')
    c = specification.get('refate_takeover')
    require(isinstance(c, dict) and 'causal_history' in c, 'Call Model.takeover before worker launch')
    require(run.name == c['session'], 'Run directory/session identity mismatch')
    cfg = specification['task_config']
    require(cfg == task_config(), 'Pinned original teacher configuration changed')
    require(cfg.get('baseline') is True and cfg['simulator']['env_task'] == TASK
            and cfg['test'].get('shared_plan') is None,
            'This is the public-recipe primitive baseline on the existing source scene')
    require(cfg['agent']['preset'] == 'codex' and cfg['agent']['isolation'] == 'docker'
            and cfg['agent']['image'] == 'embodiedevo-agent:latest',
            'Original agent image/runtime identity changed')
    require(cfg['agent']['model'] == 'gt-6-as-a' and cfg['agent']['reasoning_effort'] == 'xhigh',
            'API request identity changed; publish a new candidate')
    require(cfg['test']['budget'] == 100 and cfg['test']['submission_timeout_sec'] == 1200,
            'Preserve author execution-request and submission budgets')


def append_context_instruction(run):
    with (Path(run) / 'workspace/instruction.md').open('a') as stream:
        stream.write('\n\nExisting-environment takeover: the simulator is already at an executed '
                     'control boundary. Do not reset it. get_observation() includes the observable '
                     'causal_history up to takeover_native_index and the actual remaining_steps. '
                     'Use the original primitives and current physical feedback to finish the '
                     'native task. The original total limit is not a fresh allowance.\n')


def record_api_identity(run, *, request_id, gateway_reported_model, evidence_ref):
    """Main's optional receipt sink; no request, route change or inference occurs.

    IDs/report names must come from an observed request/response, not the old
    run or this candidate's configured model. Missing values remain None.
    """
    run = Path(run)
    require(run.is_absolute() and run.resolve() == run and run.is_relative_to(OUTPUT / 'sessions'),
            'API metadata must stay in the private session output')
    require(isinstance(evidence_ref, str) and evidence_ref, 'Missing API identity evidence provenance')
    for value in (request_id, gateway_reported_model):
        require(value is None or (isinstance(value, str) and 0 < len(value) <= 256
                                and '\n' not in value and '\r' not in value), 'Invalid public API identity')
    record = {'requested_model': 'gt-6-as-a', 'requested_reasoning_effort': 'xhigh',
              'request_id': request_id, 'gateway_reported_model': gateway_reported_model,
              'evidence_ref': evidence_ref, 'teacher_condition': 'primitive_baseline_existing_env_candidate'}
    with (run / 'agent/api-identity.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    return record
