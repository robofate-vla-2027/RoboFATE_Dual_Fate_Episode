"""Direct per-action wrapper for main's ALREADY INITIALIZED RoboDojoTask."""
import copy

from .takeover import require, integer, check_dispatch


class ExistingEnvTeacher:
    """No simulator, service or resource creation in this wrapper.

    Main owns the single-writer lock, native control_step and passive recorder.
    The supplied original Model owns only its author coding-agent worker.
    """
    def __init__(self, model, task, *, owner_snapshot):
        require(callable(owner_snapshot), 'Main must provide its live action-owner fence')
        self.model, self.task, self.owner_snapshot = model, task, owner_snapshot
        self.pending = None
        self.last_sequence = 0
        self.bound = False
        self.failed = False

    def _snapshot(self, obs):
        task = self.task
        require(getattr(task, '_failure', None) is None, 'Native task has an unresolved failure')
        require(task.official_action_limit is True, 'Native action limit must be enabled')
        used = integer(task.control_steps, 'RoboDojoTask.control_steps')
        limit = integer(task.native_step_limit, 'RoboDojoTask.native_step_limit', 1)
        require(used == int(task.native.take_action_cnt[0]), 'Native/cumulative counter mismatch')
        require(limit == task.max_steps == task.native.step_lim, 'Native budget was reset or changed')
        for key in ('success', 'terminated', 'truncated'):
            require(type(getattr(task, key)) is bool, f'Invalid RoboDojoTask.{key}')
        return {**copy.deepcopy(self.owner_snapshot()), 'native_used': used,
                'native_limit': limit, 'native_terminal': task.done,
                'last_sequence': self.last_sequence, 'public_state': obs['state']}

    def takeover(self, observation, causal_history, remaining_steps):
        require(not self.bound, 'ExistingEnvTeacher is one-shot')
        current = self._snapshot(observation)
        context = self.model.cfg['refate_takeover']
        for key in ('campaign', 'chat', 'session', 'source_id', 'owner', 'epoch',
                    'native_used', 'native_limit', 'dm_submissions_disabled',
                    'dm_inflight', 'dm_queue_discarded', 'native_terminal'):
            require(key in current and type(current[key]) is type(context[key])
                    and current[key] == context[key], f'Takeover fence mismatch: {key}')
        result = self.model.takeover(observation, causal_history, remaining_steps)
        self.bound = True
        return result

    def predict(self, currentobs):
        """Return one original native action, or None while the author agent thinks.

        None means NO physics advancement, not a hold action. Main must retain
        its sole-writer lock/fence until it executes a returned action.
        """
        require(self.bound and not self.failed, 'Unbound or failed teacher adapter')
        require(self.pending is None, 'Deliver actual after_action feedback before another prediction')
        current = self._snapshot(currentobs)
        if current['native_terminal']:
            return None
        require(current['native_used'] < current['native_limit'], 'Native budget exhausted')
        self.model.update_obs(currentobs)
        self.failed = True
        actions = self.model.get_action()
        if not actions:
            self.failed = False
            return None
        require(len(actions) == 1, 'Original EmbodiedEvo must yield one action at a time')
        self.failed = True
        request = self.model.get_takeover_request()
        # Re-read ownership after possible model waiting. Never adopt an owner
        # from the action message itself.
        current = self._snapshot(currentobs)
        action = check_dispatch(request, current)
        self.pending = request
        self.failed = False
        return action

    def after_action(self, obs_after, *, dense_trace_ref):
        """Call immediately after control_step, INCLUDING the final native action.

        The reference is supplied by main's real passive dense recorder. This
        wrapper does not manufacture actuator/substep records or new frames.
        """
        require(self.bound and not self.failed and self.pending is not None, 'No valid pending action')
        self.failed = True
        current = self._snapshot(obs_after)
        meta = self.pending['refate']
        for key in ('campaign', 'chat', 'session', 'source_id', 'owner', 'epoch', 'native_limit'):
            require(type(current.get(key)) is type(meta[key]) and current[key] == meta[key],
                    f'Owner changed during execution: {key}')
        require(current['dm_submissions_disabled'] is True and current['dm_inflight'] == 0
                and current['dm_queue_discarded'] is True, 'DM fence was lost during execution')
        message = {'kind': 'transition', 'observation': obs_after,
                   'success': self.task.success, 'terminated': self.task.terminated,
                   'truncated': self.task.truncated,
                   'refate': {**meta, 'native_after': current['native_used'],
                              'dense_trace_ref': dense_trace_ref}}
        self.model.accept_transition(message)
        self.last_sequence = meta['sequence']
        self.pending = None
        self.failed = False
        return message
