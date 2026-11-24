"""Original fresh worker specification, with external runtime binding."""
import copy

def specification(runtime, *, main_binding_ref, context, observation, causal_history):
    binding = runtime.validate_binding()
    need = runtime.need
    need(type(context) is dict, 'Actual takeover context required')
    for key, value in binding['identity'].items():
        need(type(context.get(key)) is type(value) and context[key] == value,
             'Worker identity differs from Native binding: ' + key)
    total, used = binding['native_limit'], binding['t1']
    for key, expected in (('native_limit', total), ('native_used', used),
                          ('remaining_steps', total-used)):
        need(type(context.get(key)) is int and context[key] == expected,
             'Worker budget differs from original source: ' + key)
    checked = runtime.takeover.validate_context(context, observation, causal_history, total-used)
    runtime.main_binding.load_binding(main_binding_ref, session=binding['session'], context=checked)
    config = runtime.task_config()
    result = dict(data_kind=binding['data_kind'], runtime_binding=copy.deepcopy(binding),
                  run_dir=binding['run_dir'], flat_episode=True, phase='test', seconds=3600,
                  task_config=config, refate_main_binding_ref=copy.deepcopy(main_binding_ref),
                  refate_takeover=checked)
    runtime.takeover.validate_worker_spec(result, 0)
    return result


def model(runtime, **kwargs):
    """Construct the original unbound Model; common_teacher performs takeover."""
    result = runtime.Model(specification(runtime, **kwargs))
    runtime.need(result.worker is None and result.connection is None,
                              'Model construction unexpectedly started a worker')
    return result
