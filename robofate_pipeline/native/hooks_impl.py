"""Original Native construction hooks; external roots and pinned assets."""
import ast
import hashlib
from pathlib import Path
PROFILE = 'pt_spatial4_collection_candidate_v1'

def creation_source(raw):
    """Change only proven node path/device/config merge points in the exact upstream body."""
    need(hashlib.sha256(raw).hexdigest() == UPSTREAM_SHA, 'Unexpected upstream task body')
    tree = ast.parse(raw)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RoboDojoTask')
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_create_native')
    text = ast.get_source_segment(raw.decode(), function)
    replacements = (
        ('True, True, "cuda:0"', 'True, True, "cuda:1"'),
        ('self.app = AppLauncher(args).app',
         'args.experience = str(_experience)\n        self.app = _make_app(AppLauncher, args).app'),
        ('device_id=0,', 'device_id=1,'),
        ('cfg.camera.default_frequency = eval_cfg["observation"]["collect_freq"]',
         'cfg.camera.default_frequency = eval_cfg["observation"]["collect_freq"]\n        _node_adjust_config(cfg)'),
        ('self.native.setup_sim_cfg = configure_with_private_logs',
         'self.native.setup_sim_cfg = configure_with_private_logs\n        _node_prepare_native(self.native)'),
    )
    for old, new in replacements:
        need(text.count(old) == 1, 'Node merge point changed: '+old)
        text = text.replace(old, new)
    return text


def _adjust_config(cfg):
    need(cfg.sim.device == 'cpu' and cfg.eval_cfg.device_id == 1, 'CPU physics state / GPU1 renderer binding differs')
    need(cfg.eval_cfg.observation.collect_freq == cfg.camera.default_frequency == 25,
         'Original native25 camera cadence differs')
    cfg.sim.use_fabric = False
    for robot in cfg.robot.robots:
        path = Path(_planner_refs[robot.robot_name]['path'])
        _read_pin(_planner_refs[robot.robot_name])
        need(path.is_file(), 'Existing private planner config missing')
        robot.curobo_yml_path = str(path)
    import torch
    torch.cuda.set_device(1)


def _prepare_native(native):
    import torch
    from types import MethodType
    original = native.setup_physics

    def setup(self, sim):
        original(sim)
        self.physics_interface.overwrite_gpu_setting(1)
        need(self.physics_interface.get_overwrite_gpu_setting() == 1, 'Actual physics GPU binding failed')
        import carb
        settings = carb.settings.get_settings()
        settings.set_bool('/physics/updateToUsd', True)
        settings.set_bool('/physics/updateParticlesToUsd', True)

    native.setup_physics = MethodType(setup, native)
    # Preserve the original planner and its semantics, allocating it on policy GPU0.
    planner = native.robot_manager._setup_planner
    def setup_planner(robot):
        need(not str(native.robot_manager.device).startswith('cuda'), 'Unexpected GPU state planner binding')
        with torch.cuda.device(0):
            return planner(robot)
    native.robot_manager._setup_planner = setup_planner
    _remap_offline(native)


def verify_initial(task, preparation):
    source = preparation['source']
    total = preparation['native_step_limit']
    need(task.task_name == source['task'] and task.official_action_limit is True, 'Actual task identity differs')
    need(all(type(n) is int and n == total for n in (task.max_steps, task.native_step_limit, task.native.step_lim)),
         'Original native task budget differs')
    need(task.control_steps == 0 and list(task.native.take_action_cnt) == [0], 'New baseline must begin at zero')
    need(float(task.native.dt) == .004 and int(task.native.obs_manager.collect_interval) == 10,
         'Original native control/camera interval differs')
    need(task.bank == source['native_bank'] and task.ordinal == source['native_ordinal'], 'Actual bank/layout ordinal differs')
    need(Path(task.layout_path).resolve() == Path(source['resolved_layout_path'])
         and hashlib.sha256(Path(task.layout_path).read_bytes()).hexdigest() == source['layout_ref']['sha256'],
         'Actual raw layout differs')
    return {'task':task.task_name, 'native_total_steps':total, 'native_used':0,
            'layout':str(task.layout_path), 'logical_policy_gpu':0, 'logical_simulation_gpu':1,
            'actual_render_device_verified':False}
