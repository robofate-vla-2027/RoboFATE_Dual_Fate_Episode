"""Original private-network OCI staging; all deployment paths are external."""
import hashlib
import os
from pathlib import Path
from .search import disabled_search_argv
AUTHOR_PROMPT = 'Read instruction.md. Work autonomously toward the phase goal. Interact using scripts/env.py; finish the session when done. Camera frames in observations/ are native-resolution PNGs. Use the current_cam_*.png files; never open raw_cam_* files or create copies in scratch, and inspect no more than two current frames per iteration to keep model requests focused.'

def original_command(cfg):
    need(cfg.baseline and cfg.playground is None and cfg.protocol == 'workspace-session-v1',
         'only the original Test-only baseline is supported')
    need(cfg.agent.preset == 'codex' and cfg.agent.model == 'gt-6-as-a'
         and cfg.agent.reasoning_effort == 'xhigh', 'fixed requested model/effort required')
    need(cfg.test.budget == 100 and cfg.test.submission_timeout_sec == 1200,
         'preserve original execution request budgets')
    from orchestrator import launch_agent as launch
    return disabled_search_argv(launch.adapt_for_docker(launch.agent_argv(cfg.agent.preset, AUTHOR_PROMPT,
        model=cfg.agent.model, effort=cfg.agent.reasoning_effort), preset=cfg.agent.preset))


def canonical(path, parent):
    path = Path(path)
    need(path.is_absolute() and path.resolve(strict=True) == path and path.is_relative_to(parent),
         'expected canonical path within isolated staging scope')
    return path


def bind(source, destination, *, readonly=True):
    return dict(destination=destination, type='bind', source=str(source),
        options=['rbind', 'rprivate', 'nosuid', 'nodev', 'ro' if readonly else 'rw'])


def configuration(run, socket_dir, command):
    workspace = run / 'workspace'
    mounts = [bind('/usr', '/usr'),
        dict(destination='/proc', type='proc', source='proc', options=['ro','nosuid','nodev','noexec']),
        dict(destination='/sys', type='tmpfs', source='tmpfs', options=['ro','nosuid','nodev','noexec','mode=555']),
        dict(destination='/dev', type='tmpfs', source='tmpfs', options=['nosuid','strictatime','mode=755','size=65536k']),
        dict(destination='/tmp', type='tmpfs', source='tmpfs', options=['nosuid','nodev','mode=1777']),
        bind(workspace, '/workspace', readonly=False),
        bind(socket_dir/'agent-home', '/home/agent', readonly=False),
        bind(socket_dir, '/run/embodiedevo'), bind(run/'gateway', '/gateway'),
        bind(run/'command.json', '/command.json'), bind(HERE/'guest_entry.py', '/inner.py'),
        bind(run/'agent-bin', '/agent-bin'),
        bind(PACKAGE/'bin/codex', '/agent-bin/codex'),
        bind(PACKAGE/'bin/codex-code-mode-host', '/agent-bin/codex-code-mode-host'),
        bind(PACKAGE/'codex-resources', '/codex-resources')]
    for part in ('instruction.md', 'primitives', 'scripts', 'observations'):
        mounts.append(bind(workspace/part, '/workspace/'+part))
    env = dict(PATH='/agent-bin:/usr/bin:/bin', HOME='/home/agent', CODEX_HOME='/home/agent/.codex',
        EMBODIEDEVO_WORKSPACE='/workspace', EMBODIEDEVO_SOCKET='/run/embodiedevo/session.sock',
        LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1',
        CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void', CODEX_CI='1',
        GATEWAY_PLACEHOLDER='unused', NO_PROXY='localhost,127.0.0.1')
    return dict(ociVersion='1.0.2', root=dict(path='rootfs', readonly=True), hostname='refate-evo-agent',
        process=dict(terminal=False, user=dict(uid=0,gid=0), args=['/usr/bin/python3','/inner.py'],
            env=[k+'='+v for k,v in env.items()], cwd='/workspace', noNewPrivileges=True,
            capabilities={k:[] for k in ('bounding','effective','inheritable','permitted','ambient')}),
        mounts=mounts, linux=dict(namespaces=[dict(type=n) for n in ('user','pid','network','ipc','uts','mount')],
            uidMappings=[dict(containerID=0,hostID=os.geteuid(),size=1)],
            gidMappings=[dict(containerID=0,hostID=os.getegid(),size=1)],
            maskedPaths=['/proc/acpi','/proc/kcore','/proc/keys','/proc/timer_list','/proc/sched_debug',
                         '/proc/scsi','/proc/driver/nvidia'],
            readonlyPaths=['/proc/bus','/proc/fs','/proc/irq','/proc/sys','/proc/sysrq-trigger']))


def stage(cfg, run, socket_dir, *, owner, synthetic):
    host_guard()
    need(synthetic is False, 'Synthetic staging cannot create an actor')
    need(owner == OWNER and type(synthetic) is bool, 'explicit owner and test/actual distinction required')
    scope = SHARED/'sessions'
    run = canonical(run, scope)
    socket_dir = canonical(socket_dir, run/'runtime')
    need((run/'workspace').is_dir(), 'original create_workspace must run first')
    for part in ('instruction.md','primitives','scripts','observations'):
        canonical(run/'workspace'/part, run/'workspace')
    for path in (RUNSC, PACKAGE/'bin/codex', PACKAGE/'bin/codex-code-mode-host', PACKAGE/'codex-resources'):
        need(path.exists(), 'missing existing isolated runtime dependency: '+str(path))
    command = original_command(cfg)
    for name in ('gateway','agent-bin','oci'):
        (run/name).mkdir(mode=0o700)
    home=socket_dir/'agent-home/.codex'
    home.mkdir(mode=0o700, parents=True)
    # A placeholder accesses the existing host proxy; no host credential enters the guest.
    config = ('model = "gt-6-as-a"\nmodel_provider = "gateway"\nmodel_reasoning_effort = "xhigh"\n'
        '[model_providers.gateway]\nname = "gateway"\nbase_url = "http://127.0.0.1:8080/v1"\n'
        'wire_api = "responses"\nenv_key = "GATEWAY_PLACEHOLDER"\n')
    with (home/'config.toml').open('x') as handle:
        handle.write(config)
    (home/'config.toml').chmod(0o600)
    for name in ('codex','codex-code-mode-host'):
        (run/'agent-bin'/name).touch(mode=0o755, exist_ok=False)
    write_new(run/'command.json', command)
    value = configuration(run, socket_dir, command)
    root=run/'oci/rootfs'; root.mkdir()
    (run/'oci/state').mkdir(mode=0o700)
    for name in ('usr','proc','sys','dev','tmp','workspace','home/agent','run/embodiedevo','gateway','agent-bin','codex-resources'):
        (root/name).mkdir(parents=True,exist_ok=True)
    for name in ('command.json','inner.py'):
        (root/name).touch()
    for name,target in (('bin','usr/bin'),('lib','usr/lib'),('lib64','usr/lib64')):
        (root/name).symlink_to(target)
    write_new(run/'oci/config.json', value)
    receipt = dict(schema='refate.evo.gvisor_cold_stage.v1', synthetic=synthetic, owner=owner,
        requested_model=cfg.agent.model, reasoning_effort=cfg.agent.reasoning_effort,
        original_prompt_sha256=hashlib.sha256(AUTHOR_PROMPT.encode()).hexdigest(),
        task=cfg.simulator.env_task, run=str(run), execution_entry_wired=True,
        network_or_model_or_native_started=False, data_qualification=False,
        resource_lifecycle_qualification=False)
    write_new(run/'COLD_STAGE.json', receipt)
    return receipt
