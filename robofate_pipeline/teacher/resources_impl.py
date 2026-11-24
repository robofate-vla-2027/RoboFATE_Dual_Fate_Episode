"""Original owned worker resources; instantiated by the runtime factory."""
import hashlib
import json
import os
from pathlib import Path
import socketserver
import sys
import threading
import time
from urllib.parse import urlsplit

def identity(pid):
    stat = Path(f'/proc/{pid}/stat').read_text().rpartition(')')[2].split()
    return dict(pid=pid, start_ticks=int(stat[19]),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


class AgentResources:
    def __init__(self, specification, run):
        stage.host_guard()
        stage.need(specification.get('data_kind') == 'actual'
                   and specification.get('runtime_binding', {}).get('data_kind') == 'actual',
                   'Synthetic fixtures cannot create an actual actor')
        self.context = specification['refate_takeover']
        stage.need(self.context['active_chat_id'] == stage.OWNER
                   and self.context['chat'] == stage.CHAT, 'Wrong execution owner')
        stage.need(specification.get('seconds') == 3600, 'Original 3600s wall limit required')
        self.run = stage.canonical(run, stage.SHARED / 'sessions')
        stage.need(self.run.name == self.context['session'], 'Session/run mismatch')
        self.process = self.created = self.server = self.thread = self.gate = None
        self.prepared = False
        self.socket_identity = None
        self.timing_sink = None
        self.identity = dict(container_name='refate-teacher', session=self.context['session'])

    def prepare(self, cfg, run, socket_dir, phase):
        stage.need(not self.prepared and Path(run) == self.run and phase == 'test',
                   'One Test actor per original worker; no automatic restart')
        stage.need(os.environ.get('MODEL_UPSTREAM') == stage.MODEL_UPSTREAM,
                   'Only existing authorized model forwarding route is supported')
        stage.stage(cfg, run, socket_dir, owner=stage.OWNER, synthetic=False)
        proxy = stage.transport()
        from . import timing, provider_safety as safety
        self.timing_sink = timing.JsonlSink(run/'agent/request_timing.jsonl')
        handlers, measured = timing.prepare_private_module(proxy, self.timing_sink)
        self.gate = safety.Gate(run/'agent/provider_safety.json')
        guarded = safety.install(proxy, handlers, measured, self.gate)
        handler = type('ModelOnly', (guarded('model'),), dict(
            upstream=urlsplit(stage.MODEL_UPSTREAM), allowed_prefixes=('/v1/',), key_file=None))
        previous = Path.cwd()
        try:
            os.chdir(run/'gateway')
            self.server = socketserver.ThreadingUnixStreamServer('model.sock', handler)
            self.server.daemon_threads = True
            self.server.block_on_close = False
            os.chmod('model.sock', 0o600)
            st = Path('model.sock').lstat()
            self.socket_identity = (st.st_dev, st.st_ino)
        finally:
            os.chdir(previous)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.config_sha = hashlib.sha256((run/'oci/config.json').read_bytes()).hexdigest()
        self.prepared = True
        return self.identity['container_name'], [sys.executable, '-B', '-m', 'robofate_pipeline.teacher.supervisor',
            str(run), self.config_sha, str(os.getpid()),
            stage.policy_ref['path'], stage.policy_ref['sha256']]

    def require_open(self):
        if self.gate is not None:
            self.gate.require_open()

    def capture_created(self, process):
        stage.need(self.prepared and self.process is None, 'Actor launch was not prepared')
        self.process = process
        self.created = identity(process.pid)
        receipt = self.run/'oci/OWNED_RUNSC.json'
        deadline = time.monotonic() + 20
        while True:
            stage.need(time.monotonic() < deadline, 'gVisor creation identity timeout')
            try:
                value = json.loads(receipt.read_text())
                break
            except (FileNotFoundError, json.JSONDecodeError):
                stage.need(process.poll() is None, 'gVisor exited before complete creation identity; preserve logs')
                time.sleep(0.05)
        stage.need(value['supervisor_pid'] == process.pid and value['parent_pid'] == os.getpid()
                   and value['config_sha256'] == self.config_sha, 'Wrong actual gVisor creation')
        stage.write_new(self.run/'agent/owned_process.json', dict(
            execution_owner=stage.OWNER, session=self.context['session'], backend='gvisor-private-netns',
            supervisor=self.created, worker=identity(os.getpid()), runsc=value,
            teacher_system='EmbodiedEvo public-recipe primitive baseline', qualification=False))

    def cleanup(self):
        error = None
        try:
            if self.process is not None:
                if self.process.poll() is None:
                    stage.need(identity(self.process.pid) == self.created, 'Supervisor identity changed')
                    self.process.terminate()
                raw_exit = self.process.wait(timeout=25)
                path = self.run/'oci/LIFECYCLE.json'
                stage.need(path.is_file(), 'Actual gVisor lifecycle missing')
                result = json.loads(path.read_text())
                stage.need(result['run_root'] == str(self.run)
                           and result['config_sha256'] == self.config_sha
                           and result['supervisor_pid'] == self.process.pid, 'Wrong lifecycle scope')
                wait_record = dict(supervisor_raw_exit=raw_exit, lifecycle=str(path),
                                   owned_cleanup_complete=result.get('owned_cleanup_complete'),
                                   raw_scientific_result_unchanged=True)
                wait_path = self.run/'agent/worker_wait.json'
                if wait_path.exists():
                    stage.need(json.loads(wait_path.read_text()) == wait_record, 'Conflicting actual wait record')
                else:
                    stage.write_new(wait_path, wait_record)
                stage.need(result.get('owned_cleanup_complete') is True, 'Owned sandbox cleanup incomplete')
        except Exception as exc:
            error = exc
        finally:
            if self.server is not None:
                self.server.shutdown()
                self.server.server_close()
                self.server = None
            if self.thread is not None:
                self.thread.join(timeout=5)
                stage.need(not self.thread.is_alive(), 'Model listener cleanup incomplete')
                self.thread = None
            if self.timing_sink is not None:
                self.timing_sink.close()
                self.timing_sink = None
            if self.socket_identity is not None:
                path = self.run/'gateway/model.sock'
                st = path.lstat()
                stage.need((st.st_dev, st.st_ino) == self.socket_identity, 'Model socket identity changed')
                path.unlink()
                self.socket_identity = None
        if error is not None:
            raise error

    def stop_agent_client(self, process):
        if process is None or process.poll() is not None:
            return
        stage.need(process is self.process and identity(process.pid) == self.created,
                   'Unowned supervisor; no signal sent')
        process.terminate()
        process.wait(timeout=25)


def agent_command(cfg, run, socket_dir, phase, *, resources):
    return resources.prepare(cfg, run, socket_dir, phase)
