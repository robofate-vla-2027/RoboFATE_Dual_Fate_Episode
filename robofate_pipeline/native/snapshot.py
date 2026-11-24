"""Original runtime evidence capture, separate from simulation state."""
import hashlib
import os
from pathlib import Path
import sys
from ..evidence import need

def runtime_snapshot(task):
    """Node/runtime metadata, separate from shared physical, RNG and camera payloads."""
    raw = Path('/proc/self/stat').read_text()
    tail = raw.rsplit(')',1)[1].split()
    need(int(raw.split(' (',1)[0]) == os.getpid(), 'Caller PID changed')
    modules = {}
    for name in ('robofate_pipeline.recording.core','robofate_pipeline.recording.observation_io',
                 'robofate_pipeline.recording.pt_capture','robofate_pipeline.recording.rtx_profile',
                 'simulators.robodojo'):
        module = sys.modules[name]
        path = Path(module.__file__)
        modules[name] = {'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    return {'schema':'robofate.native-runtime.v1', 'pid':os.getpid(), 'start_ticks':int(tail[19]),
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'task':task.task_name, 'native_used':task.control_steps, 'native_limit':task.native.step_lim,
            'actual_imports':modules, 'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'scope':'runtime metadata only; shared physics/RNG/camera records remain separate'}
