"""Original committed-byte Native observation persistence."""
import copy
import os
from pathlib import Path
import stat
import time
import numpy as np
from . import files as h

def policy_observation(value):
    return {k: v for k, v in value.items() if k in h.PUBLIC_KEYS}

def check_recording(task, recorder):
    recorder.writer.check()
    h.require(recorder.task is task and task._failure is None, 'Task/recorder failed or replaced')
    used = native_integer(task.control_steps, 'control_steps')
    h.require(recorder.control_index == 10 * used and recorder.capture_count == used + 1,
              'Native25 control/capture count mismatch')
    h.require(float(task.native.dt) == .004 and int(task.native.obs_manager.collect_interval) == 10,
              'Frozen physics/observation interval changed')


def record_folder(run, index):
    return Path(run) / ('initial' if index == 0 else f'macro_obs_{index:06d}')


class MainFiles(h.Files):
    def __init__(self, root, *, authorized_root, camera_budget):
        super().__init__(root, authorized_root=authorized_root)
        self.camera_budget = camera_budget

    def publish(self, name, value):
        path = self.path(name)
        raw = value if isinstance(value, bytes) else h.json_bytes(value)
        h.require(0 < len(raw) <= h.MAX_BYTES, 'Empty/oversized IPC artifact')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
        with temporary.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        finally:
            temporary.unlink()
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return {'path': name, 'sha256': h.sha(raw)}


    def bind_recorder(self, recorder, wait_for_publication):
        h.require(callable(wait_for_publication), 'Existing publication wait required')
        self.recorder = recorder
        self.wait_for_publication = wait_for_publication

    def publish_existing_regular_file(self, name, source, *, run):
        """Copy committed bytes exactly as in the deployed storage binding; no encoding."""
        run, source = Path(run), Path(source)
        h.require(source.is_absolute() and source.resolve() == source and source.parent.parent == run
                  and (source.parent.name == 'initial' or h.re.fullmatch(r'macro_obs_[0-9]{6}', source.parent.name)),
                  'Not a canonical current-run native observation file')
        h.require(source.name in {'observation.json', 'arrays.npz', 'image_stats.json',
                                 'cam_head.png', 'cam_left_wrist.png', 'cam_right_wrist.png'},
                  'Not a public recorded observation component')
        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        def stable(st):
            return st.st_dev, st.st_ino, st.st_mode, st.st_size, st.st_mtime_ns
        with os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
            before = os.fstat(stream.fileno())
            h.require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= h.MAX_BYTES,
                      'Not a bounded published observation')
            raw = stream.read(h.MAX_BYTES + 1)
            expected = h.sha(raw)
            h.require(len(raw) == before.st_size and stable(os.fstat(stream.fileno())) == stable(before)
                      and stable(source.stat(follow_symlinks=False)) == stable(before),
                      'Native observation changed while reading')
            with target.open('xb') as output:
                h.require(output.write(raw) == len(raw), 'Short native publication write')
                output.flush()
            copied = target.stat(follow_symlinks=False)
            h.require(stat.S_ISREG(copied.st_mode) and copied.st_size == before.st_size
                      and h.sha(target.read_bytes()) == expected
                      and stable(target.stat(follow_symlinks=False)) == stable(copied)
                      and stable(source.stat(follow_symlinks=False)) == stable(before),
                      'Published native byte copy or source changed')
        # No per-frame fsync/re-encoding. Recorder drain remains main's responsibility.
        return {'path': name, 'sha256': expected}

    def host_message(self, name):
        path = self.path(name, writer='host')
        h.require(path.is_file() and 0 < path.stat().st_size <= h.MAX_BYTES, 'Invalid host message')
        raw = path.read_bytes()
        h.require(len(raw) <= h.MAX_BYTES, 'Oversized host message')
        return h.strict_json(raw), h.sha(raw)


    def public_observation(self, sequence, observation):
        recorder = self.recorder
        used = native_integer(recorder.task.control_steps, 'native_used')
        self.wait_for_publication(recorder, used)
        check_recording(recorder.task, recorder)
        folder = record_folder(recorder.run, used)
        stem = f'artifacts/public_{sequence:06d}'
        refs = {name: self.publish_existing_regular_file(stem + '/' + name, folder / name, run=recorder.run)
                for name in ('observation.json', 'arrays.npz', 'cam_head.png',
                             'cam_left_wrist.png', 'cam_right_wrist.png')}
        directory = os.open(self.path(stem), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        structure = self.document(refs['observation.json'])
        tags = [structure['vision'][camera]['color'] for camera in
                ('cam_head', 'cam_left_wrist', 'cam_right_wrist')]
        descriptor = {'json': refs['observation.json'], 'arrays': refs['arrays.npz']}
        if all(isinstance(tag, dict) and 'png_key' in tag for tag in tags):
            descriptor.update(format='refate-observation-native25-v1',
                              pngs={name: ref for name, ref in refs.items() if name.endswith('.png')})
        else:
            h.require(all(isinstance(tag, dict) and 'npz_key' in tag for tag in tags),
                      'Mixed/unrecognized recorded RGB tags')
        # The raw recorder stays unchanged. Only the declared, measured camera
        # metadata is added to the teacher's public JSON view; RGB bytes are preserved.
        from . import camera as public_camera
        from ..evidence import document, file_ref as absolute_ref
        from .plain import plain
        camera_path = Path(recorder.run) / f'camera_{used:06d}.json'
        cameras = document(camera_path)['cameras']
        expected = public_camera.enrich(self.observation(descriptor), cameras, used, budget=self.camera_budget)
        h.require(equal_public(expected, policy_observation(observation)),
                  'Committed raw observation plus actual camera metadata differs from current state')
        public_structure = copy.deepcopy(structure)
        for camera in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
            for key in ('intrinsic_matrix', 'extrinsic_matrix', 'shape'):
                public_structure['vision'][camera][key] = plain(expected['vision'][camera][key])
        public_structure['additional_info'] = plain(expected.get('additional_info', {}))
        raw_descriptor = copy.deepcopy(descriptor)
        descriptor['json'] = self.publish(stem + '/public_observation.json', public_structure)
        self.publish(stem + '/public_camera_mapping.json', {
            'raw_recorded_descriptor': raw_descriptor, 'public_descriptor': descriptor,
            'actual_camera_snapshot': absolute_ref(camera_path),
            'public_camera_code': absolute_ref(Path(public_camera.__file__)),
            'native_used': used, 'RGB_reencoded': False, 'qualified': False,
        })
        h.require(equal_public(self.observation(descriptor), policy_observation(observation)),
                  'Derived public camera view differs from current observation')
        return descriptor


def equal_public(a, b):
    if isinstance(a, np.generic):
        a = a.item()
    if isinstance(b, np.generic):
        b = b.item()
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b))
    if isinstance(a, dict) or isinstance(b, dict):
        return (isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b)
                and all(equal_public(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))
                and len(a) == len(b) and all(equal_public(x, y) for x, y in zip(a, b)))
    return type(a) is type(b) and a == b


def native_integer(value, name):
    h.require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)),
              'Non-integer actual ' + name)
    return h.integer(int(value), name)
