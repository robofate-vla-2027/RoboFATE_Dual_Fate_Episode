"""Lossless synchronous observation I/O; dense RGB uses PNG references only."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import zipfile

import numpy as np
from PIL import Image


def _safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', value):
        raise ValueError('Unsafe observation filename or camera name')
    return value


def _fsync_file(stream):
    stream.flush()
    os.fsync(stream.fileno())


def _fsync_dir(folder):
    fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write(payload)
        _fsync_file(stream)
    temporary.replace(path)


def _save_png(path, frame):
    with path.open('xb') as stream:
        Image.fromarray(frame).save(stream, format='PNG', compress_level=1)
        _fsync_file(stream)


def _save_npz(path, arrays, dense):
    with path.open('xb') as stream:
        if dense:
            with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=1, allowZip64=True) as archive:
                for key, array in arrays.items():
                    with archive.open(key + '.npy', 'w', force_zip64=True) as member:
                        np.lib.format.write_array(member, array, allow_pickle=False)
        else:
            np.savez_compressed(stream, **arrays)
        _fsync_file(stream)


def _scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError('Unsupported or nonfinite observation scalar: ' + type(value).__name__)


def save_observation(folder, obs):
    """Save one boundary, joining three PNG workers before metadata publication.

    Dense children and exact macro_obs_NNNNNN folders select png_key RGB.
    Initial and other folders retain the legacy npz_key schema and compression.
    Array dtype/shape/bytes and legacy JSON scalar semantics are preserved.
    A failed write propagates, retains diagnostic files, and unpublishes the
    observation manifest; callers must not continue the physical trajectory.
    """
    folder = Path(folder)
    dense = folder.parent.name == 'dense' or re.fullmatch(r'macro_obs_[0-9]{6}', folder.name) is not None
    if not isinstance(obs, dict) or not isinstance(obs.get('vision'), dict) or len(obs['vision']) != 3:
        raise ValueError('Exactly three RGB camera observations are required')
    frames, stats = {}, {}
    for camera, data in obs['vision'].items():
        _safe_name(camera)
        frame = data.get('color') if isinstance(data, dict) else None
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3 or min(frame.shape[:2]) < 1:
            raise ValueError('Malformed RGB observation')
        frame = np.array(frame, copy=True, order='C', subok=False)
        frame.setflags(write=False)
        frames[camera] = frame
        stats[camera] = {'shape': list(frame.shape), 'mean': float(frame.mean()),
                         'std': float(frame.std()), 'sha256': hashlib.sha256(frame.tobytes()).hexdigest()}
        stats[camera]['constant_pixel_values'] = stats[camera]['std'] == 0
    arrays = {}

    def encode(value, trail=()):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError('Object array in observation')
            rgb = len(trail) == 3 and trail[0] == 'vision' and trail[2] == 'color'
            array = frames[trail[1]] if rgb else np.array(value, copy=True, order='K', subok=False)
            metadata = {'shape': list(array.shape), 'dtype': str(array.dtype)}
            if dense and rgb:
                return {'png_key': trail[1] + '.png', **metadata}
            key = f'array_{len(arrays):04d}'
            arrays[key] = array
            return {'npz_key': key, **metadata}
        if isinstance(value, dict):
            if len({str(key) for key in value}) != len(value):
                raise ValueError('Observation dictionary key collision')
            return {str(key): encode(item, trail + (str(key),)) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item, trail + (index,)) for index, item in enumerate(value)]
        return _scalar(value)

    structure = encode(obs)
    folder.mkdir(parents=True)
    try:
        # Context-manager shutdown joins every worker even when one write fails.
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix='observation_png') as pool:
            futures = [pool.submit(_save_png, folder / (camera + '.png'), frame)
                       for camera, frame in frames.items()]
            _save_npz(folder / 'arrays.npz', arrays, dense)
            for future in futures:
                future.result()
        _fsync_dir(folder)
        _write_json(folder / 'image_stats.json', stats)
        _write_json(folder / 'observation.json', structure)
        _fsync_dir(folder)
        _fsync_dir(folder.parent)
    except BaseException as primary:
        try:
            (folder / 'observation.json').unlink(missing_ok=True)
            _fsync_dir(folder)
        except Exception as cleanup_error:
            primary.add_note('Manifest cleanup also failed: ' + repr(cleanup_error))
        raise
    return str(folder)


def _owned_file(folder, name, suffix):
    _safe_name(name)
    if not name.endswith(suffix):
        raise ValueError('Unexpected observation file suffix')
    path = folder / name
    if path.is_symlink() or path.resolve(strict=True).parent != folder or not path.is_file():
        raise ValueError('Observation reference is not an owned regular file')
    return path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def decode_observation(folder):
    """Read npz_key and png_key tags; reject unsafe paths or mismatched metadata."""
    folder = Path(folder).resolve(strict=True)
    manifest = _owned_file(folder, 'observation.json', '.json')
    structure = json.loads(manifest.read_text(encoding='utf-8'), object_pairs_hook=_unique_object)
    npz_path = _owned_file(folder, 'arrays.npz', '.npz')
    with np.load(npz_path, allow_pickle=False) as arrays:
        if any(not re.fullmatch(r'array_[0-9]{4,}', key) for key in arrays.files):
            raise ValueError('Invalid NPZ member key')

        def decode(value):
            if isinstance(value, dict) and ('npz_key' in value or 'png_key' in value):
                tag = 'npz_key' if 'npz_key' in value else 'png_key'
                if set(value) != {tag, 'shape', 'dtype'}:
                    raise ValueError('Malformed observation array reference')
                shape, dtype, key = value['shape'], value['dtype'], value[tag]
                if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape) or not isinstance(dtype, str) or not isinstance(key, str):
                    raise ValueError('Invalid shape, dtype or key type')
                if tag == 'png_key':
                    if dtype != 'uint8' or len(shape) != 3 or shape[-1] != 3 or min(shape[:2]) < 1:
                        raise ValueError('Invalid RGB PNG metadata')
                    with Image.open(_owned_file(folder, key, '.png')) as image:
                        if image.format != 'PNG' or image.mode != 'RGB' or image.size != (shape[1], shape[0]):
                            raise ValueError('PNG type, channels or shape mismatch')
                        array = np.array(image, copy=True)
                else:
                    if not re.fullmatch(r'array_[0-9]{4,}', key) or key not in arrays.files:
                        raise ValueError('Invalid or missing NPZ key')
                    array = arrays[key].copy()
                if array.dtype.hasobject or list(array.shape) != shape or str(array.dtype) != dtype:
                    raise ValueError('Observation array shape or dtype mismatch')
                return array
            if isinstance(value, dict):
                return {key: decode(item) for key, item in value.items()}
            if isinstance(value, list):
                return [decode(item) for item in value]
            return _scalar(value)

        return decode(structure)


# New sources keep the legacy decoder and wire representation, but commit only
# native camera boundaries through a bounded, ordered writer.
class OrderedObservationWriter:
    def __init__(self, root, capacity=3, workers=3, encoder=None, committer=None):
        import threading
        from collections import deque
        self.root = Path(root)
        self.capacity = capacity
        self.condition = threading.Condition(threading.RLock())
        self.pending = deque()
        self.outstanding = 0
        self.submitted = 0
        self.committed = 0
        self.last_committed = -1
        self.first_error = None
        self.cleanup_errors = []
        self.closing = False
        self.closed = False
        self.touched = set()
        self.streams = {}
        self.metrics = {'observation_copy': 0.0, 'queue_wait': 0.0,
                        'encode_write': 0.0, 'final_drain_persistence': 0.0}
        self.encode_seconds = 0.0
        self.pending_worker_cleanup = False
        self.publication_in_flight = None
        self.publication_in_flight_at_failure = None
        self.max_outstanding = 0
        self.encoder = encoder or self._encode
        self.committer = committer or self._commit
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='native25_encode')
        self.thread = threading.Thread(target=self._run, name='native25_ordered_writer', daemon=True)
        self.thread.start()

    def fail(self, error):
        with self.condition:
            if self.first_error is None:
                self.first_error = error
                self.publication_in_flight_at_failure = self.publication_in_flight
            elif error is not self.first_error:
                self.cleanup_errors.append(repr(error))
            self.condition.notify_all()

    def check(self):
        with self.condition:
            if self.first_error is not None:
                raise self.first_error
            if self.closing:
                raise RuntimeError('Observation writer is closing; action admission forbidden')

    def admit(self, operation, *args, **kwargs):
        # This lock is also the writer-error latch: queued native actions and
        # actuator submissions cannot pass a confirmed failure.
        with self.condition:
            self.check()
            return operation(*args, **kwargs)

    @staticmethod
    def _freeze(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError('Object array in observation')
            out = np.array(value, copy=True, order='C', subok=False)
            out.setflags(write=False)
            return out
        if isinstance(value, dict):
            if len({str(k) for k in value}) != len(value):
                raise ValueError('Observation dictionary key collision')
            return {str(k): OrderedObservationWriter._freeze(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [OrderedObservationWriter._freeze(v) for v in value]
        return _scalar(value)

    @staticmethod
    def _encode(obs):
        import io
        if obs is None:
            return None
        if set(obs.get('vision', {})) != {'cam_head', 'cam_left_wrist', 'cam_right_wrist'}:
            raise ValueError('Native25 requires the three original camera aliases')
        arrays, files, stats = {}, {}, {}
        for name, data in obs['vision'].items():
            frame = data['color']
            if frame.dtype != np.uint8 or frame.shape != (480, 640, 3):
                raise ValueError('Native25 RGB shape or dtype changed')
            std = float(frame.std())
            # Frame freshness is checked by pt_capture, not pixel contrast.
            stats[name] = {'shape': list(frame.shape), 'mean': float(frame.mean()),
                           'std': std, 'sha256': hashlib.sha256(frame.tobytes()).hexdigest(),
                           'constant_pixel_values': std == 0}
            output = io.BytesIO()
            Image.fromarray(frame).save(output, format='PNG', compress_level=1)
            files[name + '.png'] = output.getvalue()
        def encode(value, trail=()):
            if isinstance(value, np.ndarray):
                metadata = {'shape': list(value.shape), 'dtype': str(value.dtype)}
                if len(trail) == 3 and trail[0] == 'vision' and trail[2] == 'color':
                    return {'png_key': trail[1] + '.png', **metadata}
                key = f'array_{len(arrays):04d}'
                arrays[key] = value
                return {'npz_key': key, **metadata}
            if isinstance(value, dict):
                return {k: encode(v, trail + (k,)) for k, v in value.items()}
            if isinstance(value, list):
                return [encode(v, trail + (i,)) for i, v in enumerate(value)]
            return _scalar(value)
        structure = encode(obs)
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=1, allowZip64=True) as archive:
            for key, array in arrays.items():
                with archive.open(key + '.npy', 'w', force_zip64=True) as member:
                    np.lib.format.write_array(member, array, allow_pickle=False)
        files['arrays.npz'] = output.getvalue()
        files['image_stats.json'] = (json.dumps(stats, allow_nan=False) + '\n').encode()
        files['observation.json'] = (json.dumps(structure, allow_nan=False) + '\n').encode()
        return files

    def submit(self, index, folder, obs, documents, appends):
        import time
        started = time.perf_counter()
        with self.condition:
            while self.outstanding >= self.capacity:
                self.check()
                self.condition.wait(.05)
            self.check()
            if index != self.submitted:
                raise ValueError('Native boundary submission gap')
            self.outstanding += 1
            self.max_outstanding = max(self.max_outstanding, self.outstanding)
        self.metrics['queue_wait'] += time.perf_counter() - started
        try:
            started = time.perf_counter()
            frozen = self._freeze(obs) if obs is not None else None
            documents = json.loads(json.dumps(documents, allow_nan=False))
            appends = json.loads(json.dumps(appends, allow_nan=False))
            self.metrics['observation_copy'] += time.perf_counter() - started
            def encode():
                begin = time.perf_counter()
                try:
                    return self.encoder(frozen)
                finally:
                    with self.condition:
                        self.encode_seconds += time.perf_counter() - begin
            future = self.pool.submit(encode)
            def completed(result):
                try:
                    error = result.exception()
                    if error is not None:
                        self.fail(error)
                except BaseException as error:
                    self.fail(error)
            future.add_done_callback(completed)
            with self.condition:
                self.check()
                self.pending.append((index, str(folder), future, documents, appends))
                self.submitted += 1
                self.condition.notify_all()
        except BaseException as error:
            self.fail(error)
            raise

    def _new_file(self, path, content):
        path = Path(path)
        if not path.is_relative_to(self.root):
            raise ValueError('Writer file outside owned source')
        with path.open('xb') as stream:
            stream.write(content)
        self.touched.add(path)

    def _commit(self, item, files):
        index, folder, future, documents, appends = item
        folder = Path(folder)
        if files is not None:
            folder.mkdir()
            for name, content in files.items():
                if name != 'observation.json':
                    self._new_file(folder / name, content)
        elif not (folder / 'observation.json').is_file():
            raise ValueError('Reused initial observation was not published')
        for name, value in documents.items():
            self._new_file(self.root / name, (json.dumps(value, allow_nan=False) + '\n').encode())
        def publish():
            if files is not None:
                self._new_file(folder / 'observation.json', files['observation.json'])
            for name, records in appends.items():
                path = self.root / name
                if path not in self.streams:
                    self.streams[path] = path.open('x', encoding='utf-8')
                    self.touched.add(path)
                stream = self.streams[path]
                for record in records:
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                stream.flush()
        return publish

    def _run(self):
        import time
        try:
            while True:
                with self.condition:
                    while not self.pending and not self.closing and self.first_error is None:
                        self.condition.wait(.05)
                    if self.first_error is not None:
                        return
                    if not self.pending and self.closing:
                        return
                    item = self.pending.popleft()
                encoded = item[2].result()
                with self.condition:
                    if self.first_error is not None:
                        return
                begin = time.perf_counter()
                publish = self.committer(item, encoded)
                with self.condition:
                    if self.first_error is not None:
                        return
                    self.publication_in_flight = item[0]
                # An admitted file publication can finish after failure, like
                # any in-flight operation. Do not hold action/error admission
                # across filesystem I/O; the failed source never qualifies.
                if publish is not None:
                    publish()
                with self.condition:
                    self.metrics['encode_write'] += time.perf_counter() - begin
                    self.last_committed = item[0]
                    self.committed += 1
                    self.outstanding -= 1
                    self.publication_in_flight = None
                    self.condition.notify_all()
        except BaseException as error:
            self.fail(error)

    def close(self):
        import time
        begin = time.perf_counter()
        deadline = begin + 120.0
        with self.condition:
            self.closing = True
            self.condition.notify_all()
            while self.committed < self.submitted and self.first_error is None:
                if time.perf_counter() >= deadline:
                    self.fail(TimeoutError('Native writer drain exceeded 120s; infrastructure interruption'))
                    break
                self.condition.wait(.05)
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            self.fail(RuntimeError('Writer did not stop; source cannot be published'))
        try:
            self.pending_worker_cleanup = self.first_error is not None
            self.pool.shutdown(wait=not self.pending_worker_cleanup, cancel_futures=True)
        except BaseException as error:
            self.fail(error)
        for stream in self.streams.values():
            try:
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
            except BaseException as error:
                self.fail(error)
        # Durability is required once at the closed-source boundary, not on
        # every image or low-level control submission.
        for path in sorted(self.touched):
            try:
                with path.open('rb') as stream:
                    os.fsync(stream.fileno())
            except BaseException as error:
                self.fail(error)
        for folder in sorted({p.parent for p in self.touched} | {self.root}):
            try:
                _fsync_dir(folder)
            except BaseException as error:
                self.fail(error)
        self.closed = not self.thread.is_alive()
        self.metrics['final_drain_persistence'] += time.perf_counter() - begin
        if self.first_error is not None:
            raise self.first_error

    def receipt(self):
        return {'status': 'failed' if self.first_error else ('closed' if self.closed else 'open'),
                'failed': self.first_error is not None, 'submitted_boundaries': self.submitted,
                'committed_boundaries': self.committed, 'last_committed_boundary': self.last_committed,
                'primary_error': repr(self.first_error) if self.first_error else None,
                'cleanup_errors': list(self.cleanup_errors), 'max_outstanding': self.max_outstanding,
                'capacity': self.capacity, 'publication_in_flight':self.publication_in_flight,
                'publication_in_flight_at_failure':self.publication_in_flight_at_failure,
                'pending_worker_cleanup':self.pending_worker_cleanup, 'phase_seconds': dict(self.metrics),
                'encoder_worker_cumulative_seconds': self.encode_seconds}


def native_observation_index(run):
    """Resolve true native boundaries without reading/duplicating image payloads."""
    run = Path(run).resolve(strict=True)
    contract = run / 'recording_contract.json'
    if contract.exists():
        if json.loads(contract.read_text()).get('schema') != 'refate-native25-v1':
            raise ValueError('Unknown native recording contract')
        raw = (run / 'native_observations.jsonl').read_text()
        if not raw.endswith('\n'):
            raise ValueError('Uncommitted native observation index')
        records = [json.loads(line) for line in raw.splitlines()]
        for i, row in enumerate(records):
            expected = run / ('initial' if i == 0 else f'macro_obs_{i:06d}')
            if row['index'] != i or row['control_index'] != 10*i or row['observation_dir'] != str(expected):
                raise ValueError('Native observation index/path/control mismatch')
        return records
    count = json.loads((run / 'result.json').read_text())['native_steps']
    return [{'index': i, 'control_index': 10*i, 'observation_dir': str(run/f'macro_obs_{i:06d}'),
             'legacy_native_boundary_view': True} for i in range(count+1)]


def standard_joint_observation(obs):
    """Author v2.1 field order, not future targets or actuator-command relabeling.

    Native gripper state uses its exposed normalized control value; measured
    gripper joint feedback remains separately recorded in each low-level row.
    """
    names = ('left_arm_joint_state', 'left_ee_joint_state',
             'right_arm_joint_state', 'right_ee_joint_state')
    arrays = [np.asarray(obs['state'][key], dtype=np.float32) for key in names]
    if [a.shape for a in arrays] != [(6,), (1,), (6,), (1,)]:
        raise ValueError('ARX X5 joint/gripper dimensions differ from author mapping')
    state = np.concatenate(arrays)
    if not np.isfinite(state).all():
        raise ValueError('Nonfinite official-view joint state')
    result = {'observation.state': state}
    for source, target in (('cam_head', 'cam_high'), ('cam_left_wrist', 'cam_left_wrist'),
                           ('cam_right_wrist', 'cam_right_wrist')):
        image = obs['vision'][source]['color']
        if image.dtype != np.uint8 or image.shape != (480, 640, 3):
            raise ValueError('Official-view RGB shape/dtype mismatch')
        result['observation.images.' + target] = image
    return result
