"""Original bounded IPC decoder and exclusive durable publication."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import zipfile
import numpy as np
from ..recording.takeover import ContractError, require
from ..recording.teacher import PUBLIC_KEYS, integer
MAX_BYTES = 256 * 1024 * 1024

def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(data):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'Duplicate JSON key')
            result[k] = v
        return result
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ContractError('Nonfinite JSON')))


class Files:
    def __init__(self, root, *, authorized_root):
        ROOT = Path(authorized_root)
        require(ROOT.is_absolute() and ROOT.resolve(strict=True) == ROOT and ROOT.is_dir(),
                'Explicit canonical authorized root required')
        self.root = Path(root)
        require(self.root.is_absolute() and self.root.resolve() == self.root
                and self.root.is_relative_to(ROOT), 'IPC must be canonical under the authorized project root')
        require(self.root.name == 'teacher_ipc' and self.root.is_dir(), 'Missing main-provisioned teacher_ipc')

    def path(self, relative, *, writer='main'):
        p = Path(relative)
        require(not p.is_absolute() and '..' not in p.parts and p.parts, 'Invalid relative IPC path')
        host_name = p.name in {'host_claim.json', 'finish.json'} or re.fullmatch(r'response_[0-9]{6}\.json', p.name)
        if writer == 'host':
            require(len(p.parts) == 1 and bool(host_name), 'Host may publish only its protocol files')
        else:
            require(not (len(p.parts) == 1 and host_name), 'Main evidence cannot refer to host output')
        path = self.root / p
        require(path.resolve() == path, 'IPC symlink is forbidden')
        return path

    def read(self, ref):
        require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'Expected {path, sha256} file reference')
        require(isinstance(ref['sha256'], str) and re.fullmatch('[0-9a-f]{64}', ref['sha256']), 'Invalid SHA256')
        path = self.path(ref['path'])
        require(path.is_file() and 0 < path.stat().st_size <= MAX_BYTES, 'Missing/empty/oversized main artifact')
        raw = path.read_bytes()
        require(len(raw) <= MAX_BYTES and sha(raw) == ref['sha256'], 'Main artifact hash mismatch')
        return raw

    def document(self, ref):
        return strict_json(self.read(ref))

    def message(self, name):
        path = self.path(name)
        require(path.is_file() and 0 < path.stat().st_size <= MAX_BYTES, 'Invalid main message')
        raw = path.read_bytes()
        return strict_json(raw), sha(raw)

    def publish(self, name, data):
        path = self.path(name, writer='host')
        raw = json_bytes(data)
        temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
        with temporary.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Atomic create-without-overwrite, including the initial host claim.
            os.link(temporary, path)
        finally:
            temporary.unlink()
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return sha(raw)

    def observation(self, descriptor):
        require(isinstance(descriptor, dict), 'Missing observation descriptor')
        native25 = descriptor.get('format') == 'refate-observation-native25-v1'
        require(set(descriptor) == ({'format', 'json', 'arrays', 'pngs'} if native25 else {'json', 'arrays'}),
                'Observation requires legacy NPZ or pinned native25 PNG descriptor')
        pngs = descriptor['pngs'] if native25 else {}
        require(isinstance(pngs, dict) and (len(pngs) == 3 if native25 else not pngs),
                'Native25 requires exactly three pinned native RGB files')
        png_parent = Path(descriptor['json']['path']).parent
        for name, ref in pngs.items():
            require(isinstance(name, str) and Path(name).name == name
                    and name.endswith('.png') and isinstance(ref, dict)
                    and ref.get('path') == str(png_parent / name), 'Invalid colocated native PNG reference')
        structure = self.document(descriptor['json'])
        raw = self.read(descriptor['arrays'])
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            require(sum(info.file_size for info in archive.infolist()) <= 2 * MAX_BYTES,
                    'Oversized expanded observation')
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            used = set()
            used_pngs = set()
            def decode(value):
                if isinstance(value, dict):
                    if 'png_key' in value:
                        require(native25 and set(value) == {'png_key', 'shape', 'dtype'},
                                'Invalid native PNG tag')
                        key = value['png_key']
                        require(isinstance(key, str) and key in pngs and key not in used_pngs
                                and value['shape'] == [480, 640, 3] and value['dtype'] == 'uint8',
                                'Invalid/reused/missing native RGB PNG')
                        data = self.read(pngs[key])
                        require(data[:16] == b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
                                and data[16:24] == b'\x00\x00\x02\x80\x00\x00\x01\xe0',
                                'PNG header is not original 640x480')
                        from PIL import Image
                        with Image.open(io.BytesIO(data)) as image:
                            require(image.format == 'PNG' and image.mode == 'RGB'
                                    and image.size == (640, 480), 'Native PNG mode/size changed')
                            array = np.array(image, dtype=np.uint8)
                        require(array.shape == (480, 640, 3), 'Native PNG decode failed')
                        used_pngs.add(key)
                        return array
                    if 'npz_key' in value:
                        require(set(value) == {'npz_key', 'shape', 'dtype'}, 'Invalid author array tag')
                        key = value['npz_key']
                        require(key in archive.files, 'Missing observation array')
                        array = archive[key]
                        require(not array.dtype.hasobject and list(array.shape) == value['shape']
                                and str(array.dtype) == value['dtype'], 'Observation array metadata mismatch')
                        used.add(key)
                        return array.copy()
                    return {k: decode(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [decode(v) for v in value]
                return value
            observation = decode(structure)
            require(used == set(archive.files), 'Unreferenced observation arrays')
            require(used_pngs == set(pngs), 'Unreferenced native RGB files')
        if native25:
            # Decode and validate the complete original archive first. Only the
            # author's public fields cross into online teacher observation.
            require(isinstance(observation, dict)
                    and set(observation) <= PUBLIC_KEYS | {'action', 'data_format_version', 'env_idx'},
                    'Unexpected field in original native25 recording')
            observation = {key: value for key, value in observation.items() if key in PUBLIC_KEYS}
        require(isinstance(observation, dict) and set(observation) <= PUBLIC_KEYS,
                'Non-public observation field')
        require(isinstance(observation.get('state'), dict) and observation['state'], 'Missing public state')
        require(isinstance(observation.get('instruction'), str) and observation['instruction'], 'Missing instruction')
        cameras = observation.get('vision')
        require(isinstance(cameras, dict) and cameras, 'Missing real camera observation')
        if native25:
            require(set(cameras) == {'cam_head', 'cam_left_wrist', 'cam_right_wrist'},
                    'Native25 camera set changed')
            require(all(isinstance(v, dict) and isinstance(v.get('color'), dict)
                        and 'png_key' in v['color'] for v in structure['vision'].values()),
                    'Each native25 camera must bind its own recorded PNG')
        for camera in cameras.values():
            require(isinstance(camera, dict) and isinstance(camera.get('color'), np.ndarray), 'Missing RGB array')
            a = camera['color']
            require(a.dtype == np.uint8 and a.ndim == 3 and a.shape[-1] == 3, 'Invalid native RGB shape/type')
            require(not native25 or a.shape == (480, 640, 3), 'Native RGB resolution changed')
        return observation
