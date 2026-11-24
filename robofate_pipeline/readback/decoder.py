"""Original native25 decoder with deployed archive compatibility edits."""
import io
import zipfile
from .contracts import CAMERAS, decode, need, path, verify_ref
PUBLIC_KEYS = {'instruction', 'vision', 'state', 'action', 'additional_info'}
MAX_BYTES = 64 * 1024 * 1024

def decode_archive(descriptor, read, *, archive_folder):
    """Narrow native25 lift of reviewed Host Files.observation; pure CPU only.

    NumPy/Pillow are loaded only here, never Host/Model/task modules. Complete NPZ
    and RGB validation precedes filtering, so public action arrays remain covered.
    """
    import numpy as np
    from PIL import Image
    need(set(descriptor) == {'format', 'json', 'arrays', 'pngs'}
         and descriptor['format'] == 'refate-observation-native25-v1', 'Native25 descriptor required')
    def raw(ref):
        data = verify_ref(ref, read, absolute=False)
        need(0 < len(data) <= MAX_BYTES, 'Oversized archive member')
        return data
    import re
    folder = path(archive_folder)
    need(folder.name == 'initial' or re.fullmatch(r'macro_obs_[0-9]{6}', folder.name),
         'Explicit initial/macro archive folder required')
    structure, data = decode(raw(descriptor['json'])), raw(descriptor['arrays'])
    need(type(structure.get('vision')) is dict and set(structure['vision']) == set(CAMERAS),
         'Exact archive camera set required')
    modes = set()
    for camera in CAMERAS:
        tag = structure['vision'][camera]['color']
        need(type(tag) is dict and set(tag) in ({'npz_key', 'shape', 'dtype'}, {'png_key', 'shape', 'dtype'}),
             'Unsupported/mixed RGB tag')
        mode = 'npz_key' if 'npz_key' in tag else 'png_key'
        modes.add(mode)
        need(tag['shape'] == [480, 640, 3] and tag['dtype'] == 'uint8', 'RGB shape/dtype differs')
        if mode == 'png_key':
            need(tag[mode] == camera + '.png', 'Camera PNG swap')
    need(len(modes) == 1, 'Mixed RGB representations unsupported')
    legacy_initial = modes == {'npz_key'}
    need(not legacy_initial or folder.name == 'initial', 'NPZ RGB is legacy-initial only')
    pngs = descriptor['pngs']
    need(set(pngs) == {camera + '.png' for camera in CAMERAS}, 'Exact RGB files required')
    parent = path(descriptor['json']['path'], absolute=False).parent
    for name, ref in pngs.items():
        need(ref['path'] == str(parent / name), 'Non-colocated PNG')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        need(len({v.filename for v in members}) == len(members)
             and sum(v.file_size for v in members) <= 2*MAX_BYTES, 'Duplicate/oversized expanded archive')
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        need(set(v.filename for v in members) == {key + '.npy' for key in archive.files},
             'Extra/non-array ZIP members')
        used, used_pngs = set(), set()
        def png_array(key):
            need(key in pngs and key not in used_pngs, 'Missing/reused RGB PNG')
            payload = raw(pngs[key])
            with Image.open(io.BytesIO(payload)) as image:
                need(image.format == 'PNG' and image.mode == 'RGB' and image.size == (640, 480),
                     'RGB mode/size differs')
                array = np.array(image)
            need(array.dtype == np.uint8 and array.shape == (480, 640, 3)
                 and np.isfinite(array).all(), 'RGB decode dtype/shape/nonfinite')
            used_pngs.add(key)
            return array
        def unpack(value, trail=()):
            if isinstance(value, dict):
                if 'png_key' in value:
                    need(set(value) == {'png_key', 'shape', 'dtype'}, 'Invalid PNG tag')
                    key = value['png_key']
                    need(key in pngs and key not in used_pngs and value['shape'] == [480, 640, 3]
                         and value['dtype'] == 'uint8', 'Wrong/reused RGB tag')
                    need(len(trail) == 3 and trail[0] == 'vision' and trail[2] == 'color'
                         and trail[1] in CAMERAS and key == trail[1] + '.png', 'PNG outside named camera')
                    return png_array(key)
                if 'npz_key' in value:
                    need(set(value) == {'npz_key', 'shape', 'dtype'}, 'Invalid NPZ tag')
                    key = value['npz_key']
                    need(key in archive.files and key not in used, 'Missing/reused NPZ array')
                    array = archive[key]
                    need(not array.dtype.hasobject and list(array.shape) == value['shape']
                         and str(array.dtype) == value['dtype'], 'NPZ shape/type differs')
                    need(array.dtype.kind not in 'fc' or np.isfinite(array).all(), 'Nonfinite array')
                    used.add(key)
                    return array.copy()
                return {key: unpack(item, trail + (key,)) for key, item in value.items()}
            if isinstance(value, list):
                return [unpack(item, trail + (i,)) for i, item in enumerate(value)]
            return value
        observation = unpack(structure)
        if legacy_initial:
            for camera in CAMERAS:
                array = observation['vision'][camera]['color']
                need(array.dtype == np.uint8 and array.shape == (480, 640, 3)
                     and np.isfinite(array).all(), 'Legacy RGB dtype/shape/nonfinite')
                need(np.array_equal(array, png_array(camera + '.png')), 'Legacy NPZ/PNG camera mismatch')
        need(used == set(archive.files) and used_pngs == set(pngs), 'Unreferenced archive arrays/RGB')
    need(set(observation) <= PUBLIC_KEYS | {'data_format_version', 'env_idx'}, 'Non-public archive field')
    public = {key: value for key, value in observation.items() if key in PUBLIC_KEYS}
    need(type(public.get('instruction')) is str and bool(public['instruction'])
         and type(public.get('state')) is dict and bool(public['state']), 'Missing public state/instruction')
    need(set(public['vision']) == set(CAMERAS), 'Camera set differs')
    for name in CAMERAS:
        need(public['vision'][name]['color'].dtype == np.uint8 and public['vision'][name]['color'].shape == (480, 640, 3), 'Decoded RGB contract')
    return public
