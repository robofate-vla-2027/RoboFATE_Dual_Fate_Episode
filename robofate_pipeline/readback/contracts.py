"""Original pure archive contracts, without historical collection bindings."""
import hashlib
import json
import math
from pathlib import PurePosixPath
import re
CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')

def need(ok, message):
    if not ok:
        raise ValueError(message)


def integer(value, expected=None, minimum=0):
    need(type(value) is int and value >= minimum
         and (expected is None or value == expected), 'Invalid exact integer/count')
    return value


def finite(value):
    need(type(value) in (int, float) and math.isfinite(value), 'Nonfinite physical time')
    return value


def exact(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def decode(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, 'Duplicate JSON key')
            out[key] = value
        return out
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def path(value, *, absolute=True):
    need(type(value) is str and bool(value), 'Missing path')
    p = PurePosixPath(value)
    need(str(p) == value and p.is_absolute() is absolute
         and '..' not in p.parts and value != '.', 'Noncanonical path')
    return p


def file_ref(value, *, absolute=True):
    need(type(value) is dict and set(value) == {'path', 'sha256'}, 'Exact FileRef required')
    path(value['path'], absolute=absolute)
    need(type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256']), 'Invalid hash')
    return value


def verify_ref(value, read, *, absolute=True):
    file_ref(value, absolute=absolute)
    raw = read(value['path'])
    need(type(raw) is bytes and sha(raw) == value['sha256'], 'Reference bytes changed: ' + value['path'])
    return raw
