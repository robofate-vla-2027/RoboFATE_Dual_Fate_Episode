"""Isolate the pinned WebSocket ping RNG from simulation process-global RNG."""
import ast
import hashlib
from pathlib import Path
import random



_installed = None
_receipt = None


def install(connection, source_sha256):
    SOURCE_SHA256 = source_sha256
    global _installed, _receipt
    path = Path(connection.__file__)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('Unsupported WebSocket connection source')
    tree = ast.parse(raw)
    uses = [n for n in ast.walk(tree) if isinstance(n,ast.Name) and n.id=='random' and isinstance(n.ctx,ast.Load)]
    calls = [n for n in ast.walk(tree) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name)
             and n.value.id=='random' and n.attr=='getrandbits']
    if len(uses)!=1 or len(calls)!=1:
        raise ValueError('Transport RNG use is no longer the exact ping-only seam')
    if _installed is not None:
        if connection.random is not _installed:
            raise ValueError('Transport RNG identity changed after installation')
        return dict(_receipt)
    if connection.random is not random:
        raise ValueError('Unknown pre-existing transport RNG adapter')
    before = random.getstate()
    private = random.Random()
    if random.getstate()!=before:
        raise ValueError('Private transport generator changed global RNG')
    connection.random = private
    _installed = private
    _receipt = dict(schema='refate.transport_rng_isolation.v1',
        source=dict(path=str(path),sha256=SOURCE_SHA256),generator='process_private_random.Random',
        scope='default WebSocket ping payload only',source_file_changed=False,
        ping_implementation_changed=False,keepalive_configuration_changed=False,
        request_or_control_semantics_changed=False,simulation_rng_changed=False)
    return dict(_receipt)
