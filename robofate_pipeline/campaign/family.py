"""Original task-content family normalization."""
from copy import deepcopy
import hashlib
from ..evidence import canonical

def family(layout):
    if type(layout) is not dict or not layout:
        raise ValueError('Nonempty full layout required')
    canonical(layout)
    normalized = deepcopy(layout)
    for kind in ('Rigid', 'Geometry'):
        if kind not in normalized:
            continue
        collection = normalized[kind]
        if type(collection) is not dict:
            raise ValueError('Object category must be a mapping')
        for objects in collection.values():
            if type(objects) is not list:
                raise ValueError('Object rows must be a list')
            for row in objects:
                if type(row) is not dict or 'default_pos' not in row or 'default_ori' not in row:
                    raise ValueError('Explicit rigid/geometry initial pose required')
                del row['default_pos']
                del row['default_ori']
    return {'algorithm': 'full_layout_except_Rigid_Geometry_initial_pose_v1',
            'content_sha256': hashlib.sha256(canonical(normalized).encode()).hexdigest(),
            'normalized_layout': normalized,
            'only_removed_object_fields': ['Rigid/Geometry.default_pos', 'Rigid/Geometry.default_ori'],
            'dynamic_and_unknown_fields_retained': True,
            'independence_claimed': False, 'clean_eval_claimed': False}
