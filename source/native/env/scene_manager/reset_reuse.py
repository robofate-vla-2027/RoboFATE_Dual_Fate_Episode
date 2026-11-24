"""Validate pose-only rigid-layout resets without mutating a live GPU stage."""
from copy import deepcopy
import math


class LayoutRestartRequired(RuntimeError):
    """A changed asset/topology needs a fresh simulator process."""


def pose_updates(previous, requested):
    allowed = {"Rigid", "Geometry"}
    for kind in ("Dynamic", "Articulation", "Garment", "Fluid"):
        if previous.get(kind) or requested.get(kind):
            raise LayoutRestartRequired(f"{kind} reset requires a fresh process")
    fixtures_old = {k: v for k, v in previous.items() if k not in allowed}
    fixtures_new = {k: v for k, v in requested.items() if k not in allowed}
    if fixtures_old != fixtures_new:
        raise LayoutRestartRequired("Fixture/background change requires a fresh process")

    def indexed(layout):
        records = {}
        labels = set()
        for kind in sorted(allowed):
            for category, instances in layout.get(kind, {}).items():
                for instance in instances:
                    label = instance.get("label")
                    if not label or label in labels:
                        raise LayoutRestartRequired("Missing or duplicate semantic label")
                    labels.add(label)
                    records[(kind, category, label)] = instance
        return records

    old, new = indexed(previous), indexed(requested)
    if old.keys() != new.keys():
        raise LayoutRestartRequired("Object identity/count change requires a fresh process")
    updates = []
    for key, record in new.items():
        fixed_old = {k: v for k, v in old[key].items() if k not in ("default_pos", "default_ori")}
        fixed_new = {k: v for k, v in record.items() if k not in ("default_pos", "default_ori")}
        if fixed_old != fixed_new:
            raise LayoutRestartRequired(f"Asset/physics/appearance change for {key}")
        pos, quat = record["default_pos"], record["default_ori"]
        if len(pos) != 3 or len(quat) != 4 or not all(math.isfinite(v) for v in (*pos, *quat)):
            raise ValueError(f"Invalid pose for {key}")
        if not 0.99 <= sum(v * v for v in quat) <= 1.01:
            raise ValueError(f"Invalid quaternion for {key}")
        updates.append((key[0], key[2], deepcopy(pos), deepcopy(quat)))
    return updates
