"""Unexecuted-in-Kit observer for a future owned profile; no rendering or writes.

Pass existing carb settings and omni.graph.core handles. Call at existing
before/after-capture checkpoints only. The returned settings are the carb view,
not a resolved USD RenderSettings view or a snapshot of RTX internal RNG state.
"""

from math import isfinite

DISPATCHER = "/Render/PostProcess/SDGPipeline/PostProcessDispatcher"
READ_KEYS = (
    "/rtx/rendermode",
    "/rtx/post/aa/op",
    "/rtx/shadows/constantSeed",
    "/rtx/shadows/enabled",
    "/rtx/caustics/enabled",
    "/rtx/raytracing/caustics/useFrameSeed",
    "/rtx/post/tvNoise/enabled",
    "/rtx/post/tvNoise/fixedTimeSeed",
    "/rtx/post/tvNoise/fixedTimeSeedCount",
    "/rtx/externalFrameCounter",
    "/rtx-transient/resetPtAccumOnlyWhenExternalFrameCounterChanges",
    "/rtx/rendering/useSimulationTime",
    "/omni/replicator/globalSeed",
    "/omni/replicator/RTSubframes",
    "/app/updateOrder/checkForHydraRenderComplete",
    "/app/renderer/waitIdle",
    "/app/hydraEngine/waitIdle",
)


def _read_setting(settings, key):
    try:
        value = settings.get(key)
    except Exception as exc:
        return {"status": "error", "error_type": type(exc).__name__}
    if value is None:
        return {"status": "missing", "value": None}
    if type(value) not in (bool, int, float, str):
        return {"status": "invalid_type", "type": type(value).__name__}
    if type(value) is float and not isfinite(value):
        return {"status": "nonfinite"}
    return {"status": "observed", "value": value}


def _dispatcher_stamp(graph):
    try:
        node = graph.get_node_by_path(DISPATCHER)
        if not node:
            return {"status": "missing"}
        def read_pair():
            return [node.get_attribute("outputs:referenceTimeNumerator").get(),
                    node.get_attribute("outputs:referenceTimeDenominator").get()]
        first, second = read_pair(), read_pair()
        if any(type(v) is not int for v in first + second):
            return {"status": "invalid_type"}
        if first[1] <= 0 or second[1] <= 0:
            return {"status": "invalid_denominator", "first": first, "second": second}
        return {"status": "stable_read" if first == second else "changed_during_read",
                "first": first, "second": second}
    except Exception as exc:
        return {"status": "error", "error_type": type(exc).__name__}


def snapshot(settings, graph, *, phase, native_count, physics_time,
             capture_index, actual_render_calls=None):
    """Return JSON-safe observations; never infer RTX frame/seed from physics."""
    for value in (native_count, capture_index):
        if type(value) is not int or value < 0:
            raise ValueError("counts must be nonnegative integers")
    if actual_render_calls is not None:
        if type(actual_render_calls) is not int or actual_render_calls < 0:
            raise ValueError("actual_render_calls must be observed, nonnegative, or None")
    if type(physics_time) not in (int, float) or not isfinite(physics_time):
        raise ValueError("physics_time must be finite")
    if not isinstance(phase, str) or not phase:
        raise ValueError("phase must identify the existing checkpoint")
    return {
        "schema": "v007.read_only_capture.1", "qualified": False,
        "phase": phase, "native_count": native_count, "physics_time": physics_time,
        "capture_index": capture_index, "actual_render_calls": actual_render_calls,
        "rtx_sample_index": None, "rtx_rng_state": None,
        "carb_settings": {key: _read_setting(settings, key) for key in READ_KEYS},
        "dispatcher_reference_time": _dispatcher_stamp(graph),
    }
