"""Spawn-safe original worker entry; synthetic bindings never start an actor."""
import os


def run_worker(connection, specification, observation, step_limit, episode):
    from .runtime import install
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"
    runtime = install(specification["runtime_binding"])
    runtime.execution_guard()
    return runtime.original_worker.run_worker(connection, specification, observation, step_limit, episode)
