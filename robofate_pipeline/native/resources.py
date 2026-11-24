"""Explicit two-device admission; never select devices or touch peer processes."""
import os
from pathlib import Path
import subprocess

from ..evidence import need
from .memory import memory_headroom


def validate_allocation(devices, gpu_csv, processes_csv, environment):
    rows = [[part.strip() for part in row.split(",")] for row in gpu_csv.splitlines() if row.strip()]
    need(len(rows) == 2 and all(len(row) == 5 for row in rows), "Expected exactly two visible devices")
    need([int(row[0]) for row in rows] == devices["visible_gpu_indices"]
         and [row[1] for row in rows] == devices["gpu_uuids"], "Visible device identity mismatch")
    need(len(set(devices["physical_gpu_ids"])) == 2
         and all(type(i) is int and i >= 0 for i in devices["physical_gpu_ids"]),
         "Explicit physical device mapping required")
    need(int(rows[0][2]) >= 45000 and int(rows[1][2]) >= 35000,
         "Original policy/planner and simulation headroom admission failed")
    assignments = {k: v for k, v in environment.items() if k.startswith(("SLURM_", "PBS_", "LSB_"))}
    for key in ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS"):
        if key in assignments:
            need(set(map(str, devices["physical_gpu_ids"])).issubset(assignments[key].split(",")),
                 "Scheduler does not allocate the requested physical devices")
    need(not any(k in assignments for k in ("PBS_JOBID", "LSB_JOBID")),
         "Unverified scheduler allocation")
    return dict(gpu_rows=gpu_csv, existing_processes=processes_csv, scheduler_environment=assignments,
        authorized_physical_gpus=devices["physical_gpu_ids"], gpu_uuids=devices["gpu_uuids"],
        peer_processes_are_not_managed=True)


def inspect(policy, run):
    policy.host_guard()
    need(policy.value.get("execution_enabled") is True, "Explicit execution enable required")
    query = ["nvidia-smi", "--format=csv,noheader,nounits"]
    gpu = subprocess.check_output(query + ["--query-gpu=index,uuid,memory.free,memory.total,utilization.gpu"],
                                  text=True, timeout=15)
    peers = subprocess.check_output(query + ["--query-compute-apps=pid,gpu_uuid,used_memory"],
                                    text=True, timeout=15)
    allocation = validate_allocation(policy.value["native_devices"], gpu, peers, os.environ)
    memory = memory_headroom(120)
    for scope, margin in ((Path(run), 25 * 1024**3), (Path("/"), 256 * 1024**2)):
        disk = os.statvfs(scope)
        need(disk.f_bavail * disk.f_frsize > margin, "Insufficient original disk safety margin")
    return dict(allocation=allocation, memory=memory)
