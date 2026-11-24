# Prefix/Success Format v1

Each `episodes/<task>/<episode_key>.tar.gz` expands to `<episode_key>/` with:

| Member | Content |
| --- | --- |
| `episode.json` | Portable observation/action sequence, instruction and actual boundary |
| `prefix/observations/<native_index>/` | Original DM prefix observations, including original DM t1 |
| `teacher/observations/<native_index>/` | Teacher t1 through successful terminal observation |
| `controls.jsonl.gz` | Selected dense execution target/feedback/time records |
| `provenance.json` | Original source hash, identity and unmodified qualification fields |
| `checksums.json` | SHA256 and byte count of every other member |

Observation directories retain original `observation.json`, `arrays.npz` and
the three camera PNGs unchanged. Historical metadata inside original files can
contain old host paths; use portable paths in `episode.json`, not those paths.

For N actions there are N+1 flattened observations and one additional original
DM boundary observation. Action i consumes observation i; observation i+1 is
post-action. Prefix action indices are `[0,t1)` and teacher action indices are
`[t1,N)`. Teacher restoration introduces a recorded capture boundary, so the
original and newly captured t1 views are not asserted to be pixel-identical.

The exported held-arm target is the last control's actuator target; the final
two targets must agree. Each native command has ten aligned control records.
The original native request is retained, including end-effector pose commands
and pre-clamp gripper values. A single held target is not asserted to reproduce
the complete internal IK control trajectory; use dense controls for that.

`supervise=false` marks prefix actions by default, not a negative outcome label.
All episodes contain a successful teacher suffix. Original source/group/split
and qualification flags retain their actual meanings. Group-aware dataset
splits should be made from preserved source/group identity, not frame randomization.

## Exporter Input

`python export_training.py --plan plan.json` expects:

```json
{
  "expected_hostname": "EXPLICITLY_AUTHORIZED_HOST",
  "output": "/isolated/release/training_export_v1",
  "episodes": [
    {
      "episode_key": "task__sourcehash",
      "source_id": "original-source-id",
      "task": "original-task-name",
      "t1_native": 25,
      "native_total_budget": 400,
      "source_manifest": "/original/manifest.json",
      "source_manifest_sha256": "FULL_SHA256",
      "inventory_path": "/audit/episode_inventory.json",
      "branches": {
        "dm_failure": {"resolved_path": "/original/baseline-run"},
        "teacher_success": {"resolved_path": "/original/teacher-run"}
      }
    }
  ]
}
```

The immutable inventory has `files` entries with `source`, `bytes`, `mtime_ns`.
`inventory_episodes.py --plan` accepts an explicit inventory plan. Relocation
requires explicit `physical_runs`; no historical machine path is inferred.
Its full-branch inventory is an audit input, not authorization to publish full
branches.

The exporter checks source manifest hashes, inventory sizes/mtimes, PNG chunk
CRCs/deflate scanlines, numeric state, action/control alignment, native success,
held targets and contiguous indices. It does not launch or validate a trainer.
An existing `.ready.json` receipt is reused only after archive size, full SHA256
and source identity checks; run the reader with its expected SHA256 for
independent complete revalidation.
