# RoboFATE Dual Fate Episode

Private production-lineage source and portable training-data tools.

Dataset: https://www.modelscope.cn/datasets/mxk1998/RoboFATE_Dual_Fate_Episode

## Dataset Scope

92 positive sources across 15 tasks, from ledger revision 6209:
21 from Chat1 and 71 from Chat2. The published training view includes only the
original shared DM prefix and the teacher successful suffix.

Excluded: DM failure suffixes, 67 standalone DM-success negatives, teacher
replayed prefixes, dense auxiliary RGB, simulator restore snapshots, renderer
diagnostics, model weights, private teacher executables, and credentials.

Each episode preserves three 640x480 camera streams, native 25 Hz observation
cadence, 14-dimensional state/action vectors, instruction, actual t1 boundary,
teacher supervision mask, original requested actions, selected dense execution
records, provenance, and SHA256 inventories.

The 14D action is `[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]`.
Arm labels are recorded actuator hold targets, not measured observation joint
positions. Pose requests are projected from recorded IK targets. Gripper
requests use the native API's `[0,1]` clamp. Native gripper state is the recorded
normalized previous control, not an independent physical gripper sensor.

## Read And Validate

Python 3.8+ standard library is sufficient for validation and indexing:

```sh
python -B -m unittest -v test_dataset_reader.py
python dataset_reader.py EPISODE.tar.gz --sha256 EXPECTED_SHA256
python dataset_reader.py EPISODE.tar.gz --extract-to ./episode
```

Extraction requires a new destination and checks every member before promotion.
No path traversal, links, duplicate members, or overwriting are permitted.

```python
from dataset_reader import load_episode, iter_transitions, action_chunk

episode = load_episode("./episode")
for sample in iter_transitions("./episode"):
    # PIL/numpy/torch conversion is a trainer concern.
    # sample["images"] contains head/left_wrist/right_wrist PNG paths.
    state, target = sample["state"], sample["action"]
    assert len(state) == len(target) == 14
targets, teacher_loss_mask = action_chunk(episode, start=0, horizon=50)
```

Keep prefix observations as causal context; default supervised transitions are
teacher-only. `action_chunk` returns shorter terminal chunks rather than
inventing padded labels. The original DM t1 observation is kept separately;
the flattened trajectory uses the teacher's fresh observation after restore.

This is real prefix/success imitation data, not the full original dual-branch
corpus. It is not a drop-in match for the referenced 3,400-source/17-task trainer:
actual t1 is 10-55, not that trainer's >=175 history contract; only 44 sources
have matching task names. No missing history, negative branch, or label was
fabricated. Trainer adaptation is intentionally outside this publication.

## Pipeline Template

Chat2 production lineage is the selected template and shares Chat1's teacher
and recording core. Its larger observed yield is not a controlled proof of
better speed or stability: the task sets differ and full-attempt denominators
are unavailable. All 542 referenced frozen runtime files passed hash checks.

- `chat2_production_source.tar.gz`: 2,989 remote source/config files.
- `chat2_local_orchestration.tar.gz`: 1,851 local Python/shell source files and
  the historical checkpoint; includes versioned helper dependencies.
- `SOURCE_BUNDLES.json`: archive hashes, byte counts and lineage.
- `unpack_sources.py`: verified, non-executing source extraction.
- `export_training.py`: selected prefix/success exporter; see `FORMAT.md`.
- `dataset_reader.py`: portable reader and complete member checksum validator.
- `SETUP.md`: external dependencies, historical bindings and collection stages.

```sh
python unpack_sources.py --output ./source
```

The archives are intentionally frozen to preserve historical source and hash
pins. They are not a portable one-command simulator deployment. Old host/GPU
IDs and paused-work checkpoint fields are provenance, never permission to
launch or resume work. No collection/model/GPU job was launched for publication.

## Rights And Privacy

This repository and the dataset are private. No new open-source or dataset
redistribution license is granted. Retain embedded third-party license notices.
Supply licensed simulator/assets/weights and separately authorized teacher
credentials externally; none are included in this repository.
