# RoboFATE Dual Fate Episode

Reusable source for dual-branch robot data collection and prefix/success training
exports. This project contains **code, public task definitions and synthetic
tests**, not collected episodes, sample inventories, experiment results or
machine configurations.

## Quick Start

Python 3.11+ is required. Keep environments and generated build files outside
the release tree.

```sh
python3 -m venv ../robofate-env
../robofate-env/bin/python -m pip install numpy Pillow gymnasium
make check PYTHON=../robofate-env/bin/python
```

To install the CLI without generating build files in the checkout:

```sh
../robofate-env/bin/python -m pip install setuptools wheel
../robofate-env/bin/python tools/build_wheel.py --output ../wheels
../robofate-env/bin/python -m pip install ../wheels/robofate_dual_fate_episode-*.whl
```

The package includes the collection seams, recorder, teacher takeover and data
tools. `source/` supplies the Native task/environment, OpenDM interfaces, teacher
orchestration, IsaacLab and cuRobo source. Keep the checkout when deploying these
dependencies; the Python wheel does not install the separate simulator stack.

| Component | Location |
| --- | --- |
| Source catalog, contiguous registration and locked campaign ledger | `robofate_pipeline/campaign/` |
| Frozen DM, Native-first parent and complete baseline reader | `robofate_pipeline/baseline/` |
| Baseline loop, causal prefix replay, teacher loop | `robofate_pipeline/collection.py` |
| 25 Hz RGB/state/action recording and ordered publication | `robofate_pipeline/recording/` |
| Teacher owner/dispatch/transition checks | `robofate_pipeline/recording/teacher.py`, `predict.py`, `takeover.py` |
| Original source-bound restore/context/physics comparator | `robofate_pipeline/restoration/` |
| Original camera budget and durable teacher journal | `robofate_pipeline/persistence/` |
| Replay, camera, journal and takeover composition | `robofate_pipeline/continuation.py` |
| Production Model, isolated worker and owned supervisor | `robofate_pipeline/teacher/` |
| Native creation hooks, layout checks and diagnostics | `robofate_pipeline/native/` |
| Reserved-source preparation and Native-child execution | `robofate_pipeline/execution.py` |
| Foreground deadline, owned cleanup and inherited simulator lock | `robofate_pipeline/foreground.py` |
| External environment, Conveyor pins and Vulkan source/build | `robofate_pipeline/native/deployment.py`, `vulkan_build.py`, `vulkan/` |
| Independent three-condition readback and one-shot settlement | `robofate_pipeline/readback/` |
| Native environment, task definitions and DM adapter | `source/native/` |
| Teacher workspace/session implementation | `source/teacher/` |
| Inventory and training export | `inventory_episodes.py`, `export_training.py` |
| Archive validation and training reader | `dataset_reader.py` |
| Clean-release audit and exact file hashes | `robofate_pipeline/privacy.py`, `SOURCE_INDEX.json` |

## Collection Boundary

The source pipeline is integrated end to end, with **offline acceptance only**.
Historical machine-specific launch chains are not distributed. The original
restore, persistence and readback implementations are included with external
source/owner/path/rule bindings. The production teacher, owned supervisor and
Native construction hooks are now included, with a reserved-source execution
API. The foreground parent, external deployment environment, original Conveyor
checks and Vulkan compatibility source are wired. The frozen-DM baseline,
independent baseline readback, source reservation, positive/negative ledger
transactions and finite foreground campaign entry are included. No historical
run is needed to bootstrap a fresh campaign, and no bundled configuration grants
permission to execute one. See [campaign setup](docs/CAMPAIGN.md).
See [runtime components](docs/RUNTIME.md), [deployment](docs/DEPLOYMENT.md) and
[restoration/readback](docs/RESTORATION.md).

The reusable code does not relax these gates: unknown action outcomes latch
failure; source admission and restored-t1 verification remain mandatory; provider
stops are checked again before control; teacher feedback requires committed
recording. Numerical/pixel agreement is not itself a restore qualification.

CPU tests validate these code paths using explicitly synthetic peers. They do
not establish real simulator readiness, collection yield, speed, restore quality
or training performance.

The source-bound continuation uses the original restore comparator and journal,
not no-op restore or persistence callbacks. Synthetic readback never grants
positive credits or publishes a production manifest. An interrupted readback
requires explicit reconciliation, not automatic retry.

## Training Data

Export scope is the **original shared DM/teacher prefix and successful teacher
suffix only**. The original DM boundary and the teacher's fresh t1 observation
remain distinct. Supervision defaults to teacher actions; the prefix is context.
See [format](FORMAT.md).

```sh
robofate inventory --plan /external/inventory-plan.json
python tools/make_export_plan.py --help
robofate export --plan /external/export-plan.json
python dataset_reader.py /external/episode.tar.gz --sha256 EXPECTED_SHA256
```

External plans and generated recordings may contain private provenance. Keep
them outside this code project and audit data exports separately.

## Clean Distribution

```sh
robofate audit --repo .
# Optional: private historical literals, never included in the project:
robofate audit --repo . --denylist /external/private-denylist.json
```

The audit checks every path and file byte, including hidden files and upstream
binary meshes, rejects archives, run payloads, links and generated files, and
requires an exact SHA256 source index. Reports contain rule identifiers instead
of matched private text. Pattern matching is not a universal privacy proof;
release selection and the private historical-literal audit are complementary.

Robot meshes and task recipes are upstream source resources, not episode
recordings. Public URLs, copyright notices, generic loopback addresses and
container-internal mount paths are retained where needed.

## Rights

No new license is granted for the assembled project. Preserve the original
notices under `source/native/`, its third-party dependencies and
`source/teacher/THIRD_PARTY_NOTICES.md`. cuRobo assets have separate terms.
Simulator assets, weights, licensed binaries and credentials are supplied
externally. The robot-mesh references in two cuRobo URDF files use relative paths
instead of an upstream developer's absolute home directory.
