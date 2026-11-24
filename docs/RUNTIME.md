# Production Runtime Components

The production teacher is in `robofate_pipeline/teacher/`, not the generic
upstream integration. Model, worker, takeover, prompt, OCI mounts, provider-stop
parser, request timing, byte transport and owned supervisor were extracted from
the effective source chain. The worker checks the provider gate before the
session loop and before handling each request. Unknown action outcomes latch
failure; no automatic model/episode restart is installed.

Original settings remain: Test-only primitive baseline, one fresh session,
native cumulative budget, 3600-second actor budget, 100 execution requests,
1200-second submission limit, requested model `gt-6-as-a`, effort `xhigh`,
and disabled optional web search. Model and effort are configuration evidence,
not proof of a provider-reported model identity. Credentials stay in the
externally provisioned forwarding service, not in the guest or repository.

## External Policy

`configs/execution-policy.example.json` is deliberately incomplete and disabled.
No real host, account, dataset, source ID, GPU UUID, endpoint, runtime package,
credential or historical sample list is shipped.

`manifest_ref` is a path/SHA256 reference to:

```json
{"schema": "robofate.runtime-files.v1", "files": []}
```

Fill the manifest outside this project. It must pin every installed pipeline
Python source, all upstream teacher Python files, the Native model template,
task configuration, handwritten workspace, runtime binaries/resources, planner
configuration, Native source/configuration, offline asset manifest and assets,
IsaacLab experience/launcher and selected raw layout. Empty Python package
markers can be pinned; empty metadata is still rejected.

`teacher_config_ref` pins an external `task.toml` with the original settings
above and an explicitly relocated public baseline workspace. Each task and its
native action budget must appear in `task_totals`. `planner_refs` maps robot
names to exact path/SHA256 references. External exclusions and admission records
are mandatory; absence of an identifier from this repository is not admission.

The deployment must already have its reviewed simulator dependencies. The
foreground parent constructs the Native environment from pinned external paths,
including `ROBOFATE_OPTIX_MANIFEST`, the Vulkan/RTX compatibility layer, offline
assets and runtime library paths. It does not inherit arbitrary shell hooks,
credentials, proxy variables or renderer overrides. The host check compares the external
expected hostname with the actual hostname and requires
`ROBOFATE_SSH_ALIAS` and `ROBOFATE_EXECUTION_OWNER` to match. Logical device zero
is the planner/policy role; logical device one is rendering/physics. Physical
IDs, UUIDs, scheduler allocation and visibility are external, never inferred.
The original free-memory thresholds, cgroup memory accounting and disk margins
remain admission checks, not reservations or permission to interfere with peers.

## APIs

`teacher.install(runtime_binding)` verifies sources and binds the original
implementations without starting a process or making a request.
`teacher.model(...)` creates a fresh unstarted Model.
`SourceContinuation.create_teacher(...)` constructs it from the real restored
owner/counter context and attaches the original Journal before takeover.
Synthetic policy is rejected before worker creation, staging or Native launch.

`execution.prepare_teacher(dispatch_ref, restore_policy_ref, execution_policy_ref)`
accepts an independently reserved source, prepares original handwritten inputs
and writes a one-shot context. Preparation is not source allocation.
`foreground.foreground(context_ref)` is the one-reserved-source parent API, not
a campaign launcher. It locks deployment state, verifies prepared inputs, builds
the environment, probes only the configured local listener, checks resources,
launches one Native child and waits for at most the original 3600 seconds. It
records PID, start ticks, boot ID and process group; cleanup verifies that identity
before both TERM and KILL, with 20-second and 5-second waits. Cleanup failures
preserve the primary error. Exit codes do not grant positive credit or trigger retries.
After waiting, a read-only process-group scan must also find no residual members;
leader exit alone is not cleanup success. Residuals after leader exit are reported,
not signaled using a stale group identity. This receipt covers the Native group
only; isolated worker/runsc lifecycle evidence remains independently required.

`execution.run_teacher(context_ref, lease_ref=..., lock_fd=...)` is the guarded
child API. It requires the parent identity and inherited locked file descriptor;
it does not compete for a second lock. The descriptor becomes non-inheritable
after validation. A parent-death signal is armed before Native startup. The child
rechecks the exact environment/resources, constructs Native, restores the prefix,
runs the production teacher, persists the unqualified result and closes owned
objects. There is no standalone bypass or enabled example.

Independent saved-data settlement follows execution. Raw Native success,
successful worker exit, cleanup and data qualification are distinct facts.
Only the existing reader determines data acceptance; it does not commit the
source ledger or grant collection quota.

## Native Deployment

All `deployment` fields in the example policy must be provided externally.
Interpreter and ffmpeg entries have an absolute `path` and canonical
`target_ref`; this permits an explicitly pinned virtual-environment symlink
without silently substituting a different interpreter. Every referenced file,
preload, ICD library, OptiX asset and extension file must also be in the runtime
inventory. Directory search paths are explicit; they do not imply that every
installed operating-system dependency is bundled or hermetically audited.

`native/vulkan/` contains the original image-copy plan and layer implementation.
Only the old fixed host/alias/owner guard is replaced with an external compile-time
header, and the manifest uses a relative library path. Build outside this checkout:

```sh
python -m robofate_pipeline.native.vulkan_build \
  --output /external/new-layer-build --compiler /external/canonical-cxx \
  --binding /external/vulkan-target.json --sha256 EXPECTED_SHA256
```

The target JSON has schema `robofate.vulkan-target.v1`, `expected_hostname`,
`ssh_alias` and `execution_owner`. The build requires Linux Vulkan headers and a
C++17 compiler. It compiles but never loads the library, creates no GPU context
and does not grant launch permission. Pin the resulting `build-evidence/BUILD.json`, manifest
and ELF library in the external execution policy. Build output contains private
deployment identifiers and must not be distributed. Source correspondence does
not assert binary equivalence to an archived compiler output.

For `pick_from_conveyor_by_image`, the pinned source context must carry exactly
`native_kit_extensions: ["isaacsim.asset.gen.conveyor"]`, budget 700 and
`native_extension_pins`. The policy's Conveyor root and `pins_ref` must agree.
The pin document uses schema `robofate.native-extension.v1`, extension name,
version `1.0.23`, task, `native_total_steps: 700`, and ordered `files` references
for `config/extension.toml` and `bin/libisaacsim.asset.gen.conveyor.plugin.so`.
The original version/module/plugin/ELF checks run before the Native-only enable
flag is added. Missing, altered or unexpected extensions are rejected.

## Verification Limit

CPU tests use temporary synthetic files and mocked peers. They run no sockets,
subprocesses, signals, GPU calls, simulator or model. Static correspondence
covers the generated original definitions; newly written factories/composition
are separately identified and tested, not called byte-identical launchers.

The source-reservation, baseline policy/readback and ledger chain are integrated
with this teacher parent by `campaign.runner`. `baseline.runtime` installs the
original WebSocket RNG isolation before Native construction. The DM service
preserves the original frozen FP32 loading, history and action chunk semantics.
Source-level/offline acceptance is not a live production-equivalence claim.
No real episode is part of this acceptance. CPU compiler
checks, when run separately from the guarded Python suite, exercise only synthetic
copy-plan structures and compile the layer; they do not load Vulkan.
