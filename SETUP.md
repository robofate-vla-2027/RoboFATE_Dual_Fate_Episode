# Reusing The Frozen Pipeline

## Safe Inspection

Run `python unpack_sources.py --output ./source`. This verifies both archive
SHA256 values and extracts source without importing or executing it.

The remote bundle contains `runtime/`, `source_tree/`, and `SOURCE_MANIFEST.json`.
The local bundle contains `local_orchestration/` and `LOCAL_SOURCE_MANIFEST.json`.
The manifests preserve exact source hashes and historical bindings.

## Selected Releases

- Teacher: `runtime/RUNTIME_H20_CHAT2_TEACHER_20261109_016.json`,
  SHA256 `429673ae98f9154d82ead80968e2dc9b0a7e9565aaa55239e1e7781d4bf520aa`.
- Baseline: `runtime/RUNTIME_H20_CHAT2_CONTENT_20261109_038.json`,
  SHA256 `b17e4a405102d19f5178d6d192cef04bccad605bc8b3051edc68ec820d6ecfab`.
- Teacher public-recipe lineage: EmbodiedEvo commit
  `470278c478bdb5e4b6f17dde1ea0393761f230f0`.

Do not replace `actual_adoption=false` with an inferred success. These are
latest checkpoint releases, not proof every historical episode used them.

## Collection Stages

Use `local_orchestration/CHECKPOINT.json` for exact historical helper mappings:

1. Reserve a source through the original ledger lock/CAS transaction.
2. Bind source identity, baseline release and explicit node/resources.
3. Prepare and arm the baseline. Record actual start separately from intent.
4. Observe baseline metadata/terminal status and settle original evidence.
5. Reserve and bind teacher work at the actual t1, preserving remaining budget.
6. Prepare teacher restore, run foreground collection and observe termination.
7. Export settlement metadata and settle only genuinely qualified results.
8. Inventory immutable positive pairs, export prefix/success view, verify hashes.

Key helper versions are included with their imported predecessors. In particular
local stream helpers depend on earlier helper source and `changes/`; do not
copy just one latest entrypoint out of the archive.

## External Dependencies

Publication tools: Python 3.8+ standard library. Optional image/tensor consumers
can use Pillow, NumPy, and their trainer's pinned PyTorch version.

Collection additionally needs the originally pinned RoboDojo/native simulator
environment and assets, GPU driver/Isaac/Omniverse compatibility, baseline policy
environment and weights, cuRobo/robot assets where referenced, and the private
teacher runner/API access. The private teacher executable and credentials are
not redistributed. Upstream dependency declarations are retained under
`source_tree/.../teacher/upstream/EmbodiedEvo/` and `runtime/curobo_original_895c651/`.
This publication does not claim a fresh dependency installation was validated.

Historical absolute paths and hash-pinned bindings must be mapped deliberately
to a new authorized deployment. Rebind paths in a new deployment copy, retain
the immutable source snapshot, generate new hashes/manifests, and validate
readiness before collection. Never run archived launch scripts blindly:
historical GPU IDs, thread-owner guards and paused checkpoint work are not
current resource allocations or authorization to resume.

## Non-Included Operational State

The live 99 GB local handoff, live shared ledger, locks, credentials, sessions,
PID files, weights and incomplete runs are not part of the source archive.
A new deployment needs its own explicit operational state and ledger ownership;
do not take over another chat's lock or resume its paused work.
