# Campaign Contract

## Scope

The finite foreground chain is:

1. Verify an external source catalog, exact task AST budget, raw initialization,
   numeric layout decoder, family and split.
2. Under the stable ledger lock, reserve one fresh baseline source/negative slot.
   Only a genuinely empty campaign can bootstrap without a predecessor.
3. Start Native with an inherited simulator lease. Save the initial boundary
   before loading the frozen DM service. Keep the original 900-second policy
   allowance inside the finite Native deadline, and clean both owned groups.
4. Run the original native25/PT/RGB/control/timestamp baseline reader once.
   A complete native DM success receives label 0. A complete native DM failure
   receives no quota credit and can become a teacher source.
5. Independently reserve one positive slot, publish pinned source admission,
   prepare the original restore context and teacher inputs, and execute the
   owned teacher parent. Restore and t1 verification precede teacher actions.
6. Run original three-condition readback and exclusive local manifest
   publication; commit a successful teacher branch once as label 1.
7. Export only the original shared prefix and successful teacher suffix using
   the existing inventory/export/training reader.

This is local filesystem publication, not an upload to a service. Raw terminal
results, scientific qualification, resource closure and quota accounting remain
separate. Errors stop the foreground plan; no job, heartbeat or automatic actor
retry is installed.

## External Configuration

Examples under `configs/` are deliberately disabled or empty. They contain no
deployment identity, source inventory, weights, receipts or grants.
All `*_ref` fields are exact `{"path": absolute_path, "sha256": digest}` objects.

Use a private `data_root` outside the checkout with existing `runs/`, `sessions/`,
`ledger/` and `config/` directories. The campaign `config_root` must be inside
this data root. Runtime/dependency source roots must be separate from data.
Never move or publish a live data root as part of the code release.

The execution policy extends `docs/RUNTIME.md` with:

- `source_catalog_ref`: a `robofate.source-catalog.v1` document with matching
  data kind, campaign and execution owner, and a `tasks` mapping.
- `campaign_config_ref`: a `robofate.campaign-config.v1` document with matching
  identity, `ledger_root`, `config_root`, external `source_commit`, optional
  `target_at`, and a `tasks` mapping.
- `frozen_dm`: `model_root`, `checkpoint_manifest_ref`, `config_ref`, executable
  `python` binding, `library_dirs`, `cuda_visible_devices` and `transport_ref`.
  Policy CUDA visibility remains the original container-logical `"0"`, not an
  authorization to use any physical card.

Each source-catalog task supplies `directory`, `base_ref`, `initial_count`,
`reserved_final`, `task_ref`, and optional `mixed_collection_base: true`.
The base is an externally pinned `refate.split_manifest.v1`, with complete
`layouts`, `families`, `layout_to_group`, contiguous `initial_source_sequence`,
`original_partition`, `owner_chat`, `campaign_id` and task fields.
Each layout row contains `layout_id`, `native_bank: 0`, `layout_ref`,
`initialization_content_sha256`, `initialization_distinct_from_preceding`,
`group_id` and `split`. Family IDs are computed by the original `family()`
normalization, not assigned by collection outcomes. Run `SourceCatalog.prepare`
to validate every field; it grants no launch authority.

`SourceCatalog.append` verifies the original contiguous next-layout and lineage
rules, preserving all previous fields and sealed layouts. Mixed-label collection
bases and their append chains use the original exact-membership checks and
retain the explicit non-independent-development/evaluation limitation. They are
not a general permission to cross train/dev families.

Each campaign-config task supplies nonnegative integer `positive_quota`,
`negative_quota`, and the selected original `native_wall_seconds` (greater than
900 and at most 3600). Teacher wall scope is fixed at 3600. The baseline path
requires a reserved negative slot before execution; it never silently expands
quotas when a quota is full.

The checkpoint manifest has schema `robofate.checkpoint-files.v1` and an exact
`files` list of SHA256 references for every file under `model_root`. Weights are
verified by streaming, never copied or deserialized during configuration checks.
Small executable/code/library pins stay in the runtime manifest. The transport
pin identifies the installed `websockets.sync.connection` implementation; the
original ping-only RNG seam must still match.

The DM config retains the original validated settings: frozen DM0.5/OpenDM,
native direct CUDA FP32 load, zero trainable parameters, absolute 14-dimensional
joint actions, predict 50/execute 25, 20 history slots at 1 Hz with 25 Hz runtime,
16 tokens per slot and 10 diffusion steps. No RNG reset after load, fallback
checkpoint or alternate policy model is allowed.

## Explicit Commands

These commands are deployment documentation, not part of the CPU acceptance.
`initialize`, `status` and metadata `reconcile` do not start actors.
`run` requires an actual, explicitly enabled execution policy and exact runtime
host/owner/allocation checks before any process launch.

```sh
robofate campaign initialize --policy "$POLICY" --policy-sha256 "$POLICY_SHA"
robofate campaign status --policy "$POLICY" --policy-sha256 "$POLICY_SHA"
robofate campaign run --policy "$POLICY" --policy-sha256 "$POLICY_SHA" \
  --plan "$PLAN" --plan-sha256 "$PLAN_SHA"
```

The `robofate.campaign-plan.v1` document pins the execution policy, restore
template and readback policy. `entries` is a finite list of explicit `task`,
`layout_id`, `split_ref`, `baseline_run_id`, `teacher_run_id`, `session_id` and
`t1` values. Run/session tokens have twelve decimal digits, an underscore, and
1-60 alphanumeric/underscore/hyphen characters. Tokens are never recycled.
The restore template supplies the original external rule and code pins; the
runner replaces only its admission reference with the ledger-reserved source.
The teacher is skipped for DM success or a failure terminating at/before t1.

## Interrupted Work

The ledger writes a fsynced `INTENT.json`, atomically publishes a CAS-checked
state and writes a fsynced `COMMITTED.json`. Existing operation IDs may return
only an exact matching saved result. The state is checked against the committed
journal; foreign nodes/tasks/sources and campaign clocks cannot be reset.

```sh
robofate campaign reconcile --policy "$POLICY" --policy-sha256 "$POLICY_SHA" \
  --operation-id "$OPERATION"
```

Reconciliation finishes only an existing metadata intent whose before/after
hash matches the ledger. It never recomputes a transaction or reruns an actor.
Missing intents or diverged state require manual evidence review. Armed/failed
data readers likewise do not automatically retry. Infrastructure-failed
baselines are quarantined with zero quota and cannot become teacher sources.
Provider-stop evidence remains a refusal, not a retryable transport error.

If data settlement completed before cleanup was confirmed, independently
recheck the original owned process identities and isolated worker lifecycle:

```sh
robofate campaign confirm-cleanup --policy "$POLICY" --policy-sha256 "$POLICY_SHA" \
  --context "$CONTEXT" --context-sha256 "$CONTEXT_SHA"
```

This only releases the current settled run's resource gate. It adds no sample
credit and does not signal, kill or manage another process.

## Acceptance Boundary

Synthetic tests exercise the original baseline reader/decoder, source and
family rules, reservations, both fate transactions, durable journal failures,
owned-parent ordering and the pre-existing restore/teacher/export chain.
The test runner rejects real network, process creation and signaling. Separate
compiler checks are CPU-only and never load Vulkan.

The code and offline integration are accepted without generating real episodes.
Live throughput/yield, current checkpoint identity, rendering readiness and
provider availability must not be inferred from these tests. Historical
transport release pin correspondence and historical Vulkan binary/compiler
equivalence are not established; source correspondence is recorded separately.
