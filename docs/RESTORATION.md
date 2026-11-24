# Source-Bound Restoration And Readback

All actual values below belong in private external files. No historical source
IDs, task layouts, episode outcomes, deployment hosts or credentials are shipped.
The checked-in policy examples are intentionally incomplete.

## Restore Policy

`RestoreBinding(policy_ref)` takes an independently supplied `{path, sha256}`
reference to `robofate.restore-policy.v1`. Paths must be canonical and absolute.

| Field | Required Meaning |
| --- | --- |
| `data_kind` | `actual` for real preparation, or `synthetic` for CPU fixtures |
| `data_root` | Private collection root, separate from package code |
| `config_root` | Private pinned policy/rule directory |
| `execution_owner`, `partition` | Explicit current owner, never inherited |
| `task_totals` | Reviewed original native budget for each task |
| `excluded_source_ids` | Explicit excluded/refused/consumed source IDs |
| `admission_ref` | Independently pinned admission ledger |
| `rules.profile_template` | Pinned schema/threshold/unit template |
| `rules.invariants` | Pinned exact discrete checks and coverage requirements |
| `code_pins` | Exact pins for all five `restoration/*_impl.py` files |

The admission ledger has schema `robofate.source-admission.v1`, the same
`data_kind`/`execution_owner`, and a `sources` mapping keyed by exact source ID.
Each row requires `source_id`, `task`, `status: reserved`, `consumed: false`,
`refused: false`, `context_filename`, `context_ref`, `manifest_ref`, and
`readback_ref`. Every evidence pin is read again; modifying a consumed/refused
ledger invalidates the old policy rather than silently admitting the source.

`source_material(context_ref, manifest_ref, readback_ref, t1=...)` preserves the
original terminal DM-failure, identity, renderer, state-shape and budget checks.
It derives the source-specific schema from the admitted initial state, keeping
the externally pinned numeric thresholds, units and missing-state requirements.
Source metadata is not restoration qualification.

`prepare_context(..., enabled=False)` writes the original profile, missing-state
scope and native/restore configuration with exclusive creation. Synthetic
preparation cannot set `enabled=True`. `load` instantiates the original
context/task/comparator/support definitions for that one source. `bind_actual`
validates the real current Native counters against the six pinned documents.

The `*_impl.py` modules are factory templates. They are compiled from verified
package bytes into per-binding namespaces; import `RestoreBinding`, not those
templates directly. No old runtime module is imported.

## Teacher Persistence

`persistence.CameraBudget` binds task, total, t1 and remaining native budget.
The public camera observer adds measured matrices/conventions only. It never
renders, resets, advances control, or adds object/task ground truth.

The original `Journal` stores exact causal-history references, observations,
proposals and committed transition references. Its storage binding copies
already committed NPZ/PNG bytes exclusively, without re-encoding images.
The native publication barrier still runs before feedback. No callback that
merely returns a string replaces this committed-data chain.

## Saved-Data Settlement

```sh
robofate settle --context /external/context.json --context-sha256 EXPECTED_HASH \
  --policy /external/readback-policy.json --policy-sha256 EXPECTED_HASH
```

The readback policy has schema `robofate.readback-policy.v1`, `data_kind`,
`data_root`, `manifest_root`, `execution`, `code_pins` and `protocol_refs`.
`execution` supplies `host`, `hostname`, `owner_chat`, and `campaign`.
`protocol_refs` pins `outline` and `directive`. `code_pins` names all six files:
`reader.py`, `decoder.py`, `contracts.py`, `content.py`, `publisher_impl.py`,
and `settlement.py`.

Context identifies `data_kind`, `source_id`, `task`, `run_dir`, `source_root`,
`identity`, `execution_owner`, `campaign`, `native_limit`, `t1`,
`remaining_steps`, and `source_readback_ref`. Teacher/source runs must be distinct
children of `data_root/runs`. The source readback and inventory use the original
saved metadata schema, not a caller-provided Boolean approval.

Settlement locks a once-only directory. Native success is necessary but not
sufficient. The original reader rechecks source references, original prefix,
control timing, terminal success, persisted teacher action links and decoded
RGB/state boundaries. Constant pixels remain valid data with diagnostics.
Process/API diagnostics remain separate from data acceptance.

An incomplete attempt retains `ARMED.json`/`FAILED.json` and cannot blindly
retry. A completed report can be reused only with the same context/policy and
unchanged reviewed files. Publication uses exclusive per-source slots;
interrupted or conflicting manifests are rejected. No ledger credit is
committed automatically.

Synthetic settlement emits explicitly labeled validation reports only, zero
positive/negative credits and no production publication. Synthetic publisher
unit tests exercise the original transaction directly in temporary directories,
with `synthetic_fixture: true` and no ledger registration.

## Acceptance Limits

The synthetic suite covers these components with no real network, subprocess,
model, robot, simulator or GPU calls. The enclosing remote test orchestration
may use authorized SSH. Static source correspondence and CPU tests do not
establish whole-pipeline equivalence, live throughput, yield or training quality.
The integrated launch/session/baseline chain is documented in `CAMPAIGN.md`.
