<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Architecture decision records

The engine-facing decisions from the project's decision log, federated so that the
engine's "ADR NNNN" citations resolve to a document. Each record
is redacted for publication: removed material is marked inline as
`[redacted: <category>]`, and a reference to a record that is not federated reads
`ADR NNNN [not federated]`. The source log is frozen, so these copies do not
drift. Background: [book_src/adrs.md](../../book_src/adrs.md).

| ADR | Title | Status |
|---|---|---|
| [0034](0034-blut-scope-charter.md) | BLUT is the domain control plane — own the nouns, delegate the verbs | in-progress |
| [0037](0037-blut-cookbook-architecture.md) | BLUT cookbook architecture — domain-agnostic orchestrator + loadable cookbooks | in-progress |
| [0044](0044-blut-api-standardization.md) | BLUT API standardization: one launch contract, placement-as-runtime, durability + retry as a Stage contract | in-progress |
| [0046](0046-blut-resource-broker.md) | BLUT resource broker (never-OOM, exact-resource training) | complete |
| [0047](0047-broker-measured-envelope-and-codec-dataloader.md) | BLUT broker measured-envelope hardening + codec dataloader efficiency + production handoff | complete |
| [0055](0055-ingredient-taxonomy-and-registry.md) (cited as 0051 in older code) | Ingredient taxonomy + IngredientSpec registry (full trainer decomposition) + the Course grouping layer | complete |
| [0061](0061-blut-p2p-compute.md) | BLUT P2P distributed compute (trust model + encrypted task dispatch) | complete |
| [0067](0067-remaining-work-toward-vision.md) | Remaining work toward the BLUT vision (the gated TODO) | in-progress |
| [0072](0072-blut-error-domains-and-fuzzing.md) | BLUT error-domain + fault-attribution model, adversarial hardening, fuzz testing | complete |
| [0078](0078-blut-starlark-typed-dynamic-dag.md) | Typed dynamic DAGs: scripts compose registered stages into a checked graph | complete |
| [0079](0079-blut-symmetric-mesh.md) | Symmetric P2P mesh — every host is server, worker, and (optionally) scheduler | complete |
| [0080](0080-blut-async-dp-training.md) | Async distributed training with differential privacy (FedAvg + DP-SGD) | complete |
| [0082](0082-blut-cloud-queue.md) | BLUT cloud compute queue (object-store dispatch, billed on compute) | complete |
| [0083](0083-blut-interface-stack.md) | BLUT interface stack: CLI engine, sidecar TUI/web crates, and the charter boundary | complete |
| [0085](0085-blut-deployment-registry.md) | Give BLUT a versioned plan registry so a fingerprinted DAG can be published, promoted to a named pointer, and rolled back atomically | complete |
| [0086](0086-blut-secrets-management.md) | Give BLUT a SecretRef indirection backed by an age-encrypted store, so credentials reach stages by reference and never land in logs or the content-addressed cache | complete |
| [0087](0087-blut-gpu-aware-scheduling.md) | Schedule GPU stages by topology and VRAM under a raised semaphore, with the never-OOM broker still gating RAM | complete |
| [0088](0088-blut-durable-execution-planspec-1-1.md) | Extend durable resume to mid-epoch via PlanSpec v1.1 additive per-node retry/timeout and a checkpoint-cadence contract | complete |
| [0090](0090-blut-experiment-model-dataset-registry.md) | Give BLUT first-class experiment, model, and dataset registries addressed by content-hashed URIs | complete |
| [0091](0091-blut-check-stages-standard-cookbook.md) | Ship passthrough data-quality check stages and a standard "checks" cookbook that fail-close on assertion breach | complete |
| [0093](0093-blut-web-control-plane.md) | Ship a `blut-web` sidecar (axum + Leptos/WASM) that serves a read-only dashboard v1 over status.jsonl and escalates to live control v2 through the CLI exec-bridge, keeping the engine HTTP-free | complete |
| [0094](0094-blut-eventing-triggers-sla.md) | Turn BLUT's sensor substrate into an event/data-driven trigger system — a file/spool/cron sensor daemon, sidecar webhook ingress, a `blut-notify` sink crate, and declarative SLA rules | complete |
| [0095](0095-blut-rbac-auth.md) | Gate every mutating BLUT control action behind capability-scoped tokens (viewer/operator/admin) with an audit trail, defaulting to read-only and opening a path to OIDC | complete |
| [0096](0096-blut-multi-tenancy.md) | Give BLUT projects/domains with per-tenant namespaces for cache, registry, and plans — and make the clinical zone a hard tenant boundary a Restricted tenant's data can never cross | complete |
| [0097](0097-blut-metrics-tracing-export.md) | Export Prometheus/OpenMetrics and OpenTelemetry telemetry from a sidecar crate so the engine stays HTTP-free | complete |
| [0098](0098-blut-cost-accounting-dashboards.md) | Build cost/chargeback dashboards and fail-closed budget guards over the cloud-queue cost ledger | complete |
| [0099](0099-blut-lineage-graph-model-cards.md) | Expose a full artifact-provenance graph, per-model cards, and a run-diff tool over the lineage_db substrate | complete |
| [0100](0100-blut-data-catalog.md) | Build a searchable, typed, lineage-linked dataset catalog over BLUT artifacts and deliberately stop short of a DVC-style dedup hub | complete |
| [0101](0101-blut-backfills-partitions.md) | Add a partition abstraction with partition-aware execution, backfill windows, and a status matrix | complete |
| [0102](0102-blut-dag-optimization-advanced.md) | Extend dag_opt beyond schedule-hints/DCE with stage fusion, speculative execution, pipeline parallelism, and cache/priority scheduling | complete |
| [0103](0103-blut-async-compute-completion.md) | Promote the flag-OFF async primitives to the default path with backpressure and broker-integrated admission | complete |
| [0106](0106-blut-ha-control-plane-raft.md) | Make the scheduler capability highly available via Raft — leader election and replicated scheduler state that survives a coordinator crash without losing in-flight dispatch | in-progress |
| [0109](0109-blut-hpo-advanced.md) | Run Bayesian, multi-objective, and cost-aware HPO as a DAG-native surrogate search that fans trials out into cache-keyed nodes | in-progress |
| [0110](0110-blut-auto-tuning.md) | Ship auto batch-size, LR-finder, AMP, grad-accum, and loss-spike tuning as opt-in control policies that only ever propose — the broker still decides | complete |
| [0111](0111-blut-python-sdk.md) | Ship a Python client SDK that emits the PlanSpec IR and drives runs over the CLI | complete |
| [0112](0112-blut-ecosystem-integrations.md) | Deliver ecosystem integrations as registered connector cookbooks that shell out, never as dynamic plugins | complete |
| [0133](0133-blut-typed-declaration-admission.md) | Typed-declaration admission — the engine derives launch footprints from Stage declarations, never from interpreting cookbook recipe JSON | complete |
| [0137](0137-blut-engine-m3-m6-milestone-constitution.md) | Commit the BLUT-engine M3–M6 milestone constitution — bind each remaining premier milestone to its constituent ADRs and one fail-closed gate, and fix "engine 1.0" as the crates.io release distinct from the ADR-0118 [not federated] family gate | in-progress |
| [0154](0154-blut-owns-the-run-ledger.md) | BLUT owns an append-only run ledger; a run is a noun, a dashboard is a verb | in-progress |
| [0156](0156-blut-experiment-execution-ux.md) (cited as 0071 in older code) | BLUT experiment-execution UX (advisory stages, admission auto-tune, result retrieval) | accepted |
| [0170](0170-one-dataset-command-surface.md) | One dataset command surface | complete |
