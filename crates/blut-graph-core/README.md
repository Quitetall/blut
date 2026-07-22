# blut-graph-core

`blut-graph-core` is the domain-neutral, `no_std + alloc` semantic compiler for
capability-driven ABIR node graphs. It verifies typed ports and capability,
proof, policy, fidelity, resource, effect, and target contracts before producing
a deterministic `CompiledPlan` shared by MCU AOT, host/stream, and BLUT durable
execution realms.

The crate owns no biosignal semantics, filesystem, network, async runtime, or
plugin process. Those remain implementation concerns behind the kernel,
transaction, and process-host traits.

`GraphId` covers the normalized semantic graph, including capability, proof,
policy, and fidelity contracts. `PlanId` additionally covers the selected
realm, kernels, layouts, buffers, and fusion regions. Callers set an admission
ceiling with `Compiler::with_memory_limit`; untrusted AOT readers also enforce
`PlanLimits::max_peak_bytes` before returning an executable plan.

`KernelDescriptor::implements` names the exact versioned semantic chain executed
by an implementation. Fusion selects one of those explicit implementations;
the compiler never relabels an ordinary single-node kernel as fused. Physical
layout conversions are likewise registered kernels and appear as `StepId`s in
the physical topology, attempt receipts, liveness analysis, and memory plan,
while semantic completion receipts contain only the original `NodeId`s.

Kernel selection is a bounded deterministic whole-plan search. It can reject a
cheap local choice in favor of a feasible global layout, insert a shortest
typed conversion path, and choose the admitted plan with the smallest arena
peak. Explicit fused implementations of arbitrary linear-chain length compete
with unfused partitions and with one another; a cheap infeasible fused kernel
cannot mask a feasible alternative. Buffer slots are reused only after the
prior alias lifetime ends.

Raw `CompiledPlan::from_aot_bytes` is structural inspection, not authorization.
Execution requires `AuthorizedPlan`, produced either by local compilation or by
`KernelRegistry::decode_authorized_plan` after realm, trusted `PlanId`, kernel
implementation, resource, determinism, lowering, effect, and conversion checks.
Effects distinguish prepare/commit transactions, idempotent work, and
explicitly weaker at-most-once or at-least-once execution. Invocation-bound
idempotency keys and partial failure receipts make retries auditable.
External invocation values bind canonical named input ports; zero-input source
nodes cannot receive undeclared data. Nodes that permit partial output must
declare `ExplicitGaps` and emit structured, domain-checked gap receipts.

For firmware, `AuthorizedPlan::mcu_arena_requirements` validates the supported
static subset and returns exact caller-owned arena dimensions without
allocating. The firmware execution loop remains a realm implementation rather
than a disguised use of the generic allocating host executor.

Semantic graphs use schema version 2 and the physical-step/ordered-port contract
is encoded as `BGP2`. The earlier
alpha `BGP1` postcard layout is intentionally not reinterpreted because it had
no physical conversion identity or registry-bound execution contract.

The BLUT engine exposes `adapt_durable_plan`, which maps an authorized
`BlutDurable` physical plan into the existing kind-checked durable DAG. The
adapter preserves semantic graph/plan identities and per-step implementation
lineage plus ordered physical port identities in the durable recipe record. It
rejects ambiguous multi-port mappings, invocation inputs, explicit gaps,
non-pure effects, resource under-declaration, retry drift, determinism drift,
or missing implementation/checkpoint/policy rechecks.
