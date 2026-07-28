# blut-graph-core

`blut-graph-core` is the domain-neutral, `no_std + alloc` semantic compiler for
capability-driven ABIR node graphs. It verifies typed ports and capability,
proof, policy, fidelity, resource, effect, and target contracts before producing
a deterministic `CompiledPlan` shared by MCU AOT, host/stream, and BLUT durable
execution realms.

Node configuration is a sealed exact-value algebra (`bool`, signed/unsigned
integers, bounded text/choice/bytes) validated against a normalized descriptor
schema. Defaults are materialized before semantic identity is calculated, so
implicit and explicit defaults compile to the same `GraphId` and `PlanId`.
Unknown, missing, mistyped, and out-of-range values fail before kernel search.

Every physical port carries its ABIR root/view semantic type plus proof, policy,
fidelity, extent, layout, and lease contract. These contracts survive fusion,
layout conversion, AOT serialization, and durable-plan adaptation; an edge is
admitted only when the producer contract satisfies the consumer contract.

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

State is explicit and bounded. `StateContract` distinguishes invocation,
session, and durable state and binds checkpoint mode, snapshot size, and maximum
checkpoint interval. Cross-invocation feedback uses a positive-delay
`FeedbackEdge`, a dense `FeedbackPlan`, and a session contract; the plan records
the exact persistent-state arena size. The generic one-shot executor and MCU
subset fail closed on these constructs until their realm supplies a state
store. Hierarchical decompositions are content-identified `SubgraphSchema`s;
their identity covers repeated local instances, typed config, internal edges,
interface bindings, and nested child identities. Registered lowerings require
descriptor-compatible child interfaces, exact port maps, and type-compatible
outer-to-inner configuration bindings. `KernelRegistry::materialize_subgraph`
applies one canonical outer instance to a concrete reference graph that callers
can compile with fusion enabled or disabled. Ordinary compilation still keeps
the outer node physical and preserves root subgraph identity in its step; BGP3
does not implicitly inline-expand inner DAGs.

Semantic graphs use schema version 3 and the physical-step/ordered-port contract
is encoded as `BGP3`. Earlier alpha `BGP1` and `BGP2` postcard layouts are
intentionally not reinterpreted. `BGP3` adds typed canonical configuration,
per-port semantic contracts, explicit bounded state/feedback/session records,
and identity-bound hierarchical lowering.

The process-plugin control plane is `BPC2`, a bounded canonical postcard wire.
It binds a domain-separated BLAKE3 executable digest, declared capabilities,
a canonical unsigned Ed25519 manifest digest and verifier key identifier,
request and invocation identity, startup/request/heartbeat
deadlines, maximum inflight/frame sizes, and graceful-then-kill or immediate
teardown policy. Its lifecycle only advances from spawn through handshake,
ready, draining, and termination; it cannot re-enter readiness after teardown.
The crate defines the supervisory contract but never spawns a process itself.

The BLUT engine exposes `adapt_durable_plan`, which maps an authorized
`BlutDurable` physical plan into the existing kind-checked durable DAG. The
adapter preserves semantic graph/plan identities and per-step implementation
lineage plus ordered physical port identities in the durable recipe record. It
rejects ambiguous multi-port mappings, invocation inputs, explicit gaps,
non-pure effects, resource under-declaration, retry drift, determinism drift,
or missing implementation/checkpoint/policy rechecks.
