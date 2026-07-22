# blut-graph-core

`blut-graph-core` is the domain-neutral, `no_std + alloc` semantic compiler for
capability-driven ABIR node graphs. It verifies typed ports and capability,
proof, policy, fidelity, resource, effect, and target contracts before producing
a deterministic `CompiledPlan` shared by MCU AOT, host/stream, and BLUT durable
execution realms.

The crate owns no biosignal semantics, filesystem, network, async runtime, or
plugin process. Those remain implementation concerns behind the kernel,
transaction, and process-host traits.
