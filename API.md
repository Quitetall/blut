# BLUT — API reference

> **WIP stub.** Surfaces are stabilizing during the repo→module refactor (the
> Python pipeline — snn / dataset / oracle / decoder / student / common — just
> landed under `python/lamquant/`). This stub keeps the README link live; the
> full surface table lands once the package settles.

**Owns:** the universal trainer — compile-time typed DAG (stages → recipes), the TUI
cockpit, and the **Python data pipeline** (encode → label → split → load): manifest
builder, `LmaDataset`/`LmaL3Dataset` loaders, the hash-keyed L3 cache, seizure-split
builder. Consumes `LamQuant-Lossless` (codec) and `LamQuant-Neural` (models).

To document next: `blut` CLI + `blut tui`, recipe registry + `RecipeDef` contract,
dataset loader API, split-manifest schema, the L3 cache key/invalidently contract.

`.lml`/`.lma` format owned by `LamQuant-Lossless/API.md`; model defs by `LamQuant-Neural/API.md`.

See also: [README](README.md) · meta index [`../API.md`](../API.md).
