---
name: Feature request
about: Propose a new capability or a generic seam in the blut engine
title: "[feature] "
labels: enhancement
---

## Problem / motivation
<!-- What are you trying to do that blut makes hard or impossible today? -->

## Proposed solution
<!-- What you'd like to see. If it touches the public API, sketch the surface. -->

## Alternatives considered
<!-- Other ways you've tried or thought about. -->

## Engine vs cookbook
<!-- IMPORTANT: blut is a domain-AGNOSTIC framework — it ships no concrete
     recipes/stages/backends. Domain-specific logic (a particular model, data
     format, training loop) belongs in a downstream COOKBOOK crate that depends
     on blut, not in the engine. Does your request need a new GENERIC seam in
     the engine, or is it cookbook-level work? If you're unsure, describe the
     seam (the trait / hook you need) rather than the domain use case. -->
