# Native BLUT receipt prototype — 2026-10-09

Source: 49ed0907c51dc33559e945f634ab43d16dc2d2d1. Four public CPU/receipt tests passed; optional-feature all-target Clippy and formatting passed on Rust 1.99.0. The actual executor produced the receipt. Synthetic in-memory test keys were used; no operator secret, human authorization or assurance mark was created.

Controls observed: actual clean CPU completion, cancelled attempt, existing-job preservation, wrong trusted key, wrong attempt, altered expected plan or producer binary, altered signed payload, changed native output and unavailable native job. The genuine RED was the public producer returning not-implemented; preceding cold-build deadlines and fixture compilation failures are retained separately.

This is bounded implementation evidence, not full integration or qualification. Additional warning/deadline/failure/malformed-authenticated-payload controls, provider transport, OpenWarrant adapter/callback, real recorded Dispatch integration, independent qualification and operator key custody remain. Retained test logs do not stand in for production receipts or a consenting-user study.

The worktree includes three pre-existing bootstrap commits (b4fc30d, b6cefbd, 6eedf20) above BLUT main; these are dependency history, not this prototype's implementation. No BLUT main merge is claimed.
