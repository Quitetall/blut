# Gate receipts

`war evidence record <alias>` writes receipts for an authorized Warrant's
cited gates. A probe with `war gate --run` is not a recorded acceptance
receipt, even though v1.0.0 writes transient receipt/stdout/stderr files here.
Those probe files are ignored by Git; governed evidence lives under each
Warrant's `gate-runs/`. Do not fabricate or hand-edit receipt contents;
preserve UNKNOWN when a gate cannot run.
