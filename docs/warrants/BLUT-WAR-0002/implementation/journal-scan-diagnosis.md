# Bounded journal secret-scan exception

Hosted job 114063301621 flagged generic-api-key matches on the two draft-created journal events (BLUT-WAR-0001 and BLUT-WAR-0002). Both idempotency_key values were independently recomputed as SHA-256(warrant_uuid + newline + type + newline + payload), matching OpenWarrant journal_cmd.rs. They are public event deduplication digests, not credentials.

Local installed gitleaks controls: original configuration refuses both journal copies (two findings); candidate configuration accepts both known records (zero findings); candidate still detects a different 64-character idempotency_key and a synthetic GitHub PAT in the same file (generic-api-key and github-pat, two findings). All scans redact values in retained logs and reports.

The first credential control used a sequential alphabet fixture that the installed detector did not recognize; that assertion failed and its log is retained. A deterministic non-sequential public synthetic marker was then detected. This is not evidence about real credential validity.

The exception covers only two exact field/value matches, not arbitrary digest shapes, paths, entire journals or detector rules. Original journal records remain unchanged. Hosted confirmation remains pending. Temporary copied journals and planted fixtures were removed.
