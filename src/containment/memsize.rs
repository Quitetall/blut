// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Memory-size formatting/parsing + cgroup-knob resolution, shared by every
//! containment backend. Moved verbatim from the lamquant runner so the engine
//! owns the mechanism and all backends (systemd / cgroup2 / generic) format
//! caps identically. The systemd `Memory peak:` parser lives here too.

/// Resolve one cgroup memory knob: TYPED bytes (formatted) wins, then the env
/// var, then the hard-coded default. The resolution order is the ADR 0046
/// contract — typed field is the source of truth, env is a back-compat
/// fallback for callers that still set `MEMMAX`/etc.
pub fn resolve_mem_knob(typed: Option<u64>, env_key: &str, default: &str) -> String {
    // `Some(0)` is rejected, NOT honored: systemd reads `MemoryMax=0` as
    // "no limit" — a 0 cap would silently DEFEAT containment. Fall through to
    // env/default so a 0 can never produce an uncapped unit (ADR 0046 safety).
    if let Some(bytes) = typed.filter(|&b| b > 0) {
        return fmt_bytes_as_memsize(bytes);
    }
    std::env::var(env_key).unwrap_or_else(|_| default.to_string())
}

/// Resolve a cgroup memory knob to BYTES (for backends that write raw byte
/// counts, e.g. cgroup-v2 files): TYPED bytes (wins) → env var (parsed) →
/// default (parsed). The byte-returning analogue of [`resolve_mem_knob`]. Used
/// by RSS-capping backends that can safely apply a default; NOT by the
/// `RLIMIT_AS` backend (which caps virtual address space and must never apply
/// a default that would break a CUDA process's huge VA reservations).
pub fn resolve_mem_bytes(typed: Option<u64>, env_key: &str, default: &str) -> Option<u64> {
    if let Some(bytes) = typed.filter(|&b| b > 0) {
        return Some(bytes);
    }
    if let Ok(s) = std::env::var(env_key)
        && let Some(b) = parse_memsize(&s)
    {
        return Some(b);
    }
    parse_memsize(default)
}

/// Format a byte count as a systemd memory-size string. Emits an exact integer
/// suffix when the value divides evenly (`<n>G` / `<n>M` / `<n>K`) so the
/// cgroup limit is byte-precise; otherwise emits the raw byte count (systemd
/// accepts a bare integer as bytes). Never lossy.
pub fn fmt_bytes_as_memsize(bytes: u64) -> String {
    const K: u64 = 1024;
    const M: u64 = 1024 * 1024;
    const G: u64 = 1024 * 1024 * 1024;
    if bytes != 0 && bytes.is_multiple_of(G) {
        format!("{}G", bytes / G)
    } else if bytes != 0 && bytes.is_multiple_of(M) {
        format!("{}M", bytes / M)
    } else if bytes != 0 && bytes.is_multiple_of(K) {
        format!("{}K", bytes / K)
    } else {
        // Bare integer = bytes in systemd's MemoryMax= grammar.
        bytes.to_string()
    }
}

/// Parse a systemd memory-size string to bytes — the inverse of
/// [`fmt_bytes_as_memsize`]. Accepts an exact suffix (`16G` / `256M` / `8K`)
/// or a bare integer (bytes). Returns `None` for anything else (empty,
/// non-numeric, unknown suffix) so a malformed line never poisons the
/// calibration store.
///
/// systemd's `Memory peak:` value is emitted with IEC suffixes (powers of
/// 1024), matching `fmt_bytes_as_memsize`'s base. A fractional value (e.g.
/// `16.2G`) is parsed too — the float is multiplied and truncated.
pub fn parse_memsize(s: &str) -> Option<u64> {
    const K: u64 = 1024;
    const M: u64 = 1024 * 1024;
    const G: u64 = 1024 * 1024 * 1024;
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    // Bare integer = bytes.
    if let Ok(bytes) = s.parse::<u64>() {
        return Some(bytes);
    }
    // <number><suffix>. Suffix is the final char; the rest is the number.
    let (num_part, mult) = match s.chars().last()? {
        'G' | 'g' => (&s[..s.len() - 1], G),
        'M' | 'm' => (&s[..s.len() - 1], M),
        'K' | 'k' => (&s[..s.len() - 1], K),
        // A trailing 'B' (e.g. "16GB"/"512MB") — strip it and recurse on
        // the suffix before it.
        'B' | 'b' if s.len() >= 2 => {
            return parse_memsize(&s[..s.len() - 1]);
        }
        _ => return None,
    };
    let num: f64 = num_part.trim().parse().ok()?;
    if !num.is_finite() || num < 0.0 {
        return None;
    }
    // Truncate toward zero; the value is already conservative-high as a peak,
    // and the broker MAX-merges, so flooring can't under-report a larger later
    // sample.
    Some((num * mult as f64) as u64)
}

/// Extract the byte count from a systemd-run client `Memory peak: <N>` stderr
/// line. Tolerant of leading whitespace / log prefixes: it finds the
/// `Memory peak:` marker anywhere in the line and parses the token after it.
/// Returns `None` for any other line.
pub fn parse_memory_peak_line(line: &str) -> Option<u64> {
    let marker = "Memory peak:";
    let idx = line.find(marker)?;
    let rest = line[idx + marker.len()..].trim();
    // The value is the first whitespace-delimited token after the marker.
    let token = rest.split_whitespace().next()?;
    parse_memsize(token)
}

/// Whether a systemd-run client stderr line reports the unit's result, and if
/// so whether that result is an OOM kill. `systemd-run --wait` prints
/// `Finished with result: <result>` when the unit stops; the result is
/// `oom-kill` when the kernel OOM-killed a process of the unit. `None` for any
/// other line.
pub fn parse_unit_result_oom(line: &str) -> Option<bool> {
    let marker = "Finished with result:";
    let idx = line.find(marker)?;
    let result = line[idx + marker.len()..].split_whitespace().next()?;
    Some(result == "oom-kill")
}

/// The `oom_kill` counter of a cgroup-v2 `memory.events` file: the number of
/// processes in the cgroup the kernel OOM-killed.
pub fn parse_memory_events_oom_kill(events: &str) -> Option<u64> {
    events.lines().find_map(|line| {
        let mut fields = line.split_whitespace();
        (fields.next() == Some("oom_kill")).then(|| fields.next()?.parse().ok())?
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_exact_suffixes() {
        assert_eq!(fmt_bytes_as_memsize(16 * 1024 * 1024 * 1024), "16G");
        assert_eq!(fmt_bytes_as_memsize(256 * 1024 * 1024), "256M");
        assert_eq!(fmt_bytes_as_memsize(8 * 1024), "8K");
        // Non-round → bare bytes.
        assert_eq!(fmt_bytes_as_memsize(1234567), "1234567");
    }

    #[test]
    fn parse_roundtrips() {
        assert_eq!(parse_memsize("16G"), Some(16 * 1024 * 1024 * 1024));
        assert_eq!(parse_memsize("512MB"), Some(512 * 1024 * 1024));
        assert_eq!(parse_memsize("1024"), Some(1024));
        assert_eq!(parse_memsize(""), None);
        assert_eq!(parse_memsize("garbage"), None);
    }

    #[test]
    fn peak_line_extracts() {
        assert_eq!(
            parse_memory_peak_line("Memory peak: 16.2G"),
            Some((16.2 * 1024.0 * 1024.0 * 1024.0) as u64)
        );
        assert_eq!(parse_memory_peak_line("no marker here"), None);
    }

    #[test]
    fn resolve_rejects_zero() {
        // Some(0) must NOT produce "0" (systemd = unlimited); falls to default.
        assert_eq!(resolve_mem_knob(Some(0), "NONEXISTENT_ENV_X", "44G"), "44G");
        assert_eq!(
            resolve_mem_knob(Some(8 * 1024 * 1024 * 1024), "X", "44G"),
            "8G"
        );
    }

    #[test]
    fn unit_result_line_tells_an_oom_kill_from_other_results() {
        assert_eq!(
            parse_unit_result_oom("Finished with result: oom-kill"),
            Some(true)
        );
        assert_eq!(
            parse_unit_result_oom("Finished with result: exit-code"),
            Some(false)
        );
        assert_eq!(
            parse_unit_result_oom("Finished with result: success"),
            Some(false)
        );
        assert_eq!(parse_unit_result_oom("MemoryError: out of memory"), None);
        assert_eq!(parse_unit_result_oom("Finished with result:"), None);
    }

    #[test]
    fn memory_events_oom_kill_is_read_by_exact_key() {
        let events = "low 0\nhigh 12\nmax 40\noom 3\noom_kill 2\noom_group_kill 1\n";
        assert_eq!(parse_memory_events_oom_kill(events), Some(2));
        assert_eq!(
            parse_memory_events_oom_kill("oom 3\noom_group_kill 1\n"),
            None
        );
        assert_eq!(parse_memory_events_oom_kill("oom_kill x\n"), None);
    }
}
