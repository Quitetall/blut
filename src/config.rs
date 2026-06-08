//! BLUT runtime configuration: locate llama.cpp tools + model
//! registry. Vendored from lamu-core's config.rs during the BLUT
//! repo split. Kept self-contained so BLUT has no upward Rust
//! dependency on the LAMU monorepo.

use std::path::PathBuf;

use crate::error::{Result, TrainError};

pub mod launcher;
pub mod sweep;

// Explicit imports ONLY — never `use lerna::*`: lerna re-exports its own
// `Launcher`/`BasicLauncher` at crate root, which would collide with the
// blut-owned `Launcher` trait in `launcher.rs`.
use lerna::{ConfigLoader, ConfigValue};

use crate::framework::artifact::ContentHash;
use crate::framework::cache::CacheHandle;

/// A composed + frozen configuration: the live `lerna` value, its
/// canonical JSON projection, and a content-hash fingerprint suitable
/// for cache keying / run identity.
#[derive(Clone, Debug)]
pub struct ResolvedConfig {
    /// The raw lerna config tree (defaults-list merged + overrides applied).
    pub value: ConfigValue,
    /// JSON freeze of `value` (see `config_to_json`).
    pub json: serde_json::Value,
    /// Domain-separated SHA-256 over the canonical JSON + code/data hashes.
    pub fingerprint: ContentHash,
}

/// Hydra-style compose: load `<config_name>` from `<config_dir>`, apply
/// `overrides` (defaults-list group selection + dotted value overrides +
/// `+add`/`~delete`), then freeze to JSON and fingerprint.
///
/// The defaults-list merge, group=config selection, dotted overrides and
/// `+`/`~` semantics are all handled by lerna's `load_config`; we do not
/// reimplement them.
pub fn compose(config_dir: &str, config_name: &str, overrides: &[String]) -> Result<ResolvedConfig> {
    let loader = ConfigLoader::from_config_dir(config_dir);
    let value = loader
        .load_config(Some(config_name), overrides)
        .map_err(|e| TrainError::other(format!("config compose: {e}")))?;
    let json = config_to_json(&value);
    // code_hash / data_hash are placeholders until a real code/data probe
    // lands; the fingerprint is still value-sensitive + order-invariant.
    let fingerprint = fingerprint_config(&json, ContentHash([0u8; 32]), ContentHash([0u8; 32]));
    Ok(ResolvedConfig {
        value,
        json,
        fingerprint,
    })
}

/// Freeze a `lerna::ConfigValue` tree to a `serde_json::Value`.
///
/// `ConfigValue` derives only `Clone`/`Debug`/`PartialEq` (no `Serialize`),
/// so this hand-written walker is the freeze core.
///
/// INTERPOLATION PASS-THROUGH: `lerna::load_config` only merges — it does
/// NOT resolve `${...}` interpolations — so an `Interpolation` lands here as
/// a literal string and is emitted as a JSON string. This means the frozen
/// JSON (and therefore the fingerprint) reflects UNRESOLVED interpolations.
/// TODO: wire `lerna::config::resolve` / a `ResolverContext` pre-freeze; note
/// this affects fingerprint reproducibility for interpolation-heavy configs.
fn config_to_json(v: &ConfigValue) -> serde_json::Value {
    use serde_json::Value;
    match v {
        ConfigValue::Null => Value::Null,
        ConfigValue::Bool(b) => Value::Bool(*b),
        ConfigValue::Int(i) => Value::Number((*i).into()),
        ConfigValue::Float(f) => {
            // JSON has no NaN/Inf. Map non-finite floats to their STRING form
            // (e.g. "inf", "NaN") rather than Null — collapsing to Null would
            // make a non-finite float collide with ConfigValue::Null in the
            // frozen JSON AND the fingerprint (distinct configs, same hash).
            serde_json::Number::from_f64(*f)
                .map_or_else(|| Value::String(f.to_string()), Value::Number)
        }
        ConfigValue::String(s) | ConfigValue::Interpolation(s) => Value::String(s.clone()),
        ConfigValue::List(l) => Value::Array(l.iter().map(config_to_json).collect()),
        ConfigValue::Dict(d) => Value::Object(
            d.iter()
                .map(|(k, vv)| (k.to_string(), config_to_json(vv)))
                .collect(),
        ),
        // Mirror lerna's `Display` rendering of a missing value.
        ConfigValue::Missing => Value::String("???".to_string()),
    }
}

/// Freeze a `ResolvedConfig` to a stable, pretty-printed JSON string.
pub fn freeze_to_json_string(rc: &ResolvedConfig) -> Result<String> {
    serde_json::to_string_pretty(&rc.json)
        .map_err(|e| TrainError::other(format!("config freeze: {e}")))
}

/// Domain-separated content fingerprint of a composed config.
///
/// This is the lane's "⊕ code/data hash": a domain-separated CONCAT into a
/// single SHA-256 (NOT a bitwise XOR — XOR would let a code-hash flip cancel
/// a data-hash flip). It reuses `CacheHandle::canonical_json_bytes` so the
/// config canonicalization is byte-identical to the cache's (key-sorted),
/// and SHA-256 over BLAKE3 deliberately (see `cache.rs`).
pub fn fingerprint_config(
    cfg_json: &serde_json::Value,
    code_hash: ContentHash,
    data_hash: ContentHash,
) -> ContentHash {
    use sha2::{Digest, Sha256};
    let canon = CacheHandle::canonical_json_bytes(cfg_json);
    let mut h = Sha256::new();
    h.update(b"blut.config.fingerprint.v1");
    h.update([0u8]);
    h.update(&canon);
    h.update(code_hash.0);
    h.update(data_hash.0);
    ContentHash(h.finalize().into())
}

/// Expand a sweep into one fingerprinted `SweepEntry` per cartesian combo.
/// Thin re-export over `sweep::expand` for callers that only `use crate::config`.
pub fn expand_and_fingerprint(
    config_dir: &str,
    config_name: &str,
    base_overrides: &[String],
    sweep_overrides: &[String],
) -> Result<Vec<sweep::SweepEntry>> {
    sweep::expand(config_dir, config_name, base_overrides, sweep_overrides)
}

/// Root of the user's `llama.cpp` checkout. Resolution order:
///   1. `$LAMU_LLAMACPP_DIR` env var (legacy, accepted for back-compat)
///   2. `$BLUT_LLAMACPP_DIR` env var (preferred new name)
///   3. `~/llama.cpp` (default for users who built from source in $HOME)
pub fn llamacpp_dir() -> PathBuf {
    if let Ok(p) = std::env::var("BLUT_LLAMACPP_DIR") {
        return PathBuf::from(p);
    }
    if let Ok(p) = std::env::var("LAMU_LLAMACPP_DIR") {
        return PathBuf::from(p);
    }
    dirs::home_dir().unwrap_or_default().join("llama.cpp")
}

/// Locate a llama.cpp tool by name. Tries:
///   1. `<llamacpp_dir>/build/bin/<name>` — standard cmake build layout
///   2. `<llamacpp_dir>/<name>` — flat layout / older builds
///   3. `<name>` on `$PATH` (via `which`)
pub fn llama_tool(name: &str) -> Result<PathBuf> {
    let base = llamacpp_dir();
    let candidates = [base.join("build").join("bin").join(name), base.join(name)];
    for c in candidates {
        if c.exists() {
            return Ok(c);
        }
    }
    if let Ok(p) = which::which(name) {
        return Ok(p);
    }
    Err(TrainError::Other(format!(
        "llama.cpp tool '{name}' not found in {} or on $PATH. \
         Set $BLUT_LLAMACPP_DIR to your llama.cpp checkout.",
        base.display()
    )))
}

/// Location of the BLUT model registry YAML.
///
/// Default: `~/.config/blut/registry.yaml`
/// Override: `$BLUT_REGISTRY_PATH` (preferred) or `$LAMU_REGISTRY_PATH`
/// (back-compat).
pub fn registry_path() -> PathBuf {
    if let Ok(p) = std::env::var("BLUT_REGISTRY_PATH") {
        return PathBuf::from(p);
    }
    if let Ok(p) = std::env::var("LAMU_REGISTRY_PATH") {
        return PathBuf::from(p);
    }
    dirs::config_dir()
        .unwrap_or_default()
        .join("blut")
        .join("registry.yaml")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn llamacpp_dir_default_under_home() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev_blut = std::env::var("BLUT_LLAMACPP_DIR").ok();
        let prev_lamu = std::env::var("LAMU_LLAMACPP_DIR").ok();
        // SAFETY: ENV_LOCK serializes mutations in this module.
        unsafe {
            std::env::remove_var("BLUT_LLAMACPP_DIR");
            std::env::remove_var("LAMU_LLAMACPP_DIR");
        }
        let p = llamacpp_dir();
        assert!(p.ends_with("llama.cpp"));
        unsafe {
            if let Some(v) = prev_blut {
                std::env::set_var("BLUT_LLAMACPP_DIR", v);
            }
            if let Some(v) = prev_lamu {
                std::env::set_var("LAMU_LLAMACPP_DIR", v);
            }
        }
    }

    #[test]
    fn llamacpp_dir_blut_env_takes_priority() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev_blut = std::env::var("BLUT_LLAMACPP_DIR").ok();
        let prev_lamu = std::env::var("LAMU_LLAMACPP_DIR").ok();
        unsafe {
            std::env::set_var("BLUT_LLAMACPP_DIR", "/opt/blut/llama");
            std::env::set_var("LAMU_LLAMACPP_DIR", "/opt/lamu/llama");
        }
        assert_eq!(llamacpp_dir(), PathBuf::from("/opt/blut/llama"));
        unsafe {
            match prev_blut {
                Some(v) => std::env::set_var("BLUT_LLAMACPP_DIR", v),
                None => std::env::remove_var("BLUT_LLAMACPP_DIR"),
            }
            match prev_lamu {
                Some(v) => std::env::set_var("LAMU_LLAMACPP_DIR", v),
                None => std::env::remove_var("LAMU_LLAMACPP_DIR"),
            }
        }
    }

    #[test]
    fn llama_tool_errors_with_blut_env_hint() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev = std::env::var("BLUT_LLAMACPP_DIR").ok();
        let dir = tempfile::tempdir().unwrap();
        unsafe {
            std::env::set_var("BLUT_LLAMACPP_DIR", dir.path());
        }
        let err = llama_tool("definitely-not-here").expect_err("must err");
        assert!(format!("{err}").contains("BLUT_LLAMACPP_DIR"));
        unsafe {
            match prev {
                Some(v) => std::env::set_var("BLUT_LLAMACPP_DIR", v),
                None => std::env::remove_var("BLUT_LLAMACPP_DIR"),
            }
        }
    }

    // --- LANE 2: config compose / freeze / fingerprint ---

    use lerna::ConfigDict;
    use std::io::Write;

    fn write_config(dir: &std::path::Path, name: &str, body: &str) {
        let mut f = std::fs::File::create(dir.join(name)).unwrap();
        f.write_all(body.as_bytes()).unwrap();
    }

    #[test]
    fn compose_applies_dotted_override() {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), "config.yaml", "db:\n  port: 3306\n");
        let rc = compose(
            dir.path().to_str().unwrap(),
            "config",
            &["db.port=5432".to_string()],
        )
        .unwrap();
        assert_eq!(rc.json["db"]["port"], serde_json::json!(5432));
    }

    #[test]
    fn freeze_json_round_trips_value_types() {
        let mut d = ConfigDict::new();
        d.insert("i".to_string(), ConfigValue::Int(7));
        d.insert("b".to_string(), ConfigValue::Bool(true));
        d.insert("f".to_string(), ConfigValue::Float(1.5));
        d.insert("s".to_string(), ConfigValue::String("hi".to_string()));
        d.insert(
            "interp".to_string(),
            ConfigValue::Interpolation("${x.y}".to_string()),
        );
        let json = config_to_json(&ConfigValue::Dict(d));
        assert_eq!(json["i"], serde_json::json!(7));
        assert_eq!(json["b"], serde_json::json!(true));
        assert_eq!(json["f"], serde_json::json!(1.5));
        assert_eq!(json["s"], serde_json::json!("hi"));
        // Interpolation passes through as a literal JSON string (unresolved).
        assert_eq!(json["interp"], serde_json::json!("${x.y}"));
    }

    #[test]
    fn fingerprint_is_deterministic_and_order_invariant() {
        let a = serde_json::json!({"lr": 1e-3, "bs": 16});
        let b = serde_json::json!({"bs": 16, "lr": 1e-3});
        let zero = ContentHash([0u8; 32]);
        let fa = fingerprint_config(&a, zero, zero);
        let fb = fingerprint_config(&b, zero, zero);
        assert_eq!(fa.0, fb.0, "key order must not change the fingerprint");

        let c = serde_json::json!({"lr": 1e-4, "bs": 16});
        let fc = fingerprint_config(&c, zero, zero);
        assert_ne!(fa.0, fc.0, "a changed value must change the fingerprint");
    }

    #[test]
    fn fingerprint_changes_with_code_hash() {
        let cfg = serde_json::json!({"lr": 1e-3});
        let f0 = fingerprint_config(&cfg, ContentHash([0u8; 32]), ContentHash([0u8; 32]));
        let f1 = fingerprint_config(&cfg, ContentHash([1u8; 32]), ContentHash([0u8; 32]));
        assert_ne!(
            f0.0, f1.0,
            "code_hash must contribute (domain-separated concat, not XOR-collapse)"
        );
    }

    #[test]
    fn freeze_to_json_string_is_pretty() {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), "config.yaml", "a: 1\nb: 2\n");
        let rc = compose(dir.path().to_str().unwrap(), "config", &[]).unwrap();
        let s = freeze_to_json_string(&rc).unwrap();
        assert!(s.contains('\n'), "pretty JSON must contain newlines");
        assert!(s.contains("\"a\""));
    }
}
