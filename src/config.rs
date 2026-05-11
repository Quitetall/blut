//! BLUT runtime configuration: locate llama.cpp tools + model
//! registry. Vendored from lamu-core's config.rs during the BLUT
//! repo split. Kept self-contained so BLUT has no upward Rust
//! dependency on the LAMU monorepo.

use std::path::PathBuf;

use crate::error::{Result, TrainError};

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
}
