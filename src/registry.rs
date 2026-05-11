//! Minimal model registry for BLUT.
//!
//! After a successful training run, `register_model` writes here
//! so subsequent `blut` invocations + LAMU (when integrated) can
//! discover the new model. YAML format identical to LAMU's
//! `lamu_core::registry` so the two can share a registry file by
//! pointing `$BLUT_REGISTRY_PATH` at LAMU's `models.yaml`.
//!
//! Atomic writes via tmp+rename so a crash mid-write never
//! corrupts the file.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Result, TrainError};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    #[default]
    Gguf,
    SafeTensors,
    Hf,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendType {
    #[default]
    LlamaCpp,
    Megakernel,
    Dflash,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Chat,
    Code,
    Reasoning,
    Vision,
    Embedding,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStatus {
    #[default]
    Available,
    Deprecated,
    Broken,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub path: PathBuf,
    pub format: ModelFormat,
    pub backend: BackendType,
    pub arch: String,
    pub params_b: f32,
    pub quant: String,
    pub vram_mb: u32,
    pub context_max: u32,
    pub capabilities: Vec<Capability>,
    #[serde(default)]
    pub notes: String,
    #[serde(default)]
    pub status: ModelStatus,
}

#[derive(Debug, Serialize, Deserialize)]
struct RegistryFile {
    models: HashMap<String, ModelEntry>,
}

/// Load the registry from `path`. Returns an empty Vec if the
/// file doesn't exist.
pub fn load_registry(path: &Path) -> Result<Vec<ModelEntry>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let body = std::fs::read_to_string(path).map_err(|source| TrainError::Io {
        path: path.into(),
        source,
    })?;
    let parsed: RegistryFile = serde_yaml::from_str(&body)
        .map_err(|e| TrainError::Registry(format!("parse {}: {e}", path.display())))?;
    let mut entries: Vec<ModelEntry> = parsed
        .models
        .into_iter()
        .map(|(name, mut entry)| {
            // The HashMap key wins if the entry's `name` field is
            // stale.
            entry.name = name;
            entry
        })
        .collect();
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(entries)
}

/// Append (or replace, if `replace=true`) a single entry. Atomic
/// write — failure mid-write never corrupts the file.
pub fn add_entry(entry: ModelEntry, path: &Path, replace: bool) -> Result<()> {
    let mut current = load_registry(path)?;
    let existing_idx = current.iter().position(|e| e.name == entry.name);
    match existing_idx {
        Some(_) if !replace => {
            return Err(TrainError::Registry(format!(
                "entry '{}' already in registry; pass replace=true to overwrite",
                entry.name
            )));
        }
        Some(i) => {
            current[i] = entry;
        }
        None => {
            current.push(entry);
        }
    }
    write_registry(&current, path)
}

fn write_registry(entries: &[ModelEntry], path: &Path) -> Result<()> {
    use std::io::Write;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent).map_err(|source| TrainError::Io {
        path: parent.into(),
        source,
    })?;
    let map: HashMap<String, ModelEntry> = entries
        .iter()
        .map(|e| (e.name.clone(), e.clone()))
        .collect();
    let file = RegistryFile { models: map };
    let yaml = serde_yaml::to_string(&file)
        .map_err(|e| TrainError::Registry(format!("serialize: {e}")))?;
    let tmp = path.with_extension("yaml.tmp");
    {
        let mut f = std::fs::File::create(&tmp).map_err(|source| TrainError::Io {
            path: tmp.clone(),
            source,
        })?;
        f.write_all(yaml.as_bytes()).map_err(|source| TrainError::Io {
            path: tmp.clone(),
            source,
        })?;
        let _ = f.sync_all();
    }
    if let Err(source) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(TrainError::Io {
            path: path.into(),
            source,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ModelEntry {
        ModelEntry {
            name: "demo".into(),
            path: PathBuf::from("/tmp/demo.gguf"),
            format: ModelFormat::Gguf,
            backend: BackendType::LlamaCpp,
            arch: "qwen3".into(),
            params_b: 7.0,
            quant: "Q4_K_M".into(),
            vram_mb: 0,
            context_max: 4096,
            capabilities: vec![Capability::Chat],
            notes: String::new(),
            status: ModelStatus::default(),
        }
    }

    #[test]
    fn add_then_load_round_trip() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("registry.yaml");
        add_entry(sample(), &path, false).unwrap();
        let entries = load_registry(&path).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "demo");
    }

    #[test]
    fn add_entry_refuses_duplicate_without_replace() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("registry.yaml");
        add_entry(sample(), &path, false).unwrap();
        let r = add_entry(sample(), &path, false);
        assert!(matches!(r, Err(TrainError::Registry(_))));
    }

    #[test]
    fn add_entry_replaces_with_replace_flag() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("registry.yaml");
        add_entry(sample(), &path, false).unwrap();
        let mut updated = sample();
        updated.notes = "updated".into();
        add_entry(updated, &path, true).unwrap();
        let entries = load_registry(&path).unwrap();
        assert_eq!(entries[0].notes, "updated");
    }

    #[test]
    fn load_returns_empty_when_file_missing() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("missing.yaml");
        let entries = load_registry(&path).unwrap();
        assert!(entries.is_empty());
    }
}
