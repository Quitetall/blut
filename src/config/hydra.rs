// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Native Hydra-style config compose — the blut-owned replacement for the
//! former `lerna` git-dependency (removed so the engine can publish to
//! crates.io, which forbids git deps).
//!
//! ## What this is
//! A self-contained implementation of the slice of
//! [Hydra](https://github.com/facebookresearch/hydra)'s configuration model
//! that BLUT's `recipe run --config-dir/--set/--sweep` mode uses. The only
//! runtime dependency is `serde_yaml` (already in the tree). The public
//! surface — [`ConfigValue`], [`ConfigDict`], [`ConfigLoader`],
//! [`expand_simple_sweeps`] — matches what the engine previously imported from
//! `lerna`, so `config.rs` / `config/sweep.rs` changed only their import line.
//!
//! ## Hydra semantics covered
//!   * YAML config loading from a config dir (`<dir>/<name>.yaml`)
//!   * the `defaults:` list — compose order, `_self_`, `group: option`, and
//!     `optional` entries
//!   * config groups (`<dir>/<group>/<option>.yaml`) with `# @package`
//!     headers (`_global_` → root, `a.b` → nested package path)
//!   * the override grammar: `key=val`, `+key=val` (add), `~key` (delete),
//!     dotted paths (`a.b.c=val`), and dot-less `group=option` defaults-list
//!     selection
//!   * typed value parsing (bool / null / int / float / `[list]` / quoted str)
//!   * sweep expansion: choice (`k=a,b,c`) + `range(start,stop[,step])`
//!
//! ## Hydra compatibility — known edges (additive, API stays stable)
//! These are PARSED-and-preserved or not-yet-handled, so a real Hydra tree
//! composes identically *except* where it relies on them:
//!   * `${...}` interpolation is preserved as a literal [`ConfigValue::Interpolation`]
//!     and NOT resolved (matches the prior dependency; `config.rs` freezes the
//!     unresolved form — see its `config_to_json` note).
//!   * `_self_` is recognised but the primary config is always merged LAST
//!     (after the defaults list), rather than at the `_self_` position.
//!   * `++key` force-add, glob sweeps, and `@package _group_`/`_here_`
//!     keywords are not handled.
//!
//! The compose result for the inputs BLUT actually feeds (a primary YAML +
//! dotted value overrides + cartesian sweeps) is byte-stable, which is what
//! the config fingerprint in `config.rs` keys on.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Failure composing a config tree. Display-only context (path + cause) so a
/// `recipe run` reader can tell *which* config file broke without a backtrace.
#[derive(Debug, thiserror::Error)]
pub enum HydraError {
    /// A referenced config (`<name>.yaml` or a `group/option.yaml`) was not
    /// found under the config dir.
    #[error("config '{path}' not found under '{dir}'")]
    NotFound { dir: String, path: String },
    /// The config file could not be read.
    #[error("read config '{path}': {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    /// The config file was not valid YAML.
    #[error("parse YAML '{path}': {source}")]
    Yaml {
        path: String,
        #[source]
        source: serde_yaml::Error,
    },
    /// A config file parsed to a non-mapping top-level value (Hydra configs
    /// must be a mapping at the root).
    #[error("config '{path}' top-level is not a mapping")]
    NotAMapping { path: String },
}

// ---------------------------------------------------------------------------
// ConfigValue / ConfigDict — the composed-config value model
// ---------------------------------------------------------------------------

/// A configuration value: one node of a composed config tree. Mirrors the YAML
/// scalar/collection types plus Hydra's `Interpolation`/`Missing` markers.
#[derive(Clone, Debug, PartialEq, Default)]
pub enum ConfigValue {
    /// Null / `~` / None.
    #[default]
    Null,
    /// Boolean.
    Bool(bool),
    /// 64-bit signed integer.
    Int(i64),
    /// 64-bit float.
    Float(f64),
    /// Plain string (no `${...}`).
    String(String),
    /// Ordered list of values.
    List(Vec<ConfigValue>),
    /// Insertion-ordered mapping.
    Dict(ConfigDict),
    /// An unresolved interpolation, e.g. `"${foo.bar}"` (preserved literally).
    Interpolation(String),
    /// Hydra's mandatory-but-unset marker (`???`).
    Missing,
}

impl ConfigValue {
    /// `true` for [`ConfigValue::Null`].
    pub fn is_null(&self) -> bool {
        matches!(self, ConfigValue::Null)
    }

    /// `true` for [`ConfigValue::Missing`].
    pub fn is_missing(&self) -> bool {
        matches!(self, ConfigValue::Missing)
    }

    /// `true` for [`ConfigValue::Interpolation`].
    pub fn is_interpolation(&self) -> bool {
        matches!(self, ConfigValue::Interpolation(_))
    }

    /// Borrow as bool (only when this is a [`ConfigValue::Bool`]).
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ConfigValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Borrow as int (only when this is a [`ConfigValue::Int`]).
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ConfigValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Borrow as float (an `Int` widens to float, matching YAML numeric reads).
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ConfigValue::Float(f) => Some(*f),
            ConfigValue::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Borrow as `&str` (string or interpolation literal).
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ConfigValue::String(s) | ConfigValue::Interpolation(s) => Some(s),
            _ => None,
        }
    }

    /// Borrow as list.
    pub fn as_list(&self) -> Option<&Vec<ConfigValue>> {
        match self {
            ConfigValue::List(l) => Some(l),
            _ => None,
        }
    }

    /// Borrow as dict.
    pub fn as_dict(&self) -> Option<&ConfigDict> {
        match self {
            ConfigValue::Dict(d) => Some(d),
            _ => None,
        }
    }

    /// Mutably borrow as dict.
    pub fn as_dict_mut(&mut self) -> Option<&mut ConfigDict> {
        match self {
            ConfigValue::Dict(d) => Some(d),
            _ => None,
        }
    }
}

impl From<bool> for ConfigValue {
    fn from(b: bool) -> Self {
        ConfigValue::Bool(b)
    }
}
impl From<i64> for ConfigValue {
    fn from(i: i64) -> Self {
        ConfigValue::Int(i)
    }
}
impl From<i32> for ConfigValue {
    fn from(i: i32) -> Self {
        ConfigValue::Int(i as i64)
    }
}
impl From<f64> for ConfigValue {
    fn from(f: f64) -> Self {
        ConfigValue::Float(f)
    }
}
impl From<String> for ConfigValue {
    /// A string containing a `${...}` span becomes an [`ConfigValue::Interpolation`].
    fn from(s: String) -> Self {
        if is_interpolation_literal(&s) {
            ConfigValue::Interpolation(s)
        } else {
            ConfigValue::String(s)
        }
    }
}
impl From<&str> for ConfigValue {
    fn from(s: &str) -> Self {
        ConfigValue::from(s.to_string())
    }
}
impl From<Vec<ConfigValue>> for ConfigValue {
    fn from(v: Vec<ConfigValue>) -> Self {
        ConfigValue::List(v)
    }
}
impl From<ConfigDict> for ConfigValue {
    fn from(d: ConfigDict) -> Self {
        ConfigValue::Dict(d)
    }
}

impl fmt::Display for ConfigValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigValue::Null => write!(f, "null"),
            ConfigValue::Bool(b) => write!(f, "{b}"),
            ConfigValue::Int(i) => write!(f, "{i}"),
            ConfigValue::Float(fl) => write!(f, "{fl}"),
            ConfigValue::String(s) | ConfigValue::Interpolation(s) => write!(f, "{s}"),
            ConfigValue::List(l) => {
                write!(f, "[")?;
                for (i, v) in l.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "]")
            }
            ConfigValue::Dict(d) => write!(f, "{d:?}"),
            ConfigValue::Missing => write!(f, "???"),
        }
    }
}

/// `true` if a string carries a `${...}` interpolation span.
fn is_interpolation_literal(s: &str) -> bool {
    if let Some(start) = s.find("${") {
        s[start + 2..].contains('}')
    } else {
        false
    }
}

/// An insertion-ordered string→value map (the dict node of a config tree).
///
/// Order is preserved so the JSON freeze + content fingerprint in `config.rs`
/// are deterministic for a given compose. A side `index` keeps lookups O(1).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConfigDict {
    entries: Vec<(String, ConfigValue)>,
    index: HashMap<String, usize>,
}

impl ConfigDict {
    /// Empty dict.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or overwrite `key`. A new key appends (preserving order); an
    /// existing key updates in place.
    pub fn insert(&mut self, key: String, value: ConfigValue) {
        if let Some(&idx) = self.index.get(&key) {
            self.entries[idx].1 = value;
        } else {
            let idx = self.entries.len();
            self.entries.push((key.clone(), value));
            self.index.insert(key, idx);
        }
    }

    /// Borrow the value at `key`.
    pub fn get(&self, key: &str) -> Option<&ConfigValue> {
        self.index.get(key).map(|&idx| &self.entries[idx].1)
    }

    /// Mutably borrow the value at `key`.
    pub fn get_mut(&mut self, key: &str) -> Option<&mut ConfigValue> {
        match self.index.get(key) {
            Some(&idx) => Some(&mut self.entries[idx].1),
            None => None,
        }
    }

    /// `true` if `key` is present (and live).
    pub fn contains_key(&self, key: &str) -> bool {
        self.index.contains_key(key)
    }

    /// Remove `key`, returning its prior value. The entry is fully removed and
    /// the index rebuilt, so the remaining keys keep their insertion order AND a
    /// later re-insert of `key` appends cleanly. (A tombstone-keep design would
    /// double-emit the key from `iter`/`keys` after a re-insert, breaking the
    /// deterministic JSON freeze; dicts are small, so the O(n) rebuild is moot.)
    pub fn remove(&mut self, key: &str) -> Option<ConfigValue> {
        let idx = self.index.remove(key)?;
        let (_, old) = self.entries.remove(idx);
        self.index.clear();
        for (i, (k, _)) in self.entries.iter().enumerate() {
            self.index.insert(k.clone(), i);
        }
        Some(old)
    }

    /// Number of live entries.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// `true` if there are no live entries.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Iterate live `(key, value)` pairs in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ConfigValue)> {
        self.entries
            .iter()
            .filter(|(k, _)| self.index.contains_key(k))
            .map(|(k, v)| (k.as_str(), v))
    }

    /// Live keys, in insertion order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.iter().map(|(k, _)| k)
    }

    /// Live values, in insertion order.
    pub fn values(&self) -> impl Iterator<Item = &ConfigValue> {
        self.iter().map(|(_, v)| v)
    }

    /// Resolve a dotted path (`"a.b.c"`) to a cloned value, or `None` if any
    /// segment is missing / not a dict.
    pub fn select(&self, path: &str) -> Option<ConfigValue> {
        let parts: Vec<&str> = path.split('.').collect();
        self.select_parts(&parts)
    }

    fn select_parts(&self, parts: &[&str]) -> Option<ConfigValue> {
        let (key, rest) = parts.split_first()?;
        let value = self.get(key)?;
        if rest.is_empty() {
            return Some(value.clone());
        }
        match value {
            ConfigValue::Dict(d) => d.select_parts(rest),
            _ => None,
        }
    }

    /// Deep-merge `other` INTO `self`: dict-vs-dict recurses; every other pair
    /// (including list-vs-list) replaces. Matches Hydra's default merge.
    pub fn merge(&mut self, other: &ConfigDict) {
        for (key, value) in other.iter() {
            let deep = matches!(
                (self.get(key), value),
                (Some(ConfigValue::Dict(_)), ConfigValue::Dict(_))
            );
            if deep {
                if let (ConfigValue::Dict(other_dict), Some(ConfigValue::Dict(self_dict))) =
                    (value, self.get_mut(key))
                {
                    self_dict.merge(other_dict);
                    continue;
                }
            }
            self.insert(key.to_string(), value.clone());
        }
    }
}

// ---------------------------------------------------------------------------
// ConfigLoader — Hydra compose over a single config directory
// ---------------------------------------------------------------------------

/// One loaded config file: its mapping body plus the `# @key value` header
/// directives parsed from the leading comment block (notably `@package`).
struct LoadedConfig {
    dict: ConfigDict,
    header: HashMap<String, String>,
}

/// Loads + composes configs rooted at a single config directory.
///
/// Mirrors Hydra's compose: read `<name>.yaml`, apply its `defaults:` list
/// (with group selection from dot-less overrides), merge group configs at
/// their `@package`, merge the primary body last, then apply value overrides.
pub struct ConfigLoader {
    config_dir: PathBuf,
}

impl ConfigLoader {
    /// A loader rooted at `config_dir` (configs resolved relative to it).
    pub fn from_config_dir(config_dir: &str) -> Self {
        Self {
            config_dir: PathBuf::from(config_dir),
        }
    }

    /// Compose `<config_name>` with `overrides` applied, returning the merged
    /// tree as a [`ConfigValue::Dict`].
    ///
    /// Overrides split into *defaults-list selections* (dot-less `group=opt`,
    /// applied to the `defaults:` list) and *value overrides* (everything
    /// else: `a.b=v`, `+a=v`, `~a`), applied after the merge.
    pub fn load_config(
        &self,
        config_name: Option<&str>,
        overrides: &[String],
    ) -> Result<ConfigValue, HydraError> {
        let (default_overrides, value_overrides): (Vec<&String>, Vec<&String>) =
            overrides.iter().partition(|o| is_default_override(o));
        let default_map = build_default_override_map(&default_overrides);

        let mut merged = ConfigDict::new();

        if let Some(name) = config_name {
            let primary = self.load_single_config(name)?;

            if let Some(ConfigValue::List(defaults)) = primary.dict.get("defaults") {
                let modified = apply_default_overrides(defaults, &default_map);
                self.process_defaults(&modified, &mut merged)?;
            }
            // Merge the primary body (minus its `defaults:` directive) LAST.
            for (key, value) in primary.dict.iter() {
                if key != "defaults" {
                    // A dict body deep-merges over anything the defaults set.
                    if let (ConfigValue::Dict(src), Some(ConfigValue::Dict(dst))) =
                        (value, merged.get_mut(key))
                    {
                        dst.merge(src);
                    } else {
                        merged.insert(key.to_string(), value.clone());
                    }
                }
            }
        }

        for ov in &value_overrides {
            apply_override(&mut merged, ov);
        }

        Ok(ConfigValue::Dict(merged))
    }

    /// `true` if `config_name` (or `group/option`) resolves to a file.
    pub fn config_exists(&self, config_path: &str) -> bool {
        self.resolve_path(config_path).is_some()
    }

    /// List the option names in a config group (`<dir>/<group>/*.yaml`).
    pub fn list_group(&self, group_path: &str) -> Vec<String> {
        let dir = self.config_dir.join(group_path);
        let mut items = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for entry in rd.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("yaml") {
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        items.push(stem.to_string());
                    }
                }
            }
        }
        items.sort();
        items.dedup();
        items
    }

    /// Resolve a config path (`name` or `group/option`) to a `.yaml` / `.yml`
    /// file under the config dir.
    fn resolve_path(&self, config_path: &str) -> Option<PathBuf> {
        for ext in ["yaml", "yml"] {
            let p = self.config_dir.join(format!("{config_path}.{ext}"));
            if p.exists() {
                return Some(p);
            }
        }
        None
    }

    /// Read + parse a single config file to its mapping body + header.
    fn load_single_config(&self, config_path: &str) -> Result<LoadedConfig, HydraError> {
        let path = self
            .resolve_path(config_path)
            .ok_or_else(|| HydraError::NotFound {
                dir: self.config_dir.display().to_string(),
                path: config_path.to_string(),
            })?;
        let content = std::fs::read_to_string(&path).map_err(|e| HydraError::Io {
            path: path.display().to_string(),
            source: e,
        })?;
        let header = extract_header(&content);
        let yaml: serde_yaml::Value =
            serde_yaml::from_str(&content).map_err(|e| HydraError::Yaml {
                path: path.display().to_string(),
                source: e,
            })?;
        // An empty document parses to Null — treat as an empty mapping.
        let dict = match yaml_to_config_value(&yaml) {
            ConfigValue::Dict(d) => d,
            ConfigValue::Null => ConfigDict::new(),
            _ => {
                return Err(HydraError::NotAMapping {
                    path: path.display().to_string(),
                });
            }
        };
        Ok(LoadedConfig { dict, header })
    }

    /// Apply a `defaults:` list to `merged`: each entry pulls in a config (a
    /// bare name at root, or a `group: option` at its `@package`).
    fn process_defaults(
        &self,
        defaults: &[ConfigValue],
        merged: &mut ConfigDict,
    ) -> Result<(), HydraError> {
        for default in defaults {
            match default {
                // Bare config name (or `_self_`, which we honour by merging the
                // primary body last — see module docs).
                ConfigValue::String(name) => {
                    if name != "_self_" {
                        let cfg = self.load_single_config(name)?;
                        merged.merge(&cfg.dict);
                    }
                }
                // `group: option` selection (possibly several per entry).
                ConfigValue::Dict(dict) => {
                    let is_optional = dict
                        .get("optional")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    for (group, value) in dict.iter() {
                        if group == "optional" {
                            continue;
                        }
                        let option = match value {
                            ConfigValue::String(s) => s.clone(),
                            // `group: null` means "skip this group".
                            _ => continue,
                        };
                        let path = format!("{group}/{option}");
                        match self.load_single_config(&path) {
                            Ok(cfg) => {
                                let package = cfg
                                    .header
                                    .get("package")
                                    .map(String::as_str)
                                    .unwrap_or(group);
                                merge_at_package(merged, &cfg.dict, package);
                            }
                            Err(e) => {
                                if !is_optional {
                                    return Err(e);
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// `true` for a defaults-list selection override (`group=option`): has `=`, a
/// dot-less key, and no `+`/`~` prefix.
fn is_default_override(override_str: &str) -> bool {
    match override_str.find('=') {
        Some(eq) => {
            let key = &override_str[..eq];
            !key.contains('.') && !key.starts_with('+') && !key.starts_with('~')
        }
        None => false,
    }
}

/// Map `group -> option` from dot-less selection overrides.
fn build_default_override_map(overrides: &[&String]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for o in overrides {
        if let Some(eq) = o.find('=') {
            map.insert(o[..eq].to_string(), o[eq + 1..].to_string());
        }
    }
    map
}

/// Rewrite a `defaults:` list, replacing each `group: option` whose group is
/// selected on the CLI with the chosen option.
fn apply_default_overrides(
    defaults: &[ConfigValue],
    override_map: &HashMap<String, String>,
) -> Vec<ConfigValue> {
    defaults
        .iter()
        .map(|default| match default {
            ConfigValue::Dict(dict) => {
                let mut new_dict = ConfigDict::new();
                for (group, value) in dict.iter() {
                    match override_map.get(group) {
                        Some(opt) => {
                            new_dict.insert(group.to_string(), ConfigValue::String(opt.clone()));
                        }
                        None => new_dict.insert(group.to_string(), value.clone()),
                    }
                }
                ConfigValue::Dict(new_dict)
            }
            other => other.clone(),
        })
        .collect()
}

/// Merge `source` into `target` at a `@package` path: `_global_`/empty → root,
/// otherwise nest under the dotted package.
fn merge_at_package(target: &mut ConfigDict, source: &ConfigDict, package: &str) {
    if package == "_global_" || package.is_empty() {
        target.merge(source);
        return;
    }
    let parts: Vec<&str> = package.split('.').collect();
    merge_at_path(target, source, &parts);
}

fn merge_at_path(target: &mut ConfigDict, source: &ConfigDict, path: &[&str]) {
    let Some((key, rest)) = path.split_first() else {
        target.merge(source);
        return;
    };
    if !target.contains_key(key) {
        target.insert(key.to_string(), ConfigValue::Dict(ConfigDict::new()));
    }
    if let Some(ConfigValue::Dict(nested)) = target.get_mut(key) {
        merge_at_path(nested, source, rest);
    }
}

/// Apply one value override (`key=val`, `+key=val`, `~key[=val]`) to `config`.
fn apply_override(config: &mut ConfigDict, override_str: &str) {
    // Deletion: `~key` (with or without a trailing `=value`).
    if let Some(rest) = override_str.strip_prefix('~') {
        let key = rest.split('=').next().unwrap_or(rest);
        delete_at_path(config, key);
        return;
    }
    let Some(eq) = override_str.find('=') else {
        return;
    };
    let key = &override_str[..eq];
    let value = parse_override_value(&override_str[eq + 1..]);
    // `+key` adds (creating intermediate dicts); plain `key` sets where the
    // path already exists.
    let (key, create) = match key.strip_prefix('+') {
        Some(k) => (k, true),
        None => (key, false),
    };
    set_at_path(config, key, value, create);
}

/// Parse a CLI override value string into a typed [`ConfigValue`].
fn parse_override_value(value_str: &str) -> ConfigValue {
    let trimmed = value_str.trim();
    match trimmed {
        "true" => return ConfigValue::Bool(true),
        "false" => return ConfigValue::Bool(false),
        "null" | "~" => return ConfigValue::Null,
        "???" => return ConfigValue::Missing,
        _ => {}
    }
    if let Ok(i) = trimmed.parse::<i64>() {
        return ConfigValue::Int(i);
    }
    // Require a digit before accepting a float, so bare `inf`/`nan`/`infinity`
    // (which `f64::from_str` DOES accept) stay strings — matching Hydra, and
    // keeping a `nan` from poisoning `ConfigValue`'s `PartialEq`.
    if trimmed.bytes().any(|b| b.is_ascii_digit()) {
        if let Ok(f) = trimmed.parse::<f64>() {
            return ConfigValue::Float(f);
        }
    }
    if trimmed.starts_with('[') && trimmed.ends_with(']') {
        let inner = &trimmed[1..trimmed.len() - 1];
        if inner.trim().is_empty() {
            return ConfigValue::List(Vec::new());
        }
        let items = inner
            .split(',')
            .map(|s| parse_override_value(s.trim()))
            .collect();
        return ConfigValue::List(items);
    }
    let unquoted = strip_matching_quotes(trimmed);
    ConfigValue::from(unquoted)
}

/// Strip one layer of matching single/double quotes, if present.
fn strip_matching_quotes(s: &str) -> &str {
    let bytes = s.as_bytes();
    if bytes.len() >= 2 {
        let (first, last) = (bytes[0], bytes[bytes.len() - 1]);
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return &s[1..s.len() - 1];
        }
    }
    s
}

/// Set `value` at a dotted path, creating intermediate dicts only when `create`.
fn set_at_path(config: &mut ConfigDict, path: &str, value: ConfigValue, create: bool) {
    let parts: Vec<&str> = path.split('.').collect();
    set_at_path_parts(config, &parts, value, create);
}

fn set_at_path_parts(config: &mut ConfigDict, parts: &[&str], value: ConfigValue, create: bool) {
    let Some((key, rest)) = parts.split_first() else {
        return;
    };
    if rest.is_empty() {
        config.insert(key.to_string(), value);
        return;
    }
    if !config.contains_key(key) {
        if !create {
            return;
        }
        config.insert(key.to_string(), ConfigValue::Dict(ConfigDict::new()));
    }
    if let Some(ConfigValue::Dict(nested)) = config.get_mut(key) {
        set_at_path_parts(nested, rest, value, create);
    }
}

/// Delete the value at a dotted path (no-op if absent).
fn delete_at_path(config: &mut ConfigDict, path: &str) {
    let parts: Vec<&str> = path.split('.').collect();
    delete_at_path_parts(config, &parts);
}

fn delete_at_path_parts(config: &mut ConfigDict, parts: &[&str]) {
    let Some((key, rest)) = parts.split_first() else {
        return;
    };
    if rest.is_empty() {
        config.remove(key);
        return;
    }
    if let Some(ConfigValue::Dict(nested)) = config.get_mut(key) {
        delete_at_path_parts(nested, rest);
    }
}

// ---------------------------------------------------------------------------
// YAML → ConfigValue + header extraction
// ---------------------------------------------------------------------------

/// Convert a parsed `serde_yaml::Value` to a [`ConfigValue`]. Numbers narrow
/// to `Int` when integral; `"${...}"` strings become `Interpolation`; `"???"`
/// becomes `Missing`; a `!tag`'d node unwraps to its inner value.
fn yaml_to_config_value(yaml: &serde_yaml::Value) -> ConfigValue {
    match yaml {
        serde_yaml::Value::Null => ConfigValue::Null,
        serde_yaml::Value::Bool(b) => ConfigValue::Bool(*b),
        serde_yaml::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                ConfigValue::Int(i)
            } else if let Some(f) = n.as_f64() {
                ConfigValue::Float(f)
            } else {
                ConfigValue::Null
            }
        }
        serde_yaml::Value::String(s) => {
            if s == "???" {
                ConfigValue::Missing
            } else {
                ConfigValue::from(s.clone())
            }
        }
        serde_yaml::Value::Sequence(seq) => {
            ConfigValue::List(seq.iter().map(yaml_to_config_value).collect())
        }
        serde_yaml::Value::Mapping(map) => {
            let mut dict = ConfigDict::new();
            for (key, value) in map {
                // Hydra config keys are strings; non-string keys are ignored.
                if let serde_yaml::Value::String(k) = key {
                    dict.insert(k.clone(), yaml_to_config_value(value));
                }
            }
            ConfigValue::Dict(dict)
        }
        // serde_yaml 0.9 tagged node (`!Foo value`) — unwrap to the inner value.
        serde_yaml::Value::Tagged(t) => yaml_to_config_value(&t.value),
    }
}

/// Extract `# @key value` directives from the leading comment block. Stops at
/// the first non-comment, non-blank, non-`---` line. Used for `@package`.
fn extract_header(content: &str) -> HashMap<String, String> {
    let mut header = HashMap::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(comment) = trimmed.strip_prefix('#') {
            let comment = comment.trim();
            let Some(directive) = comment.strip_prefix('@') else {
                continue;
            };
            // `@key value`, `@key: value` (trailing colon tolerated), or `@key:value`.
            if let Some((key, value)) = directive.split_once(char::is_whitespace) {
                let (key, value) = (key.trim().trim_end_matches(':'), value.trim());
                if !key.is_empty() && !value.is_empty() {
                    header.insert(key.to_string(), value.to_string());
                }
            } else if let Some((key, value)) = directive.split_once(':') {
                let (key, value) = (key.trim(), value.trim());
                if !key.is_empty() && !value.is_empty() {
                    header.insert(key.to_string(), value.to_string());
                }
            }
        } else if !trimmed.starts_with("---") {
            break;
        }
    }
    header
}

// ---------------------------------------------------------------------------
// Sweep expansion
// ---------------------------------------------------------------------------

/// Expand sweep override strings into the cartesian product of their axes.
///
/// Recognised per-axis grammar (matching Hydra's basic sweeper):
///   * choice — `key=a,b,c` → one combo arm per value
///   * range  — `key=range(start,stop[,step])` → `start..stop` step `step`
///     (defaults: start 0, stop 10, step 1; integer arms)
///   * anything else (including `key=[list]` / `key={dict}` / a bare flag) is
///     a single fixed arm.
///
/// Returns a `Vec` of override-string sets, one per cartesian combo. An empty
/// input yields a single empty combo (so a no-sweep run still composes once).
pub fn expand_simple_sweeps(overrides: &[&str]) -> Vec<Vec<String>> {
    let mut dimensions: Vec<Vec<String>> = Vec::new();
    for ovr in overrides {
        dimensions.push(expand_one_axis(ovr));
    }
    cartesian_product(&dimensions)
}

/// Expand a single sweep axis to its arm list (see [`expand_simple_sweeps`]).
fn expand_one_axis(ovr: &str) -> Vec<String> {
    let Some(eq) = ovr.find('=') else {
        // Not a `key=value` form (e.g. a `~delete` or a flag) — one fixed arm.
        return vec![ovr.to_string()];
    };
    let key = &ovr[..eq];
    let value = &ovr[eq + 1..];

    // Range FIRST — `range(...)` contains commas that are not choice separators.
    if let Some(inner) = value
        .strip_prefix("range(")
        .and_then(|v| v.strip_suffix(')'))
    {
        let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
        let start: i64 = parts.first().and_then(|s| s.parse().ok()).unwrap_or(0);
        let stop: i64 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);
        // An explicit step must parse to a POSITIVE integer. A float / negative /
        // zero / unparseable step is NOT silently coerced to 1 (which would mangle
        // the sweep) — the override falls through to a single fixed arm instead.
        // (Float / descending ranges are a documented additive gap.)
        let step: i64 = match parts.get(2) {
            None => 1,
            Some(s) => match s.parse::<i64>() {
                Ok(st) if st > 0 => st,
                _ => return vec![ovr.to_string()],
            },
        };
        return (start..stop)
            .step_by(step as usize)
            .map(|v| format!("{key}={v}"))
            .collect();
    }

    // Choice — `a,b,c`, but NOT a `[list]`/`{dict}` literal value.
    if value.contains(',') && !value.starts_with('[') && !value.starts_with('{') {
        return value
            .split(',')
            .map(|v| format!("{key}={}", v.trim()))
            .collect();
    }

    // Fixed single arm.
    vec![ovr.to_string()]
}

/// Cartesian product of per-axis arm lists. Empty input → one empty combo.
fn cartesian_product(dimensions: &[Vec<String>]) -> Vec<Vec<String>> {
    let mut result: Vec<Vec<String>> = vec![vec![]];
    for dim in dimensions {
        let mut next = Vec::with_capacity(result.len() * dim.len());
        for combo in &result {
            for arm in dim {
                let mut extended = combo.clone();
                extended.push(arm.clone());
                next.push(extended);
            }
        }
        result = next;
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::Path;

    fn write_config(dir: &Path, name: &str, body: &str) {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(body.as_bytes()).unwrap();
    }

    // --- ConfigValue / ConfigDict ---

    #[test]
    fn value_accessors() {
        assert!(ConfigValue::Null.is_null());
        assert!(ConfigValue::Missing.is_missing());
        assert_eq!(ConfigValue::Bool(true).as_bool(), Some(true));
        assert_eq!(ConfigValue::Int(42).as_int(), Some(42));
        assert_eq!(ConfigValue::Int(7).as_float(), Some(7.0)); // int widens
        assert_eq!(ConfigValue::Float(3.5).as_float(), Some(3.5));
        assert_eq!(ConfigValue::from("hi").as_str(), Some("hi"));
    }

    #[test]
    fn interpolation_detection() {
        assert!(ConfigValue::from("${foo.bar}").is_interpolation());
        assert!(ConfigValue::from("a ${x} b").is_interpolation());
        assert!(!ConfigValue::from("plain").is_interpolation());
        assert!(!ConfigValue::from("${unterminated").is_interpolation());
    }

    #[test]
    fn dict_insert_get_order_and_remove() {
        let mut d = ConfigDict::new();
        d.insert("a".into(), ConfigValue::Int(1));
        d.insert("b".into(), ConfigValue::Int(2));
        d.insert("a".into(), ConfigValue::Int(10)); // overwrite in place
        let keys: Vec<&str> = d.keys().collect();
        assert_eq!(
            keys,
            vec!["a", "b"],
            "insertion order preserved on overwrite"
        );
        assert_eq!(d.get("a").unwrap().as_int(), Some(10));
        assert_eq!(d.len(), 2);
        assert_eq!(d.remove("a").unwrap().as_int(), Some(10));
        assert!(!d.contains_key("a"));
        assert_eq!(d.len(), 1);
        let keys: Vec<&str> = d.keys().collect();
        assert_eq!(keys, vec!["b"], "tombstoned key is skipped");
    }

    #[test]
    fn dict_select_dotted() {
        let mut inner = ConfigDict::new();
        inner.insert("value".into(), ConfigValue::Int(42));
        let mut outer = ConfigDict::new();
        outer.insert("nested".into(), ConfigValue::Dict(inner));
        assert_eq!(outer.select("nested.value").unwrap().as_int(), Some(42));
        assert!(outer.select("nested.missing").is_none());
        assert!(outer.select("nested.value.too.deep").is_none());
    }

    #[test]
    fn dict_deep_merge() {
        let mut base = ConfigDict::new();
        let mut base_inner = ConfigDict::new();
        base_inner.insert("x".into(), ConfigValue::Int(1));
        base_inner.insert("y".into(), ConfigValue::Int(2));
        base.insert("g".into(), ConfigValue::Dict(base_inner));
        base.insert("top".into(), ConfigValue::Int(0));

        let mut overlay = ConfigDict::new();
        let mut over_inner = ConfigDict::new();
        over_inner.insert("y".into(), ConfigValue::Int(20)); // override
        over_inner.insert("z".into(), ConfigValue::Int(3)); // add
        overlay.insert("g".into(), ConfigValue::Dict(over_inner));

        base.merge(&overlay);
        let g = base.get("g").unwrap().as_dict().unwrap();
        assert_eq!(g.get("x").unwrap().as_int(), Some(1), "untouched key kept");
        assert_eq!(g.get("y").unwrap().as_int(), Some(20), "nested override");
        assert_eq!(g.get("z").unwrap().as_int(), Some(3), "nested add");
        assert_eq!(base.get("top").unwrap().as_int(), Some(0));
    }

    // --- YAML → ConfigValue + header ---

    #[test]
    fn yaml_scalar_types() {
        let y: serde_yaml::Value = serde_yaml::from_str(
            "i: 7\nf: 1.5\nb: true\nn: null\ns: hi\ninterp: ${a.b}\nmiss: '???'\n",
        )
        .unwrap();
        let cv = yaml_to_config_value(&y);
        let d = cv.as_dict().unwrap();
        assert_eq!(d.get("i").unwrap(), &ConfigValue::Int(7));
        assert_eq!(d.get("f").unwrap(), &ConfigValue::Float(1.5));
        assert_eq!(d.get("b").unwrap(), &ConfigValue::Bool(true));
        assert_eq!(d.get("n").unwrap(), &ConfigValue::Null);
        assert_eq!(d.get("s").unwrap(), &ConfigValue::String("hi".into()));
        assert!(d.get("interp").unwrap().is_interpolation());
        assert_eq!(d.get("miss").unwrap(), &ConfigValue::Missing);
    }

    #[test]
    fn header_extraction() {
        let h = extract_header("# @package db\n# @group foo\nkey: 1\n# @ignored after body\n");
        assert_eq!(h.get("package").map(String::as_str), Some("db"));
        assert_eq!(h.get("group").map(String::as_str), Some("foo"));
        assert_eq!(h.len(), 2, "stops at first body line");

        let h2 = extract_header("# @package:nested.path\nkey: 1\n");
        assert_eq!(h2.get("package").map(String::as_str), Some("nested.path"));

        // Leading blanks + doc separator do not stop the scan.
        let h3 = extract_header("\n---\n# @package x\nkey: 1\n");
        assert_eq!(h3.get("package").map(String::as_str), Some("x"));
    }

    // --- ConfigLoader compose ---

    #[test]
    fn load_simple_config() {
        let dir = tempfile::tempdir().unwrap();
        write_config(
            dir.path(),
            "config.yaml",
            "db:\n  host: localhost\n  port: 3306\n",
        );
        let loader = ConfigLoader::from_config_dir(dir.path().to_str().unwrap());
        let cfg = loader.load_config(Some("config"), &[]).unwrap();
        let db = cfg.as_dict().unwrap().get("db").unwrap().as_dict().unwrap();
        assert_eq!(db.get("host").unwrap().as_str(), Some("localhost"));
        assert_eq!(db.get("port").unwrap().as_int(), Some(3306));
    }

    #[test]
    fn load_with_dotted_override() {
        let dir = tempfile::tempdir().unwrap();
        write_config(
            dir.path(),
            "config.yaml",
            "db:\n  host: localhost\n  port: 3306\n",
        );
        let loader = ConfigLoader::from_config_dir(dir.path().to_str().unwrap());
        let cfg = loader
            .load_config(
                Some("config"),
                &["db.host=remote".to_string(), "db.port=5432".to_string()],
            )
            .unwrap();
        let db = cfg.as_dict().unwrap().get("db").unwrap().as_dict().unwrap();
        assert_eq!(db.get("host").unwrap().as_str(), Some("remote"));
        assert_eq!(db.get("port").unwrap().as_int(), Some(5432));
    }

    #[test]
    fn add_and_delete_overrides() {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), "config.yaml", "keep: 1\ndrop: 2\n");
        let loader = ConfigLoader::from_config_dir(dir.path().to_str().unwrap());
        let cfg = loader
            .load_config(
                Some("config"),
                &["+added.deep=9".to_string(), "~drop".to_string()],
            )
            .unwrap();
        let d = cfg.as_dict().unwrap();
        assert_eq!(d.get("keep").unwrap().as_int(), Some(1));
        assert!(!d.contains_key("drop"), "~drop removed it");
        let added = d.get("added").unwrap().as_dict().unwrap();
        assert_eq!(
            added.get("deep").unwrap().as_int(),
            Some(9),
            "+ creates nesting"
        );
    }

    #[test]
    fn defaults_list_with_package_header() {
        let dir = tempfile::tempdir().unwrap();
        write_config(
            dir.path(),
            "db/mysql.yaml",
            "# @package db\ndriver: mysql\nport: 3306\n",
        );
        write_config(
            dir.path(),
            "config.yaml",
            "defaults:\n  - db: mysql\n\napp_name: myapp\n",
        );
        let loader = ConfigLoader::from_config_dir(dir.path().to_str().unwrap());
        let cfg = loader.load_config(Some("config"), &[]).unwrap();
        let d = cfg.as_dict().unwrap();
        assert_eq!(d.get("app_name").unwrap().as_str(), Some("myapp"));
        let db = d.get("db").unwrap().as_dict().unwrap();
        assert_eq!(db.get("driver").unwrap().as_str(), Some("mysql"));
        assert_eq!(db.get("port").unwrap().as_int(), Some(3306));
    }

    #[test]
    fn default_group_selection_override() {
        let dir = tempfile::tempdir().unwrap();
        write_config(
            dir.path(),
            "db/mysql.yaml",
            "# @package db\ndriver: mysql\n",
        );
        write_config(
            dir.path(),
            "db/postgres.yaml",
            "# @package db\ndriver: postgres\n",
        );
        write_config(dir.path(), "config.yaml", "defaults:\n  - db: mysql\n");
        let loader = ConfigLoader::from_config_dir(dir.path().to_str().unwrap());
        // dot-less `db=postgres` re-selects the group option.
        let cfg = loader
            .load_config(Some("config"), &["db=postgres".to_string()])
            .unwrap();
        let db = cfg.as_dict().unwrap().get("db").unwrap().as_dict().unwrap();
        assert_eq!(db.get("driver").unwrap().as_str(), Some("postgres"));
    }

    #[test]
    fn primary_overrides_defaults() {
        // The primary body is merged LAST, so it wins over a defaults entry.
        let dir = tempfile::tempdir().unwrap();
        write_config(
            dir.path(),
            "base/a.yaml",
            "shared: from_default\nonly_default: 1\n",
        );
        write_config(
            dir.path(),
            "config.yaml",
            "defaults:\n  - base: a\nshared: from_primary\n",
        );
        let loader = ConfigLoader::from_config_dir(dir.path().to_str().unwrap());
        let cfg = loader.load_config(Some("config"), &[]).unwrap();
        let d = cfg.as_dict().unwrap();
        assert_eq!(d.get("shared").unwrap().as_str(), Some("from_primary"));
        // `base: a` has no @package → merges at the `base` package path.
        let base = d.get("base").unwrap().as_dict().unwrap();
        assert_eq!(base.get("only_default").unwrap().as_int(), Some(1));
    }

    #[test]
    fn optional_default_missing_is_tolerated() {
        // Hydra's `optional` marker sits in the SAME defaults entry as the
        // group (`- db: nope` + `optional: true` → one mapping).
        let dir = tempfile::tempdir().unwrap();
        write_config(
            dir.path(),
            "config.yaml",
            "defaults:\n  - db: nope\n    optional: true\nx: 1\n",
        );
        let loader = ConfigLoader::from_config_dir(dir.path().to_str().unwrap());
        let cfg = loader.load_config(Some("config"), &[]).unwrap();
        assert_eq!(cfg.as_dict().unwrap().get("x").unwrap().as_int(), Some(1));
    }

    #[test]
    fn missing_required_default_errors() {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), "config.yaml", "defaults:\n  - db: nope\n");
        let loader = ConfigLoader::from_config_dir(dir.path().to_str().unwrap());
        let err = loader.load_config(Some("config"), &[]).unwrap_err();
        assert!(matches!(err, HydraError::NotFound { .. }));
    }

    #[test]
    fn value_parsing_types() {
        assert_eq!(parse_override_value("true"), ConfigValue::Bool(true));
        assert_eq!(parse_override_value("false"), ConfigValue::Bool(false));
        assert_eq!(parse_override_value("null"), ConfigValue::Null);
        assert_eq!(parse_override_value("~"), ConfigValue::Null);
        assert_eq!(parse_override_value("42"), ConfigValue::Int(42));
        assert_eq!(parse_override_value("1e-3"), ConfigValue::Float(0.001));
        assert_eq!(
            parse_override_value("'quoted'"),
            ConfigValue::String("quoted".into())
        );
        assert_eq!(
            parse_override_value("bare"),
            ConfigValue::String("bare".into())
        );
        assert_eq!(
            parse_override_value("[1, 2, 3]"),
            ConfigValue::List(vec![
                ConfigValue::Int(1),
                ConfigValue::Int(2),
                ConfigValue::Int(3)
            ])
        );
        assert_eq!(parse_override_value("[]"), ConfigValue::List(vec![]));
    }

    #[test]
    fn config_exists_and_list_group() {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), "config.yaml", "v: 1\n");
        write_config(dir.path(), "db/mysql.yaml", "driver: mysql\n");
        write_config(dir.path(), "db/postgres.yaml", "driver: postgres\n");
        let loader = ConfigLoader::from_config_dir(dir.path().to_str().unwrap());
        assert!(loader.config_exists("config"));
        assert!(!loader.config_exists("nope"));
        assert_eq!(
            loader.list_group("db"),
            vec!["mysql".to_string(), "postgres".to_string()]
        );
    }

    // --- sweep expansion ---

    #[test]
    fn sweep_choice() {
        let r = expand_simple_sweeps(&["db=mysql,postgres", "server=dev,prod"]);
        assert_eq!(r.len(), 4);
        assert!(r.contains(&vec!["db=mysql".to_string(), "server=dev".to_string()]));
        assert!(r.contains(&vec!["db=postgres".to_string(), "server=prod".to_string()]));
    }

    #[test]
    fn sweep_range() {
        let r = expand_simple_sweeps(&["x=range(1,4)"]);
        assert_eq!(
            r,
            vec![
                vec!["x=1".to_string()],
                vec!["x=2".to_string()],
                vec!["x=3".to_string()]
            ]
        );
        let stepped = expand_simple_sweeps(&["x=range(0,10,3)"]);
        assert_eq!(
            stepped,
            vec![
                vec!["x=0".to_string()],
                vec!["x=3".to_string()],
                vec!["x=6".to_string()],
                vec!["x=9".to_string()],
            ]
        );
    }

    #[test]
    fn sweep_mixed_and_fixed() {
        let r = expand_simple_sweeps(&["db=mysql,postgres", "port=3306"]);
        assert_eq!(r.len(), 2);
        assert!(r.contains(&vec!["db=mysql".to_string(), "port=3306".to_string()]));
        assert!(r.contains(&vec!["db=postgres".to_string(), "port=3306".to_string()]));
    }

    #[test]
    fn sweep_float_choice_preserved() {
        // The arms keep their literal text (e.g. `1e-3`), which the loader
        // later parses as a float override.
        let r = expand_simple_sweeps(&["lr=1e-3,1e-4", "bs=8,16"]);
        assert_eq!(r.len(), 4);
        assert!(r.contains(&vec!["lr=1e-3".to_string(), "bs=8".to_string()]));
        assert!(r.contains(&vec!["lr=1e-4".to_string(), "bs=16".to_string()]));
    }

    #[test]
    fn sweep_empty_is_single_empty_combo() {
        assert_eq!(expand_simple_sweeps(&[]), vec![Vec::<String>::new()]);
    }

    #[test]
    fn sweep_list_value_is_fixed_arm() {
        // A `[a,b]` literal must NOT be split on its inner comma.
        let r = expand_simple_sweeps(&["k=[1,2]"]);
        assert_eq!(r, vec![vec!["k=[1,2]".to_string()]]);
    }

    #[test]
    fn sweep_large_cartesian() {
        let r = expand_simple_sweeps(&["a=1,2,3", "b=4,5,6", "c=7,8,9"]);
        assert_eq!(r.len(), 27);
    }

    // --- regressions from the adversarial review ---

    #[test]
    fn remove_then_reinsert_does_not_duplicate_key() {
        // Tombstone bug: ~x then +x=1 must NOT double-emit "x" from iter/keys,
        // or the JSON freeze fingerprint would be non-deterministic.
        let mut d = ConfigDict::new();
        d.insert("x".into(), ConfigValue::Int(0));
        d.insert("y".into(), ConfigValue::Int(9));
        assert_eq!(d.remove("x").unwrap().as_int(), Some(0));
        d.insert("x".into(), ConfigValue::Int(1));
        let keys: Vec<&str> = d.keys().collect();
        assert_eq!(keys, vec!["y", "x"], "remaining order kept, no stale dup");
        assert_eq!(d.len(), 2);
        assert_eq!(d.iter().count(), 2, "len() and iter() agree");
        assert_eq!(d.get("x").unwrap().as_int(), Some(1));
    }

    #[test]
    fn remove_reinsert_via_overrides_keeps_single_key() {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), "config.yaml", "x: 0\n");
        let loader = ConfigLoader::from_config_dir(dir.path().to_str().unwrap());
        let cfg = loader
            .load_config(Some("config"), &["~x".to_string(), "+x=1".to_string()])
            .unwrap();
        let d = cfg.as_dict().unwrap();
        assert_eq!(d.keys().count(), 1, "exactly one 'x' survives");
        assert_eq!(d.get("x").unwrap().as_int(), Some(1));
    }

    #[test]
    fn non_finite_values_stay_strings() {
        // Hydra keeps bare inf/nan as strings; a Float(nan) would also break
        // ConfigValue PartialEq.
        assert_eq!(
            parse_override_value("inf"),
            ConfigValue::String("inf".into())
        );
        assert_eq!(
            parse_override_value("nan"),
            ConfigValue::String("nan".into())
        );
        assert_eq!(
            parse_override_value("Infinity"),
            ConfigValue::String("Infinity".into())
        );
        // real numbers still parse
        assert_eq!(parse_override_value(".5"), ConfigValue::Float(0.5));
        assert_eq!(parse_override_value("1."), ConfigValue::Float(1.0));
        assert_eq!(parse_override_value("2e3"), ConfigValue::Float(2000.0));
    }

    #[test]
    fn range_bad_step_is_fixed_arm_not_mangled() {
        // float step → single fixed arm (not silently step-1)
        assert_eq!(
            expand_simple_sweeps(&["x=range(0,1,0.1)"]),
            vec![vec!["x=range(0,1,0.1)".to_string()]]
        );
        // negative step → single fixed arm
        assert_eq!(
            expand_simple_sweeps(&["x=range(0,10,-2)"]),
            vec![vec!["x=range(0,10,-2)".to_string()]]
        );
        // positive step still expands
        assert_eq!(
            expand_simple_sweeps(&["x=range(0,6,2)"]),
            vec![
                vec!["x=0".to_string()],
                vec!["x=2".to_string()],
                vec!["x=4".to_string()]
            ]
        );
    }

    #[test]
    fn header_package_colon_space_form() {
        // `# @package: db` (colon + space) must still resolve to package "db".
        let h = extract_header("# @package: db\nkey: 1\n");
        assert_eq!(h.get("package").map(String::as_str), Some("db"));
    }
}
