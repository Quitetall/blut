// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Ecosystem connectors (ADR 0112) — a compiled-in `connectors` cookbook whose
//! stages delegate an external verb to a SUBPROCESS (ADR 0034) and exchange data
//! through content-addressed artifacts with a CLOSED set of typed I/O kinds. No
//! dynamic library loading; the `from_erased_graph` kind-checker (ADR 0078)
//! type-checks an integration graph exactly as it checks a native one.
//!
//! The four kinds carry IDENTITY + HASH, never bulk data — an `ObjectRef` is an
//! object-store URI + content fingerprint, so the cache key
//! `hash(code_sha ⊕ canon_args ⊕ input_hash)` is unchanged. The bulk lives
//! remotely; the local artifact is a small manifest.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::framework::artifact::{Artifact, ContentHash};
use crate::framework::cookbook::{Cookbook, Registry};
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{ErasedStageCtor, Stage, StageContext};
use crate::recipes::recipe::RecipeDef;

// ── the closed set of connector I/O kinds ──────────────────────────

macro_rules! connector_kind {
    ($ty:ident, $kind:literal, $($field:ident : $fty:ty),* $(,)?) => {
        #[derive(Clone, Debug, Serialize, Deserialize)]
        pub struct $ty {
            $(pub $field: $fty,)*
            /// The identity fingerprint (from the external tool / the ref), NOT
            /// recomputed from local bytes.
            pub content_hash: ContentHash,
            /// Local manifest/data path (a small descriptor for a remote ref;
            /// the data itself for a local one).
            pub path: PathBuf,
        }
        impl Artifact for $ty {
            const KIND: &'static str = $kind;
            const SCHEMA: u32 = 1;
            // Identity+hash: the hash is the ref's fingerprint, not the local
            // manifest's bytes — never re-walk it.
            const HASH_CONTENTS: bool = false;
            fn content_hash(&self) -> ContentHash { self.content_hash }
            fn primary_path(&self) -> &Path { &self.path }
        }
    };
}

connector_kind!(ObjectRef, "connector.object_ref", uri: String);
connector_kind!(TableRef, "connector.table_ref", table: String);
connector_kind!(DatasetRef, "connector.dataset_ref",);
connector_kind!(Blob, "connector.blob",);

/// The closed set of connector kind tags — a connector stage's declared I/O must
/// be one of these (nothing else is a connector kind).
pub const CONNECTOR_KINDS: &[&str] = &[
    ObjectRef::KIND,
    TableRef::KIND,
    DatasetRef::KIND,
    Blob::KIND,
];

// ── connector stages (each delegates to a subprocess) ──────────────

/// Declare an object-store reference from args (a graph SOURCE) — the entry
/// point of a connector graph (`input = ()`).
#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ObjectRefArgs {
    /// The object-store URI (e.g. `s3://bucket/key`).
    pub uri: String,
    /// The object's content hash (hex) — its identity.
    pub content_hash: String,
}

pub struct DeclareObject;
#[async_trait]
impl Stage for DeclareObject {
    const NAME: &'static str = "connector_object";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = ObjectRef;
    type Args = ObjectRefArgs;
    async fn run(
        &self,
        ctx: &StageContext,
        _input: (),
        args: &ObjectRefArgs,
    ) -> Result<ObjectRef, StageError> {
        let content_hash = ContentHash::from_hex(&args.content_hash)
            .map_err(|e| StageError::Backend(anyhow::anyhow!("bad content_hash: {e}")))?;
        std::fs::create_dir_all(&ctx.stage_dir).ok();
        let path = ctx.stage_dir.join("object.ref.json");
        let manifest = serde_json::json!({ "uri": args.uri, "content_hash": args.content_hash });
        std::fs::write(&path, manifest.to_string())
            .map_err(|e| StageError::Backend(anyhow::anyhow!("write object manifest: {e}")))?;
        Ok(ObjectRef {
            uri: args.uri.clone(),
            content_hash,
            path,
        })
    }
}

/// Args for the shell-out connectors: the external tool binary + optional verb.
#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ToolArgs {
    /// The external tool to delegate to (e.g. `aws`, `rclone`, `duckdb`).
    pub tool: String,
    /// For `connector_store`: the destination object-store URI to PUT to (the
    /// returned `ObjectRef.uri`). `None` ⇒ a content-addressed
    /// `connector://<hash>` sink (round-trippable only within this store).
    #[serde(default)]
    pub dest_uri: Option<String>,
    /// Extra args threaded verbatim after the verb.
    #[serde(default)]
    pub extra: Vec<String>,
}

/// Fetch an object to a local dataset — delegates the GET to `tool` as a
/// subprocess (ADR 0034). `ObjectRef → DatasetRef`.
pub struct ObjectFetch;
#[async_trait]
impl Stage for ObjectFetch {
    const NAME: &'static str = "connector_fetch";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu, Resource::Network];
    type Input = ObjectRef;
    type Output = DatasetRef;
    type Args = ToolArgs;
    async fn run(
        &self,
        ctx: &StageContext,
        input: ObjectRef,
        args: &ToolArgs,
    ) -> Result<DatasetRef, StageError> {
        std::fs::create_dir_all(&ctx.stage_dir).ok();
        let dest = ctx.stage_dir.join("data");
        // Delegate the verb to the external tool (subprocess, never a dylib) on a
        // blocking-safe async spawn. `--` separates the fixed verb from the
        // user-supplied URI so a `-`-leading value can't be read as a flag.
        let status = tokio::process::Command::new(&args.tool)
            .arg("get")
            .args(&args.extra) // verbatim tool FLAGS come before `--`
            .arg("--")
            .arg(&input.uri)
            .arg(&dest)
            .status()
            .await
            .map_err(|e| {
                StageError::Backend(anyhow::anyhow!("spawn connector tool '{}': {e}", args.tool))
            })?;
        if !status.success() {
            return Err(StageError::Backend(anyhow::anyhow!(
                "connector '{}' get {} exited {:?}",
                args.tool,
                input.uri,
                status.code()
            )));
        }
        // Content-address integrity: the fetched bytes MUST hash to the declared
        // identity — a corrupt/tampered download must never silently produce a
        // DatasetRef with the wrong bytes but the declared hash (which would
        // poison every downstream cache key).
        let got = ContentHash::hash_file(&dest)
            .map_err(|e| StageError::Backend(anyhow::anyhow!("hash fetched object: {e}")))?;
        if got != input.content_hash {
            return Err(StageError::Backend(anyhow::anyhow!(
                "connector fetch integrity: {} hashed {} but the ref declared {}",
                input.uri,
                got.to_hex(),
                input.content_hash.to_hex()
            )));
        }
        Ok(DatasetRef {
            content_hash: got,
            path: dest,
        })
    }
}

/// Store a local dataset back to the object store — delegates PUT to `tool`.
/// `DatasetRef → ObjectRef`.
pub struct ObjectStore;
#[async_trait]
impl Stage for ObjectStore {
    const NAME: &'static str = "connector_store";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu, Resource::Network];
    type Input = DatasetRef;
    type Output = ObjectRef;
    type Args = ToolArgs;
    async fn run(
        &self,
        ctx: &StageContext,
        input: DatasetRef,
        args: &ToolArgs,
    ) -> Result<ObjectRef, StageError> {
        // The REAL destination the object lands at (returned so a later
        // `connector_fetch` can round-trip it). Defaults to the content-addressed
        // sink only when the recipe didn't name one.
        let uri = args
            .dest_uri
            .clone()
            .unwrap_or_else(|| format!("connector://{}", input.content_hash.to_hex()));
        let status = tokio::process::Command::new(&args.tool)
            .arg("put")
            .args(&args.extra) // verbatim tool FLAGS come before `--`
            .arg("--")
            .arg(&input.path)
            .arg(&uri)
            .status()
            .await
            .map_err(|e| {
                StageError::Backend(anyhow::anyhow!("spawn connector tool '{}': {e}", args.tool))
            })?;
        if !status.success() {
            return Err(StageError::Backend(anyhow::anyhow!(
                "connector '{}' put exited {:?}",
                args.tool,
                status.code()
            )));
        }
        std::fs::create_dir_all(&ctx.stage_dir).ok();
        let path = ctx.stage_dir.join("stored.ref.json");
        std::fs::write(&path, serde_json::json!({ "uri": uri }).to_string())
            .map_err(|e| StageError::Backend(anyhow::anyhow!("write stored manifest: {e}")))?;
        Ok(ObjectRef {
            uri,
            content_hash: input.content_hash,
            path,
        })
    }
}

// ── descriptors + `blut connectors list --check` ───────────────────

/// A connector's declared identity: name + external tool + typed I/O kinds.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectorDescriptor {
    pub name: &'static str,
    pub input_kind: &'static str,
    pub output_kind: &'static str,
}

/// Every registered connector stage's descriptor.
pub fn list() -> Vec<ConnectorDescriptor> {
    vec![
        ConnectorDescriptor {
            name: DeclareObject::NAME,
            input_kind: "()",
            output_kind: ObjectRef::KIND,
        },
        ConnectorDescriptor {
            name: ObjectFetch::NAME,
            input_kind: ObjectRef::KIND,
            output_kind: DatasetRef::KIND,
        },
        ConnectorDescriptor {
            name: ObjectStore::NAME,
            input_kind: DatasetRef::KIND,
            output_kind: ObjectRef::KIND,
        },
    ]
}

/// Validate the connector registry (`--check`): every connector's declared I/O
/// kind must be `()` (a source) or one of the CLOSED connector-kind set — a
/// connector that leaks a non-connector kind into an integration graph is a hard
/// error. Returns the list of problems (empty ⇒ healthy).
pub fn check() -> Vec<String> {
    let ok = |k: &str| k == "()" || CONNECTOR_KINDS.contains(&k);
    let mut problems = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for d in list() {
        if !seen.insert(d.name) {
            problems.push(format!("{}: duplicate connector name", d.name));
        }
        if !ok(d.input_kind) {
            problems.push(format!(
                "{}: input kind '{}' not a connector kind",
                d.name, d.input_kind
            ));
        }
        if !ok(d.output_kind) {
            problems.push(format!(
                "{}: output kind '{}' not a connector kind",
                d.name, d.output_kind
            ));
        }
    }
    problems
}

/// The built-in `connectors` cookbook.
pub struct ConnectorsCookbook;
impl Cookbook for ConnectorsCookbook {
    fn name(&self) -> &'static str {
        "connectors"
    }
    fn recipes(&self) -> &'static [&'static RecipeDef] {
        &[]
    }
    fn stages_erased(&self) -> &'static [(&'static str, ErasedStageCtor)] {
        static S: &[(&str, ErasedStageCtor)] = &[
            (DeclareObject::NAME, || std::sync::Arc::new(DeclareObject)),
            (ObjectFetch::NAME, || std::sync::Arc::new(ObjectFetch)),
            (ObjectStore::NAME, || std::sync::Arc::new(ObjectStore)),
        ];
        S
    }
}

/// Register the built-in `connectors` cookbook into `reg`.
pub fn register(reg: &mut Registry) {
    reg.register(Box::new(ConnectorsCookbook));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connector_kinds_are_distinct_and_closed() {
        let mut k = CONNECTOR_KINDS.to_vec();
        k.sort_unstable();
        k.dedup();
        assert_eq!(k.len(), 4, "four distinct connector kinds");
        assert!(CONNECTOR_KINDS.contains(&ObjectRef::KIND));
    }

    #[test]
    fn connectors_check_is_healthy() {
        // Every shipped connector's I/O is `()` or a connector kind.
        assert!(check().is_empty(), "registry problems: {:?}", check());
        // list() enumerates the three connector stages.
        assert_eq!(list().len(), 3);
    }

    #[test]
    fn roundtrip_graph_kind_checks_mismatch_fails() {
        use crate::framework::plan_spec::{PlanSpec, SpecNode};
        let mut reg = Registry::new();
        register(&mut reg);
        let node = |stage: &str, args: serde_json::Value| SpecNode {
            stage: stage.into(),
            args,
            retry: None,
            timeout: None,
            priority: None,
        };
        let tool = serde_json::json!({ "tool": "true" });
        // object → fetch → store: ObjectRef → DatasetRef → ObjectRef. Kind-checks.
        let ok_spec = PlanSpec {
            name: "s3_roundtrip".into(),
            nodes: vec![
                node(
                    "connector_object",
                    serde_json::json!({ "uri": "s3://b/k", "content_hash": "aa".repeat(32) }),
                ),
                node("connector_fetch", tool.clone()),
                node("connector_store", tool.clone()),
            ],
            edges: vec![(0, 1), (1, 2)],
            expansions: Vec::new(),
            version: 1,
        };
        assert!(
            ok_spec.compile(&reg).is_ok(),
            "the connector roundtrip must kind-check"
        );

        // A mismatched wiring (fetch → fetch: DatasetRef ≠ ObjectRef) is refused.
        let bad_spec = PlanSpec {
            name: "bad".into(),
            nodes: vec![
                node(
                    "connector_object",
                    serde_json::json!({ "uri": "s3://b/k", "content_hash": "aa".repeat(32) }),
                ),
                node("connector_fetch", tool.clone()),
                node("connector_fetch", tool),
            ],
            edges: vec![(0, 1), (1, 2)],
            expansions: Vec::new(),
            version: 1,
        };
        assert!(
            bad_spec.compile(&reg).is_err(),
            "a kind mismatch must be refused"
        );
    }
}
