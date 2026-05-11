//! Materializer — `materialize_dataset_path`.
//!
//! Graph-input stage that turns a path or registered dataset name
//! into a typed `DatasetJsonl`. The split between
//! `materialize_conversations` (pulls from lamu-mcp's conversation
//! db) and this stage isolates the source-of-truth choice from the
//! rest of the pipeline — recipe 1 wires the former, recipe 2 the
//! latter.
//!
//! Args carry exactly one of `path` or `registered_name`. Recipes
//! that have a path call this with `path` set. Recipes that ship a
//! dataset name pre-registered in `datasets_db` call this with
//! `registered_name` set; the stage resolves the name to its
//! recorded `source_path` and re-hashes the file.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::DatasetJsonl;
use crate::datasets_db;
use crate::framework::artifact::ContentHash;
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};

pub struct MaterializeDatasetPath;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// Path to a JSONL dataset on disk. Exclusive with `registered_name`.
    #[serde(default)]
    pub path: Option<PathBuf>,
    /// Registered dataset name in `datasets_db`. Exclusive with `path`.
    #[serde(default)]
    pub registered_name: Option<String>,
}

#[async_trait]
impl Stage for MaterializeDatasetPath {
    const NAME: &'static str = "materialize_dataset_path";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Disk];
    type Input = ();
    type Output = DatasetJsonl;
    type Args = Args;

    async fn run(
        &self,
        _ctx: &StageContext,
        _input: (),
        args: &Args,
    ) -> Result<DatasetJsonl, StageError> {
        let resolved_path = match (&args.path, &args.registered_name) {
            (Some(p), None) => p.clone(),
            (None, Some(name)) => {
                let conn =
                    datasets_db::open().map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;
                let rec = datasets_db::get_by_name(&conn, name)
                    .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?
                    .ok_or_else(|| {
                        StageError::BadInput(format!("registered dataset '{name}' not found"))
                    })?;
                rec.source_path
            }
            (Some(_), Some(_)) => {
                return Err(StageError::BadInput(
                    "materialize_dataset_path: pass exactly one of path / registered_name, not both"
                        .into(),
                ));
            }
            (None, None) => {
                return Err(StageError::BadInput(
                    "materialize_dataset_path: one of path / registered_name is required".into(),
                ));
            }
        };

        if !resolved_path.exists() {
            return Err(StageError::BadInput(format!(
                "dataset path not found: {}",
                resolved_path.display()
            )));
        }

        let content_hash =
            ContentHash::hash_file(&resolved_path).map_err(|source| StageError::Io {
                path: resolved_path.clone(),
                source,
            })?;
        let n_examples = count_jsonl_examples(&resolved_path)?;

        Ok(DatasetJsonl {
            path: resolved_path,
            content_hash,
            n_examples,
        })
    }
}

fn count_jsonl_examples(path: &std::path::Path) -> Result<i64, StageError> {
    use std::io::BufRead;
    let f = std::fs::File::open(path).map_err(|source| StageError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut n = 0i64;
    for line in std::io::BufReader::new(f).lines() {
        let line = line.map_err(|source| StageError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if !line.trim().is_empty() {
            n += 1;
        }
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(td: &std::path::Path) -> StageContext {
        std::fs::create_dir_all(td.join("stage")).unwrap();
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    #[tokio::test]
    async fn loads_dataset_from_path() {
        let td = tempfile::tempdir().unwrap();
        let p = td.path().join("ds.jsonl");
        std::fs::write(&p, "{\"a\":1}\n{\"b\":2}\n{\"c\":3}\n").unwrap();
        let out = MaterializeDatasetPath
            .run(
                &ctx(td.path()),
                (),
                &Args {
                    path: Some(p.clone()),
                    registered_name: None,
                },
            )
            .await
            .unwrap();
        assert_eq!(out.path, p);
        assert_eq!(out.n_examples, 3);
    }

    #[tokio::test]
    async fn rejects_missing_path() {
        let td = tempfile::tempdir().unwrap();
        let r = MaterializeDatasetPath
            .run(
                &ctx(td.path()),
                (),
                &Args {
                    path: Some(td.path().join("missing")),
                    registered_name: None,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[tokio::test]
    async fn rejects_both_path_and_name() {
        let td = tempfile::tempdir().unwrap();
        let r = MaterializeDatasetPath
            .run(
                &ctx(td.path()),
                (),
                &Args {
                    path: Some(td.path().join("x")),
                    registered_name: Some("y".into()),
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[tokio::test]
    async fn rejects_neither_path_nor_name() {
        let td = tempfile::tempdir().unwrap();
        let r = MaterializeDatasetPath
            .run(
                &ctx(td.path()),
                (),
                &Args {
                    path: None,
                    registered_name: None,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }
}
