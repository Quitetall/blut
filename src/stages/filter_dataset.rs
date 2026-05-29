//! Stage 4 — `filter_dataset`.
//!
//! Drops examples that don't meet quality thresholds. Reads the
//! input JSONL line-by-line, applies predicates, writes the kept
//! lines to `<stage_dir>/filtered.jsonl`. Pure-CPU work; no GPU,
//! no network.
//!
//! Predicates (all optional):
//! - `min_turns`: an example's `messages` array must have at least
//!   this many entries (`>= min_turns`).
//! - `max_msg_bytes`: any single message whose UTF-8 length exceeds
//!   this drops the entire example. Defense against pathological
//!   long-context outliers.
//! - `drop_errors`: when true, drop any example whose role is
//!   `tool` and content matches `*Error*` / `*error*` / `Traceback`.

use std::io::{BufRead, BufWriter, Write};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::DatasetJsonl;
use crate::framework::artifact::ContentHash;
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};

pub struct FilterDataset;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    #[serde(default)]
    pub min_turns: u32,
    #[serde(default)]
    pub max_msg_bytes: u32,
    #[serde(default)]
    pub drop_errors: bool,
}

#[async_trait]
impl Stage for FilterDataset {
    const NAME: &'static str = "filter_dataset";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = DatasetJsonl;
    type Output = DatasetJsonl;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        input: DatasetJsonl,
        args: &Args,
    ) -> Result<DatasetJsonl, StageError> {
        let out_path = ctx.stage_dir.join("filtered.jsonl");
        let infile = std::fs::File::open(&input.path).map_err(|source| StageError::Io {
            path: input.path.clone(),
            source,
        })?;
        let reader = std::io::BufReader::new(infile);

        let outfile = std::fs::File::create(&out_path).map_err(|source| StageError::Io {
            path: out_path.clone(),
            source,
        })?;
        let mut writer = BufWriter::new(outfile);

        let mut kept: i64 = 0;
        let mut malformed: i64 = 0;
        for line in reader.lines() {
            let line = line.map_err(|source| StageError::Io {
                path: input.path.clone(),
                source,
            })?;
            if line.trim().is_empty() {
                continue;
            }
            match classify(&line, args) {
                Verdict::Keep => {
                    writeln!(writer, "{line}").map_err(|source| StageError::Io {
                        path: out_path.clone(),
                        source,
                    })?;
                    kept += 1;
                }
                Verdict::DropPredicate => {}
                Verdict::DropMalformed => malformed += 1,
            }
        }
        writer.flush().map_err(|source| StageError::Io {
            path: out_path.clone(),
            source,
        })?;

        if malformed > 0 {
            tracing::warn!(
                "filter_dataset: dropped {malformed} malformed JSONL line(s) from {}",
                input.path.display()
            );
        }

        if kept == 0 {
            return Err(StageError::BadInput(format!(
                "filter_dataset removed all {} examples (min_turns={}, max_msg_bytes={}, drop_errors={}, malformed={})",
                input.n_examples, args.min_turns, args.max_msg_bytes, args.drop_errors, malformed
            )));
        }

        let content_hash = ContentHash::hash_file(&out_path).map_err(|source| StageError::Io {
            path: out_path.clone(),
            source,
        })?;

        // R21 post: kept > 0 (we just rejected kept == 0 above);
        // kept <= input.n_examples; output file exists.
        debug_assert!(kept > 0, "kept must be > 0 after the kept==0 guard");
        debug_assert!(
            kept <= input.n_examples.max(kept),
            "kept cannot exceed input n_examples"
        );
        Ok(DatasetJsonl {
            path: out_path,
            content_hash,
            n_examples: kept,
        })
    }
}

/// Verdict for a single line. `DropMalformed` is distinguished from
/// `DropPredicate` so the stage can count malformed input separately
/// — silent drops mask schema drift / upstream bugs.
enum Verdict {
    Keep,
    DropPredicate,
    DropMalformed,
}

/// Per-example predicate. Operates on the raw JSONL line so we
/// don't deserialize into a typed struct (the upstream JSONL schema
/// may vary across producers — keep this loose).
fn classify(line: &str, args: &Args) -> Verdict {
    let v: serde_json::Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(_) => return Verdict::DropMalformed,
    };
    // Missing `messages` key is schema-mismatch, not malformed.
    let messages = match v.get("messages").and_then(|m| m.as_array()) {
        Some(m) => m,
        None => return Verdict::DropPredicate,
    };
    if args.min_turns > 0 && (messages.len() as u32) < args.min_turns {
        return Verdict::DropPredicate;
    }
    for msg in messages {
        let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
        if args.max_msg_bytes > 0 && (content.len() as u32) > args.max_msg_bytes {
            return Verdict::DropPredicate;
        }
        if args.drop_errors {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
            if role == "tool"
                && (content.contains("Error")
                    || content.contains("error")
                    || content.contains("Traceback"))
            {
                return Verdict::DropPredicate;
            }
        }
    }
    Verdict::Keep
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::stage::Stage;

    fn write_jsonl(td: &std::path::Path, lines: &[&str]) -> std::path::PathBuf {
        let p = td.join("in.jsonl");
        std::fs::write(&p, lines.join("\n")).unwrap();
        p
    }

    fn ctx(td: &std::path::Path) -> StageContext {
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    #[tokio::test]
    async fn drops_short_examples() {
        let td = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(td.path().join("stage")).unwrap();
        let p = write_jsonl(
            td.path(),
            &[
                r#"{"messages":[{"role":"user","content":"hi"}]}"#,
                r#"{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]}"#,
            ],
        );
        let input = DatasetJsonl {
            path: p,
            content_hash: ContentHash::of_bytes(b""),
            n_examples: 2,
        };
        let args = Args {
            min_turns: 2,
            max_msg_bytes: 0,
            drop_errors: false,
        };
        let out = FilterDataset
            .run(&ctx(td.path()), input, &args)
            .await
            .unwrap();
        assert_eq!(out.n_examples, 1);
    }

    #[tokio::test]
    async fn drops_oversize_messages() {
        let td = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(td.path().join("stage")).unwrap();
        let huge = "x".repeat(2000);
        let body = format!(r#"{{"messages":[{{"role":"user","content":"{huge}"}}]}}"#);
        let small = r#"{"messages":[{"role":"user","content":"ok"}]}"#;
        let p = write_jsonl(td.path(), &[&body, small]);
        let input = DatasetJsonl {
            path: p,
            content_hash: ContentHash::of_bytes(b""),
            n_examples: 2,
        };
        let args = Args {
            min_turns: 0,
            max_msg_bytes: 1000,
            drop_errors: false,
        };
        let out = FilterDataset
            .run(&ctx(td.path()), input, &args)
            .await
            .unwrap();
        assert_eq!(out.n_examples, 1);
    }

    #[tokio::test]
    async fn fails_when_all_filtered() {
        let td = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(td.path().join("stage")).unwrap();
        let p = write_jsonl(
            td.path(),
            &[r#"{"messages":[{"role":"user","content":"a"}]}"#],
        );
        let input = DatasetJsonl {
            path: p,
            content_hash: ContentHash::of_bytes(b""),
            n_examples: 1,
        };
        let args = Args {
            min_turns: 10,
            max_msg_bytes: 0,
            drop_errors: false,
        };
        let r = FilterDataset.run(&ctx(td.path()), input, &args).await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[tokio::test]
    async fn drop_errors_filters_tool_errors() {
        let td = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(td.path().join("stage")).unwrap();
        let p = write_jsonl(
            td.path(),
            &[
                r#"{"messages":[{"role":"tool","content":"Traceback (most recent call last)"}]}"#,
                r#"{"messages":[{"role":"tool","content":"ok"}]}"#,
            ],
        );
        let input = DatasetJsonl {
            path: p,
            content_hash: ContentHash::of_bytes(b""),
            n_examples: 2,
        };
        let args = Args {
            min_turns: 0,
            max_msg_bytes: 0,
            drop_errors: true,
        };
        let out = FilterDataset
            .run(&ctx(td.path()), input, &args)
            .await
            .unwrap();
        assert_eq!(out.n_examples, 1);
    }
}
