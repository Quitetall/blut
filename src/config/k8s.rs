// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `LaunchTarget::K8s` submission (ADR 0067 T4.5 · B3).
//!
//! K8s is an ADAPTER, not the control plane (ADR 0037): the engine emits a
//! `BlutPlan` manifest as TEXT and shells `kubectl apply -f -`. There is ZERO
//! kube-rs dependency in the engine — the typed CRDs + the reconcile loop live
//! entirely in the `blut-operator` crate. This keeps the ADR 0034 no-in-engine-
//! server / no-kube-in-core charter intact.

use std::io::Write;
use std::process::{Command, Stdio};

use crate::error::{Result, TrainError};

/// A minimal spec for the `BlutPlan` a `--launcher k8s` run submits. Mirrors
/// the operator CRD's fields; kept as plain data so the engine needs no kube
/// types. `data_class_ceiling` is `Public`/`Internal` only — `Restricted` is
/// not a value the engine will emit (ADR 0061 clinical hard-block).
pub struct PlanSubmission {
    pub name: String,
    pub cookbook_image: String,
    /// The PlanSpec JSON (or Starlark) to run, delivered inline.
    pub plan_content: String,
    /// `Public` or `Internal` (never `Restricted`).
    pub data_class_ceiling: String,
    /// Optional args JSON.
    pub args: Option<String>,
    /// Target namespace (`default` if `None`).
    pub namespace: Option<String>,
    /// GPUs the plan's pod requests — rendered as a `gpus:` spec field the
    /// operator maps to a `nvidia.com/gpu` resource limit (ADR 0087 distributed
    /// tail). `0` ⇒ the field is omitted, so the manifest is byte-identical to
    /// the pre-ADR render.
    pub gpus: u32,
}

/// YAML-escape a scalar as a double-quoted string (covers `"`, `\`, and
/// newlines — enough for image refs, names, and JSON blobs).
fn yaml_quote(s: &str) -> String {
    let escaped = s
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t");
    format!("\"{escaped}\"")
}

/// Render a `BlutPlan` manifest as YAML text. Pure + testable — no cluster, no
/// kube types. Rejects a `Restricted` ceiling fail-closed (defense in depth:
/// the operator CRD also can't represent it).
pub fn render_blut_plan_manifest(sub: &PlanSubmission) -> Result<String> {
    // Fail-fast whitelist: the engine emits ONLY Public/Internal. Restricted is
    // named explicitly for a clear clinical-block error; any other value is
    // rejected too (defense in depth atop the CRD's unrepresentable-Restricted).
    match sub.data_class_ceiling.as_str() {
        "Public" | "Internal" => {}
        "Restricted" => {
            return Err(TrainError::other(
                "refusing to submit a Restricted-class plan to a cluster \
                 (ADR 0061 clinical hard-block)",
            ));
        }
        other => {
            return Err(TrainError::other(format!(
                "invalid dataClassCeiling '{other}' (expected Public or Internal)"
            )));
        }
    }
    let ns = sub.namespace.as_deref().unwrap_or("default");
    let mut y = String::new();
    y.push_str("apiVersion: blut.lamquant.dev/v1alpha1\n");
    y.push_str("kind: BlutPlan\n");
    y.push_str("metadata:\n");
    y.push_str(&format!("  name: {}\n", yaml_quote(&sub.name)));
    y.push_str(&format!("  namespace: {}\n", yaml_quote(ns)));
    y.push_str("spec:\n");
    y.push_str("  planSource:\n");
    y.push_str("    inline:\n");
    y.push_str(&format!(
        "      content: {}\n",
        yaml_quote(&sub.plan_content)
    ));
    y.push_str(&format!(
        "  cookbookImage: {}\n",
        yaml_quote(&sub.cookbook_image)
    ));
    y.push_str(&format!(
        "  dataClassCeiling: {}\n",
        yaml_quote(&sub.data_class_ceiling)
    ));
    if let Some(args) = &sub.args {
        y.push_str(&format!("  args: {}\n", yaml_quote(args)));
    }
    // ADR 0087: only emit `gpus:` when the plan actually asks for GPUs — 0 keeps
    // the manifest byte-identical to the pre-ADR render. The operator maps this
    // to a `nvidia.com/gpu` limit on the pod.
    if sub.gpus > 0 {
        // `gpus` is a `u32` — safe to render as a bare YAML integer (no
        // `yaml_quote` needed, unlike the string scalars above).
        y.push_str(&format!("  gpus: {}\n", sub.gpus));
    }
    Ok(y)
}

/// Submit a plan to a cluster: render the manifest and pipe it to
/// `kubectl apply -f -`. The engine never links kube — this is a subprocess.
pub fn submit_plan(sub: &PlanSubmission) -> Result<()> {
    let manifest = render_blut_plan_manifest(sub)?;
    let mut child = Command::new("kubectl")
        .args(["apply", "-f", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| TrainError::other(format!("kubectl spawn failed (is it installed?): {e}")))?;
    // Write the manifest, then ALWAYS reap the child (don't `?`-return on a
    // write error before waiting — that would drop the Child unreaped). Take +
    // drop stdin so kubectl sees EOF.
    let write_result = {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| TrainError::other("kubectl stdin unavailable"))?;
        stdin.write_all(manifest.as_bytes())
    };
    let out = child
        .wait_with_output()
        .map_err(|e| TrainError::other(format!("kubectl wait failed: {e}")))?;
    write_result.map_err(|e| TrainError::other(format!("kubectl stdin write failed: {e}")))?;
    if !out.status.success() {
        return Err(TrainError::other(format!(
            "kubectl apply failed (exit {}): {}",
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sub() -> PlanSubmission {
        PlanSubmission {
            name: "demo".into(),
            cookbook_image: "ghcr.io/lamquant/blut:latest".into(),
            plan_content: "{\"nodes\":[]}".into(),
            data_class_ceiling: "Internal".into(),
            args: None,
            namespace: Some("blut".into()),
            gpus: 0,
        }
    }

    #[test]
    fn manifest_has_kind_ceiling_and_escaped_json() {
        let y = render_blut_plan_manifest(&sub()).unwrap();
        assert!(y.contains("kind: BlutPlan"));
        assert!(y.contains("dataClassCeiling: \"Internal\""));
        assert!(y.contains("namespace: \"blut\""));
        // The JSON plan content is escaped inside a quoted scalar.
        assert!(y.contains("content: \"{\\\"nodes\\\":[]}\""));
    }

    #[test]
    fn restricted_ceiling_is_refused() {
        let mut s = sub();
        s.data_class_ceiling = "Restricted".into();
        let err = render_blut_plan_manifest(&s).unwrap_err();
        assert!(format!("{err}").contains("Restricted"));
    }

    #[test]
    fn default_namespace_when_unset() {
        let mut s = sub();
        s.namespace = None;
        let y = render_blut_plan_manifest(&s).unwrap();
        assert!(y.contains("namespace: \"default\""));
    }

    #[test]
    fn args_included_only_when_present() {
        let mut s = sub();
        assert!(!render_blut_plan_manifest(&s).unwrap().contains("args:"));
        s.args = Some("{\"tier\":3}".into());
        assert!(render_blut_plan_manifest(&s).unwrap().contains("args:"));
    }

    #[test]
    fn gpus_emitted_only_when_nonzero() {
        // ADR 0087: gpus=0 keeps the manifest byte-identical to the pre-ADR render.
        let mut s = sub();
        assert!(!render_blut_plan_manifest(&s).unwrap().contains("gpus:"));
        s.gpus = 4;
        assert!(render_blut_plan_manifest(&s).unwrap().contains("gpus: 4\n"));
    }
}
