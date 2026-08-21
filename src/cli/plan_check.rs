// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `blut plan check` — typecheck a `.json` PlanSpec without publishing it.
//!
//! # Why this exists separately from `plan publish`
//!
//! [`crate::registry_db::publish`] already typechecks fail-closed: a spec that
//! does not compile never enters the `deployments` table. But publish also
//! *writes* — it opens the registry database and inserts an immutable row. So
//! anything that wanted to ask only the question "would this spec be accepted?"
//! had to either accept a side effect it did not want, or reimplement the
//! check. This asks the question and stops.
//!
//! Two callers want exactly that: CI, which should reject a bad spec in review
//! rather than at deploy time; and a foreign tool that generates PlanSpecs and
//! needs to know whether BLUT would take one before it claims it can drive
//! BLUT.
//!
//! # The one property that makes it worth having
//!
//! [`check`] calls the same two functions publish calls, in the same order —
//! `serde_json::from_str::<PlanSpec>` then [`PlanSpec::compile`]. It does not
//! reimplement either. That is deliberate: a second validator written to
//! approximate the first drifts, and once it drifts, a caller trusting the
//! cheap check is trusting a different question than the one deployment asks.
//! The test `check_and_publish_agree_on_every_case` holds the two to the same
//! verdict.
//!
//! # What acceptance does and does not mean
//!
//! Accepted means: the JSON matches the schema (`deny_unknown_fields`, so an
//! unknown field is a refusal rather than a silent drop), every named stage
//! resolves in *this binary's* registered cookbooks, and the graph kind-checks.
//!
//! It does not mean the plan will succeed, and it is not transferable between
//! binaries. Stage resolution is against compiled-in cookbooks, and blut-core
//! ships none — so the same spec can be accepted by a cookbook binary and
//! refused by a bare one. Callers that record a verdict should record which
//! binary produced it; `recipes_registered` in the `--json` output exists to
//! make a bare registry visible rather than surprising.

use crate::framework::Registry;
use crate::framework::plan_spec::PlanSpec;

/// A spec that typechecked, and the id publishing it would key on.
#[derive(Debug)]
pub struct Accepted {
    pub spec: PlanSpec,
    /// The ADR-0078 fingerprint, from [`crate::registry_db::fingerprint`] —
    /// the same function `publish` keys its row by, so a caller can bind
    /// evidence to spec content and have that id still mean something after
    /// the spec is deployed.
    pub fingerprint: String,
}

/// Typecheck `text` as a PlanSpec against `reg`. `Err` carries the refusal
/// reason already phrased for a human.
///
/// This signature is the non-mutation guarantee: it is handed a registry and a
/// string, and returns a verdict. It is given no database connection, no path,
/// and no clock.
pub fn check(reg: &Registry, text: &str) -> Result<Accepted, String> {
    let spec: PlanSpec =
        serde_json::from_str(text).map_err(|e| format!("PlanSpec JSON does not parse: {e}"))?;
    spec.compile(reg)
        .map_err(|e| format!("PlanSpec does not typecheck: {e}"))?;
    let fingerprint = crate::registry_db::fingerprint(&spec);
    Ok(Accepted { spec, fingerprint })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::LamuTrainerBackend;
    use crate::framework::Compatible;
    use crate::framework::artifact::{Artifact, ContentHash};
    use crate::framework::cookbook::Cookbook;
    use crate::framework::error::StageError;
    use crate::framework::resource::Resource;
    use crate::framework::stage::{ErasedStageCtor, Stage, StageContext};
    use crate::recipes::recipe::RecipeDef;
    use async_trait::async_trait;
    use serde::{Deserialize, Serialize};
    use std::path::Path;
    use std::sync::Arc;

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct Unit;
    impl Artifact for Unit {
        const KIND: &'static str = "plancheck.unit";
        const SCHEMA: u32 = 1;
        const INLINE: bool = true;
        fn content_hash(&self) -> ContentHash {
            ContentHash::of_bytes(b"plancheck.unit")
        }
        fn primary_path(&self) -> &Path {
            Path::new(".")
        }
    }

    #[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
    struct NoArgs {}

    struct MakeUnit;
    impl Compatible<LamuTrainerBackend> for MakeUnit {}
    #[async_trait]
    impl Stage for MakeUnit {
        const NAME: &'static str = "plancheck_make_unit";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = ();
        type Output = Unit;
        type Args = NoArgs;
        async fn run(&self, _c: &StageContext, _i: (), _a: &NoArgs) -> Result<Unit, StageError> {
            Ok(Unit)
        }
    }

    static ERASED: &[(&str, ErasedStageCtor)] = &[("plancheck_make_unit", || Arc::new(MakeUnit))];
    static NO_RECIPES: &[&RecipeDef] = &[];
    struct ToyCookbook;
    impl Cookbook for ToyCookbook {
        fn name(&self) -> &'static str {
            "plancheck_toy"
        }
        fn recipes(&self) -> &'static [&'static RecipeDef] {
            NO_RECIPES
        }
        fn stages_erased(&self) -> &'static [(&'static str, ErasedStageCtor)] {
            ERASED
        }
    }
    fn toy_registry() -> Registry {
        let mut reg = Registry::new();
        reg.register(Box::new(ToyCookbook));
        reg
    }

    const GOOD: &str = r#"{
      "name": "one-stage",
      "nodes": [{ "stage": "plancheck_make_unit", "args": {} }],
      "edges": [],
      "version": 1
    }"#;

    #[test]
    fn a_resolvable_spec_is_accepted_and_carries_publishs_fingerprint() {
        let reg = toy_registry();
        let ok = check(&reg, GOOD).expect("a spec naming a registered stage typechecks");
        assert_eq!(ok.spec.name, "one-stage");
        // The reported id must be the one `publish` would key the row by, or
        // a caller that recorded it has recorded something unresolvable.
        assert_eq!(ok.fingerprint, crate::registry_db::fingerprint(&ok.spec));
    }

    /// blut-core ships no cookbook, so the SAME text a cookbook binary accepts
    /// is refused by a bare registry. The message must name the stage, because
    /// "does not typecheck" alone sends the author looking at their JSON.
    #[test]
    fn the_same_spec_is_refused_by_a_registry_with_no_cookbooks() {
        let why = check(&Registry::new(), GOOD).expect_err("no cookbook can resolve the stage");
        assert!(why.contains("plancheck_make_unit"), "{why}");
        assert!(why.contains("not in any registered cookbook"), "{why}");
    }

    /// `deny_unknown_fields` is the reason a typo in a generated spec is a
    /// refusal rather than a silently ignored field. If that attribute is ever
    /// dropped from `PlanSpec`, this is what notices.
    #[test]
    fn an_unknown_field_is_refused_not_ignored() {
        let text = r#"{
          "name": "typo",
          "nodes": [],
          "edgs": [],
          "version": 1
        }"#;
        let why = check(&toy_registry(), text).expect_err("an unknown field must be refused");
        assert!(why.contains("does not parse"), "{why}");
        assert!(why.contains("edgs"), "unknown field must be named: {why}");
    }

    #[test]
    fn malformed_json_is_refused_before_the_registry_is_consulted() {
        let why = check(&toy_registry(), "not json at all").expect_err("must refuse");
        assert!(why.contains("does not parse"), "{why}");
    }

    /// The property the whole module rests on: `check` and `publish` must
    /// agree. If they ever disagree, a spec could pass CI and be refused at
    /// deploy — the failure mode a pre-flight check exists to prevent.
    ///
    /// Publishing needs a database, so this compares `check`'s verdict against
    /// publish's own typecheck expression (`PlanSpec::compile`) rather than
    /// against a live insert.
    #[test]
    fn check_and_publish_agree_on_every_case() {
        let reg = toy_registry();
        let bare = Registry::new();
        let cases: &[(&Registry, &str)] = &[
            (&reg, GOOD),
            (&bare, GOOD),
            (
                &reg,
                r#"{"name":"empty","nodes":[],"edges":[],"version":1}"#,
            ),
        ];
        for (registry, text) in cases {
            let mine = check(registry, text).is_ok();
            // Exactly what `registry_db::publish` gates on, inlined.
            let theirs = serde_json::from_str::<PlanSpec>(text)
                .ok()
                .is_some_and(|s| s.compile(registry).is_ok());
            assert_eq!(mine, theirs, "check and publish disagreed on: {text}");
        }
    }
}
