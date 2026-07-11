// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! ADR 0095 acceptance gate: the capability matrix (viewer/operator/admin/none ×
//! mutating actions) holds exactly, a `restricted`-tenant action is denied to
//! every shared-tenant token including `admin` (ADR 0061 clinical hard-block),
//! and every attempt — allow AND deny — produces one `audit.jsonl` row.

use blut::rbac::{Action, Principal, Role, authorize, enforce};
use blut::tenant::Tenant;

fn prin(id: &str, role: Role, tenant: &str) -> Principal {
    Principal {
        token_id: id.into(),
        role,
        tenant: Tenant::parse(tenant).unwrap(),
    }
}

#[test]
fn capability_matrix_holds_exactly() {
    let research = Tenant::parse("research/dev").unwrap();
    let viewer = prin("v", Role::Viewer, "research/dev");
    let operator = prin("o", Role::Operator, "research/dev");
    let admin = prin("a", Role::Admin, "research/dev");

    // (action, viewer, operator, admin, anonymous) — expected `allowed`.
    let matrix = [
        (Action::ReadStatus, true, true, true, true),
        (Action::ReadLineage, true, true, true, true),
        (Action::Run, false, true, true, false),
        (Action::Cancel, false, true, true, false),
        (Action::Retry, false, true, true, false),
        (Action::PlanPromote, false, false, true, false),
        (Action::PlanRollback, false, false, true, false),
        (Action::SecretSet, false, false, true, false),
        (Action::TokenAdmin, false, false, true, false),
        (Action::TenantAdmin, false, false, true, false),
    ];
    for (action, ev, eo, ea, en) in matrix {
        assert_eq!(
            authorize(Some(&viewer), action, &research).allowed,
            ev,
            "viewer {action:?}"
        );
        assert_eq!(
            authorize(Some(&operator), action, &research).allowed,
            eo,
            "operator {action:?}"
        );
        assert_eq!(
            authorize(Some(&admin), action, &research).allowed,
            ea,
            "admin {action:?}"
        );
        assert_eq!(
            authorize(None, action, &research).allowed,
            en,
            "anonymous {action:?}"
        );
    }
}

#[test]
fn clinical_boundary_dominates_role() {
    let clinical = Tenant::parse("clinical/prod").unwrap();
    let shared_admin = prin("a", Role::Admin, "research/dev");

    // NO shared-tenant token — not even admin — reaches a restricted tenant, for
    // ANY action (even a read), and neither does an anonymous caller.
    for action in [Action::ReadStatus, Action::Run, Action::PlanPromote] {
        assert!(
            !authorize(Some(&shared_admin), action, &clinical).allowed,
            "shared admin must be denied clinical {action:?}"
        );
        assert!(
            !authorize(None, action, &clinical).allowed,
            "anonymous must be denied clinical {action:?}"
        );
    }

    // A token minted IN the clinical tenant is subject only to its own role.
    let clinical_admin = prin("ca", Role::Admin, "clinical/prod");
    let clinical_op = prin("co", Role::Operator, "clinical/prod");
    assert!(authorize(Some(&clinical_admin), Action::PlanPromote, &clinical).allowed);
    assert!(authorize(Some(&clinical_op), Action::Run, &clinical).allowed);
    assert!(
        !authorize(Some(&clinical_op), Action::PlanPromote, &clinical).allowed,
        "role still applies within the restricted tenant"
    );
}

#[test]
fn every_attempt_allow_or_deny_is_audited() {
    let td = tempfile::tempdir().unwrap();
    let audit = td.path().join("audit.jsonl");
    let research = Tenant::parse("research/dev").unwrap();
    let clinical = Tenant::parse("clinical/prod").unwrap();
    let admin = prin("a", Role::Admin, "research/dev");

    let attempts: Vec<(Option<Principal>, Action, Tenant)> = vec![
        (Some(admin.clone()), Action::PlanPromote, research.clone()), // allow
        (Some(admin.clone()), Action::PlanPromote, clinical.clone()), // deny (clinical)
        (None, Action::Run, research.clone()),                        // deny (anon mutation)
        (Some(admin.clone()), Action::ReadStatus, research.clone()),  // allow
    ];
    let (mut allow, mut deny) = (0, 0);
    for (i, (p, action, t)) in attempts.iter().enumerate() {
        let d = enforce(p.as_ref(), *action, t, &audit, 1000 + i as i64);
        if d.allowed {
            allow += 1;
        } else {
            deny += 1;
        }
    }
    assert_eq!(allow, 2);
    assert_eq!(deny, 2);

    // EXACTLY one audit row per attempt (allow AND deny), and the denied
    // clinical attempt is on the record.
    let body = std::fs::read_to_string(&audit).unwrap();
    assert_eq!(
        body.lines().count(),
        attempts.len(),
        "one audit row per attempt"
    );
    assert!(body.contains("\"allowed\":false"));
    assert!(body.contains("clinical/prod"));
}
