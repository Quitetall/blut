// SPDX-License-Identifier: AGPL-3.0-or-later
// The ADR 0094 acceptance gate (SLA/notify half): a forced max_runtime breach
// for a RESTRICTED run produces an sla.jsonl row and EXACTLY ONE redacted
// (PHI-free) notification through a local sink — and that same restricted
// breach is REFUSED at an off-box sink (clinical content never leaves the box).

use blut_notify::{NotificationEnvelope, NotifySink, SinkBoundary, notify_breaches, read_breaches};
use blut_types::sla::{SlaBreach, SlaKind};
use blut_types::trust::DataClass;

/// Captures every envelope it is handed so the test can inspect the payload.
struct CapturingSink {
    boundary: SinkBoundary,
    seen: Vec<NotificationEnvelope>,
}

impl NotifySink for CapturingSink {
    fn boundary(&self) -> SinkBoundary {
        self.boundary
    }
    fn send(&mut self, envelope: &NotificationEnvelope) -> Result<(), String> {
        self.seen.push(envelope.clone());
        Ok(())
    }
}

#[test]
fn sla_breach_writes_a_row_and_notifies_once_redacted() {
    let td = tempfile::tempdir().unwrap();
    let sla_path = td.path().join("sla.jsonl");

    // A forced max_runtime breach on a RESTRICTED (clinical) run. The producer
    // (`blut sla check`) writes a PHI-FREE summary — job id + seconds only.
    let breach = SlaBreach {
        rule: "clinical-cap-1h".into(),
        kind: SlaKind::MaxRuntime,
        job_id: "job-eeg-77".into(),
        tenant: "clinical/prod".into(),
        data_class: DataClass::Restricted,
        observed_secs: 4200,
        limit_secs: 3600,
        summary: "max_runtime breach on job job-eeg-77: 4200s vs limit 3600s".into(),
        detected_unix: 1_700_000_000,
    };
    // Write the sla.jsonl row (proving the shared keystone format round-trips
    // from producer to consumer).
    {
        use std::io::Write as _;
        let mut f = std::fs::File::create(&sla_path).unwrap();
        writeln!(f, "{}", breach.to_line()).unwrap();
    }

    // The row exists and parses back.
    let breaches = read_breaches(&sla_path).unwrap();
    assert_eq!(breaches.len(), 1, "one sla.jsonl row");
    assert_eq!(breaches[0], breach);

    // LOCAL sink: exactly one notification, and its payload is PHI-free.
    let mut local = CapturingSink {
        boundary: SinkBoundary::Local,
        seen: Vec::new(),
    };
    let delivered = notify_breaches(&mut local, &breaches).unwrap();
    assert_eq!(
        delivered, 1,
        "exactly one redacted notification for the breach"
    );
    assert_eq!(local.seen.len(), 1);
    let payload = &local.seen[0];
    assert!(payload.tenant.is_restricted());
    assert_eq!(payload.summary, breach.summary);
    assert!(
        !payload.summary.contains("patient") && !payload.summary.contains("name"),
        "the notification must be PHI-free"
    );

    // OFF-BOX sink: the SAME restricted breach is refused — 0 delivered, and the
    // sink is never even handed the payload (custody chokepoint, fail-closed).
    let mut offbox = CapturingSink {
        boundary: SinkBoundary::OffBox,
        seen: Vec::new(),
    };
    let delivered_off = notify_breaches(&mut offbox, &breaches).unwrap();
    assert_eq!(delivered_off, 0, "a restricted breach never leaves the box");
    assert!(offbox.seen.is_empty(), "off-box sink saw no payload");
}
