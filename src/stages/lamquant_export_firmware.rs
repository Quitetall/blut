//! Stage — `lamquant_export_firmware`. (HardenedCkpt, JointCkpt, SnnCkpt) → FirmwareBundle.
//!
//! Wraps `firmware/export_firmware.py`. Takes the hardened encoder
//! (Route B), the matching decoder ckpt from `JointCkpt`, and the
//! Mamba SNN ckpt; emits C headers + a flash-ready `.bin` under
//! `bundle_dir`. `target` selects per-MCU memory layouts (RP2350 /
//! NRF54L15 / ESP32-P4 / STM32N6).
//!
//! Deterministic per-input-byte-state — outputs are a pure function
//! of the input checkpoints + target config.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::lamquant::stat_fingerprint;
use crate::artifacts::{FirmwareBundle, HardenedCkpt, JointCkpt, SnnCkpt};
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};
use crate::lamquant_backend::{LamquantBackend, LamquantInvocation};
use crate::stages::lamquant_helpers::{
    blut_env, progress_forwarder, python_for, resolve_home, safe_join, script_path,
};

pub struct LamquantExportFirmware;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    #[serde(default)]
    pub lamquant_home: String,
    /// Target MCU. Drives memory layout in the export script.
    /// Accepted: rp2350 | nrf54l15 | esp32p4 | stm32n6.
    #[serde(default = "default_target")]
    pub target: String,
    /// Output bundle directory relative to lamquant_home. Empty =
    /// builder default `firmware/export/<target>/`.
    #[serde(default)]
    pub bundle_dir_rel: String,
    /// Pass `--include-snn` so the SNN ckpt is bundled. Defaults
    /// true since SNN is part of the production firmware payload.
    #[serde(default = "default_true")]
    pub include_snn: bool,
}

fn default_target() -> String {
    "rp2350".into()
}

fn default_true() -> bool {
    true
}

#[async_trait]
impl Stage for LamquantExportFirmware {
    const NAME: &'static str = "lamquant_export_firmware";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const DETERMINISTIC: bool = true;
    type Input = (HardenedCkpt, JointCkpt, SnnCkpt);
    type Output = FirmwareBundle;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        input: Self::Input,
        args: &Args,
    ) -> Result<FirmwareBundle, StageError> {
        let valid_targets = ["rp2350", "nrf54l15", "esp32p4", "stm32n6"];
        if !valid_targets.contains(&args.target.as_str()) {
            return Err(StageError::BadInput(format!(
                "target must be one of {valid_targets:?}, got {:?}",
                args.target
            )));
        }
        let home = resolve_home(&args.lamquant_home)?;
        let python = python_for(&home);
        let script = script_path(&home, &["firmware", "export_firmware.py"])?;

        let (hardened, joint, snn) = input;
        let bundle_dir = if args.bundle_dir_rel.is_empty() {
            home.join("firmware").join("export").join(&args.target)
        } else {
            safe_join(&home, &args.bundle_dir_rel)?
        };
        std::fs::create_dir_all(&bundle_dir).map_err(|source| StageError::Io {
            path: bundle_dir.clone(),
            source,
        })?;
        let bin_path = bundle_dir.join(format!("lamquant_{}.bin", args.target));

        let mut cmd_args = vec![
            "--target".into(),
            args.target.clone(),
            "--encoder".into(),
            hardened.path.display().to_string(),
            "--decoder".into(),
            joint.decoder_path.display().to_string(),
            "--output".into(),
            bundle_dir.display().to_string(),
        ];
        if args.include_snn {
            cmd_args.push("--snn".into());
            cmd_args.push(snn.path.display().to_string());
        }

        let inv = LamquantInvocation {
            python,
            script,
            cwd: home,
            args: cmd_args,
            env: blut_env(&ctx.job_dir, Self::NAME),
            expected_outputs: vec![bundle_dir.clone()],
            run_manifest_path: None,
        };
        let mut backend = LamquantBackend::new();
        backend
            .run(
                inv,
                Some(progress_forwarder(Self::NAME, ctx.status_tx.clone())),
            )
            .await
            .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;

        let content_hash =
            stat_fingerprint(b"lamquant.firmware_bundle", &bundle_dir).map_err(|source| {
                StageError::Io {
                    path: bundle_dir.clone(),
                    source,
                }
            })?;
        Ok(FirmwareBundle {
            bundle_dir,
            bin_path,
            target: args.target.clone(),
            content_hash,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::artifact::ContentHash;
    use std::path::PathBuf;

    fn ctx(td: &std::path::Path) -> StageContext {
        std::fs::create_dir_all(td.join("stage")).unwrap();
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    fn hardened() -> HardenedCkpt {
        HardenedCkpt {
            path: PathBuf::from("/tmp/h"),
            content_hash: ContentHash::of_bytes(b""),
            route_b: true,
        }
    }

    fn joint() -> JointCkpt {
        JointCkpt {
            encoder_path: PathBuf::from("/tmp/enc"),
            decoder_path: PathBuf::from("/tmp/dec"),
            content_hash: ContentHash::of_bytes(b""),
            final_loss: 0.0,
            tier: 3,
            preset: "production".into(),
        }
    }

    fn snn() -> SnnCkpt {
        SnnCkpt {
            path: PathBuf::from("/tmp/snn"),
            content_hash: ContentHash::of_bytes(b""),
            head_size_kb: 8.0,
            final_loss: 0.0,
        }
    }

    #[tokio::test]
    async fn rejects_invalid_target() {
        let td = tempfile::tempdir().unwrap();
        let r = LamquantExportFirmware
            .run(
                &ctx(td.path()),
                (hardened(), joint(), snn()),
                &Args {
                    lamquant_home: td.path().display().to_string(),
                    target: "esp32-s3".into(),
                    bundle_dir_rel: String::new(),
                    include_snn: true,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[tokio::test]
    async fn accepts_all_four_canonical_targets() {
        // Each target should reach the "home not found" / "script not
        // found" failure rather than the target-validation error.
        for tgt in ["rp2350", "nrf54l15", "esp32p4", "stm32n6"] {
            let td = tempfile::tempdir().unwrap();
            let r = LamquantExportFirmware
                .run(
                    &ctx(td.path()),
                    (hardened(), joint(), snn()),
                    &Args {
                        lamquant_home: td.path().join("nope").display().to_string(),
                        target: tgt.into(),
                        bundle_dir_rel: String::new(),
                        include_snn: true,
                    },
                )
                .await;
            assert!(matches!(r, Err(StageError::BadInput(_))));
        }
    }
}
