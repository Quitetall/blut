//! LamQuant concrete stages (cookbook-local).
//!
//! Each `pub mod` here implements `blut::framework::Stage` for one
//! atomic unit of LamQuant work (manifest build, LMA convert/encode,
//! precompute, train, PCCP gate, harden, firmware export). Recipes (in
//! `crate::recipes`) compose these into typed Plans over
//! `crate::backends::LamquantBackend`. The `Compatible<...>` impls live
//! in `compat_impls` (orphan-rule home — both the stage types and the
//! backend are local to this crate). Carved out of blut-core's
//! `stages/` at C2a.

pub mod lamquant_build_manifest;
pub mod lamquant_build_split_manifest;
pub mod lamquant_convert_lma;
pub mod lamquant_encode_lma;
pub mod lamquant_export_firmware;
pub mod lamquant_generate_snn_labels;
pub mod lamquant_harden_artifacts;
pub(crate) mod lamquant_helpers;
pub mod lamquant_pccp_gate_encoder;
pub mod lamquant_pccp_gate_snn;
pub mod lamquant_precompute_fullband;
pub mod lamquant_precompute_l3;
pub mod lamquant_pretrain_mae;
pub mod lamquant_train_combined;
pub mod lamquant_train_joint;
pub mod lamquant_train_l3_teacher;
pub mod lamquant_train_mamba_snn;
pub mod lamquant_train_student;
pub mod lamquant_train_teacher;
pub mod lamquant_train_vocos_decoder;

mod compat_impls;

pub use lamquant_build_manifest::LamquantBuildManifest;
pub use lamquant_build_split_manifest::LamquantBuildSplitManifest;
pub use lamquant_convert_lma::LamquantConvertLma;
pub use lamquant_encode_lma::LamquantEncodeLma;
pub use lamquant_export_firmware::LamquantExportFirmware;
pub use lamquant_generate_snn_labels::LamquantGenerateSnnLabels;
pub use lamquant_harden_artifacts::LamquantHardenArtifacts;
pub use lamquant_pccp_gate_encoder::{LamquantPccpGateDecoder, LamquantPccpGateEncoder};
pub use lamquant_pccp_gate_snn::LamquantPccpGateSnn;
pub use lamquant_precompute_fullband::LamquantPrecomputeFullband;
pub use lamquant_precompute_l3::LamquantPrecomputeL3;
pub use lamquant_pretrain_mae::LamquantPretrainMae;
pub use lamquant_train_combined::LamquantTrainCombined;
pub use lamquant_train_joint::LamquantTrainJoint;
pub use lamquant_train_l3_teacher::LamquantTrainL3Teacher;
pub use lamquant_train_mamba_snn::LamquantTrainMambaSnn;
pub use lamquant_train_student::LamquantTrainStudent;
pub use lamquant_train_teacher::LamquantTrainTeacher;
pub use lamquant_train_vocos_decoder::LamquantTrainVocosDecoder;
