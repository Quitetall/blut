//! LamQuant typed artifacts (cookbook-local).
//!
//! Each artifact references on-disk bytes and implements
//! `blut::framework::Artifact`. Together with the cookbook's stages
//! that produce / consume them, they define the LamQuant typed lattice
//! (LmaCorpus → manifests → checkpoints → PCCP verdicts → firmware
//! bundle). Re-exported flat at `crate::artifacts::*` so the moved
//! recipe / stage files resolve `crate::artifacts::Manifest` etc.
//! intra-cookbook (mirrors the pre-C2a blut-core layout).

pub mod lamquant;

pub use lamquant::{
    FirmwareBundle, FullbandMemmap, HardenedCkpt, JointCkpt, L3Cache, LmaCorpus, MaeCkpt, Manifest,
    PccpVerdict, SnnCkpt, SnnLabels, SplitManifest, TeacherCkpt,
};
