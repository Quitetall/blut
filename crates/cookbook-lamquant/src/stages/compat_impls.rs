//! `Compatible<LamquantBackend>` impls for every LamQuant stage.
//!
//! Centralized here (vs sprinkled in each stage file) so an auditor can
//! see all backend-compatibility decisions in one place. A new LamQuant
//! stage added without a `Compatible<LamquantBackend>` impl will fail to
//! wire into any `Plan<O, LamquantBackend>` at compile time — a useful
//! "did you forget to declare backend compatibility?" guard.
//!
//! Every `lamquant_*` stage shells out to a LamQuant kernel script, so
//! all are tagged `Compatible<LamquantBackend>`. The trait
//! (`blut::framework::Compatible`) lives in blut-core; the backend
//! (`crate::backends::LamquantBackend`) and the stage types are
//! cookbook-local, so the orphan rule is satisfied here (NOT in
//! blut-core, where neither would be local after C2a).

use crate::backends::LamquantBackend;
use crate::stages::*;
use blut::framework::Compatible;

impl Compatible<LamquantBackend> for LamquantBuildManifest {}
impl Compatible<LamquantBackend> for LamquantBuildSplitManifest {}
impl Compatible<LamquantBackend> for LamquantConvertLma {}
impl Compatible<LamquantBackend> for LamquantEncodeLma {}
impl Compatible<LamquantBackend> for LamquantExportFirmware {}
impl Compatible<LamquantBackend> for LamquantGenerateSnnLabels {}
impl Compatible<LamquantBackend> for LamquantHardenArtifacts {}
impl Compatible<LamquantBackend> for LamquantPrecomputeFullband {}
impl Compatible<LamquantBackend> for LamquantPrecomputeL3 {}
impl Compatible<LamquantBackend> for LamquantPretrainMae {}
impl Compatible<LamquantBackend> for LamquantTrainTeacher {}
impl Compatible<LamquantBackend> for LamquantTrainL3Teacher {}
impl Compatible<LamquantBackend> for LamquantTrainJoint {}
impl Compatible<LamquantBackend> for LamquantTrainStudent {}
impl Compatible<LamquantBackend> for LamquantTrainVocosDecoder {}
impl Compatible<LamquantBackend> for LamquantTrainCombined {}
impl Compatible<LamquantBackend> for LamquantTrainMambaSnn {}
impl Compatible<LamquantBackend> for LamquantPccpGateSnn {}
impl Compatible<LamquantBackend> for LamquantPccpGateEncoder {}
impl Compatible<LamquantBackend> for LamquantPccpGateDecoder {}

#[cfg(test)]
mod tests {
    use super::*;
    use blut::framework::Stage;

    // Compile-time witness: each named stage typechecks against the
    // LamQuant backend. If a future stage refactor breaks the bound,
    // this `fn _f<...>()` line fails to compile — exactly the auditing
    // property we want.
    fn _lamquant_witness<S: Stage + Compatible<LamquantBackend>>() {}

    #[test]
    fn lamquant_witnesses() {
        _lamquant_witness::<LamquantBuildManifest>();
        _lamquant_witness::<LamquantTrainMambaSnn>();
        _lamquant_witness::<LamquantTrainJoint>();
        _lamquant_witness::<LamquantPccpGateSnn>();
    }
}
