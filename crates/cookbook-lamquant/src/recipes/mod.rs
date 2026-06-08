//! LamQuant recipes (cookbook-local) — saved compositions of the
//! cookbook's stages. Each takes typed args and compiles to a
//! `blut::framework::plan::Plan<(), crate::backends::LamquantBackend>`.
//!
//! The catalog the cookbook hands BLUT is [`LAMQUANT_RECIPES`]; the
//! cookbook's `LamquantCookbook::recipes()` returns it and the runtime
//! `blut::framework::Registry` composes it with the other registered
//! cookbooks. (Carved out of blut-core's `recipes::recipe::RECIPES`
//! union at C2a — blut-core now ships only the lamu recipes.)

pub mod lamquant_combined_decoder;
pub mod lamquant_data_prep;
pub mod lamquant_encoder;
pub mod lamquant_full_pipeline;
pub mod lamquant_joint_codec;
pub mod lamquant_oracle;
pub mod lamquant_snn;

pub use lamquant_combined_decoder::LamquantCombinedDecoder;
pub use lamquant_data_prep::LamquantDataPrep;
pub use lamquant_encoder::LamquantEncoder;
pub use lamquant_full_pipeline::LamquantFullPipeline;
pub use lamquant_joint_codec::LamquantJointCodec;
pub use lamquant_oracle::LamquantOracle;
pub use lamquant_snn::LamquantSnn;

use blut::recipes::recipe::RecipeDef;

/// Return all LamQuant recipes whose `(input_kinds, output_kind)` tuple
/// matches `of`'s — i.e. drop-in swap candidates within this cookbook's
/// catalog. Excludes `of` itself by name. Mirrors
/// `blut::recipes::recipe::swap_candidates` but iterates
/// [`LAMQUANT_RECIPES`] (blut-core's `swap_candidates` reads blut-core's
/// own `RECIPES`, which no longer contains the lamquant recipes post-C2a).
pub fn swap_candidates(of: &'static RecipeDef) -> impl Iterator<Item = &'static RecipeDef> {
    let want_in = of.input_kinds;
    let want_out = of.output_kind;
    let name = of.name;
    LAMQUANT_RECIPES
        .iter()
        .copied()
        .filter(move |r| r.name != name && r.input_kinds == want_in && r.output_kind == want_out)
}

/// Recipes owned by the **blut-lamquant** cookbook (neural EEG codec).
/// This is the catalog the cookbook hands BLUT; the runtime
/// [`blut::framework::Registry`] composes it with the other registered
/// cookbooks. Carved out of blut-core's `RECIPES` union at C2a.
pub static LAMQUANT_RECIPES: &[&RecipeDef] = &[
    &lamquant_data_prep::DEF,
    &lamquant_combined_decoder::DEF,
    &lamquant_snn::DEF,
    &lamquant_encoder::DEF,
    &lamquant_joint_codec::DEF,
    &lamquant_oracle::DEF,
    &lamquant_full_pipeline::DEF,
];
