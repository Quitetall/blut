//! `blut` binary — the LamQuant build of BLUT. Composes the cookbook
//! registry (LamQuant + lamu) and hands off to the domain-agnostic BLUT
//! CLI in `blut::cli`. blut-core itself ships no recipes; the cookbook
//! crates supply them. See `[[project_blut_cookbook_split]]`.

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Register the cookbooks this binary ships with. (LamQuant currently
    // re-exports from blut; it moves into this crate at C2a, no change
    // here. lamu rides along until it splits to blut-lamu at C2b.)
    let mut reg = cookbook_lamquant::registry();
    reg.register(Box::new(blut::framework::LamuCookbook));
    blut::cli::run(reg).await
}
