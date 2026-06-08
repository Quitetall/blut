//! `blut` binary — the LamQuant build of BLUT. Composes the LamQuant
//! cookbook registry and hands off to the domain-agnostic BLUT CLI in
//! `blut::cli`. blut-core itself ships no recipes; this crate supplies
//! the LamQuant cookbook. The lamu cookbook moved to its own
//! `blut-lamu` binary at C2b. See `[[project_blut_cookbook_split]]`.

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let reg = cookbook_lamquant::registry();
    blut::cli::run(reg).await
}
