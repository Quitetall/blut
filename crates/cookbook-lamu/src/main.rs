//! `blut-lamu` binary — the lamu build of BLUT. Composes the lamu
//! cookbook registry (generic-LLM training: SFT / DPO / distill / eval)
//! and hands off to the domain-agnostic BLUT CLI in `blut::cli`.
//! blut-core itself ships no recipes; this crate supplies them. See
//! `[[project_blut_cookbook_split]]`.

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let reg = cookbook_lamu::registry();
    blut::cli::run(reg).await
}
