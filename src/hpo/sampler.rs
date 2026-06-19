//! HPO samplers (BLUT v0.20).
//!
//! A [`Sampler`] suggests the next trial's [`Overlay`]. [`RandomSampler`]
//! ignores history (independent random search); model-based samplers (TPE, a
//! later phase) condition on the completed [`TrialResult`]s.

use rand::SeedableRng;
use rand::rngs::StdRng;

use super::space::{Overlay, SearchSpace, TrialResult};

/// Suggests trial configurations. `Send` so an HPO run can hold one across the
/// async executor; not `Sync` (the run loop is the sole caller).
pub trait Sampler: Send {
    /// Suggest the next trial config. `completed` carries finished trials +
    /// their objectives for model-based samplers; `RandomSampler` ignores it.
    fn ask(&mut self, space: &SearchSpace, completed: &[TrialResult]) -> Overlay;
}

/// Independent random search — each dim sampled independently from its
/// distribution. Seedable, so an HPO run is reproducible.
pub struct RandomSampler {
    rng: StdRng,
}

impl RandomSampler {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Sampler for RandomSampler {
    fn ask(&mut self, space: &SearchSpace, _completed: &[TrialResult]) -> Overlay {
        space.sample(&mut self.rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hpo::space::Dist;

    fn space() -> SearchSpace {
        let mut s = SearchSpace::default();
        s.dims.insert(
            "lr".into(),
            Dist::Uniform {
                low: 0.0,
                high: 1.0,
            },
        );
        s
    }

    #[test]
    fn random_sampler_seed_reproducible_and_advances() {
        let (mut s1, mut s2) = (RandomSampler::new(7), RandomSampler::new(7));
        let sp = space();
        let a1 = s1.ask(&sp, &[]);
        let a2 = s2.ask(&sp, &[]);
        assert_eq!(a1, a2, "same seed → same first suggestion");
        let b1 = s1.ask(&sp, &[]);
        assert_ne!(a1, b1, "successive suggestions advance the RNG");
    }
}
