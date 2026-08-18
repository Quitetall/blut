// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! ADR 0109's `Gp` surrogate: Gaussian-process regression with a Matérn 5/2
//! kernel, and expected hypervolume improvement (EHVI) as the multi-objective
//! acquisition.
//!
//! **Scope, stated plainly.** The ADR names "expected-hypervolume-improvement
//! (qEHVI)". What ships here is the **q=1** case computed by Monte Carlo:
//! sample each objective's posterior, measure how much the sampled point would
//! grow the Pareto front's hypervolume, average. That is unbiased and converges
//! to the true EHVI, and it is honest about being sampled rather than a closed
//! form. The `q>1` batch case — which needs the joint posterior over a batch
//! plus inclusion–exclusion over overlapping boxes — is NOT implemented, and
//! [`ehvi_mc`] takes one candidate to make that impossible to mistake.
//!
//! **No linear-algebra dependency.** The Cholesky factorisation, the two
//! substitutions, and the kernel are ~60 lines here. Pulling `nalgebra` into
//! the engine to avoid them would be a real dependency decision on a crate that
//! ADR 0034 keeps deliberately lean, for arithmetic that fits on a page.
//!
//! **Objectives are modelled independently**, one GP per objective. That is the
//! standard EHVI setup and it is what the ADR's "GP w/ Matérn kernel" implies;
//! it does not model correlation *between* objectives, so a study where
//! quality and cost are strongly coupled gets an optimistic variance. Recorded
//! rather than hidden.

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::pareto::{Direction, ParetoPoint, hypervolume_2d};
use super::space::{Dist, Overlay, SearchSpace, TrialResult};
use super::tpe::{support_t, to_t};

/// GP hyperparameters. Fixed rather than marginal-likelihood-optimised: an
/// inner optimisation loop is a second thing to get wrong, and on the trial
/// counts an HPO study actually reaches (tens, not thousands) a sane fixed
/// length scale over a normalised space is not the binding constraint.
#[derive(Clone, Debug)]
pub struct GpConfig {
    /// Matérn length scale, in units of the normalised [0,1] space.
    pub length_scale: f64,
    /// Kernel amplitude. Applied to STANDARDISED targets, so 1.0 means "one
    /// standard deviation of the observed objective".
    pub signal_variance: f64,
    /// Observation noise. Also the numerical floor that keeps the Gram matrix
    /// positive definite when two trials sit at nearly the same config.
    pub noise_variance: f64,
}

impl Default for GpConfig {
    fn default() -> Self {
        Self {
            length_scale: 0.25,
            signal_variance: 1.0,
            noise_variance: 1e-6,
        }
    }
}

/// Matérn 5/2 covariance for a scaled distance `r = d / length_scale`.
///
/// Chosen over the squared exponential deliberately: the SE kernel assumes an
/// infinitely differentiable response surface, which makes it overconfident
/// between observations. Matérn 5/2 assumes twice-differentiable and is the
/// standard default for Bayesian optimisation for exactly that reason.
fn matern52(d: f64, cfg: &GpConfig) -> f64 {
    let r = (d / cfg.length_scale.max(1e-12)).abs();
    let s5r = 5.0_f64.sqrt() * r;
    cfg.signal_variance * (1.0 + s5r + 5.0 * r * r / 3.0) * (-s5r).exp()
}

/// Encode one overlay as a point in normalised feature space.
///
/// Continuous dims map through the same t-space transform every other sampler
/// uses (so `log_uniform` is compared in log space) and are scaled to [0,1] by
/// their declared support — never by the observed data, which would make the
/// encoding shift as trials arrive and silently invalidate a fitted model.
/// Categorical dims get one coordinate per choice, at `1/sqrt(2)` so that two
/// different categories sit exactly distance 1 apart, matching a full sweep of
/// a continuous dim.
fn encode(space: &SearchSpace, overlay: &Overlay) -> Vec<f64> {
    let mut out = Vec::new();
    for (name, dist) in &space.dims {
        let value = overlay.iter().find(|(k, _)| k == name).map(|(_, v)| v);
        match dist {
            Dist::Choice { choices } => {
                let idx = value.and_then(|v| choices.iter().position(|c| c == v));
                for i in 0..choices.len() {
                    // An absent/unknown category leaves every coordinate at 0,
                    // which is distance 1/sqrt(2) from each real category —
                    // uninformative, not a match for any of them.
                    out.push(if Some(i) == idx {
                        std::f64::consts::FRAC_1_SQRT_2
                    } else {
                        0.0
                    });
                }
            }
            _ => {
                let (lo, hi) = support_t(dist).unwrap_or((0.0, 1.0));
                let span = (hi - lo).abs().max(1e-12);
                let t = value
                    .and_then(|v| v.as_f64())
                    .map(|x| (to_t(dist, x) - lo) / span)
                    .unwrap_or(0.5); // absent ⇒ the middle, not an endpoint
                out.push(t.clamp(0.0, 1.0));
            }
        }
    }
    out
}

fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
}

/// Cholesky `A = L Lᵀ` for symmetric positive-definite `A`, returning `None`
/// when a pivot is non-positive (i.e. `A` was not PD at this jitter).
fn cholesky(a: &[Vec<f64>], jitter: f64) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            if i == j {
                s += jitter;
            }
            s -= l[i][..j]
                .iter()
                .zip(l[j][..j].iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
            if i == j {
                // NaN must fail here too: `s <= 0.0` alone is false for NaN, so
                // the finiteness check is what actually rejects it.
                if !s.is_finite() || s <= 0.0 {
                    return None;
                }
                l[i][j] = s.sqrt();
            } else {
                let d = l[j][j];
                if d.abs() < 1e-300 {
                    return None;
                }
                l[i][j] = s / d;
            }
        }
    }
    Some(l)
}

/// Solve `L z = b`.
fn forward_sub(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut z = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        // `z.iter().enumerate().take(i)` rather than `for k in 0..i`: the
        // borrow ends with the inner loop, so the `z[i] = …` write below is
        // still fine, and MSRV clippy (1.88) rejects the index form under
        // `-D warnings` even though a newer clippy does not.
        for (k, zk) in z.iter().enumerate().take(i) {
            s -= l[i][k] * zk;
        }
        z[i] = s / l[i][i];
    }
    z
}

/// Solve `Lᵀ x = z`.
fn back_sub(l: &[Vec<f64>], z: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = z[i];
        for k in (i + 1)..n {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i];
    }
    x
}

/// A fitted single-objective GP.
#[derive(Clone, Debug)]
pub struct GpModel {
    cfg: GpConfig,
    /// The space this model was FITTED on. Owned rather than re-supplied at
    /// predict time: `encode` lays out coordinates per declared dimension, and
    /// `sq_dist` zips, so encoding a query against a different space silently
    /// truncates to the shorter vector and returns a CONFIDENT wrong answer
    /// (measured: a model fitted on one dim, queried against two, reported
    /// mu=10.0 sd=0.001 instead of refusing). Owning the space deletes that
    /// failure mode rather than asserting against it.
    space: SearchSpace,
    xs: Vec<Vec<f64>>,
    /// `K⁻¹ y` in standardised target space.
    alpha: Vec<f64>,
    chol: Vec<Vec<f64>>,
    y_mean: f64,
    y_std: f64,
}

impl GpModel {
    /// Fit to `(overlay, objective)` pairs.
    ///
    /// Returns `None` when there is nothing to fit or the Gram matrix cannot be
    /// factorised even after escalating jitter — a refusal, never a silently
    /// degenerate model that would report confident nonsense.
    pub fn fit(space: &SearchSpace, trials: &[TrialResult], cfg: GpConfig) -> Option<GpModel> {
        // Non-finite objectives are dropped, not ranked: they are unmeasured.
        let used: Vec<&TrialResult> = trials.iter().filter(|t| t.objective.is_finite()).collect();
        if used.is_empty() {
            return None;
        }
        let xs: Vec<Vec<f64>> = used.iter().map(|t| encode(space, &t.overlay)).collect();
        let ys: Vec<f64> = used.iter().map(|t| t.objective).collect();

        // Standardise targets so the GP's zero prior mean is the sample mean.
        // Without this, a study whose objective lives near 1000 is modelled as
        // a huge excursion from 0 and the posterior is dominated by the prior.
        let n = ys.len() as f64;
        let y_mean = ys.iter().sum::<f64>() / n;
        let var = ys.iter().map(|y| (y - y_mean) * (y - y_mean)).sum::<f64>() / n;
        // A constant objective has zero spread; unit std keeps it finite and
        // the model correctly predicts the constant with no signal.
        let y_std = if var > 1e-12 { var.sqrt() } else { 1.0 };
        let yz: Vec<f64> = ys.iter().map(|y| (y - y_mean) / y_std).collect();

        let m = xs.len();
        let mut k = vec![vec![0.0_f64; m]; m];
        for i in 0..m {
            for j in 0..m {
                k[i][j] = matern52(sq_dist(&xs[i], &xs[j]).sqrt(), &cfg);
            }
        }

        // Escalate jitter rather than fail on the first ill-conditioned Gram
        // matrix: duplicate configs are NORMAL in HPO (a re-proposed config is
        // a cache hit, per the ADR), and they make rows identical.
        let mut jitter = cfg.noise_variance.max(1e-10);
        let chol = loop {
            if let Some(l) = cholesky(&k, jitter) {
                break l;
            }
            jitter *= 10.0;
            if jitter > 1.0 {
                return None;
            }
        };
        let alpha = back_sub(&chol, &forward_sub(&chol, &yz));
        Some(GpModel {
            cfg,
            space: space.clone(),
            xs,
            alpha,
            chol,
            y_mean,
            y_std,
        })
    }

    /// Posterior mean and standard deviation at `overlay`, in the ORIGINAL
    /// objective units. `sd` is clamped at zero — a tiny negative from
    /// round-off is not evidence of negative variance.
    pub fn predict(&self, overlay: &Overlay) -> (f64, f64) {
        let x = encode(&self.space, overlay);
        let ks: Vec<f64> = self
            .xs
            .iter()
            .map(|xi| matern52(sq_dist(&x, xi).sqrt(), &self.cfg))
            .collect();
        let mean_z: f64 = ks.iter().zip(self.alpha.iter()).map(|(a, b)| a * b).sum();
        let v = forward_sub(&self.chol, &ks);
        let var_z = (self.cfg.signal_variance - v.iter().map(|a| a * a).sum::<f64>()).max(0.0);
        (mean_z * self.y_std + self.y_mean, var_z.sqrt() * self.y_std)
    }

    /// Number of observations backing this fit.
    pub fn len(&self) -> usize {
        self.xs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.xs.is_empty()
    }
}

/// Standard normal draw (Box–Muller).
fn std_normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
    let u2: f64 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// What an EHVI score is measured against: the incumbent front, the per-axis
/// directions, and the reference point bounding the dominated region.
///
/// Grouped rather than passed loose because they are only meaningful together —
/// a front interpreted under the wrong `dirs`, or against a reference on the
/// wrong side of it, yields a confident number that means nothing.
#[derive(Clone, Copy, Debug)]
pub struct HypervolumeTarget<'a> {
    pub front: &'a [ParetoPoint],
    pub dirs: &'a [Direction],
    pub reference: [f64; 2],
}

/// Monte-Carlo **expected hypervolume improvement** for ONE candidate (q=1).
///
/// Draws `samples` joint outcomes from the per-objective posteriors, measures
/// how much each would grow the front's dominated hypervolume against
/// `reference`, and averages. Improvement is clamped at zero per sample, which
/// is what makes this an *improvement* rather than a signed change: a candidate
/// that lands inside the existing front contributes nothing, it does not
/// subtract.
///
/// Restricted to two objectives because [`hypervolume_2d`] is. Returns `0.0`
/// for any other arity rather than a number that looks meaningful.
pub fn ehvi_mc(
    models: &[GpModel],
    candidate: &Overlay,
    target: &HypervolumeTarget<'_>,
    samples: usize,
    rng: &mut StdRng,
) -> f64 {
    let HypervolumeTarget {
        front,
        dirs,
        reference,
    } = *target;
    if models.len() != 2 || dirs.len() != 2 || samples == 0 {
        return 0.0;
    }
    let base = hypervolume_2d(front, reference, dirs);
    let posteriors: Vec<(f64, f64)> = models.iter().map(|m| m.predict(candidate)).collect();

    let mut acc = 0.0;
    for _ in 0..samples {
        let drawn: Vec<f64> = posteriors
            .iter()
            .map(|(mu, sd)| mu + sd * std_normal(rng))
            .collect();
        if drawn.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let mut augmented = front.to_vec();
        augmented.push(ParetoPoint::new(u32::MAX, drawn));
        let grown = hypervolume_2d(&augmented, reference, dirs);
        acc += (grown - base).max(0.0);
    }
    acc / samples as f64
}

/// Single-objective expected improvement, for a study that declared one
/// objective. Kept beside EHVI so a 1-objective study is not forced through a
/// 2-objective code path.
///
/// `xi` is the exploration margin; 0.0 is pure greedy EI.
pub fn expected_improvement(mean: f64, sd: f64, best: f64, xi: f64, maximize: bool) -> f64 {
    if !sd.is_finite() || sd <= 0.0 {
        // No posterior uncertainty ⇒ EI is exactly the (clamped) mean gain.
        let gain = if maximize { mean - best } else { best - mean };
        return gain.max(0.0);
    }
    let improvement = if maximize {
        mean - best - xi
    } else {
        best - mean - xi
    };
    let z = improvement / sd;
    let pdf = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf = 0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2));
    (improvement * cdf + sd * pdf).max(0.0)
}

/// Abramowitz–Stegun 7.1.26 error function (|ε| < 1.5e-7), so EI needs no
/// statistics dependency.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

/// The `Gp` surrogate as a [`Sampler`](super::sampler::Sampler).
pub struct GpSampler {
    cfg: GpConfig,
    /// Draw randomly until this many finite trials exist. A GP fitted to one
    /// or two points is a prior with decoration.
    pub n_startup: usize,
    /// Candidate configs scored per `ask`.
    pub n_candidates: usize,
    /// `true` = larger objective is better.
    pub maximize: bool,
    rng: StdRng,
}

impl GpSampler {
    pub fn new(cfg: GpConfig, seed: u64) -> Self {
        Self {
            cfg,
            n_startup: 5,
            n_candidates: 64,
            maximize: true,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl super::sampler::Sampler for GpSampler {
    fn ask(&mut self, space: &SearchSpace, completed: &[TrialResult]) -> Overlay {
        let finite = completed.iter().filter(|t| t.objective.is_finite()).count();
        if finite < self.n_startup {
            return space.sample(&mut self.rng);
        }
        let Some(model) = GpModel::fit(space, completed, self.cfg.clone()) else {
            return space.sample(&mut self.rng);
        };
        let best = completed
            .iter()
            .map(|t| t.objective)
            .filter(|o| o.is_finite())
            .fold(
                if self.maximize {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                },
                |a, b| {
                    if self.maximize { a.max(b) } else { a.min(b) }
                },
            );

        let mut chosen: Option<(f64, Overlay)> = None;
        for _ in 0..self.n_candidates.max(1) {
            let cand = space.sample(&mut self.rng);
            let (mu, sd) = model.predict(&cand);
            let ei = expected_improvement(mu, sd, best, 0.0, self.maximize);
            if ei.is_finite() && chosen.as_ref().is_none_or(|(b, _)| ei > *b) {
                chosen = Some((ei, cand));
            }
        }
        chosen
            .map(|(_, c)| c)
            .unwrap_or_else(|| space.sample(&mut self.rng))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hpo::sampler::Sampler;
    use serde_json::{Value, json};
    use std::collections::BTreeMap;

    fn space_of(dims: Vec<(&str, Dist)>) -> SearchSpace {
        let mut m = BTreeMap::new();
        for (k, v) in dims {
            m.insert(k.to_string(), v);
        }
        SearchSpace { dims: m }
    }

    fn tr(pairs: Vec<(&str, Value)>, objective: f64) -> TrialResult {
        TrialResult {
            overlay: pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
            objective,
        }
    }

    fn unit() -> SearchSpace {
        space_of(vec![(
            "x",
            Dist::Uniform {
                low: 0.0,
                high: 1.0,
            },
        )])
    }

    fn at(x: f64) -> Overlay {
        vec![("x".to_string(), json!(x))]
    }

    // ── kernel + linear algebra ─────────────────────────────────────────
    #[test]
    fn matern_is_maximal_at_zero_and_decays_monotonically() {
        let c = GpConfig::default();
        let k0 = matern52(0.0, &c);
        assert!((k0 - c.signal_variance).abs() < 1e-12, "k(0) = variance");
        let mut prev = k0;
        for step in 1..20 {
            let k = matern52(step as f64 * 0.1, &c);
            assert!(k < prev, "not monotone at {step}: {k} !< {prev}");
            assert!(k >= 0.0);
            prev = k;
        }
    }

    #[test]
    fn cholesky_reconstructs_its_input() {
        // A known SPD matrix; L Lᵀ must return it.
        let a = vec![
            vec![4.0, 2.0, 0.6],
            vec![2.0, 5.0, 1.0],
            vec![0.6, 1.0, 3.0],
        ];
        let l = cholesky(&a, 0.0).expect("SPD input must factorise");
        for i in 0..3 {
            for j in 0..3 {
                let v: f64 = (0..3).map(|k| l[i][k] * l[j][k]).sum();
                assert!((v - a[i][j]).abs() < 1e-10, "({i},{j}) {v} != {}", a[i][j]);
            }
        }
    }

    #[test]
    fn cholesky_refuses_a_non_positive_definite_matrix() {
        // Indefinite: eigenvalues straddle zero.
        let a = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        assert!(
            cholesky(&a, 0.0).is_none(),
            "must refuse, not return garbage"
        );
    }

    #[test]
    fn substitutions_invert_the_factor() {
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let l = cholesky(&a, 0.0).unwrap();
        let b = vec![1.0, 2.0];
        let x = back_sub(&l, &forward_sub(&l, &b));
        // A x must equal b.
        for i in 0..2 {
            let v: f64 = (0..2).map(|j| a[i][j] * x[j]).sum();
            assert!((v - b[i]).abs() < 1e-10, "row {i}: {v} != {}", b[i]);
        }
    }

    // ── GP behaviour ────────────────────────────────────────────────────
    #[test]
    fn gp_interpolates_its_observations() {
        let sp = unit();
        let obs = vec![
            tr(vec![("x", json!(0.1))], 1.0),
            tr(vec![("x", json!(0.5))], 5.0),
            tr(vec![("x", json!(0.9))], 2.0),
        ];
        let m = GpModel::fit(&sp, &obs, GpConfig::default()).unwrap();
        for (x, y) in [(0.1, 1.0), (0.5, 5.0), (0.9, 2.0)] {
            let (mu, sd) = m.predict(&at(x));
            assert!(
                (mu - y).abs() < 0.05,
                "at {x}: predicted {mu}, observed {y}"
            );
            assert!(
                sd < 0.2,
                "uncertainty at an observed point should be small: {sd}"
            );
        }
    }

    #[test]
    fn uncertainty_grows_away_from_the_data() {
        let sp = unit();
        // Everything observed at one end; the far end is unexplored.
        let obs = vec![
            tr(vec![("x", json!(0.02))], 1.0),
            tr(vec![("x", json!(0.05))], 1.1),
        ];
        let m = GpModel::fit(&sp, &obs, GpConfig::default()).unwrap();
        let (_, near) = m.predict(&at(0.03));
        let (_, far) = m.predict(&at(0.98));
        assert!(
            far > near,
            "sd must grow away from data: far {far} !> near {near}"
        );
    }

    #[test]
    fn gp_refuses_to_fit_nothing_and_drops_non_finite_targets() {
        let sp = unit();
        assert!(GpModel::fit(&sp, &[], GpConfig::default()).is_none());
        let all_bad = vec![
            tr(vec![("x", json!(0.2))], f64::NAN),
            tr(vec![("x", json!(0.4))], f64::INFINITY),
        ];
        assert!(
            GpModel::fit(&sp, &all_bad, GpConfig::default()).is_none(),
            "no finite target ⇒ no model, not a model of nothing"
        );
        let mixed = vec![
            tr(vec![("x", json!(0.2))], f64::NAN),
            tr(vec![("x", json!(0.4))], 3.0),
        ];
        let m = GpModel::fit(&sp, &mixed, GpConfig::default()).unwrap();
        assert_eq!(m.len(), 1, "only the finite trial is used");
    }

    #[test]
    fn duplicate_configs_do_not_break_the_factorisation() {
        // Re-proposing an evaluated config is normal (it is a cache hit), and
        // it makes two Gram rows identical — the jitter escalation must cope.
        let sp = unit();
        let obs = vec![
            tr(vec![("x", json!(0.3))], 2.0),
            tr(vec![("x", json!(0.3))], 2.0),
            tr(vec![("x", json!(0.3))], 2.0),
            tr(vec![("x", json!(0.7))], 4.0),
        ];
        let m = GpModel::fit(&sp, &obs, GpConfig::default()).expect("must still fit");
        let (mu, sd) = m.predict(&at(0.3));
        assert!(mu.is_finite() && sd.is_finite(), "mu {mu} sd {sd}");
        assert!((mu - 2.0).abs() < 0.2, "predicted {mu}");
    }

    #[test]
    fn a_constant_objective_is_predicted_without_dividing_by_zero() {
        let sp = unit();
        let obs = vec![
            tr(vec![("x", json!(0.2))], 7.0),
            tr(vec![("x", json!(0.8))], 7.0),
        ];
        let m = GpModel::fit(&sp, &obs, GpConfig::default()).unwrap();
        let (mu, sd) = m.predict(&at(0.5));
        assert!(
            (mu - 7.0).abs() < 1e-6,
            "constant target ⇒ constant model: {mu}"
        );
        assert!(sd.is_finite() && sd >= 0.0);
    }

    #[test]
    fn predictions_are_finite_across_a_mixed_space() {
        let sp = space_of(vec![
            (
                "lr",
                Dist::LogUniform {
                    low: 1e-5,
                    high: 1e-1,
                },
            ),
            ("k", Dist::IntUniform { low: 1, high: 8 }),
            (
                "mode",
                Dist::Choice {
                    choices: vec![json!("a"), json!("b")],
                },
            ),
        ]);
        let obs = vec![
            tr(
                vec![("lr", json!(1e-4)), ("k", json!(2)), ("mode", json!("a"))],
                1.0,
            ),
            tr(
                vec![("lr", json!(1e-2)), ("k", json!(6)), ("mode", json!("b"))],
                3.0,
            ),
        ];
        let m = GpModel::fit(&sp, &obs, GpConfig::default()).unwrap();
        for lr in [1e-5, 1e-3, 1e-1] {
            for k in [1, 4, 8] {
                for mode in ["a", "b"] {
                    let o = vec![
                        ("lr".to_string(), json!(lr)),
                        ("k".to_string(), json!(k)),
                        ("mode".to_string(), json!(mode)),
                    ];
                    let (mu, sd) = m.predict(&o);
                    assert!(
                        mu.is_finite() && sd.is_finite() && sd >= 0.0,
                        "{lr} {k} {mode}"
                    );
                }
            }
        }
    }

    // ── acquisition ─────────────────────────────────────────────────────
    #[test]
    fn expected_improvement_rewards_both_mean_and_uncertainty() {
        // Same mean, more uncertainty ⇒ strictly more EI.
        let low = expected_improvement(1.0, 0.1, 1.0, 0.0, true);
        let high = expected_improvement(1.0, 1.0, 1.0, 0.0, true);
        assert!(high > low, "{high} !> {low}");
        // Same uncertainty, better mean ⇒ strictly more EI.
        let worse = expected_improvement(0.5, 0.5, 1.0, 0.0, true);
        let better = expected_improvement(1.5, 0.5, 1.0, 0.0, true);
        assert!(better > worse, "{better} !> {worse}");
        // Never negative.
        assert!(expected_improvement(-100.0, 0.3, 1.0, 0.0, true) >= 0.0);
    }

    #[test]
    fn expected_improvement_respects_minimize() {
        let ei_min = expected_improvement(0.2, 0.3, 1.0, 0.0, false);
        let ei_max = expected_improvement(0.2, 0.3, 1.0, 0.0, true);
        assert!(ei_min > ei_max, "below best is good when minimising");
    }

    #[test]
    fn erf_matches_known_values() {
        for (x, want) in [(0.0, 0.0), (0.5, 0.5205), (1.0, 0.8427), (2.0, 0.9953)] {
            assert!((erf(x) - want).abs() < 1e-3, "erf({x}) = {}", erf(x));
            assert!((erf(-x) + want).abs() < 1e-3, "erf is odd");
        }
    }

    #[test]
    fn ehvi_is_zero_for_a_candidate_certain_to_be_dominated() {
        // Minimise both. Front already holds (1,1); a candidate confidently at
        // (9,9) is dominated and cannot grow the volume.
        let sp = unit();
        let obs = vec![
            tr(vec![("x", json!(0.0))], 9.0),
            tr(vec![("x", json!(1.0))], 9.0),
        ];
        let cfg = GpConfig {
            noise_variance: 1e-10,
            ..Default::default()
        };
        let m0 = GpModel::fit(&sp, &obs, cfg.clone()).unwrap();
        let m1 = GpModel::fit(&sp, &obs, cfg).unwrap();
        let front = vec![ParetoPoint::new(1, vec![1.0, 1.0])];
        let dirs = [Direction::Minimize, Direction::Minimize];
        let mut rng = StdRng::seed_from_u64(4);
        let target = HypervolumeTarget {
            front: &front,
            dirs: &dirs,
            reference: [10.0, 10.0],
        };
        let v = ehvi_mc(&[m0, m1], &at(0.0), &target, 200, &mut rng);
        assert!(v < 0.5, "a dominated candidate should barely register: {v}");
    }

    #[test]
    fn ehvi_prefers_the_candidate_that_extends_the_front() {
        // Two objectives, both minimised. Observations make x≈0 predict LOW
        // values (good) and x≈1 predict HIGH values (bad).
        let sp = unit();
        let obs = vec![
            tr(vec![("x", json!(0.0))], 1.0),
            tr(vec![("x", json!(0.1))], 1.2),
            tr(vec![("x", json!(0.9))], 8.0),
            tr(vec![("x", json!(1.0))], 8.5),
        ];
        let cfg = GpConfig::default();
        let m0 = GpModel::fit(&sp, &obs, cfg.clone()).unwrap();
        let m1 = GpModel::fit(&sp, &obs, cfg).unwrap();
        let front = vec![ParetoPoint::new(1, vec![5.0, 5.0])];
        let dirs = [Direction::Minimize, Direction::Minimize];
        let target = HypervolumeTarget {
            front: &front,
            dirs: &dirs,
            reference: [10.0, 10.0],
        };
        let good = {
            let mut r = StdRng::seed_from_u64(11);
            ehvi_mc(&[m0.clone(), m1.clone()], &at(0.05), &target, 400, &mut r)
        };
        let bad = {
            let mut r = StdRng::seed_from_u64(11);
            ehvi_mc(&[m0, m1], &at(0.95), &target, 400, &mut r)
        };
        assert!(
            good > bad,
            "EHVI should favour the improving region: {good} !> {bad}"
        );
    }

    #[test]
    fn ehvi_refuses_arities_it_cannot_compute() {
        let sp = unit();
        let obs = vec![tr(vec![("x", json!(0.5))], 1.0)];
        let m = GpModel::fit(&sp, &obs, GpConfig::default()).unwrap();
        let front = vec![ParetoPoint::new(1, vec![1.0, 1.0])];
        let dirs = [Direction::Minimize, Direction::Minimize];
        let mut rng = StdRng::seed_from_u64(1);
        let target = HypervolumeTarget {
            front: &front,
            dirs: &dirs,
            reference: [9.0, 9.0],
        };
        // One model for two objectives — not computable, must be 0.0 not a guess.
        let one = std::slice::from_ref(&m);
        assert_eq!(ehvi_mc(one, &at(0.5), &target, 32, &mut rng), 0.0);
        // Zero samples likewise.
        assert_eq!(
            ehvi_mc(&[m.clone(), m], &at(0.5), &target, 0, &mut rng),
            0.0
        );
    }

    // ── sampler ─────────────────────────────────────────────────────────
    #[test]
    fn gp_sampler_draws_randomly_before_startup_then_uses_the_model() {
        let sp = unit();
        let mut s = GpSampler::new(GpConfig::default(), 3);
        s.n_startup = 4;
        let few = vec![tr(vec![("x", json!(0.5))], 1.0)];
        let o = s.ask(&sp, &few);
        assert!((0.0..=1.0).contains(&o[0].1.as_f64().unwrap()));

        // With enough data concentrated on a good region, the model should
        // steer there rather than sampling uniformly.
        let mut obs = Vec::new();
        for i in 0..12 {
            let x = 0.8 + (i as f64) * 0.005;
            obs.push(tr(vec![("x", json!(x))], 10.0));
        }
        for i in 0..12 {
            let x = 0.05 + (i as f64) * 0.005;
            obs.push(tr(vec![("x", json!(x))], 0.0));
        }
        let mut near_good = 0;
        for seed in 0..10u64 {
            let mut s = GpSampler::new(GpConfig::default(), seed);
            s.n_startup = 1;
            let o = s.ask(&sp, &obs);
            if o[0].1.as_f64().unwrap() > 0.5 {
                near_good += 1;
            }
        }
        assert!(
            near_good >= 7,
            "should concentrate on the high region; got {near_good}/10"
        );
    }

    #[test]
    fn gp_sampler_is_seed_reproducible() {
        let sp = unit();
        let obs: Vec<TrialResult> = (0..8)
            .map(|i| tr(vec![("x", json!(i as f64 / 8.0))], i as f64))
            .collect();
        let a = GpSampler::new(GpConfig::default(), 42).ask(&sp, &obs);
        let b = GpSampler::new(GpConfig::default(), 42).ask(&sp, &obs);
        assert_eq!(a, b);
    }
}
