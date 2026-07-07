//! The trained model plus the two pure scoring functions.
//!
//! `Model` is a plain data record: the softmax weight matrix, per-class
//! intercepts, the class prior and smoothing `alpha` the history features were
//! built with, and the two version tags the shell's load-time guards check
//! (`feature_version`, `embedding_model_id`). Fitting it is effectful and lives
//! in `shell/fit.rs`; scoring it — [`predict_proba`] and [`predict`] — is a pure
//! function of the assembled feature vector and stays here.

use serde::{Deserialize, Serialize};

use super::features::FEATURE_DIM;
use super::labels::Priority;

/// Bumped whenever the feature layout in [`super::features::assemble`] changes.
/// A serialized `Model` carries the version it was trained under; the shell
/// refuses to load a model whose `feature_version` differs from this (a silent
/// feature-order skew between train and inference is otherwise invisible).
pub const FEATURE_VERSION: u32 = 1;

/// A trained multinomial logistic-regression model over the 392-dim feature
/// vector. Serialized to a single `models/model.json` (see `shell/persist.rs`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    /// Feature layout version this model was trained under. Checked at load.
    pub feature_version: u32,
    /// Embedding model the 384 embedding dims came from. Checked at load — a
    /// mismatch makes those dims silently meaningless while the file still
    /// deserializes cleanly, so it is the more dangerous skew.
    pub embedding_model_id: String,
    /// Class prior `[p1, p2, p3]`, summing to 1. Used by the core to build the
    /// history blocks for senders with no (or partial) history; stored so
    /// inference reproduces exactly the smoothing training used.
    pub class_prior: [f32; 3],
    /// Dirichlet smoothing strength for the history proportions. Stored for the
    /// same reason as `class_prior`.
    pub alpha: f32,
    /// Softmax weight matrix, one row of [`FEATURE_DIM`] weights per class,
    /// indexed by [`Priority::to_index`]: `weights[class][feature]`. Rows are
    /// `Vec<f32>` (not `[f32; 392]`) because serde cannot derive `Deserialize`
    /// for arrays longer than 32; every row is still exactly `FEATURE_DIM` long
    /// by construction (the solver and the tests build them that way).
    pub weights: Vec<Vec<f32>>,
    /// Per-class intercepts (bias), indexed by [`Priority::to_index`].
    pub intercepts: [f32; 3],
}

/// Class-probability distribution over the three priorities for one feature
/// vector: softmax of the per-class linear scores `weights·x + intercept`.
///
/// The output is indexed by [`Priority::to_index`] and sums to 1. `x` must be
/// exactly [`FEATURE_DIM`] long (the core's `features_for`/`classify` guarantee
/// this by assembling `x` themselves).
pub fn predict_proba(model: &Model, x: &[f32]) -> [f32; 3] {
    debug_assert_eq!(x.len(), FEATURE_DIM, "feature vector must be {FEATURE_DIM} dims");

    // Per-class linear scores.
    let mut scores = [0.0f32; 3];
    for c in 0..3 {
        let mut s = model.intercepts[c];
        let w = &model.weights[c];
        for j in 0..FEATURE_DIM {
            s += w[j] * x[j];
        }
        scores[c] = s;
    }

    softmax(scores)
}

/// The argmax priority for one feature vector. Ties (identical scores) resolve
/// to the lower class index, which is the more conservative priority.
pub fn predict(model: &Model, x: &[f32]) -> Priority {
    let proba = predict_proba(model, x);
    let mut best = 0usize;
    for c in 1..3 {
        if proba[c] > proba[best] {
            best = c;
        }
    }
    // `best` is always in 0..3, so `from_index` cannot fail.
    Priority::from_index(best).expect("argmax index is a valid class")
}

/// Numerically stable softmax over the three class scores. Shifts by the max
/// before exponentiating so large scores do not overflow.
fn softmax(scores: [f32; 3]) -> [f32; 3] {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps = [0.0f32; 3];
    let mut sum = 0.0f32;
    for c in 0..3 {
        let e = (scores[c] - max).exp();
        exps[c] = e;
        sum += e;
    }
    for c in 0..3 {
        exps[c] /= sum;
    }
    exps
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model whose weights read the last three feature slots as a one-hot
    /// class signal: feature `FEATURE_DIM-3+c` drives class `c`. Everything else
    /// is zero, so predictions are fully controlled by those three slots.
    fn onehot_model() -> Model {
        let mut weights = vec![vec![0.0f32; FEATURE_DIM]; 3];
        for c in 0..3 {
            weights[c][FEATURE_DIM - 3 + c] = 10.0;
        }
        Model {
            feature_version: FEATURE_VERSION,
            embedding_model_id: "test".to_string(),
            class_prior: [1.0 / 3.0; 3],
            alpha: 1.0,
            weights,
            intercepts: [0.0, 0.0, 0.0],
        }
    }

    fn onehot_x(class: usize) -> Vec<f32> {
        let mut x = vec![0.0f32; FEATURE_DIM];
        x[FEATURE_DIM - 3 + class] = 1.0;
        x
    }

    fn approx_sum_one(p: &[f32; 3]) {
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "probs sum to {s}, not 1");
    }

    #[test]
    fn proba_sums_to_one() {
        let m = onehot_model();
        for c in 0..3 {
            approx_sum_one(&predict_proba(&m, &onehot_x(c)));
        }
        // Also for an all-zero feature vector (uniform expected).
        approx_sum_one(&predict_proba(&m, &vec![0.0f32; FEATURE_DIM]));
    }

    #[test]
    fn all_zero_features_are_uniform() {
        let m = onehot_model();
        let p = predict_proba(&m, &vec![0.0f32; FEATURE_DIM]);
        for c in 0..3 {
            assert!((p[c] - 1.0 / 3.0).abs() < 1e-5, "class {c} = {}", p[c]);
        }
    }

    #[test]
    fn argmax_picks_the_driven_class() {
        let m = onehot_model();
        assert_eq!(predict(&m, &onehot_x(0)), Priority::P1);
        assert_eq!(predict(&m, &onehot_x(1)), Priority::P2);
        assert_eq!(predict(&m, &onehot_x(2)), Priority::P3);
    }

    #[test]
    fn intercepts_break_a_tie() {
        // Zero features: scores are just the intercepts. Bias class P3.
        let mut m = onehot_model();
        m.intercepts = [0.0, 0.0, 1.0];
        assert_eq!(predict(&m, &vec![0.0f32; FEATURE_DIM]), Priority::P3);
    }
}
