//! (private) The solver: fit a multinomial logistic regression over assembled
//! feature vectors via linfa's L-BFGS, and pack the result into the core's
//! [`Model`].
//!
//! This is the one place the L-BFGS solver lives (design → *Solver invocation*):
//! deterministic given its inputs, but heavy compute returning `Result`, so it
//! stays in the shell. It takes feature vectors the core already assembled
//! (`features_for`) plus their labels, and returns a `Model` ready to persist.
//!
//! **Weight layout translation.** linfa's fitted `params()` is
//! `(n_features, n_classes)` and its columns follow linfa's *sorted* class
//! labels, which need not be all three priorities (a training set missing a class
//! yields fewer columns). The core's `Model.weights` is `[class][feature]` over
//! all three priorities. We transpose and scatter linfa's columns into the right
//! `Priority` rows by consulting `classes()`, zero-filling any priority absent
//! from the training data. Getting this mapping wrong would silently apply each
//! class's weights to the wrong class, so it is done explicitly here.

use linfa::prelude::*;
use linfa_logistic::MultiLogisticRegression;
use ndarray::{Array1, Array2};

use crate::core::{Model, Priority, FEATURE_DIM, FEATURE_VERSION};

/// L-BFGS iteration cap. Generous — the problem is small (392 features) and
/// convex, so it converges well within this; the cap only bounds a pathological
/// non-converging run.
const MAX_ITERATIONS: u64 = 1000;

/// One labeled training example: an assembled feature vector and its confirmed
/// priority. The caller builds `features` with `core::features_for` so training
/// and inference smooth history identically.
pub struct Example {
    pub features: Vec<f32>,
    pub label: Priority,
}

/// Fit a model over the given examples and pack it into a [`Model`].
///
/// `class_prior` and `alpha` are stored on the returned model (they are the
/// history-smoothing parameters the features were built with, not solver
/// hyperparameters); `embedding_model_id` is stored for the load-time guard.
/// Returns an error if there are no examples, the feature vectors are the wrong
/// length, or the solver fails.
pub fn fit(
    examples: &[Example],
    class_prior: [f32; 3],
    alpha: f32,
    embedding_model_id: &str,
) -> Result<Model, String> {
    if examples.is_empty() {
        return Err("no training examples".to_string());
    }
    let n_features = examples[0].features.len();
    if n_features != FEATURE_DIM {
        return Err(format!(
            "feature vectors are {n_features} dims, expected {FEATURE_DIM}"
        ));
    }

    let dataset = build_dataset(examples, n_features)?;

    let fitted = MultiLogisticRegression::default()
        .max_iterations(MAX_ITERATIONS)
        .with_intercept(true)
        .fit(&dataset)
        .map_err(|e| format!("logistic regression fit failed: {e}"))?;

    let (weights, intercepts) = pack_params(fitted.params(), fitted.intercept(), fitted.classes());

    Ok(Model {
        feature_version: FEATURE_VERSION,
        embedding_model_id: embedding_model_id.to_string(),
        class_prior,
        alpha,
        weights,
        intercepts,
    })
}

/// Assemble the examples into the `(records, targets)` dataset linfa fits. Labels
/// are the priority class indices (`0..3`); linfa sorts them, so the fitted
/// `classes()` reports which column is which — see [`pack_params`].
fn build_dataset(
    examples: &[Example],
    n_features: usize,
) -> Result<Dataset<f64, usize, ndarray::Ix1>, String> {
    let n = examples.len();
    let mut records = Array2::<f64>::zeros((n, n_features));
    let mut targets = Array1::<usize>::zeros(n);
    for (i, ex) in examples.iter().enumerate() {
        if ex.features.len() != n_features {
            return Err(format!(
                "example {i} has {} features, expected {n_features}",
                ex.features.len()
            ));
        }
        for (j, &v) in ex.features.iter().enumerate() {
            records[[i, j]] = v as f64;
        }
        targets[i] = ex.label.to_index();
    }
    Ok(Dataset::new(records, targets))
}

/// Translate linfa's fitted params into the core's `[class][feature]` weight
/// matrix and per-class intercepts, scattering by `classes()` so each column
/// lands in its true [`Priority`] row. Priorities absent from the training set
/// (not in `classes`) stay all-zero.
///
/// `params` is `(n_features, n_classes)`; `intercept` is `(n_classes)`; both are
/// column-indexed by `classes` (linfa's sorted label list, each an index in
/// `0..3`).
fn pack_params(
    params: &Array2<f64>,
    intercept: &Array1<f64>,
    classes: &[usize],
) -> (Vec<Vec<f32>>, [f32; 3]) {
    let n_features = params.nrows();
    let mut weights = vec![vec![0.0f32; n_features]; 3];
    let mut intercepts = [0.0f32; 3];
    for (col, &class_idx) in classes.iter().enumerate() {
        // class_idx is a Priority index (0..3) since that is what we labeled with.
        for f in 0..n_features {
            weights[class_idx][f] = params[[f, col]] as f32;
        }
        intercepts[class_idx] = intercept[col] as f32;
    }
    (weights, intercepts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::predict;

    /// Build one example whose last three feature slots one-hot the class, the
    /// rest zero — a trivially separable set for the solver to recover.
    fn onehot_example(label: Priority) -> Example {
        let mut features = vec![0.0f32; FEATURE_DIM];
        features[FEATURE_DIM - 3 + label.to_index()] = 1.0;
        Example { features, label }
    }

    #[test]
    fn separates_a_tiny_synthetic_set() {
        // A few copies of each class so the solver has something to fit.
        let mut examples = Vec::new();
        for _ in 0..5 {
            for p in Priority::ALL {
                examples.push(onehot_example(p));
            }
        }
        let model = fit(&examples, [1.0 / 3.0; 3], 1.0, "test").unwrap();

        // ~100% train accuracy on the separable set.
        for p in Priority::ALL {
            let x = onehot_example(p).features;
            assert_eq!(predict(&model, &x), p, "misclassified {p:?}");
        }
    }

    #[test]
    fn empty_examples_is_an_error() {
        assert!(fit(&[], [1.0 / 3.0; 3], 1.0, "test").is_err());
    }

    #[test]
    fn wrong_length_features_is_an_error() {
        let bad = Example {
            features: vec![0.0f32; 10],
            label: Priority::P1,
        };
        assert!(fit(&[bad], [1.0 / 3.0; 3], 1.0, "test").is_err());
    }
}
