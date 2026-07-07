//! (private) Sender history: the `ClassCounts` seam type plus the two pure
//! reductions the features are built from — Dirichlet-smoothed `proportions`
//! and a `confidence` scalar. Both are ≈ [0,1] by construction, so no
//! normalization step is needed downstream.

/// Per-class observation counts for one sender (domain or address), indexed by
/// `Priority::to_index()`: `[p1, p2, p3]`. The shell fills these from notmuch
/// tag-count queries; the core only ever sees the plain array.
pub type ClassCounts = [u32; 3];

/// The "no history" counts. The shell falls back to this when a count query
/// fails; [`proportions`] then returns exactly `prior` and [`confidence`] is 0.
pub const ZERO: ClassCounts = [0, 0, 0];

/// Dirichlet-smoothed class proportions, summing to 1.
///
/// With counts `n_i`, prior `π_i` (itself summing to 1), and smoothing strength
/// `alpha`, the smoothed proportion is
///
/// ```text
///   (n_i + alpha * π_i) / (N + alpha)      where N = Σ n_i
/// ```
///
/// At `N = 0` this returns the prior exactly; as `N → ∞` it approaches the
/// empirical ratio `n_i / N`. `alpha` controls how fast that transition happens.
pub fn proportions(counts: &ClassCounts, prior: &[f32; 3], alpha: f32) -> [f32; 3] {
    let total: u32 = counts.iter().sum();
    let denom = total as f32 + alpha;
    let mut out = [0.0f32; 3];
    for i in 0..3 {
        out[i] = (counts[i] as f32 + alpha * prior[i]) / denom;
    }
    out
}

/// A [0,1] confidence in a sender's history, `min(1, ln(1 + N) / ln(1000))`,
/// where `N` is the total number of observations. 0 at `N = 0`, monotonically
/// increasing, and capped at 1 (reached at `N = 999`).
pub fn confidence(counts: &ClassCounts) -> f32 {
    let total: u32 = counts.iter().sum();
    let raw = (1.0 + total as f32).ln() / 1000.0_f32.ln();
    raw.min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNIFORM: [f32; 3] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-5, "expected {b}, got {a}");
    }

    /// Looser tolerance for "approaches X" claims, where smoothing deliberately
    /// leaves a small residual that never fully vanishes at finite counts/alpha.
    fn near(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-3, "expected ~{b}, got {a}");
    }

    #[test]
    fn proportions_sum_to_one() {
        for counts in [[0, 0, 0], [5, 2, 1], [1000, 0, 3], [0, 0, 7]] {
            let p = proportions(&counts, &UNIFORM, 1.0);
            approx(p.iter().sum::<f32>(), 1.0);
        }
    }

    #[test]
    fn zero_counts_return_prior_exactly() {
        let prior = [0.2, 0.5, 0.3];
        let p = proportions(&ZERO, &prior, 1.0);
        approx(p[0], prior[0]);
        approx(p[1], prior[1]);
        approx(p[2], prior[2]);
    }

    #[test]
    fn shrinks_toward_prior_at_low_counts() {
        // One observation in class p3 with a skewed prior: the smoothed
        // estimate should sit between the prior and the empirical [0,0,1].
        let prior = [0.6, 0.3, 0.1];
        let p = proportions(&[0, 0, 1], &prior, 4.0);
        // p3 pulled up from its prior of 0.1, but nowhere near 1.
        assert!(p[2] > prior[2] && p[2] < 0.5, "p3 = {}", p[2]);
        // p1 pulled down from its prior but still the largest share.
        assert!(p[0] < prior[0] && p[0] > p[1] && p[0] > p[2]);
    }

    #[test]
    fn approaches_empirical_ratio_at_high_counts() {
        // With huge counts, alpha's influence vanishes.
        let p = proportions(&[900, 100, 0], &UNIFORM, 1.0);
        near(p[0], 0.9);
        near(p[1], 0.1);
        assert!(p[2] < 1e-3);
    }

    #[test]
    fn alpha_zero_is_pure_empirical() {
        let p = proportions(&[3, 1, 0], &UNIFORM, 0.0);
        approx(p[0], 0.75);
        approx(p[1], 0.25);
        approx(p[2], 0.0);
    }

    #[test]
    fn large_alpha_dominated_by_prior() {
        let prior = [0.6, 0.3, 0.1];
        let p = proportions(&[1, 1, 1], &prior, 10_000.0);
        // With a huge alpha the tiny counts barely move the prior.
        near(p[0], prior[0]);
        near(p[1], prior[1]);
        near(p[2], prior[2]);
    }

    #[test]
    fn confidence_zero_at_no_history() {
        approx(confidence(&ZERO), 0.0);
    }

    #[test]
    fn confidence_is_monotone() {
        let mut prev = -1.0;
        for n in [0u32, 1, 5, 50, 500, 999] {
            let c = confidence(&[n, 0, 0]);
            assert!(c >= prev, "confidence decreased at n={n}");
            prev = c;
        }
    }

    #[test]
    fn confidence_capped_at_one() {
        approx(confidence(&[999, 0, 0]), 1.0);
        // Beyond the cap it stays at 1, not above.
        approx(confidence(&[10_000, 5_000, 5_000]), 1.0);
    }
}
