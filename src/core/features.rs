//! (private) `assemble`: lay the embedding and the two sender-history blocks out
//! into one fixed-order feature vector.
//!
//! Layout (392 dims, all ≈ [0,1] by construction — see `docs/architecture.md` →
//! *The feature vector*):
//!
//! ```text
//! [ embedding            (384, unit-norm from MiniLM)
//! | domain:  p1,p2,p3 proportions (sum to 1), confidence
//! | address: p1,p2,p3 proportions (sum to 1), confidence ]
//! ```
//!
//! The order is a stable contract: it is baked into every serialized `Model`, so
//! a change here is a `feature_version` bump (guarded at load — see
//! `shell/persist.rs`). The golden test below pins it.

/// Dimensions of the MiniLM embedding block.
pub const EMBED_DIM: usize = 384;

/// Dimensions of one sender-history block: 3 smoothed proportions + 1
/// confidence scalar. Applies to both the domain and the address block.
pub const HIST_DIM: usize = 4;

/// Total assembled feature length: embedding + domain block + address block.
pub const FEATURE_DIM: usize = EMBED_DIM + HIST_DIM + HIST_DIM; // 392

/// One sender-history block, ready to drop into the feature vector: the three
/// Dirichlet-smoothed class proportions followed by the confidence scalar. The
/// core builds these from [`super::history`] outputs; the type just keeps the
/// two pieces together across the `assemble` boundary.
#[derive(Debug, Clone, Copy)]
pub struct HistBlock {
    /// Smoothed class proportions `[p1, p2, p3]`, summing to 1.
    pub proportions: [f32; 3],
    /// [0,1] confidence in this sender's history.
    pub confidence: f32,
}

impl HistBlock {
    /// The block for a sender with no history: exactly the prior, zero
    /// confidence. `hist_block` uses this for the `ZERO_COUNTS` case (an absent
    /// sender or a count query that fell back).
    pub fn from_prior(prior: &[f32; 3]) -> HistBlock {
        HistBlock {
            proportions: *prior,
            confidence: 0.0,
        }
    }
}

/// Assemble the fixed-order 392-dim feature vector from the embedding and the
/// two history blocks. The output length is always [`FEATURE_DIM`]; the caller
/// (the core's `features_for`/`classify`) guarantees `embedding` is exactly
/// [`EMBED_DIM`] long.
pub fn assemble(embedding: &[f32], domain_hist: &HistBlock, addr_hist: &HistBlock) -> Vec<f32> {
    debug_assert_eq!(
        embedding.len(),
        EMBED_DIM,
        "embedding must be exactly {EMBED_DIM} dims"
    );
    let mut out = Vec::with_capacity(FEATURE_DIM);
    out.extend_from_slice(embedding);
    push_hist(&mut out, domain_hist);
    push_hist(&mut out, addr_hist);
    out
}

/// Append one history block in canonical order: the three proportions, then the
/// confidence scalar.
fn push_hist(out: &mut Vec<f32>, hist: &HistBlock) {
    out.extend_from_slice(&hist.proportions);
    out.push(hist.confidence);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(p: [f32; 3], c: f32) -> HistBlock {
        HistBlock {
            proportions: p,
            confidence: c,
        }
    }

    #[test]
    fn length_is_exactly_392() {
        let embedding = vec![0.0f32; EMBED_DIM];
        let d = block([0.2, 0.5, 0.3], 0.4);
        let a = block([0.1, 0.1, 0.8], 0.9);
        assert_eq!(assemble(&embedding, &d, &a).len(), 392);
        assert_eq!(FEATURE_DIM, 392);
    }

    #[test]
    fn order_is_stable_golden() {
        // A hand-built input whose every slot is distinguishable, so any
        // reordering of the layout would move an observable value.
        let mut embedding = vec![0.0f32; EMBED_DIM];
        embedding[0] = 1.0; // first embedding slot
        embedding[EMBED_DIM - 1] = 2.0; // last embedding slot
        let d = block([0.11, 0.12, 0.13], 0.14);
        let a = block([0.21, 0.22, 0.23], 0.24);

        let f = assemble(&embedding, &d, &a);

        // Embedding occupies [0, 384).
        assert_eq!(f[0], 1.0);
        assert_eq!(f[EMBED_DIM - 1], 2.0);
        // Domain block: proportions then confidence, at [384, 388).
        assert_eq!(&f[EMBED_DIM..EMBED_DIM + 4], &[0.11, 0.12, 0.13, 0.14]);
        // Address block: proportions then confidence, at [388, 392).
        assert_eq!(&f[EMBED_DIM + 4..FEATURE_DIM], &[0.21, 0.22, 0.23, 0.24]);
    }

    #[test]
    fn from_prior_is_prior_with_zero_confidence() {
        let prior = [0.6, 0.3, 0.1];
        let b = HistBlock::from_prior(&prior);
        assert_eq!(b.proportions, prior);
        assert_eq!(b.confidence, 0.0);
    }
}
