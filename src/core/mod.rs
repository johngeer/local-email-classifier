//! THE core interface (pure functions only).
//!
//! ENFORCED BOUNDARY: this module and everything under `core/` must have **no**
//! `use` of `notmuch`, `fastembed`, or `std::fs`. The core is a pure function of
//! data the shell has already gathered. The dependency is one-directional:
//! `shell` depends on `core`, never the reverse.
//!
//! Checklist status: §1 leaves (`labels`, `domain`, `history`, `text`) and the
//! §2 `features`/`model` math are in place; the two public entry points
//! (`classify`, `features_for`) compose them below.

pub mod labels;

// §1–§2 leaves. Private (`mod`, not `pub mod`): they exist for isolation and
// testing, not as public surface. `history`/`model` types are re-exported below
// since they are the seam types the shell fills and persists.
mod domain;
mod features;
mod history;
mod model;
mod text;

pub use domain::registrable_domain;
pub use history::{ClassCounts, ZERO as ZERO_COUNTS};
pub use labels::Priority;
pub use model::{Model, FEATURE_VERSION};

/// The seam type the shell hands to the core: one parsed email, already read off
/// disk. Defined here because the core defines what it consumes; the shell's
/// `mailfile` adapter produces it.
#[derive(Debug, Clone)]
pub struct RawEmail {
    /// Raw From header value (may include a display name).
    pub from: String,
    /// Subject header, or empty if absent.
    pub subject: String,
    /// Best-available body text (text/plain part preferred by the shell).
    pub body: String,
    /// Arrival timestamp, unix seconds. Used by the shell for time-based
    /// scoping and splits; the core does not depend on it for v1 features.
    pub ts: i64,
}

/// Prepare the single text string handed to the embedder: subject-first, body
/// cleaned and truncated. The embedder itself is a shell concern (it is IO and a
/// heavy model), so the core hands *back* the text for the shell to embed rather
/// than embedding here. Both entry points then take the resulting embedding.
pub fn prepared_text(email: &RawEmail) -> String {
    text::prepare_text(&email.subject, &email.body)
}

/// Classify one email into a [`Priority`]. The whole inference pipeline behind
/// one call: `proportions/confidence → assemble → predict_proba → argmax`.
///
/// `embedding` is the [`features::EMBED_DIM`]-long unit vector the shell
/// produced from [`prepared_text`]; the embedder is IO and stays in the shell,
/// so the core takes the result rather than running it.
pub fn classify(
    model: &Model,
    embedding: &[f32],
    domain_counts: &ClassCounts,
    addr_counts: &ClassCounts,
) -> Priority {
    let x = features_for(model, embedding, domain_counts, addr_counts);
    model::predict(model, &x)
}

/// Build the 392-dim feature vector for one email — `classify`'s pure first
/// half, exposed on its own so `train` can assemble a feature matrix without
/// scoring. Uses the model's stored `class_prior`/`alpha` so training and
/// inference smooth history identically.
pub fn features_for(
    model: &Model,
    embedding: &[f32],
    domain_counts: &ClassCounts,
    addr_counts: &ClassCounts,
) -> Vec<f32> {
    let domain_hist = hist_block(domain_counts, &model.class_prior, model.alpha);
    let addr_hist = hist_block(addr_counts, &model.class_prior, model.alpha);
    features::assemble(embedding, &domain_hist, &addr_hist)
}

/// Turn one sender's raw counts into the smoothed proportions + confidence block
/// the feature vector carries. At `ZERO_COUNTS` this yields the prior with zero
/// confidence, matching [`features::HistBlock::from_prior`].
fn hist_block(counts: &ClassCounts, prior: &[f32; 3], alpha: f32) -> features::HistBlock {
    features::HistBlock {
        proportions: history::proportions(counts, prior, alpha),
        confidence: history::confidence(counts),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model that ignores the embedding (all-zero weights there) and reads the
    /// address block's three proportion slots as the class signal, so a sender's
    /// history fully determines the prediction. Lets us test `classify` end to
    /// end without a solver.
    fn addr_history_model() -> Model {
        use features::{EMBED_DIM, FEATURE_DIM, HIST_DIM};
        // Address proportions sit at [EMBED_DIM + HIST_DIM, EMBED_DIM + HIST_DIM + 3).
        let addr_p0 = EMBED_DIM + HIST_DIM;
        let mut weights = vec![vec![0.0f32; FEATURE_DIM]; 3];
        for c in 0..3 {
            weights[c][addr_p0 + c] = 20.0;
        }
        Model {
            feature_version: FEATURE_VERSION,
            embedding_model_id: "test".to_string(),
            class_prior: [1.0 / 3.0; 3],
            alpha: 1.0,
            weights,
            intercepts: [0.0; 3],
        }
    }

    fn zero_embedding() -> Vec<f32> {
        vec![0.0f32; features::EMBED_DIM]
    }

    #[test]
    fn features_for_has_full_length() {
        let m = addr_history_model();
        let f = features_for(&m, &zero_embedding(), &[1, 2, 3], &[4, 5, 6]);
        assert_eq!(f.len(), features::FEATURE_DIM);
    }

    #[test]
    fn classify_follows_the_dominant_sender_class() {
        let m = addr_history_model();
        let e = zero_embedding();
        // Address history overwhelmingly P3 → classify P3.
        assert_eq!(classify(&m, &e, &ZERO_COUNTS, &[0, 0, 200]), Priority::P3);
        // Overwhelmingly P1 → classify P1.
        assert_eq!(classify(&m, &e, &ZERO_COUNTS, &[200, 0, 0]), Priority::P1);
    }

    #[test]
    fn no_history_yields_the_prior_block() {
        // With zero counts the address block is exactly the (uniform) prior, so
        // the three driven slots tie and argmax falls to the lowest class.
        let m = addr_history_model();
        assert_eq!(
            classify(&m, &zero_embedding(), &ZERO_COUNTS, &ZERO_COUNTS),
            Priority::P1
        );
    }

    #[test]
    fn prepared_text_puts_subject_first() {
        let email = RawEmail {
            from: "a@b.com".to_string(),
            subject: "Urgent".to_string(),
            body: "hello there".to_string(),
            ts: 0,
        };
        assert!(prepared_text(&email).starts_with("Urgent"));
    }
}
