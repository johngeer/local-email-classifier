//! THE shell interface: [`train`] and [`classify_new`] only.
//!
//! All IO, caching, and the linfa solver live here. These two entry points
//! compose the §4 adapters below: they gather `RawEmail`, history counts, and an
//! embedding, hand them to the pure core, and persist / apply what comes back.
//! `main.rs` calls only these two functions.

// §3 persistence: JSON save/load + load-time guards. Private (`mod`): the entry
// points below are the shell's public surface.
mod persist;

// §4 adapters: all IO and stateful adapters, each private. `mailfile` parses a
// maildir file, `embed` fronts fastembed, `notmuch` runs the tag queries behind
// two HashMap caches, `fit` runs the L-BFGS solve. The §5 entry points below
// compose these; nothing here is public.
mod embed;
mod fit;
mod mailfile;
mod notmuch;

use std::path::Path;

use crate::core::{self, RawEmail};
use embed::{Embedder, FastEmbedder, EMBEDDING_MODEL_ID};

/// Classification cutoff: only mail arriving on or after this date is classified,
/// leaving the pre-cutoff backlog untouched (design → *Scope*). The cutoff gates
/// classification **only** — training and history counts see the full archive.
const CLASSIFY_CUTOFF: &str = "2026-07-01";

/// The class prior and Dirichlet smoothing `alpha` a freshly trained model is
/// built with. Both are stored on the `Model` so inference reproduces training's
/// history smoothing exactly. A uniform prior is the honest default until the
/// label distribution argues otherwise.
const CLASS_PRIOR: [f32; 3] = [1.0 / 3.0; 3];
const ALPHA: f32 = 1.0;

/// The post-new hook entry point: classify in-scope new mail and write a guess.
///
/// Selects mail on or after [`CLASSIFY_CUTOFF`] that is not already
/// human-confirmed (the skip-confirmed rule lives in the notmuch query), then for
/// each message: parse → embed the prepared text → gather memoized domain +
/// address history counts → `core::classify` → write `prio-*` + `auto`.
///
/// Per-message failures (an unreadable file, a missing message id, a failed tag
/// write) are logged and skipped — one bad message never aborts the batch. An
/// embedding-model failure *does* abort, per the failure policy: a garbage vector
/// would be silently misclassified.
pub fn classify_new(model_path: &Path) -> Result<(), String> {
    let embedder = FastEmbedder::new().map_err(|e| format!("loading embedder: {e}"))?;
    let model = persist::load(model_path, embedder.model_id())
        .map_err(|e| format!("loading model {}: {e}", model_path.display()))?;

    let files = notmuch::new_mail_files(CLASSIFY_CUTOFF)?;
    let mut nm = notmuch::Notmuch::new();

    let mut classified = 0usize;
    for path in &files {
        let (email, message_id) = match mailfile::parse_file_with_id(path) {
            Ok(parsed) => parsed,
            Err(e) => {
                eprintln!("skipping {}: parse failed: {e}", path.display());
                continue;
            }
        };
        let Some(message_id) = message_id else {
            eprintln!("skipping {}: no Message-ID header to tag by", path.display());
            continue;
        };

        // Embed the prepared text. A model failure here aborts the whole batch.
        let text = core::prepared_text(&email);
        let embedding = embedder
            .embed(&text)
            .map_err(|e| format!("embedding {}: {e}", path.display()))?;

        let (domain_counts, addr_counts) = sender_counts(&mut nm, &email);
        let priority = core::classify(&model, &embedding, &domain_counts, &addr_counts);

        if let Err(e) = notmuch::write_guess(&message_id, priority) {
            eprintln!("tagging {message_id} failed: {e}");
            continue;
        }
        classified += 1;
    }

    eprintln!("classified {classified}/{} in-scope message(s)", files.len());
    Ok(())
}

/// The training entry point: fit a fresh model over every confirmed label.
///
/// Selects the confirmed-label set (all dates, no cutoff — the full archive is
/// signal) → parse each → build the pure feature vector with `core::features_for`
/// (using the same prior/alpha inference will use) → `fit` the L-BFGS solve →
/// save the JSON model. Returns an error if there are no confirmed labels, the
/// embedder or notmuch fails, or the solve fails.
///
/// Per-message parse failures are logged and skipped; an embedding failure aborts
/// (a training example with a garbage vector would poison the fit).
pub fn train(model_path: &Path) -> Result<(), String> {
    let embedder = FastEmbedder::new().map_err(|e| format!("loading embedder: {e}"))?;

    let files = notmuch::confirmed_label_files()?;

    // Per-class training set sizes, in Priority index order. Logged up front so a
    // run makes plain how much signal (and how skewed) the confirmed-label set is.
    let mut label_counts = [0usize; 3];
    for (_, label) in &files {
        label_counts[label.to_index()] += 1;
    }
    eprintln!(
        "confirmed labels: {} total  ({} prio-low, {} prio-normal, {} prio-high)",
        files.len(),
        label_counts[core::Priority::P1.to_index()],
        label_counts[core::Priority::P2.to_index()],
        label_counts[core::Priority::P3.to_index()],
    );

    let mut nm = notmuch::Notmuch::new();

    // A model carrying just the smoothing params, so `features_for` builds the
    // history blocks the same way training and inference both must.
    let smoothing = core::Model {
        feature_version: core::FEATURE_VERSION,
        embedding_model_id: EMBEDDING_MODEL_ID.to_string(),
        class_prior: CLASS_PRIOR,
        alpha: ALPHA,
        weights: vec![vec![0.0; core::FEATURE_DIM]; 3],
        intercepts: [0.0; 3],
    };

    eprintln!("embedding {} message(s)…", files.len());
    let mut examples = Vec::new();
    for (i, (path, label)) in files.iter().enumerate() {
        if i > 0 && i % 100 == 0 {
            eprintln!("  embedded {i}/{}…", files.len());
        }
        let email = match mailfile::parse_file(path) {
            Ok(email) => email,
            Err(e) => {
                eprintln!("skipping {}: parse failed: {e}", path.display());
                continue;
            }
        };

        let text = core::prepared_text(&email);
        let embedding = embedder
            .embed(&text)
            .map_err(|e| format!("embedding {}: {e}", path.display()))?;

        let (domain_counts, addr_counts) = sender_counts(&mut nm, &email);
        let features = core::features_for(&smoothing, &embedding, &domain_counts, &addr_counts);
        examples.push(fit::Example {
            features,
            label: *label,
        });
    }

    if examples.is_empty() {
        return Err("no confirmed labels to train on".to_string());
    }

    eprintln!("fitting multinomial logistic regression on {} example(s)…", examples.len());
    let model = fit::fit(&examples, CLASS_PRIOR, ALPHA, EMBEDDING_MODEL_ID)?;
    persist::save(&model, model_path)
        .map_err(|e| format!("saving model {}: {e}", model_path.display()))?;
    eprintln!("trained on {} example(s), saved {}", examples.len(), model_path.display());
    Ok(())
}

/// Gather one email's memoized domain + address history counts. The address is
/// the sender's exact address; the domain is its registrable eTLD+1. A `From`
/// with no extractable address yields [`core::ZERO_COUNTS`] for both — the core
/// maps that to the prior with zero confidence.
fn sender_counts(nm: &mut notmuch::Notmuch, email: &RawEmail) -> (core::ClassCounts, core::ClassCounts) {
    let Some(addr) = extract_address(&email.from) else {
        return (core::ZERO_COUNTS, core::ZERO_COUNTS);
    };
    let addr_counts = nm.addr_counts(&addr);
    let domain_counts = match core::registrable_domain(&addr) {
        Some(domain) => nm.domain_counts(&domain),
        None => core::ZERO_COUNTS,
    };
    (domain_counts, addr_counts)
}

/// Pull the bare `local@domain` address out of a raw `From` header value, which
/// may be `Display Name <local@domain>` or a bare address. Returns `None` if no
/// `@` is present. Lowercased so history lookups are case-insensitive.
fn extract_address(from: &str) -> Option<String> {
    let inner = match (from.find('<'), from.find('>')) {
        (Some(lt), Some(gt)) if lt < gt => &from[lt + 1..gt],
        _ => from,
    };
    let addr = inner.trim();
    if addr.contains('@') && !addr.is_empty() {
        Some(addr.to_lowercase())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_address_from_display_name_form() {
        assert_eq!(
            extract_address("Alice Example <Alice@Example.com>").as_deref(),
            Some("alice@example.com")
        );
    }

    #[test]
    fn extracts_bare_address() {
        assert_eq!(
            extract_address("bob@example.org").as_deref(),
            Some("bob@example.org")
        );
    }

    #[test]
    fn no_at_sign_yields_none() {
        assert_eq!(extract_address("Mailer Daemon"), None);
        assert_eq!(extract_address(""), None);
        assert_eq!(extract_address("<>"), None);
    }
}
