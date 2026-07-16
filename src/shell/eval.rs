//! (private) The time-held-out evaluation harness — a sanity check, not part of
//! the production path (design → *Evaluation*, checklist §6).
//!
//! Splits the confirmed-label set by date at a cutoff, trains on everything
//! *before* it, and scores predictions against the confirmed labels *after* it.
//! The split is by time, not random, to mimic deployment: fit on the past, guess
//! the future.
//!
//! **Honest history scope.** A test email's per-sender history features are built
//! from counts restricted to the train window (`date:..cutoff`), so no test email
//! ever sees a post-cutoff label — not even its own. This is the deployment-
//! faithful choice: at classify time the history is whatever existed then. It is
//! stricter than the shipped `train`/`classify` path (which counts the full
//! archive and accepts the leak, design → *Training-time leak note*), so these
//! numbers read pessimistically relative to production, not optimistically.
//!
//! Output is a 3×3 confusion matrix plus overall accuracy, per-class recall (p3
//! recall matters most), and an adjacent-vs-distant error split (a p1↔p3 miss is
//! worse than a p2↔p3 one).

use crate::core::{self, Priority};
use crate::shell::embed::{Embedder, FastEmbedder, CachingEmbedder, EMBEDDING_MODEL_ID};
use crate::shell::{fit, mailfile, notmuch, sender_counts, ALPHA, CLASS_PRIOR};

/// Run the time-held-out evaluation at `cutoff` (a notmuch date, e.g.
/// `2026-07-01`). Trains on confirmed labels dated `..cutoff`, scores confirmed
/// labels dated `cutoff..`. History counts on both sides are bounded to
/// `..cutoff` (see the module docs). Prints the confusion matrix and metrics.
pub fn evaluate(cutoff: &str) -> Result<(), String> {
    let before = format!("..{cutoff}");
    let after = format!("{cutoff}..");

    let raw_embedder = FastEmbedder::new().map_err(|e| format!("loading embedder: {e}"))?;
    let sanitized_id = crate::shell::embed::sanitize_model_id(raw_embedder.model_id());
    let cache_path = std::path::Path::new("cache").join(format!("embeddings-{sanitized_id}.redb"));
    let embedder = CachingEmbedder::open(raw_embedder, &cache_path);

    // History counts are bounded to the train window on BOTH sides, so a test
    // email's sender features are as-of-cutoff — never leaking its own future
    // label. One bounded adapter, shared by train and test feature assembly.
    let mut nm = notmuch::Notmuch::with_date_bound(&before);

    // --- Train side: labels before the cutoff. ---
    let train_files = notmuch::confirmed_label_files_dated(Some(&before))?;
    log_split("train", &train_files);
    let examples = build_examples(&embedder, &mut nm, &train_files)?;
    if examples.is_empty() {
        return Err("no pre-cutoff labels to train on".to_string());
    }
    log!("fitting on {} pre-cutoff example(s)…", examples.len());
    let model = fit::fit(&examples, CLASS_PRIOR, ALPHA, EMBEDDING_MODEL_ID)?;

    // --- Test side: labels on/after the cutoff, scored against the model. ---
    let test_files = notmuch::confirmed_label_files_dated(Some(&after))?;
    log_split("test", &test_files);
    if test_files.is_empty() {
        return Err("no post-cutoff labels to evaluate against".to_string());
    }

    // confusion[true][pred] over Priority index order (P1, P2, P3).
    let mut confusion = [[0u32; 3]; 3];
    let mut scored = 0usize;
    for (path, truth) in &test_files {
        let email = match mailfile::parse_file(path) {
            Ok(email) => email,
            Err(e) => {
                log!("skipping {}: parse failed: {e}", path.display());
                continue;
            }
        };
        let text = core::prepared_text(&email);
        let embedding = embedder
            .embed(&text)
            .map_err(|e| format!("embedding {}: {e}", path.display()))?;
        let (domain_counts, addr_counts) = sender_counts(&mut nm, &email);
        let pred = core::classify(&model, &embedding, &domain_counts, &addr_counts);
        confusion[truth.to_index()][pred.to_index()] += 1;
        scored += 1;
    }

    embedder.flush();
    log!(
        "embeddings: {} from cache, {} regenerated ({} total)",
        embedder.hits(),
        embedder.misses(),
        embedder.hits() + embedder.misses()
    );

    report(cutoff, &confusion, scored);
    Ok(())
}

/// Parse → embed → assemble features for a labeled file set, exactly as `train`
/// does (same prior/alpha smoothing model), reusing the shared bounded `nm` for
/// history counts. Per-message parse failures are logged and skipped; an
/// embedding failure aborts.
fn build_examples(
    embedder: &CachingEmbedder<FastEmbedder>,
    nm: &mut notmuch::Notmuch,
    files: &[(std::path::PathBuf, Priority)],
) -> Result<Vec<fit::Example>, String> {
    let smoothing = core::Model {
        feature_version: core::FEATURE_VERSION,
        embedding_model_id: EMBEDDING_MODEL_ID.to_string(),
        class_prior: CLASS_PRIOR,
        alpha: ALPHA,
        weights: vec![vec![0.0; core::FEATURE_DIM]; 3],
        intercepts: [0.0; 3],
    };
    let mut examples = Vec::new();
    for (i, (path, label)) in files.iter().enumerate() {
        if i > 0 && i % 100 == 0 {
            log!("  embedded {i}/{}…", files.len());
        }
        let email = match mailfile::parse_file(path) {
            Ok(email) => email,
            Err(e) => {
                log!("skipping {}: parse failed: {e}", path.display());
                continue;
            }
        };
        let text = core::prepared_text(&email);
        let embedding = embedder
            .embed(&text)
            .map_err(|e| format!("embedding {}: {e}", path.display()))?;
        let (domain_counts, addr_counts) = sender_counts(nm, &email);
        let features = core::features_for(&smoothing, &embedding, &domain_counts, &addr_counts);
        examples.push(fit::Example {
            features,
            label: *label,
        });
    }
    Ok(examples)
}

/// Log a split's per-class size in Priority order, so the balance of each side is
/// on the record next to the metrics.
fn log_split(name: &str, files: &[(std::path::PathBuf, Priority)]) {
    let mut counts = [0usize; 3];
    for (_, label) in files {
        counts[label.to_index()] += 1;
    }
    log!(
        "{name}: {} total  ({} prio-low, {} prio-normal, {} prio-high)",
        files.len(),
        counts[Priority::P1.to_index()],
        counts[Priority::P2.to_index()],
        counts[Priority::P3.to_index()],
    );
}

/// Print the confusion matrix and the metrics from design → *Evaluation*:
/// overall accuracy, per-class recall, and the adjacent-vs-distant error split.
fn report(cutoff: &str, confusion: &[[u32; 3]; 3], scored: usize) {
    let labels = ["prio-low ", "prio-norm", "prio-high"];

    println!();
    println!("time-held-out evaluation, cutoff {cutoff}  ({scored} test message(s))");
    println!("rows = true label, cols = predicted\n");
    println!("            {}   {}   {}", labels[0], labels[1], labels[2]);
    for t in 0..3 {
        print!("{}  ", labels[t]);
        for p in 0..3 {
            print!("{:>11}", confusion[t][p]);
        }
        let row_total: u32 = confusion[t].iter().sum();
        let recall = if row_total == 0 {
            f32::NAN
        } else {
            confusion[t][t] as f32 / row_total as f32
        };
        println!("   | recall {:>5.1}%", recall * 100.0);
    }

    let correct: u32 = (0..3).map(|i| confusion[i][i]).sum();
    let total: u32 = confusion.iter().flatten().sum();
    let acc = if total == 0 {
        f32::NAN
    } else {
        correct as f32 / total as f32
    };

    // Distant errors are p1↔p3 confusions (index distance 2); adjacent are the
    // p1↔p2 / p2↔p3 ones (distance 1). The diagonal is correct, not an error.
    let mut adjacent = 0u32;
    let mut distant = 0u32;
    for t in 0..3 {
        for p in 0..3 {
            match (t as i32 - p as i32).unsigned_abs() {
                1 => adjacent += confusion[t][p],
                2 => distant += confusion[t][p],
                _ => {}
            }
        }
    }

    println!();
    println!("overall accuracy : {:.1}%  ({correct}/{total})", acc * 100.0);
    println!("adjacent errors  : {adjacent}  (p1↔p2, p2↔p3)");
    println!("distant  errors  : {distant}  (p1↔p3 — the costly ones)");
}
