//! (private) The embedding adapter: text → a 384-dim unit vector.
//!
//! A small [`Embedder`] trait fronts the concrete `fastembed` backend so the
//! embedder is a swappable seam (the design's model2vec escape hatch drops in
//! here with the same 1-vector-per-text contract). The shell hands
//! [`crate::core::prepared_text`]'s output in and gets the `[f32; 384]` back to
//! pass to the core — the core never touches the model.
//!
//! Failure policy (design → *Failure & cold-start*): an embedding-model failure
//! *aborts*. A zero/garbage embedding would be silently misclassified, which is
//! worse than stopping — so every fallible call here propagates its error rather
//! than substituting a default vector.

use std::error::Error;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use crate::core::EMBED_DIM;

/// The embedding model id the whole pipeline is keyed on. Written into every
/// serialized `Model` and checked by the persistence load-time guard — a
/// mismatch means the 384 embedding dims are meaningless (see `shell/persist.rs`).
pub const EMBEDDING_MODEL_ID: &str = "all-MiniLM-L6-v2";

/// A swappable text embedder: one unit vector per text. The trait is the seam
/// the design calls for — later backends (model2vec) implement the same contract.
pub trait Embedder {
    /// The model id this embedder produces vectors for. Must match the id stored
    /// in a `Model` for the persistence guard to accept it.
    fn model_id(&self) -> &str;

    /// Embed one prepared text into a [`EMBED_DIM`]-long unit vector. Returns an
    /// error on any model failure — the caller aborts rather than proceeding with
    /// a garbage vector.
    fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn Error>>;
}

/// The `fastembed` backend: all-MiniLM-L6-v2 (384-dim, unit-norm, ONNX). The
/// weights are auto-downloaded to `fastembed`'s own cache dir on first use — they
/// are a separate write-once artifact and never live in `models/`.
pub struct FastEmbedder {
    model: TextEmbedding,
}

impl FastEmbedder {
    /// Load the embedding model, downloading the ONNX weights on first use.
    /// Returns an error if the model cannot be initialized.
    pub fn new() -> Result<FastEmbedder, Box<dyn Error>> {
        let model = TextEmbedding::try_new(InitOptions::new(EmbeddingModel::AllMiniLML6V2))?;
        Ok(FastEmbedder { model })
    }
}

impl Embedder for FastEmbedder {
    fn model_id(&self) -> &str {
        EMBEDDING_MODEL_ID
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn Error>> {
        // One text in, one vector out. `embed` batches; we hand it a singleton.
        let mut out = self.model.embed(vec![text], None)?;
        let vector = out
            .pop()
            .ok_or("embedder returned no vector for one input")?;
        if vector.len() != EMBED_DIM {
            return Err(format!(
                "embedder returned {} dims, expected {EMBED_DIM}",
                vector.len()
            )
            .into());
        }
        Ok(vector)
    }
}
