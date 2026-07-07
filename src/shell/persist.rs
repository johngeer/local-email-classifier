//! (private) Model persistence: JSON save/load for the single `models/model.json`
//! file, plus the two load-time guards.
//!
//! The guards exist because a deserialized `Model` can be structurally valid yet
//! semantically wrong for the running code:
//! - `feature_version` skew means the 392 weights line up against a *different*
//!   feature layout — every weight is applied to the wrong feature.
//! - `embedding_model_id` skew means the 384 embedding dims came from a
//!   different embedder; the file still deserializes cleanly but the model now
//!   scores confident nonsense. This is the more dangerous skew (wrong answers,
//!   not a shape error), so it is checked explicitly rather than assumed.
//!
//! Both must pass or the load refuses — better to stop than to classify with a
//! silently mismatched model.

use std::fs;
use std::io;
use std::path::Path;

use crate::core::{Model, FEATURE_VERSION};

/// What went wrong loading a model. Split from a plain string so the caller can
/// tell an IO/parse failure from a guard rejection.
#[derive(Debug)]
pub enum LoadError {
    /// The file could not be read.
    Io(io::Error),
    /// The bytes were not valid model JSON.
    Parse(serde_json::Error),
    /// The file deserialized but is for a different feature layout.
    FeatureVersion { found: u32, expected: u32 },
    /// The file deserialized but was trained against a different embedder.
    EmbeddingModel { found: String, expected: String },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "reading model file: {e}"),
            LoadError::Parse(e) => write!(f, "parsing model JSON: {e}"),
            LoadError::FeatureVersion { found, expected } => write!(
                f,
                "model feature_version {found} does not match this build's {expected}; \
                 retrain the model"
            ),
            LoadError::EmbeddingModel { found, expected } => write!(
                f,
                "model embedding_model_id {found:?} does not match the embedder in use \
                 ({expected:?}); retrain the model"
            ),
        }
    }
}

impl std::error::Error for LoadError {}

/// Serialize a model to pretty JSON at `path`, creating parent dirs as needed.
/// Pretty rather than compact so the file is readable and diffable by eye, as
/// the design calls for.
pub fn save(model: &Model, path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let json = serde_json::to_string_pretty(model)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    fs::write(path, json)
}

/// Load and validate a model. Deserializes the JSON, then runs both load-time
/// guards against the current build's `FEATURE_VERSION` and the given
/// `embedding_model_id`. Returns a [`LoadError`] rather than panicking so the
/// caller can report a clear "retrain" message.
pub fn load(path: &Path, embedding_model_id: &str) -> Result<Model, LoadError> {
    let bytes = fs::read(path).map_err(LoadError::Io)?;
    let model: Model = serde_json::from_slice(&bytes).map_err(LoadError::Parse)?;
    check_guards(&model, embedding_model_id)?;
    Ok(model)
}

/// The two load-time guards. Factored out so it can be unit-tested without
/// touching the filesystem.
fn check_guards(model: &Model, embedding_model_id: &str) -> Result<(), LoadError> {
    if model.feature_version != FEATURE_VERSION {
        return Err(LoadError::FeatureVersion {
            found: model.feature_version,
            expected: FEATURE_VERSION,
        });
    }
    if model.embedding_model_id != embedding_model_id {
        return Err(LoadError::EmbeddingModel {
            found: model.embedding_model_id.clone(),
            expected: embedding_model_id.to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{classify, Priority, ZERO_COUNTS};

    const FEATURE_DIM: usize = 392;
    const EMBED_ID: &str = "all-MiniLM-L6-v2";

    /// A small but non-trivial model: distinct weights and intercepts so a
    /// round-trip that dropped or reordered anything would change predictions.
    fn sample_model() -> Model {
        let mut weights = vec![vec![0.0f32; FEATURE_DIM]; 3];
        for c in 0..3 {
            weights[c][FEATURE_DIM - 3 + c] = 5.0 + c as f32;
        }
        Model {
            feature_version: FEATURE_VERSION,
            embedding_model_id: EMBED_ID.to_string(),
            class_prior: [0.5, 0.3, 0.2],
            alpha: 2.0,
            weights,
            intercepts: [0.1, -0.2, 0.3],
        }
    }

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("email_classifier_persist_test_{name}.json"));
        p
    }

    fn zero_embedding() -> Vec<f32> {
        vec![0.0f32; 384]
    }

    #[test]
    fn round_trip_preserves_predictions() {
        let path = tmp_path("round_trip");
        let model = sample_model();
        save(&model, &path).unwrap();
        let loaded = load(&path, EMBED_ID).unwrap();

        // Identical predictions on a handful of inputs is the property that
        // actually matters — not byte-equality of the struct.
        let e = zero_embedding();
        for addr in [ZERO_COUNTS, [10, 0, 0], [0, 0, 10]] {
            assert_eq!(
                classify(&model, &e, &ZERO_COUNTS, &addr),
                classify(&loaded, &e, &ZERO_COUNTS, &addr),
            );
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn round_trip_preserves_scalar_fields() {
        let path = tmp_path("scalars");
        let model = sample_model();
        save(&model, &path).unwrap();
        let loaded = load(&path, EMBED_ID).unwrap();
        assert_eq!(loaded.alpha, model.alpha);
        assert_eq!(loaded.class_prior, model.class_prior);
        assert_eq!(loaded.intercepts, model.intercepts);
        assert_eq!(loaded.feature_version, model.feature_version);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn guard_rejects_mismatched_embedding_model() {
        let mut model = sample_model();
        model.embedding_model_id = "some-other-embedder".to_string();
        match check_guards(&model, EMBED_ID) {
            Err(LoadError::EmbeddingModel { found, expected }) => {
                assert_eq!(found, "some-other-embedder");
                assert_eq!(expected, EMBED_ID);
            }
            other => panic!("expected EmbeddingModel error, got {other:?}"),
        }
    }

    #[test]
    fn guard_rejects_mismatched_feature_version() {
        let mut model = sample_model();
        model.feature_version = FEATURE_VERSION + 1;
        match check_guards(&model, EMBED_ID) {
            Err(LoadError::FeatureVersion { found, expected }) => {
                assert_eq!(found, FEATURE_VERSION + 1);
                assert_eq!(expected, FEATURE_VERSION);
            }
            other => panic!("expected FeatureVersion error, got {other:?}"),
        }
    }

    #[test]
    fn guards_pass_for_a_matching_model() {
        assert!(check_guards(&sample_model(), EMBED_ID).is_ok());
    }

    // A predictable failure path: guarantee `load` surfaces a missing file as
    // `Io`, not a panic.
    #[test]
    fn load_missing_file_is_io_error() {
        let path = tmp_path("does_not_exist_ever");
        let _ = fs::remove_file(&path);
        match load(&path, EMBED_ID) {
            Err(LoadError::Io(_)) => {}
            other => panic!("expected Io error, got {other:?}"),
        }
    }
}
