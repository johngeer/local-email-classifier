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

use std::collections::HashMap;
use std::error::Error;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::atomic::AtomicUsize;
use std::sync::Mutex;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use redb::ReadableTableMetadata;

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

const EMBEDDINGS_TABLE: redb::TableDefinition<u64, &[u8]> = redb::TableDefinition::new("embeddings");

/// A caching decorator that wraps an inner `Embedder` and persistent redb store.
pub struct CachingEmbedder<E: Embedder> {
    inner: E,
    db: Option<redb::Database>,
    hits: AtomicUsize,
    misses: AtomicUsize,
    pending: Mutex<HashMap<u64, Vec<f32>>>,
}

impl<E: Embedder> CachingEmbedder<E> {
    /// Open a persistent cache at the given path, wrapping the inner embedder.
    /// Fallback to bypass mode if the cache fails to open.
    pub fn open(inner: E, path: &Path) -> Self {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    log!("warning: failed to create cache directory {}: {}; caching disabled", parent.display(), e);
                    return Self::bypass(inner);
                }
            }
        }

        let db = match redb::Database::create(path) {
            Ok(db) => db,
            Err(e) => {
                log!("warning: failed to open/create cache file {}: {}; caching disabled", path.display(), e);
                return Self::bypass(inner);
            }
        };

        // Read the current entry count, creating the table lazily only if it does
        // not exist yet. A brand-new store needs one write transaction to
        // materialize the table; an existing store answers from a read
        // transaction alone, so the common warm-open path — every `classify_new`
        // hook fire — pays no write/fsync just to open.
        let entry_count = match (|| -> Result<u64, Box<dyn Error>> {
            let read_txn = db.begin_read()?;
            match read_txn.open_table(EMBEDDINGS_TABLE) {
                Ok(table) => Ok(table.len()?),
                Err(redb::TableError::TableDoesNotExist(_)) => {
                    // Fresh store: create the table (and probe write access).
                    drop(read_txn);
                    let write_txn = db.begin_write()?;
                    {
                        let _table = write_txn.open_table(EMBEDDINGS_TABLE)?;
                    }
                    write_txn.commit()?;
                    Ok(0)
                }
                Err(e) => Err(e.into()),
            }
        })() {
            Ok(len) => len,
            Err(e) => {
                log!("warning: failed to open cache file {}: {}; caching disabled", path.display(), e);
                return Self::bypass(inner);
            }
        };

        log!(
            "opened cache/{} ({} entries)",
            path.file_name().unwrap_or_default().to_string_lossy(),
            entry_count
        );

        CachingEmbedder {
            inner,
            db: Some(db),
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
            pending: Mutex::new(HashMap::new()),
        }
    }

    /// Construct a CachingEmbedder in bypass mode (no caching).
    pub fn bypass(inner: E) -> Self {
        CachingEmbedder {
            inner,
            db: None,
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
            pending: Mutex::new(HashMap::new()),
        }
    }

    /// Get the hit tally.
    pub fn hits(&self) -> usize {
        self.hits.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get the miss tally.
    pub fn misses(&self) -> usize {
        self.misses.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Flush all buffered inserts to the database in a single transaction.
    /// Errors are logged and swallowed.
    pub fn flush(&self) {
        let Some(ref db) = self.db else {
            return;
        };

        let mut pending = match self.pending.lock() {
            Ok(guard) => guard,
            Err(e) => {
                log!("warning: failed to lock pending inserts for flush: {}", e);
                return;
            }
        };
        if pending.is_empty() {
            return;
        }

        let res = (|| -> Result<(), Box<dyn Error>> {
            let write_txn = db.begin_write()?;
            {
                let mut table = write_txn.open_table(EMBEDDINGS_TABLE)?;
                for (key, vector) in pending.drain() {
                    let mut bytes = Vec::with_capacity(vector.len() * 4);
                    for &f in &vector {
                        bytes.extend_from_slice(&f.to_le_bytes());
                    }
                    table.insert(key, bytes.as_slice())?;
                }
            }
            write_txn.commit()?;
            Ok(())
        })();

        if let Err(e) = res {
            log!("warning: failed to write/commit pending cache inserts: {}", e);
        }
    }

    fn read_cache(&self, db: &redb::Database, key: u64) -> Result<Option<Vec<f32>>, Box<dyn Error>> {
        let read_txn = db.begin_read()?;
        let table = read_txn.open_table(EMBEDDINGS_TABLE)?;
        let Some(guard) = table.get(key)? else {
            return Ok(None);
        };
        let bytes = guard.value();

        if bytes.len() != EMBED_DIM * 4 {
            return Ok(None); // Mismatch: treat as miss
        }

        let mut vector = Vec::with_capacity(EMBED_DIM);
        for chunk in bytes.chunks_exact(4) {
            let chunk_array: [u8; 4] = chunk.try_into().unwrap();
            vector.push(f32::from_le_bytes(chunk_array));
        }

        Ok(Some(vector))
    }
}

impl<E: Embedder> Embedder for CachingEmbedder<E> {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn Error>> {
        let Some(ref db) = self.db else {
            return self.inner.embed(text);
        };

        let key = hash_text(text);

        // Try pending buffer first (for same-run duplicate texts)
        if let Ok(pending) = self.pending.lock() {
            if let Some(vector) = pending.get(&key) {
                self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(vector.clone());
            }
        }

        // Try database cache
        match self.read_cache(db, key) {
            Ok(Some(vector)) => {
                self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(vector);
            }
            Ok(None) => {}
            Err(_) => {} // Treat read error as miss
        }

        // Real embed
        let vector = self.inner.embed(text)?;
        self.misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Buffer the insert
        if let Ok(mut pending) = self.pending.lock() {
            pending.insert(key, vector.clone());
        }

        Ok(vector)
    }
}

impl<E: Embedder> Drop for CachingEmbedder<E> {
    fn drop(&mut self) {
        self.flush();
    }
}

/// Injective sanitizer for model ids to be filesystem friendly.
pub fn sanitize_model_id(id: &str) -> String {
    let mut sanitized = String::new();
    for c in id.chars() {
        match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' | '%' => {
                sanitized.push_str(&format!("%{:02X}", c as u32));
            }
            _ => sanitized.push(c),
        }
    }
    sanitized
}

fn hash_text(text: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyEmbedder {
        call_count: AtomicUsize,
    }

    impl DummyEmbedder {
        fn new() -> Self {
            DummyEmbedder {
                call_count: AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.call_count.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    impl Embedder for DummyEmbedder {
        fn model_id(&self) -> &str {
            "dummy/model-v1"
        }

        fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn Error>> {
            self.call_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut vec = vec![0.0f32; EMBED_DIM];
            if !vec.is_empty() {
                vec[0] = text.len() as f32;
            }
            Ok(vec)
        }
    }

    fn temp_db_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("email_classifier_cache_test_{name}.redb"));
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn test_sanitize_model_id() {
        assert_eq!(sanitize_model_id("sentence-transformers/all-MiniLM-L6-v2"), "sentence-transformers%2Fall-MiniLM-L6-v2");
        assert_eq!(sanitize_model_id("a\\b:c*d?e\"f<g>h|i%j"), "a%5Cb%3Ac%2Ad%3Fe%22f%3Cg%3Eh%7Ci%25j");
    }

    #[test]
    fn test_cache_hit_and_miss() {
        let path = temp_db_path("hit_miss");
        let dummy = DummyEmbedder::new();
        let cached = CachingEmbedder::open(dummy, &path);

        assert_eq!(cached.hits(), 0);
        assert_eq!(cached.misses(), 0);

        // First embed: miss
        let v1 = cached.embed("hello").unwrap();
        assert_eq!(cached.hits(), 0);
        assert_eq!(cached.misses(), 1);
        assert_eq!(cached.inner.calls(), 1);
        assert_eq!(v1[0], 5.0);

        // Second embed: hit the buffered insert before flush
        let v2 = cached.embed("hello").unwrap();
        assert_eq!(cached.hits(), 1);
        assert_eq!(cached.misses(), 1);
        assert_eq!(cached.inner.calls(), 1);
        assert_eq!(v1, v2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_cache_hit_and_miss_with_flush() {
        let path = temp_db_path("hit_miss_flush");
        let dummy = DummyEmbedder::new();
        let cached = CachingEmbedder::open(dummy, &path);

        let v1 = cached.embed("hello").unwrap();
        assert_eq!(cached.inner.calls(), 1);

        cached.flush();

        let v2 = cached.embed("hello").unwrap();
        assert_eq!(cached.hits(), 1);
        assert_eq!(cached.misses(), 1);
        assert_eq!(cached.inner.calls(), 1);
        assert_eq!(v1, v2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_persistence_round_trip() {
        let path = temp_db_path("persistence");

        {
            let dummy = DummyEmbedder::new();
            let cached = CachingEmbedder::open(dummy, &path);
            let _ = cached.embed("hello").unwrap();
            cached.flush();
        }

        {
            let dummy = DummyEmbedder::new();
            let cached = CachingEmbedder::open(dummy, &path);
            let v = cached.embed("hello").unwrap();
            assert_eq!(cached.hits(), 1);
            assert_eq!(cached.misses(), 0);
            assert_eq!(cached.inner.calls(), 0);
            assert_eq!(v[0], 5.0);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_bypass_mode() {
        let dummy = DummyEmbedder::new();
        let cached = CachingEmbedder::bypass(dummy);

        assert!(cached.db.is_none());
        let _ = cached.embed("hello").unwrap();
        let _ = cached.embed("hello").unwrap();
        assert_eq!(cached.inner.calls(), 2);
        assert_eq!(cached.hits(), 0);
        assert_eq!(cached.misses(), 0);
    }

    #[test]
    fn test_corrupt_file_bypasses() {
        let path = temp_db_path("corrupt");
        std::fs::write(&path, b"not a redb database").unwrap();

        let dummy = DummyEmbedder::new();
        let cached = CachingEmbedder::open(dummy, &path);
        assert!(cached.db.is_none());

        let _ = cached.embed("hello").unwrap();
        assert_eq!(cached.inner.calls(), 1);
        assert_eq!(cached.hits(), 0);
        assert_eq!(cached.misses(), 0);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_wrong_length_entry_misses() {
        let path = temp_db_path("wrong_length");

        {
            let db = redb::Database::create(&path).unwrap();
            let write_txn = db.begin_write().unwrap();
            {
                let mut table = write_txn.open_table(EMBEDDINGS_TABLE).unwrap();
                let key = hash_text("hello");
                table.insert(key, vec![0u8; 10].as_slice()).unwrap();
            }
            write_txn.commit().unwrap();
        }

        let dummy = DummyEmbedder::new();
        let cached = CachingEmbedder::open(dummy, &path);

        let v = cached.embed("hello").unwrap();
        assert_eq!(cached.hits(), 0);
        assert_eq!(cached.misses(), 1);
        assert_eq!(cached.inner.calls(), 1);
        assert_eq!(v[0], 5.0);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_missing_file_opens_cold_some() {
        // A non-existent path yields an empty *created* store — `db` is `Some`
        // (cold), not `None` (bypass). A miss embedded now must warm the file:
        // after a drop/reopen it is a hit. This pins cold-`Some` vs bypass-`None`.
        let path = temp_db_path("missing_cold");
        assert!(!path.exists());

        {
            let cached = CachingEmbedder::open(DummyEmbedder::new(), &path);
            assert!(cached.db.is_some());
            let _ = cached.embed("hello").unwrap();
            assert_eq!(cached.misses(), 1);
            cached.flush();
        }

        {
            let cached = CachingEmbedder::open(DummyEmbedder::new(), &path);
            let _ = cached.embed("hello").unwrap();
            assert_eq!(cached.hits(), 1);
            assert_eq!(cached.inner.calls(), 0);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_distinct_texts_miss_independently() {
        let path = temp_db_path("distinct");
        let cached = CachingEmbedder::open(DummyEmbedder::new(), &path);

        let _ = cached.embed("alpha").unwrap();
        let _ = cached.embed("bravo").unwrap();
        assert_eq!(cached.misses(), 2);
        assert_eq!(cached.hits(), 0);
        assert_eq!(cached.inner.calls(), 2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_hit_miss_counter_mix() {
        // A mix of repeated and fresh texts: the tallies the effectiveness
        // summary is built from must match the expected split.
        let path = temp_db_path("counter_mix");
        let cached = CachingEmbedder::open(DummyEmbedder::new(), &path);

        for t in ["a", "b", "a", "c", "b", "a"] {
            let _ = cached.embed(t).unwrap();
        }
        // 3 distinct texts → 3 misses; the remaining 3 repeats → 3 hits.
        assert_eq!(cached.misses(), 3);
        assert_eq!(cached.hits(), 3);
        assert_eq!(cached.inner.calls(), 3);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_locked_file_bypasses() {
        // Holding the redb lock on a path, then opening a second CachingEmbedder
        // on it, must yield `db = None` (bypass), not an error or a hang. The
        // instance still serves vectors (all misses) and writes nothing.
        let path = temp_db_path("locked");
        let _holder = redb::Database::create(&path).unwrap();

        let cached = CachingEmbedder::open(DummyEmbedder::new(), &path);
        assert!(cached.db.is_none(), "second open on a locked file must bypass");

        let _ = cached.embed("hello").unwrap();
        let _ = cached.embed("hello").unwrap();
        assert_eq!(cached.inner.calls(), 2);
        assert_eq!(cached.hits(), 0);
        assert_eq!(cached.misses(), 0);

        drop(_holder);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_write_failure_is_swallowed() {
        // A flush against a store whose file was deleted/made-unwritable mid-run
        // must not surface from `embed` or `flush` — the vector is still returned
        // from the inner embedder (Failure policy: a cache write never aborts a run).
        let path = temp_db_path("write_fail");
        let cached = CachingEmbedder::open(DummyEmbedder::new(), &path);

        let v = cached.embed("hello").unwrap();
        assert_eq!(v[0], 5.0);

        // Corrupt the backing file out from under the store, then flush. The
        // commit fails internally; `flush` logs and swallows without panicking.
        std::fs::write(&path, b"corrupted out from under the store").unwrap();
        cached.flush(); // must not panic

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_slashed_model_id_round_trips_through_filename() {
        // An id containing `/` must create exactly one real file (no phantom
        // subdirectory, no failed-open → silent cold cache), and two ids that
        // differ only in a sanitized character must map to *distinct* files.
        let dir = std::env::temp_dir().join(format!(
            "email_classifier_slashid_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let id = "sentence-transformers/all-MiniLM-L6-v2";
        let file = format!("embeddings-{}.redb", sanitize_model_id(id));
        let path = dir.join(&file);
        assert!(!file.contains('/'), "sanitized filename must not contain a slash");

        {
            let cached = CachingEmbedder::open(DummyEmbedder::new(), &path);
            assert!(cached.db.is_some());
            let _ = cached.embed("hello").unwrap();
            cached.flush();
        }
        assert!(path.exists(), "exactly one real file should be created");

        // Injectivity: `a/b` and `a-b` must not collide onto one filename.
        assert_ne!(sanitize_model_id("a/b"), sanitize_model_id("a-b"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}

