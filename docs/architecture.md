# Architecture

A high-level map of how `email_classifier` is put together and where each
responsibility lives, and the living reference for the architecture: it points at
the code and states the invariants that hold across it, leaving fine detail to the
code itself. For the original rationale — the pre-implementation spec the design
was worked out against — see `docs/designs/done/design.md`.

## What the program does

Assign each incoming email one of three ordered priorities and apply it as a
notmuch tag:

| Priority | Tag           | Meaning         |
|----------|---------------|-----------------|
| `P1`     | `prio-low`    | least important |
| `P2`     | `prio-normal` | —               |
| `P3`     | `prio-high`   | most important  |

Classification is multinomial logistic regression (softmax, L-BFGS) over a
392-dim feature vector: a 384-dim text embedding plus two 4-dim sender-history
blocks. There is no daemon and no separate database — notmuch tags are the entire
user-facing API, and the model runs from notmuch's post-new hook.

## The two-module rule (functional core / imperative shell)

The load-bearing architectural rule. Everything is one of two kinds:

- **`src/core/`** — pure functions of already-gathered data. No `use` of
  `notmuch`, `fastembed`, or `std::fs`; the boundary is checkable by inspecting
  imports. Unit-tested in isolation.
- **`src/shell/`** — all IO, caching, and the linfa solver. Gathers inputs
  (`RawEmail`, history counts, embeddings), hands them to the core, and persists
  or applies what comes back.

The dependency is one-directional: **`shell` depends on `core`, never the
reverse.** The seam types the shell fills (`RawEmail`, `ClassCounts`) live in
`core` because the core defines what it consumes.

Both sides are **deep modules** — a small public surface over many private
leaves. A caller of the core sees ~2 functions and a couple of types; a caller of
the shell sees exactly `train`, `classify_new`, and `evaluate`. The leaf files
(`domain`, `history`, `text`, `features`, `mailfile`, `notmuch`, …) stay private
(`mod`, not `pub mod`); they exist for isolation and testing, not as public API.

Rule of thumb: if it touches notmuch, the filesystem, the embedding model, or the
L-BFGS solver, it is shell. Otherwise it is core.

## Module map

```
src/
├── main.rs         arg parse → dispatch train / classify / eval. Calls only the
│                   shell entry points; exits non-zero on error.
├── log.rs          log! macro — stderr lines prefixed with elapsed/delta clocks.
├── core/           THE core interface (pure); mod.rs re-exports ~2 fns + types.
│   ├── mod.rs        classify, features_for, prepared_text; RawEmail seam type.
│   ├── labels.rs     Priority enum ↔ usize ↔ tag string (the shared vocabulary).
│   ├── domain.rs     registrable_domain — psl-based eTLD+1.            (private)
│   ├── history.rs    ClassCounts, proportions (Dirichlet), confidence. (private)
│   ├── text.rs       prepare_text — subject-first, cleaned, truncated. (private)
│   ├── features.rs   assemble — the fixed 392-dim layout.              (private)
│   └── model.rs      Model record + pure predict_proba / predict.
└── shell/          THE shell interface: train / classify_new / evaluate only.
    ├── mod.rs        the entry points; sender_counts / extract_address helpers.
    ├── notmuch.rs    search + count adapter, two HashMap caches, the filter. (private)
    ├── embed.rs      Embedder trait + fastembed impl + CachingEmbedder. (private)
    ├── mailfile.rs   mail-parser → RawEmail.                           (private)
    ├── persist.rs    JSON save/load + load-time guards.                (private)
    ├── fit.rs        linfa L-BFGS solve → Model.                       (private)
    └── eval.rs       time-held-out confusion-matrix sanity check.      (private)
```

Only `labels` and `model` carry public types out of the core; the four leaves
(`domain`/`history`/`text`/`features`) are private, with `history`'s `ClassCounts`
re-exported by `core/mod.rs` because it is a seam type. Nothing under `shell/` is
public except the three entry points.

## The two interfaces

**Core** (`src/core/mod.rs`) — pure, takes the embedding as an argument so the
embedder stays in the shell:

- `prepared_text(&RawEmail) -> String` — the text to embed (hands the shell the
  string; does not embed).
- `classify(&Model, embedding, domain_counts, addr_counts) -> Priority` — hides
  the whole `proportions/confidence → assemble → predict_proba → argmax` pipeline.
- `features_for(&Model, embedding, domain_counts, addr_counts) -> Vec<f32>` —
  `classify`'s pure first half, exposed so `train` can build a feature matrix
  without scoring.

**Shell** (`src/shell/mod.rs`) — all IO behind three functions:

- `train(model_path)` — query confirmed labels (all dates) → parse → embed →
  `features_for` → `fit` → save JSON.
- `classify_new(model_path)` — the post-new hook path: select in-scope new mail →
  parse → embed → `classify` → write `prio-*` + `auto`.
- `evaluate(cutoff)` — a sanity check; trains before `cutoff`, scores on/after,
  writes no model and touches no tags.

## The feature vector (392 dims, all ≈ [0,1] by construction)

Assembled in a fixed order by `core/features.rs::assemble`. The order is a stable
contract baked into every serialized model; changing it is a `FEATURE_VERSION`
bump (guarded at load).

```
[ embedding            (384, unit-norm from MiniLM)
| domain:  p1,p2,p3 smoothed proportions (sum to 1), confidence   (4)
| address: p1,p2,p3 smoothed proportions (sum to 1), confidence   (4) ]
```

Features are scaled by construction (≈[0,1]), so there is no normalization step
and no stored stats. Each history block is three Dirichlet-smoothed class
proportions plus a `min(1, ln(1+N)/ln(1000))` confidence scalar — the address
block is sharp where the domain is not (a few key people on a big freemail
domain). See `core/history.rs` and README → *What the four history numbers mean*.

## Data flow

**Inference / `classify_new`** (║ marks the shell→core→shell seam):

```
shell │ notmuch: select in-scope new mail (date ≥ cutoff, unread, unconfirmed)
      │ mailfile: parse → RawEmail;  embed: prepared_text → [f32; 384]
      │ notmuch: memoized domain + address ClassCounts
      ╠══ classify(model, embedding, domain_counts, addr_counts) ══
core  │ proportions/confidence · assemble · predict_proba · argmax → Priority
      ╠══ return Priority ══
shell │ notmuch: write prio-* + auto by message id
```

Everything the core does is a pure function of the three inputs the shell
gathered; nothing inside the core reaches back out to notmuch or the model.

**Training** is the same seam without the argmax: select confirmed labels (all
dates) → parse → embed → `features_for` → collect into `fit::Example`s →
`fit::fit` (L-BFGS) → `persist::save`.

## The tag / labeling model

One visible priority tag; a flat `auto` marker tracks whether a human has stood
behind it. This is the whole feedback loop — no explicit training UI.

- The model writes `prio-*` **and** `auto` on a fresh guess. `auto` means
  *unconfirmed*.
- **Reading** a guess drops `unread` (via notmuch's `synchronize_flags`) — that
  counts as agreeing. **Correcting** it in the mail client re-sets `prio-*` and
  clears `auto`. Either way it becomes a *confirmed* label.
- A **confirmed label** is any `prio-*` message that is not an unreviewed guess:
  `not (tag:auto and tag:unread)`.

Corrections are picked up on the next run with no retrain, because history counts
already filter to confirmed labels; retrain the weights in batch occasionally.

The tag ↔ `Priority` mapping lives only in `core/labels.rs`; the confirmed-label
filter lives only in `shell/notmuch.rs::CONFIRMED_FILTER`. Neither string appears
anywhere else.

## Key invariants

These are easy to break accidentally; each lives in exactly one place.

- **Core purity.** No `use` of `notmuch`/`fastembed`/`std::fs` under `core/` —
  the "core is pure" claim is a checkable import property, asserted in
  `core/mod.rs`'s header.
- **The confirmed-label filter** (`not (tag:auto and tag:unread)`) is AND-ed into
  every history count and the training selection, so sender proportions are never
  built from the model's own unreviewed guesses. One constant,
  `notmuch.rs::CONFIRMED_FILTER`.
- **Classification scope.** Only mail on or after `CLASSIFY_CUTOFF`
  (`shell/mod.rs`) is classified; the cutoff gates classification *only* —
  training and history counts see the full archive.
- **Load-time guards** (`shell/persist.rs`). A model is refused unless both
  `feature_version` and `embedding_model_id` match the running code — the second
  is the dangerous skew, since a mismatched embedder makes the 384 dims silently
  meaningless while the JSON still deserializes cleanly.
- **Dedup by message, not file.** Training selection passes `--duplicate=1` so a
  message forwarded across accounts (indexed once per maildir copy) contributes
  one example, not one per copy. A training-set correctness fix, not a perf nicety.
- **Skip confirmed mail.** `classify_new` never overwrites a message already
  carrying `prio-*` without `auto`; the rule lives in its notmuch query, and
  guesses are written by message id, so a message with no Message-ID is skipped
  rather than mistagged.

## Failure & cold-start policy

- A notmuch **count** query that errors → treated as *unknown history*
  (`ZERO_COUNTS`, which the core maps to the prior with zero confidence); logged
  once, never aborts the batch.
- An **embedding-model** failure *does* abort — a garbage vector would be silently
  misclassified, which is worse than stopping.
- Per-message parse / tag-write failures in `classify_new` are logged and skipped;
  one bad message never aborts the run. The post-new hook itself is invoked
  non-fatally, so a classifier failure never blocks `notmuch new`.

## Persistence

One human-readable JSON file, `models/model.json` (gitignored, regenerable):
`feature_version`, `embedding_model_id`, `class_prior`, `alpha`, `weights`,
`intercepts`. Save/load and the load-time guards live in `shell/persist.rs`; the
file is never sharded. The MiniLM ONNX weights are a *separate* artifact that
`fastembed` auto-downloads to its own cache — never co-locate them in `models/`.

## Crates at the seams

| Concern            | Crate                       | Where            |
|--------------------|-----------------------------|------------------|
| Email parsing      | `mail-parser`               | `shell/mailfile` |
| Registrable domain | `psl`                       | `core/domain`    |
| Embeddings         | `fastembed` (all-MiniLM-L6-v2) | `shell/embed` |
| Embedding cache    | `redb` (embedded KV store)  | `shell/embed`    |
| Regression         | `linfa` + `linfa-logistic`  | `shell/fit`      |
| Arrays             | `ndarray`                   | `shell/fit`      |
| Tags / history     | `notmuch` CLI (shelled out) | `shell/notmuch`  |
| Persistence        | `serde` + `serde_json`      | `core`/`persist` |

The `Embedder` trait in `shell/embed.rs` is the swappable embedding seam: the
`CachingEmbedder` decorator (below) drops in behind it, and a later `model2vec`
backend drops in the same way — both with the same one-vector-per-email contract,
leaving the core untouched.

## The embedding cache

A persistent cache of text embeddings so any run — `train`, `evaluate`, or the
`classify_new` hook — reuses vectors an earlier run already computed instead of
re-embedding the same messages. `CachingEmbedder<E>` (`shell/embed.rs`) is a
decorator that *is* an `Embedder`, wrapping a concrete backend (`FastEmbedder`)
and an `Option<redb::Database>`. All three entry points hold a
`CachingEmbedder<FastEmbedder>` instead of a bare `FastEmbedder`; their embed
loops are unchanged, and the core never sees the cache.

- **Pure speed optimization.** A warm cache returns the same vector the embedder
  would have; correctness never depends on it. See
  `docs/designs/done/embedding-cache.md` for the full rationale.
- **Key + value.** Key is `hash(prepared_text)` (a `u64`); it self-invalidates if
  `core::prepare_text` changes. Value is the raw little-endian `f32` bytes of the
  vector (`EMBED_DIM × 4`), length-checked on read — a wrong-width entry is a miss,
  never a garbage vector.
- **The model-id guard is the filename.** The cache lives at
  `cache/embeddings-<sanitized-id>.redb`; a different embedder opens a different
  file, so a cross-model mismatch is structurally impossible to load. The id is
  passed through an *injective* filename sanitizer (percent-encoding path-hostile
  characters) so a HuggingFace-style `foo/bar` id cannot name a phantom
  subdirectory or collide with `foo-bar`. This mirrors `persist.rs`'s
  `embedding_model_id` guard.
- **Failure policy = one `Option`.** `db` is `Some` on a clean open-or-create and
  `None` on *any* open failure (corrupt, permission, disk, or **locked by a
  concurrent run** — the `classify_new` hook can overlap an mbsync cycle). `None`
  bypasses caching; a cache failure never aborts a run or blocks tagging. Writes
  are buffered and flushed under a single redb transaction per run.

`cache/` is a machine-local regenerable artifact — created on demand, gitignored,
and *not* under `models/` (which holds only `model.json`). Deleting it is the
whole maintenance story; the next run rebuilds what it needs.

## Planned extensions

Not yet in the code; captured so the direction is documented.

- **Deferred (Phase 3, `docs/designs/done/design.md`)** — chronological-replay training (removes the
  v1 training-time count leak), proportional-odds ordinal regression, batched
  embedding for `train`/`eval`, a Tranco-rank feature, and `confidence · P(pₙ)`
  history interaction terms.
```
