# Local Email Priority Classifier — Plan (Rust, simplified v1)

Classify emails into ordered priorities **p1 < p2 < p3** (p3 most important) using
multinomial logistic regression over a local embedding + domain **and address**
history features. Functional core / imperative shell; core unit tested.

Design principle: **features are scaled by construction** (everything ≈ [0,1]),
so there is no normalization step, no stored stats, and a single JSON model file.

## Tag interface (the whole API is notmuch tags)

How the classifier reaches the mail: notmuch is the API for *tags, queries, and
triggering*, but message **text** is read from the maildir files directly.
`notmuch search --output=files` hands back paths, which the shell then parses
with `mail-parser`. The classifier never writes, moves, or renames maildir files
— the one flag interaction (`unread` ↔ maildir `Seen`) is notmuch's own
`synchronize_flags`, not ours.

```d2
direction: down

hook: "notmuch post-new hook" {shape: hexagon}
cls: "email_classifier\n(train / classify)" {shape: rectangle}

nm: "notmuch" {
  tags: "tags + search/count\n(prio-*, auto, unread)" {shape: rectangle}
  index: "index (Xapian)" {shape: cylinder}
}

md: "maildir\n(message files on disk)" {shape: page}
mbsync: "mbsync" {shape: rectangle}

hook -> cls: "fires each `notmuch new`"

cls -> nm.tags: "write prio-* / auto (tag)\nquery labels + history counts"
cls -> md: "read files & parse\n(mail-parser)" {style.stroke-dash: 3}

nm.tags <-> nm.index
nm.index -> md: "indexes" {style.stroke-dash: 3}
mbsync -> md: "delivers new mail"

nm.tags -> md: "--output=files\n(returns paths)" {style.stroke-dash: 3}
```

The dashed edges are file reads / path resolution; the solid `cls -> nm.tags`
edge is the whole write side of the API (tagging). Note the classifier has two
arrows out — one to notmuch for tags/queries, one straight to the maildir for
text — which is the coupling discussed under *Load labeled messages* below.

See `notes_email_setup.txt` → *Priority Labels* / *Priority Classifier* for how
these tags are applied by hand in aerc and where the classifier hooks in.

**One visible priority tag; an `auto` marker tracks whether a human has stood
behind it.** There is a single `prio-*` namespace so the existing aerc folders
(`tag:prio-high` etc.) just work no matter how the label got there. A flat `auto`
tag rides alongside to distinguish an unconfirmed model guess from a
human-confirmed label.

- **Priority tags:** `prio-low`, `prio-normal`, `prio-high`, mutually exclusive.
  Map to the priorities used throughout this doc: `prio-low = P1`,
  `prio-normal = P2`, `prio-high = P3` (higher = more important). This mapping
  lives in `labels.rs` as the single source of truth — the tag strings appear
  nowhere else in the core.
- **`auto` marker:** the model writes `prio-*` **and** `auto` on a fresh guess.
  `auto` means *unconfirmed* — no human has agreed with this label yet. Any human
  touch removes it (see below). It is a flat tag, not per-level, because its only
  job is the confirmed/unconfirmed bit.
- **How `auto` clears (this is the whole labeling model):**
  - *Correcting* a guess — pressing `p1/p2/p3` in aerc — re-sets the priority and
    **removes `auto`**. The binding must add `-auto`; the notes say each binding
    already sets one prio tag and clears the other two, so this is one more token.
  - *Agreeing* with a guess is a **no-op gesture**: you just read the mail.
    Reading sets the maildir Seen flag → notmuch drops `unread` (relies on
    `maildir.synchronize_flags = true`, already configured). So a guess that is
    still `auto` **and** `unread` = never reviewed; `auto` but **not** `unread` =
    read-and-let-stand = agreed.
  - Fresh hand-labeling of never-seen mail: `p1/p2/p3` sets `prio-*`, and there is
    no `auto` to clear — same clean result.
- **What "agreed" really means:** *read*, not explicitly endorsed. A mail you
  open to skim, mean to re-file, and don't, counts as agreed. Accepted tradeoff —
  it keeps agreeing free, and reading priority mail is the signal we have. Stated
  here so the training set is understood as *read-approved*, not hand-verified.
- **Trigger:** the classifier runs from notmuch's **post-new hook** on each
  `notmuch new` (every mbsync cycle) — a plain subprocess, no daemon. It writes
  `prio-*` + `auto` on in-scope new mail only (see Scope below), and never
  overwrites a message that already carries `prio-*` without `auto` (a
  human-confirmed label is never re-guessed).
- **Code location:** `email/email_classifier/` in the dotfiles repo, a cargo
  project with `train` / `classify` subcommands; the single serialized model lives
  at `models/model.json` (gitignored — machine-local and regenerable). See
  *Directory layout* below.

**Scope: the cutoff gates classification only.** The `2026-07-01` cutoff applies
to *one* of the three things the system does — leave the other two on the full
archive:
- **Classify** — cutoff applies. Only messages with `ts >= 2026-07-01` get a
  priority assigned; the backlog is left untouched. Shell-level filter
  (`notmuch search date:2026-07-01..`), applied before parsing, so the core never
  sees an out-of-scope email.
- **Train** — no cutoff. Fit on every **confirmed** label regardless of date:
  hand labels, corrections, and read-and-agreed guesses. Query:
  `(tag:prio-low or tag:prio-normal or tag:prio-high) and not (tag:auto and tag:unread)`.
  That excludes only unreviewed model guesses (`auto and unread`), which would be
  circular to train on. Old confirmed mail is exactly the signal we want.
- **History counts** — no cutoff, and **confirmed labels only** (same
  `not (tag:auto and tag:unread)` filter). Sender proportions must not be built
  from the model's own unreviewed guesses. Older mail is signal about a sender
  even if we never classify it.

## Crates

| Concern | Crate | Notes |
|---|---|---|
| Email parsing | `mail-parser` | From, subject, body (prefer text/plain part) |
| Registrable domain | `psl` | eTLD+1 extraction |
| Embeddings | `fastembed` | all-MiniLM-L6-v2 (384-dim, unit-norm, ONNX, auto-download) |
| Regression | `linfa` + `linfa-logistic` | `MultiLogisticRegression` (softmax, L-BFGS) |
| Arrays | `ndarray` | feature matrices |
| History source | `notmuch` (CLI or crate) | labels = tags; counts via tag queries |
| Persistence | `serde` + `serde_json` | one human-readable model file |

Embedding swap path: small `Embedder` trait; later drop in `model2vec-rs`
(potion-base-8M) if MiniLM is too slow. Same 1-vector-per-email contract.

## Features (392 dims, all ≈ [0,1] by construction)

```
[ embedding (384, unit-norm from MiniLM)
| domain:  p1, p2, p3 smoothed proportions (sum to 1),  ln(1+n)/ln(1000) capped at 1
| address: p1, p2, p3 smoothed proportions (sum to 1),  ln(1+n)/ln(1000) capped at 1 ]
```

Address history is kept alongside domain history deliberately: priority mail is
often a few specific people on big freemail domains (gmail.com), where domain
proportions are uninformative but address proportions are sharp. The smoothed
address feature degrades gracefully to the prior for rarely-seen addresses.

Cut from v1 (revisit only if the confusion matrix demands it): Tranco rank,
is_subdomain, z-score normalization.

## Architecture: functional core, imperative shell

The rule: **all IO, caching, and the linfa solver live in the shell; the core is
pure functions of already-gathered data.** The shell's only job is to gather
`RawEmail`, counts, and an embedding, hand them to the core, and persist what
comes back. If a function touches notmuch, the filesystem, the embedding model,
or the L-BFGS solver, it is shell. Everything else is core and unit-tested in
isolation.

**Two deep modules.** The core/shell seam is drawn so each side is a *deep*
module — a small public interface hiding most of the complexity. The many small
functions the plan lists (`registrable_domain`, `proportions`, `prepare_text`,
`assemble`, …) are **private** leaves inside these two modules, not top-level
public surface. They keep their own files for testing and readability, but a
caller of the core sees only ~2 functions and a type; a caller of the shell sees
only `train` and `classify_new`.

### Directory layout

```
email_classifier/
├── Cargo.toml
├── src/
│   ├── main.rs           // arg parse → dispatch train/classify. ~50 lines.
│   ├── core/
│   │   ├── mod.rs        // THE core interface: re-exports Priority + ~2 fns.
│   │   ├── labels.rs     //   Priority enum + tag mapping (pub — the vocabulary)
│   │   ├── domain.rs     //   (private) registrable_domain
│   │   ├── history.rs    //   (private) ClassCounts, proportions, confidence
│   │   ├── text.rs       //   (private) prepare_text
│   │   ├── features.rs   //   (private) assemble
│   │   └── model.rs      //   Model struct; predict_proba, predict (pure)
│   └── shell/
│       ├── mod.rs        // THE shell interface: train() and classify_new() only.
│       ├── notmuch.rs    //   (private) search + count adapter, the HashMap cache
│       ├── embed.rs      //   (private) Embedder trait + fastembed impl
│       ├── mailfile.rs   //   (private) mail-parser → RawEmail
│       ├── persist.rs    //   (private) JSON save/load + load-time guards
│       └── fit.rs        //   (private) linfa L-BFGS solve
└── models/               // gitignored, machine-local, regenerable
    └── model.json        // the single serialized regression model
```

**Enforced boundary.** `core/` has no `use` of `notmuch`, `fastembed`, or
`std::fs` — "core is pure" is checkable by inspecting `core/mod.rs`'s imports.
The dependency is one-directional: the seam types `RawEmail` and `ClassCounts`
live in `core` (the core defines what it consumes), and `shell` depends on
`core`, never the reverse.

**The two interfaces:**
```
core::mod.rs
  pub use labels::Priority;
  pub struct Model                                    // weights, intercepts, prior,
                                                      //  alpha, versions
  pub fn classify(&Model, &RawEmail,
                  domain_counts, addr_counts) -> Priority
  pub fn features_for(&RawEmail,
                  domain_counts, addr_counts) -> Vec<f32>   // train's pure half

shell::mod.rs
  pub fn train(model_path) -> Result<()>              // query→parse→features→fit→save
  pub fn classify_new(model_path) -> Result<()>       // the post-new hook entry point
```
`classify` hides the whole `domain → proportions/confidence → assemble →
predict_proba → argmax` pipeline behind one call; `main.rs` calls only the two
shell functions.

**Embedding weights are not our file.** The MiniLM ONNX weights are a separate,
large, write-once artifact keyed by `embedding_model_id`; `fastembed`
auto-downloads them to its own cache dir. `models/` holds *only* `model.json` —
never co-locate the ONNX blob there (different size, lifecycle, and owner).

### Shell (IO + stateful adapters; integration-tested only)
- **Load labeled messages**: confirmed labels are the `prio-*` tags minus
  unreviewed guesses. `notmuch search --output=files tag:prio-<level> and not
  (tag:auto and tag:unread)` → parse each file with `mail-parser` →
  `RawEmail { from, subject, body, ts }`.

  **Deduplicate by message, not file (`--duplicate=1`).** Plain `--output=files`
  returns one path *per maildir file*, and one logical message often has several —
  the same mail is forwarded between accounts, so it lands in each account's
  maildir and notmuch indexes every copy under the same Message-ID. Training on
  that file list counts such a message once per copy, silently over-weighting
  exactly the mail that crosses accounts (and inflating the logged per-class
  sizes: e.g. 1211 files vs. 785 messages on the current archive). The selection
  therefore passes `--duplicate=1`, which makes notmuch emit exactly one
  representative file per *message* — its own identity does the collapse, so we
  never dedup paths ourselves. One `search` per level, no N+1 per-id reads. This
  is a training-set correctness fix, not a perf nicety — note it here so the
  file-vs-message distinction is not re-lost. Classification is unaffected either
  way (guesses are written by `id:<msgid>`, so duplicate files of one message
  collapse onto one tag write).
- **History-count adapter**: `notmuch count tag:prio-<level> and not (tag:auto and
  tag:unread) from:<addr>` and the same `from:<domain-pattern>` per class. The
  `not (tag:auto and tag:unread)` filter is mandatory here — sender proportions
  must never be built from the model's own unreviewed guesses, or it feeds its
  bias back into its features. No separate database.
  A `HashMap<String, ClassCounts>` (one for domains, one for addresses) sits in
  front of the queries — first miss populates, repeated senders cost one query
  total. This cache is deliberately outside the core: it turns effectful,
  order-dependent lookups into the plain `ClassCounts` the core consumes. Per-
  process; corrections (retags) are picked up on the next run.
- **Embedding adapter**: invoke the model, return a `[f32; 384]` unit vector.
- **Solver invocation**: call `model::fit` (below) — deterministic given a seed,
  but heavy compute returning `Result`, so it stays in the shell.
- **Persistence**: model save/load (JSON), including the load-time guard checks.

**Failure & cold-start policy (shell):**
- A notmuch count query that errors is treated as *unknown history*, not a crash:
  fall back to `ClassCounts::ZERO`, which the core already maps to the prior with
  `confidence = 0`. Log once; never abort a batch classify for one bad sender.
- An embedding-model failure *does* abort — a zero/garbage embedding would be
  silently misclassified, which is worse than stopping.

**Training-time leak note:** history features for a training example should use
counts *as of that email's arrival*. v1 uses final tag counts and accepts the
leak — but be honest about the cost: the leak inflates exactly the history
proportions that carry the most signal, so v1 train accuracy will read
optimistically. Treat v1 numbers as a floor on error, not a real estimate.
v2 removes it: replay emails in date order, folding counts through the HashMap —
a pure fold with zero notmuch count queries during training.

### Core (pure functions; unit tested)

`mod.rs` re-exports `Priority`, `Model`, `classify`, and `features_for`; the
`domain`/`history`/`text`/`features` leaves below stay **private** (`mod`, not
`pub mod`) — they exist for isolation and testing, not as public surface.

```
labels.rs   (pub — the shared vocabulary)
  Priority::{P1, P2, P3} <-> usize                    // class index
  Priority <-> "prio-low"|"prio-normal"|"prio-high"   // priority tag (P1=low, P2=normal, P3=high)
  // "auto" is a flat marker managed by the shell, not a Priority — not modeled here
  // single source of truth for the tag<->priority mapping; tag strings live nowhere else

model.rs    (pub: Model, classify, features_for; predict_proba/predict pure helpers)
  predict_proba(model, x) -> [f32; 3]                 // softmax over weights·x
  predict(model, x) -> Priority                       // argmax
  // classify + features_for compose the private leaves below into the two
  // public entry points; fit() is effectful and lives in the shell (fit.rs)

domain.rs   (private)
  registrable_domain(from_addr) -> Option<String>     // psl-based eTLD+1

history.rs  (private)
  ClassCounts = [u32; 3]                              // seam type, re-exported by mod.rs
  proportions(counts, prior, alpha) -> [f32; 3]       // Dirichlet-smoothed, sums to 1
  confidence(counts) -> f32                           // min(1, ln(1+n)/ln(1000))

text.rs     (private)
  prepare_text(subject, body) -> String
  // subject first, "\n", cleaned body, truncated to char budget (~2k chars;
  // model truncates at 512 tokens anyway — subject-first guarantees the
  // densest signal survives)
  // cleaning: strip HTML if no text/plain; drop quoted reply chains
  // ("> " lines, "On ... wrote:"); drop signature after "-- " line

features.rs (private)
  assemble(embedding, domain_hist, addr_hist) -> Vec<f32>   // fixed order, 392 dims
```

## Persistence

One JSON file, `models/model.json`: `{ feature_version, class_prior, alpha,
weights, intercepts, embedding_model_id }`. Readable and diffable by eye. Save/
load and the guard checks below live in `shell/persist.rs`; never shard this
into multiple files.

**Load-time guards (shell, both must pass or refuse to run):**
- `feature_version` must match the code's current version — guards against
  feature-order skew between train and inference.
- `embedding_model_id` must match the embedder in use — a different model makes
  the 384 embedding dims silently meaningless while the file still deserializes
  cleanly. This is the more dangerous skew because it produces confident wrong
  answers rather than a shape error, so it is checked explicitly, not assumed.

## Data flow

**Inference** (║ marks the shell→core→shell seam):
```
shell │ select in-scope mail (date:2026-07-01..), parse, embed
      │ prepare_text(subject,body), memoized domain + address counts
      ╠══ hand (embedding, domain_counts, addr_counts) to core ══
core  │ registrable_domain · proportions/confidence · assemble
      │ · predict_proba · argmax → Priority
      ╠══ return Priority to shell ══
shell │ write prio-{low,normal,high} + auto  (skip if already prio-* w/o auto)
```
Runs from the post-new hook on each `notmuch new`.
Everything the core does is a pure function of the three inputs the shell
gathered; nothing inside the core reaches back out to notmuch or the model.

**Feedback loop:** reading a guess (drops `unread`) confirms it; correcting it in
aerc re-sets `prio-*` and clears `auto`. Either way it becomes a confirmed label,
so history counts (which filter out `auto and unread`) are automatically right on
the next run, no retrain. Retrain weights in batch occasionally (e.g., after N
corrections).

**Training:** `notmuch search (tag:prio-low or tag:prio-normal or tag:prio-high)
and not (tag:auto and tag:unread)` (**all dates**, no cutoff) → parse files →
warm-cache feature extraction → linfa fit → save JSON model.

## Unit tests (core)

- `proportions`: sums to 1; shrinks toward prior at low counts; approaches
  empirical ratio at high counts; alpha edge cases
- `confidence`: 0 at n=0; monotone; capped at 1
- `registrable_domain`: gmail.com, co.uk multi-part TLDs, subdomains, invalid input
- `prepare_text`: subject always first; quoted reply block removed; signature
  after `-- ` removed; HTML stripped; char budget respected; empty body safe
- `assemble`: output length exactly 392; order stable (golden test)
- `predict_proba`/`predict` (pure core): on a hand-built weight matrix, probs
  sum to 1 and argmax picks the expected class — no solver involved
- `fit` (shell): tiny linearly separable synthetic set → ~100% train accuracy
  (integration-level; exercises linfa, not a core unit test)
- label mapping round-trips
- model JSON round-trips (serialize → deserialize → identical predictions)

## Phasing

1. **Baseline:** embeddings → multinomial LR. CLI: `train`, `classify`.
2. **History:** notmuch-backed domain + address counts with HashMap memoization;
   corrections are just notmuch retags.
3. **Optional / as-needed:** chronological-replay training (fixes the leak),
   ordinal (proportional-odds) regression, model2vec embedder, Tranco rank if
   novel-domain errors show up in the confusion matrix.

## Evaluation

**Evaluate only against confirmed labels** — the same
`not (tag:auto and tag:unread)` set used for training. Grading the model on its
own unreviewed guesses would score it against itself and read falsely high. A
corrected guess is a genuine test point: it's a case the model got wrong and a
human fixed, so it belongs in the held-out set like any other confirmed label.

Held-out split by time (not random) to mimic deployment. Track overall accuracy,
per-class recall (p3 recall matters most), and adjacent-vs-distant confusion
(p1↔p3 mistakes are worse than p2↔p3).

## Implementation checklist

Ordered bottom-up: the **pure core leaves first** (each unit-testable in
isolation with no IO), then the **model math**, then the **shell adapters** that
feed them, then the two **shell entry points**, then **wiring**. Each item is
checkable on its own before the next depends on it. `[core]` = pure, unit-tested;
`[shell]` = IO, integration-tested. This realizes Phase 1–2 of *Phasing*; the
Phase 3 items are listed last as deferred.

**0. Project skeleton** ✅ complete
- [x] `cargo new email_classifier` at `email/email_classifier/`; add the *Crates*
  table deps to `Cargo.toml`.
- [x] Create the `core/` and `shell/` module dirs with empty `mod.rs` files so the
  boundary exists from commit one; add `models/` to `.gitignore`.

**1. Core vocabulary and leaves** (no deps between most of these — parallelizable) ✅ complete
- [x] `core/labels.rs` — `Priority` enum, `<->usize`, `<->` tag strings. *Test:*
  round-trips. Everything else imports this, so it goes first.
- [x] `core/domain.rs` — `registrable_domain` (psl). *Test:* gmail.com, co.uk,
  subdomains, invalid input.
- [x] `core/history.rs` — `ClassCounts`, `proportions`, `confidence`. *Test:*
  sums-to-1, shrink-to-prior, monotone confidence, alpha edges.
- [x] `core/text.rs` — `prepare_text`. *Test:* subject-first, quote/sig strip,
  HTML strip, char budget, empty body.

**2. Feature assembly + model math** (depends on §1) ✅ complete
- [x] `core/features.rs` — `assemble(embedding, domain_hist, addr_hist)`. *Test:*
  length exactly 392, stable order (golden). Needs `history` outputs to exist.
- [x] `core/model.rs` — `Model` struct + pure `predict_proba` / `predict`. *Test:*
  hand-built weight matrix → probs sum to 1, argmax picks expected class. No
  solver yet.
- [x] `core/mod.rs` — compose the leaves into the two public fns `classify` and
  `features_for`; re-export `Priority`, `Model`, `ClassCounts`. This closes the
  core interface — verify `core/mod.rs` has **no** `use` of notmuch/fastembed/fs.
  (`classify`/`features_for` take the embedding as an argument; `prepared_text`
  hands the shell the string to embed, keeping the embedder out of the core.)

**3. Persistence** (depends on `Model` existing) ✅ complete
- [x] `shell/persist.rs` — JSON save/load + load-time guards (`feature_version`,
  `embedding_model_id`). *Test:* serialize→deserialize→identical predictions;
  guard rejects a mismatched `embedding_model_id`.

**4. Shell adapters** (independent of each other; all depend on §2 seam types) ✅ complete
- [x] `shell/mailfile.rs` — `mail-parser` → `RawEmail`.
- [x] `shell/embed.rs` — `Embedder` trait + fastembed impl → `[f32; 384]`.
  Embedding failure aborts (per failure policy).
- [x] `shell/notmuch.rs` — search/count adapter + the two `HashMap` caches;
  count errors fall back to `ClassCounts::ZERO`. Carries the
  `not (tag:auto and tag:unread)` filter.
- [x] `shell/fit.rs` — linfa L-BFGS solve → weights/intercepts. *Test:* tiny
  linearly-separable set → ~100% train accuracy.

**5. Shell entry points** (compose everything above) ✅ complete
- [x] `shell/mod.rs::classify_new` — select `date:2026-07-01..` → parse → embed →
  warm counts → `core::classify` → write `prio-*` + `auto`, skipping confirmed
  mail. This is the post-new hook path; build it before `train` so you can
  eyeball predictions with a hand-made model.json. The skip-confirmed rule is
  folded into the notmuch query (`not ((prio-*) and not tag:auto)`); guesses are
  written by notmuch message id (`id:<msgid>`), so a message with no Message-ID
  is skipped rather than mistagged. Per-message failures are logged and skipped;
  an embedding failure aborts.
- [x] `shell/mod.rs::train` — query confirmed labels (all dates) → parse →
  `features_for` → `fit` → `persist::save`. Labels come from one
  `search --output=files --duplicate=1` per priority level (the query supplies the
  label), so no per-file tag read is needed and each message contributes one
  example — a message forwarded across accounts is no longer trained once per
  maildir copy (see *Load labeled messages* → `--duplicate=1`).

**6. Wiring + deploy**
- [x] `main.rs` — arg parse → dispatch `train` / `classify`, optional
  `--model <path>`. Only calls the two shell fns; exits non-zero on error.
- [x] Run `train` on the real archive (`task build-train` → `models/model.json`);
  `train` logs per-class set sizes up front.
- [x] **Deduplicate training files by message** (see *Load labeled messages* →
  `--duplicate=1`): `search --output=files --duplicate=1` collapses
  forwarded-across-accounts copies to one example per notmuch message, so accuracy
  numbers no longer double-count cross-account mail.
- [ ] Sanity-check the confusion matrix against a time-held-out split
  (*Evaluation*); classify once by hand and eyeball the guesses.
- [ ] Install as notmuch **post-new hook**; confirm it fires on an mbsync cycle and
  respects Scope + the skip-confirmed rule.

**Deferred (Phase 3, only if the confusion matrix demands it):** chronological-
replay training (removes the training-time leak), proportional-odds ordinal
regression, `model2vec` embedder swap, Tranco rank feature.
