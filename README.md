# email_classifier

A local-only email priority classifier. It assigns incoming mail one of three
ordered priorities and applies them as notmuch tags:

| Priority | Tag           | Meaning         |
|----------|---------------|-----------------|
| P1       | `prio-low`    | least important |
| P2       | `prio-normal` | —               |
| P3       | `prio-high`   | most important  |

Classification uses multinomial logistic regression over a text embedding
(all-MiniLM-L6-v2, 384-dim) plus smoothed per-domain and per-address tag-history
proportions — 392 features total, all scaled to ≈[0,1] by construction. The
whole interface is notmuch tags; the model runs from notmuch's post-new hook,
with no daemon and no separate database. See `design.md` for the full
specification and rationale.

## Building and testing

This project uses [Task](https://taskfile.dev) (go-task):

```
task build   # cargo build --release
task test    # cargo test
task --list  # list tasks
```

## Architecture

Functional core / imperative shell (see `design.md` and `CLAUDE.md`):

- `src/core/` — pure functions, unit-tested, no IO.
- `src/shell/` — all IO, caching, and the linfa solver.
- `models/model.json` — the single serialized model (gitignored, regenerable).

## Implementation status

Built bottom-up following the checklist in `design.md`. Current state:

**Done — §1, core vocabulary and leaves:**
- `core/labels.rs` — `Priority` enum and tag-string mapping (single source of
  truth for the tag vocabulary).
- `core/domain.rs` — `registrable_domain` (eTLD+1 via `psl`).
- `core/history.rs` — `ClassCounts`, Dirichlet-smoothed `proportions`,
  `confidence`.
- `core/text.rs` — `prepare_text` (subject-first, quote/signature/HTML stripping,
  char budget).
- `core/mod.rs` — declares the leaves and defines the `RawEmail` seam type;
  re-exports `Priority`, `ClassCounts`.

**Done — §2, feature assembly + model math:**
- `core/features.rs` — `assemble` (fixed 392-dim layout, golden-tested order).
- `core/model.rs` — `Model` struct and pure `predict_proba`/`predict` (softmax +
  argmax), plus `FEATURE_VERSION`.
- `core/mod.rs` — composes the leaves into the two public entry points
  `classify` and `features_for`, plus `prepared_text` (the shell embeds its
  output — the core does not touch the embedder). Boundary verified: `core/` has
  no `use` of notmuch/fastembed/std::fs.

**Done — §3, persistence:**
- `shell/persist.rs` — JSON `save`/`load` for `models/model.json` with both
  load-time guards (`feature_version`, `embedding_model_id`). Round-trip and
  guard-rejection tests included.

**Done — §4, shell adapters:**
- `shell/mailfile.rs` — `mail-parser` → `RawEmail`, preferring the text/plain
  part and falling back to HTML.
- `shell/embed.rs` — the `Embedder` trait plus the `fastembed`
  (all-MiniLM-L6-v2) backend; embedding failure aborts, per the failure policy.
- `shell/notmuch.rs` — the `notmuch`-CLI search/count adapter with the two
  `HashMap` caches; count errors fall back to `ZERO_COUNTS`, and the mandatory
  `not (tag:auto and tag:unread)` filter lives here in one place.
- `shell/fit.rs` — the linfa L-BFGS solve, packing linfa's params into the
  core `Model` (with the class-column → `Priority`-row mapping). Includes the
  tiny-separable-set integration test.

**Done — §5, shell entry points:**
- `shell/mod.rs::classify_new` — the post-new hook path: select in-scope new mail
  (`date:2026-07-01..`, with the skip-confirmed rule folded into the query),
  parse → embed → warm domain + address counts → `core::classify` → write
  `prio-*` + `auto` addressed by notmuch message id. Per-message failures are
  logged and skipped; an embedding failure aborts.
- `shell/mod.rs::train` — fit over every confirmed label (all dates): one
  `search --output=files` per priority level supplies each file's label, parse →
  `core::features_for` → `shell/fit.rs::fit` → `persist::save`.

**Done — §6, wiring:**
- `main.rs` — `train` / `classify` subcommand dispatch with an optional
  `--model <path>` (default `models/model.json`); calls only the two shell entry
  points. Exit status is 1 on error so the post-new hook surfaces a failed run.
- `Taskfile.yml` — `task train`, `task classify`, and `task build-train`
  (release-build then train and persist), all teeing output under `output/`.
- Trained on the real archive: `task build-train` fits over the confirmed-label
  file set and writes `models/model.json`. `train` logs per-class set sizes up
  front. Selection uses `search --output=files --duplicate=1`, so a message
  forwarded across accounts is one training example, not one per maildir copy.

**Not yet implemented:**
- §6 — post-new hook install (confirm it fires on an mbsync cycle) and the
  time-held-out confusion-matrix sanity check.

See the *Implementation checklist* in `design.md` for the full ordered plan.
