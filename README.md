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

**Not yet implemented:**
- §2 — `core/features.rs` (`assemble`), `core/model.rs` (`Model`,
  `predict_proba`/`predict`), and the composed public entry points (`classify`,
  `features_for`).
- §3–§5 — the shell: `persist`, `mailfile`, `embed`, `notmuch`, `fit`, and the
  `train` / `classify_new` entry points. `src/shell/mod.rs` is still a stub.
- §6 — `main.rs` CLI dispatch, real-archive training, and post-new hook install.

See the *Implementation checklist* in `design.md` for the full ordered plan.
