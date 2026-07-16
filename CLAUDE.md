# CLAUDE.md

Guidance for agents working in this repository.

## What this project is

A local-only email priority classifier. It sorts incoming email into three
ordered priorities (`prio-low < prio-normal < prio-high`) using multinomial
logistic regression over a text embedding plus per-domain and per-address tag
history. The entire user-facing API is **notmuch tags** — there is no daemon and
no separate database; the classifier runs from notmuch's post-new hook.

`docs/architecture.md` is the living reference for how the code is put together:
the tag/labeling model, the feature layout, the core/shell architecture, and the
invariants (e.g. the `not (tag:auto and tag:unread)` filter, load-time guards)
that are easy to break accidentally. **Read `docs/architecture.md` before making
non-trivial changes** — most design questions are already answered there. Keep it
current as the architecture evolves.

`docs/designs/done/design.md` is the original pre-implementation spec — the
rationale the design was worked out against, now realized in the code. Consult it
for the *why* behind a decision; `architecture.md` is the source of truth for the
*what*. Satellite design docs for individual features live alongside it in
`docs/designs/done/` once built (see e.g. `embedding-cache.md`).

`README.md` is the user-facing overview. Keep it current as behavior changes.

## Running tasks

Use [Task](https://taskfile.dev) (go-task) for common commands — prefer these
over calling cargo directly, so there is one obvious way to run each thing:

- `task test` — run the test suite (`cargo test`)
- `task build` — release build (`cargo build --release`)
- `task run` — run the classifier (release), teeing all output to a
  timestamped file under `output/`
- `task --list` — see available tasks

`task run` passes any extra arguments through to the binary, e.g.
`task run -- classify`. It writes combined stdout+stderr to
`output/run-<timestamp>.log` (via `tee`) so a run can be inspected afterwards
without re-running. The `output/` directory is created on demand and is
gitignored (machine-local logs, like `models/`).

When adding a common workflow (lint, train, classify, etc.), add it to
`Taskfile.yml` rather than documenting a bare cargo/shell incantation.

## Coding style

**Functional core, imperative shell.** This is the load-bearing architectural
rule, not a preference:

- **`core/`** is pure functions of already-gathered data. It must have **no**
  `use` of `notmuch`, `fastembed`, or `std::fs` — the boundary is checkable by
  inspecting imports. Everything here is unit-tested in isolation.
- **`shell/`** owns all IO, caching, and the linfa solver. It gathers inputs
  (`RawEmail`, counts, embeddings), hands them to the core, and persists results.
- The dependency is one-directional: `shell` depends on `core`, never the
  reverse. Seam types (`RawEmail`, `ClassCounts`) live in `core` because the core
  defines what it consumes.
- Both sides are **deep modules**: a small public surface hiding the internals.
  The many small leaf functions stay private (`mod`, not `pub mod`); a caller of
  the core sees ~2 functions and a type, a caller of the shell sees `train` and
  `classify_new`.

When in doubt about where code belongs: if it touches notmuch, the filesystem,
the embedding model, or the L-BFGS solver, it is shell. Otherwise it is core.

Match the surrounding code's naming, comment density, and idiom. Add unit tests
alongside new core functions (see the *Unit tests* section of
`docs/designs/done/design.md`).
