# Embedding Cache — Design

A satellite design doc to `design.md`. It specifies a persistent cache of text
embeddings so any run — `train`, `eval`, or the `classify_new` hook — reuses the
vectors an earlier run already computed, instead of re-embedding the same
messages every time. The store is a single-file embedded KV database (redb),
queried per-key rather than loaded whole, which is what lets the cache sit safely
on the frequently-firing classify path. Read `design.md` first — this doc assumes
its vocabulary (core/shell, `RawEmail`, `prepared_text`, the load-time guards)
and only describes the delta.

## Motivation

Embedding is the dominant cost in `train`, `eval`, **and `classify_new`**
(`design.md` → *Features*, "batch the embedding calls"). `train`/`eval`
re-embed **every** confirmed label on **every** run; `classify_new` re-embeds
every in-scope message — mail since the cutoff that is still `unread` and not
confirmed — on **every** `notmuch new`, so an unread `auto` guess is re-embedded
run after run until it is read. But the embedding of a message is a pure
function of two things only:

- the **prepared text** (`core::prepared_text(&email)`), and
- the **embedding model** (`EMBEDDING_MODEL_ID`).

Neither changes between two runs over the same archive. So the second run's
embedding work is almost entirely redundant — on all three paths. A cache keyed
on those two inputs turns a cold re-embed into a key lookup for every message
seen before.

This is a pure speed optimization. It must not change any model's weights: a run
with a warm cache and a run with a cold cache must produce byte-identical
feature vectors (a cache hit returns the *same* vector the embedder would have).

## What this is *not*

- Not a daemon and not a shared service — one machine-local, single-file
  embedded store (redb), opened by a classifier process while it runs, in
  the spirit of the existing `models/` and `.fastembed_cache/` artifacts. "No
  separate database" in `design.md` means no *server* and no second source of
  truth for the tag model; an embedded on-disk KV file used purely as a
  regenerable cache is not that. This is the intended *single-writer* case — but
  it is not *guaranteed*: because `classify_new` fires from the `notmuch new` hook
  every 1–5 min, two runs can overlap (an mbsync cycle running long into the
  next), and redb takes an exclusive file lock. The loser of that race does not
  abort — it falls back to running without the cache, handled as a *Failure
  policy* case below, so tagging is never blocked on cache access.
- Not a correctness mechanism. If the cache is deleted, corrupted, or stale for
  any reason, the only consequence is a slower run — never a wrong vector (see
  *Failure policy*).

## Core / shell placement

**The entire cache is shell.** It touches the filesystem (now via redb) and
fronts the embedding model — by `design.md`'s own boundary rule ("if it touches
notmuch, the filesystem, the embedding model, or the L-BFGS solver, it is
shell"), it cannot be anything else. The core is untouched: embeddings still
flow into `features_for` / `classify` as plain `Vec<f32>` data, exactly as
today. There is no new `use` of `std::fs`, `redb`, or `fastembed` anywhere under
`core/`.

The cache is a direct sibling of the two caches `design.md` already describes as
"deliberately outside the core" — the domain and address `HashMap<String,
ClassCounts>` in front of the notmuch count queries. Same shape (a lookup
fronting an effectful computation), same rationale (turn repeated effectful work
into the plain data the core consumes). Two twists: this store is *persisted*
between runs rather than per-process, and it is a redb file queried by key
rather than an in-memory `HashMap` loaded whole (see *Storage* for why).

### The decorator seam

The cache slots in **behind the existing `Embedder` trait**, not beside it. That
trait already exists precisely as the swappable embedding seam (`shell/embed.rs`
— the model2vec escape hatch is meant to drop in here). We add a caching
decorator that *is* an `Embedder`, wrapping a concrete backend:

```
trait Embedder { fn model_id(&self) -> &str; fn embed(&self, text: &str) -> Result<Vec<f32>>; }

FastEmbedder          — the real ONNX model (unchanged)
CachingEmbedder<E>    — wraps an inner E: Embedder + an Option<redb store>
```

```
struct CachingEmbedder<E: Embedder> {
    inner: E,
    db:    Option<redb::Database>,   // None = caching disabled (see Failure policy)
    hits:  usize,                    // (AtomicUsize only if embed is &self + shared)
    misses: usize,
}
```

**The `Option` is the whole failure policy in one field.** `db` is `Some` when the
store opened (or was created) cleanly and `None` when opening failed for *any*
reason — missing-and-uncreatable, corrupt-beyond-open, permission, disk-full, or
**locked by a concurrent run**. `embed` branches on it once: `Some` → the
get-or-miss cache path; `None` → forward straight to `inner.embed`, caching
nothing. That single match replaces the per-failure-mode recovery prose the
earlier draft spread across sections — every open failure funnels to the same
bypass state, and there is exactly one place that decides caching is off. (Note
`None` is *not* the same as a cold `Some`: a freshly-created empty store is
`Some` and still writes its misses; `None` never writes at all. See *Failure
policy*.)

`CachingEmbedder::embed` is, when `db` is `Some`: hash the text → `get` the key
from redb → on hit, return the stored vector; on miss, call `inner.embed`,
buffer the insert, return. When `db` is `None` it is just `inner.embed`. Because
it implements `Embedder`, all three entry points keep calling
`embedder.embed(&text)` at the same call sites (`shell/mod.rs::classify_new`,
`shell/mod.rs::train`, `shell/eval.rs`) with **no change to their loops** — they
just hold a `CachingEmbedder<FastEmbedder>` instead of a `FastEmbedder`. The core
never sees any of this.

Because redb is queried by key, there is no whole-file load and no whole-file
write-back: the decorator opens the store when constructed and reads/writes
individual keys as it serves each `embed`. Opening the store (and its per-run
hit/miss tallies) is the enclosing shell's job at each entry point, where the
other persistence already lives; the file IO itself is redb's, transactional and
per-key rather than a bulk rewrite. This is what makes the cache safe on the
frequently-firing `classify_new` hook (see *Location and lifecycle*): a classify
run inserts only its handful of misses instead of rewriting the whole cache.

## The cache key

`key = hash(prepared_text)`, a `u64` (a fast non-cryptographic hash, e.g.
`xxhash`/`ahash` — collision-resistance is not a security property here, just a
table key). The store is a single redb table `u64 → &[u8]`, the bytes being the
raw little-endian `f32` values of the embedding (see *Storage*).

**Why hash the prepared text and not the Message-ID:** the text hash
*self-invalidates* on the one change that would otherwise poison the cache
silently — an edit to `core::prepared_text` (how subject/body are assembled and
truncated). Change that logic and every key changes, so stale vectors simply are
never found; the run recomputes and repopulates. A Message-ID key would keep
returning vectors for the *old* text after such an edit — a silent-wrong-vector
bug, exactly the class of failure `design.md`'s embedding-model guard exists to
prevent. The text hash makes the correct behavior automatic.

Storing the raw text as the key instead of a hash would also work and be even
more obviously correct, at the cost of a much larger file (the truncated bodies).
The hash is the compact form of the same idea; a collision would return the wrong
vector, but at `u64` width over an archive of ~10³ messages the probability is
negligible and, if paranoia warrants, the stored value can carry the text length
as a cheap tripwire. **v1: bare `u64` key.**

## The model-id guard (mirrors persistence)

A cached vector is meaningless under a different embedder — the same silent skew
`persist.rs` guards against for the model file ("the 384 embedding dims came from
a different embedder ... scores confident nonsense"). The cache carries the same
guard, resolved the simplest way: **the model id is in the filename**.

```
cache/embeddings-all-MiniLM-L6-v2.redb
```

Opening the cache for embedder *X* opens `cache/embeddings-<X>.redb`. A different
embedder opens a different file — a mismatch is structurally impossible to load,
so there is no in-file `embedding_model_id` field to check and no cross-model
contamination. Switch embedders and the old file is simply never opened (and can
be deleted at leisure). This is the file-naming analogue of `persist.rs`'s
`EmbeddingModel` guard.

**The id must be sanitized for the filename — injectively.** The example above
(`all-MiniLM-L6-v2`) is filename-clean and hides a bug: a HuggingFace id like
`sentence-transformers/all-MiniLM-L6-v2` (the form `design.md` uses) contains a
`/`, so `cache/embeddings-sentence-transformers/all-MiniLM-L6-v2.redb` names a
file inside a nonexistent `sentence-transformers/` subdirectory. The open fails,
and — per *Failure policy* — silently falls back to a cold cache **on every run**:
a cache that quietly never works, the worst outcome for a pure optimization
(no error, no speedup, forever). So the id is passed through a sanitizer before
it goes in the filename.

The sanitizer must be **injective**: two distinct model ids must never map to the
same filename, or we reintroduce exactly the cross-model contamination the
per-file guard exists to prevent (e.g. a naive `/`→`-` replacement collapses
`a/b` and `a-b` onto one file). Percent-encode the path-hostile characters (`/`,
and any other separator disallowed in a filename) rather than replacing them with
a character that could already appear in an id. This keeps the mapping reversible
and collision-free.

Note this stays purely a *filename* transform — **there is no in-file
`embedding_model_id` field and no metadata table.** Adding one would reintroduce
the in-file check this section deliberately eliminated; the sanitized filename
*is* the guard. (If a tamper tripwire is ever wanted, it is the same optional
idea as the key's length tripwire in *The cache key*, not a required table.)

## Storage: redb (values are raw little-endian `f32` bytes)

The store is [`redb`](https://docs.rs/redb): a single-file, pure-Rust,
ACID embedded key–value store. One redb table maps `u64 → &[u8]`, where the
bytes are the embedding's `EMBED_DIM` `f32` values written back-to-back in
little-endian order — no serialization framing at all (see *Value encoding*
below for why there is no `serde`/`postcard` in the value path). What changed
from the flat-file design is the **container**: a redb table indexed by key
rather than one flat blob loaded whole into a `HashMap`.

**Why a keyed store and not a whole-file map.** Extending the cache to
`classify_new` — which runs from the `notmuch new` hook every 1–5 minutes —
breaks two assumptions the flat-file design rested on:

- **Load cost on the hot path.** A flat file must be deserialized *in full* to
  answer even one lookup. On a hook that fires every few minutes, paying a
  whole-map load per run to serve a handful of lookups is pure waste, and it
  grows with the file. redb answers a `get` by walking its on-disk B-tree — no
  whole-file load — so the hook's cost is proportional to the few keys it
  actually touches.
- **Write amplification and drive wear.** The flat design wrote the *entire* map
  back once per run. A classify run adds only a few vectors; rewriting the whole
  file (tens to hundreds of MB as it grows — see *Sizing*) to append a few KB,
  every few minutes, is large and needless write volume against SSD endurance.
  redb commits only the changed pages in a transaction, so a classify run writes
  ~KB, not the whole store. (Reads never wear the drive; the concern is
  strictly the repeated whole-file *writes* the flat model implied.)

Neither bit until the cache went onto the frequently-firing, ever-growing
classify path; on `train`/`eval` alone a flat file was fine. The keyed store is
what makes "check the cache before every embedding, everywhere" safe.

**Why redb specifically.** The workload is the simplest possible KV: point `get`
by `u64`, `insert` on miss, one writer at a time, no ranges, no secondary
indexes, no cross-key transactions. That makes operational robustness and
dependency weight the deciding factors, not features or throughput — everything
here is fast enough.

- **redb** — pure Rust, single file, ACID, KV-native API (`insert`/`get`). No C
  toolchain, integrates like any crate, matches this `cargo`-native tree. Chosen.
- **sqlite (`rusqlite`)** — the most proven storage engine there is, and the
  main alternative considered. Passed over: it pulls in a C build, and we'd use
  ~1% of it (one `(key, blob)` table, no SQL of substance). Its unique strength,
  "never corrupts," is largely wasted on a *regenerable* cache whose failure
  policy is already "start empty and rebuild." A defensible conservative pick if
  a C dep is ever preferred; not worth it here.
- **fjall** — pure-Rust LSM KV. Fine, but LSM machinery (compaction, tiered
  files) targets write volumes far above this cache's; more moving parts than the
  job needs.
- **rocksdb / sled / heed(LMDB)** — rejected: heavyweight C++ dep and tuning
  surface (rocksdb); perpetual-beta maturity concerns (sled); C dep and mmap
  footguns with no advantage over redb here (heed).

### Value encoding: raw little-endian `f32` bytes, no serializer

The value is a **fixed-shape** payload — always exactly `EMBED_DIM` (384) `f32`s,
a shape that never grows a field — so there is nothing for a serialization crate
to earn. The store holds each vector as `EMBED_DIM × 4` bytes: on write,
`f.to_le_bytes()` for each `f32` concatenated; on read, `chunks_exact(4)` +
`f32::from_le_bytes`. No `serde`, no `postcard`, no `unsafe`.

**Why this replaced the earlier `postcard` value encoding.** The earlier draft
reached for `postcard` (over JSON, `rkyv`, `bitcode`) for a compact, maintained,
serde-native blob. But every argument for it dissolves once the value is a
fixed-length `[f32; 384]`:

- **No schema to evolve.** `postcard`'s payoff is variable-length,
  field-additive structs. A fixed-width vector never changes shape; the one thing
  that *could* change it (`EMBED_DIM`) travels with the model id, and the model id
  is already in the filename (*The model-id guard*) — so that change invalidates
  the file by giving it a new name, exactly the regenerable-cache story ("delete
  the file, rebuild"). There is no migration path to preserve.
- **Not a speedup — a wash, minus a dependency.** Raw bytes are *not* faster than
  `postcard` here: both copy. redb hands back an unaligned `&[u8]`, so the bytes
  can't be zero-copy-cast to `&[f32]` alignment-safely anyway, and `fit` copies
  every vector into an `ndarray` `f64` row regardless (the same reason the earlier
  draft dismissed `rkyv`'s zero-copy). So the honest gain is: *same cost, one
  fewer dependency (`postcard`, and `serde` on the value path), and a stricter
  corruption check* — not throughput. The `chunks_exact` loop optimizes to a
  straight copy, often SIMD.
- **The advisory reason evaporates cleanly.** `postcard` was partly justified as a
  RUSTSEC-flagged-`bincode` replacement; dropping to `to_le_bytes` removes the
  serialization crate from the value path entirely, so it reintroduces no flagged
  dependency — there is simply no serializer here to advise about.

**Alignment and the length check.** Because redb's `&[u8]` is not guaranteed
4-byte aligned, the decode must not cast raw pointers; `chunks_exact(4)` +
`from_le_bytes` is the safe, endian-portable path (`try_into().unwrap()` on a
statically-length-4 chunk cannot panic). On read, verify `bytes.len() ==
EMBED_DIM * 4` **before** decoding and treat a mismatch as a miss (a corrupt or
wrong-width entry → recompute), not a decode of garbage. This length guard is
load-bearing, not decorative: it is also the value-side analogue of the optional
key tripwire in *The cache key* — the cheapest possible catch for a corrupt entry
or a silent `EMBED_DIM` change, folding straight into the *Failure policy* "a
corrupt entry costs speed, never correctness" rule.

If anyone later reaches for `bytemuck`/`zerocopy` to skip the copy: don't. The
copy is the price of staying `unsafe`-free, it is cheap at these sizes, and the
downstream `f64` copy in `fit` spends any zero-copy win anyway.

## Sizing

What the cache accumulates is one entry per **distinct prepared text embedded**,
so on the classify path it grows at the mail **arrival** rate, not the (much
slower) rate at which mail is confirmed into labels. Concretely, from the current
archive:

- **~1.55 KB/entry** (384 × 4-byte `f32` = 1,536 B, plus key and framing).
- Confirmed labels today: **~1,665** (the `train`/`eval` reuse set).
- Arrival rate: **~16k messages/year** and rising ~15%/year.

So the classify cache reaches order **~16k entries (~25 MB) after a year**,
crossing ~100k entries / ~150 MB — the scale where a *flat-file* whole-map load
would start to bite — in roughly **4–5 years**. redb removes that cliff entirely
(no whole-map load), so size is bounded only by disk, which is a non-issue at
these magnitudes; hashing by prepared text also dedups repeated boilerplate,
bending the curve below raw arrival. Growth is monotonic and self-maintaining:
the store only accumulates, and the entire maintenance story remains "delete the
file, the next run rebuilds what it needs."

## Location and lifecycle

- **Directory:** a new `cache/` at the repo root, **not** `models/`. `design.md`
  is explicit that `models/` holds *only* `model.json`; the embedding cache is a
  separate regenerable artifact, like `.fastembed_cache/`. `cache/` is created on
  demand and gitignored (machine-local, regenerable).
- **Open:** at the start of `classify_new`/`train`/`eval`, open (creating
  `cache/` and the file if absent — the same create-parent-dirs pattern as
  `persist::save`) `cache/embeddings-<id>.redb` into the `CachingEmbedder`'s `db`
  field. redb opens by mapping the file, not by reading it whole, so this is cheap
  regardless of size — the property that keeps it safe on the `classify_new` hook.
  The open resolves to the `Some`/`None` state above: a clean open-or-create is
  `Some(db)`; **any** failure (locked, corrupt-beyond-open, permission, disk) is
  `None` — caching off for this run, never an abort. The open is **logged**: on
  `Some`, the path and current entry count
  (`opened cache/embeddings-<id>.redb (N entrie(s))`); on `None`, a one-line
  warning naming the reason and that the run proceeds without the cache.
- **Writes:** transactional and incremental, not a bulk end-of-run rewrite of the
  whole store. A run's misses are inserted under redb write transactions and redb
  commits only the changed pages, so a run writes ~KB (its handful of misses on
  classify; its cold-run misses on `train`/`eval`), never the tens-to-hundreds of
  MB a flat-file whole-map rewrite implied — the write-side half of *why a keyed
  store*. There is no separate "save" step; when the run ends the store is durable.

  **Commit granularity is per *run*, not per *key*.** A redb write transaction can
  hold many inserts and commit once, so the misses of a run are grouped into as
  few commits as the access pattern allows rather than one `commit` per embedded
  message. This matters most on `train`/`eval`, which embed the whole archive: a
  cold first run has ~1.6k misses, and committing each insert separately would be
  the classic write-amplification / slow-fsync-per-row pattern (N commits for N
  vectors). One transaction over the run's misses turns that into a handful of
  commits. Concretely, the `CachingEmbedder` batches its pending inserts and
  flushes them under a single `WriteTransaction` — e.g. buffer misses and commit
  once when the embed loop finishes, or in bounded chunks if memory of the pending
  set is a concern. This is a shell-internal detail entirely under the decorator;
  it changes neither the `Embedder` call sites nor the per-key `get` on the read
  path (reads still walk the B-tree without loading the file whole). It also
  composes cleanly with the future batched `embed` (see *Interaction with batched
  embedding*): the batched-miss `insert` and the single-transaction `commit` are
  the same flush.
- **Growth:** the file only accumulates. Entries for deleted mail linger
  harmlessly (they are simply never looked up). If the file ever wants trimming,
  deleting it is the entire maintenance story — the next run rebuilds what it
  needs. No eviction logic in v1.

## Logging

The cache reports its open and its effectiveness, so a run's log makes plain how
much work the cache saved — no more, no less. It does **not** log per-message
hit/miss chatter in the embed loop, nor per-key redb writes; that would drown the
existing progress lines for zero insight.

- **Open** is logged once, as described under *Location and lifecycle* above —
  path plus current entry count (or the cold-cache note). There is no separate
  save log, since writes are per-key and transactional rather than a single
  end-of-run flush.
- **Effectiveness summary:** after the embed loop finishes (before the fit in
  `train`/`eval`; before writing guesses in `classify_new`), one line reports how
  many embeddings came from the cache versus how many were regenerated this run —
  e.g. `embeddings: 742 from cache, 43 regenerated (785 total)`.
  The `CachingEmbedder` tallies hits and misses as it serves them; the enclosing
  entry point reads those counters and emits the summary. This is the number a
  reader actually wants — it shows the cache working (or, on a cold run, all
  misses) at a glance. On the frequently-firing `classify_new` hook it is
  typically a small line (few new messages), which is exactly the point.

## Failure policy

Consistent with `design.md`'s shell failure policy, but note the asymmetry with
the *live* embedder. The whole policy collapses into the `db: Option<Database>`
state from *The decorator seam*: **any open failure → `None` → bypass caching;
never abort.** The three states are kept distinct:

- **Open failed → `db = None` (bypass).** *Every* reason the store won't open
  funnels here — corrupt beyond opening, permission, disk-full, and **locked by a
  concurrent run** (redb's exclusive lock). There is no per-reason recovery path:
  log the reason once, set `db = None`, and run without the cache — every text is
  a miss served by `inner.embed`, and nothing is written. The overlapping run that
  holds the lock populates the store; the loser just pays full embedding cost this
  once. As with every path here, **a cache failure must never abort tagging** on
  `classify_new`. (Verify redb's behavior when the lock is held: if the second
  open *blocks* rather than *errors*, treat a bounded wait the same way — do not
  let the hook hang, fall back to `None` instead.)
- **Missing file → `db = Some(fresh store)` (cold, not bypass).** A missing cache
  is *not* an open failure: create it and proceed with an empty `Some` store,
  which warms as it writes its misses. This is deliberately distinct from `None` —
  a cold `Some` still populates the cache for next run; `None` never writes. Same
  outcome for *speed* on the first run (all misses), different outcome for the
  file (one leaves a warm cache behind, one leaves nothing).
- **Write failed mid-run → swallow (still `Some`).** A redb `insert`/`commit`
  error *after* a successful open (e.g. the disk fills partway through) is not an
  open failure and the `Option` does not cover it: log and swallow it, on every
  path. The run's real output — the trained model, or the `prio-*` guess tags —
  does not depend on the cache write succeeding; a dropped insert just means that
  vector is recomputed next run. (An implementation may downgrade `Some → None`
  after a write error to stop retrying a dead store for the rest of the run; that
  is an optimization, not required.) This holds for `classify_new` too: a cache
  write failure must never abort tagging.
- A live embedding failure still **aborts**, unchanged — that policy is about the
  model producing a garbage vector, which the cache never does (a hit returns a
  previously-valid vector; a miss defers to the real embedder, which aborts on
  failure exactly as before).

## Interaction with batched embedding (future)

This composes with the batched-embedding step on `design.md`'s roadmap rather
than competing with it. With the cache in place, only cache **misses** need the
model — so the natural combined form is: look every text up, collect the misses,
embed the misses in one batched call, insert them. The cache cuts the *warm*-run
cost to near zero; batching cuts the *cold*-run (and first-run) cost. v1 ships
the cache with the current one-text-at-a-time `embed`; batching is a later,
independent change at the same seam.

## Testing

Per `design.md`'s test split, the cache is shell and is exercised by shell/
integration tests, not core unit tests (the core is unchanged and its existing
tests already pin that embeddings-as-data contract). Worthwhile shell tests:

- **Hit returns the inner vector unchanged:** a `CachingEmbedder` over a stub
  `Embedder` that records call count — second `embed` of the same text returns
  the identical vector and does **not** call the inner embedder again.
- **Distinct texts miss independently:** two different texts each hit the inner
  embedder exactly once.
- **Persistence round-trip:** populate a `CachingEmbedder` over a temp-dir redb
  file, drop it, open a fresh `CachingEmbedder` on the same path, confirm a
  previously-computed text is now a hit (inner call count stays zero) — the
  cross-run reuse the whole cache exists for.
- **Missing file opens cold (`Some`):** a non-existent path yields an empty
  *created* store — `db` is `Some`, and a miss embedded now is a hit after a
  drop/reopen (it warmed the file). Distinguishes cold-`Some` from bypass-`None`.
- **Corrupt/locked file bypasses (`None`):** pointing the opener at garbage bytes
  — or holding the redb lock on a path with an open store, then opening a second
  `CachingEmbedder` on it — yields `db = None`, not an error. `embed` still returns
  vectors (all misses via the inner embedder), the instance performs **no** cache
  writes, and after a drop/reopen the file is unchanged (nothing was written).
  This pins the *Failure policy* open-failure → `None` case for both corruption
  and the concurrent-lock race.
- **Slashed model id round-trips through the filename:** opening a
  `CachingEmbedder` for an id containing `/` (e.g.
  `sentence-transformers/all-MiniLM-L6-v2`) creates and reopens exactly one file
  (no phantom subdirectory, no failed open → silent cold cache), and two ids that
  differ only in a sanitized character map to *distinct* files — the injectivity
  guarantee from *The model-id guard*.
- **Write failure is swallowed:** a redb `insert` error (e.g. a read-only path)
  does not surface from `embed` — the vector is still returned from the inner
  embedder, matching the *Failure policy* rule that a cache write must never
  abort a run.
- **Hit/miss counters:** over a mix of repeated and fresh texts, the
  `CachingEmbedder`'s hit and miss tallies match the expected split — the numbers
  the effectiveness summary is built from.
- **Value round-trips through raw LE bytes:** a stored vector, read back after a
  drop/reopen, is bit-for-bit the `f32`s that went in — the `to_le_bytes` /
  `chunks_exact` + `from_le_bytes` encoding preserves the vector exactly (the
  byte-identical-vector contract from *Motivation*).
- **Wrong-length entry is treated as a miss:** an entry whose bytes are not
  `EMBED_DIM * 4` long (corruption, or a stale wider/narrower `EMBED_DIM`) does
  not decode to a garbage vector — the length check rejects it and the text is
  recomputed, per *Value encoding* → the length guard.

Use a temp-dir redb path per test so the store is real (exercising the actual
open/insert/get path) but isolated and cleaned up; the inner embedder stays a
stub — already the pattern the trait was built for — so no test touches the real
ONNX model and the tests stay fast.

## Implementation checklist (append to `design.md` §Implementation checklist)

1. Add `redb` (and the chosen hasher, if not `std`) to `Cargo.toml`. No
   serialization crate is needed for the value path — vectors are stored as raw
   little-endian `f32` bytes (see *Value encoding*).
2. Add `/cache/` to `.gitignore`.
3. In `shell/embed.rs`: add `CachingEmbedder<E: Embedder>` implementing
   `Embedder` and holding `db: Option<redb::Database>`, plus an `open(path) ->
   CachingEmbedder` helper that tries to open-or-create the file (or a small
   `cache` sibling module if `embed.rs` grows unwieldy). A clean open is
   `Some(db)`; **any** open failure — locked, corrupt, permission, disk — is
   `None`, logged once, caching off (see *The decorator seam* / *Failure policy*).
   `embed` matches on `db`: `None` → `inner.embed` (bypass); `Some` → get →
   (length-check + decode) → hit, or miss → `inner.embed` + buffer-the-insert.
   Hit/miss counters live on the struct. Values are raw little-endian `f32` bytes — encode with
   `to_le_bytes`, decode with a `bytes.len() == EMBED_DIM * 4` guard then
   `chunks_exact(4)` + `from_le_bytes`, and treat a wrong-length entry as a miss
   (see *Value encoding*); no `serde`/`postcard`. Misses are flushed under a
   single redb `WriteTransaction` (per-run, not per-key — see *Location and
   lifecycle* → Writes); there is no separate `save` step but there **is** a
   run-end flush of buffered inserts.
4. Add the model-id **filename sanitizer** (injective; percent-encode `/` and any
   other path-hostile character — see *The model-id guard*) and derive the cache
   path `cache/embeddings-<sanitized-id>.redb` through it. Guards against a
   HuggingFace-style id (`sentence-transformers/…`) naming a phantom subdirectory
   and silently opening cold every run.
5. In `shell/mod.rs::classify_new`, `shell/mod.rs::train`, and `shell/eval.rs`:
   build a `CachingEmbedder` wrapping `FastEmbedder`, opening the sanitized cache
   path up front. Call sites in the embed loops are unchanged. Log the open and
   the from-cache-vs-regenerated summary per *Logging* above. A cache open failure
   — **including a lock held by a concurrent run** — falls back to running without
   the cache; a cache open or write failure must never abort the run (esp. classify
   tagging), per *Failure policy*.
6. Shell tests per *Testing* above.
7. Update `README.md`: note the embedding cache (all three paths) as
   implemented, and `Taskfile` if a cache-clearing convenience target is wanted
   (`task clean-cache` = `rm -rf cache/`).
