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

- `src/core/` — pure functions of already-gathered data, unit-tested, no IO.
- `src/shell/` — all IO, caching, and the linfa solver: notmuch queries, the
  embedder, mail parsing, persistence.
- `models/model.json` — the single serialized model (gitignored, regenerable).

### Predictor variables (392 features, all ≈[0,1] by construction)

- **Text embedding (384):** the all-MiniLM-L6-v2 unit vector of the prepared
  message text (subject first, then body with quoted replies, signatures, and
  HTML stripped).
- **Sender-domain history (4):** over the sender's registrable domain (eTLD+1),
  three smoothed class proportions plus one confidence scalar (see below).
- **Sender-address history (4):** the same four numbers for the exact sender
  address — sharp where the domain is not (e.g. a few key people on gmail.com).

#### What the four history numbers mean

For a sender, the three proportions are smoothed estimates of P(p1), P(p2),
P(p3) — how that sender's past confirmed emails were labeled. With `n_i` emails
seen in class `i`, prior `π` (uniform `⅓` today), and smoothing `alpha` (1.0):

```
P(pᵢ | sender) = (nᵢ + alpha·πᵢ) / (N + alpha)      N = Σ nⱼ    (the three sum to 1)
```

This is Dirichlet (Laplace) smoothing, not the raw ratio `nᵢ/N`, so a
never-seen sender returns the prior (`0.33, 0.33, 0.33`) rather than a confident
guess, and sparse history shrinks toward it.

Because the smoothing pulls toward the prior, **the amount of history already
leaks a little into the proportions** — "1 email, all p3" and "500 emails, all
p3" do *not* give the same P(p3):

```
N=1,   all p3:  P(p3) = (1 + 1·⅓) / (1 + 1)     ≈ 0.67   (prior still has real pull)
N=500, all p3:  P(p3) = (500 + 1·⅓) / (500 + 1) ≈ 0.998  (prior washed out)
```

But that in-proportion signal is weak and saturates fast, so the fourth number
makes "how much history" explicit: a confidence scalar
`min(1, ln(1+N) / ln(1000))` — 0 at no history, rising with the total count,
capped at 1 (reached at N=999). It lets the regression lean on a sender's
history only when there is enough of it, and fall back to the text embedding
otherwise.

All history counts come only from *confirmed* labels
(`not (tag:auto and tag:unread)`), so the model never trains on its own
unreviewed guesses. (v1 uses final tag counts rather than counts as of each
email's arrival — a known leak that inflates exactly these proportions; see
`design.md` → *Features* and *Training-time leak note*.)

For the current build status and the ordered implementation checklist, see
`design.md`.

## Deployment (notmuch post-new hook)

The classifier runs from notmuch's **post-new hook** — no daemon, no cron. After
every `notmuch new` (each mbsync cycle) the hook invokes `classify`, which tags
in-scope new mail with `prio-*` + `auto`.

The hook script lives in the dotfiles repo and is symlinked into the maildir's
notmuch hook directory:

```
~/Documents/dotphiles/email/notmuch-hooks/post-new   # source (edit here)
  → symlinked to →
~/Mail/.notmuch/hooks/post-new                        # where notmuch looks
```

`notmuch new` runs the hook with the **maildir as cwd and a minimal `PATH`**, so
the classifier stanza uses absolute paths for both the binary and the model and
does not rely on `models/model.json` resolving relative to cwd:

```sh
classifier=~/Documents/scripts/email_classifier/target/release/email_classifier
model=~/Documents/scripts/email_classifier/models/model.json
if [ -x "$classifier" ]; then
    "$classifier" classify --model "$model" || echo "post-new: email_classifier failed (non-fatal)" >&2
fi
```

The call is **non-fatal**: a classifier failure logs to stderr but does not abort
`notmuch new`, so mail tagging is never blocked by a bad model or a missing
binary. The stanza runs *after* the account/spam/sent tagging in the same hook,
so the classifier's date/tag scope filter sees accurate tags.

To (re)deploy: `task build` (produces the release binary the hook points at) and
`task train` (writes `models/model.json`). The hook picks up the new binary and
model on the next `notmuch new` — nothing else to restart.
