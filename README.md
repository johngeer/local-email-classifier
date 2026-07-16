# email_classifier

A local-only email priority classifier. It assigns incoming mail one of three
ordered priorities and applies them as notmuch tags:

| Priority | Tag           | Meaning         |
|----------|---------------|-----------------|
| P1       | `prio-low`    | least important |
| P2       | `prio-normal` |                 |
| P3       | `prio-high`   | most important  |

Classification uses multinomial logistic regression over a text embedding
(all-MiniLM-L6-v2, 384-dim) plus smoothed per-domain and per-address tag-history
proportions. This uses 392 total features, all between [0,1]. 

The whole interface is notmuch tags; the model runs from notmuch's post-new hook,
with no daemon and no separate database. 

See `docs/architecture.md` for the high-level map, and
`docs/designs/done/design.md` for the original specification and rationale.

## Quickstart

### Install

Download the git repository and make sure it has access to notmuch (local email index).

### Building and testing

This project uses [Task](https://taskfile.dev) (go-task, like make) to run commands.

To train the model on your email, run:

```
task train
```

## Performance

For such a simple model, this does a good job. I find it works better than most email-service classifiers.

I suspect one big advantage it has is the explicitly labeled emails it uses for training.

It is pretty fast. However, building the embeddings for the training data is the primary bottleneck. To address this, a persistent embedding cache (redb key-value store) is implemented, which reuses the vectors from earlier runs. Subsequent runs with a warm cache skip re-embedding.

Here is an example training run on my laptop with a cold cache:

```
[+  0.000 Δ 0.000] training over confirmed labels → models/model.json
[+  0.714 Δ 0.714] confirmed labels: 1451 total  (170 prio-low, 529 prio-normal, 752 prio-high)
[+  0.716 Δ 0.002] embedding 1451 message(s)…
[+ 24.549 Δ23.833]   embedded 100/1451…
[+ 33.661 Δ 9.112]   embedded 200/1451…
[+ 41.230 Δ 7.568]   embedded 300/1451…
[+ 48.231 Δ 7.002]   embedded 400/1451…
[+ 54.910 Δ 6.678]   embedded 500/1451…
[+ 62.910 Δ 8.001]   embedded 600/1451…
[+ 70.178 Δ 7.268]   embedded 700/1451…
[+ 75.445 Δ 5.267]   embedded 800/1451…
[+ 80.652 Δ 5.208]   embedded 900/1451…
[+ 85.459 Δ 4.807]   embedded 1000/1451…
[+ 90.855 Δ 5.396]   embedded 1100/1451…
[+ 97.103 Δ 6.247]   embedded 1200/1451…
[+101.512 Δ 4.410]   embedded 1300/1451…
[+106.687 Δ 5.175]   embedded 1400/1451…
[+108.450 Δ 1.763] fitting multinomial logistic regression on 1451 example(s)…
[+108.664 Δ 0.214] trained on 1451 example(s), saved models/model.json
```

## Architecture

Functional core / imperative shell. See `docs/architecture.md` for the
high-level map (module layout, the two interfaces, data flow, and invariants) and
`CLAUDE.md` for coding guidance; `docs/designs/done/design.md` is the original
pre-implementation spec and rationale.

- `src/core/`: pure functions of already-gathered data, unit-tested, no IO.
- `src/shell/`: all IO, caching, and the linfa solver: notmuch queries, the
  embedder, mail parsing, persistence.
- `models/model.json`: the single serialized model (gitignored, regenerable).
- `cache/`: the persistent embedding cache (redb key-value store, gitignored, regenerable).

### Predictor variables

- **Text embedding (384):** the all-MiniLM-L6-v2 unit vector of the prepared
  message text (subject first, then body with quoted replies, signatures, and
  HTML stripped).
- **Sender-domain history (4):** over the sender's registrable domain (eTLD+1),
  three smoothed class proportions plus one confidence scalar (see below).
- **Sender-address history (4):** the same four numbers for the exact sender
  address. It is sharp where the domain is not (e.g. a few key people on gmail.com).

#### What the four history numbers mean

For a sender, the three proportions are smoothed estimates of P(p1), P(p2),
P(p3), i.e. how that sender's past confirmed emails were labeled. With `n_i` emails
seen in class `i`, prior `π` (uniform `⅓` today), and smoothing `alpha` (1.0):

```
P(pᵢ | sender) = (nᵢ + alpha·πᵢ) / (N + alpha)      N = Σ nⱼ    (the three sum to 1)
```

This is Dirichlet (Laplace) smoothing, not the raw ratio `nᵢ/N`, so a
never-seen sender returns the prior (`0.33, 0.33, 0.33`) rather than a confident
guess, and sparse history shrinks toward it.

Because the smoothing pulls toward the prior, **the amount of history
leaks a little into the proportions**: "1 email, all p3" and "500 emails, all
p3" do *not* give the same P(p3):

```
N=1,   all p3:  P(p3) = (1 + 1·⅓) / (1 + 1)     ≈ 0.67   (prior still has real pull)
N=500, all p3:  P(p3) = (500 + 1·⅓) / (500 + 1) ≈ 0.998  (prior washed out)
```

But that in-proportion signal is weak and saturates fast, so the fourth number
makes "how much history" explicit: a confidence scalar
`min(1, ln(1+N) / ln(1000))`: 0 at no history, rising with the total count,
capped at 1 (reached at N=999). It lets the regression lean on a sender's
history only when there is enough of it, and fall back to the text embedding
otherwise.

All history counts come only from *confirmed* labels
(`not (tag:auto and tag:unread)`), so the model never trains on its own
unreviewed guesses.

## Deployment (notmuch post-new hook)

The classifier runs from notmuch's **post-new hook**: no daemon, no cron. After
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
model on the next `notmuch new`. Nothing to restart.
