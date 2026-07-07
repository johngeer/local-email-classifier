//! (private) The notmuch adapter: file selection, per-sender history counts, and
//! the two `HashMap` caches in front of the count queries.
//!
//! notmuch is the API for *tags, queries, and triggering* (the message text is
//! read straight off disk by `mailfile`). This module shells out to the `notmuch`
//! CLI — the crate is not a build dependency — and owns two invariants from the
//! design:
//!
//! - **The confirmed-label filter is carried here.** Every history count and the
//!   training file selection AND-in `not (tag:auto and tag:unread)`, so sender
//!   proportions are never built from the model's own unreviewed guesses (which
//!   would feed its bias back into its features). It lives in one place —
//!   [`CONFIRMED_FILTER`] — so it cannot drift between call sites.
//! - **Count-query failures are non-fatal.** A count that errors is treated as
//!   *unknown history* — [`ZERO_COUNTS`], which the core maps to the prior with
//!   zero confidence — logged once, never aborting a batch. (An *embedding*
//!   failure aborts; a bad sender count does not.)
//!
//! A `HashMap<String, ClassCounts>` sits in front of the counts (one map for
//! domains, one for addresses): the first lookup of a sender runs three count
//! queries and caches the result; repeated senders in the same run are free. The
//! cache is per-process — corrections (retags) are picked up on the next run.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

use crate::core::{ClassCounts, Priority, ZERO_COUNTS};

/// The confirmed-label filter, AND-ed into every history count and the training
/// selection. Excludes only unreviewed model guesses (`auto and unread`); hand
/// labels, corrections, and read-and-agreed guesses all remain. Mandatory —
/// see the module docs and `design.md` → *Scope* / *History counts*.
pub const CONFIRMED_FILTER: &str = "not (tag:auto and tag:unread)";

/// Cached, memoized access to notmuch history counts for one run.
///
/// Holds the two per-sender caches (domains, addresses) and turns effectful,
/// order-dependent count lookups into the plain [`ClassCounts`] the core
/// consumes. Construct once per `train`/`classify_new` invocation.
pub struct Notmuch {
    domain_counts: HashMap<String, ClassCounts>,
    addr_counts: HashMap<String, ClassCounts>,
}

impl Notmuch {
    /// A fresh adapter with empty caches.
    pub fn new() -> Notmuch {
        Notmuch {
            domain_counts: HashMap::new(),
            addr_counts: HashMap::new(),
        }
    }

    /// Confirmed-label class counts for one registrable domain, memoized. Matches
    /// senders on that domain with `from:*@<domain>`. A failed query falls back to
    /// [`ZERO_COUNTS`] (logged once); the value is cached either way so a bad
    /// sender is not re-queried within the run.
    pub fn domain_counts(&mut self, domain: &str) -> ClassCounts {
        if let Some(c) = self.domain_counts.get(domain) {
            return *c;
        }
        let counts = counts_for_from(&format!("*@{domain}"));
        self.domain_counts.insert(domain.to_string(), counts);
        counts
    }

    /// Confirmed-label class counts for one exact address, memoized. Same
    /// fallback and caching behaviour as [`domain_counts`](Self::domain_counts).
    pub fn addr_counts(&mut self, addr: &str) -> ClassCounts {
        if let Some(c) = self.addr_counts.get(addr) {
            return *c;
        }
        let counts = counts_for_from(addr);
        self.addr_counts.insert(addr.to_string(), counts);
        counts
    }
}

/// The three confirmed-label counts for a `from:` pattern, one query per class.
/// Any query error degrades that class to 0 (logged once) rather than aborting —
/// unknown history, not a crash.
fn counts_for_from(from_pattern: &str) -> ClassCounts {
    let mut counts = ZERO_COUNTS;
    for p in Priority::ALL {
        let query = format!(
            "tag:{} and {CONFIRMED_FILTER} and from:{from_pattern}",
            p.to_tag()
        );
        counts[p.to_index()] = count(&query).unwrap_or_else(|e| {
            log!("notmuch count failed for {query:?}, treating as no history: {e}");
            0
        });
    }
    counts
}

/// Labeled message file paths for the whole confirmed set, all dates — the
/// training selection. One `search --output=files --duplicate=1` per priority, so
/// each file's label comes from *which* query returned it rather than a separate
/// per-file tag read, and each logical message contributes exactly one training
/// example (see [`search_files`] on the cross-account dedup). Query per level:
/// `tag:prio-<level> and <confirmed>`. Returns an error if any notmuch invocation
/// fails (training cannot proceed without its labels).
///
/// A message carrying more than one `prio-*` tag (which the mutually-exclusive
/// namespace should prevent) would appear under multiple levels; that is an
/// upstream tagging bug, and duplicating it across labels is a faithful, harmless
/// reflection of the actual tags rather than something to silently paper over.
pub fn confirmed_label_files() -> Result<Vec<(PathBuf, Priority)>, String> {
    let mut labeled = Vec::new();
    for p in Priority::ALL {
        let query = format!("tag:{} and {CONFIRMED_FILTER}", p.to_tag());
        for path in search_files(&query)? {
            labeled.push((path, p));
        }
    }
    Ok(labeled)
}

/// Message file paths for in-scope new mail to classify: on or after the
/// classification cutoff (`date:2026-07-01..`) and not already confirmed. The
/// skip-confirmed rule is folded into the query itself — a message carrying a
/// `prio-*` tag *without* `auto` is human-confirmed and must never be re-guessed,
/// so it is excluded here. Unlabeled mail and stale `auto` guesses remain in
/// scope. Returns an error if the notmuch invocation fails.
pub fn new_mail_files(cutoff: &str) -> Result<Vec<PathBuf>, String> {
    search_files(&new_mail_query(cutoff))
}

/// The in-scope-new-mail selection query. Pure (no IO) so its shape — the date
/// cutoff AND the skip-confirmed exclusion — is unit-testable without notmuch.
fn new_mail_query(cutoff: &str) -> String {
    let confirmed = format!(
        "(tag:{} or tag:{} or tag:{}) and not tag:auto",
        Priority::P1.to_tag(),
        Priority::P2.to_tag(),
        Priority::P3.to_tag(),
    );
    format!("date:{cutoff}.. and not ({confirmed})")
}

/// Write a fresh guess onto one message, addressed by its notmuch message id:
/// set the given priority tag, add `auto` (the unconfirmed marker), and clear the
/// other two priority tags so the `prio-*` namespace stays mutually exclusive.
///
/// The message id is notmuch's own key (the `Message-ID` header), so `id:<msgid>`
/// targets exactly the message the shell parsed. Returns an error if the tag
/// invocation fails — the caller logs it and moves on rather than aborting the
/// batch.
pub fn write_guess(message_id: &str, priority: Priority) -> Result<(), String> {
    let tags = guess_tag_ops(priority);
    let query = format!("id:{message_id}");
    let mut args: Vec<&str> = vec!["tag"];
    args.extend(tags.iter().map(String::as_str));
    args.push("--");
    args.push(&query);
    run(&args).map(|_| ())
}

/// The `+/-` tag operations for a fresh guess of `priority`: add its tag and
/// `auto`, remove the other two priority tags. Pure so the mutual-exclusion
/// invariant is unit-testable without notmuch.
fn guess_tag_ops(priority: Priority) -> Vec<String> {
    let mut tags = vec![format!("+{}", priority.to_tag()), "+auto".to_string()];
    for other in Priority::ALL {
        if other != priority {
            tags.push(format!("-{}", other.to_tag()));
        }
    }
    tags
}

/// `notmuch search --output=files --duplicate=1 <query>` → one path per line.
///
/// `--duplicate=1` yields exactly one representative file per *message*, not one
/// per maildir file. One logical message forwarded across accounts is indexed
/// under a single Message-ID but lands in several maildirs; without this flag it
/// would appear once per copy, over-weighting cross-account mail in the training
/// set (785 messages read as 1211 files on the current archive). notmuch's own
/// identity does the dedup — we never collapse paths ourselves.
fn search_files(query: &str) -> Result<Vec<PathBuf>, String> {
    let output = run(&["search", "--output=files", "--duplicate=1", query])?;
    Ok(output
        .lines()
        .filter(|l| !l.is_empty())
        .map(PathBuf::from)
        .collect())
}

/// `notmuch count <query>` → the message count.
fn count(query: &str) -> Result<u32, String> {
    let output = run(&["count", query])?;
    output
        .trim()
        .parse::<u32>()
        .map_err(|e| format!("unparsable count {output:?}: {e}"))
}

/// Run a notmuch subcommand and return its stdout as a string. Errors carry the
/// subcommand and stderr so a failing query is diagnosable.
fn run(args: &[&str]) -> Result<String, String> {
    let output = Command::new("notmuch")
        .args(args)
        .output()
        .map_err(|e| format!("spawning notmuch {args:?}: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "notmuch {args:?} exited {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map_err(|e| format!("notmuch {args:?} produced non-utf8 output: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confirmed_filter_excludes_unreviewed_guesses() {
        // The filter must exclude exactly `auto and unread` and nothing else.
        assert_eq!(CONFIRMED_FILTER, "not (tag:auto and tag:unread)");
    }

    #[test]
    fn count_query_carries_tag_filter_and_sender() {
        // Reconstruct the query the way `counts_for_from` does, to pin its shape
        // without touching notmuch. (The private format is the invariant we care
        // about: every count is confirmed-filtered and sender-scoped.)
        let p = Priority::P3;
        let query = format!(
            "tag:{} and {CONFIRMED_FILTER} and from:{}",
            p.to_tag(),
            "alice@example.com"
        );
        assert_eq!(
            query,
            "tag:prio-high and not (tag:auto and tag:unread) and from:alice@example.com"
        );
    }

    #[test]
    fn new_mail_query_gates_by_date_and_skips_confirmed() {
        // In scope: on/after the cutoff. Out of scope: a confirmed prio-* (a
        // priority tag without `auto`) — those are never re-guessed. A stale
        // `auto` guess stays in scope (it is not excluded by `not tag:auto`).
        assert_eq!(
            new_mail_query("2026-07-01"),
            "date:2026-07-01.. and not ((tag:prio-low or tag:prio-normal or tag:prio-high) \
             and not tag:auto)"
        );
    }

    #[test]
    fn guess_sets_one_prio_plus_auto_and_clears_the_others() {
        // A P3 guess: +prio-high +auto, and both other prio tags removed so the
        // namespace stays mutually exclusive.
        assert_eq!(
            guess_tag_ops(Priority::P3),
            vec!["+prio-high", "+auto", "-prio-low", "-prio-normal"]
        );
        assert_eq!(
            guess_tag_ops(Priority::P1),
            vec!["+prio-low", "+auto", "-prio-normal", "-prio-high"]
        );
    }

    #[test]
    fn caches_are_independent_and_memoize() {
        // Without a notmuch database we cannot exercise a real count, but we can
        // verify the cache short-circuits: pre-seed a value and read it back
        // without a query (a query with no DB would error and cache ZERO).
        let mut nm = Notmuch::new();
        nm.domain_counts.insert("example.com".to_string(), [1, 2, 3]);
        nm.addr_counts.insert("a@example.com".to_string(), [4, 5, 6]);
        assert_eq!(nm.domain_counts("example.com"), [1, 2, 3]);
        assert_eq!(nm.addr_counts("a@example.com"), [4, 5, 6]);
    }
}
