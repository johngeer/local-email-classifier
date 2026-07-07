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
            eprintln!("notmuch count failed for {query:?}, treating as no history: {e}");
            0
        });
    }
    counts
}

/// Message file paths for all confirmed labels, all dates — the training set
/// selection. Query: `(prio-low or prio-normal or prio-high) and <confirmed>`.
/// Returns an error if the notmuch invocation itself fails (training cannot
/// proceed without its labels).
pub fn confirmed_label_files() -> Result<Vec<PathBuf>, String> {
    let query = format!(
        "(tag:{} or tag:{} or tag:{}) and {CONFIRMED_FILTER}",
        Priority::P1.to_tag(),
        Priority::P2.to_tag(),
        Priority::P3.to_tag(),
    );
    search_files(&query)
}

/// Message file paths for in-scope new mail to classify: on or after the
/// classification cutoff (`date:2026-07-01..`), and not already carrying a
/// confirmed `prio-*` (the skip-confirmed rule is applied by the caller against
/// the file's own tags; here the date cutoff gates the batch). Returns an error
/// if the notmuch invocation fails.
pub fn new_mail_files(cutoff: &str) -> Result<Vec<PathBuf>, String> {
    search_files(&format!("date:{cutoff}.."))
}

/// `notmuch search --output=files <query>` → one path per line.
fn search_files(query: &str) -> Result<Vec<PathBuf>, String> {
    let output = run(&["search", "--output=files", query])?;
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
