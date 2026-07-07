//! THE core interface (pure functions only).
//!
//! ENFORCED BOUNDARY: this module and everything under `core/` must have **no**
//! `use` of `notmuch`, `fastembed`, or `std::fs`. The core is a pure function of
//! data the shell has already gathered. The dependency is one-directional:
//! `shell` depends on `core`, never the reverse.
//!
//! Checklist status: §1 leaves (`labels`, `domain`, `history`, `text`) are in
//! place. `features`/`model` and the two public entry points (`classify`,
//! `features_for`) land in §2 — hence the leaf modules below are declared but
//! the composed interface is not re-exported yet.

pub mod labels;

// §1 leaves. Private (`mod`, not `pub mod`): they exist for isolation and
// testing, not as public surface. `history` types are re-exported below since
// they are seam types the shell fills.
mod domain;
mod history;
mod text;

pub use history::{ClassCounts, ZERO as ZERO_COUNTS};
pub use labels::Priority;

/// The seam type the shell hands to the core: one parsed email, already read off
/// disk. Defined here because the core defines what it consumes; the shell's
/// `mailfile` adapter produces it.
#[derive(Debug, Clone)]
pub struct RawEmail {
    /// Raw From header value (may include a display name).
    pub from: String,
    /// Subject header, or empty if absent.
    pub subject: String,
    /// Best-available body text (text/plain part preferred by the shell).
    pub body: String,
    /// Arrival timestamp, unix seconds. Used by the shell for time-based
    /// scoping and splits; the core does not depend on it for v1 features.
    pub ts: i64,
}
