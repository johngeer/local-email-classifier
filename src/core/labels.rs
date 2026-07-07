//! The shared vocabulary: the `Priority` enum and the single source of truth for
//! the tag <-> priority mapping. The `prio-*` tag strings appear nowhere else in
//! the core.
//!
//! Mapping (higher = more important):
//!   P1 = "prio-low"    (usize 0)
//!   P2 = "prio-normal" (usize 1)
//!   P3 = "prio-high"   (usize 2)
//!
//! The flat "auto" marker is managed by the shell, not modeled here — it is not
//! a `Priority`.

use serde::{Deserialize, Serialize};

/// Ordered email priority: `P1 < P2 < P3`, with P3 the most important.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Priority {
    P1,
    P2,
    P3,
}

impl Priority {
    /// All priorities in class-index order. Handy for iterating the three
    /// classes without hard-coding the count.
    pub const ALL: [Priority; 3] = [Priority::P1, Priority::P2, Priority::P3];

    /// Class index used to position this priority in count/probability arrays.
    pub fn to_index(self) -> usize {
        match self {
            Priority::P1 => 0,
            Priority::P2 => 1,
            Priority::P3 => 2,
        }
    }

    /// Inverse of [`to_index`]. Returns `None` for out-of-range indices.
    pub fn from_index(i: usize) -> Option<Priority> {
        match i {
            0 => Some(Priority::P1),
            1 => Some(Priority::P2),
            2 => Some(Priority::P3),
            _ => None,
        }
    }

    /// The notmuch priority tag for this priority. This is the *only* place the
    /// tag strings are written.
    pub fn to_tag(self) -> &'static str {
        match self {
            Priority::P1 => "prio-low",
            Priority::P2 => "prio-normal",
            Priority::P3 => "prio-high",
        }
    }

    /// Parse a notmuch priority tag back into a `Priority`. Returns `None` for
    /// any string that is not one of the three priority tags.
    pub fn from_tag(tag: &str) -> Option<Priority> {
        match tag {
            "prio-low" => Some(Priority::P1),
            "prio-normal" => Some(Priority::P2),
            "prio-high" => Some(Priority::P3),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_round_trips() {
        for p in Priority::ALL {
            assert_eq!(Priority::from_index(p.to_index()), Some(p));
        }
    }

    #[test]
    fn index_order_matches_priority_order() {
        // P1 < P2 < P3 must line up with 0 < 1 < 2.
        assert_eq!(Priority::P1.to_index(), 0);
        assert_eq!(Priority::P2.to_index(), 1);
        assert_eq!(Priority::P3.to_index(), 2);
    }

    #[test]
    fn from_index_rejects_out_of_range() {
        assert_eq!(Priority::from_index(3), None);
        assert_eq!(Priority::from_index(usize::MAX), None);
    }

    #[test]
    fn tag_round_trips() {
        for p in Priority::ALL {
            assert_eq!(Priority::from_tag(p.to_tag()), Some(p));
        }
    }

    #[test]
    fn tag_mapping_is_exact() {
        assert_eq!(Priority::P1.to_tag(), "prio-low");
        assert_eq!(Priority::P2.to_tag(), "prio-normal");
        assert_eq!(Priority::P3.to_tag(), "prio-high");
    }

    #[test]
    fn from_tag_rejects_unknown() {
        assert_eq!(Priority::from_tag("auto"), None);
        assert_eq!(Priority::from_tag("prio-high "), None);
        assert_eq!(Priority::from_tag(""), None);
        assert_eq!(Priority::from_tag("inbox"), None);
    }
}
