//! (private) Registrable-domain (eTLD+1) extraction from a From address, via the
//! Public Suffix List.

/// Extract the registrable domain (eTLD+1) from a From address.
///
/// Accepts a bare address (`alice@mail.example.co.uk`) or a display-name form
/// (`Alice <alice@example.com>`); anything after the last `@` up to a closing
/// `>` is treated as the host. Returns lowercased eTLD+1, or `None` when there
/// is no host or no registrable domain can be determined.
pub fn registrable_domain(from_addr: &str) -> Option<String> {
    let host = extract_host(from_addr)?;
    // psl operates on bytes; feed it the lowercased host.
    let host_lc = host.to_ascii_lowercase();
    let domain = psl::domain(host_lc.as_bytes())?;
    // psl returns the registrable domain (suffix + one label). Re-borrow as str.
    std::str::from_utf8(domain.as_bytes())
        .ok()
        .map(|s| s.to_string())
}

/// Pull the host portion out of a From value, tolerating display-name wrappers
/// and trailing `>`. Returns `None` if there is no `@` or the host is empty.
fn extract_host(from_addr: &str) -> Option<&str> {
    let after_at = from_addr.rsplit_once('@')?.1;
    // Strip an angle-bracket close and any surrounding whitespace.
    let host = after_at
        .trim()
        .trim_end_matches('>')
        .trim();
    if host.is_empty() {
        None
    } else {
        Some(host)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_gmail() {
        assert_eq!(
            registrable_domain("alice@gmail.com").as_deref(),
            Some("gmail.com")
        );
    }

    #[test]
    fn subdomain_collapses_to_etld_plus_one() {
        assert_eq!(
            registrable_domain("bob@mail.corp.example.com").as_deref(),
            Some("example.com")
        );
    }

    #[test]
    fn multi_part_tld_co_uk() {
        assert_eq!(
            registrable_domain("carol@news.example.co.uk").as_deref(),
            Some("example.co.uk")
        );
    }

    #[test]
    fn display_name_form() {
        assert_eq!(
            registrable_domain("Alice Example <alice@example.com>").as_deref(),
            Some("example.com")
        );
    }

    #[test]
    fn case_is_normalized() {
        assert_eq!(
            registrable_domain("Alice@Example.COM").as_deref(),
            Some("example.com")
        );
    }

    #[test]
    fn invalid_no_at() {
        assert_eq!(registrable_domain("not-an-address"), None);
    }

    #[test]
    fn invalid_empty_host() {
        assert_eq!(registrable_domain("alice@"), None);
        assert_eq!(registrable_domain("alice@>"), None);
    }

    #[test]
    fn empty_input() {
        assert_eq!(registrable_domain(""), None);
    }
}
