//! (private) `prepare_text`: turn a subject + body into the single string handed
//! to the embedder.
//!
//! Subject goes first (then `\n`, then the cleaned body) so the densest signal
//! survives the model's 512-token truncation. Cleaning drops quoted reply
//! chains and signatures and strips HTML when the body looks like markup, then
//! truncates to a character budget.

/// Character budget for the assembled text. The embedder truncates at 512
/// tokens anyway (~2k chars of English); subject-first guarantees the most
/// informative text is inside the budget.
const CHAR_BUDGET: usize = 2000;

/// Assemble the text handed to the embedder: `subject`, a newline, then the
/// cleaned `body`, truncated to [`CHAR_BUDGET`] characters.
///
/// `body` is expected to already be the text/plain part when one exists; if it
/// still looks like HTML (no text/plain part was available) tags are stripped
/// here as a fallback.
pub fn prepare_text(subject: &str, body: &str) -> String {
    let cleaned_body = clean_body(body);
    let mut out = String::with_capacity(subject.len() + 1 + cleaned_body.len());
    out.push_str(subject.trim());
    out.push('\n');
    out.push_str(&cleaned_body);
    truncate_chars(out.trim(), CHAR_BUDGET)
}

/// Strip HTML if the body looks like markup, then drop quoted reply chains and a
/// trailing signature.
fn clean_body(body: &str) -> String {
    let text = if looks_like_html(body) {
        strip_html(body)
    } else {
        body.to_string()
    };
    let text = drop_signature(&text);
    drop_quoted_lines(&text)
}

/// Heuristic: treat the body as HTML if it contains a tell-tale tag. Cheap and
/// good enough as a fallback for when no text/plain part existed.
fn looks_like_html(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("<html")
        || lower.contains("<body")
        || lower.contains("<div")
        || lower.contains("<p>")
        || lower.contains("<br")
        || lower.contains("<table")
}

/// Remove HTML tags, collapsing them to spaces so words don't fuse, then
/// collapse runs of whitespace.
fn strip_html(body: &str) -> String {
    let mut out = String::with_capacity(body.len());
    let mut in_tag = false;
    for ch in body.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                out.push(' ');
            }
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    collapse_whitespace(&out)
}

/// Collapse runs of whitespace into single spaces, preserving newlines is not
/// needed here since HTML lost its line structure anyway.
fn collapse_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Drop everything from the signature delimiter line (`-- ` on its own line)
/// onward. Only the first such delimiter matters.
fn drop_signature(text: &str) -> String {
    let mut kept: Vec<&str> = Vec::new();
    for line in text.lines() {
        // The RFC 3676 sig delimiter is exactly "-- " (dash dash space).
        if line == "-- " || line.trim_end() == "--" && line.starts_with("--") {
            break;
        }
        kept.push(line);
    }
    kept.join("\n")
}

/// Drop quoted reply material: lines beginning with `>` and the `On … wrote:`
/// attribution line that introduces them.
fn drop_quoted_lines(text: &str) -> String {
    let kept: Vec<&str> = text
        .lines()
        .filter(|line| {
            let t = line.trim_start();
            if t.starts_with('>') {
                return false;
            }
            // "On <date>, <name> wrote:" attribution line.
            if t.starts_with("On ") && t.trim_end().ends_with("wrote:") {
                return false;
            }
            true
        })
        .collect();
    kept.join("\n").trim().to_string()
}

/// Truncate to at most `budget` characters (not bytes), so multibyte input is
/// never split mid-character.
fn truncate_chars(s: &str, budget: usize) -> String {
    match s.char_indices().nth(budget) {
        Some((byte_idx, _)) => s[..byte_idx].to_string(),
        None => s.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subject_is_always_first() {
        let out = prepare_text("Quarterly report", "Body text here.");
        assert!(out.starts_with("Quarterly report\n"));
    }

    #[test]
    fn quoted_reply_block_removed() {
        let body = "Sounds good, thanks.\n\nOn Mon, Jan 1, Alice wrote:\n> the original\n> message here";
        let out = prepare_text("Re: plan", body);
        assert!(out.contains("Sounds good"));
        assert!(!out.contains("original"));
        assert!(!out.contains("Alice wrote"));
    }

    #[test]
    fn signature_after_dash_dash_removed() {
        let body = "Please review.\n-- \nJohn Geer\nSenior Engineer\njohn@example.com";
        let out = prepare_text("Review", body);
        assert!(out.contains("Please review"));
        assert!(!out.contains("Senior Engineer"));
        assert!(!out.contains("john@example.com"));
    }

    #[test]
    fn html_is_stripped_as_fallback() {
        let body = "<html><body><p>Hello <b>world</b></p></body></html>";
        let out = prepare_text("Hi", body);
        assert!(out.contains("Hello"));
        assert!(out.contains("world"));
        assert!(!out.contains('<'));
        assert!(!out.contains('>'));
    }

    #[test]
    fn plain_text_with_angle_brackets_is_not_mangled() {
        // No HTML tell-tales, so the body is left alone (aside from quote rules).
        let body = "Use a < b to compare values.";
        let out = prepare_text("Math", body);
        assert!(out.contains("Use a < b to compare"));
    }

    #[test]
    fn char_budget_respected() {
        let long_body = "x".repeat(5000);
        let out = prepare_text("Subj", &long_body);
        assert!(out.chars().count() <= 2000, "len = {}", out.chars().count());
    }

    #[test]
    fn char_budget_counts_chars_not_bytes() {
        // Multibyte chars must not be split and must be counted as one each.
        let long_body = "é".repeat(5000);
        let out = prepare_text("S", &long_body);
        assert!(out.chars().count() <= 2000);
        // Still valid UTF-8 (String guarantees this; a bad split would panic).
        assert!(out.chars().all(|c| c == 'é' || c == 'S' || c == '\n'));
    }

    #[test]
    fn empty_body_is_safe() {
        let out = prepare_text("Only a subject", "");
        assert_eq!(out, "Only a subject");
    }

    #[test]
    fn empty_subject_and_body_is_safe() {
        let out = prepare_text("", "");
        assert_eq!(out, "");
    }
}
