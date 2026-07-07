//! (private) Maildir file → [`RawEmail`]: parse one message file off disk into
//! the seam type the core consumes.
//!
//! notmuch hands the shell *paths* (`search --output=files`); the text itself is
//! read straight from the maildir file and parsed with `mail-parser`. This
//! adapter owns that parse and the text/plain-preferring body extraction — the
//! core never touches a file. Cleaning (quote/signature/HTML stripping) is a core
//! concern (`text::prepare_text`); here we only pull out the raw header/body
//! fields, preferring the text/plain part and falling back to HTML.

use std::fs;
use std::io;
use std::path::Path;

use mail_parser::MessageParser;

use crate::core::RawEmail;

/// Parse the message file at `path` into a [`RawEmail`].
///
/// - `from`   — the raw From header value (display name kept; the core's
///   `registrable_domain` tolerates the wrapper).
/// - `subject`— the Subject header, or empty if absent.
/// - `body`   — the text/plain body if present, else the HTML body (the core
///   strips markup as a fallback), else empty.
/// - `ts`     — the Date header as unix seconds, or 0 if it is missing/unparsable
///   (the core does not use `ts` for v1 features; the shell scopes on notmuch's
///   own `date:` query, so a missing header here is harmless).
///
/// Returns an [`io::Error`] if the file cannot be read, or a
/// [`io::ErrorKind::InvalidData`] error if `mail-parser` cannot parse it.
pub fn parse_file(path: &Path) -> io::Result<RawEmail> {
    parse_file_with_id(path).map(|(email, _id)| email)
}

/// Parse the message file at `path` into a [`RawEmail`] paired with its
/// `Message-ID` header value (stripped of the surrounding angle brackets, if any).
///
/// The message id is what the shell tags against: it is notmuch's own key, so
/// `id:<msgid>` addresses exactly this message when writing a guess. Returns
/// `None` for the id if the header is missing — such a message cannot be tagged
/// by id and the caller skips it (logged) rather than mis-tagging.
pub fn parse_file_with_id(path: &Path) -> io::Result<(RawEmail, Option<String>)> {
    let bytes = fs::read(path)?;
    let message = MessageParser::default()
        .parse(&bytes)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "unparsable message"))?;

    let from = message
        .header_raw("From")
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let subject = message.subject().unwrap_or_default().to_string();
    let body = extract_body(&message);
    let ts = message.date().map(|d| d.to_timestamp()).unwrap_or(0);
    let message_id = message.message_id().map(|s| s.trim().to_string());

    Ok((
        RawEmail {
            from,
            subject,
            body,
            ts,
        },
        message_id,
    ))
}

/// Pull the best-available body text: the text/plain part if the message has one,
/// otherwise the HTML part (returned as-is; the core strips its tags), otherwise
/// an empty string.
fn extract_body(message: &mail_parser::Message) -> String {
    if let Some(text) = message.body_text(0) {
        return text.into_owned();
    }
    if let Some(html) = message.body_html(0) {
        return html.into_owned();
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_tmp(name: &str, contents: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("email_classifier_mailfile_test_{name}.eml"));
        fs::write(&p, contents).unwrap();
        p
    }

    #[test]
    fn parses_headers_and_plain_body() {
        let raw = b"From: Alice <alice@example.com>\r\n\
                    Subject: Lunch?\r\n\
                    Date: Tue, 1 Jul 2025 10:00:00 +0000\r\n\
                    \r\n\
                    Are you free today?\r\n";
        let path = write_tmp("plain", raw);
        let email = parse_file(&path).unwrap();
        assert!(email.from.contains("alice@example.com"));
        assert_eq!(email.subject, "Lunch?");
        assert!(email.body.contains("Are you free today?"));
        assert!(email.ts > 0);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn extracts_message_id_without_brackets() {
        let raw = b"From: Alice <alice@example.com>\r\n\
                    Subject: hi\r\n\
                    Message-ID: <abc123@example.com>\r\n\
                    \r\n\
                    body\r\n";
        let path = write_tmp("msgid", raw);
        let (_email, id) = parse_file_with_id(&path).unwrap();
        assert_eq!(id.as_deref(), Some("abc123@example.com"));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn missing_message_id_is_none() {
        let raw = b"From: a@example.com\r\n\
                    Subject: no id\r\n\
                    \r\n\
                    body\r\n";
        let path = write_tmp("no_msgid", raw);
        let (_email, id) = parse_file_with_id(&path).unwrap();
        assert_eq!(id, None);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn prefers_text_plain_over_html() {
        let raw = b"From: bob@example.org\r\n\
                    Subject: multipart\r\n\
                    MIME-Version: 1.0\r\n\
                    Content-Type: multipart/alternative; boundary=\"b\"\r\n\
                    \r\n\
                    --b\r\n\
                    Content-Type: text/plain\r\n\
                    \r\n\
                    plain wins\r\n\
                    --b\r\n\
                    Content-Type: text/html\r\n\
                    \r\n\
                    <p>html loses</p>\r\n\
                    --b--\r\n";
        let path = write_tmp("multipart", raw);
        let email = parse_file(&path).unwrap();
        assert!(email.body.contains("plain wins"));
        assert!(!email.body.contains("html loses"));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn falls_back_to_html_when_no_plain_part() {
        let raw = b"From: c@example.net\r\n\
                    Subject: html only\r\n\
                    Content-Type: text/html\r\n\
                    \r\n\
                    <p>hello</p>\r\n";
        let path = write_tmp("html_only", raw);
        let email = parse_file(&path).unwrap();
        assert!(email.body.contains("hello"));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn missing_subject_and_date_are_defaulted() {
        let raw = b"From: d@example.com\r\n\
                    \r\n\
                    body\r\n";
        let path = write_tmp("no_subject", raw);
        let email = parse_file(&path).unwrap();
        assert_eq!(email.subject, "");
        assert_eq!(email.ts, 0);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn unreadable_file_is_io_error() {
        let path = std::env::temp_dir().join("email_classifier_mailfile_test_missing.eml");
        let _ = fs::remove_file(&path);
        assert!(parse_file(&path).is_err());
    }
}
