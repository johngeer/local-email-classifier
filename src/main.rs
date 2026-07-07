//! arg parse → dispatch `train` / `classify`. Only calls the two shell entry
//! points ([`shell::train`], [`shell::classify_new`]); all IO lives behind them.
//!
//! Usage:
//!   email_classifier train     — fit a fresh model over every confirmed label
//!   email_classifier classify  — the post-new hook path: guess in-scope new mail
//!
//! The single model file is `models/model.json` (design → *Persistence*);
//! `--model <path>` overrides it. Exit status is 0 on success, 1 on error so the
//! post-new hook surfaces a failed run.

#[macro_use]
mod log;
mod core;
mod shell;

use std::path::Path;
use std::process::ExitCode;

/// The single serialized model, relative to the working directory (the post-new
/// hook runs from the maildir root; adjust via `--model` if that is not where
/// `models/` lives). Gitignored and regenerable — see design → *Persistence*.
const DEFAULT_MODEL_PATH: &str = "models/model.json";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Parse the subcommand and optional `--model <path>`, then dispatch. Split out
/// from `main` so the `?`-style error flows to a single reporting site.
fn run(args: &[String]) -> Result<(), String> {
    let mut command: Option<&str> = None;
    let mut model_path = DEFAULT_MODEL_PATH;
    let mut cutoff: Option<&str> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_path = args.get(i).ok_or("--model needs a path argument")?;
            }
            "--cutoff" => {
                i += 1;
                cutoff = Some(args.get(i).ok_or("--cutoff needs a date argument")?);
            }
            "-h" | "--help" => {
                print_usage();
                return Ok(());
            }
            other if command.is_none() => command = Some(other),
            other => return Err(format!("unexpected argument {other:?} (see --help)")),
        }
        i += 1;
    }

    let model_path = Path::new(model_path);
    match command {
        Some("train") => {
            log!("training over confirmed labels → {}", model_path.display());
            shell::train(model_path)
        }
        Some("classify") => {
            log!("classifying in-scope new mail with {}", model_path.display());
            shell::classify_new(model_path)
        }
        Some("eval") => {
            // Default cutoff mirrors the production CLASSIFY_CUTOFF so the sanity
            // check evaluates the same past→future boundary deployment uses.
            let cutoff = cutoff.unwrap_or("2026-07-01");
            log!("time-held-out evaluation at cutoff {cutoff} (no model written)");
            shell::evaluate(cutoff)
        }
        Some(other) => {
            Err(format!("unknown command {other:?} (expected `train`, `classify`, or `eval`)"))
        }
        None => {
            print_usage();
            Err("no command given".to_string())
        }
    }
}

/// The one-screen usage text, printed for `--help` and on a missing/unknown
/// command.
fn print_usage() {
    eprintln!(
        "usage: email_classifier <train|classify|eval> [--model <path>] [--cutoff <date>]\n\
         \n\
         Commands:\n  \
           train     fit a fresh model over every confirmed label (all dates)\n  \
           classify  guess in-scope new mail and write prio-* + auto (post-new hook)\n  \
           eval      time-held-out confusion-matrix sanity check (writes no model)\n\
         \n\
         Options:\n  \
           --model <path>   model file (default: {DEFAULT_MODEL_PATH})\n  \
           --cutoff <date>  eval split date (default: 2026-07-01); train before, test on/after"
    );
}
