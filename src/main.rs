// arg parse → dispatch train/classify. Kept to ~50 lines; only calls the two
// shell entry points. Wiring is filled in at checklist §6.

// The §1 core leaves land before the code that consumes them (features/model in
// §2, the shell in §4–5), so they read as dead until those steps wire them in.
// Remove this once `main` dispatches to the shell entry points (§6).
#![allow(dead_code, unused_imports)]

mod core;
mod shell;

fn main() {
    // Dispatch is wired in checklist §6 (main.rs). For now the skeleton just
    // establishes the module boundary: `core` (pure) and `shell` (IO).
}
