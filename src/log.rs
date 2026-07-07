//! Timestamped logging to stderr, so a `task run` log makes plain both *when*
//! each step happened and *how long* it took relative to the previous line.
//!
//! [`log!`] takes the same `format!` arguments as `eprintln!` and prefixes the
//! line with two clocks: total elapsed since process start, and the delta since
//! the last `log!` call — the "time between logs" that lets a run be read as a
//! timeline. Both are measured off a monotonic clock, immune to wall-clock jumps.

use std::sync::Mutex;
use std::time::Instant;

/// Process start and the instant of the previous [`log!`] call. `start` is fixed
/// for the run; `prev` advances each line so the reported delta is line-to-line.
/// Behind a `Mutex` so concurrent logs stay coherent; contention is nil in this
/// single-threaded-logging tool.
struct Clock {
    start: Instant,
    prev: Instant,
}

static CLOCK: Mutex<Option<Clock>> = Mutex::new(None);

/// The `[+total Δdelta]` prefix for one log line: seconds since process start and
/// seconds since the previous `log!`. Lazily seeds the clock on first call, so no
/// explicit init is needed — the first line reports `+0.000 Δ0.000`.
#[doc(hidden)]
pub fn prefix() -> String {
    let now = Instant::now();
    let mut guard = CLOCK.lock().unwrap_or_else(|p| p.into_inner());
    let clock = guard.get_or_insert(Clock { start: now, prev: now });
    let total = now.duration_since(clock.start).as_secs_f64();
    let delta = now.duration_since(clock.prev).as_secs_f64();
    clock.prev = now;
    format!("[+{total:7.3} Δ{delta:6.3}]")
}

/// `eprintln!` with a leading elapsed/delta timestamp — see the module docs. Use
/// for progress and status lines; the format arguments are identical to
/// `eprintln!`.
#[macro_export]
macro_rules! log {
    ($($arg:tt)*) => {
        eprintln!("{} {}", $crate::log::prefix(), format_args!($($arg)*))
    };
}
