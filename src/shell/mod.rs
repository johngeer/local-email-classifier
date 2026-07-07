// THE shell interface: train() and classify_new() only.
//
// All IO, caching, and the linfa solver live here. The remaining adapters and
// the two entry points are filled in at checklist §4–§5.

// §3 persistence: JSON save/load + load-time guards. Private (`mod`): the entry
// points below are the shell's public surface.
mod persist;

// §4 adapters: all IO and stateful adapters, each private. `mailfile` parses a
// maildir file, `embed` fronts fastembed, `notmuch` runs the tag queries behind
// two HashMap caches, `fit` runs the L-BFGS solve. The §5 entry points compose
// these; nothing here is public until then.
mod embed;
mod fit;
mod mailfile;
mod notmuch;
