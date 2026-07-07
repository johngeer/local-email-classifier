// THE shell interface: train() and classify_new() only.
//
// All IO, caching, and the linfa solver live here. The remaining adapters and
// the two entry points are filled in at checklist §4–§5.

// §3 persistence: JSON save/load + load-time guards. Private (`mod`): the entry
// points below are the shell's public surface.
mod persist;
