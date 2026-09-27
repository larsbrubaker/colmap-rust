// First-paint gate: the app's "ready" signal. A host marks it after the first frame has been
// painted and presented; the web shell (next step) exposes it to Playwright as the page's ready
// flag, and a reactive host uses `should_paint_tick` so that first frame is never skipped.
// Same idea as AtomArtist's `first_paint.rs`.

use std::cell::Cell;

/// One-shot latch: false until a host reports a presented frame.
#[derive(Debug, Default)]
pub struct FirstPaintGate {
    painted: Cell<bool>,
}

impl FirstPaintGate {
    /// A gate that has not yet seen a presented frame.
    pub const fn new() -> Self {
        Self {
            painted: Cell::new(false),
        }
    }

    /// True once [`Self::mark_painted`] has run.
    pub fn has_painted(&self) -> bool {
        self.painted.get()
    }

    /// Record that a frame was painted and presented. Call only after a real present, so a
    /// tick that bailed out early (no surface texture yet) leaves the gate open.
    pub fn mark_painted(&self) {
        self.painted.set(true);
    }

    /// Whether a host tick should paint: always until the first present, afterwards on a
    /// resize or when the host's own reactive predicate asks (evaluated lazily).
    pub fn should_paint_tick(&self, resized: bool, host_wants_draw: impl FnOnce() -> bool) -> bool {
        !self.painted.get() || resized || host_wants_draw()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forces_paint_until_marked() {
        let gate = FirstPaintGate::new();
        assert!(!gate.has_painted());
        assert!(gate.should_paint_tick(false, || false));
        gate.mark_painted();
        assert!(gate.has_painted());
        assert!(!gate.should_paint_tick(false, || false));
        assert!(gate.should_paint_tick(true, || false));
        assert!(gate.should_paint_tick(false, || true));
    }
}
