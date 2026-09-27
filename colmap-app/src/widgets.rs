// Small generic widgets the app's views are built from:
//
// - `Tagged` gives any widget a reflection id (`Widget::id`), so tests and shells can find it
//   with `find_widget_by_id` / `find_widget_screen_rect`. agg-gui's stock widgets have no id
//   setter, so this wrapper is how the app names them.
// - `BoundLabel` is a `Label` whose text comes from a closure over app state, re-read on every
//   layout, which is how data-driven text (stage status, backend info) follows state changes.
//
// Both are layout-transparent: they size to their child and forward its anchors and margins.

use std::sync::Arc;

use agg_gui::text::Font;
use agg_gui::{
    DrawCtx, Event, EventResult, HAnchor, Insets, Label, Rect, Size, Spacer, VAnchor, Widget,
};

/// Wraps one child and gives it an id.
pub struct Tagged {
    id: String,
    bounds: Rect,
    children: Vec<Box<dyn Widget>>,
}

impl Tagged {
    pub fn new(id: impl Into<String>, child: Box<dyn Widget>) -> Self {
        Self {
            id: id.into(),
            bounds: Rect::default(),
            children: vec![child],
        }
    }
}

/// Lay out the single child at the wrapper's origin and adopt its size. On an axis where the
/// child stretches, the child gets the whole `available` extent rather than its natural size:
/// a parent re-lays the wrapper out at the final slot size, and the child must fill that slot
/// (not just report into it) for its background and top-anchored content to land right.
fn layout_single(children: &mut [Box<dyn Widget>], bounds: &mut Rect, available: Size) -> Size {
    let size = match children.first_mut() {
        Some(child) => {
            let natural = child.layout(available);
            let width = if child.h_anchor().is_stretch() {
                available.width
            } else {
                natural.width
            };
            let height = if child.v_anchor().is_stretch() {
                available.height
            } else {
                natural.height
            };
            let s = Size::new(width, height);
            if s != natural {
                child.layout(s);
            }
            child.set_bounds(Rect::new(0.0, 0.0, s.width, s.height));
            s
        }
        None => Size::new(0.0, 0.0),
    };
    *bounds = Rect::new(bounds.x, bounds.y, size.width, size.height);
    size
}

/// Forwarded layout properties of the first child (defaults when there is none).
macro_rules! forward_layout_props {
    () => {
        fn margin(&self) -> Insets {
            self.children
                .first()
                .map(|c| c.margin())
                .unwrap_or(Insets::ZERO)
        }
        fn h_anchor(&self) -> HAnchor {
            self.children
                .first()
                .map(|c| c.h_anchor())
                .unwrap_or(HAnchor::FIT)
        }
        fn v_anchor(&self) -> VAnchor {
            self.children
                .first()
                .map(|c| c.v_anchor())
                .unwrap_or(VAnchor::FIT)
        }
        fn min_size(&self) -> Size {
            self.children
                .first()
                .map(|c| c.min_size())
                .unwrap_or(Size::ZERO)
        }
        fn max_size(&self) -> Size {
            self.children
                .first()
                .map(|c| c.max_size())
                .unwrap_or(Size::MAX)
        }
    };
}

impl Widget for Tagged {
    fn type_name(&self) -> &'static str {
        "Tagged"
    }
    fn id(&self) -> Option<&str> {
        Some(&self.id)
    }
    fn bounds(&self) -> Rect {
        self.bounds
    }
    fn set_bounds(&mut self, bounds: Rect) {
        self.bounds = bounds;
    }
    fn children(&self) -> &[Box<dyn Widget>] {
        &self.children
    }
    fn children_mut(&mut self) -> &mut Vec<Box<dyn Widget>> {
        &mut self.children
    }
    forward_layout_props!();
    fn layout(&mut self, available: Size) -> Size {
        layout_single(&mut self.children, &mut self.bounds, available)
    }
    fn paint(&mut self, _ctx: &mut dyn DrawCtx) {}
    fn on_event(&mut self, _event: &Event) -> EventResult {
        EventResult::Ignored
    }
    fn is_visible(&self) -> bool {
        self.children.first().is_some_and(|c| c.is_visible())
    }
}

/// Horizontal filler for a `FlexRow` (added with `add_flex`). A bare `Spacer` reports the
/// row's full height as its natural height, which makes the row claim its whole slot; capping
/// its height keeps the row at its content's height.
pub fn row_filler() -> Spacer {
    Spacer::new().with_max_size(Size::new(f64::MAX, 0.0))
}

/// Source of a [`BoundLabel`]'s text.
pub type TextSource = Box<dyn Fn() -> String>;

/// A label bound to app state: its text is `source()` as of the latest layout.
pub struct BoundLabel {
    id: Option<String>,
    bounds: Rect,
    children: Vec<Box<dyn Widget>>,
    source: TextSource,
    text: String,
}

impl BoundLabel {
    /// `configure` styles the inner [`Label`] (font size, color, anchors).
    pub fn new(
        font: Arc<Font>,
        source: impl Fn() -> String + 'static,
        configure: impl FnOnce(Label) -> Label,
    ) -> Self {
        let text = source();
        let label = configure(Label::new(text.clone(), font));
        Self {
            id: None,
            bounds: Rect::default(),
            children: vec![Box::new(label)],
            source: Box::new(source),
            text,
        }
    }

    /// Give the label a reflection id.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// The text shown as of the latest layout.
    pub fn text(&self) -> &str {
        &self.text
    }
}

impl Widget for BoundLabel {
    fn type_name(&self) -> &'static str {
        "BoundLabel"
    }
    fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }
    fn bounds(&self) -> Rect {
        self.bounds
    }
    fn set_bounds(&mut self, bounds: Rect) {
        self.bounds = bounds;
    }
    fn children(&self) -> &[Box<dyn Widget>] {
        &self.children
    }
    fn children_mut(&mut self) -> &mut Vec<Box<dyn Widget>> {
        &mut self.children
    }
    forward_layout_props!();
    fn layout(&mut self, available: Size) -> Size {
        let text = (self.source)();
        if text != self.text {
            if let Some(label) = self.children.first_mut() {
                label.set_label_text(&text);
            }
            self.text = text;
        }
        layout_single(&mut self.children, &mut self.bounds, available)
    }
    fn paint(&mut self, _ctx: &mut dyn DrawCtx) {}
    fn on_event(&mut self, _event: &Event) -> EventResult {
        EventResult::Ignored
    }
    fn properties(&self) -> Vec<(&'static str, String)> {
        vec![("text", self.text.clone())]
    }
}
