//! The game's screen-space UI: [`Widget`]s from the scene, laid out by anchor
//! and drawn with egui, which the renderer composites over the scene. The
//! editor draws the same widgets over its viewport to preview and edit them.

use egui::{Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};
use verity_core::{Widget, WidgetKind};

/// Widget lengths are pixels at this screen height; other heights scale them.
pub const REFERENCE_HEIGHT: f32 = 1080.0;

/// How many points one reference pixel takes on `screen`.
pub fn scale(screen: Rect) -> f32 {
    screen.height() / REFERENCE_HEIGHT
}

/// Where `widget` sits on `screen`.
pub fn rect(widget: &Widget, screen: Rect) -> Rect {
    let scale = scale(screen);
    let size = Vec2::from(widget.size) * scale;
    let anchor = screen.min + screen.size() * Vec2::from(widget.anchor);
    let min = anchor + Vec2::from(widget.offset) * scale - size * Vec2::from(widget.pivot);
    Rect::from_min_size(min, size)
}

fn color([r, g, b, a]: [u8; 4]) -> Color32 {
    Color32::from_rgba_unmultiplied(r, g, b, a)
}

/// Draw the visible widgets over `screen`, back to front, and return the
/// indices of buttons clicked this frame. Buttons only take input when
/// `interactive`, so an editor can preview them without pressing them.
pub fn show(ui: &mut egui::Ui, screen: Rect, widgets: &[Widget], interactive: bool) -> Vec<usize> {
    let painter = ui.painter_at(screen);
    let scale = scale(screen);
    let mut clicked = Vec::new();
    for (index, widget) in widgets.iter().enumerate() {
        if !widget.visible {
            continue;
        }
        let rect = rect(widget, screen);
        let radius = widget.corner_radius * scale;
        let mut background = color(widget.background);
        if widget.kind == WidgetKind::Button && interactive {
            let response = ui.interact(rect, ui.id().with(("game_ui", index)), Sense::click());
            if response.clicked() {
                clicked.push(index);
            }
            if response.is_pointer_button_down_on() {
                background = background.gamma_multiply(0.7);
            } else if response.hovered() {
                background = background.lerp_to_gamma(Color32::WHITE, 0.15);
                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            }
        }
        match widget.kind {
            WidgetKind::Text => {}
            WidgetKind::Bar => {
                painter.rect_filled(rect, radius, background);
                let mut fill = rect;
                fill.set_width(rect.width() * widget.value.clamp(0.0, 1.0));
                painter.rect_filled(fill, radius, color(widget.color));
            }
            WidgetKind::Panel | WidgetKind::Button => {
                painter.rect_filled(rect, radius, background);
            }
        }
        if !widget.text.is_empty() && widget.kind != WidgetKind::Bar {
            let font = FontId::proportional((widget.font_size * scale).max(1.0));
            let galley =
                painter.layout(widget.text.clone(), font, color(widget.color), rect.width());
            let at = Align2::CENTER_CENTER
                .align_size_within_rect(galley.size(), rect)
                .min;
            painter.galley(at, galley, Color32::WHITE);
        }
    }
    clicked
}

/// The topmost visible widget under `point`.
pub fn hit(widgets: &[Widget], screen: Rect, point: Pos2) -> Option<usize> {
    (widgets.iter().enumerate().rev())
        .find(|(_, w)| w.visible && rect(w, screen).contains(point))
        .map(|(index, _)| index)
}

/// An outline around a widget, for editors.
pub fn outline(painter: &egui::Painter, widget: &Widget, screen: Rect, stroke: Stroke) {
    painter.rect_stroke(
        rect(widget, screen).expand(1.0),
        2.0,
        stroke,
        StrokeKind::Outside,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_pivot_and_offset_place_the_widget_and_scale_with_height() {
        let screen = Rect::from_min_size(Pos2::new(100.0, 50.0), Vec2::new(960.0, 540.0));
        // Bottom-right corner, pivot at its own bottom right, 20 px in.
        let widget = Widget {
            anchor: [1.0, 1.0],
            pivot: [1.0, 1.0],
            offset: [-20.0, -20.0],
            size: [200.0, 100.0],
            ..Widget::default()
        };
        let placed = rect(&widget, screen);
        assert_eq!(placed.max, Pos2::new(1060.0 - 10.0, 590.0 - 10.0));
        assert_eq!(placed.size(), Vec2::new(100.0, 50.0));
        assert_eq!(hit(&[widget.clone()], screen, placed.center()), Some(0));
        let hidden = Widget {
            visible: false,
            ..widget
        };
        assert_eq!(hit(&[hidden], screen, placed.center()), None);
    }
}
