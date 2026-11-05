//! Frame timing: a small FPS readout in the viewport corner that opens a
//! profiler window with per-stage CPU times and a frame graph.

use crate::ui::{ACCENT, BORDER, CYAN, MUTED, PINK, SURFACE, TEXT};
use egui::{Color32, Pos2, Rect, RichText, Sense, Stroke, vec2};
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

const HISTORY: usize = 300;
/// Frames the readout averages over.
const RECENT: usize = 30;

/// One frame's CPU times in milliseconds.
#[derive(Clone, Copy, Default)]
struct Sample {
    /// Time since the previous frame started.
    frame: f32,
    /// Building the editor UI.
    ui: f32,
    /// Recording and submitting the renderer's frame.
    render: f32,
}

#[derive(Default)]
pub struct Profiler {
    samples: VecDeque<Sample>,
    last: Option<Instant>,
    pub open: bool,
}

fn ms(duration: Duration) -> f32 {
    duration.as_secs_f32() * 1000.0
}

impl Profiler {
    /// Record a frame that spent `ui` building the UI and `render` rendering.
    pub fn record(&mut self, ui: Duration, render: Duration) {
        let now = Instant::now();
        let Some(last) = self.last.replace(now) else {
            return;
        };
        if self.samples.len() == HISTORY {
            self.samples.pop_front();
        }
        self.samples.push_back(Sample {
            frame: ms(now - last),
            ui: ms(ui),
            render: ms(render),
        });
    }

    /// Mean frame time of the latest frames.
    fn recent_frame(&self) -> Option<f32> {
        let n = self.samples.len().min(RECENT);
        (n > 0).then(|| {
            self.samples
                .iter()
                .rev()
                .take(n)
                .map(|s| s.frame)
                .sum::<f32>()
                / n as f32
        })
    }

    /// The corner readout over `viewport`; clicking it toggles the window.
    pub fn badge(&mut self, ui: &egui::Ui, viewport: Rect) {
        let frame = self.recent_frame();
        let text = match frame {
            Some(f) => format!("{:.0} FPS  {f:.1} ms", 1000.0 / f.max(0.001)),
            None => "— FPS".into(),
        };
        let color = match frame {
            Some(f) if f > 33.4 => PINK,
            Some(f) if f > 16.7 => Color32::from_rgb(255, 196, 64),
            _ => ACCENT,
        };
        let galley = ui
            .painter()
            .layout_no_wrap(text, egui::FontId::monospace(11.0), color);
        let rect = Rect::from_min_size(
            Pos2::new(
                viewport.right() - galley.size().x - 22.0,
                viewport.top() + 10.0,
            ),
            galley.size() + vec2(12.0, 6.0),
        );
        let response = ui
            .interact(rect, ui.id().with("profiler_badge"), Sense::click())
            .on_hover_text("Open the profiler");
        let painter = ui.painter_at(viewport);
        let fill = if response.hovered() {
            SURFACE
        } else {
            Color32::from_black_alpha(150)
        };
        painter.rect(
            rect,
            6,
            fill,
            Stroke::new(1.0, BORDER),
            egui::StrokeKind::Inside,
        );
        painter.galley(rect.min + vec2(6.0, 3.0), galley, color);
        if response.clicked() {
            self.open = !self.open;
        }
    }

    /// The in-depth window: stage statistics, the frame graph and `details`.
    pub fn window(&mut self, ctx: &egui::Context, details: &[(&str, String)]) {
        let mut open = self.open;
        egui::Window::new("Profiler")
            .open(&mut open)
            .default_width(380.0)
            .resizable(false)
            .show(ctx, |ui| {
                egui::Grid::new("profiler_stats")
                    .num_columns(5)
                    .spacing(vec2(14.0, 3.0))
                    .show(ui, |ui| {
                        for header in ["CPU ms", "avg", "min", "max", "1% high"] {
                            ui.label(RichText::new(header).size(11.0).color(MUTED));
                        }
                        ui.end_row();
                        let stages: [(&str, Color32, fn(&Sample) -> f32); 3] = [
                            ("Frame", TEXT, |s| s.frame),
                            ("UI", CYAN, |s| s.ui),
                            ("Render", PINK, |s| s.render),
                        ];
                        for (name, color, get) in stages {
                            let mut values: Vec<f32> = self.samples.iter().map(get).collect();
                            values.sort_by(f32::total_cmp);
                            ui.label(RichText::new(name).color(color));
                            if values.is_empty() {
                                ui.end_row();
                                continue;
                            }
                            let avg = values.iter().sum::<f32>() / values.len() as f32;
                            let high = values[(values.len() * 99 / 100).min(values.len() - 1)];
                            for v in [avg, values[0], values[values.len() - 1], high] {
                                ui.monospace(format!("{v:6.2}"));
                            }
                            ui.end_row();
                        }
                    });
                ui.add_space(6.0);
                self.graph(ui);
                ui.label(
                    RichText::new(
                        "Frame bars: UI (cyan) and render (pink) stacked; lines at 60 and 30 FPS",
                    )
                    .size(10.5)
                    .color(MUTED),
                );
                if !details.is_empty() {
                    ui.separator();
                    egui::Grid::new("profiler_details")
                        .num_columns(2)
                        .show(ui, |ui| {
                            for (name, value) in details {
                                ui.label(RichText::new(*name).color(MUTED));
                                ui.label(value);
                                ui.end_row();
                            }
                        });
                }
            });
        self.open = open;
    }

    fn graph(&self, ui: &mut egui::Ui) {
        let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 110.0), Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 6, Color32::from_black_alpha(120));
        // Scale to the slowest frame, at least 40 ms so the 30 FPS line shows.
        let top = self.samples.iter().map(|s| s.frame).fold(40.0, f32::max);
        let y = |v: f32| rect.bottom() - (v / top).min(1.0) * rect.height();
        let width = rect.width() / HISTORY as f32;
        let start = rect.right() - width * self.samples.len() as f32;
        for (i, s) in self.samples.iter().enumerate() {
            let x = start + width * i as f32;
            let bar = |from: f32, to: f32, color| {
                painter.rect_filled(
                    Rect::from_x_y_ranges(x..=x + width.max(1.0), y(to)..=y(from)),
                    0,
                    color,
                );
            };
            bar(0.0, s.frame, Color32::from_gray(70));
            bar(0.0, s.ui, CYAN);
            bar(s.ui, s.ui + s.render, PINK);
        }
        for (budget, label) in [(1000.0 / 60.0, "60"), (1000.0 / 30.0, "30")] {
            let line = y(budget);
            painter.hline(
                rect.x_range(),
                line,
                Stroke::new(1.0, ACCENT.gamma_multiply(0.5)),
            );
            painter.text(
                Pos2::new(rect.left() + 4.0, line - 1.0),
                egui::Align2::LEFT_BOTTOM,
                label,
                egui::FontId::monospace(9.5),
                MUTED,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_a_bounded_history_after_the_first_frame() {
        let mut profiler = Profiler::default();
        profiler.record(Duration::ZERO, Duration::ZERO);
        assert!(
            profiler.recent_frame().is_none(),
            "the first frame has no interval"
        );
        for _ in 0..HISTORY + 10 {
            profiler.record(Duration::from_millis(2), Duration::from_millis(3));
        }
        assert_eq!(profiler.samples.len(), HISTORY);
        assert!(
            profiler
                .samples
                .iter()
                .all(|s| s.ui == 2.0 && s.render == 3.0)
        );
    }
}
