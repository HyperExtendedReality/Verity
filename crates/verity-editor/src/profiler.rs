//! Frame timing: a small FPS readout in the viewport corner that opens a
//! profiler window with a CPU and GPU tree of timed rows, each with history.

use crate::ui::{ACCENT, BORDER, CYAN, MUTED, PINK, SURFACE, TEXT};
use egui::{Color32, Pos2, Rect, RichText, Sense, Stroke, vec2};
use std::{
    collections::{HashSet, VecDeque},
    time::{Duration, Instant},
};

const HISTORY: usize = 300;
/// Frames the readout averages over.
const RECENT: usize = 30;
/// The root CPU row: time since the previous frame started.
const FRAME: &str = "Frame";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Cpu,
    Gpu,
}

/// One timed row of the tree, in milliseconds.
struct Row {
    /// Kind, ancestors and name: stays the same across frames.
    id: String,
    name: String,
    depth: usize,
    kind: Kind,
    history: VecDeque<f32>,
    /// This frame's time, summed over repeated records.
    current: Option<f32>,
}

#[derive(Default)]
pub struct Profiler {
    /// In tree order: each row's children follow it, deeper by one.
    rows: Vec<Row>,
    /// Frames in each row's history.
    frames: usize,
    last: Option<Instant>,
    /// Where a row first seen this frame goes: after the last recorded row.
    cursor: usize,
    /// Ancestor names of the next record, per kind.
    stack: [Vec<String>; 2],
    counters: Vec<(String, String)>,
    /// Last frame's counters, which the window shows.
    shown: Vec<(String, String)>,
    paused: bool,
    selected: Option<String>,
    collapsed: HashSet<String>,
    pub open: bool,
}

fn ms(duration: Duration) -> f32 {
    duration.as_secs_f32() * 1000.0
}

impl Profiler {
    /// Close the previous frame's rows and start a new frame. Rows not
    /// recorded in a frame record zero.
    pub fn begin_frame(&mut self) {
        let now = Instant::now();
        if self.rows.is_empty() {
            self.rows.push(Row {
                id: format!("{:?}//{FRAME}", Kind::Cpu),
                name: FRAME.into(),
                depth: 0,
                kind: Kind::Cpu,
                history: VecDeque::new(),
                current: None,
            });
        }
        if let Some(last) = self.last.replace(now) {
            self.rows[0].current = Some(ms(now - last));
            if !self.paused {
                for row in &mut self.rows {
                    if row.history.len() == HISTORY {
                        row.history.pop_front();
                    }
                    row.history.push_back(row.current.unwrap_or(0.0));
                }
                self.frames = (self.frames + 1).min(HISTORY);
            }
        }
        for row in &mut self.rows {
            row.current = None;
        }
        self.cursor = 1;
        self.stack = Default::default();
        self.stack[Kind::Cpu as usize].push(FRAME.into());
        self.shown = std::mem::take(&mut self.counters);
    }

    /// Record CPU time for `name` under the latest row of depth `depth - 1`.
    pub fn cpu(&mut self, name: &str, depth: usize, time: Duration) {
        self.record(Kind::Cpu, name, depth, ms(time));
    }

    pub fn gpu(&mut self, name: &str, depth: usize, ms: f32) {
        self.record(Kind::Gpu, name, depth, ms);
    }

    /// Run `f`, recording its time as a CPU row.
    pub fn time<R>(&mut self, name: &str, depth: usize, f: impl FnOnce() -> R) -> R {
        let start = Instant::now();
        let result = f();
        self.cpu(name, depth, start.elapsed());
        result
    }

    /// A value shown under the tree, such as a count.
    pub fn counter(&mut self, name: &str, value: String) {
        self.counters.push((name.into(), value));
    }

    fn record(&mut self, kind: Kind, name: &str, depth: usize, value: f32) {
        let stack = &mut self.stack[kind as usize];
        stack.truncate(depth);
        let id = format!("{kind:?}/{}/{name}", stack.join("/"));
        stack.push(name.into());
        let index = match self.rows.iter().position(|row| row.id == id) {
            Some(index) => index,
            None => {
                let index = self.cursor.min(self.rows.len());
                self.rows.insert(
                    index,
                    Row {
                        id,
                        name: name.into(),
                        depth,
                        kind,
                        history: std::iter::repeat_n(0.0, self.frames).collect(),
                        current: None,
                    },
                );
                index
            }
        };
        let row = &mut self.rows[index];
        row.current = Some(row.current.unwrap_or(0.0) + value);
        self.cursor = index + 1;
    }

    fn frame_history(&self) -> Option<&VecDeque<f32>> {
        self.rows.first().map(|row| &row.history)
    }

    /// Mean frame time of the latest frames.
    fn recent_frame(&self) -> Option<f32> {
        let history = self.frame_history()?;
        let n = history.len().min(RECENT);
        (n > 0).then(|| history.iter().rev().take(n).sum::<f32>() / n as f32)
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

    /// The Profiler panel: the CPU and GPU trees, the selected row's graph
    /// and last frame's counters.
    pub fn panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let label = if self.paused { "Resume" } else { "Pause" };
            if ui
                .button(label)
                .on_hover_text("Freeze the history to inspect a spike")
                .clicked()
            {
                self.paused = !self.paused;
            }
            let name = self.selected_row().map_or(FRAME, |row| row.name.as_str());
            ui.label(RichText::new(format!("Graph: {name}")).color(MUTED));
        });
        self.graph(ui);
        ui.label(
            RichText::new("Click a row to graph it; lines at 60 and 30 FPS")
                .size(10.5)
                .color(MUTED),
        );
        ui.separator();
        egui::ScrollArea::vertical()
            .auto_shrink([false, true])
            .max_height(ui.available_height() - 120.0)
            .show(ui, |ui| {
                egui::Grid::new("profiler_tree")
                    .num_columns(5)
                    .spacing(vec2(14.0, 3.0))
                    .striped(true)
                    .show(ui, |ui| {
                        for kind in [Kind::Cpu, Kind::Gpu] {
                            self.tree(ui, kind);
                        }
                    });
            });
        if !self.shown.is_empty() {
            ui.separator();
            egui::Grid::new("profiler_counters")
                .num_columns(2)
                .show(ui, |ui| {
                    for (name, value) in &self.shown {
                        ui.label(RichText::new(name).color(MUTED));
                        ui.label(value);
                        ui.end_row();
                    }
                });
        }
    }

    fn selected_row(&self) -> Option<&Row> {
        let id = self.selected.as_ref()?;
        self.rows.iter().find(|row| &row.id == id)
    }

    /// One kind's rows with avg, min, max and 1% high; collapsed parents hide
    /// their children.
    fn tree(&mut self, ui: &mut egui::Ui, kind: Kind) {
        let indices: Vec<usize> = (0..self.rows.len())
            .filter(|&i| self.rows[i].kind == kind)
            .collect();
        if indices.is_empty() {
            return;
        }
        let header = match kind {
            Kind::Cpu => "CPU ms",
            Kind::Gpu => "GPU ms",
        };
        for header in [header, "avg", "min", "max", "1% high"] {
            ui.label(RichText::new(header).size(11.0).color(MUTED));
        }
        ui.end_row();
        let mut hide_below = None;
        for (n, &i) in indices.iter().enumerate() {
            let row = &self.rows[i];
            if hide_below.is_some_and(|depth| row.depth > depth) {
                continue;
            }
            hide_below = None;
            let parent = indices
                .get(n + 1)
                .is_some_and(|&next| self.rows[next].depth > row.depth);
            let collapsed = self.collapsed.contains(&row.id);
            if parent && collapsed {
                hide_below = Some(row.depth);
            }
            let id = row.id.clone();
            let color = match (row.depth, kind) {
                (0, _) => TEXT,
                (_, Kind::Cpu) => CYAN,
                (_, Kind::Gpu) => PINK,
            };
            let selected = self.selected.as_ref() == Some(&id);
            let stats = stats(&row.history);
            let name = row.name.clone();
            let depth = row.depth;
            ui.horizontal(|ui| {
                ui.add_space(depth as f32 * 12.0);
                if parent {
                    let arrow = if collapsed { "▸" } else { "▾" };
                    if ui.small_button(arrow).clicked() && !self.collapsed.remove(&id) {
                        self.collapsed.insert(id.clone());
                    }
                } else {
                    ui.add_space(18.0);
                }
                let text = RichText::new(name).color(color);
                if ui.selectable_label(selected, text).clicked() {
                    self.selected = (!selected).then(|| id.clone());
                }
            });
            match stats {
                Some(values) => {
                    for v in values {
                        ui.monospace(format!("{v:6.2}"));
                    }
                }
                None => {
                    for _ in 0..4 {
                        ui.monospace("     —");
                    }
                }
            }
            ui.end_row();
        }
    }

    /// The selected row (or the frame) as bars, over the frame time in grey.
    fn graph(&self, ui: &mut egui::Ui) {
        let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 110.0), Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 6, Color32::from_black_alpha(120));
        let Some(frames) = self.frame_history() else {
            return;
        };
        let selected = self.selected_row();
        let (values, color) = match selected {
            Some(row) if row.kind == Kind::Gpu => (&row.history, PINK),
            Some(row) => (&row.history, CYAN),
            None => (frames, ACCENT.gamma_multiply(0.7)),
        };
        // Scale to the slowest frame, at least 40 ms so the 30 FPS line shows.
        let top = frames
            .iter()
            .chain(values)
            .fold(40.0, |a, &b| f32::max(a, b));
        let y = |v: f32| rect.bottom() - (v / top).min(1.0) * rect.height();
        let width = rect.width() / HISTORY as f32;
        let bar = |i: usize, len: usize, v: f32, color| {
            let x = rect.right() - width * (len - i) as f32;
            painter.rect_filled(
                Rect::from_x_y_ranges(x..=x + width.max(1.0), y(v)..=rect.bottom()),
                0,
                color,
            );
        };
        if selected.is_some() {
            for (i, &v) in frames.iter().enumerate() {
                bar(i, frames.len(), v, Color32::from_gray(70));
            }
        }
        for (i, &v) in values.iter().enumerate() {
            bar(i, values.len(), v, color);
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

/// Avg, min, max and 1% high of `history`.
fn stats(history: &VecDeque<f32>) -> Option<[f32; 4]> {
    if history.is_empty() {
        return None;
    }
    let mut values: Vec<f32> = history.iter().copied().collect();
    values.sort_by(f32::total_cmp);
    let n = values.len();
    let avg = values.iter().sum::<f32>() / n as f32;
    Some([
        avg,
        values[0],
        values[n - 1],
        values[(n * 99 / 100).min(n - 1)],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_a_bounded_history_after_the_first_frame() {
        let mut profiler = Profiler::default();
        profiler.begin_frame();
        assert!(
            profiler.recent_frame().is_none(),
            "the first frame has no interval"
        );
        for _ in 0..HISTORY + 10 {
            profiler.cpu("UI", 1, Duration::from_millis(2));
            profiler.gpu("GPU frame", 0, 3.0);
            profiler.begin_frame();
        }
        assert!(profiler.rows.iter().all(|r| r.history.len() == HISTORY));
        assert!(profiler.rows[1].history.iter().all(|&v| v == 2.0));
        assert!(profiler.rows[2].history.iter().all(|&v| v == 3.0));
    }

    #[test]
    fn nested_rows_keep_order_and_depth_and_missing_rows_record_zero() {
        let mut profiler = Profiler::default();
        profiler.begin_frame();
        profiler.cpu("Simulation", 1, Duration::from_millis(1));
        profiler.cpu("Render", 1, Duration::from_millis(4));
        profiler.cpu("Submit", 2, Duration::from_millis(1));
        profiler.begin_frame();
        // A child seen later goes under its parent, not at the end.
        profiler.cpu("Simulation", 1, Duration::from_millis(1));
        profiler.cpu("Solve", 2, Duration::from_millis(1));
        profiler.cpu("Render", 1, Duration::from_millis(4));
        profiler.begin_frame();
        let tree: Vec<_> = (profiler.rows.iter())
            .map(|r| (r.name.as_str(), r.depth))
            .collect();
        assert_eq!(
            tree,
            [
                ("Frame", 0),
                ("Simulation", 1),
                ("Solve", 2),
                ("Render", 1),
                ("Submit", 2)
            ]
        );
        let history = |name: &str| {
            let row = profiler.rows.iter().find(|r| r.name == name).unwrap();
            row.history.iter().copied().collect::<Vec<_>>()
        };
        assert_eq!(history("Submit"), [1.0, 0.0]);
        assert_eq!(history("Solve"), [0.0, 1.0]);
    }
}
