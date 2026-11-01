use crate::ui::{ACCENT, ACCENT_DARK, BORDER, CYAN, MUTED, PINK, SURFACE, TEXT};
use std::fs;
use std::path::{Path, PathBuf};

/// Actions that require the editor to load an asset.
pub enum AssetAction {
    OpenScene(PathBuf),
    ImportModel(PathBuf),
}

struct Entry {
    name: String,
    path: PathBuf,
    is_dir: bool,
    size: u64,
}

fn entry_row(ui: &mut egui::Ui, entry: &Entry, selected: bool) -> egui::Response {
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 29.0), egui::Sense::click());
    let response = response.on_hover_text(&entry.name);
    if selected || response.hovered() {
        ui.painter()
            .rect_filled(rect, 7, if selected { ACCENT_DARK } else { SURFACE });
    }
    if selected {
        ui.painter().rect_stroke(
            rect,
            7,
            egui::Stroke::new(1.0, BORDER),
            egui::StrokeKind::Inside,
        );
    }
    let icon = egui::Rect::from_center_size(
        rect.left_center() + egui::vec2(16.0, 0.0),
        egui::vec2(23.0, 19.0),
    );
    ui.painter().rect_filled(icon, 4, SURFACE);
    let ext = entry
        .path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    let (kind, color) = if entry.is_dir {
        ("DIR", CYAN)
    } else if ext.eq_ignore_ascii_case("glb") {
        ("3D", PINK)
    } else {
        ("FILE", MUTED)
    };
    ui.painter().text(
        icon.center(),
        egui::Align2::CENTER_CENTER,
        kind,
        egui::FontId::proportional(8.0),
        color,
    );
    let text_rect = egui::Rect::from_min_max(
        rect.left_top() + egui::vec2(34.0, 0.0),
        rect.right_bottom() - egui::vec2(6.0, 0.0),
    );
    ui.painter().with_clip_rect(text_rect).text(
        text_rect.left_center(),
        egui::Align2::LEFT_CENTER,
        &entry.name,
        egui::FontId::proportional(12.0),
        TEXT,
    );
    response
}

/// A filesystem browser rooted at the workspace's assets directory.
pub struct AssetBrowser {
    current: PathBuf,
    history: Vec<PathBuf>,
    search: String,
    entries: Vec<Entry>,
    selected: Option<PathBuf>,
    error: Option<String>,
    loaded: bool,
}

impl Default for AssetBrowser {
    fn default() -> Self {
        Self {
            current: PathBuf::new(),
            history: Vec::new(),
            search: String::new(),
            entries: Vec::new(),
            selected: None,
            error: None,
            loaded: false,
        }
    }
}

impl AssetBrowser {
    fn assets_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("assets")
    }

    // Resolve both paths on each filesystem operation so a changed symlink cannot
    // lead the browser outside the assets directory.
    fn resolve(relative: &Path) -> Result<PathBuf, String> {
        let root = Self::assets_root()
            .canonicalize()
            .map_err(|error| format!("Cannot access assets directory: {error}"))?;
        let path = root
            .join(relative)
            .canonicalize()
            .map_err(|error| format!("Cannot access {}: {error}", relative.display()))?;
        if !path.starts_with(&root) {
            return Err("This path is outside the assets directory.".to_owned());
        }
        Ok(path)
    }

    fn refresh(&mut self) {
        self.loaded = true;
        self.entries.clear();
        self.error = None;
        let result = (|| {
            let directory = Self::resolve(&self.current)?;
            let entries = fs::read_dir(directory)
                .map_err(|error| format!("Cannot read directory: {error}"))?;
            for entry in entries {
                let entry = match entry {
                    Ok(entry) => entry,
                    Err(error) => {
                        self.error = Some(format!("Some entries could not be read: {error}"));
                        continue;
                    }
                };
                let relative = self.current.join(entry.file_name());
                // Ignore symlinks entirely; they may escape the root or form cycles.
                let metadata = match fs::symlink_metadata(entry.path()) {
                    Ok(metadata) if !metadata.file_type().is_symlink() => metadata,
                    Ok(_) => continue,
                    Err(error) => {
                        self.error = Some(format!("Some entries could not be read: {error}"));
                        continue;
                    }
                };
                if !metadata.is_dir() && !metadata.is_file() {
                    continue;
                }
                if Self::resolve(&relative).is_err() {
                    continue;
                }
                self.entries.push(Entry {
                    name: entry.file_name().to_string_lossy().into_owned(),
                    path: relative,
                    is_dir: metadata.is_dir(),
                    size: metadata.len(),
                });
            }
            Ok::<(), String>(())
        })();
        if let Err(error) = result {
            self.error = Some(error);
        }
        self.entries.sort_by(|a, b| {
            b.is_dir
                .cmp(&a.is_dir)
                .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
                .then_with(|| a.name.cmp(&b.name))
        });
        if self
            .selected
            .as_ref()
            .is_some_and(|selected| !self.entries.iter().any(|entry| &entry.path == selected))
        {
            self.selected = None;
        }
    }

    fn navigate(&mut self, path: PathBuf) {
        if path != self.current {
            self.history
                .push(std::mem::replace(&mut self.current, path));
            self.selected = None;
            self.refresh();
        }
    }

    /// Show the browser and return an action for an activated supported asset.
    pub fn show(&mut self, ui: &mut egui::Ui) -> Option<AssetAction> {
        if !self.loaded {
            self.refresh();
        }

        let mut destination = None;
        let mut back = false;
        let mut refresh = false;
        ui.horizontal(|ui| {
            back = ui
                .add_enabled(!self.history.is_empty(), egui::Button::new("Back"))
                .clicked();
            if ui
                .add_enabled(
                    !self.current.as_os_str().is_empty(),
                    egui::Button::new("Up"),
                )
                .clicked()
            {
                destination = Some(self.current.parent().unwrap_or(Path::new("")).to_path_buf());
            }
            refresh = ui.button("Refresh").clicked();
            ui.add(
                egui::TextEdit::singleline(&mut self.search)
                    .hint_text("Filter assets...")
                    .desired_width(180.0),
            );
        });

        ui.horizontal_wrapped(|ui| {
            if ui
                .link(egui::RichText::new("ASSETS").strong().color(ACCENT))
                .clicked()
            {
                destination = Some(PathBuf::new());
            }
            let mut path = PathBuf::new();
            for component in self.current.components() {
                path.push(component.as_os_str());
                ui.label("/");
                if ui.link(component.as_os_str().to_string_lossy()).clicked() {
                    destination = Some(path.clone());
                }
            }
        });

        if back {
            if let Some(path) = self.history.pop() {
                self.current = path;
                self.selected = None;
                self.refresh();
            }
        } else if let Some(path) = destination.take() {
            self.navigate(path);
        } else if refresh {
            self.refresh();
        }

        if let Some(error) = &self.error {
            ui.colored_label(egui::Color32::LIGHT_RED, error);
        }

        let filter = self.search.to_lowercase();
        let mut activated = None;
        let mut found = false;
        // Leave room for the selected file's details in a short bottom panel.
        let details_height = if self.selected.is_some() { 44.0 } else { 0.0 };
        let list_height = (ui.available_height() - details_height).max(35.0);
        egui::ScrollArea::vertical()
            .id_salt("asset_browser_entries")
            .max_height(list_height)
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 2.0;
                for entry in &self.entries {
                    if !filter.is_empty() && !entry.name.to_lowercase().contains(&filter) {
                        continue;
                    }
                    found = true;
                    let response =
                        entry_row(ui, entry, self.selected.as_ref() == Some(&entry.path));
                    if response.clicked() {
                        self.selected = Some(entry.path.clone());
                    }
                    if response.double_clicked() {
                        if entry.is_dir {
                            destination = Some(entry.path.clone());
                        } else {
                            activated = Some(entry.path.clone());
                        }
                    }
                }
                if !found && self.error.is_none() {
                    ui.weak(if filter.is_empty() {
                        "This folder is empty."
                    } else {
                        "No assets match the search."
                    });
                }
            });

        if let Some(path) = destination {
            self.navigate(path);
        }
        if let Some(path) = activated {
            match Self::resolve(&path) {
                Ok(absolute) if absolute.is_file() => {
                    let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
                    if extension.eq_ignore_ascii_case("glb") {
                        return Some(AssetAction::ImportModel(Path::new("assets").join(path)));
                    }
                    if extension.eq_ignore_ascii_case("json")
                        || extension.eq_ignore_ascii_case("hyperon")
                    {
                        return Some(AssetAction::OpenScene(absolute));
                    }
                }
                Ok(_) => self.error = Some("The selected asset is no longer a file.".to_owned()),
                Err(error) => self.error = Some(error),
            }
        }

        if let Some(entry) = self
            .entries
            .iter()
            .find(|entry| self.selected.as_ref() == Some(&entry.path))
        {
            let extension = entry
                .path
                .extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("");
            let kind = if entry.is_dir {
                "Folder"
            } else if extension.eq_ignore_ascii_case("glb") {
                "glTF binary model"
            } else if extension.eq_ignore_ascii_case("json")
                || extension.eq_ignore_ascii_case("hyperon")
            {
                "Scene (JSON)"
            } else {
                "File"
            };
            ui.separator();
            ui.label(format!(
                "{kind}  |  {}",
                Path::new("assets").join(&entry.path).display()
            ));
            if !entry.is_dir {
                ui.weak(format!("{} bytes", entry.size));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_row_accepts_single_and_double_clicks() {
        let context = egui::Context::default();
        let entry = Entry {
            name: "model.glb".into(),
            path: PathBuf::from("model.glb"),
            is_dir: false,
            size: 0,
        };
        let frame = |events: Vec<egui::Event>| {
            let mut rect = egui::Rect::NOTHING;
            let mut clicked = false;
            let mut double_clicked = false;
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(500.0, 300.0),
                )),
                events,
                ..Default::default()
            };
            let mut output = context.run_ui(input, |ui| {
                let response = entry_row(ui, &entry, false);
                rect = response.rect;
                clicked = response.clicked();
                double_clicked |= response.double_clicked();
            });
            output.textures_delta.clear();
            (rect, clicked, double_clicked)
        };
        let pos = frame(vec![]).0.center();
        frame(vec![egui::Event::PointerMoved(pos)]);
        let mut clicks = 0;
        let mut double_clicked = false;
        for _ in 0..2 {
            frame(vec![egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            }]);
            let (_, clicked, double) = frame(vec![egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            }]);
            clicks += usize::from(clicked);
            double_clicked |= double;
        }
        assert_eq!(clicks, 2);
        assert!(double_clicked);
    }
}
