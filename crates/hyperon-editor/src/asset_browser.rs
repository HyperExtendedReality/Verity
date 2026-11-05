use crate::ui::{
    ACCENT, ACCENT_DARK, AMBER, BACKGROUND, BORDER, CYAN, HOVER, MUTED, PINK, SURFACE, TEXT,
};
use egui::{Color32, FontId, Rect, Sense, Stroke, StrokeKind, vec2};
use std::collections::HashMap;
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

impl Entry {
    /// Badge text, color and description, by file type.
    fn kind(&self) -> (String, Color32, &'static str) {
        let ext = self
            .path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let (color, description) = match ext.as_str() {
            _ if self.is_dir => (AMBER, "Folder"),
            "glb" => (PINK, "glTF binary model"),
            "gltf" => (PINK, "glTF model (pack to GLB to import)"),
            "hyperon" | "json" => (ACCENT, "Scene"),
            "png" | "jpg" | "jpeg" | "ktx2" | "dds" | "hdr" | "exr" => (CYAN, "Texture"),
            _ => (MUTED, "File"),
        };
        (ext.to_ascii_uppercase(), color, description)
    }
}

fn human_size(bytes: u64) -> String {
    let mut size = bytes as f64;
    for unit in ["B", "KB", "MB", "GB"] {
        if size < 1024.0 || unit == "GB" {
            return if unit == "B" {
                format!("{bytes} B")
            } else {
                format!("{size:.1} {unit}")
            };
        }
        size /= 1024.0;
    }
    unreachable!()
}

fn folder_icon(painter: &egui::Painter, rect: Rect, color: Color32) {
    let tab = Rect::from_min_size(rect.min, vec2(rect.width() * 0.42, rect.height() * 0.3));
    painter.rect_filled(tab, 2, color.gamma_multiply(0.7));
    let body = Rect::from_min_max(rect.min + vec2(0.0, rect.height() * 0.16), rect.max);
    painter.rect_filled(body, 3, color);
}

/// One folder line in the tree: icon and name, highlighted when current.
fn tree_row(ui: &mut egui::Ui, name: &str, current: bool) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), 20.0), Sense::click());
    if current || response.hovered() {
        ui.painter()
            .rect_filled(rect, 4, if current { ACCENT_DARK } else { HOVER });
    }
    let icon = Rect::from_min_size(rect.left_center() + vec2(4.0, -6.0), vec2(14.0, 12.0));
    folder_icon(ui.painter(), icon, AMBER);
    ui.painter().with_clip_rect(rect).text(
        rect.left_center() + vec2(24.0, 0.0),
        egui::Align2::LEFT_CENTER,
        name,
        FontId::proportional(12.0),
        if current { ACCENT } else { TEXT },
    );
    response
}

/// A grid tile: a large type icon over the name, like Unity's Project window.
fn entry_tile(ui: &mut egui::Ui, entry: &Entry, selected: bool, size: f32) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(vec2(size, size + 30.0), Sense::click());
    let response = response.on_hover_text(&entry.name);
    let painter = ui.painter();
    if selected {
        painter.rect_filled(rect, 6, ACCENT_DARK);
        painter.rect_stroke(rect, 6, Stroke::new(1.0, ACCENT), StrokeKind::Inside);
    } else if response.hovered() {
        painter.rect_filled(rect, 6, HOVER);
    }
    let (badge, color, _) = entry.kind();
    let thumb = Rect::from_min_size(rect.min, vec2(size, size)).shrink(size * 0.14);
    if entry.is_dir {
        folder_icon(
            painter,
            thumb.shrink2(vec2(0.0, thumb.height() * 0.1)),
            color,
        );
    } else {
        // An asset card with a type-colored strip along its foot, like Unreal's.
        painter.rect_filled(thumb, 5, SURFACE);
        painter.rect_stroke(thumb, 5, Stroke::new(1.0, BORDER), StrokeKind::Inside);
        let strip = Rect::from_min_max(thumb.left_bottom() - vec2(0.0, 4.0), thumb.right_bottom());
        painter.rect_filled(
            strip,
            egui::CornerRadius {
                sw: 5,
                se: 5,
                ..Default::default()
            },
            color,
        );
        painter.text(
            thumb.center() - vec2(0.0, 2.0),
            egui::Align2::CENTER_CENTER,
            if badge.is_empty() { "FILE" } else { &badge },
            FontId::proportional((size * 0.17).clamp(9.0, 18.0)),
            color,
        );
    }
    let mut job = egui::text::LayoutJob::simple(
        entry.name.clone(),
        FontId::proportional(11.0),
        if selected {
            TEXT
        } else {
            TEXT.gamma_multiply(0.9)
        },
        size - 6.0,
    );
    job.wrap.max_rows = 2;
    job.wrap.break_anywhere = true;
    let galley = painter.layout_job(job);
    let pos = egui::pos2(
        rect.center().x - galley.size().x / 2.0,
        rect.top() + size - 2.0,
    );
    painter.galley(pos, galley, TEXT);
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
    /// Tile width in points.
    tile: f32,
    /// Subfolders of each folder the tree has shown, until the next refresh.
    folders: HashMap<PathBuf, Vec<PathBuf>>,
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
            tile: 84.0,
            folders: HashMap::new(),
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
        self.folders.clear();
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

    /// The subfolders of `dir`, skipping symlinks and unreadable entries as
    /// `refresh` does.
    fn subfolders(&mut self, dir: &Path) -> Vec<PathBuf> {
        self.folders
            .entry(dir.to_path_buf())
            .or_insert_with(|| {
                let Ok(entries) = Self::resolve(dir)
                    .and_then(|path| fs::read_dir(path).map_err(|error| error.to_string()))
                else {
                    return Vec::new();
                };
                let mut folders: Vec<PathBuf> = entries
                    .flatten()
                    .filter(|entry| fs::symlink_metadata(entry.path()).is_ok_and(|m| m.is_dir()))
                    .map(|entry| dir.join(entry.file_name()))
                    .filter(|path| Self::resolve(path).is_ok())
                    .collect();
                folders.sort_by_key(|path| path.to_string_lossy().to_lowercase());
                folders
            })
            .clone()
    }

    /// The folder tree under `dir`, recording a clicked folder in `destination`.
    fn folder_tree(
        &mut self,
        ui: &mut egui::Ui,
        dir: &Path,
        name: &str,
        destination: &mut Option<PathBuf>,
    ) {
        let children = self.subfolders(dir);
        let current = dir == self.current;
        if children.is_empty() {
            ui.horizontal(|ui| {
                ui.add_space(ui.spacing().indent);
                if tree_row(ui, name, current).clicked() {
                    *destination = Some(dir.to_path_buf());
                }
            });
            return;
        }
        let id = ui.make_persistent_id(("asset_tree", dir));
        let root = dir.as_os_str().is_empty();
        let (_, header, _) =
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, root)
                .show_header(ui, |ui| tree_row(ui, name, current))
                .body(|ui| {
                    for child in &children {
                        let name = child.file_name().unwrap_or_default().to_string_lossy();
                        self.folder_tree(ui, child, &name, destination);
                    }
                });
        if header.inner.clicked() {
            *destination = Some(dir.to_path_buf());
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
                .add_enabled(!self.history.is_empty(), egui::Button::new("⏴"))
                .on_hover_text("Back")
                .clicked();
            if ui
                .add_enabled(!self.current.as_os_str().is_empty(), egui::Button::new("⏶"))
                .on_hover_text("Up")
                .clicked()
            {
                destination = Some(self.current.parent().unwrap_or(Path::new("")).to_path_buf());
            }
            refresh = ui.button("⟳").on_hover_text("Refresh").clicked();
            ui.add_space(6.0);
            // Breadcrumbs: Assets › folder › folder.
            if ui
                .link(egui::RichText::new("Assets").strong().color(ACCENT))
                .clicked()
            {
                destination = Some(PathBuf::new());
            }
            let mut path = PathBuf::new();
            for component in self.current.components() {
                path.push(component.as_os_str());
                ui.label(egui::RichText::new("›").color(MUTED));
                if ui
                    .link(egui::RichText::new(component.as_os_str().to_string_lossy()).color(TEXT))
                    .clicked()
                {
                    destination = Some(path.clone());
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut self.search)
                        .hint_text("🔍 Search")
                        .desired_width(200.0),
                );
            });
        });
        ui.add_space(4.0);

        egui::Panel::left("asset_folders")
            .resizable(true)
            .default_size(190.0)
            .size_range(120.0..=360.0)
            .frame(
                egui::Frame::new()
                    .fill(BACKGROUND)
                    .corner_radius(6)
                    .inner_margin(6),
            )
            .show(ui, |ui| {
                egui::ScrollArea::both()
                    .id_salt("asset_folders_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.spacing_mut().item_spacing.y = 1.0;
                        self.folder_tree(ui, Path::new(""), "Assets", &mut destination);
                    });
            });

        let filter = self.search.to_lowercase();
        let visible = || {
            self.entries
                .iter()
                .filter(|entry| filter.is_empty() || entry.name.to_lowercase().contains(&filter))
        };
        let count = visible().count();
        let selected_entry = self
            .entries
            .iter()
            .find(|entry| self.selected.as_ref() == Some(&entry.path));
        let status = match selected_entry {
            Some(entry) => {
                let (_, _, description) = entry.kind();
                let path = Path::new("assets").join(&entry.path);
                if entry.is_dir {
                    format!("{description}  ·  {}", path.display())
                } else {
                    format!(
                        "{description}  ·  {}  ·  {}",
                        path.display(),
                        human_size(entry.size)
                    )
                }
            }
            None => format!("{count} item{}", if count == 1 { "" } else { "s" }),
        };
        egui::Panel::bottom("asset_status")
            .show_separator_line(false)
            .frame(egui::Frame::new().inner_margin(egui::Margin::symmetric(8, 3)))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(status).size(11.0).color(MUTED));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.spacing_mut().slider_width = 90.0;
                        ui.add(egui::Slider::new(&mut self.tile, 56.0..=140.0).show_value(false))
                            .on_hover_text("Tile size");
                    });
                });
            });

        let mut activated = None;
        egui::Frame::new()
            .fill(BACKGROUND)
            .corner_radius(6)
            .inner_margin(8)
            .outer_margin(egui::Margin {
                left: 4,
                ..Default::default()
            })
            .show(ui, |ui| {
                if let Some(error) = &self.error {
                    ui.colored_label(Color32::LIGHT_RED, error);
                }
                egui::ScrollArea::vertical()
                    .id_salt("asset_browser_entries")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.spacing_mut().item_spacing = vec2(6.0, 6.0);
                        ui.horizontal_wrapped(|ui| {
                            for entry in visible() {
                                let selected = self.selected.as_ref() == Some(&entry.path);
                                let response = entry_tile(ui, entry, selected, self.tile);
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
                        });
                        if count == 0 && self.error.is_none() {
                            ui.weak(if filter.is_empty() {
                                "This folder is empty."
                            } else {
                                "No assets match the search."
                            });
                        }
                    });
            });
        if back {
            if let Some(path) = self.history.pop() {
                self.current = path;
                self.selected = None;
                self.refresh();
            }
        } else if let Some(path) = destination {
            self.navigate(path);
        } else if refresh {
            self.refresh();
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
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_tile_accepts_single_and_double_clicks() {
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
                let response = entry_tile(ui, &entry, false, 84.0);
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
