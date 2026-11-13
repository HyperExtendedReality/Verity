//! The dockable panel layout, kept between sessions in the user's config folder.

use egui_dock::{DockState, Node, NodeIndex};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Panel {
    Viewport,
    Hierarchy,
    Inspector,
    Assets,
    Profiler,
    Console,
}

impl Panel {
    pub fn title(self) -> &'static str {
        match self {
            Self::Viewport => "Viewport",
            Self::Hierarchy => "Hierarchy",
            Self::Inspector => "Inspector",
            Self::Assets => "Content Browser",
            Self::Profiler => "Profiler",
            Self::Console => "Console",
        }
    }
}

/// Hierarchy left, Inspector right, content and console below the viewport.
pub fn default_layout() -> DockState<Panel> {
    let mut dock = DockState::new(vec![Panel::Viewport]);
    let tree = dock.main_surface_mut();
    let [center, _] = tree.split_left(NodeIndex::root(), 0.17, vec![Panel::Hierarchy]);
    let [center, _] = tree.split_right(center, 0.78, vec![Panel::Inspector]);
    tree.split_below(center, 0.7, vec![Panel::Assets, Panel::Console]);
    dock
}

fn layout_path() -> Option<PathBuf> {
    let config = std::env::var_os("APPDATA")
        .or_else(|| std::env::var_os("XDG_CONFIG_HOME"))
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))?;
    Some(config.join("Verity").join("editor_layout.json"))
}

/// The saved layout, or the default when there is none or it is unreadable.
pub fn load_layout() -> DockState<Panel> {
    layout_path()
        .and_then(|path| std::fs::read(path).ok())
        .and_then(|data| serde_json::from_slice::<DockState<Panel>>(&data).ok())
        // The viewport cannot be closed, so a layout without it is not ours.
        .filter(|dock| dock.find_tab(&Panel::Viewport).is_some())
        .unwrap_or_else(default_layout)
}

/// `dock` as JSON. Undrawn nodes hold infinite rects, which JSON writes as
/// null and cannot read back; the dock recomputes rects every frame anyway.
fn to_json(dock: &DockState<Panel>) -> serde_json::Result<Vec<u8>> {
    let mut dock = dock.clone();
    for (_, node) in dock.iter_all_nodes_mut() {
        node.set_rect(egui::Rect::ZERO);
        if let Node::Leaf(leaf) = node {
            leaf.viewport = egui::Rect::ZERO;
        }
    }
    serde_json::to_vec_pretty(&dock)
}

pub fn save_layout(dock: &DockState<Panel>) {
    let Some(path) = layout_path() else { return };
    let result = to_json(dock).map_err(|e| e.to_string()).and_then(|data| {
        std::fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
        std::fs::write(&path, data).map_err(|e| e.to_string())
    });
    if let Err(error) = result {
        eprintln!(
            "Cannot save the editor layout to {}: {error}",
            path.display()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_layout_round_trips_through_json() {
        let dock = default_layout();
        let json = to_json(&dock).unwrap();
        let back: DockState<Panel> = serde_json::from_slice(&json).unwrap();
        let tabs = |d: &DockState<Panel>| d.iter_all_tabs().map(|(_, &t)| t).collect::<Vec<_>>();
        assert_eq!(tabs(&back), tabs(&dock));
        assert!(back.find_tab(&Panel::Viewport).is_some());
    }
}
