//! The dockable panel layout, kept between sessions in the user's config folder.

use egui_dock::{DockState, Node, NodeIndex, Split};
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
    pub const ALL: [Self; 6] = [
        Self::Viewport,
        Self::Hierarchy,
        Self::Inspector,
        Self::Assets,
        Self::Console,
        Self::Profiler,
    ];

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

/// Activate an open panel or restore it without rearranging the other panels.
pub fn open_panel(dock: &mut DockState<Panel>, panel: Panel) {
    if dock.find_tab(&panel).is_none() {
        let bottom = [Panel::Assets, Panel::Console, Panel::Profiler];
        if bottom.contains(&panel)
            && let Some(path) = bottom.iter().find_map(|tab| dock.find_tab(tab))
        {
            dock.set_focused_node_and_surface(path.node_path());
            dock.push_to_focused_leaf(panel);
        } else if let Some(viewport) = dock.find_tab(&Panel::Viewport) {
            let (split, fraction) = match panel {
                Panel::Hierarchy => (Split::Left, 0.22),
                Panel::Inspector => (Split::Right, 0.72),
                _ => (Split::Below, 0.70),
            };
            dock.split(
                viewport.node_path(),
                split,
                fraction,
                Node::leaf_with(vec![panel]),
            );
        } else {
            dock.push_to_first_leaf(panel);
        }
    }
    if let Some(path) = dock.find_tab(&panel) {
        dock.set_focused_node_and_surface(path.node_path());
        let _ = dock.set_active_tab(path);
    }
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

    #[test]
    fn window_menu_restores_and_activates_panels_without_duplicates() {
        let mut dock = default_layout();
        let inspector = dock.find_tab(&Panel::Inspector).unwrap();
        dock.remove_tab(inspector);
        for panel in [Panel::Inspector, Panel::Console, Panel::Profiler] {
            open_panel(&mut dock, panel);
            open_panel(&mut dock, panel);
            let path = dock.find_tab(&panel).unwrap();
            assert_eq!(dock.leaf(path.node_path()).unwrap().active, path.tab);
            assert_eq!(
                dock.iter_all_tabs()
                    .filter(|(_, tab)| **tab == panel)
                    .count(),
                1
            );
        }
        let viewport = dock.find_tab(&Panel::Viewport).unwrap();
        assert_eq!(
            dock.leaf(viewport.node_path()).unwrap().active,
            viewport.tab
        );
        let browser = dock.find_tab(&Panel::Assets).unwrap();
        let profiler = dock.find_tab(&Panel::Profiler).unwrap();
        assert_eq!(browser.node_path(), profiler.node_path());
    }
}
