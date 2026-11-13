//! Procedural content. Cities first: a scene stores only a [`City`] recipe
//! (nodes, the spline edges between them, a seed and rules), and everything
//! else is a deterministic function of it: lanes, junction signals and
//! sidewalks ([`network`]), city blocks and lots ([`lots`]), buildings and
//! their interiors ([`building`]) and meshes ([`mesh`]). Traffic and
//! pedestrians ([`sim`]) are the engine's generic `verity-agents` driven
//! over the network this crate builds.

pub mod building;
pub mod lots;
pub mod mesh;
pub mod network;
pub mod sim;

use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};
use verity_nav::spline;

pub use building::Building;
pub use lots::{Block, Lot};
pub use mesh::Mesh;
pub use network::Network;
pub use sim::Simulation;
pub use verity_agents::{hash, unit};
pub use verity_nav::Path;

/// Ground height under a point of the plane, or `None` to keep authored heights.
pub type Height<'a> = &'a (dyn Fn(f32, f32) -> Option<f32> + Sync);

/// Ground that keeps every authored height.
pub fn flat(_: f32, _: f32) -> Option<f32> {
    None
}

/// Spacing of the points a spline is sampled at, in metres.
const SPACING: f32 = 2.0;
/// Most lanes each way an edge may have.
pub const MAX_LANES: u8 = 6;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Kind {
    #[default]
    Road,
    Highway,
    River,
}

impl Kind {
    pub const ALL: [Kind; 3] = [Kind::Road, Kind::Highway, Kind::River];

    pub fn name(self) -> &'static str {
        match self {
            Kind::Road => "Road",
            Kind::Highway => "Highway",
            Kind::River => "River",
        }
    }

    /// Lanes each way and surface width of a new edge of this kind.
    pub fn defaults(self) -> ([u8; 2], f32) {
        match self {
            Kind::Road => ([1, 1], 7.0),
            Kind::Highway => ([3, 3], 22.0),
            Kind::River => ([0, 0], 18.0),
        }
    }

    /// Speed limit in metres per second.
    pub fn speed(self) -> f32 {
        match self {
            Kind::Road => 13.9,
            Kind::Highway => 27.8,
            Kind::River => 0.0,
        }
    }

    /// Whether vehicles drive on it, and it bounds city blocks.
    pub fn drives(self) -> bool {
        self != Kind::River
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub position: [f32; 3],
}

/// A spline from node `a` through `controls` to node `b`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub a: u32,
    pub b: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub controls: Vec<[f32; 3]>,
    #[serde(default)]
    pub kind: Kind,
    /// Lanes from `a` to `b`, then from `b` to `a`.
    pub lanes: [u8; 2],
    /// Width of the carriageway or the water, in metres.
    pub width: f32,
}

/// How the city fills in between its roads, and how busy it is.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Rules {
    /// Width of a sidewalk on each side of a road.
    pub sidewalk: f32,
    /// Street frontage of a lot.
    pub lot_width: f32,
    pub lot_depth: f32,
    /// Gap between a lot's edges and its building.
    pub setback: f32,
    /// Fewest and most floors of a building.
    pub floors: [u32; 2],
    pub floor_height: f32,
    /// How far from the city's centre buildings stay tall, in metres.
    pub downtown: f32,
    /// Vehicles per kilometre of lane, and pedestrians per kilometre of sidewalk.
    pub vehicles_per_km: f32,
    pub walkers_per_km: f32,
    /// Seconds each approach of a signalled junction stays green.
    pub green: f32,
}

impl Default for Rules {
    fn default() -> Self {
        Self {
            sidewalk: 3.0,
            lot_width: 14.0,
            lot_depth: 18.0,
            setback: 1.5,
            floors: [2, 14],
            floor_height: 3.2,
            downtown: 250.0,
            vehicles_per_km: 40.0,
            walkers_per_km: 80.0,
            green: 12.0,
        }
    }
}

/// The recipe a scene saves.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct City {
    pub seed: u32,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub rules: Rules,
}

fn finite(p: [f32; 3]) -> bool {
    p.iter().all(|v| v.is_finite())
}

impl City {
    pub fn node(&self, index: u32) -> Vec3 {
        Vec3::from_array(self.nodes[index as usize].position)
    }

    /// Whether an edge can be built: scene files are not trusted.
    pub fn valid(&self, edge: &Edge) -> bool {
        let n = self.nodes.len() as u32;
        edge.a < n
            && edge.b < n
            && edge.a != edge.b
            && finite(self.nodes[edge.a as usize].position)
            && finite(self.nodes[edge.b as usize].position)
            && edge.controls.iter().all(|&c| finite(c))
            && edge.controls.len() <= 256
            && edge.width.is_finite()
            && (1.0..=200.0).contains(&edge.width)
            && edge.lanes.iter().all(|&l| l <= MAX_LANES)
    }

    /// The edge's spline points: node `a`, its controls, node `b`.
    pub fn points(&self, edge: usize) -> Vec<Vec3> {
        let e = &self.edges[edge];
        std::iter::once(self.node(e.a))
            .chain(e.controls.iter().map(|&c| Vec3::from_array(c)))
            .chain([self.node(e.b)])
            .collect()
    }

    /// The edge's centre line, laid on the ground.
    pub fn centre(&self, edge: usize, height: Height) -> Path {
        let points = spline::catmull_rom(&self.points(edge), SPACING);
        Path::new(points.into_iter().map(|p| match height(p.x, p.y) {
            Some(z) if z.is_finite() => Vec3::new(p.x, p.y, z),
            _ => p,
        }))
    }

    /// Node nearest `p` in the ground plane, within `radius`.
    pub fn nearest_node(&self, p: Vec3, radius: f32) -> Option<u32> {
        (self.nodes.iter().enumerate())
            .map(|(i, n)| (i as u32, Vec3::from_array(n.position).truncate().distance(p.truncate())))
            .filter(|&(_, d)| d <= radius)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }

    /// Valid edge whose curve passes nearest `p`, within `radius`, and the
    /// point on it.
    pub fn nearest_edge(&self, p: Vec3, radius: f32) -> Option<(usize, Vec3)> {
        (0..self.edges.len())
            .filter(|&e| self.valid(&self.edges[e]))
            .filter_map(|e| {
                let path = self.centre(e, &flat);
                let (s, d) = path.nearest(p);
                (d <= radius).then(|| (e, path.at(s).0, d))
            })
            .min_by(|a, b| a.2.total_cmp(&b.2))
            .map(|(e, p, _)| (e, p))
    }

    pub fn add_node(&mut self, p: Vec3) -> u32 {
        self.nodes.push(Node {
            position: p.to_array(),
        });
        self.nodes.len() as u32 - 1
    }

    /// Split an edge at the point of its curve nearest `p`, joining both
    /// halves at a new node there, which it returns.
    pub fn split_edge(&mut self, edge: usize, p: Vec3) -> u32 {
        let path = self.centre(edge, &flat);
        let (s, _) = path.nearest(p);
        let at = path.at(s).0;
        // Controls before the split stay with the first half.
        let controls = self.edges[edge].controls.clone();
        let cut = controls
            .iter()
            .take_while(|&&c| path.nearest(Vec3::from_array(c)).0 < s)
            .count();
        let node = self.add_node(at);
        let mut second = self.edges[edge].clone();
        second.a = node;
        second.controls = controls[cut..].to_vec();
        let first = &mut self.edges[edge];
        first.b = node;
        first.controls = controls[..cut].to_vec();
        self.edges.push(second);
        node
    }

    /// The node at `p`: an existing one within `snap`, else a new one, which
    /// splits an edge it lands on.
    pub fn node_at(&mut self, p: Vec3, snap: f32) -> u32 {
        if let Some(node) = self.nearest_node(p, snap) {
            return node;
        }
        match self.nearest_edge(p, snap) {
            Some((edge, at)) => self.split_edge(edge, at),
            None => self.add_node(p),
        }
    }

    /// Add an edge of `kind` through `points`, joining or splitting what it
    /// meets at its ends. `None` if it would start and end at one node.
    pub fn add_edge(&mut self, points: &[Vec3], kind: Kind, snap: f32) -> Option<usize> {
        let (&first, &last) = (points.first()?, points.last()?);
        if points.len() < 2 || !points.iter().all(|p| p.is_finite()) {
            return None;
        }
        let a = self.node_at(first, snap);
        let b = self.node_at(last, snap);
        if a == b {
            return None;
        }
        let (lanes, width) = kind.defaults();
        self.edges.push(Edge {
            a,
            b,
            controls: points[1..points.len() - 1].iter().map(|p| p.to_array()).collect(),
            kind,
            lanes,
            width,
        });
        Some(self.edges.len() - 1)
    }

    /// Remove an edge and any node it leaves unused.
    pub fn remove_edge(&mut self, edge: usize) {
        if edge >= self.edges.len() {
            return;
        }
        self.edges.remove(edge);
        let mut used = vec![false; self.nodes.len()];
        for e in &self.edges {
            for n in [e.a, e.b] {
                if let Some(u) = used.get_mut(n as usize) {
                    *u = true;
                }
            }
        }
        let mut remap = vec![u32::MAX; self.nodes.len()];
        let mut kept = 0;
        for (i, &u) in used.iter().enumerate() {
            if u {
                remap[i] = kept;
                kept += 1;
            }
        }
        let mut i = 0;
        self.nodes.retain(|_| {
            i += 1;
            used[i - 1]
        });
        for e in &mut self.edges {
            e.a = remap.get(e.a as usize).copied().unwrap_or(u32::MAX);
            e.b = remap.get(e.b as usize).copied().unwrap_or(u32::MAX);
        }
    }

    /// Move point `k` of an edge's spline: its first and last are its nodes.
    pub fn move_point(&mut self, edge: usize, k: usize, p: Vec3) {
        let Some(e) = self.edges.get_mut(edge) else {
            return;
        };
        let last = e.controls.len() + 1;
        let node = match k {
            0 => e.a,
            k if k == last => e.b,
            k if k < last => {
                e.controls[k - 1] = p.to_array();
                return;
            }
            _ => return,
        };
        if let Some(n) = self.nodes.get_mut(node as usize) {
            n.position = p.to_array();
        }
    }

    /// Mean position of the nodes in use, which the city is tallest around.
    pub fn centre_of_town(&self) -> Vec2 {
        let used: Vec<Vec2> = (self.edges.iter())
            .filter(|e| self.valid(e))
            .flat_map(|e| [self.node(e.a).truncate(), self.node(e.b).truncate()])
            .collect();
        used.iter().sum::<Vec2>() / used.len().max(1) as f32
    }
}

/// Everything static a city derives from its recipe.
pub struct Model {
    pub network: Network,
    pub blocks: Vec<Block>,
    pub buildings: Vec<Building>,
}

impl Model {
    pub fn build(city: &City, height: Height) -> Self {
        let network = Network::build(city, height);
        let blocks = lots::blocks(&network);
        let lots = lots::lots(city, &network, &blocks);
        // Lots come in a fixed order, so building seeds follow from the city's.
        let centre = city.centre_of_town();
        let buildings = (lots.into_iter())
            .map(|lot| Building::plan(lot, &city.rules, centre))
            .collect();
        Self {
            network,
            blocks,
            buildings,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A `size` by `size` grid of roads `spacing` apart, centred on the origin.
    pub fn grid(size: usize, spacing: f32) -> City {
        let mut city = City::default();
        let offset = (size - 1) as f32 * spacing * 0.5;
        for y in 0..size {
            for x in 0..size {
                city.add_node(Vec3::new(x as f32 * spacing - offset, y as f32 * spacing - offset, 0.0));
            }
        }
        let (lanes, width) = Kind::Road.defaults();
        let mut road = |a: usize, b: usize| {
            city.edges.push(Edge {
                a: a as u32,
                b: b as u32,
                controls: Vec::new(),
                kind: Kind::Road,
                lanes,
                width,
            })
        };
        for y in 0..size {
            for x in 0..size {
                let i = y * size + x;
                if x + 1 < size {
                    road(i, i + 1);
                }
                if y + 1 < size {
                    road(i, i + size);
                }
            }
        }
        city
    }

    #[test]
    fn drawing_onto_a_road_splits_it_into_a_junction() {
        let mut city = City::default();
        let straight = [Vec3::new(-50.0, 0.0, 0.0), Vec3::new(50.0, 0.0, 0.0)];
        city.add_edge(&straight, Kind::Road, 4.0).unwrap();
        let side = [Vec3::new(0.0, 40.0, 0.0), Vec3::new(1.0, 1.5, 0.0)];
        city.add_edge(&side, Kind::Road, 4.0).unwrap();
        assert_eq!(city.edges.len(), 3);
        assert_eq!(city.nodes.len(), 4);
        let junction = city.edges[1].a;
        assert!(city.node(junction).distance(Vec3::new(1.0, 0.0, 0.0)) < 0.5);
        assert!(city.edges.iter().all(|e| city.valid(e)));
        // Removing the side road leaves the split road and drops its far node.
        city.remove_edge(1);
        assert_eq!(city.nodes.len(), 3);
        assert!(city.edges.iter().all(|e| city.valid(e)));
    }

    #[test]
    fn the_recipe_round_trips_through_json() {
        let city = grid(3, 60.0);
        let json = serde_json::to_value(&city).unwrap();
        assert_eq!(serde_json::from_value::<City>(json).unwrap(), city);
    }

    #[test]
    fn invalid_edges_are_rejected() {
        let mut city = grid(2, 50.0);
        city.edges[0].b = 99;
        city.edges[1].width = f32::NAN;
        assert!(!city.valid(&city.edges[0]) && !city.valid(&city.edges[1]));
        // Building skips them rather than panicking.
        let model = Model::build(&city, &flat);
        assert!(!model.network.lanes.is_empty());
    }
}
