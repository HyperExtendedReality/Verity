//! The scene's procedural city in the editor: its built model on the GPU as
//! retained meshes, the viewport tool that draws and edits its splines, and
//! the agents of a running scene drawn near the camera.

use std::{
    collections::{BTreeSet, HashSet},
    sync::Arc,
};

use egui::{Color32, Pos2, Rect, Sense, Stroke};
use glam::{Mat4, Vec2, Vec3};
use verity_core::{Material, Scene};
use verity_pcg::{City, Kind, Mesh, Model, Rules, mesh};
use verity_render::{Camera, DeformingMesh, Renderer, RetainedMesh, Vertex};
use verity_runtime::Runtime;
use verity_terra::Landscape;

/// Sets city keys apart from terrain chunks (low bits) and planet patches (bit 63).
const CITY: u64 = 1 << 62;
const EDGE: u64 = CITY | 1 << 56;
const JUNCTION: u64 = CITY | 2 << 56;
const BLOCK: u64 = CITY | 3 << 56;
const INTERIOR: u64 = CITY | 4 << 56;
/// Buildings this near the camera show their insides when asked.
const INTERIOR_RADIUS: f32 = 40.0;
/// Agents drawn: only those this near, and at most this many.
const AGENT_RADIUS: f32 = 250.0;
const MAX_AGENTS: usize = 8000;
/// How near, in points, a click must be to pick a line or point.
const PICK: f32 = 8.0;
/// Snap distance for road ends, in metres.
const SNAP: f32 = 6.0;

/// The scene's city recipe; one that does not parse counts as none.
pub fn recipe(scene: &Scene) -> Option<City> {
    serde_json::from_value(scene.city.clone()?).ok()
}

/// Ground under a point, from the landscape when there is one.
pub fn height(ground: &Option<Arc<Landscape>>) -> impl Fn(f32, f32) -> Option<f32> + Sync + '_ {
    move |x, y| ground.as_ref().map(|g| g.height(x, y))
}

/// The viewport tool's state.
#[derive(Default)]
pub struct CityTool {
    /// Drawing a new edge of this kind.
    pub drawing: Option<Kind>,
    /// Points of the edge being drawn.
    pub pending: Vec<Vec3>,
    pub edge: Option<usize>,
    /// The selected edge's point being dragged.
    dragging: Option<usize>,
    pub interiors: bool,
    /// Show lanes and sidewalks.
    pub overlay: bool,
    pub status: Option<String>,
    /// The landscape clicks land on; set each frame.
    pub ground: Option<Arc<Landscape>>,
    /// Lane and walk segments of the built city.
    pub lines: Arc<Vec<[Vec3; 2]>>,
}

fn retained(mesh: Mesh) -> Option<RetainedMesh> {
    (!mesh.is_empty()).then(|| RetainedMesh {
        bounds: mesh.bounds(),
        vertices: vertices(&mesh),
        indices: mesh.indices,
        plain: true,
    })
}

fn vertices(mesh: &Mesh) -> Vec<Vertex> {
    (mesh.positions.iter().zip(&mesh.normals).zip(&mesh.colors))
        .map(|((p, n), c)| Vertex {
            position: [p[0], p[1], p[2], 0.0],
            normal: [n[0], n[1], n[2], 0.0],
            tangent: [0.0; 4],
            color: [c[0], c[1], c[2], 1.0],
        })
        .collect()
}

#[derive(Default)]
pub struct CityView {
    /// The recipe and ground the model was built from.
    applied: Option<(City, usize)>,
    model: Option<Model>,
    rules: Rules,
    shown: HashSet<u64>,
    /// Buildings showing their insides.
    inside: BTreeSet<usize>,
    lines: Arc<Vec<[Vec3; 2]>>,
}

impl CityView {
    /// Follow the scene's recipe, laid on `ground`. While `holding` (a drag
    /// in progress) the old model stays, so editing stays smooth.
    pub fn update(
        &mut self,
        wanted: Option<City>,
        ground: &Option<Arc<Landscape>>,
        renderer: &mut Renderer,
        eye: Vec3,
        interiors: bool,
        holding: bool,
    ) {
        let key = wanted.map(|c| (c, ground.as_ref().map_or(0, |g| Arc::as_ptr(g) as usize)));
        if key != self.applied && !holding {
            for k in self.shown.drain() {
                renderer.set_retained_mesh(k, None);
            }
            self.inside.clear();
            self.model = key
                .as_ref()
                .map(|(city, _)| Model::build(city, &height(ground)));
            self.rules = key.as_ref().map(|(c, _)| c.rules).unwrap_or_default();
            self.applied = key;
            self.lines = Arc::new(self.model.as_ref().map(lines).unwrap_or_default());
            if let Some(model) = &self.model {
                let net = &model.network;
                for e in 0..net.edges.len() {
                    set(
                        &mut self.shown,
                        renderer,
                        EDGE | e as u64,
                        retained(mesh::edge(net, e)),
                    );
                }
                for n in 0..net.junctions.len() {
                    set(
                        &mut self.shown,
                        renderer,
                        JUNCTION | n as u64,
                        retained(mesh::junction(net, n)),
                    );
                }
                for b in 0..model.blocks.len() {
                    let block = self.block(b);
                    set(&mut self.shown, renderer, BLOCK | b as u64, retained(block));
                }
            }
        }
        // Insides of the buildings around the camera, in place of their shells.
        let Some(model) = &self.model else { return };
        let wanted: BTreeSet<usize> = if interiors {
            (model.buildings.iter().enumerate())
                .filter(|(_, b)| b.lot.centre.truncate().distance(eye.truncate()) < INTERIOR_RADIUS)
                .map(|(i, _)| i)
                .collect()
        } else {
            BTreeSet::new()
        };
        if wanted == self.inside {
            return;
        }
        let changed: BTreeSet<usize> = wanted.symmetric_difference(&self.inside).copied().collect();
        self.inside = wanted;
        let blocks: BTreeSet<u32> = changed
            .iter()
            .map(|&i| model.buildings[i].lot.block)
            .collect();
        for &i in &changed {
            let mesh = self.inside.contains(&i).then(|| {
                let mut mesh = Mesh::default();
                model.buildings[i].interior(&self.rules, &mut mesh);
                mesh
            });
            set(
                &mut self.shown,
                renderer,
                INTERIOR | i as u64,
                mesh.and_then(retained),
            );
        }
        for b in blocks {
            let block = self.block(b as usize);
            set(&mut self.shown, renderer, BLOCK | b as u64, retained(block));
        }
    }

    /// Shells of a block's buildings that are not showing their insides.
    fn block(&self, b: usize) -> Mesh {
        let mut mesh = Mesh::default();
        let Some(model) = &self.model else {
            return mesh;
        };
        for (i, building) in model.buildings.iter().enumerate() {
            if building.lot.block as usize == b && !self.inside.contains(&i) {
                building.shell(&self.rules, &mut mesh);
            }
        }
        mesh
    }

    pub fn lines(&self) -> Arc<Vec<[Vec3; 2]>> {
        self.lines.clone()
    }

    pub fn status(&self) -> Option<String> {
        let model = self.model.as_ref()?;
        Some(format!(
            "{} edges · {} lanes · {} blocks · {} buildings",
            model.network.edges.len(),
            model.network.lanes.len(),
            model.blocks.len(),
            model.buildings.len()
        ))
    }
}

fn set(shown: &mut HashSet<u64>, renderer: &mut Renderer, key: u64, mesh: Option<RetainedMesh>) {
    if mesh.is_some() {
        shown.insert(key);
    } else {
        shown.remove(&key);
    }
    renderer.set_retained_mesh(key, mesh);
}

/// Every lane and walk segment, raised a little to show over the surface.
fn lines(model: &Model) -> Vec<[Vec3; 2]> {
    let lift = Vec3::Z * 0.2;
    let net = &model.network;
    let lanes = net
        .lanes
        .paths
        .iter()
        .flat_map(|p| p.points.windows(2).map(|w| [w[0] + lift, w[1] + lift]));
    let walks = (0..net.walks.targets.len())
        .filter(|&e| net.walks.sources[e] < net.walks.targets[e])
        .map(|e| {
            [
                net.walks.positions[net.walks.sources[e] as usize] + lift,
                net.walks.positions[net.walks.targets[e] as usize] + lift,
            ]
        });
    lanes.chain(walks).collect()
}

/// Agents of a running scene near `eye`: crowds, pedestrians and vehicles,
/// as one mesh of simple figures.
pub fn agents(runtime: Option<&Runtime>, eye: Vec3) -> Option<DeformingMesh> {
    let runtime = runtime?;
    let mut mesh = Mesh::default();
    let mut drawn = 0;
    let near = |p: Vec3| p.distance_squared(eye) < AGENT_RADIUS * AGENT_RADIUS;
    let crowd = &runtime.agents.crowd;
    for i in 0..crowd.len() {
        if drawn < MAX_AGENTS && near(crowd.position[i]) {
            let color = runtime
                .agents
                .colors
                .get(crowd.group[i] as usize)
                .copied()
                .unwrap_or([0.8; 3]);
            figure(
                &mut mesh,
                crowd.position[i],
                crowd.velocity[i],
                crowd.radius[i],
                crowd.phase[i],
                color,
            );
            drawn += 1;
        }
    }
    if let Some(city) = &runtime.city {
        let people = &city.sim.crowd;
        for i in 0..people.len() {
            if drawn < MAX_AGENTS && near(people.position[i]) {
                let h = verity_pcg::hash(i as u32, 0xC0A7);
                let color = [
                    0.2 + verity_pcg::unit(h) * 0.6,
                    0.2 + verity_pcg::unit(h >> 3) * 0.5,
                    0.25 + verity_pcg::unit(h >> 7) * 0.5,
                ];
                figure(
                    &mut mesh,
                    people.position[i],
                    people.velocity[i],
                    people.radius[i],
                    people.phase[i],
                    color,
                );
                drawn += 1;
            }
        }
        let traffic = &city.sim.traffic;
        for i in 0..traffic.len() as u32 {
            let (front, heading) = traffic.pose(i);
            if drawn < MAX_AGENTS && near(front) {
                vehicle(&mut mesh, front, heading, traffic.length[i as usize], i);
                drawn += 1;
            }
        }
    }
    (!mesh.is_empty()).then(|| DeformingMesh {
        vertices: vertices(&mesh),
        indices: mesh.indices,
        material: Material {
            base_color: [1.0; 3],
            metallic: 0.0,
            roughness: 0.6,
        },
        textured: None,
    })
}

/// A body with legs that swing with the walk cycle.
fn figure(mesh: &mut Mesh, at: Vec3, velocity: Vec3, radius: f32, phase: f32, color: [f32; 3]) {
    let scale = radius / 0.3;
    let forward = velocity
        .truncate()
        .try_normalize()
        .unwrap_or(Vec2::X)
        .extend(0.0);
    let side = Vec3::Z.cross(forward);
    let height = 1.75 * scale;
    let hip = height * 0.5;
    mesh.cuboid(
        at + Vec3::Z * (hip + height * 0.22),
        [
            forward * 0.14 * scale,
            side * 0.22 * scale,
            Vec3::Z * height * 0.22,
        ],
        color,
        color,
    );
    mesh.cuboid(
        at + Vec3::Z * (height * 0.9),
        [
            forward * 0.1 * scale,
            side * 0.1 * scale,
            Vec3::Z * height * 0.08,
        ],
        [0.85, 0.68, 0.55],
        [0.15, 0.1, 0.08],
    );
    let swing = (phase * std::f32::consts::TAU).sin()
        * 0.25
        * scale
        * velocity.truncate().length().min(1.0);
    for (k, sign) in [(0.0, 1.0f32), (1.0, -1.0)] {
        let foot = at + side * (k - 0.5) * 0.2 * scale + forward * swing * sign;
        let knee = (at + Vec3::Z * hip + side * (k - 0.5) * 0.2 * scale + foot) * 0.5;
        mesh.cuboid(
            knee,
            [
                forward * 0.07 * scale,
                side * 0.07 * scale,
                Vec3::Z * hip * 0.5,
            ],
            [0.12, 0.12, 0.16],
            [0.12, 0.12, 0.16],
        );
    }
}

fn vehicle(mesh: &mut Mesh, front: Vec3, heading: Vec3, length: f32, i: u32) {
    let along = heading
        .truncate()
        .try_normalize()
        .unwrap_or(Vec2::X)
        .extend(0.0);
    let side = Vec3::Z.cross(along);
    let centre = front - along * length * 0.5;
    let h = verity_pcg::hash(i, 0xCA2);
    let paint = [
        verity_pcg::unit(h),
        verity_pcg::unit(h >> 5),
        verity_pcg::unit(h >> 11),
    ]
    .map(|c| 0.08 + c * 0.75);
    let tall = if length > 10.0 { 1.5 } else { 0.55 };
    mesh.cuboid(
        centre + Vec3::Z * (0.25 + tall * 0.5),
        [along * length * 0.5, side * 0.9, Vec3::Z * tall * 0.5],
        paint,
        paint,
    );
    if length <= 10.0 {
        let glass = [0.05, 0.07, 0.09];
        mesh.cuboid(
            centre - along * 0.3 + Vec3::Z * (0.25 + tall + 0.25),
            [along * length * 0.28, side * 0.82, Vec3::Z * 0.25],
            glass,
            paint,
        );
    }
}

/// World to viewport points, as the renderer projects.
fn view_proj(camera: &Camera, viewport: Rect) -> Option<Mat4> {
    let forward = camera.target - camera.eye;
    if viewport.width() < 16.0
        || viewport.height() < 16.0
        || !forward.is_finite()
        || forward.cross(verity_render::UP).length_squared() < 1e-8
        || !(0.01..3.13).contains(&camera.vertical_fov)
    {
        return None;
    }
    let mut projection = Mat4::perspective_infinite_reverse_rh(
        camera.vertical_fov,
        viewport.width() / viewport.height(),
        0.1,
    );
    projection.y_axis.y *= -1.0;
    Some(projection * Mat4::look_at_rh(camera.eye, camera.target, verity_render::UP))
}

fn project(m: &Mat4, viewport: Rect, p: Vec3) -> Option<Pos2> {
    let clip = *m * p.extend(1.0);
    if !clip.is_finite() || clip.w <= 0.1 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    Some(Pos2::new(
        viewport.left() + (ndc.x + 1.0) * 0.5 * viewport.width(),
        viewport.top() + (ndc.y + 1.0) * 0.5 * viewport.height(),
    ))
}

/// Where the ray under `point` meets the landscape, else the plane z = 0.
fn ground_hit(
    m: &Mat4,
    camera: &Camera,
    viewport: Rect,
    point: Pos2,
    ground: Option<&Landscape>,
) -> Option<Vec3> {
    let ndc = Vec3::new(
        (point.x - viewport.left()) * 2.0 / viewport.width() - 1.0,
        (point.y - viewport.top()) * 2.0 / viewport.height() - 1.0,
        1.0,
    );
    let far = m.inverse().project_point3(ndc);
    let direction = (far - camera.eye).try_normalize()?;
    let origin = camera.eye;
    if let Some(ground) = ground {
        let above = |t: f32| {
            let p = origin + direction * t;
            p.z - ground.height(p.x, p.y)
        };
        let (mut t, mut last) = (0.0, 0.0);
        while t < 4000.0 {
            let gap = above(t);
            if gap <= 0.0 {
                let (mut lo, mut hi) = (last, t);
                for _ in 0..20 {
                    let mid = (lo + hi) * 0.5;
                    if above(mid) > 0.0 { lo = mid } else { hi = mid }
                }
                return Some(origin + direction * hi);
            }
            last = t;
            t += (gap * 0.5).clamp(0.25, 20.0);
        }
    }
    (direction.z < -1e-4).then(|| origin + direction * (-origin.z / direction.z))
}

fn polyline(m: &Mat4, viewport: Rect, points: &[Vec3]) -> Vec<Pos2> {
    points
        .iter()
        .filter_map(|&p| project(m, viewport, p))
        .collect()
}

fn distance_to(points: &[Pos2], p: Pos2) -> f32 {
    points
        .windows(2)
        .map(|w| {
            let (a, b) = (w[0], w[1]);
            let ab = b - a;
            let t = ((p - a).dot(ab) / ab.length_sq().max(1e-6)).clamp(0.0, 1.0);
            p.distance(a + ab * t)
        })
        .fold(f32::INFINITY, f32::min)
}

fn color(kind: Kind) -> Color32 {
    match kind {
        Kind::Road => crate::ui::AMBER,
        Kind::Highway => crate::ui::PINK,
        Kind::River => crate::ui::CYAN,
    }
}

/// The city tool in the viewport: draws the splines, places new ones, picks
/// and drags points. Returns whether the recipe changed.
pub fn viewport(
    ui: &mut egui::Ui,
    viewport: Rect,
    scene: &mut Scene,
    camera: &Camera,
    tool: &mut CityTool,
) -> bool {
    let Some(m) = view_proj(camera, viewport) else {
        return false;
    };
    let mut city = recipe(scene).unwrap_or_default();
    let before = city.clone();
    let painter = ui.painter_at(viewport);
    if tool.overlay {
        for [a, b] in tool.lines.iter() {
            if let (Some(a), Some(b)) = (project(&m, viewport, *a), project(&m, viewport, *b)) {
                painter.line_segment(
                    [a, b],
                    Stroke::new(1.0, crate::ui::CYAN.gamma_multiply(0.5)),
                );
            }
        }
    }
    let curves: Vec<Option<Vec<Pos2>>> = (0..city.edges.len())
        .map(|e| {
            city.valid(&city.edges[e]).then(|| {
                polyline(
                    &m,
                    viewport,
                    &verity_nav::spline::catmull_rom(&city.points(e), 3.0),
                )
            })
        })
        .collect();
    if tool.edge.is_some_and(|e| e >= city.edges.len()) {
        tool.edge = None;
    }
    for (e, curve) in curves.iter().enumerate() {
        if let Some(curve) = curve {
            let selected = tool.edge == Some(e);
            let width = if selected { 3.0 } else { 1.5 };
            let stroke = Stroke::new(
                width,
                if selected {
                    crate::ui::ACCENT
                } else {
                    color(city.edges[e].kind)
                },
            );
            painter.add(egui::Shape::line(curve.clone(), stroke));
        }
    }
    for node in &city.nodes {
        if let Some(p) = project(&m, viewport, Vec3::from_array(node.position)) {
            painter.circle_filled(p, 3.0, crate::ui::TEXT);
        }
    }
    let points: Vec<Option<Pos2>> = tool
        .edge
        .map(|e| {
            city.points(e)
                .iter()
                .map(|&p| project(&m, viewport, p))
                .collect()
        })
        .unwrap_or_default();
    for p in points.iter().flatten() {
        painter.circle_stroke(*p, 5.0, Stroke::new(2.0, crate::ui::ACCENT));
    }
    let response = ui.interact(
        viewport,
        ui.make_persistent_id("city_tool"),
        Sense::click_and_drag(),
    );
    let pointer = response.interact_pointer_pos().or(response.hover_pos());
    let hit = |p: Pos2| ground_hit(&m, camera, viewport, p, tool.ground.as_deref());
    if let Some(kind) = tool.drawing {
        let mut preview = tool.pending.clone();
        if let Some(p) = pointer.and_then(hit) {
            preview.push(p);
        }
        painter.add(egui::Shape::line(
            polyline(&m, viewport, &preview),
            Stroke::new(2.0, color(kind)),
        ));
        if response.clicked()
            && let Some(p) = pointer.and_then(hit)
        {
            tool.pending.push(p);
        }
        let (enter, escape) = ui.input(|i| {
            (
                i.key_pressed(egui::Key::Enter),
                i.key_pressed(egui::Key::Escape),
            )
        });
        if enter || response.double_clicked() {
            // A double-click also placed its point twice.
            tool.pending.dedup_by(|a, b| a.distance(*b) < 0.5);
            if let Some(edge) = city.add_edge(&tool.pending, kind, SNAP) {
                tool.edge = Some(edge);
            }
            tool.pending.clear();
        }
        if escape {
            tool.pending.clear();
            tool.drawing = None;
        }
    } else {
        if response.drag_started()
            && let Some(at) = pointer
        {
            // `points` is empty without a selected edge.
            tool.dragging = (points.iter().enumerate())
                .filter_map(|(k, p)| p.map(|p| (k, p.distance(at))))
                .filter(|&(_, d)| d < PICK)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(k, _)| k);
        }
        if response.dragged()
            && let (Some(e), Some(k), Some(p)) = (tool.edge, tool.dragging, pointer.and_then(hit))
        {
            city.move_point(e, k, p);
        }
        if response.drag_stopped() {
            tool.dragging = None;
        }
        if response.clicked()
            && let Some(at) = pointer
        {
            tool.edge = (curves.iter().enumerate())
                .filter_map(|(e, c)| c.as_ref().map(|c| (e, distance_to(c, at))))
                .filter(|&(_, d)| d < PICK)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(e, _)| e);
        }
    }
    if city == before {
        return false;
    }
    scene.city = Some(serde_json::to_value(&city).expect("city recipes serialize"));
    true
}
