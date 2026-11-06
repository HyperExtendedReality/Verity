//! The scene-authoring form of a joint, with material presets.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use verity_core::{Choice, Component, EntityHandle, FieldMut};

use crate::{Joint, Muscle, SoftBody, SoftMaterial, Strength};

/// Materials a joint can be made of. Presets use handbook strengths for a
/// section of the authored area; `Custom` uses the component's own fields.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum JointMaterial {
    #[default]
    Custom,
    Steel,
    Wood,
    Rubber,
    Bone,
    Ligament,
}

const MATERIALS: [JointMaterial; 6] = [
    JointMaterial::Custom,
    JointMaterial::Steel,
    JointMaterial::Wood,
    JointMaterial::Rubber,
    JointMaterial::Bone,
    JointMaterial::Ligament,
];

impl Choice for JointMaterial {
    fn options(&self) -> &'static [&'static str] {
        &["Custom", "Steel", "Wood", "Rubber", "Bone", "Ligament"]
    }
    fn index(&self) -> usize {
        *self as usize
    }
    fn select(&mut self, index: usize) {
        *self = MATERIALS[index.min(MATERIALS.len() - 1)];
    }
}

/// Yield and ultimate stress (Pa), plastic strain to failure as a share of the
/// joint's span, and Young's modulus for materials that visibly stretch.
struct Properties {
    yield_stress: f32,
    ultimate_stress: f32,
    ductility: f32,
    modulus: Option<f32>,
}

impl JointMaterial {
    fn properties(self) -> Option<Properties> {
        let (yield_stress, ultimate_stress, ductility, modulus) = match self {
            Self::Custom => return None,
            Self::Steel => (250.0e6, 400.0e6, 0.2, None),
            Self::Wood => (f32::INFINITY, 40.0e6, 0.0, None),
            Self::Rubber => (f32::INFINITY, 15.0e6, 0.0, Some(2.0e6)),
            Self::Bone => (f32::INFINITY, 130.0e6, 0.0, None),
            Self::Ligament => (50.0e6, 100.0e6, 0.1, None),
        };
        Some(Properties {
            yield_stress,
            ultimate_stress,
            ductility,
            modulus,
        })
    }
}

/// Pins this entity's body to another's, or to the world. Serialized numbers
/// use 0 for "none", as JSON has no infinity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct JointComponent {
    /// The other entity's name; empty pins to the world.
    pub connected_to: String,
    /// In this body's frame. The other anchor is wherever this one starts.
    pub anchor: [f32; 3],
    pub material: JointMaterial,
    /// Cross-section (cm²) and span (m) of a preset's material.
    pub section: f32,
    pub span: f32,
    /// Resist rotation as well as separation.
    pub welded: bool,
    /// Rotation range in degrees around this body's axes, if `limited`.
    pub limited: bool,
    pub limit_min: [f32; 3],
    pub limit_max: [f32; 3],
    /// Stiffness of the stops (N·m/rad); 0 is rigid.
    pub stop_stiffness: f32,
    /// A muscle around `muscle_axis` (this body's frame) holding `muscle_target`
    /// degrees. Size in cm², tension in N/cm², arm in m, stiffness in N·m/rad.
    pub muscle: bool,
    pub muscle_axis: [f32; 3],
    pub muscle_section: f32,
    pub muscle_tension: f32,
    pub muscle_arm: f32,
    pub activation: f32,
    pub muscle_target: f32,
    pub muscle_stiffness: f32,
    /// `Custom` material: stiffness (N/m), 0 rigid; stretch that doubles it (m),
    /// 0 never; and loads in N and N·m, 0 never.
    pub stiffness: f32,
    pub stiffening: f32,
    pub yield_force: f32,
    pub break_force: f32,
    pub stretch_to_break: f32,
    pub yield_torque: f32,
    pub break_torque: f32,
    /// Degrees of bending before breaking.
    pub bend_to_break: f32,
}

impl Default for JointComponent {
    fn default() -> Self {
        Self {
            connected_to: String::new(),
            anchor: [0.0; 3],
            material: JointMaterial::Custom,
            section: 1.0,
            span: 1.0,
            welded: false,
            limited: false,
            limit_min: [-45.0; 3],
            limit_max: [45.0; 3],
            stop_stiffness: 0.0,
            muscle: false,
            muscle_axis: [0.0, 1.0, 0.0],
            muscle_section: 20.0,
            muscle_tension: 30.0,
            muscle_arm: 0.05,
            activation: 1.0,
            muscle_target: 0.0,
            muscle_stiffness: 500.0,
            stiffness: 0.0,
            stiffening: 0.0,
            yield_force: 0.0,
            break_force: 0.0,
            stretch_to_break: 0.1,
            yield_torque: 0.0,
            break_torque: 0.0,
            bend_to_break: 30.0,
        }
    }
}

impl Component for JointComponent {
    const NAME: &'static str = "Joint";

    fn fields(&mut self) -> Vec<(&'static str, FieldMut<'_>)> {
        let positive = 0.0..=f32::MAX;
        let angle = -180.0..=180.0;
        vec![
            ("Connected to", FieldMut::Entity(&mut self.connected_to)),
            (
                "Anchor",
                FieldMut::Vec3(&mut self.anchor, f32::MIN..=f32::MAX),
            ),
            ("Material", FieldMut::Choice(&mut self.material)),
            (
                "Section cm²",
                FieldMut::Float(&mut self.section, 0.001..=f32::MAX),
            ),
            ("Span m", FieldMut::Float(&mut self.span, 0.001..=f32::MAX)),
            ("Welded", FieldMut::Bool(&mut self.welded)),
            ("Limit rotation", FieldMut::Bool(&mut self.limited)),
            (
                "Limit min °",
                FieldMut::Vec3(&mut self.limit_min, angle.clone()),
            ),
            ("Limit max °", FieldMut::Vec3(&mut self.limit_max, angle)),
            (
                "Stop stiffness N·m/rad",
                FieldMut::Float(&mut self.stop_stiffness, positive.clone()),
            ),
            ("Muscle", FieldMut::Bool(&mut self.muscle)),
            (
                "Muscle axis",
                FieldMut::Vec3(&mut self.muscle_axis, -1.0..=1.0),
            ),
            (
                "Muscle section cm²",
                FieldMut::Float(&mut self.muscle_section, positive.clone()),
            ),
            (
                "Specific tension N/cm²",
                FieldMut::Float(&mut self.muscle_tension, positive.clone()),
            ),
            (
                "Moment arm m",
                FieldMut::Float(&mut self.muscle_arm, positive.clone()),
            ),
            (
                "Activation",
                FieldMut::Float(&mut self.activation, 0.0..=1.0),
            ),
            (
                "Muscle target °",
                FieldMut::Float(&mut self.muscle_target, -180.0..=180.0),
            ),
            (
                "Muscle stiffness N·m/rad",
                FieldMut::Float(&mut self.muscle_stiffness, positive.clone()),
            ),
            (
                "Stiffness N/m",
                FieldMut::Float(&mut self.stiffness, positive.clone()),
            ),
            (
                "Stiffening m",
                FieldMut::Float(&mut self.stiffening, positive.clone()),
            ),
            (
                "Yield N",
                FieldMut::Float(&mut self.yield_force, positive.clone()),
            ),
            (
                "Break N",
                FieldMut::Float(&mut self.break_force, positive.clone()),
            ),
            (
                "Stretch to break m",
                FieldMut::Float(&mut self.stretch_to_break, positive.clone()),
            ),
            (
                "Yield N·m",
                FieldMut::Float(&mut self.yield_torque, positive.clone()),
            ),
            (
                "Break N·m",
                FieldMut::Float(&mut self.break_torque, positive.clone()),
            ),
            (
                "Bend to break °",
                FieldMut::Float(&mut self.bend_to_break, positive),
            ),
        ]
    }
}

/// `value`, with 0 meaning none.
fn or_infinite(value: f32) -> f32 {
    if value > 0.0 { value } else { f32::INFINITY }
}

impl JointComponent {
    /// The joint between `body_b` and `body_a`, whose anchor `anchor_a` the
    /// caller works out, or why the authored values are invalid.
    pub fn to_joint(
        &self,
        body_a: Option<EntityHandle>,
        body_b: EntityHandle,
        anchor_a: Vec3,
    ) -> Result<Joint, String> {
        let numbers = [
            self.section,
            self.span,
            self.stiffness,
            self.stiffening,
            self.yield_force,
            self.break_force,
            self.stretch_to_break,
            self.yield_torque,
            self.break_torque,
            self.bend_to_break,
            self.stop_stiffness,
            self.muscle_section,
            self.muscle_tension,
            self.muscle_arm,
            self.muscle_stiffness,
        ];
        if !numbers.iter().all(|v| v.is_finite() && *v >= 0.0) {
            return Err("values must be finite and nonnegative".into());
        }
        if !(self.section > 0.0 && self.span > 0.0) {
            return Err("section and span must be positive".into());
        }
        let anchor = Vec3::from_array(self.anchor);
        let (min, max) = (
            Vec3::from_array(self.limit_min),
            Vec3::from_array(self.limit_max),
        );
        if !anchor.is_finite() || !min.is_finite() || !max.is_finite() {
            return Err("anchor and limits must be finite".into());
        }
        if !min.cmple(max).all() {
            return Err("each rotation limit's minimum must not exceed its maximum".into());
        }
        let mut joint = Joint::new(body_a, body_b, anchor_a, anchor);
        if self.welded {
            joint.angular_stiffness = f32::INFINITY;
        }
        if self.limited {
            joint.limit_min = min * std::f32::consts::PI / 180.0;
            joint.limit_max = max * std::f32::consts::PI / 180.0;
            joint.limit_stiffness = or_infinite(self.stop_stiffness);
        }
        if self.muscle {
            let axis = Vec3::from_array(self.muscle_axis);
            if !axis.is_finite() || axis.length_squared() < 1.0e-6 {
                return Err("muscle axis must be nonzero".into());
            }
            if !(0.0..=1.0).contains(&self.activation) || !self.muscle_target.is_finite() {
                return Err("activation must be in [0, 1] and the target finite".into());
            }
            joint.muscle = Some(Muscle {
                axis: axis.normalize(),
                section: self.muscle_section,
                specific_tension: self.muscle_tension,
                moment_arm: self.muscle_arm,
                activation: self.activation,
                target: self.muscle_target.to_radians(),
                optimal_angle: self.muscle_target.to_radians(),
                angle_width: 2.0,
                max_speed: 15.0,
                stiffness: self.muscle_stiffness,
            });
        }
        match self.material.properties() {
            Some(material) => {
                let area = self.section * 1.0e-4;
                // A round section's bending moment at a stress is σπr³/4 = σAr/4.
                let lever = (area / std::f32::consts::PI).sqrt() / 4.0;
                let (yield_force, break_force) = (
                    material.yield_stress * area,
                    material.ultimate_stress * area,
                );
                joint.linear_strength = Strength {
                    yield_load: yield_force,
                    break_load: break_force,
                    ductility: material.ductility * self.span,
                };
                joint.angular_strength = Strength {
                    yield_load: yield_force * lever,
                    break_load: break_force * lever,
                    ductility: material.ductility,
                };
                if let Some(modulus) = material.modulus {
                    joint.linear_stiffness = modulus * area / self.span;
                    joint.stiffening = self.span;
                }
            }
            None => {
                joint.linear_stiffness = or_infinite(self.stiffness);
                joint.stiffening = or_infinite(self.stiffening);
                joint.linear_strength = custom(
                    self.yield_force,
                    self.break_force,
                    self.stretch_to_break,
                    "stretch to break",
                )?;
                joint.angular_strength = custom(
                    self.yield_torque,
                    self.break_torque,
                    self.bend_to_break.to_radians(),
                    "bend to break",
                )?;
            }
        }
        Ok(joint)
    }
}

fn custom(yields: f32, breaks: f32, ductility: f32, name: &str) -> Result<Strength, String> {
    let strength = Strength {
        yield_load: or_infinite(yields),
        break_load: or_infinite(breaks),
        ductility,
    };
    if strength.is_valid() {
        Ok(strength)
    } else {
        Err(format!("{name} must be positive for a joint that yields"))
    }
}

/// The shapes a soft body can take.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SoftShape {
    #[default]
    Cloth,
    Solid,
}

impl Choice for SoftShape {
    fn options(&self) -> &'static [&'static str] {
        &["Cloth", "Solid"]
    }
    fn index(&self) -> usize {
        *self as usize
    }
    fn select(&mut self, index: usize) {
        *self = [Self::Cloth, Self::Solid][index.min(1)];
    }
}

/// What a soft body is made of.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SoftPreset {
    #[default]
    Cloth,
    Metal,
    CrumpleZone,
    Rubber,
}

impl Choice for SoftPreset {
    fn options(&self) -> &'static [&'static str] {
        &["Cloth", "Metal", "Crumple zone", "Rubber"]
    }
    fn index(&self) -> usize {
        *self as usize
    }
    fn select(&mut self, index: usize) {
        *self = [Self::Cloth, Self::Metal, Self::CrumpleZone, Self::Rubber][index.min(3)];
    }
}

/// Which particles hold on at the start.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SoftPin {
    #[default]
    None,
    TopEdge,
    TwoCorners,
    AllEdges,
    /// Those inside the connected body: a shell around a rigid core.
    InsideBody,
}

impl Choice for SoftPin {
    fn options(&self) -> &'static [&'static str] {
        &[
            "None",
            "Top edge",
            "Two corners",
            "All edges",
            "Inside body",
        ]
    }
    fn index(&self) -> usize {
        *self as usize
    }
    fn select(&mut self, index: usize) {
        *self = [
            Self::None,
            Self::TopEdge,
            Self::TwoCorners,
            Self::AllEdges,
            Self::InsideBody,
        ][index.min(4)];
    }
}

/// A deformable body in place of this entity's cube. A cloth lies in the
/// entity's local XY plane, its top edge along +Y.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SoftBodyComponent {
    pub shape: SoftShape,
    pub material: SoftPreset,
    /// Full size in metres; a cloth uses X and Y.
    pub size: [f32; 3],
    /// Cells along each axis.
    pub cells: [f32; 3],
    /// Scales how much it takes to tear or dent.
    pub strength: f32,
    pub pin: SoftPin,
    /// The body pins hold to, by name; empty is the world.
    pub connected_to: String,
    /// Force that tears a pin out (N); 0 never.
    pub pin_strength: f32,
}

impl Default for SoftBodyComponent {
    fn default() -> Self {
        Self {
            shape: SoftShape::Cloth,
            material: SoftPreset::Cloth,
            size: [2.0, 2.0, 0.2],
            cells: [20.0, 20.0, 1.0],
            strength: 1.0,
            pin: SoftPin::TopEdge,
            connected_to: String::new(),
            pin_strength: 0.0,
        }
    }
}

impl Component for SoftBodyComponent {
    const NAME: &'static str = "Soft Body";

    fn fields(&mut self) -> Vec<(&'static str, FieldMut<'_>)> {
        vec![
            ("Shape", FieldMut::Choice(&mut self.shape)),
            ("Material", FieldMut::Choice(&mut self.material)),
            ("Size m", FieldMut::Vec3(&mut self.size, 0.001..=f32::MAX)),
            ("Cells", FieldMut::Vec3(&mut self.cells, 1.0..=128.0)),
            (
                "Strength ×",
                FieldMut::Float(&mut self.strength, 0.001..=f32::MAX),
            ),
            ("Pin", FieldMut::Choice(&mut self.pin)),
            ("Pinned to", FieldMut::Entity(&mut self.connected_to)),
            (
                "Pin strength N",
                FieldMut::Float(&mut self.pin_strength, 0.0..=f32::MAX),
            ),
        ]
    }
}

impl SoftBodyComponent {
    /// The soft body at a pose, or why the authored values are invalid.
    pub fn to_soft_body(&self, position: Vec3, rotation: Quat) -> Result<SoftBody, String> {
        let size = Vec3::from_array(self.size);
        let cells = Vec3::from_array(self.cells);
        if !size.is_finite() || size.min_element() <= 0.0 {
            return Err("size must be finite and positive".into());
        }
        if !cells.is_finite() || cells.min_element() < 1.0 || cells.max_element() > 128.0 {
            return Err("cells must be from 1 to 128".into());
        }
        if !(self.strength.is_finite() && self.strength > 0.0) {
            return Err("strength must be finite and positive".into());
        }
        if !(self.pin_strength.is_finite() && self.pin_strength >= 0.0) {
            return Err("pin strength must be finite and nonnegative".into());
        }
        let mut material = match self.material {
            SoftPreset::Cloth => SoftMaterial::cloth(),
            SoftPreset::Metal => SoftMaterial::sheet_metal(),
            SoftPreset::CrumpleZone => SoftMaterial::crumple_zone(),
            SoftPreset::Rubber => SoftMaterial::rubber(),
        };
        material.strength.yield_load *= self.strength;
        material.strength.break_load *= self.strength;
        let cells = cells.as_uvec3().to_array();
        Ok(match self.shape {
            SoftShape::Cloth => SoftBody::cloth(
                position,
                rotation,
                [size.x, size.y],
                [cells[0], cells[1]],
                material,
            ),
            SoftShape::Solid => SoftBody::lattice(position, rotation, size / 2.0, cells, material),
        })
    }

    /// Whether a particle starting at `point`, in this body's frame, is pinned.
    /// `InsideBody` is decided by the caller, which knows the other body.
    pub fn pins(&self, point: Vec3) -> bool {
        let (half, edge) = (Vec3::from_array(self.size) / 2.0, 1.0e-3);
        let at = |value: f32, limit: f32| value.abs() >= limit - edge;
        match self.pin {
            SoftPin::None | SoftPin::InsideBody => false,
            SoftPin::TopEdge => point.y >= half.y - edge,
            SoftPin::TwoCorners => point.y >= half.y - edge && at(point.x, half.x),
            SoftPin::AllEdges => at(point.x, half.x) || at(point.y, half.y),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use verity_core::{Entity, World};

    fn joint(component: &JointComponent) -> Result<Joint, String> {
        let entity = World::new().spawn(Entity::default());
        component.to_joint(None, entity, Vec3::ZERO)
    }

    #[test]
    fn defaults_are_an_unbreakable_ball_joint() {
        let made = joint(&JointComponent::default()).unwrap();
        assert!(made.linear_stiffness.is_infinite());
        assert!(made.linear_strength.break_load.is_infinite());
        assert!(made.linear_strength.yield_load.is_infinite());
        assert!(made.limit_max.x.is_infinite());
    }

    #[test]
    fn presets_scale_with_their_section() {
        let steel = JointComponent {
            material: JointMaterial::Steel,
            section: 2.0,
            ..Default::default()
        };
        let strength = joint(&steel).unwrap().linear_strength;
        assert!((strength.break_load - 80_000.0).abs() < 1.0);
        assert!((strength.yield_load - 50_000.0).abs() < 1.0);
        let rubber = JointComponent {
            material: JointMaterial::Rubber,
            ..Default::default()
        };
        assert!((joint(&rubber).unwrap().linear_stiffness - 200.0).abs() < 0.01);
    }

    #[test]
    fn invalid_authored_values_are_rejected_not_panicked() {
        let bad = JointComponent {
            yield_force: 10.0,
            break_force: 20.0,
            stretch_to_break: 0.0,
            ..Default::default()
        };
        assert!(joint(&bad).unwrap_err().contains("stretch"));
        let bad = JointComponent {
            limit_min: [10.0; 3],
            limit_max: [0.0; 3],
            limited: true,
            ..Default::default()
        };
        assert!(joint(&bad).unwrap_err().contains("limit"));
        let bad = JointComponent {
            section: f32::NAN,
            ..Default::default()
        };
        assert!(joint(&bad).is_err());
    }
}
