//! The scene-authoring form of a physics body.

use glam::Vec3;
use hyperon_core::{Choice, Component, FieldMut};
use serde::{Deserialize, Serialize};

use crate::{Collider, PhysicsBody, PhysicsMaterial, RigidBody};

/// The analytic collider shapes a scene can author.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Shape {
    #[default]
    Cuboid,
    Sphere,
    Capsule,
}

impl Choice for Shape {
    fn options(&self) -> &'static [&'static str] {
        &["Cuboid", "Sphere", "Capsule"]
    }
    fn index(&self) -> usize {
        *self as usize
    }
    fn select(&mut self, index: usize) {
        *self = [Self::Cuboid, Self::Sphere, Self::Capsule][index.min(2)];
    }
}

impl Choice for RigidBody {
    fn options(&self) -> &'static [&'static str] {
        &["Static", "Kinematic", "Dynamic"]
    }
    fn index(&self) -> usize {
        *self as usize
    }
    fn select(&mut self, index: usize) {
        *self = [Self::Static, Self::Kinematic, Self::Dynamic][index.min(2)];
    }
}

/// A rigid body with one centered collider. Only the size fields of the
/// chosen shape apply.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RigidBodyComponent {
    pub body_type: RigidBody,
    pub shape: Shape,
    pub half_extents: [f32; 3],
    pub radius: f32,
    pub half_height: f32,
    pub density: f32,
    pub friction: f32,
    pub restitution: f32,
}

impl Default for RigidBodyComponent {
    fn default() -> Self {
        let material = PhysicsMaterial::default();
        Self {
            body_type: RigidBody::Dynamic,
            shape: Shape::Cuboid,
            // The editor's cube is 1 m across.
            half_extents: [0.5; 3],
            radius: 0.5,
            half_height: 0.5,
            density: material.density,
            friction: material.friction,
            restitution: material.restitution,
        }
    }
}

impl Component for RigidBodyComponent {
    const NAME: &'static str = "Rigid Body";

    fn fields(&mut self) -> Vec<(&'static str, FieldMut<'_>)> {
        let size = 0.001..=f32::MAX;
        vec![
            ("Body type", FieldMut::Choice(&mut self.body_type)),
            ("Shape", FieldMut::Choice(&mut self.shape)),
            ("Half extents", FieldMut::Vec3(&mut self.half_extents, size.clone())),
            ("Radius", FieldMut::Float(&mut self.radius, size.clone())),
            ("Half height", FieldMut::Float(&mut self.half_height, size)),
            ("Density", FieldMut::Float(&mut self.density, 0.001..=f32::MAX)),
            ("Friction", FieldMut::Float(&mut self.friction, 0.0..=f32::MAX)),
            ("Restitution", FieldMut::Float(&mut self.restitution, 0.0..=1.0)),
        ]
    }
}

impl RigidBodyComponent {
    /// The simulation body, or why the authored values are invalid.
    pub fn to_body(&self) -> Result<PhysicsBody, String> {
        let positive = |name: &str, v: f32| {
            (v.is_finite() && v > 0.0)
                .then_some(v)
                .ok_or_else(|| format!("{name} must be finite and positive"))
        };
        let collider = match self.shape {
            Shape::Cuboid => {
                for v in self.half_extents {
                    positive("half extents", v)?;
                }
                Collider::Cuboid {
                    half_extents: Vec3::from_array(self.half_extents),
                }
            }
            Shape::Sphere => Collider::Sphere {
                radius: positive("radius", self.radius)?,
            },
            Shape::Capsule => Collider::Capsule {
                radius: positive("radius", self.radius)?,
                half_height: positive("half height", self.half_height)?,
            },
        };
        positive("density", self.density)?;
        if !(self.friction.is_finite() && self.friction >= 0.0) {
            return Err("friction must be finite and nonnegative".into());
        }
        if !(0.0..=1.0).contains(&self.restitution) {
            return Err("restitution must be in [0, 1]".into());
        }
        let mut body = PhysicsBody::new(self.body_type, collider);
        body.material = PhysicsMaterial {
            density: self.density,
            friction: self.friction,
            restitution: self.restitution,
        };
        Ok(body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_authored_values_are_rejected_not_panicked() {
        assert!(RigidBodyComponent::default().to_body().is_ok());
        let bad = RigidBodyComponent {
            shape: Shape::Sphere,
            radius: 0.0,
            ..Default::default()
        };
        assert!(bad.to_body().unwrap_err().contains("radius"));
        let bad = RigidBodyComponent {
            restitution: f32::NAN,
            ..Default::default()
        };
        assert!(bad.to_body().is_err());
    }
}
