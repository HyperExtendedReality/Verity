//! Where things can go, for any game: curves ([`spline`]), graphs with A*
//! ([`graph`]), triangle navmeshes with smoothed paths ([`navmesh`]), flow
//! fields that steer many agents to one goal ([`flow`]), and the neighbour
//! queries crowds need ([`spatial`]). Z is up throughout.

pub mod flow;
pub mod graph;
pub mod navmesh;
pub mod spatial;
pub mod spline;

pub use flow::FlowField;
pub use graph::{Csr, Graph};
pub use navmesh::NavMesh;
pub use spatial::SpatialHash;
pub use spline::Path;

/// No node, triangle or gate.
pub const NONE: u32 = u32::MAX;
