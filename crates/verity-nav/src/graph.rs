//! Directed graphs in compressed sparse row form, with A* and Dijkstra.

use std::{cmp::Ordering, collections::BinaryHeap};

use glam::Vec3;

use crate::NONE;

/// Lists of `u32` per index, stored flat.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Csr {
    start: Vec<u32>,
    items: Vec<u32>,
}

impl Csr {
    /// From `(owner, item)` pairs, kept in their order per owner.
    pub fn new(owners: usize, pairs: impl IntoIterator<Item = (u32, u32)>) -> Self {
        let pairs: Vec<(u32, u32)> = pairs.into_iter().collect();
        let mut start = vec![0u32; owners + 1];
        for &(owner, _) in &pairs {
            start[owner as usize + 1] += 1;
        }
        for i in 0..owners {
            start[i + 1] += start[i];
        }
        let mut fill = start.clone();
        let mut items = vec![0; pairs.len()];
        for (owner, item) in pairs {
            items[fill[owner as usize] as usize] = item;
            fill[owner as usize] += 1;
        }
        Self { start, items }
    }

    pub fn len(&self) -> usize {
        self.start.len().saturating_sub(1)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, owner: u32) -> &[u32] {
        let i = owner as usize;
        match (self.start.get(i), self.start.get(i + 1)) {
            (Some(&a), Some(&b)) => &self.items[a as usize..b as usize],
            _ => &[],
        }
    }
}

/// A directed graph whose edges carry a cost and an optional gate.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub positions: Vec<Vec3>,
    /// Edge ids leaving each node.
    out: Csr,
    pub sources: Vec<u32>,
    pub targets: Vec<u32>,
    /// At least the straight distance between the ends, for A*.
    pub costs: Vec<f32>,
    /// A gate the caller opens and closes, or [`NONE`].
    pub gates: Vec<u32>,
}

/// One edge: from, to, cost and gate.
pub type GraphEdge = (u32, u32, f32, u32);

#[derive(PartialEq)]
struct Open(f32, u32);
impl Eq for Open {}
impl Ord for Open {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.total_cmp(&self.0).then(self.1.cmp(&other.1))
    }
}
impl PartialOrd for Open {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Graph {
    /// Edges whose ends are not nodes are dropped.
    pub fn new(positions: Vec<Vec3>, edges: impl IntoIterator<Item = GraphEdge>) -> Self {
        let n = positions.len() as u32;
        let edges: Vec<GraphEdge> = edges
            .into_iter()
            .filter(|&(a, b, c, _)| a < n && b < n && c.is_finite() && c >= 0.0)
            .collect();
        Self {
            out: Csr::new(n as usize, edges.iter().enumerate().map(|(i, e)| (e.0, i as u32))),
            sources: edges.iter().map(|e| e.0).collect(),
            targets: edges.iter().map(|e| e.1).collect(),
            costs: edges.iter().map(|e| e.2).collect(),
            gates: edges.iter().map(|e| e.3).collect(),
            positions,
        }
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Edge ids leaving `node`.
    pub fn edges(&self, node: u32) -> &[u32] {
        self.out.get(node)
    }

    /// The edge from `a` to `b`, if one exists.
    pub fn edge(&self, a: u32, b: u32) -> Option<u32> {
        self.edges(a).iter().copied().find(|&e| self.targets[e as usize] == b)
    }

    /// The cheapest node sequence from `from` to `to`, both included.
    pub fn astar(&self, from: u32, to: u32) -> Option<Vec<u32>> {
        let n = self.len();
        if from as usize >= n || to as usize >= n {
            return None;
        }
        let goal = self.positions[to as usize];
        let mut cost = vec![f32::INFINITY; n];
        let mut came = vec![NONE; n];
        let mut open = BinaryHeap::new();
        cost[from as usize] = 0.0;
        open.push(Open(self.positions[from as usize].distance(goal), from));
        while let Some(Open(_, node)) = open.pop() {
            if node == to {
                let mut path = vec![to];
                while let Some(&last) = path.last().filter(|&&l| l != from) {
                    path.push(came[last as usize]);
                }
                path.reverse();
                return Some(path);
            }
            let here = cost[node as usize];
            for &e in self.edges(node) {
                let next = self.targets[e as usize];
                let through = here + self.costs[e as usize];
                if through < cost[next as usize] {
                    cost[next as usize] = through;
                    came[next as usize] = node;
                    let h = self.positions[next as usize].distance(goal);
                    open.push(Open(through + h, next));
                }
            }
        }
        None
    }

    /// Each node's cheapest cost to reach any of `goals`; infinite if none.
    pub fn distances_to(&self, goals: &[u32]) -> Vec<f32> {
        let n = self.len();
        let incoming = Csr::new(n, (self.targets.iter().enumerate()).map(|(e, &t)| (t, e as u32)));
        let mut cost = vec![f32::INFINITY; n];
        let mut open = BinaryHeap::new();
        for &g in goals.iter().filter(|&&g| (g as usize) < n) {
            cost[g as usize] = 0.0;
            open.push(Open(0.0, g));
        }
        while let Some(Open(c, node)) = open.pop() {
            if c > cost[node as usize] {
                continue;
            }
            for &e in incoming.get(node) {
                let prev = self.sources[e as usize];
                let through = c + self.costs[e as usize];
                if through < cost[prev as usize] {
                    cost[prev as usize] = through;
                    open.push(Open(through, prev));
                }
            }
        }
        cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `w` by `h` lattice with unit edges both ways.
    fn lattice(w: u32, h: u32) -> Graph {
        let positions = (0..w * h).map(|i| Vec3::new((i % w) as f32, (i / w) as f32, 0.0)).collect();
        let mut edges = Vec::new();
        for i in 0..w * h {
            if i % w + 1 < w {
                edges.extend([(i, i + 1, 1.0, NONE), (i + 1, i, 1.0, NONE)]);
            }
            if i / w + 1 < h {
                edges.extend([(i, i + w, 1.0, NONE), (i + w, i, 1.0, NONE)]);
            }
        }
        Graph::new(positions, edges)
    }

    #[test]
    fn astar_finds_the_manhattan_optimum() {
        let graph = lattice(8, 6);
        let path = graph.astar(0, 8 * 6 - 1).unwrap();
        assert_eq!(path.len(), 7 + 5 + 1);
        assert_eq!((path[0], *path.last().unwrap()), (0, 47));
        for pair in path.windows(2) {
            assert!(graph.edge(pair[0], pair[1]).is_some());
        }
        assert_eq!(graph.astar(3, 3), Some(vec![3]));
    }

    #[test]
    fn distances_follow_edge_direction() {
        // 0 -> 1 -> 2, and 2 -> 0 costs 10.
        let positions = vec![Vec3::ZERO, Vec3::X, Vec3::X * 2.0];
        let graph = Graph::new(positions, [(0, 1, 1.0, NONE), (1, 2, 1.0, NONE), (2, 0, 10.0, NONE)]);
        assert_eq!(graph.distances_to(&[2]), vec![2.0, 1.0, 0.0]);
        assert_eq!(graph.distances_to(&[0]), vec![0.0, 11.0, 10.0]);
        assert_eq!(graph.astar(2, 1), Some(vec![2, 0, 1]));
    }

    #[test]
    fn csr_keeps_order_and_handles_empty_owners() {
        let csr = Csr::new(3, [(2, 7), (0, 1), (2, 8)]);
        assert_eq!((csr.get(0), csr.get(1), csr.get(2), csr.get(9)), (&[1][..], &[][..], &[7, 8][..], &[][..]));
    }
}
