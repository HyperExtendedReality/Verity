//! Standalone CPU model of the voxel.slang brick hop and occupancy lifecycle.
const CAP: i32 = 512;
const SIZE: i32 = 16;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Dda {
    c: [i32; 3],
    next: [f32; 3],
    t: f32,
    i: i32,
    clear: bool,
}

fn axis(e: [f32; 3]) -> usize {
    if e[0] <= e[1] && e[0] <= e[2] {
        0
    } else if e[1] <= e[2] {
        1
    } else {
        2
    }
}

fn inside(c: [i32; 3]) -> bool {
    c.into_iter().all(|x| (0..SIZE).contains(&x))
}

impl Dda {
    fn fine(&mut self, step: [i32; 3], delta: [f32; 3], end: f32) -> bool {
        let a = axis(self.next);
        self.t = self.next[a];
        self.next[a] += delta[a];
        self.c[a] += step[a];
        if self.t > end || !inside(self.c) {
            return false;
        }
        self.i += 1;
        true
    }

    fn hop(&mut self, low: [i32; 3], step: [i32; 3], delta: [f32; 3], end: f32) -> bool {
        let exit: [_; 3] = std::array::from_fn(|a| {
            let offset = ((self.c[a] + low[a]) & 3) as usize;
            if step[a] > 0 { 3 - offset } else { offset }
        });
        let mut events = [self.next; 5];
        for j in 1..5 {
            events[j] = std::array::from_fn(|a| events[j - 1][a] + delta[a]);
        }
        let edges = std::array::from_fn(|a| events[exit[a]][a]);
        let axis = axis(edges);
        let edge = edges[axis];
        if edge > end || edge.is_nan() || self.next.iter().any(|e| e.is_nan()) {
            return false;
        }
        let taken: [_; 3] = std::array::from_fn(|a| {
            events[..4]
                .iter()
                .filter(|e| e[a] < edge || (a <= axis && e[a] == edge))
                .count()
                .min(exit[a] + 1)
        });
        let c = std::array::from_fn(|a| self.c[a] + step[a] * taken[a] as i32);
        let count = taken.iter().sum::<usize>() as i32;
        if !inside(c) || self.i + count > CAP {
            return false;
        }
        *self = Self {
            c,
            next: std::array::from_fn(|a| events[taken[a]][a]),
            t: edge,
            i: self.i + count,
            clear: true,
        };
        true
    }
}

fn brick(c: [i32; 3], low: [i32; 3]) -> [i32; 3] {
    std::array::from_fn(|a| (c[a] + low[a]) >> 2)
}

#[test]
fn hops_match_every_fine_event_including_ties_and_rounded_plateaus() {
    let mut seed = 0xc01d_u32;
    let mut rand = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        seed >> 16
    };
    let mut hopped = 0;
    for k in 0..30_000 {
        let low = [-19, 4, 23];
        let next = std::array::from_fn(|_| match k % 3 {
            0 => (rand() % 4) as f32 * 0.25,
            1 => rand() as f32 * 0.00013,
            _ => 1.0e9,
        });
        let step = std::array::from_fn(|a| if (k >> a) & 1 == 0 { 1 } else { -1 });
        let delta = std::array::from_fn(|_| 0.25 + (rand() % 7) as f32 * 0.25);
        let initial = Dda {
            c: std::array::from_fn(|_| (rand() % SIZE as u32) as i32),
            next,
            t: 0.0,
            i: k % CAP,
            clear: false,
        };
        let end = [0.5, 4.0, 1.0e9][k as usize % 3];
        let mut fine = initial;
        let b = brick(fine.c, low);
        let can_hop = loop {
            fine.clear = true;
            if !fine.fine(step, delta, end) {
                break false;
            }
            if brick(fine.c, low) != b {
                break true;
            }
            if fine.i == CAP {
                break false;
            }
        };
        let mut coarse = initial;
        assert_eq!(coarse.hop(low, step, delta, end), can_hop);
        assert_eq!(coarse, if can_hop { fine } else { initial });
        hopped += i32::from(can_hop);
    }
    assert!(hopped > 1000);
}

#[test]
fn first_hit_start_solid_and_step_cap_match() {
    // Start in two solid cells; the empty first brick beyond them must set clear.
    // A hit on the last permitted visit is visible, not one just past it.
    let trace = |coarse, end, i| {
        let mut d = Dda {
            c: [0, 0, 0],
            next: [1.0, 99.0, 99.0],
            t: 0.0,
            i,
            clear: false,
        };
        let low = [-8, 0, 0];
        let (step, delta) = ([1; 3], [1.0; 3]);
        while d.i < CAP {
            if coarse && d.c[0] / 4 == 1 && d.hop(low, step, delta, end) {
                continue;
            }
            let found = matches!(d.c[0], 0 | 1 | 8);
            if found && (d.clear || d.i >= 3) {
                return (Some(d.c[0]), d);
            }
            d.clear |= !found;
            if !d.fine(step, delta, end) {
                break;
            }
        }
        (None, d)
    };
    for (end, i, hit) in [
        (20.0, 0, Some(8)),
        (8.0, 0, Some(8)),
        (7.999, 0, None),
        (20.0, CAP - 1, Some(0)),
    ] {
        assert_eq!(trace(true, end, i), trace(false, end, i));
        assert_eq!(trace(true, end, i).0, hit);
    }
    let start = Dda {
        c: [4, 0, 0],
        next: [1.0, 99.0, 99.0],
        t: 0.0,
        i: CAP - 4,
        clear: false,
    };
    let mut d = start;
    assert!(d.hop([0; 3], [1; 3], [1.0; 3], 4.0));
    assert_eq!((d.c[0], d.i, d.clear), (8, CAP, true));
    let mut d = Dda {
        i: CAP - 3,
        ..start
    };
    assert!(!d.hop([0; 3], [1; 3], [1.0; 3], 4.0));
}

#[test]
fn toroidal_scroll_and_partial_edits_preserve_untouched_finalized_occupancy() {
    let mut cells = [false; SIZE as usize];
    let mut counts = [0_i32; SIZE as usize / 4];
    let mut rebuild = |world: i32, survives: bool| {
        let slot = (world & (SIZE - 1)) as usize;
        let b = ((world >> 2) & (SIZE / 4 - 1)) as usize;
        counts[b] -= i32::from(cells[slot]); // clear releases the aliased old voxel
        cells[slot] = survives;
        counts[b] += i32::from(survives); // a rejected/unallocated candidate adds nothing
        assert_eq!(
            counts[b],
            cells[4 * b..4 * b + 4].iter().filter(|&&x| x).count() as i32
        );
        counts[b]
    };
    assert_eq!(rebuild(-4, true), 1);
    assert_eq!(rebuild(-3, true), 2);
    assert_eq!(rebuild(-4, false), 1); // partial edit cannot forget its occupied neighbor
    assert_eq!(rebuild(12, true), 2); // scrolled-in brick aliases the old storage
    assert_eq!(rebuild(13, false), 1);
    assert_eq!(rebuild(12, false), 0);
}
