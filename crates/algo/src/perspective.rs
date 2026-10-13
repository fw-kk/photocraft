//! Edit › Perspective Warp: planes (quads) drawn over the image in Layout mode, whose corners are
//! then moved in Warp mode; the image follows a piecewise-homography map.
//!
//! Each plane maps its source quad onto its destination quad with a homography
//! `M_i = H_dst ∘ H_src⁻¹` (through the unit square, so `(u, v)` are the plane's projective
//! coordinates). Two homographies that agree on an edge's endpoints map the edge onto the same
//! segment but generally with a different spacing along it, so on its own a piecewise map tears
//! at shared edges. For every edge shared by two planes we therefore take the average of both
//! planes' images of each edge point as the target and add the difference, faded linearly
//! across the plane (weight `1 − u`, `u`, `1 − v` or `v` from that edge to the opposite one).
//! The correction vanishes at the corners (shared corners have one destination), so the map is
//! exactly continuous across shared edges, exactly the homography for a lone plane, and exactly
//! the identity when nothing moved.
//!
//! Outside every plane, a lone plane's homography carries on: the whole image lies on that
//! plane, as Photoshop has it. With two or more planes Photoshop does something else, measured
//! on its own output: the displacement the planes impose on their boundary is continued as a
//! harmonic field over the rest of the image (the Laplace equation with the planes as boundary
//! values and the image's edge free), so the surroundings follow a plane's edge where they meet
//! it and settle, a few plane-widths out, into the edge's mean shift at the original scale. A
//! corner wall drawn as three narrow planes would otherwise fling everything beside it through
//! the slivers' vanishing lines. That field is built by [`PerspectiveMap::with_surroundings`];
//! without it (the preview, a lone plane) points outside every plane use the nearest plane, with
//! `(u, v)` clamped for the corrections.
//!
//! Corners that coincide (within half a pixel) in the source are linked: they share one
//! destination (the mean of the given ones), as Photoshop's snapped planes do.

use serde::{Deserialize, Serialize};

use crate::transform::Homography;

/// One plane: source quad (Layout mode) and destination quad (Warp mode), corners clockwise from
/// top-left, document pixels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Plane {
    pub src: [[f64; 2]; 4],
    pub dst: [[f64; 2]; 4],
}

impl Plane {
    pub fn identity(q: [[f64; 2]; 4]) -> Self {
        Plane { src: q, dst: q }
    }
}

const LINK_EPS: f64 = 0.5;

/// Unit square corners in plane order.
const UNIT: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

fn unit_to_quad(q: &[[f64; 2]; 4]) -> Option<Homography> {
    Homography::rect_to_quad([0.0, 0.0, 1.0, 1.0], *q)
}

/// Groups of linked corners: `(plane, corner)` pairs that coincide in the source.
pub fn linked_corners(planes: &[Plane]) -> Vec<Vec<(usize, usize)>> {
    let mut groups: Vec<Vec<(usize, usize)>> = Vec::new();
    for (pi, p) in planes.iter().enumerate() {
        for (ci, c) in p.src.iter().enumerate() {
            let g = groups.iter_mut().find(|g| {
                let (a, b) = g[0];
                let o = planes[a].src[b];
                (o[0] - c[0]).abs() <= LINK_EPS && (o[1] - c[1]).abs() <= LINK_EPS
            });
            match g {
                Some(g) => g.push((pi, ci)),
                None => groups.push(vec![(pi, ci)]),
            }
        }
    }
    groups
}

/// Snaps linked corners to one source and one destination position (their means).
pub fn unify(planes: &mut [Plane]) {
    for g in linked_corners(planes) {
        if g.len() < 2 {
            continue;
        }
        let n = g.len() as f64;
        let s = g.iter().fold([0.0, 0.0], |a, &(p, c)| [a[0] + planes[p].src[c][0] / n, a[1] + planes[p].src[c][1] / n]);
        let d = g.iter().fold([0.0, 0.0], |a, &(p, c)| [a[0] + planes[p].dst[c][0] / n, a[1] + planes[p].dst[c][1] / n]);
        for &(p, c) in &g {
            planes[p].src[c] = s;
            planes[p].dst[c] = d;
        }
    }
}

/// Edge `k` of a plane runs from corner `k` to corner `k + 1`.
fn edge_uv(k: usize, u: f64, v: f64) -> ([f64; 2], f64) {
    // (point on the edge in plane coordinates, weight of that edge's correction)
    match k {
        0 => ([u, 0.0], 1.0 - v),
        1 => ([1.0, v], u),
        2 => ([u, 1.0], v),
        _ => ([0.0, v], 1.0 - u),
    }
}

struct PlaneMap {
    src: Homography,
    src_inv: Homography,
    dst: Homography,
    quad: [[f64; 2]; 4],
    /// For each edge: the neighbouring plane sharing it, if any.
    shared: [Option<usize>; 4],
}

/// A prepared perspective warp (forward map, source → destination).
pub struct PerspectiveMap {
    planes: Vec<PlaneMap>,
    /// The displacement of the image outside the planes, when there are several: see the
    /// module notes and [`PerspectiveMap::with_surroundings`].
    outside: Option<Field>,
}

/// The displacement of the image outside the planes: a harmonic field, solved on nested grids.
/// The coarse grid spans the image, so the far field settles against the image's own edge; the
/// finer grids span the planes' neighbourhood, where the field changes quickly and a coarse cell
/// put the content beside a plane several pixels off (on a 2000 px wide placement, 10 px for 12 px
/// cells, under 2 px for 6 px cells, measured against Photoshop). Each finer grid takes its border
/// values from the grid above it, so the levels join continuously; a point is sampled from the
/// finest grid that holds it.
struct Field {
    levels: Vec<Grid>,
}

/// A displacement grid, bilinearly sampled; constant beyond its edge.
struct Grid {
    x0: f64,
    y0: f64,
    cell: f64,
    nx: usize,
    ny: usize,
    dx: Vec<f64>,
    dy: Vec<f64>,
}

/// Grid cells across the longer side of the image on the coarse level.
const FIELD_CELLS: usize = 200;
/// Each finer level's cells are this many times smaller, over the planes grown by this share of
/// their extent.
const FIELD_LEVELS: [(f64, f64); 2] = [(4.0, 1.0), (16.0, 0.25)];
/// No level has more cells than this; its cell grows to keep under it.
const FIELD_MAX_CELLS: usize = 300_000;
/// The field is converged when no cell moves more than this (pixels) in a sweep.
const FIELD_TOLERANCE: f64 = 0.02;
const FIELD_MAX_SWEEPS: usize = 20_000;

impl Field {
    fn solve(map: &PerspectiveMap, bounds: [f64; 4]) -> Option<Field> {
        if !bounds.iter().all(|v| v.is_finite()) || !(bounds[2] > bounds[0] && bounds[3] > bounds[1]) {
            return None;
        }
        let (w, h) = (bounds[2] - bounds[0], bounds[3] - bounds[1]);
        let coarse_cell = (w.max(h) / FIELD_CELLS as f64).max(2.0);
        let mut levels = vec![Grid::solve(map, bounds, coarse_cell, None)?];
        let (px0, py0, px1, py1) = map.planes.iter().flat_map(|pm| pm.quad.iter()).fold(
            (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
            |(a, b, c, d), p| (a.min(p[0]), b.min(p[1]), c.max(p[0]), d.max(p[1])),
        );
        let extent = (px1 - px0).max(py1 - py0).max(1.0);
        for (ratio, margin) in FIELD_LEVELS {
            let cell = coarse_cell / ratio;
            if cell < 2.0 || cell >= levels.last()?.cell {
                continue;
            }
            let inner = [
                (px0 - margin * extent).max(bounds[0]),
                (py0 - margin * extent).max(bounds[1]),
                (px1 + margin * extent).min(bounds[2]),
                (py1 + margin * extent).min(bounds[3]),
            ];
            if !(inner[2] > inner[0] && inner[3] > inner[1]) {
                continue;
            }
            if let Some(grid) = Grid::solve(map, inner, cell, levels.last()) {
                levels.push(grid);
            }
        }
        Some(Field { levels })
    }

    fn sample(&self, x: f64, y: f64) -> (f64, f64) {
        let grid = self.levels.iter().rev().find(|g| g.holds(x, y)).unwrap_or(&self.levels[0]);
        grid.sample(x, y)
    }
}

impl Grid {
    /// Solves the Laplace equation for the displacement over `bounds` at `cell`: fixed where
    /// the planes are (their own map, less the identity); at the grid's edge, fixed to `parent`
    /// where there is one and free otherwise. Gauss-Seidel with over-relaxation, red-black,
    /// until converged.
    fn solve(map: &PerspectiveMap, bounds: [f64; 4], cell: f64, parent: Option<&Grid>) -> Option<Grid> {
        let (w, h) = (bounds[2] - bounds[0], bounds[3] - bounds[1]);
        let cell = cell.max(((w * h) / FIELD_MAX_CELLS as f64).sqrt()).max(2.0);
        let nx = ((w / cell).ceil() as usize + 2).max(3);
        let ny = ((h / cell).ceil() as usize + 2).max(3);
        let (x0, y0) = (bounds[0] - cell / 2.0, bounds[1] - cell / 2.0);
        let n = nx * ny;
        let mut fixed = vec![false; n];
        let (mut dx, mut dy) = (vec![0.0; n], vec![0.0; n]);
        let centre = |i: usize, j: usize| (x0 + (i as f64 + 0.5) * cell, y0 + (j as f64 + 0.5) * cell);
        // Cells whose centre lies in a plane, and cells a plane's edge crosses, so a plane
        // narrower than a cell still pins the field.
        for j in 0..ny {
            for i in 0..nx {
                let (x, y) = centre(i, j);
                if map.inside(x, y).is_some() {
                    let (mx, my) = map.map_planes(x, y);
                    let k = j * nx + i;
                    fixed[k] = true;
                    dx[k] = mx - x;
                    dy[k] = my - y;
                }
            }
        }
        for pm in &map.planes {
            for k in 0..4 {
                let (a, b) = (pm.quad[k], pm.quad[(k + 1) % 4]);
                let steps = (((b[0] - a[0]).hypot(b[1] - a[1]) / (cell / 2.0)).ceil() as usize).max(1);
                for s in 0..=steps {
                    let t = s as f64 / steps as f64;
                    let (x, y) = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]));
                    let (fi, fj) = (((x - x0) / cell).floor(), ((y - y0) / cell).floor());
                    if fi < 0.0 || fj < 0.0 || fi >= nx as f64 || fj >= ny as f64 {
                        continue;
                    }
                    let (i, j) = (fi as usize, fj as usize);
                    let k = j * nx + i;
                    if !fixed[k] {
                        let (cx, cy) = centre(i, j);
                        let (mx, my) = map.map_planes(cx, cy);
                        fixed[k] = true;
                        dx[k] = mx - cx;
                        dy[k] = my - cy;
                    }
                }
            }
        }
        if !fixed.iter().any(|f| *f) || fixed.iter().all(|f| *f) {
            return None;
        }
        // A start that is already right far away: the parent's field, or the mean fixed
        // displacement. A finer grid's border is the parent's value, held.
        let count = fixed.iter().filter(|f| **f).count() as f64;
        let mean = |v: &[f64]| v.iter().zip(&fixed).filter(|(_, f)| **f).map(|(x, _)| x).sum::<f64>() / count;
        let (mx, my) = (mean(&dx), mean(&dy));
        for j in 0..ny {
            for i in 0..nx {
                let k = j * nx + i;
                if fixed[k] {
                    continue;
                }
                let (x, y) = centre(i, j);
                match parent {
                    Some(p) => {
                        let (px, py) = p.sample(x, y);
                        dx[k] = px;
                        dy[k] = py;
                        if i == 0 || j == 0 || i + 1 == nx || j + 1 == ny {
                            fixed[k] = true;
                        }
                    }
                    None => {
                        dx[k] = mx;
                        dy[k] = my;
                    }
                }
            }
        }
        let omega = 2.0 / (1.0 + (std::f64::consts::PI / nx.max(ny) as f64).sin());
        for field in [&mut dx, &mut dy] {
            for _ in 0..FIELD_MAX_SWEEPS {
                let mut worst: f64 = 0.0;
                for colour in 0..2 {
                    for j in 0..ny {
                        for i in 0..nx {
                            if (i + j) % 2 != colour {
                                continue;
                            }
                            let k = j * nx + i;
                            if fixed[k] {
                                continue;
                            }
                            let (mut sum, mut cnt) = (0.0, 0.0);
                            if i > 0 { sum += field[k - 1]; cnt += 1.0; }
                            if i + 1 < nx { sum += field[k + 1]; cnt += 1.0; }
                            if j > 0 { sum += field[k - nx]; cnt += 1.0; }
                            if j + 1 < ny { sum += field[k + nx]; cnt += 1.0; }
                            let step = omega * (sum / cnt - field[k]);
                            field[k] += step;
                            worst = worst.max(step.abs());
                        }
                    }
                }
                if worst < FIELD_TOLERANCE {
                    break;
                }
            }
        }
        Some(Grid { x0, y0, cell, nx, ny, dx, dy })
    }

    /// Whether the point lies within the grid's cell centres (where it interpolates rather
    /// than holds its edge value).
    fn holds(&self, x: f64, y: f64) -> bool {
        let fx = (x - self.x0) / self.cell - 0.5;
        let fy = (y - self.y0) / self.cell - 0.5;
        fx >= 0.0 && fy >= 0.0 && fx <= (self.nx - 1) as f64 && fy <= (self.ny - 1) as f64
    }

    fn sample(&self, x: f64, y: f64) -> (f64, f64) {
        let fx = ((x - self.x0) / self.cell - 0.5).clamp(0.0, (self.nx - 1) as f64 - 1e-9);
        let fy = ((y - self.y0) / self.cell - 0.5).clamp(0.0, (self.ny - 1) as f64 - 1e-9);
        let (i, j) = (fx.floor() as usize, fy.floor() as usize);
        let (a, b) = (fx - i as f64, fy - j as f64);
        let at = |v: &[f64]| {
            let k = j * self.nx + i;
            (1.0 - a) * (1.0 - b) * v[k] + a * (1.0 - b) * v[k + 1] + (1.0 - a) * b * v[k + self.nx] + a * b * v[k + self.nx + 1]
        };
        (at(&self.dx), at(&self.dy))
    }
}

/// Straighten modes (Photoshop's Warp-mode buttons).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Straighten {
    Horizontal,
    Vertical,
    Auto,
}

impl Straighten {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "h" | "horizontal" | "level" => Some(Straighten::Horizontal),
            "v" | "vertical" => Some(Straighten::Vertical),
            "auto" | "both" => Some(Straighten::Auto),
            _ => None,
        }
    }
}

/// Makes the destination edges that are near-horizontal exactly horizontal and/or near-vertical
/// exactly vertical (within 45°), moving linked corners together.
pub fn straighten(planes: &mut [Plane], mode: Straighten) {
    unify(planes);
    let groups = linked_corners(planes);
    let group_of = |p: usize, c: usize| groups.iter().position(|g| g.contains(&(p, c))).unwrap_or(0);
    let mut pos: Vec<[f64; 2]> = groups.iter().map(|g| planes[g[0].0].dst[g[0].1]).collect();
    let edges: Vec<(usize, usize)> =
        (0..planes.len()).flat_map(|p| (0..4).map(move |k| (p, k))).map(|(p, k)| (group_of(p, k), group_of(p, (k + 1) % 4))).collect();
    for (a, b) in edges {
        let (pa, pb) = (pos[a], pos[b]);
        let (dx, dy) = ((pb[0] - pa[0]).abs(), (pb[1] - pa[1]).abs());
        if dy <= dx && matches!(mode, Straighten::Horizontal | Straighten::Auto) {
            let y = (pa[1] + pb[1]) / 2.0;
            pos[a][1] = y;
            pos[b][1] = y;
        } else if dx < dy && matches!(mode, Straighten::Vertical | Straighten::Auto) {
            let x = (pa[0] + pb[0]) / 2.0;
            pos[a][0] = x;
            pos[b][0] = x;
        }
    }
    for (g, p) in groups.iter().zip(pos) {
        for &(pi, ci) in g {
            planes[pi].dst[ci] = p;
        }
    }
}

impl PerspectiveMap {
    /// Prepares the map. `None` if a quad is degenerate (or there are no planes).
    pub fn new(planes: &[Plane]) -> Option<Self> {
        if planes.is_empty() {
            return None;
        }
        let mut planes = planes.to_vec();
        unify(&mut planes);
        let near = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).abs() <= LINK_EPS && (a[1] - b[1]).abs() <= LINK_EPS;
        let mut maps = Vec::with_capacity(planes.len());
        for (i, p) in planes.iter().enumerate() {
            let src = unit_to_quad(&p.src)?;
            let src_inv = src.inverse()?;
            let dst = unit_to_quad(&p.dst)?;
            let mut shared = [None; 4];
            for (k, s) in shared.iter_mut().enumerate() {
                let (a, b) = (p.src[k], p.src[(k + 1) % 4]);
                *s = planes
                    .iter()
                    .enumerate()
                    .find(|(j, o)| {
                        *j != i
                            && (0..4).any(|m| {
                                let (c, d) = (o.src[m], o.src[(m + 1) % 4]);
                                (near(a, c) && near(b, d)) || (near(a, d) && near(b, c))
                            })
                    })
                    .map(|(j, _)| j);
            }
            maps.push(PlaneMap { src, src_inv, dst, quad: p.src, shared });
        }
        Some(PerspectiveMap { planes: maps, outside: None })
    }

    /// The map with Photoshop's treatment of the image outside the planes, for an image that
    /// spans `bounds` (`[x0, y0, x1, y1]`, document pixels): with two or more planes, the
    /// surroundings follow a harmonic continuation of the planes' boundary displacement rather
    /// than the nearest plane's homography (see the module notes). A lone plane is unchanged.
    pub fn with_surroundings(mut self, bounds: [f64; 4]) -> Self {
        if self.planes.len() >= 2 {
            self.outside = Field::solve(&self, bounds);
        }
        self
    }

    /// The plane containing `p` and `p`'s plane coordinates, if any does.
    fn inside(&self, x: f64, y: f64) -> Option<(usize, f64, f64)> {
        self.planes.iter().enumerate().find_map(|(i, pm)| {
            let (u, v) = pm.src_inv.apply(x, y);
            ((-1e-9..=1.0 + 1e-9).contains(&u) && (-1e-9..=1.0 + 1e-9).contains(&v)).then_some((i, u, v))
        })
    }

    /// The plane containing `p` (or the nearest one) and `p`'s plane coordinates.
    fn locate(&self, x: f64, y: f64) -> (usize, f64, f64) {
        let mut best = (0, f64::MAX, 0.0, 0.0);
        for (i, pm) in self.planes.iter().enumerate() {
            let (u, v) = pm.src_inv.apply(x, y);
            if (-1e-9..=1.0 + 1e-9).contains(&u) && (-1e-9..=1.0 + 1e-9).contains(&v) {
                return (i, u, v);
            }
            let d = dist_to_quad(&pm.quad, [x, y]);
            if d < best.1 {
                best = (i, d, u, v);
            }
        }
        (best.0, best.2, best.3)
    }

    /// Maps a source point to the destination.
    pub fn map(&self, x: f64, y: f64) -> (f64, f64) {
        if let Some(field) = &self.outside
            && self.inside(x, y).is_none()
        {
            let (dx, dy) = field.sample(x, y);
            return (x + dx, y + dy);
        }
        self.map_planes(x, y)
    }

    /// The piecewise-homography map itself: the plane containing the point, or the nearest one.
    fn map_planes(&self, x: f64, y: f64) -> (f64, f64) {
        let (i, u, v) = self.locate(x, y);
        let pm = &self.planes[i];
        let (mut ox, mut oy) = pm.dst.apply(u, v);
        if !ox.is_finite() || !oy.is_finite() {
            return (x, y);
        }
        let (cu, cv) = (u.clamp(0.0, 1.0), v.clamp(0.0, 1.0));
        for (k, sh) in pm.shared.iter().enumerate() {
            let Some(j) = *sh else { continue };
            let (e, w) = edge_uv(k, cu, cv);
            if w <= 0.0 {
                continue;
            }
            let mine = pm.dst.apply(e[0], e[1]);
            let (qx, qy) = pm.src.apply(e[0], e[1]);
            let other = &self.planes[j];
            let (ou, ov) = other.src_inv.apply(qx, qy);
            let theirs = other.dst.apply(ou, ov);
            let target = ((mine.0 + theirs.0) / 2.0, (mine.1 + theirs.1) / 2.0);
            ox += w * (target.0 - mine.0);
            oy += w * (target.1 - mine.1);
        }
        (ox, oy)
    }

    /// Whether every plane maps onto itself (identity).
    pub fn is_identity(planes: &[Plane]) -> bool {
        planes.iter().all(|p| p.src.iter().zip(&p.dst).all(|(a, b)| (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9))
    }
}

fn dist_to_quad(q: &[[f64; 2]; 4], p: [f64; 2]) -> f64 {
    (0..4)
        .map(|k| {
            let (a, b) = (q[k], q[(k + 1) % 4]);
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let l2 = dx * dx + dy * dy;
            let t = if l2 > 0.0 { (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
            ((a[0] + t * dx - p[0]).powi(2) + (a[1] + t * dy - p[1]).powi(2)).sqrt()
        })
        .fold(f64::MAX, f64::min)
}

/// Corners of the unit square in plane order (for callers drawing plane grids).
pub fn unit_corners() -> [[f64; 2]; 4] {
    UNIT
}

/// A point of plane `p` at plane coordinates `(u, v)`, in source space.
pub fn plane_point(p: &Plane, u: f64, v: f64, dst: bool) -> [f64; 2] {
    let q = if dst { &p.dst } else { &p.src };
    unit_to_quad(q)
        .map(|h| {
            let (x, y) = h.apply(u, v);
            [x, y]
        })
        .unwrap_or(q[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: (f64, f64), b: (f64, f64), tol: f64) -> bool {
        (a.0 - b.0).abs() <= tol && (a.1 - b.1).abs() <= tol
    }

    #[test]
    fn identity_and_a_known_homography() {
        let q = [[10.0, 10.0], [90.0, 12.0], [85.0, 70.0], [12.0, 75.0]];
        let m = PerspectiveMap::new(&[Plane::identity(q)]).unwrap();
        for p in [(20.0, 20.0), (50.0, 40.0), (0.0, 0.0), (120.0, 90.0)] {
            assert!(close(m.map(p.0, p.1), p, 1e-9), "{p:?} → {:?}", m.map(p.0, p.1));
        }
        // Square → trapezoid: corners land exactly, the centre lands where the homography says.
        let src = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]];
        let dst = [[20.0, 0.0], [80.0, 0.0], [100.0, 100.0], [0.0, 100.0]];
        let m = PerspectiveMap::new(&[Plane { src, dst }]).unwrap();
        for (s, d) in src.iter().zip(&dst) {
            assert!(close(m.map(s[0], s[1]), (d[0], d[1]), 1e-9));
        }
        let h = Homography::rect_to_quad([0.0, 0.0, 100.0, 100.0], dst).unwrap();
        assert!(close(m.map(50.0, 50.0), h.apply(50.0, 50.0), 1e-9));
        // The diagonals' crossing stays the crossing: (50, 50) maps to the trapezoid's diagonal intersection.
        let c = m.map(50.0, 50.0);
        assert!((c.0 - 50.0).abs() < 1e-9 && c.1 < 50.0, "{c:?}");
    }

    #[test]
    fn the_surroundings_of_several_planes_follow_their_edges_and_settle_into_a_shift() {
        // A corner wall: a plane and a sliver beside it whose far edge is pushed up and right,
        // on an image 1000 px wide.
        let a = Plane { src: [[400.0, 100.0], [500.0, 100.0], [500.0, 400.0], [400.0, 400.0]], dst: [[400.0, 100.0], [500.0, 100.0], [500.0, 400.0], [400.0, 400.0]] };
        let b = Plane { src: [[500.0, 100.0], [530.0, 100.0], [530.0, 400.0], [500.0, 400.0]], dst: [[500.0, 100.0], [550.0, 60.0], [550.0, 420.0], [500.0, 400.0]] };
        let bounds = [0.0, 0.0, 1000.0, 500.0];
        let bare = PerspectiveMap::new(&[a.clone(), b.clone()]).unwrap();
        let m = PerspectiveMap::new(&[a.clone(), b.clone()]).unwrap().with_surroundings(bounds);
        // Inside the planes nothing changes.
        for p in [(450.0, 250.0), (515.0, 120.0)] {
            assert!(close(m.map(p.0, p.1), bare.map(p.0, p.1), 1e-9));
        }
        // Far to the right the sliver's homography would stretch the image by its own
        // vanishing geometry; Photoshop shifts it by about the edge's mean displacement
        // ((+20, 0) on average over the edge x = 530) and keeps its scale.
        let (p, q) = (m.map(900.0, 250.0), m.map(950.0, 250.0));
        assert!((p.0 - 920.0).abs() < 6.0 && p.1.abs() - 250.0 < 6.0, "{p:?}");
        assert!(((q.0 - p.0) - 50.0).abs() < 1.0, "scale kept: {p:?} {q:?}");
        let far = bare.map(900.0, 250.0);
        assert!((far.0 - 920.0).abs() > 50.0, "the homography alone would have put it at {far:?}");
        // Just outside the edge it still meets the edge's own map.
        let edge = bare.map(530.0, 250.0);
        let near = m.map(532.0, 250.0);
        assert!(close(near, (edge.0 + 2.0, edge.1), 4.0), "{near:?} vs {edge:?}");
        // To the left of the unmoved plane the image stays put.
        assert!(close(m.map(100.0, 250.0), (100.0, 250.0), 2.0));
        // A lone plane is the homography everywhere, surroundings or not.
        let lone = PerspectiveMap::new(&[b.clone()]).unwrap().with_surroundings(bounds);
        assert!(close(lone.map(900.0, 250.0), PerspectiveMap::new(&[b]).unwrap().map(900.0, 250.0), 1e-9));
    }

    #[test]
    fn shared_edges_stay_continuous() {
        // Two planes sharing the edge x = 50 (with slightly mismatched corners that get linked).
        let a = Plane { src: [[0.0, 0.0], [50.0, 0.0], [50.0, 100.0], [0.0, 100.0]], dst: [[0.0, 10.0], [50.0, 0.0], [55.0, 100.0], [0.0, 90.0]] };
        let b = Plane { src: [[50.2, 0.0], [100.0, 0.0], [100.0, 100.0], [50.0, 100.2]], dst: [[50.0, 0.0], [110.0, 20.0], [100.0, 80.0], [55.0, 100.0]] };
        let m = PerspectiveMap::new(&[a, b]).unwrap();
        // After linking, the shared edge runs from (50.1, 0) to (50, 100.1).
        for t in [0.05, 0.25, 0.5, 0.77, 0.95] {
            let (x, y) = (50.1 - 0.1 * t, 100.1 * t);
            let l = m.map(x - 1e-6, y);
            let r = m.map(x + 1e-6, y);
            assert!(close(l, r, 1e-4), "t={t}: {l:?} vs {r:?}");
        }
        assert_eq!(linked_corners(&[Plane::identity([[0.0; 2]; 4])]).len(), 1);
    }

    #[test]
    fn straighten_levels_and_plumbs_edges() {
        let mut ps =
            vec![Plane { src: [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]], dst: [[0.0, 3.0], [100.0, -2.0], [96.0, 100.0], [4.0, 104.0]] }];
        straighten(&mut ps, Straighten::Vertical);
        assert_eq!(ps[0].dst[1][0], ps[0].dst[2][0]);
        assert_eq!(ps[0].dst[0][0], ps[0].dst[3][0]);
        assert_ne!(ps[0].dst[0][1], ps[0].dst[1][1]);
        straighten(&mut ps, Straighten::Auto);
        assert_eq!(ps[0].dst[0][1], ps[0].dst[1][1]);
        assert_eq!(ps[0].dst[2][1], ps[0].dst[3][1]);
    }
}
