//! Photoshop's Perspective Warp smart filter (`Fltr` class `perspectiveWarpTransform`), as
//! Photoshop writes it on a placed layer.
//!
//! The descriptor holds the planes the user drew in Layout mode and where their corners went in
//! Warp mode: `vertices` and `warpedVertices` (lists of `Pnt ` points, document pixels), and
//! `quads`, each a list of four `indices` into them. `referenceRect` is the placement when the
//! warp was made and `PuX0..3`/`PuY0..3` its corners; neither is needed to apply the warp.
//!
//! PhotoCraft has the same warp as `layer.smartObjects.perspectiveWarp`, so the filter imports
//! as that command with the planes translated:
//!
//!   * each plane's corners go clockwise from the top-left, as the command takes them; Photoshop
//!     lists them from wherever the plane was drawn from, and read in that order a corner wall's
//!     second plane came out sheared and no longer met the first;
//!   * a plane narrower than a few pixels is a seam Photoshop draws nothing for, and one whose
//!     warped width collapses to a sliver of its drawn width is a hinge it folded away: both are
//!     left out, since through the corners they share they bend the walls beside them.
//!
//! The item is kept verbatim as a template ([`TEMPLATE_KEY`]) and written back unchanged, so a
//! file round-trips exactly; the planes are read-only here, as the filter's dialog is Photoshop's.

use photocraft_doc::SmartFilter;
use photocraft_psd::descriptor::{Descriptor, Value};
use serde_json::{Value as J, json};

use super::{blend_options, from_hex, get_desc, set, to_hex};

pub(super) const CLASS: &str = "perspectiveWarpTransform";
pub(super) const COMMAND: &str = "layer.smartObjects.perspectiveWarp";
pub(super) const TEMPLATE_KEY: &str = "__perspectiveWarpPsd";
/// Pixels: a plane narrower than this on average is a seam, not a surface.
const MIN_PLANE_WIDTH: f64 = 8.0;
/// A plane whose warped width is under this share of its drawn width is a folded hinge.
const HINGE_COLLAPSE: f64 = 0.25;
/// The descriptor is small; room for a few hundred planes.
const MAX_TEMPLATE_BYTES: usize = 1 << 20;

type Quad = [[f64; 2]; 4];

fn points(f: &Descriptor, key: &str) -> Option<Vec<[f64; 2]>> {
    let Some(Value::List(list)) = f.get(key) else { return None };
    list.iter()
        .map(|v| match v {
            Value::Descriptor(p) => Some([crate::blocks::num(p.get("Hrzn"))?, crate::blocks::num(p.get("Vrtc"))?]),
            _ => None,
        })
        .collect()
}

fn quads(f: &Descriptor) -> Option<Vec<[usize; 4]>> {
    let Some(Value::List(list)) = f.get("quads") else { return None };
    list.iter()
        .map(|v| {
            let Value::Descriptor(q) = v else { return None };
            let Some(Value::List(ix)) = q.get("indices") else { return None };
            let ix: Vec<usize> = ix.iter().map(|i| match i { Value::Integer(n) => usize::try_from(*n).ok(), _ => None }).collect::<Option<_>>()?;
            <[usize; 4]>::try_from(ix).ok()
        })
        .collect()
}

/// The quad's area, shoelace, in square pixels.
fn area(q: &Quad) -> f64 {
    (0..4).map(|i| {
        let (a, b) = (q[i], q[(i + 1) % 4]);
        a[0] * b[1] - b[0] * a[1]
    }).sum::<f64>().abs() / 2.0
}

/// The quad's mean width across its longest side.
fn width(q: &Quad) -> f64 {
    let longest = (0..4).map(|i| {
        let (a, b) = (q[i], q[(i + 1) % 4]);
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
    }).fold(0.0, f64::max);
    if longest > 0.0 { area(q) / longest } else { 0.0 }
}

/// The corner order that goes clockwise (y down) around the quad's centre, starting from the
/// corner nearest the top-left.
fn clockwise(src: &[[f64; 2]], ix: [usize; 4]) -> [usize; 4] {
    let cx = ix.iter().map(|&i| src[i][0]).sum::<f64>() / 4.0;
    let cy = ix.iter().map(|&i| src[i][1]).sum::<f64>() / 4.0;
    let mut around = ix;
    around.sort_by(|&a, &b| {
        let ta = (src[a][1] - cy).atan2(src[a][0] - cx);
        let tb = (src[b][1] - cy).atan2(src[b][0] - cx);
        ta.total_cmp(&tb)
    });
    let start = (0..4).min_by(|&a, &b| {
        let sa = src[around[a]][0] + src[around[a]][1];
        let sb = src[around[b]][0] + src[around[b]][1];
        sa.total_cmp(&sb)
    }).unwrap_or(0);
    [around[start], around[(start + 1) % 4], around[(start + 2) % 4], around[(start + 3) % 4]]
}

/// The command's planes for a filter descriptor: `None` when it does not read as a Perspective
/// Warp, or when no plane with a surface is left.
pub(super) fn planes(f: &Descriptor) -> Option<Vec<J>> {
    let src = points(f, "vertices")?;
    let dst = points(f, "warpedVertices")?;
    if src.len() != dst.len() || !src.iter().chain(&dst).all(|p| p[0].is_finite() && p[1].is_finite()) {
        return None;
    }
    let mut out = Vec::new();
    for ix in quads(f)? {
        if ix.iter().any(|&i| i >= src.len()) {
            return None;
        }
        let order = clockwise(&src, ix);
        let s: Quad = order.map(|i| src[i]);
        let d: Quad = order.map(|i| dst[i]);
        let (ws, wd) = (width(&s), width(&d));
        if ws < MIN_PLANE_WIDTH || wd < HINGE_COLLAPSE * ws {
            continue;
        }
        out.push(json!({"src": s, "dst": d}));
    }
    (!out.is_empty()).then_some(out)
}

/// `layer.smartObjects.perspectiveWarp` params for a Perspective Warp item.
pub(super) fn import_params(item: &Descriptor) -> Option<J> {
    let f = get_desc(item, "Fltr")?;
    if !f.class_id.is(CLASS) {
        return None;
    }
    let planes = planes(f)?;
    // Neutral blending in the template: the model's own fields replace it on export.
    let mut template = item.clone();
    let neutral = SmartFilter { command: String::new(), params: J::Null, blend: photocraft_color::BlendMode::Normal, opacity: 1.0, visible: true };
    set(&mut template, "blendOptions", blend_options(&neutral));
    set(&mut template, "enab", Value::Boolean(true));
    Some(json!({"planes": planes, "interpolation": "bicubic", TEMPLATE_KEY: to_hex(&template.to_bytes())}))
}

/// The item as Photoshop wrote it: the planes are not edited here, so nothing is rebuilt.
pub(super) fn export_item(params: &J) -> Result<Descriptor, String> {
    let raw = params.get(TEMPLATE_KEY).ok_or("a Perspective Warp made in PhotoCraft is not written to PSD as Photoshop's smart filter")?;
    let raw = raw.as_str().ok_or("the Perspective Warp PSD data must be hex text")?;
    if raw.len() > MAX_TEMPLATE_BYTES * 2 {
        return Err("the Perspective Warp PSD data is too large".into());
    }
    let bytes = from_hex(raw).ok_or("the Perspective Warp PSD data is not valid hex")?;
    let item = Descriptor::from_bytes(&bytes).map_err(|e| format!("the Perspective Warp PSD data is unreadable ({e})"))?;
    if !item.class_id.is("filterFX") || get_desc(&item, "Fltr").is_none_or(|f| !f.class_id.is(CLASS)) {
        return Err("the Perspective Warp PSD data is not a Perspective Warp filter".into());
    }
    Ok(item)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pnt(x: f64, y: f64) -> Value {
        Value::Descriptor(Descriptor::new("Pnt ").with("Hrzn", Value::Double(x)).with("Vrtc", Value::Double(y)))
    }

    fn quad_of(ix: [i32; 4]) -> Value {
        Value::Descriptor(Descriptor::new("perspectiveWarpQuad").with("indices", Value::List(ix.iter().map(|&i| Value::Integer(i)).collect())))
    }

    fn fltr(src: &[(f64, f64)], dst: &[(f64, f64)], quads: &[[i32; 4]]) -> Descriptor {
        Descriptor::new(CLASS)
            .with("vertices", Value::List(src.iter().map(|&(x, y)| pnt(x, y)).collect()))
            .with("warpedVertices", Value::List(dst.iter().map(|&(x, y)| pnt(x, y)).collect()))
            .with("quads", Value::List(quads.iter().map(|q| quad_of(*q)).collect()))
    }

    fn corners(p: &J, key: &str) -> Vec<[f64; 2]> {
        p[key].as_array().unwrap().iter().map(|c| [c[0].as_f64().unwrap(), c[1].as_f64().unwrap()]).collect()
    }

    #[test]
    fn corners_go_clockwise_from_top_left_however_photoshop_listed_them() {
        // A corner wall: the second plane listed from its top-right, anticlockwise.
        let src = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0), (200.0, 0.0), (200.0, 100.0)];
        let dst = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0), (190.0, -20.0), (190.0, 120.0)];
        let planes = planes(&fltr(&src, &dst, &[[0, 1, 2, 3], [4, 1, 2, 5]])).unwrap();
        assert_eq!(planes.len(), 2);
        assert_eq!(corners(&planes[1], "src"), [[100.0, 0.0], [200.0, 0.0], [200.0, 100.0], [100.0, 100.0]]);
        assert_eq!(corners(&planes[1], "dst"), [[100.0, 0.0], [190.0, -20.0], [190.0, 120.0], [100.0, 100.0]]);
    }

    #[test]
    fn seams_and_folded_hinges_are_left_out() {
        // A 9 px strip between two walls that lands 0.5 px wide, and a 4 px seam.
        let src = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0), (109.0, 0.0), (109.0, 100.0), (300.0, 0.0), (300.0, 100.0), (304.0, 0.0), (304.0, 100.0)];
        let dst = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0), (100.5, 0.0), (100.5, 100.0), (300.0, 0.0), (300.0, 100.0), (302.0, 0.0), (302.0, 100.0)];
        let planes = planes(&fltr(&src, &dst, &[[0, 1, 2, 3], [4, 1, 2, 5], [6, 4, 5, 7], [8, 6, 7, 9]])).unwrap();
        assert_eq!(planes.iter().map(|p| corners(p, "src")[0]).collect::<Vec<_>>(), [[0.0, 0.0], [109.0, 0.0]]);
    }

    #[test]
    fn a_warp_with_no_surface_left_is_not_a_filter() {
        let src = [(0.0, 0.0), (2.0, 0.0), (2.0, 100.0), (0.0, 100.0)];
        assert!(planes(&fltr(&src, &src, &[[0, 1, 2, 3]])).is_none());
        assert!(planes(&fltr(&src, &src, &[[0, 1, 2, 9]])).is_none(), "an index past the vertices");
    }

    #[test]
    fn the_item_imports_as_the_command_and_exports_as_it_was() {
        let src = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
        let dst = [(5.0, -5.0), (100.0, 0.0), (100.0, 100.0), (0.0, 110.0)];
        let item = Descriptor::new("filterFX")
            .with("Nm  ", Value::Text(photocraft_psd::descriptor::UnicodeString::new_nul("Perspective Warp...")))
            .with("enab", Value::Boolean(false))
            .with("Fltr", Value::Descriptor(fltr(&src, &dst, &[[0, 1, 2, 3]])))
            .with("filterID", Value::Integer(999));
        let params = import_params(&item).unwrap();
        assert_eq!(params["planes"].as_array().unwrap().len(), 1);
        assert_eq!(params["interpolation"], "bicubic");
        let back = export_item(&params).unwrap();
        assert!(back.class_id.is("filterFX"));
        assert!(get_desc(&back, "Fltr").unwrap().class_id.is(CLASS));
        assert_eq!(get_desc(&back, "Fltr").unwrap().to_bytes(), fltr(&src, &dst, &[[0, 1, 2, 3]]).to_bytes());
    }
}
