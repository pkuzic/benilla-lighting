//! MONKEY (daylight fixtures): **the sun, as an interior-lane light, standing in the doorway.**
//!
//! THE SEAM. With `interiorLight` on, a WMO's interior groups light from the room law —
//! `interiorAmbient` + the fixtures that claim the room, and no sun (`static_gx.wgsl`'s
//! `interior_room_light`, folded through the `1 - exp(-x*interiorExposure)` rolloff). Everything
//! around a doorway does NOT: an EXTERIOR-class group (the Goldshire inn's shell), and an
//! EXT/TRANS-class BATCH of an interior group, both keep the reference sky law by day
//! (`static_gx.wgsl` ~:1242 — `day_w = {TRANS: MOCV alpha, EXT: 1} x sun_w`, blended toward the
//! reference result). So at the threshold one plank is `ambient + diffuse*N.L` and the next is
//! `interiorAmbient` — a hard line, exactly where the artists' MOCV bake used to carry the sun
//! bleed inward and the dynamic lane threw it away. At NIGHT the seam is already gone: `sun_w` is
//! 0, `day_w` collapses, and both sides are on the room law.
//!
//! THE FIX. Give the room law a sun to find: an ordinary interior-lane point light standing in the
//! doorway, coloured like the sky and calibrated so the room law's result AT THE DOOR equals the
//! reference sunlit result the batch a plank away is rendering. The seam then closes by
//! construction on both sides, and the light falls off inward the way daylight does. It rides the
//! existing machinery whole — `LightRooms` / `LightLitRooms` claims through
//! [`benilla_formats::room_claim`], `LightLane` interior, `LightReach`, the packer's soft window —
//! so nothing in the shader, the claim table or the std430 layout changes.
//!
//! WHERE A DOORWAY IS. Two seeds, and they are the client's own two populations:
//!  1. **PORTAL** — a portal whose two sides are one INTERIOR-class group and one EXTERIOR-class
//!     group (or which only one group names at all: a portal straight to the outside). This is
//!     byte-for-byte the set `FixColorVertexAlpha` (`0x6c43d0`, `benilla_formats`'s
//!     `fix_color_vertex_alpha`) whitens a group's MOCV across — "the fade is specific to
//!     interior<->exterior openings". The reference lights its dark transition corridors with
//!     exactly this relation, so it is the authored answer to "which opening lets daylight in",
//!     not a heuristic of ours.
//!  2. **APERTURE** — an **EXT-class MOBA batch of an INTERIOR group**. That is the other half of
//!     the same fact: the batch class IS the artist saying "this surface takes the full exterior
//!     day/night law", and a batch that says so inside a room is a window or an open side. It is
//!     also the population that draws the seam from the inside (the room's INT-class floor meeting
//!     its own EXT-class window wall), which no portal describes.
//!  3. **BOUNDARY** (MONKEY (daylight fixtures: boundary)) — the GROUP-LAW seam itself, taken off
//!     the meshes: the vertices an INTERIOR-class group shares (to within [`SEAM_EPS`]) with an
//!     EXTERIOR-class group of the same root. Where the artist split one continuous floor between a
//!     room and the building's shell, the two meshes are stitched along the threshold line with
//!     coincident vertices, and THAT line is the doorway — the one the inn's front door actually
//!     is, and the only one of the three seeds that can see it (there is no portal there and no
//!     EXT-class batch in the entry group; see MEASURED below). It is the exact locus of the seam
//!     rather than an approximation of it: the hard line the owner photographed IS the polygon edge
//!     where the sky law hands over to the room law.
//!
//! MEASURED, because the two seeds are not interchangeable (`benilla-extract wmolights` + a
//! batch-class dump over the same roots, 2026-09-11):
//!  * **Goldshire inn** (Lion's Pride) — 12 groups, 10 portals, and exactly ONE of them is
//!    interior<->exterior (p0, `g0 <-> g11 room04`). Its **front door is not a portal at all**: the
//!    entry group `g3` is 100 % INT-class batches with two portals, both to interior neighbours,
//!    and the shell `g4 upstairs` that holds the threshold planks authors NO portals whatsoever.
//!    The aperture seed reaches its upstairs (`g2/g6/g7/g8/g9` each carry one EXT-class batch —
//!    the windows, diag 5.4-14.9 yd), and the BOUNDARY seed is what reaches the front door: `g3
//!    entry` shares its threshold vertices with the shell `g4`, and that stitch line is the door.
//!  * **Northshire abbey** — 14 groups, 14 portals, two of them interior<->exterior (p10
//!    `g3 Main Hall <-> g5 mainlobby2`, 40.3 yd^2; p13 `g10 Stairs2 <-> g5`, 14.3 yd^2), i.e. the
//!    portal seed's own best case.
//!
//! EXCLUSIONS. A daylight fixture is NOT a fire: no [`super::FlameFlicker`] (daylight does not
//! wobble), no [`super::SyntheticFireLight`] (so `fireLightGain` cannot dim or kill it), and
//! `benilla_app::torch_shadow` skips it as a caster candidate (`Without<DaylightFixture>`) — it is
//! a 20 yd-wide area source standing in a hole in a wall, so a point-cube shadow of it would be
//! wrong in kind, and it would outrank real fixtures for the twelve cube slots.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use benilla_formats::{PortalGraph, WmoGroupInfo};
use bevy::prelude::*;

use super::{
    DynamicInteriors, FireLightGain, LightLane, LightLitRooms, LightReach, SyntheticFireLight,
    WowLighting,
};

/// A `PointLight` that IS the daylight standing in one exterior-facing opening. Spawned with the
/// placement (`terrain_stream::spawn::fx`), re-aimed every frame by [`update_daylight_fixtures`],
/// and stripped of its `PointLight` outright after dark (see that system for why not an epsilon).
#[derive(Component, Clone, Copy, Debug)]
pub struct DaylightFixture {
    /// The `WmoPortalInstance` this opening belongs to — the placement, for the claim table's head
    /// word and for the dump.
    pub instance: Entity,
    /// The INTERIOR-class group the fixture stands in (it is nudged into it, see [`NUDGE_YD`]).
    pub group: u16,
    /// The MOPT portal index for a [`DaylightHow::Portal`] seed; `None` for an aperture seed, which
    /// is a batch and has no portal id.
    pub portal: Option<u16>,
    /// Which rule found this opening — the dump's readout, and the reason the two are separable.
    pub how: DaylightHow,
    /// The AUTHORED reach (yd), read exactly like a MOLT fixture's `attenuation_end`: the packer
    /// multiplies it by the live `interiorAttenScale` to get the effective radius `R`.
    pub reach: f32,
    /// The CALIBRATION distance (yd) — fixture to the floor fragment directly under the opening.
    pub cal_d: f32,
    /// `N.L` at that fragment for a floor normal (the fixture stands `hz` above it).
    pub cal_ndl: f32,
}

/// Which of the two rules found an opening — see the module doc's MEASURED note for why both are
/// needed and what each one reaches.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DaylightHow {
    /// An interior<->exterior MOPT portal (the client's own `FixColorVertexAlpha` relation).
    Portal,
    /// An EXT-class MOBA batch of an INTERIOR group — a window or an open side.
    Aperture,
    /// MONKEY (daylight fixtures: boundary): a run of vertices an interior group SHARES with an
    /// exterior-class group of the same root — the stitched threshold of a doorway the artist cut
    /// between two groups without authoring a portal.
    Boundary,
    /// MONKEY (portal bleed): an INTERIOR<->INTERIOR portal -- a doorway between two ROOMS, whose
    /// fixture is not the sun at all but the LIT room's own light carried across the threshold into
    /// the dark one. See the PORTAL BLEED section near the bottom of this file.
    Bleed,
}

impl DaylightHow {
    /// The dump's one-word tag.
    pub fn tag(self) -> &'static str {
        match self {
            DaylightHow::Portal => "portal",
            DaylightHow::Aperture => "aperture",
            DaylightHow::Boundary => "boundary",
            DaylightHow::Bleed => "bleed",
        }
    }
}

/// One opening, resolved in WMO MODEL space (WoW axes) — everything the spawner needs and nothing
/// about the world, so the whole selection rule is testable without a placement.
#[derive(Clone, Copy, Debug)]
pub struct DaylightSeed {
    pub group: u16,
    pub portal: Option<u16>,
    pub how: DaylightHow,
    /// Already nudged [`NUDGE_YD`] INTO `group` — the spawn position.
    pub pos: [f32; 3],
    /// The opening's bounding-box diagonal (yd) — the reach formula's input.
    pub diag: f32,
    /// The opening's area (yd^2) — the BUDGET rank key (largest first).
    pub area: f32,
    /// The seed's height above the opening's bottom edge (yd) — the calibration geometry.
    pub hz: f32,
}

/// How far INTO the interior group an opening's fixture stands (yd). Far enough that the light is
/// unambiguously on the room's side of the threshold plane (so the claim rule's containment test
/// puts it in the room rather than in whatever the doorway itself belongs to), and near enough that
/// the calibration below is measuring the doorway and not the middle of the room.
const NUDGE_YD: f32 = 0.5;
/// The widest opening the APERTURE seed will accept (bbox diagonal, yd). An EXT-class batch this
/// large is not a window: it is a whole open side of a group (the Goldshire inn's kitchen authors
/// one at 41.5 yd), and a single point light standing in the middle of it would be a sun in the
/// room rather than daylight at an edge. The portal seed has no such cap — a portal IS an opening
/// however big, and the reach clamp below bounds it anyway.
const APERTURE_MAX_DIAG: f32 = 24.0;
/// …and the narrowest. Below this an EXT-class batch is a trim strip or a sliver of frame, not an
/// aperture, and seeding one would spend a budget slot on a light nobody can see through.
const APERTURE_MIN_DIAG: f32 = 1.0;
/// An APERTURE seed this close to a PORTAL seed (yd) is the same opening described twice — the
/// window-frame batch beside the doorway it belongs to. The portal wins (it is the authored
/// relation); the aperture is dropped. Sized like `MODD_SYNTH_DEDUPE`'s argument: comfortably wider
/// than one doorway's own geometry, far narrower than the gap between two real openings.
const SEED_DEDUPE_YD: f32 = 4.0;
/// How high above an opening's SILL the fixture may stand (yd). An opening's centre is the obvious
/// place for it, and for a doorway (sill on the floor, ~1-2 yd to the middle) that is exactly right
/// — but a tall arch or a two-storey window puts its centre 4-5 yd up, and then two things go wrong
/// at once: the pool lands on the wall rather than on the floor where the seam is, and the
/// calibration cannot be met at all (the inverse-square core has already spent most of its value by
/// the time the light reaches the ground, so the solve saturates at `I = 1` and the room still
/// comes out short — MEASURED at 0.68 against a 0.83 target on the inn's `g6` and the abbey's `g9`).
/// Two yards is chest height on a threshold: low enough to light the floor, high enough that the
/// down-angle is still steep.
const SEED_MAX_HZ: f32 = 2.0;
/// MONKEY (daylight fixtures: boundary): how close two vertices of DIFFERENT groups have to be (yd)
/// to count as the same stitched point. A WMO's groups are cut out of one authored mesh, so a
/// shared edge is duplicated with bit-identical `MOVT` floats; 1/20 yd is two orders of magnitude
/// above float noise and still far below any gap an artist would leave between separate surfaces.
const SEAM_EPS: f32 = 0.05;
/// …and how far apart two stitched points may be (yd) and still belong to the SAME opening. A
/// threshold's vertex run is dense (a plank edge every few inches); a yard is comfortably wider
/// than that and comfortably narrower than the gap between a building's separate doors.
const SEAM_CLUSTER_YD: f32 = 1.0;
/// …overridable once per process by `WOW_DAYLIGHT_SEAM_EPS`, because the right value is a fact
/// about the CONTENT and the corpus disagrees with itself. The Goldshire inn's shell stitches its
/// upstairs rooms at the default and its ground floor not at all (`g4 <-> g5`: 0 coincident cells at
/// 0.05, 59 at 0.25), so a building whose seam the default misses can be re-measured live rather
/// than by another build. Read once; a bad value clamps back into `0.01..1.0`.
fn seam_eps() -> f32 {
    static EPS: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *EPS.get_or_init(|| {
        std::env::var("WOW_DAYLIGHT_SEAM_EPS")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .map_or(SEAM_EPS, |v| v.clamp(0.01, 1.0))
    })
}
/// The slack (yd) the exterior group's authored MOGI box is grown by before the pair is considered
/// at all. The boxes of two stitched groups touch rather than overlap, and an authored box is not
/// always tight around its own faces (NSabbey g3's bottom floats 1.5 yd above its floor), so a
/// strict overlap test would drop real pairs.
const SEAM_BOX_SLACK: f32 = 0.5;
/// MONKEY (daylight fixtures: boundary): a cluster taller than this (yd) is not a doorway sill. It
/// is the whole vertical seam where a room's wall is stitched to the shell's — the population that
/// would otherwise put a light against every party wall in the building. A door frame's own floor
/// line is centimetres tall and even a stair mouth stays well under three yards.
const SEAM_MAX_Z_EXTENT: f32 = 3.0;
/// …and its horizontal run must be at least this wide (yd), or it is a corner stitch rather than an
/// opening: three or four coincident vertices where two walls and a floor meet.
const SEAM_MIN_XY_EXTENT: f32 = 1.0;
/// MONKEY (daylight fixtures: boundary): the vertex budget above which the stitch pass is SKIPPED
/// (batch vertices summed over the whole root).
///
/// The pass is `27 hash probes per unique interior vertex per touching exterior neighbour`, and a
/// building pays nothing for it -- MEASURED at 6.6 ms for the Goldshire inn (20 470 batch verts)
/// and 7.4 ms for Northshire abbey (25 206), both including the other two seeds. A CITY does:
/// `Stormwind.wmo` is 306 groups / 3 088 batches / **761 733** batch verts and the whole selection
/// costs **162 ms**, synchronously, on the frame its placement spawns.
///
/// What makes the guard free rather than a trade is that the city gets nothing from the pass
/// anyway: Stormwind's eight surviving seeds are four portals and two apertures and two more
/// portals -- not one boundary cluster ranks into the budget, because a district's authored portals
/// are 338 yd^2 and a stitched threshold is tens. So the guard costs the city nothing it was
/// keeping, and buys back the hitch.
const SEAM_VERT_BUDGET: usize = 300_000;
/// Openings per placement. The rank is by AREA, so what survives a cap is the building's real
/// doorways and its largest windows. Eight is a whole facade's worth and still far under the
/// 256-slot packed table, which a city's placements share.
pub const MAX_DAYLIGHT_PER_PLACEMENT: usize = 8;

/// The EFFECTIVE reach (yd) of an opening `diag` yards across — the radius `R` the shader actually
/// windows the pool with: `clamp(1.5*diag + 4, 6, 20)`. The linear term says a wide doorway throws
/// light further in than an arrow slit; the `+4` floor keeps even a small window's pool bigger than
/// its own frame; the clamp keeps a 250 yd^2 cathedral opening from claiming a reach no interior
/// could absorb.
pub fn daylight_reach(diag: f32) -> f32 {
    (1.5 * diag + 4.0).clamp(6.0, 20.0)
}

/// …and what a daylight fixture actually CARRIES, which is one step removed: the packer windows
/// every interior fixture at `authored end x interiorAttenScale`, so an opening whose effective
/// reach is to be [`daylight_reach`] must author `daylight_reach / atten_scale`.
///
/// The division is by the cvar's DEFAULT, not its live value, for the same reason
/// [`benilla_formats::room_claim::CLAIM_ATTEN_SCALE`] exists: the number is baked once at spawn and
/// the cvar is live, so dividing by the live one would make the knob a no-op on exactly these
/// fixtures. At the default the effective reach is the formula's own answer; move the cvar and
/// daylight widens or tightens with every other fixture in the building, which is the behaviour a
/// global knob should have.
///
/// It lands where the corpus does, which is the corroboration: the Goldshire inn's own MOLT
/// fixtures author 6.97-9.53 yd, and a 5.9 yd doorway of its authors 8.1 here.
pub fn daylight_authored_reach(diag: f32) -> f32 {
    daylight_reach(diag) / benilla_formats::room_claim::CLAIM_ATTEN_SCALE
}

// MONKEY (daylight fixtures): the interior pool's PROFILE, MIRRORED from the block
// `static_gx.wgsl` and `wow_model.wgsl` carry ("MONKEY (soft falloff)") — the same mirror
// `benilla_app::torch_shadow` keeps for its caster ranking, and for the same reason: this module
// has to predict the shader's own answer at the door, and predicting it with a different curve
// would calibrate against a pool that is not the one being drawn. **Keep the four in sync.**
const INTERIOR_WRAP: f32 = 0.5;
const INTERIOR_CORE_YD: f32 = 1.75;
const INTERIOR_CORE_GAIN: f32 = 1.5;
const INTERIOR_DIRECT_POW: f32 = 10.0;
const INTERIOR_FILL_SPAN: f32 = 1.5;
const INTERIOR_FILL_POW: f32 = 1.0;

/// `(clamp01(1 - (d/r)^p))^2` — `static_gx.wgsl`'s `interior_window`.
fn interior_window(d: f32, r: f32, p: f32) -> f32 {
    let w = (1.0 - (d / r.max(1e-4)).clamp(0.0, 1.0).powf(p)).clamp(0.0, 1.0);
    w * w
}

/// The shader's DIRECT term at `d` yards on a surface whose normal sees the fixture at `ndl`, for a
/// fixture of effective radius `r_eff` and committed colour 1 — i.e. the factor the packed colour
/// is multiplied by. `atten * nl * window`, the three lines of `interior_room_light`.
fn direct_profile(d: f32, ndl: f32, r_eff: f32) -> f32 {
    let atten = INTERIOR_CORE_GAIN / (1.0 + (d / INTERIOR_CORE_YD).powi(2));
    let nl = ((ndl + INTERIOR_WRAP) / (1.0 + INTERIOR_WRAP)).max(0.0);
    atten * nl * interior_window(d, r_eff, INTERIOR_DIRECT_POW)
}

/// MONKEY (night fade): a three-line mirror of `global_light`'s private `sun_shadow_strength` — the
/// same one `blob_shadow` keeps, and for the same reason (the function is three lines and the
/// alternative is widening a private packer detail into the crate API). 0 at or below the horizon,
/// 1 by ~12 degrees of elevation, smoothstepped.
fn sun_shadow_strength(sun_height: f32) -> f32 {
    let t = (sun_height / 0.208).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Rec.709 luminance — the ONE scalar the calibration matches on. Matching per channel would be
/// wrong as well as impossible: the room law's hue comes from the fixture's normalised colour and
/// the reference's from `ambient + diffuse*N.L`, and the two agree in luminance at the door by
/// construction but not channel for channel (the ambient half is bluer than the diffuse half).
fn luminance(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// **THE CALIBRATION.** The packed intensity `I` in `[0, 1]` that makes the room law's result at the
/// door equal the reference sunlit result there.
///
/// The room law is `illum = 1 - exp(-(direct + fill + interiorAmbient) * interiorExposure)` and the
/// reference is `lit_int_base = clamp(ambient + diffuse*N.L)`, so with `T` the reference's luminance
/// the required illumination is `X = -ln(1 - T) / exposure`. Both the direct and the FILL term ride
/// the same committed colour (`c_norm`, and `c_fill` is a half-desaturated mix of it, which
/// preserves luminance), so both are linear in `I`:
///
/// ```text
/// X = hue_lum * I * (A + interiorFill * W_fill) + interiorAmbient
/// ```
///
/// `I` is clamped to 1 because the shader NORMALISES the committed colour
/// (`c_norm = c / max(1, max channel)`): past a peak of 1 the extra intensity is divided straight
/// back out, so an uncalibratable opening saturates rather than silently reading as if it carried
/// more light than it does. `I = 0` when the room already exceeds the target from its ambient floor
/// alone.
pub fn daylight_intensity(
    target_lum: f32,
    hue_lum: f32,
    a_direct: f32,
    w_fill: f32,
    knobs: &DynamicInteriors,
) -> f32 {
    // `1 - exp(-x)` never reaches 1, so a reference luminance of exactly 1 has no finite solution;
    // 0.999 is a thousandth of a byte-step short and keeps the log finite.
    let t = target_lum.clamp(0.0, 0.999);
    let x = -(1.0 - t).ln() / knobs.exposure.max(1e-3);
    let denom = hue_lum.max(1e-3) * (a_direct + knobs.fill.max(0.0) * w_fill);
    if denom <= 1e-6 {
        return 0.0;
    }
    ((x - knobs.ambient) / denom).clamp(0.0, 1.0)
}

/// The day's own colour and brightness AT A FLOOR, as the reference law renders it — the target the
/// calibration above matches, and the hue the fixture is given.
///
/// Returns `(hue, target_lum, sun_w)`: `hue` is the reference colour renormalised to peak 1 (what
/// the packed colour carries, so `c_norm` in the shader comes back as `hue * I`), `target_lum` its
/// luminance, and `sun_w` the day envelope every daylight fixture is scaled by — the SAME
/// `sun_shadow_strength(celestial_dir.y)` the shader's `day_w` is scaled by, so the two halves of
/// the threshold fade in and out of the sky law together instead of crossing.
///
/// The normal is UP because the seam the feature exists to close is a FLOOR seam — threshold plank
/// to floorboard. A wall at the same door reads a different `N.L` under both laws, and no single
/// point source can match a directional one on every normal at once; the floor is the surface the
/// eye compares across the threshold.
pub fn daylight_target(light: &WowLighting) -> ([f32; 3], f32, f32) {
    // `static_gx.wgsl`: `L = -normalize(light_sun.xyz)`, `ndotl = max(dot(N, L), 0)`.
    let l = -light.sun_dir.normalize_or_zero();
    let ndotl = l.y.max(0.0);
    let mut target = [0.0f32; 3];
    let mut peak = 0.0f32;
    for i in 0..3 {
        target[i] = (light.ambient[i] + light.diffuse[i] * ndotl).clamp(0.0, 1.0);
        peak = peak.max(target[i]);
    }
    let hue = if peak > 1e-4 {
        [target[0] / peak, target[1] / peak, target[2] / peak]
    } else {
        [0.0; 3]
    };
    (
        hue,
        luminance(target),
        sun_shadow_strength(light.celestial_dir().y),
    )
}

/// The AABB of a point cloud in WMO model space — the one measurement both seeds take of their
/// opening (centre, diagonal, and the bottom edge the calibration height is measured from).
fn bounds(points: impl Iterator<Item = [f32; 3]>) -> Option<([f32; 3], [f32; 3])> {
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    let mut any = false;
    for p in points {
        any = true;
        for a in 0..3 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    any.then_some((lo, hi))
}

/// The two largest extents of a box multiplied — the opening's AREA as the budget ranks it. An
/// aperture is a slab (a window is thin in its wall normal), so dropping the smallest extent is
/// exactly "how big is the hole", and it is the same number a portal's own polygon area approaches.
fn opening_area(lo: [f32; 3], hi: [f32; 3]) -> f32 {
    let mut e = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    e.sort_by(f32::total_cmp);
    (e[1] * e[2]).max(0.0)
}

fn diagonal(lo: [f32; 3], hi: [f32; 3]) -> f32 {
    ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt()
}

/// Where in an opening the fixture stands, and how far that is above the opening's SILL —
/// the box centre in plan, but no more than [`SEED_MAX_HZ`] up. See that constant.
fn seed_point(lo: [f32; 3], hi: [f32; 3]) -> ([f32; 3], f32) {
    let c = center(lo, hi);
    let z = c[2].min(lo[2] + SEED_MAX_HZ);
    ([c[0], c[1], z], (z - lo[2]).max(0.0))
}

/// MONKEY (daylight fixtures: boundary): where a STITCHED cluster's fixture stands — the cluster's
/// centre in plan, but a full [`SEED_MAX_HZ`] ABOVE its lowest point rather than at its own height.
///
/// The difference from [`seed_point`] is the whole geometry of the case. A portal or a window IS the
/// opening, so its own centre is where the light comes through. A boundary cluster is the THRESHOLD
/// — a line of vertices lying on the floor — and the opening is the doorway standing on top of it.
/// Seeding at the cluster's own height would put a point light in the floorboards: calibrated
/// correctly at the threshold (the solve does not care) but with a hot ring on the planks and a much
/// steeper fall inward than a doorway has.
fn seed_point_above(lo: [f32; 3], hi: [f32; 3]) -> ([f32; 3], f32) {
    let c = center(lo, hi);
    ([c[0], c[1], lo[2] + SEED_MAX_HZ], SEED_MAX_HZ)
}

fn center(lo: [f32; 3], hi: [f32; 3]) -> [f32; 3] {
    [
        0.5 * (lo[0] + hi[0]),
        0.5 * (lo[1] + hi[1]),
        0.5 * (lo[2] + hi[2]),
    ]
}

/// Nudge `from` [`NUDGE_YD`] toward the inside of `g`, preferring the direction `plane_n` when the
/// opening has an authored plane (a portal): stepping along the doorway's own normal crosses the
/// threshold squarely, where stepping toward the room's box centre can slide ALONG a wall in an
/// L-shaped room and leave the fixture in the frame. The sign is taken from the room centre either
/// way — that is the only thing that says which side is "in".
fn nudge_inward(from: [f32; 3], g: &WmoGroupInfo, plane_n: Option<[f32; 3]>) -> [f32; 3] {
    let gc = center(g.bbox_min, g.bbox_max);
    let to_room = Vec3::new(gc[0] - from[0], gc[1] - from[1], gc[2] - from[2]);
    let dir = match plane_n.map(|n| Vec3::from(n).normalize_or_zero()) {
        Some(n) if n.length_squared() > 0.5 && to_room.dot(n).abs() > 1e-3 => {
            n * to_room.dot(n).signum()
        }
        _ => to_room.normalize_or_zero(),
    };
    let p = Vec3::new(from[0], from[1], from[2]) + dir * NUDGE_YD;
    [p.x, p.y, p.z]
}

/// **THE SELECTION RULE** — every exterior-facing opening of one WMO root, in WMO model space,
/// ranked by area and capped at [`MAX_DAYLIGHT_PER_PLACEMENT`]. See the module doc for the two
/// seeds and the corpus measurement behind having both.
///
/// `batches` is `(absolute group index, EXT-class?, the batch's model-space positions)` for every
/// render batch of the root — the aperture seed's input, passed as an iterator so the caller can
/// hand it the loaded asset's parallel `submeshes` / `submesh_group` arrays without building a
/// second copy, and a test can hand it three points.
pub fn daylight_seeds<'a, I>(
    groups: &[WmoGroupInfo],
    portals: PortalGraph<'_>,
    batches: I,
) -> Vec<DaylightSeed>
where
    I: IntoIterator<Item = (u16, bool, &'a [[f32; 3]])>,
{
    let mut out = daylight_seeds_ranked(groups, portals, batches);
    out.truncate(MAX_DAYLIGHT_PER_PLACEMENT);
    out
}

/// MONKEY (portal bleed): the same rule, RANKED but not capped — the budget is now shared with the
/// bleed seeds ([`placement_openings`]), and a cap applied here would spend the eight slots on
/// daylight before a doorway between two rooms was ever compared against them.
pub fn daylight_seeds_ranked<'a, I>(
    groups: &[WmoGroupInfo],
    portals: PortalGraph<'_>,
    batches: I,
) -> Vec<DaylightSeed>
where
    I: IntoIterator<Item = (u16, bool, &'a [[f32; 3]])>,
{
    let (want_portals, want_apertures, want_boundaries) = daylight_modes();
    let mut out: Vec<DaylightSeed> = Vec::new();
    let interior_of = |g: u16| groups.get(usize::from(g)).map(|g| g.interior);
    // Collected because TWO seeds read it: the aperture rule walks it for its EXT-class batches,
    // and the boundary rule needs every batch of every group, grouped. It is one small Vec of
    // `(u16, bool, &[_])` per placement per spawn wave -- borrowed slices, no vertex copy.
    let batches: Vec<(u16, bool, &[[f32; 3]])> = batches.into_iter().collect();

    // --- 1. PORTAL seeds. Both endpoints of a MOPR edge count as "a side": a portal is authored
    // once per group that owns it, and a portal only ONE group names is a portal to the outside,
    // whose missing side is exterior by definition (there is no group out there).
    let mut sides: Vec<(u16, Vec<u16>)> = Vec::new();
    if want_portals {
        let note = |portal: u16, group: u16, sides: &mut Vec<(u16, Vec<u16>)>| {
            match sides.iter_mut().find(|(p, _)| *p == portal) {
                Some((_, gs)) => {
                    if !gs.contains(&group) {
                        gs.push(group);
                    }
                }
                None => sides.push((portal, vec![group])),
            }
        };
        for (gi, (start, count)) in portals.slices.iter().enumerate() {
            let (start, count) = (usize::from(*start), usize::from(*count));
            for r in portals.refs.get(start..start + count).unwrap_or(&[]) {
                note(r.portal, gi as u16, &mut sides);
                note(r.portal, r.group, &mut sides);
            }
        }
    }
    for (portal, gs) in &sides {
        let mut ins = gs.iter().copied().filter(|g| interior_of(*g) == Some(true));
        let (Some(inside), None) = (ins.next(), ins.next()) else {
            continue; // zero interior sides (an exterior-to-exterior seam) or two (a room to a room)
        };
        // One interior side is not yet a doorway: the other side must be exterior-CLASS, or absent
        // entirely (a portal straight to the outside, which only one group names).
        if gs.len() > 1 && !gs.iter().any(|g| interior_of(*g) == Some(false)) {
            continue;
        }
        let Some(g) = groups.get(usize::from(inside)) else {
            continue;
        };
        let info = portals.infos.get(usize::from(*portal));
        let verts = info.and_then(|i| {
            let s = usize::from(i.start_vertex);
            portals.vertices.get(s..s + usize::from(i.count))
        });
        let Some((lo, hi)) = verts.and_then(|v| bounds(v.iter().copied())) else {
            continue;
        };
        let (c, hz) = seed_point(lo, hi);
        let n = info.map(|i| [i.plane[0], i.plane[1], i.plane[2]]);
        out.push(DaylightSeed {
            group: inside,
            portal: Some(*portal),
            how: DaylightHow::Portal,
            pos: nudge_inward(c, g, n),
            diag: diagonal(lo, hi),
            // The real polygon area where the record gives one — a doorway is rarely its own box.
            area: benilla_formats::room_claim::portal_area(&portals, *portal)
                .unwrap_or_else(|| opening_area(lo, hi)),
            hz,
        });
    }
    let portal_seeds = out.len();

    // --- 2. APERTURE seeds: an EXT-class batch of an INTERIOR group (see the module doc).
    if want_apertures {
        for &(gi, ext_class, positions) in &batches {
            if !ext_class || interior_of(gi) != Some(true) {
                continue;
            }
            let Some(g) = groups.get(usize::from(gi)) else {
                continue;
            };
            let Some((lo, hi)) = bounds(positions.iter().copied()) else {
                continue;
            };
            let diag = diagonal(lo, hi);
            if !(APERTURE_MIN_DIAG..=APERTURE_MAX_DIAG).contains(&diag) {
                continue;
            }
            let (c, hz) = seed_point(lo, hi);
            let pos = nudge_inward(c, g, None);
            // The authored portal wins a shared opening — see [`SEED_DEDUPE_YD`].
            let dup = out[..portal_seeds].iter().any(|s| {
                (s.pos[0] - pos[0]).powi(2)
                    + (s.pos[1] - pos[1]).powi(2)
                    + (s.pos[2] - pos[2]).powi(2)
                    < SEED_DEDUPE_YD * SEED_DEDUPE_YD
            });
            if dup {
                continue;
            }
            out.push(DaylightSeed {
                group: gi,
                portal: None,
                how: DaylightHow::Aperture,
                pos,
                diag,
                area: opening_area(lo, hi),
                hz,
            });
        }
    }

    // --- 3. BOUNDARY seeds: the stitched interior<->exterior group seam (see the module doc).
    // Skipped outright on a city-scale root -- see [`SEAM_VERT_BUDGET`] for the measurement.
    let root_verts: usize = batches.iter().map(|(_, _, p)| p.len()).sum();
    if want_boundaries && root_verts <= SEAM_VERT_BUDGET {
        let authored = out.len();
        for (group, lo, hi, inward) in boundary_clusters(groups, &batches) {
            let (c, hz) = seed_point_above(lo, hi);
            let pos = [
                c[0] + inward[0] * NUDGE_YD,
                c[1] + inward[1] * NUDGE_YD,
                c[2],
            ];
            // The authored seeds win a shared opening: a real doorway often has BOTH a portal (or
            // an EXT batch) and a stitch line, and two lights in one door would double the pool.
            let dup = out[..authored].iter().any(|s| {
                (s.pos[0] - pos[0]).powi(2)
                    + (s.pos[1] - pos[1]).powi(2)
                    + (s.pos[2] - pos[2]).powi(2)
                    < SEED_DEDUPE_YD * SEED_DEDUPE_YD
            });
            if dup {
                continue;
            }
            let diag = diagonal(lo, hi);
            out.push(DaylightSeed {
                group,
                portal: None,
                how: DaylightHow::Boundary,
                pos,
                diag,
                // A threshold is a LINE, not a slab: `opening_area`'s two-largest-extents measure
                // would call a 4 yd doorway sill `4 x 0.3 = 1.2` yd^2 and the budget would drop it
                // under every window in the building. The opening the line REPRESENTS is about as
                // tall as it is wide, so its run squared is the honest comparable.
                area: diag * diag,
                hz,
            });
        }
    }

    // Largest opening first: what a placement keeps once the shared budget cuts is its real
    // doorways and its widest windows, and what it drops is the trim.
    out.sort_by(|a, b| b.area.total_cmp(&a.area));
    out
}

/// MONKEY (daylight fixtures: boundary): every stitched interior<->exterior opening of one root, as
/// `(interior group, cluster lo, cluster hi, inward unit direction)` in WMO model space.
///
/// **The rule.** A WMO's groups are cut out of one authored mesh, so where a room's floor runs into
/// the building shell's the two group files carry the SAME vertices -- the threshold is literally
/// duplicated geometry. For each exterior-class group `E` we hash its vertices onto a [`SEAM_EPS`]
/// grid; then for each interior-class group `G` whose authored box reaches `E`'s (grown by
/// [`SEAM_BOX_SLACK`]) we keep the `G` vertices landing within `SEAM_EPS` of one of `E`'s. The
/// survivors are clustered, and each cluster is one opening.
///
/// **The loop order is E-outer on purpose.** One hash per exterior group, reused across every
/// interior group it touches, so peak memory is the LARGEST exterior group rather than the sum --
/// Stormwind carries ~8 district shells of six figures of vertices each, and a per-pair hash would
/// have rebuilt the big one eight times over.
///
/// **Clustering is by CELL, not by point.** Single-link over points is `O(k^2)` in the worst case
/// and a dense threshold is exactly that worst case (a vertex every few inches). Instead the hits
/// are dropped onto a [`SEAM_CLUSTER_YD`] grid and the occupied CELLS are union-found over their
/// 27-neighbourhood, which links any two points closer than the cell size and at most its diagonal.
/// The difference from exact single-link is a yard of slack at the joins, which the extent filters
/// absorb.
///
/// **The filters** are the difference between "the doorway" and "every wall this room shares with
/// the shell": a cluster must run at least [`SEAM_MIN_XY_EXTENT`] horizontally (or it is a corner
/// stitch -- three vertices where two walls meet a floor), must stand no more than
/// [`SEAM_MAX_Z_EXTENT`] tall (or it is the vertical seam of a whole party wall), and must fall in
/// the same 1..24 yd size band the aperture seed uses.
fn boundary_clusters(
    groups: &[WmoGroupInfo],
    batches: &[(u16, bool, &[[f32; 3]])],
) -> Vec<(u16, [f32; 3], [f32; 3], [f32; 3])> {
    // Per-group batch slices + each group's own vertex CENTROID, which is what "inward" points at.
    // The centroid rather than the authored box centre: a box centre can sit outside an L-shaped
    // room entirely, and a threshold's inward direction is the one thing this seed cannot get wrong
    // without putting the light inside the wall.
    let mut per_group: Vec<Vec<&[[f32; 3]]>> = vec![Vec::new(); groups.len()];
    let mut sum: Vec<([f64; 3], u64)> = vec![([0.0; 3], 0); groups.len()];
    for &(gi, _, positions) in batches {
        let Some(slot) = per_group.get_mut(usize::from(gi)) else {
            continue;
        };
        slot.push(positions);
        let acc = &mut sum[usize::from(gi)];
        for p in positions {
            for a in 0..3 {
                acc.0[a] += f64::from(p[a]);
            }
            acc.1 += 1;
        }
    }
    let eps = seam_eps();
    let key = |p: [f32; 3]| {
        [
            (p[0] / eps).round() as i32,
            (p[1] / eps).round() as i32,
            (p[2] / eps).round() as i32,
        ]
    };
    // Each INTERIOR group's unique grid keys, built ONCE. A group repeats every vertex across the
    // batches that use it (the Goldshire inn authors 92 batches over 12 groups), and the walk below
    // is `27 hash probes per key per exterior neighbour` -- so deduplicating here, rather than
    // inside that walk, is the difference between paying for vertices and paying for vertex USES.
    let mut g_keys: Vec<Vec<[i32; 3]>> = vec![Vec::new(); groups.len()];
    for (gi, g) in groups.iter().enumerate() {
        if !g.interior || per_group[gi].is_empty() {
            continue;
        }
        let mut seen: HashSet<[i32; 3]> = HashSet::new();
        for slice in &per_group[gi] {
            for p in *slice {
                seen.insert(key(*p));
            }
        }
        g_keys[gi] = seen.into_iter().collect();
    }
    // Stitched hits per interior group, deduped by grid key (a group repeats each vertex across
    // every batch that uses it, and a threshold vertex is shared by several).
    let mut hits: HashMap<u16, HashSet<[i32; 3]>> = HashMap::new();
    for (ei, e) in groups.iter().enumerate() {
        if e.interior || per_group[ei].is_empty() {
            continue;
        }
        let elo = [
            e.bbox_min[0] - SEAM_BOX_SLACK,
            e.bbox_min[1] - SEAM_BOX_SLACK,
            e.bbox_min[2] - SEAM_BOX_SLACK,
        ];
        let ehi = [
            e.bbox_max[0] + SEAM_BOX_SLACK,
            e.bbox_max[1] + SEAM_BOX_SLACK,
            e.bbox_max[2] + SEAM_BOX_SLACK,
        ];
        let mut cells: HashSet<[i32; 3]> = HashSet::new();
        for slice in &per_group[ei] {
            for p in *slice {
                cells.insert(key(*p));
            }
        }
        for (gi, g) in groups.iter().enumerate() {
            if !g.interior || per_group[gi].is_empty() {
                continue;
            }
            if (0..3).any(|a| g.bbox_min[a] > ehi[a] || g.bbox_max[a] < elo[a]) {
                continue; // the two groups are nowhere near each other
            }
            let seen = hits.entry(gi as u16).or_default();
            for k in &g_keys[gi] {
                // Cheap reject first: a vertex outside the exterior group's grown box cannot be
                // within `eps` of one of its vertices, and that is most of them.
                let p = [k[0] as f32 * eps, k[1] as f32 * eps, k[2] as f32 * eps];
                if (0..3).any(|a| p[a] < elo[a] || p[a] > ehi[a]) {
                    continue;
                }
                if seen.contains(k) {
                    continue; // already stitched by another exterior neighbour
                }
                let hit = (-1..=1).any(|dx| {
                    (-1..=1).any(|dy| {
                        (-1..=1).any(|dz| cells.contains(&[k[0] + dx, k[1] + dy, k[2] + dz]))
                    })
                });
                if hit {
                    seen.insert(*k);
                }
            }
        }
    }

    let mut out = Vec::new();
    for (gi, keys) in hits {
        let (acc, n) = sum[usize::from(gi)];
        if keys.is_empty() || n == 0 {
            continue;
        }
        let centroid = [
            (acc[0] / n as f64) as f32,
            (acc[1] / n as f64) as f32,
            (acc[2] / n as f64) as f32,
        ];
        let points: Vec<[f32; 3]> = keys
            .iter()
            .map(|k| [k[0] as f32 * eps, k[1] as f32 * eps, k[2] as f32 * eps])
            .collect();
        for (lo, hi) in cluster_boxes(&points) {
            let d = diagonal(lo, hi);
            if !(APERTURE_MIN_DIAG..=APERTURE_MAX_DIAG).contains(&d) {
                continue;
            }
            if hi[2] - lo[2] > SEAM_MAX_Z_EXTENT {
                continue;
            }
            let xy = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2)).sqrt();
            if xy < SEAM_MIN_XY_EXTENT {
                continue;
            }
            let c = center(lo, hi);
            // Horizontal only: the room's centroid is usually a storey up or down from a threshold,
            // and a light nudged along that slope would leave the floor it is meant to be lighting.
            let mut inward = Vec3::new(centroid[0] - c[0], centroid[1] - c[1], 0.0);
            if inward.length_squared() < 1e-6 {
                inward = Vec3::X; // degenerate (the centroid is over the door) -- any push will do
            }
            let inward = inward.normalize();
            out.push((gi, lo, hi, [inward.x, inward.y, 0.0]));
        }
    }
    // Deterministic across runs: a `HashMap` walk is not, and the budget TRUNCATES, so an unstable
    // order would give the same building different doorways on different launches.
    out.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1[0].total_cmp(&b.1[0]))
            .then(a.1[1].total_cmp(&b.1[1]))
            .then(a.1[2].total_cmp(&b.1[2]))
    });
    out
}

/// Single-link clustering of `points` at [`SEAM_CLUSTER_YD`], by grid cell (see
/// [`boundary_clusters`] for why by cell): one `(lo, hi)` box per cluster.
fn cluster_boxes(points: &[[f32; 3]]) -> Vec<([f32; 3], [f32; 3])> {
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    let cell = |p: [f32; 3]| {
        [
            (p[0] / SEAM_CLUSTER_YD).floor() as i32,
            (p[1] / SEAM_CLUSTER_YD).floor() as i32,
            (p[2] / SEAM_CLUSTER_YD).floor() as i32,
        ]
    };
    let mut index: HashMap<[i32; 3], usize> = HashMap::new();
    let mut cells: Vec<[i32; 3]> = Vec::new();
    let mut of_point: Vec<usize> = Vec::with_capacity(points.len());
    for p in points {
        let c = cell(*p);
        let i = *index.entry(c).or_insert_with(|| {
            cells.push(c);
            cells.len() - 1
        });
        of_point.push(i);
    }
    let mut parent: Vec<usize> = (0..cells.len()).collect();
    for (i, c) in cells.iter().enumerate() {
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(&j) = index.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) {
                        let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                        if a != b {
                            parent[a] = b;
                        }
                    }
                }
            }
        }
    }
    let mut boxes: HashMap<usize, ([f32; 3], [f32; 3])> = HashMap::new();
    for (p, ci) in points.iter().zip(of_point.iter()) {
        let root = find(&mut parent, *ci);
        let e = boxes.entry(root).or_insert((*p, *p));
        for a in 0..3 {
            e.0[a] = e.0[a].min(p[a]);
            e.1[a] = e.1[a].max(p[a]);
        }
    }
    let mut out: Vec<([f32; 3], [f32; 3])> = boxes.into_values().collect();
    // Same determinism argument as `boundary_clusters`.
    out.sort_by(|a, b| {
        a.0[0]
            .total_cmp(&b.0[0])
            .then(a.0[1].total_cmp(&b.0[1]))
            .then(a.0[2].total_cmp(&b.0[2]))
    });
    out
}

impl DaylightSeed {
    /// The component this seed becomes — the calibration geometry resolved once, at spawn.
    ///
    /// `cal_d` is the distance from the fixture to the floor fragment directly under the opening:
    /// [`NUDGE_YD`] horizontally (the fixture stands that far inside the threshold) and `hz`
    /// vertically (the opening's centre sits that far above its own sill). `cal_ndl` is that
    /// fragment's `N.L` for a floor normal, which is `hz / cal_d` — so a low sill (a doorway,
    /// `hz ~ 1.1`) reads nearly straight down and a high window reads more grazing, exactly as the
    /// geometry says.
    pub fn fixture(&self, instance: Entity) -> DaylightFixture {
        let hz = self.hz.max(0.25);
        let cal_d = (NUDGE_YD * NUDGE_YD + hz * hz).sqrt().max(1e-3);
        DaylightFixture {
            instance,
            group: self.group,
            portal: self.portal,
            how: self.how,
            reach: daylight_authored_reach(self.diag),
            cal_d,
            cal_ndl: (hz / cal_d).clamp(0.0, 1.0),
        }
    }
}

/// `WOW_DAYLIGHT` — the feature's own kill switch, read once. `0`/`off` disables it entirely;
/// `portals`, `apertures` and `boundaries` each keep one seed alone. A plain env var rather than a
/// cvar because the cvar table is another agent's file this round; the A/B it buys ("is THIS seam a
/// portal opening, a window, or a stitched threshold?") is worth more than the tidier home.
pub fn daylight_modes() -> (bool, bool, bool) {
    static MODES: std::sync::OnceLock<(bool, bool, bool)> = std::sync::OnceLock::new();
    *MODES.get_or_init(|| match std::env::var("WOW_DAYLIGHT").ok().as_deref() {
        Some("0" | "off" | "false") => (false, false, false),
        Some("portals") => (true, false, false),
        Some("apertures") => (false, true, false),
        Some("boundaries") => (false, false, true),
        _ => (true, true, true),
    })
}

/// Re-aim every daylight fixture at this frame's sun: the colour, the calibrated intensity, and the
/// day/night existence switch.
///
/// Scheduled in `PostUpdate` **before** `global_light::classify_light_lanes`, which is itself
/// chained before the packer `build_light_data` — so a fixture is packed on THIS frame's sun rather
/// than last frame's. It needs no transform-propagation ordering of its own (unlike the packer): a
/// daylight fixture never moves, and nothing here reads a `GlobalTransform`.
///
/// **After dark the `PointLight` is REMOVED, not dimmed.** The packer has no zero-intensity skip —
/// it packs any `PointLight` within 300 yd — so an epsilon fixture would still hold one of the 256
/// table slots and still be walked by every interior fragment in its fill radius, all night, to
/// contribute nothing. Removing the component takes it out of the packer's query outright and costs
/// two archetype moves per game day. Everything else about the entity (its claims, its lane, its
/// reach) is untouched, so dawn re-inserts and the claim set is exactly the one built at spawn.
pub fn update_daylight_fixtures(
    mut commands: Commands,
    light: Res<WowLighting>,
    knobs: Res<DynamicInteriors>,
    // MONKEY (portal bleed): `Without<BleedFixture>` -- a bleed fixture wears `DaylightFixture` as
    // its marker (for the packer's `interiorGain` skip and `torch_shadow`'s caster refusal) but is
    // aimed by `update_bleed_fixtures` instead, from the room next door rather than from the sun.
    // The exclusion is also what makes the two systems' `&mut PointLight` access disjoint.
    mut fixtures: Query<
        (
            Entity,
            &DaylightFixture,
            Option<&LightReach>,
            Option<&mut PointLight>,
        ),
        Without<BleedFixture>,
    >,
    time: Res<Time>,
    mut last_dump: Local<f64>,
) {
    let (hue, target_lum, sun_w) = daylight_target(&light);
    // The room lane being off is the feature being off: with `interiorLight 0` every interior
    // surface is back on its MOCV bake, nothing reads the interior half of the point table, and a
    // daylight fixture would be a slot spent on a lane nobody is listening to.
    let day = knobs.enabled && sun_w > 1e-3 && target_lum > 1e-3;
    let hue_lum = luminance(hue);
    let mut lit: Vec<(Entity, DaylightFixture, f32)> = Vec::new();
    for (e, fx, reach, pl) in &mut fixtures {
        let intensity = if day {
            // The EFFECTIVE radius the shader will window this fixture with — the same
            // `interior_reach(authored, interiorAttenScale)` the packer folds into the colour row's
            // `.w`, so the profile predicted here is the profile drawn.
            let r_eff = super::interior_reach(reach.map_or(fx.reach, |r| r.0), knobs.atten_scale);
            let a = direct_profile(fx.cal_d, fx.cal_ndl, r_eff);
            let w_fill = interior_window(fx.cal_d, INTERIOR_FILL_SPAN * r_eff, INTERIOR_FILL_POW);
            daylight_intensity(target_lum, hue_lum, a, w_fill, &knobs) * sun_w
        } else {
            0.0
        };
        match pl {
            Some(mut pl) if intensity > 1e-4 => {
                pl.color = Color::linear_rgb(hue[0], hue[1], hue[2]);
                pl.intensity = 4.0 * std::f32::consts::PI * intensity;
            }
            Some(_) => {
                commands.entity(e).remove::<PointLight>();
            }
            None if intensity > 1e-4 => {
                commands
                    .entity(e)
                    .insert(daylight_point_light(hue, intensity));
            }
            None => {}
        }
        if intensity > 1e-4 {
            lit.push((e, *fx, intensity));
        }
    }
    // `WOW_POINTS_DUMP`: the DAYLIGHT rows. They ride their own block rather than the packer's
    // because the packer is another agent's file this round — the numbers are the same ones,
    // printed one stage earlier, and a DAY-only block is itself the readout that the night switch
    // above is working.
    if let Some(mode) = std::env::var_os("WOW_POINTS_DUMP") {
        let every = if mode == *"frame" { 0.0 } else { 1.0 };
        let now = time.elapsed_secs_f64();
        if now - *last_dump >= every {
            *last_dump = now;
            if !lit.is_empty() {
                eprintln!(
                    "[daylight] {} DAYLIGHT fixture(s) lit — sun_w {sun_w:.3}, target lum {target_lum:.3}, hue [{:.3},{:.3},{:.3}]",
                    lit.len(),
                    hue[0],
                    hue[1],
                    hue[2],
                );
                for (e, fx, i) in lit.iter().take(8) {
                    eprintln!(
                        "  DAYLIGHT {e}  inst {} g{}  {}{}  reach {:5.2}  cal d {:4.2} N.L {:4.2}  I {i:.4}",
                        fx.instance,
                        fx.group,
                        fx.how.tag(),
                        fx.portal.map_or(String::new(), |p| format!(" p{p}")),
                        fx.reach,
                        fx.cal_d,
                        fx.cal_ndl,
                    );
                }
            }
        }
    }
}

/// The `PointLight` a daylight fixture carries. The same 4*PI convention every other authored
/// source uses (`terrain_stream::spawn::fx::point_light`): the packer recovers `intensity/(4*PI)`
/// and multiplies the linear colour by it, so the committed colour is exactly `hue * I` — which is
/// what the calibration above solved for, and what the shader's `c_norm` reads back unchanged while
/// `I <= 1`.
pub fn daylight_point_light(hue: [f32; 3], intensity: f32) -> PointLight {
    PointLight {
        color: Color::linear_rgb(hue[0], hue[1], hue[2]),
        intensity: 4.0 * std::f32::consts::PI * intensity.max(0.0),
        // The 48 yd CANDIDACY constant every source packs — not a reach (that rides `LightReach`).
        range: 48.0,
        shadows_enabled: false,
        ..default()
    }
}

/// The claim set of a daylight fixture — the SAME rule every MOLT fixture's claims are built by
/// ([`benilla_formats::room_claim`]), seeded with the interior group it stands in as its MOLR
/// referrer. Spelled here rather than at the spawn site so the reach the claim's portal hop is
/// measured against can never drift from the reach the packer windows the fixture with.
pub fn daylight_claims(
    groups: &[WmoGroupInfo],
    portals: PortalGraph<'_>,
    seed: &DaylightSeed,
) -> Vec<benilla_formats::room_claim::Claim> {
    benilla_formats::room_claim::room_claims(
        groups,
        portals,
        seed.pos,
        benilla_formats::room_claim::claim_reach(daylight_authored_reach(seed.diag)),
        &[seed.group],
    )
}

/// The one-room visibility claim a daylight fixture carries — it exists while its own room is
/// drawn, exactly like the MOLT fixture standing in that room (`LightRooms`, decision 0689).
pub fn daylight_rooms(instance: Entity, group: u16) -> super::LightRooms {
    super::LightRooms::new(crate::wmo_portal::WmoGroupVis {
        instance,
        groups: Arc::from([group].as_slice()),
    })
}

/// A daylight fixture's lane is SETTLED interior, stamped at spawn rather than rayed.
///
/// `classify_light_lanes`'s down-ray would usually agree — the fixture stands a half-yard inside an
/// interior group — but "usually" is the problem: it stands in a DOORWAY, the one place in a
/// building where the surface under a light can belong to the exterior shell (the threshold plank),
/// and a fixture flipped to the exterior lane would stop lighting the room it was built for and
/// start pooling on the street instead. The seed already knows the answer from the authored group
/// class, so it states it and the classifier skips the entity ([`super::LightLane::SETTLED`]).
pub fn daylight_lane() -> super::LightLane {
    super::LightLane {
        interior: true,
        generation: super::LightLane::SETTLED,
    }
}

// ===========================================================================================
// MONKEY (portal bleed): **the LIT room next door, glowing in the doorway.**
//
// THE BUG. Three seeds above stand the SUN in an opening, and every one of them needs the opening
// to face OUTSIDE (a portal to an exterior group, an EXT-class batch, a stitch to the shell). A
// room whose only doorways lead to other ROOMS gets nothing from them — and with `interiorDaylight`
// defaulted to 0 (the director's call: by day light comes in at doorways and windows, not through
// the walls) it gets nothing from the enclosed day floor either. What is left is `interiorAmbient`.
//
// MEASURED, the Lion's Pride Inn's front vestibule (`benilla-extract wmolights`, 2026-09-12). It is
// group `g0`, the WMO's only TRANS-batch group, and it hangs off the end of a CHAIN of doorways:
// porch `g11` --p0--> `g0` --p1--> `g1` --p2--> the hall `g5`, where the ten MOLT fixtures live.
// No fixture CONTAINS-claims g0 or g1; both rooms are lit, if at all, by portal hops. And the hops
// have run out by the time they reach p1: the nearest fixture claiming g1 (L2, 9.0 yd from p2)
// carries 2.6 yd of unspent reach past p2's own 3.2 yd of doorway slack, and p1's centre is 5.8 yd
// from p2's — so every claim weight at p1 is EXACTLY 0. The vestibule's budget there is the bare
// `interiorAmbient x interiorGain` = 0.0075, i.e. `1 - exp(-0.0075 x 2.5)` = **0.019 x tex**, while
// the `trans_night_floor` bake share renders the same planks at **0.324 x tex**: a flat grey band
// seventeen times its own neighbour, which is what the owner photographed at 20:50 and 23:59.
//
// THE FIX, and it is the owner's own principle stated as a light: a doorway between two rooms
// TRANSMITS. Stand an ordinary interior-lane point light IN such a portal, claiming both rooms, and
// give it every frame [`BLEED_K`] times the irradiance the BRIGHTER side has at the portal plane.
// Each room then reads a fraction of its neighbour AT the threshold (so nothing steps across the
// doorway) and falls off inward on the room lane's own profile (so it is a doorway's spill, not a
// second candle).
//
// **IN the doorway, not on the dark side of it, and that is the whole direction rule.** The obvious
// design stands the fixture inside the darker room and points it at the lit one — but which room is
// darker is not a fact about the building, it is a fact about the HOUR. At midnight the inn's
// vestibule takes light from the hall through p1; at noon the porch's own DAYLIGHT fixture lights
// the vestibule through p0 and the flow reverses, g0 into g1, which is exactly the owner's "by day
// only doorways admit light". A fixture standing on one side would have to be moved, and a light
// that moves through a wall when a cloud passes is worse than the seam it fixes. An APERTURE has no
// side: it sits in the hole, it is claimed by both rooms, and only its INTENSITY and COLOUR move —
// it carries whichever neighbour is brighter this frame, in that neighbour's own hue, and the
// direction reverses without anything moving. The cost is that the brighter room also gains a
// doorway glow (bounded at `BLEED_K` of what it already has there, falling off within a yard or
// two), which is what a lit doorway looks like anyway.
//
// IT IS THE SAME ENTITY SHAPE as a daylight fixture, and it carries [`DaylightFixture`] as its
// marker for exactly that reason — three exclusions ride on that component and none of them wants
// re-deciding here: `benilla_app::torch_shadow` refuses it as a cube-shadow caster
// (`Without<DaylightFixture>` — a 6 yd-wide doorway is not a point source), the packer skips
// `interiorGain` on it (`Has<DaylightFixture>`, and this module folds that gain into the TARGET
// already, so packing it again would dim the doorway twice), and it is neither a
// [`super::SyntheticFireLight`] (no `fireLightGain` over it) nor a [`super::FlameFlicker`] (a
// doorway does not wobble). [`BleedFixture`] beside it carries what is bleed-specific — the two
// rooms and the portal plane — and doubles as the query filter that keeps the two per-frame
// systems' `PointLight` access disjoint.
//
// NO FEEDBACK, ONE LEVEL. A bleed must not read another bleed as if it were a candle, or a run of
// doorways would ring up without bound. But refusing outright breaks the very case this exists for:
// the inn's vestibule is TWO doorways from the hall, so a p1 bleed that cannot see the p2 bleed sees
// nothing at all. So [`update_bleed_fixtures`] runs the evaluation TWICE in one system over its own
// local arrays: pass A lights every bleed from the REAL fixtures only, pass B re-lights each one
// from the real fixtures plus the PASS-A value of any OTHER bleed standing in the room being
// sampled. That is exactly one level of chaining, it resolves inside a single frame (no cross-frame
// accumulation), and the order the bleeds happen to be queried in cannot change the answer.
// ===========================================================================================

/// The fraction of a neighbour's irradiance a doorway transmits, measured at the portal plane. A
/// real opening passes most of the light that reaches it (it is a hole, not a filter) and loses the
/// rest to the frame, the door leaf and the solid angle the wall subtends — so this is deliberately
/// near, but under, 1: at the threshold the dark room reads a little dimmer than the lit one, which
/// is the direction a threshold reads in life.
pub const BLEED_K: f32 = 0.6;

/// MONKEY (portal bleed): the bleed-specific half of the fixture — everything [`DaylightFixture`]
/// has no field for, plus the marker that keeps the two per-frame systems' `PointLight` access
/// disjoint (`update_daylight_fixtures` filters `Without<BleedFixture>`).
#[derive(Component, Clone, Copy, Debug)]
pub struct BleedFixture {
    /// The two interior groups this doorway joins. Both are sampled every frame and the BRIGHTER one
    /// is what the fixture carries — see the aperture argument in the block above.
    pub sides: [u16; 2],
    /// The portal's centre in BEVY WORLD space: the plane both sides are measured at, and the point
    /// the calibration matches. World rather than model space for [`super::ClaimFade::center`]'s own
    /// reason — the per-frame evaluation has world-space fixture positions and nothing else.
    pub probe: Vec3,
}

/// One interior<->interior doorway, resolved in WMO MODEL space — the bleed twin of
/// [`DaylightSeed`], and deliberately a separate type: a daylight seed's whole payload is the
/// CALIBRATION geometry (`hz`, the floor fragment under the opening), and a bleed seed's is the two
/// ROOMS and the plane between them.
#[derive(Clone, Copy, Debug)]
pub struct BleedSeed {
    /// The two interior groups the portal joins.
    pub sides: [u16; 2],
    pub portal: u16,
    /// The portal's centre, z-clamped exactly like [`seed_point`] so a two-storey arch stands near
    /// its sill rather than at its apex. This is both the spawn position and the probe point: the
    /// fixture IS the opening.
    pub pos: [f32; 3],
    pub diag: f32,
    /// The portal polygon's area — the shared budget's rank key.
    pub area: f32,
}

/// Every portal of one root whose BOTH sides are interior-class groups, ranked by area — the bleed
/// seed's whole rule, and it needs nothing but the group table and the portal graph.
///
/// A portal only one group names is excluded by construction (its missing side is the outside, which
/// is the DAYLIGHT portal seed's population, not this one), and so is any portal touching an
/// exterior-class group: the porch outside the inn's front door is dark at night and lit by the sky
/// by day, and neither is a room whose candles want carrying inward. The one size filter is
/// [`APERTURE_MIN_DIAG`] — below it a "portal" is a sliver an artist left between two halves of one
/// room, and a light in it would be a light inside a wall.
pub fn bleed_seeds(groups: &[WmoGroupInfo], portals: PortalGraph<'_>) -> Vec<BleedSeed> {
    if !bleed_enabled() {
        return Vec::new();
    }
    let mut sides: Vec<(u16, Vec<u16>)> = Vec::new();
    let note = |portal: u16, group: u16, sides: &mut Vec<(u16, Vec<u16>)>| {
        match sides.iter_mut().find(|(p, _)| *p == portal) {
            Some((_, gs)) => {
                if !gs.contains(&group) {
                    gs.push(group);
                }
            }
            None => sides.push((portal, vec![group])),
        }
    };
    for (gi, (start, count)) in portals.slices.iter().enumerate() {
        let (start, count) = (usize::from(*start), usize::from(*count));
        for r in portals.refs.get(start..start + count).unwrap_or(&[]) {
            note(r.portal, gi as u16, &mut sides);
            note(r.portal, r.group, &mut sides);
        }
    }
    let mut out = Vec::new();
    for (portal, gs) in &sides {
        if gs.len() != 2 {
            continue;
        }
        if !gs
            .iter()
            .all(|g| groups.get(usize::from(*g)).is_some_and(|g| g.interior))
        {
            continue;
        }
        let info = portals.infos.get(usize::from(*portal));
        let verts = info.and_then(|i| {
            let s = usize::from(i.start_vertex);
            portals.vertices.get(s..s + usize::from(i.count))
        });
        let Some((lo, hi)) = verts.and_then(|v| bounds(v.iter().copied())) else {
            continue;
        };
        let diag = diagonal(lo, hi);
        if diag < APERTURE_MIN_DIAG {
            continue;
        }
        let area = benilla_formats::room_claim::portal_area(&portals, *portal)
            .unwrap_or_else(|| opening_area(lo, hi));
        // …and a portal at or above [`benilla_formats::room_claim::SPLIT_PORTAL_MIN_AREA`] is NOT a
        // doorway. It is a room the artist CUT IN HALF for rendering, and the claim rule already
        // says so: `ClaimHow::Split` gives a fixture on either side a HARD claim on the other, so
        // both halves are lit by the same candles at full weight and there is nothing to transmit.
        // A bleed there would stand a light in the middle of one room and add `BLEED_K` of that
        // room's own light back to it. MEASURED, this is the whole difference between "the Goldshire
        // inn's doorways" and "Northshire abbey changed character": the abbey's four biggest portals
        // are 252.6 / 50.9 / 50.7 / 47.1 yd^2 splits of its nave and would have taken half the
        // building's budget, while its real doorways are the 14-16 yd^2 ones underneath them. The
        // inn's p8 (95.9) and p3 (35.1) go the same way, which is what lets BOTH halves of its
        // entrance chain (p2 at 20.5 and p1 at 17.3) into the eight.
        if area >= benilla_formats::room_claim::SPLIT_PORTAL_MIN_AREA {
            continue;
        }
        let (pos, _) = seed_point(lo, hi);
        out.push(BleedSeed {
            sides: [gs[0], gs[1]],
            portal: *portal,
            pos,
            diag,
            area,
        });
    }
    // Deterministic, largest first — the shared budget TRUNCATES, so an unstable order would give
    // the same building different doorways on different launches.
    out.sort_by(|a, b| b.area.total_cmp(&a.area).then(a.portal.cmp(&b.portal)));
    out
}

/// **THE ONE SELECTION ENTRY POINT** a placement calls: every opening of one WMO root that gets a
/// fixture, daylight and bleed together, sharing the [`MAX_DAYLIGHT_PER_PLACEMENT`] budget and
/// ranked against each other by AREA.
///
/// One budget rather than two because the resource being spent is the same one — slots in the packed
/// point table, and fragments walked by every interior pixel of the building — and because the
/// ranking question is the same question: which of this building's openings matter. A city's district
/// portals (338 yd^2) still take every slot; the Lion's Pride Inn's eight begin with the hall doorway
/// p2 (20.5, bleed), its porch door p0 (17.3, daylight) and the vestibule doorway p1 (17.3, bleed) —
/// i.e. both halves of its entrance chain fit, because the two portals that used to outrank them
/// (95.9 and 35.1 yd^2) are SPLIT cuts the bleed seed now refuses.
///
/// Ties go to DAYLIGHT (`d < b` is what promotes a bleed): the sun standing in an opening is the
/// bigger visual fact, and the inn's p0/p1 pair is exactly a tie at 17.34 yd^2.
pub fn placement_openings<'a, I>(
    groups: &[WmoGroupInfo],
    portals: PortalGraph<'_>,
    batches: I,
) -> (Vec<DaylightSeed>, Vec<BleedSeed>)
where
    I: IntoIterator<Item = (u16, bool, &'a [[f32; 3]])>,
{
    let day = daylight_seeds_ranked(groups, portals, batches);
    let bleed = bleed_seeds(groups, portals);
    // Two descending-by-area lists merged into one budget. A plain interleave rather than a
    // concat-and-sort so each rule's own tie-break survives (both are already deterministic).
    let (mut di, mut bi) = (0usize, 0usize);
    let (mut day_keep, mut bleed_keep) = (Vec::new(), Vec::new());
    while day_keep.len() + bleed_keep.len() < MAX_DAYLIGHT_PER_PLACEMENT {
        match (day.get(di).map(|s| s.area), bleed.get(bi).map(|c| c.area)) {
            (Some(d), Some(b)) if d < b => {
                bleed_keep.push(bleed[bi]);
                bi += 1;
            }
            (Some(_), _) => {
                day_keep.push(day[di]);
                di += 1;
            }
            (None, Some(_)) => {
                bleed_keep.push(bleed[bi]);
                bi += 1;
            }
            (None, None) => break,
        }
    }
    (day_keep, bleed_keep)
}

/// `entry * (1 - smoothstep(0, radius, max(|P - center| - slack, 0)))` — the CPU mirror of
/// `static_gx.wgsl`'s `interior_room_admits` fade arm. `radius == 0` is the hard claim.
fn claim_fade_weight(radius: f32, slack: f32, entry: f32, center: Vec3, p: Vec3) -> f32 {
    if radius <= 0.0 {
        return 1.0;
    }
    let t = (((p - center).length() - slack).max(0.0) / radius.max(1e-4)).clamp(0.0, 1.0);
    entry * (1.0 - t * t * (3.0 - 2.0 * t))
}

/// `INTERIOR_CORE_GAIN / (1 + (d/INTERIOR_CORE_YD)^2)` — the shader's soft-core inverse square, split
/// out of [`direct_profile`] because the bleed lane measures IRRADIANCE (no surface, so no `N.L`)
/// while the daylight calibration measures a floor fragment.
fn core_atten(d: f32) -> f32 {
    INTERIOR_CORE_GAIN / (1.0 + (d / INTERIOR_CORE_YD).powi(2))
}

/// `c / max(1, max channel)` — `static_gx.wgsl`'s `c_norm`. The table commits raw over-gamut colour
/// x intensity and the shader normalises it per fixture, so anything predicting the shader's answer
/// has to normalise it too.
fn commit_norm(c: [f32; 3]) -> [f32; 3] {
    let peak = c[0].max(c[1]).max(c[2]).max(1.0);
    [c[0] / peak, c[1] / peak, c[2] / peak]
}

/// `WOW_BLEED=0` — the feature's own kill switch, read once, for the same A/B reason
/// [`daylight_modes`] has one. Anything else (absent included) leaves it on.
fn bleed_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("WOW_BLEED").ok().as_deref(),
            Some("0" | "off" | "false")
        )
    })
}

impl BleedSeed {
    /// The shared component this seed becomes. `cal_d`/`cal_ndl` state the BLEED lane's calibration
    /// rather than the daylight lane's: the match is made at the PORTAL PLANE, [`NUDGE_YD`] from the
    /// fixture, as irradiance (`N.L = 1`) — a doorway has no one surface normal, and the wrapped
    /// Lambert spans only 1/3..1 anyway, so choosing a normal would add a factor to both sides of the
    /// ratio and information to neither. `group` is the first side, for the dump and the claim head;
    /// the pair that matters lives on [`BleedFixture::sides`].
    pub fn fixture(&self, instance: Entity) -> DaylightFixture {
        DaylightFixture {
            instance,
            group: self.sides[0],
            portal: Some(self.portal),
            how: DaylightHow::Bleed,
            reach: daylight_authored_reach(self.diag),
            cal_d: NUDGE_YD,
            cal_ndl: 1.0,
        }
    }

    /// The claim set — the SAME rule every other fixture's is built by, seeded with BOTH rooms as
    /// its MOLR referrers, which is the aperture argument spelled in claim terms: the doorway is not
    /// in one room, it is in the wall between two, and it lights both. Groups one hop beyond either
    /// come in soft, exactly as they do for a candle standing in the room.
    pub fn claims(
        &self,
        groups: &[WmoGroupInfo],
        portals: PortalGraph<'_>,
    ) -> Vec<benilla_formats::room_claim::Claim> {
        benilla_formats::room_claim::room_claims(
            groups,
            portals,
            self.pos,
            benilla_formats::room_claim::claim_reach(daylight_authored_reach(self.diag)),
            &self.sides,
        )
    }

    /// The VISIBILITY gate — the doorway exists while EITHER of its rooms is drawn
    /// ([`crate::wmo_portal::WmoGroupVis::drawn_by`] ORs), which is the same fail-open a prop several
    /// rooms name gets. A doorway culled with one of its two rooms would switch a light off in the
    /// room still on screen.
    pub fn rooms(&self, instance: Entity) -> super::LightRooms {
        super::LightRooms::new(crate::wmo_portal::WmoGroupVis {
            instance,
            groups: Arc::from(self.sides.as_slice()),
        })
    }
}

/// One interior-lane source as the bleed evaluation needs it — the packer's own arithmetic, minus
/// the flicker (a bleed must not breathe with the hearth it carries: the pool it feeds is a whole
/// doorway, and a wobbling doorway is a wobbling building).
struct BleedSource<'a> {
    pos: Vec3,
    /// Committed colour, NORMALISED like the shader's `c_norm`.
    c_norm: [f32; 3],
    r_eff: f32,
    claims: Option<&'a LightLitRooms>,
}

/// The irradiance (direct + fill, as a colour) the fixtures claiming `group` of `instance` deliver
/// at `p` — `static_gx.wgsl`'s `interior_room_light` with `N.L` pinned to 1 and the ambient floor
/// left out. The floor is excluded because the OTHER side has its own copy of it; carrying the
/// neighbour's across would make every doorway in the world transmit `interiorAmbient` as if it were
/// light, and every room would end up at twice its own floor.
fn room_irradiance(
    sources: &[BleedSource<'_>],
    extra: &[(Vec3, [f32; 3], f32)],
    p: Vec3,
    instance: Entity,
    group: u16,
    k_fill: f32,
) -> [f32; 3] {
    let mut direct = [0.0f32; 3];
    let mut fill = [0.0f32; 3];
    let mut add = |pos: Vec3, c_norm: [f32; 3], r_eff: f32, w: f32| {
        let d = (pos - p).length();
        if d > INTERIOR_FILL_SPAN * r_eff {
            return;
        }
        let a = core_atten(d) * interior_window(d, r_eff, INTERIOR_DIRECT_POW) * w;
        let fw = k_fill * interior_window(d, INTERIOR_FILL_SPAN * r_eff, INTERIOR_FILL_POW) * w;
        // The fill is HALF-DESATURATED and taken as a MAX over fixtures, exactly like the shader: an
        // ambient room glow that summed would scale with the candle count.
        let grey = 0.299 * c_norm[0] + 0.587 * c_norm[1] + 0.114 * c_norm[2];
        for i in 0..3 {
            direct[i] += c_norm[i] * a;
            fill[i] = fill[i].max(0.5 * (c_norm[i] + grey) * fw);
        }
    };
    for s in sources {
        // Range first, claim second -- the reverse of the shader's order, deliberately. The shader
        // tests the gate first because a fragment's room claims one or two of the table's fixtures
        // and the gate throws away the most; here the loop is over every interior source in the
        // WORLD against a handful of doorways, so the cheap squared-distance reject is what throws
        // away the most. The answer is identical either way (both terms multiply).
        if (s.pos - p).length_squared() > (INTERIOR_FILL_SPAN * s.r_eff).powi(2) {
            continue;
        }
        let w = claim_weight(s.claims, instance, group, p);
        if w > 0.0 {
            add(s.pos, s.c_norm, s.r_eff, w);
        }
    }
    // The ONE level of chaining: another doorway's bleed, standing in the room being sampled, with
    // the value pass A gave it. It claims that room hard (both its sides are MOLR seeds), so its
    // weight there is 1.
    for (pos, c_norm, r_eff) in extra {
        add(*pos, *c_norm, *r_eff, 1.0);
    }
    [
        direct[0] + fill[0],
        direct[1] + fill[1],
        direct[2] + fill[2],
    ]
}

/// The CPU mirror of `static_gx.wgsl`'s `interior_room_admits`, NON-strict arm: may this fixture
/// light room `(instance, group)`, and at what weight. Every fail-OPEN arm the shader has is kept
/// (no claims = the packer's UNGATED head, which is also what `interiorRoomGate 0` packs), because
/// predicting a gate with a stricter rule than the gate's own would calibrate a doorway against light
/// the room is not actually getting.
fn claim_weight(claims: Option<&LightLitRooms>, instance: Entity, group: u16, p: Vec3) -> f32 {
    let Some(c) = claims.filter(|c| !c.rooms.groups.is_empty()) else {
        return 1.0;
    };
    if c.rooms.instance != instance {
        return 0.0;
    }
    // The packer keeps at most `ROOM_CLAIM_MAX` claims per fixture (highest priority first), so a
    // claim past the cap is one the shader will never see.
    let n = c.rooms.groups.len().min(super::ROOM_CLAIM_MAX);
    for k in 0..n {
        if c.rooms.groups[k] & !super::LIT_ROOM_EXT_DENY != group {
            continue;
        }
        let Some(f) = c.fades.get(k) else {
            return 1.0; // a producer with no fades: every claim hard, the pre-soft-claims gate
        };
        return claim_fade_weight(f.radius, f.slack, f.entry, f.center, p);
    }
    0.0
}

/// The packed intensity in `[0, 1]` that makes this fixture deliver `BLEED_K x lit` at the portal
/// plane, and the hue it delivers it in (the brighter side's own chroma renormalised to peak 1 — what
/// the packed colour carries, so the shader's `c_norm` reads back exactly `hue * I` while `I <= 1`).
///
/// Both sides of the equality are IRRADIANCE in the room lane's own pre-rolloff units, so no exposure
/// inversion is involved and the answer cannot be perturbed by the user's `interiorExposure` — a
/// budget matched against a budget stays matched however the rolloff is tuned. That is the difference
/// from the daylight calibration, which has to hit a DISPLAY value the reference sky law is already
/// rendering a plank away.
fn bleed_intensity(lit: [f32; 3], r_eff: f32, k_fill: f32) -> ([f32; 3], f32) {
    let peak = lit[0].max(lit[1]).max(lit[2]);
    if peak <= 1e-6 {
        return ([0.0; 3], 0.0);
    }
    let hue = [lit[0] / peak, lit[1] / peak, lit[2] / peak];
    let hue_lum = luminance(hue).max(1e-3);
    // What this fixture is worth at the portal plane: it stands NUDGE_YD from it. The fill is taken
    // as ADDITIVE here where the shader takes a max over fixtures — at half a yard the bleed wins that
    // max against anything else in the room, and where it does not the fixture simply lands a touch
    // under nominal, which errs toward the dark side of the trade.
    let self_factor = core_atten(NUDGE_YD) * interior_window(NUDGE_YD, r_eff, INTERIOR_DIRECT_POW)
        + k_fill * interior_window(NUDGE_YD, INTERIOR_FILL_SPAN * r_eff, INTERIOR_FILL_POW);
    if self_factor <= 1e-6 {
        return (hue, 0.0);
    }
    (
        hue,
        (BLEED_K * luminance(lit) / (hue_lum * self_factor)).clamp(0.0, 1.0),
    )
}

/// Re-aim every bleed fixture at this frame's room light — see the PORTAL BLEED block above for the
/// rule, for why the fixture is an aperture with no side, and for why the evaluation runs in two
/// passes inside one system.
///
/// Scheduled after [`update_daylight_fixtures`] (so a doorway reading the porch's daylight fixture as
/// its brighter side reads THIS frame's sun, not last frame's) and before
/// `global_light::classify_light_lanes`, which is chained before the packer. The one-frame lag that
/// remains is structural and harmless: `update_daylight_fixtures` INSERTS and REMOVES its `PointLight`
/// through `Commands`, so on the single frame a daylight fixture switches on at dawn or off at dusk
/// the bleed reading it is one frame stale. Its steady-state value is written in place and is current.
pub fn update_bleed_fixtures(
    mut commands: Commands,
    knobs: Res<DynamicInteriors>,
    fire_gain: Res<FireLightGain>,
    mut bleeds: Query<(
        Entity,
        &DaylightFixture,
        &BleedFixture,
        &GlobalTransform,
        Option<&LightReach>,
        Option<&mut PointLight>,
    )>,
    sources: Query<
        (
            &PointLight,
            &GlobalTransform,
            Option<&LightReach>,
            Option<&LightLitRooms>,
            Option<&LightLane>,
            Option<&super::LightRooms>,
            Has<SyntheticFireLight>,
            Has<DaylightFixture>,
        ),
        Without<BleedFixture>,
    >,
    time: Res<Time>,
    mut last_dump: Local<f64>,
) {
    // The room lane being off is the feature being off, exactly as for a daylight fixture.
    let on = knobs.enabled && bleed_enabled();
    // The packed fill gain — `interiorFill x interiorGain`, the same product `build_light_data`
    // writes into `point_count.z`.
    let k_fill = knobs.fill.max(0.0) * knobs.interior_gain;
    // Which placements have a doorway to light at all. A source claiming a DIFFERENT building is
    // refused by `claim_weight`'s instance compare anyway, so dropping it here changes no answer --
    // it just stops a city's few thousand interior fixtures being walked once per doorway per side
    // per pass. An UNGATED source (no claims) is kept: the gate fails open on it, so it lights every
    // room in range, this building's included.
    let live: std::collections::HashSet<Entity> =
        bleeds.iter().map(|(_, fx, ..)| fx.instance).collect();
    let src: Vec<BleedSource<'_>> = if on && !live.is_empty() {
        sources
            .iter()
            .filter_map(|(pl, gt, reach, lit_rooms, lane, rooms, synthetic, daylight)| {
                if lit_rooms.is_some_and(|c| {
                    !c.rooms.groups.is_empty() && !live.contains(&c.rooms.instance)
                }) {
                    return None;
                }
                // The packer's own lane verdict, fallback included (a light lives a frame or two
                // before the classifier's first pass).
                if !lane.map_or_else(|| rooms.is_some(), |l| l.interior) {
                    return None;
                }
                let c = pl.color.to_linear();
                let base = pl.intensity / (4.0 * std::f32::consts::PI);
                // `fireLightGain` IS mirrored (a dial that darkens the hearth must darken what the
                // doorway carries of it); the FLICKER deliberately is not.
                let s = base * if synthetic { fire_gain.0.max(0.0) } else { 1.0 };
                // …and `interiorGain`, which the packer applies to every interior fixture EXCEPT a
                // `DaylightFixture`. Mirroring that exclusion here is what keeps the doorway from
                // being dimmed twice: a bleed fixture IS a `DaylightFixture`, so the packer will not
                // scale it, and the gain therefore has to be in the TARGET.
                let g = if daylight { 1.0 } else { knobs.interior_gain };
                let c_norm = commit_norm([
                    (c.red * s * g).max(0.0),
                    (c.green * s * g).max(0.0),
                    (c.blue * s * g).max(0.0),
                ]);
                let r = reach
                    .map(|r| r.0)
                    .filter(|r| *r > 0.5)
                    .unwrap_or_else(|| super::m2_light_reach(base));
                Some(BleedSource {
                    pos: gt.translation(),
                    c_norm,
                    r_eff: super::interior_reach(r, knobs.atten_scale),
                    claims: lit_rooms,
                })
            })
            .collect()
    } else {
        Vec::new()
    };

    /// One bleed fixture's per-frame working row — the two passes' shared scratch.
    struct Row {
        e: Entity,
        instance: Entity,
        sides: [u16; 2],
        portal: u16,
        authored: f32,
        probe: Vec3,
        pos: Vec3,
        r_eff: f32,
        hue: [f32; 3],
        i: f32,
    }
    // The brighter of the two sides, as a colour: the aperture carries whichever neighbour has more
    // light at the plane THIS frame, which is how the direction reverses between midnight and noon
    // without the fixture moving.
    let brighter = |a: [f32; 3], b: [f32; 3]| if luminance(a) >= luminance(b) { a } else { b };

    // --- PASS A: every bleed lit by the REAL fixtures alone.
    let mut rows: Vec<Row> = Vec::new();
    for (e, fx, bl, gt, reach, _) in &bleeds {
        let r_eff = super::interior_reach(reach.map_or(fx.reach, |r| r.0), knobs.atten_scale);
        let lit = if on {
            brighter(
                room_irradiance(&src, &[], bl.probe, fx.instance, bl.sides[0], k_fill),
                room_irradiance(&src, &[], bl.probe, fx.instance, bl.sides[1], k_fill),
            )
        } else {
            [0.0; 3]
        };
        let (hue, i) = bleed_intensity(lit, r_eff, k_fill);
        rows.push(Row {
            e,
            instance: fx.instance,
            sides: bl.sides,
            portal: fx.portal.unwrap_or(u16::MAX),
            authored: fx.reach,
            probe: bl.probe,
            pos: gt.translation(),
            r_eff,
            hue,
            i,
        });
    }
    // --- PASS B: …and again, with each OTHER bleed's PASS-A value standing in the room being
    // sampled. One level, and no more: `extra` is built from `rows` as pass A left it, never from the
    // values being written now, so a ring of doorways cannot ring up. A doorway's own contribution is
    // excluded from both of its own sides (`j != i`) — it must never calibrate against itself.
    // Grouped by placement first: a doorway can only ever chain to a doorway of its OWN building, and
    // a city holds hundreds of both, so the alternative is a quadratic scan across every building in
    // the frame to find the seven rows that could matter.
    let mut by_instance: HashMap<Entity, Vec<usize>> = HashMap::new();
    for (i, r) in rows.iter().enumerate() {
        by_instance.entry(r.instance).or_default().push(i);
    }
    let mut out: Vec<([f32; 3], f32, [f32; 3])> = Vec::with_capacity(rows.len());
    for (i, r) in rows.iter().enumerate() {
        let siblings = by_instance.get(&r.instance).map_or(&[][..], |v| &v[..]);
        let side_extra = |g: u16| -> Vec<(Vec3, [f32; 3], f32)> {
            siblings
                .iter()
                .map(|j| (*j, &rows[*j]))
                .filter(|(j, o)| *j != i && o.i > 1e-4 && o.sides.contains(&g))
                .map(|(_, o)| {
                    (
                        o.pos,
                        [o.hue[0] * o.i, o.hue[1] * o.i, o.hue[2] * o.i],
                        o.r_eff,
                    )
                })
                .collect()
        };
        let lit = if on {
            brighter(
                room_irradiance(
                    &src,
                    &side_extra(r.sides[0]),
                    r.probe,
                    r.instance,
                    r.sides[0],
                    k_fill,
                ),
                room_irradiance(
                    &src,
                    &side_extra(r.sides[1]),
                    r.probe,
                    r.instance,
                    r.sides[1],
                    k_fill,
                ),
            )
        } else {
            [0.0; 3]
        };
        let (hue, intensity) = bleed_intensity(lit, r.r_eff, k_fill);
        out.push((hue, intensity, lit));
    }
    for (r, (hue, intensity, _)) in rows.iter().zip(out.iter()) {
        // Same existence rule as a daylight fixture, and for the same reason: the packer has no
        // zero-intensity skip, so a doorway between two unlit rooms must leave the table outright
        // rather than hold one of its 256 slots to contribute nothing.
        let Ok((_, _, _, _, _, pl)) = bleeds.get_mut(r.e) else {
            continue;
        };
        match pl {
            Some(mut pl) if *intensity > 1e-4 => {
                pl.color = Color::linear_rgb(hue[0], hue[1], hue[2]);
                pl.intensity = 4.0 * std::f32::consts::PI * intensity;
            }
            Some(_) => {
                commands.entity(r.e).remove::<PointLight>();
            }
            None if *intensity > 1e-4 => {
                commands
                    .entity(r.e)
                    .insert(daylight_point_light(*hue, *intensity));
            }
            None => {}
        }
    }
    // `WOW_POINTS_DUMP`: the BLEED rows, tagged like the daylight ones and carrying the number the
    // whole rule turns on — the brighter side's irradiance at the portal plane.
    if let Some(mode) = std::env::var_os("WOW_POINTS_DUMP") {
        let every = if mode == *"frame" { 0.0 } else { 1.0 };
        let now = time.elapsed_secs_f64();
        if now - *last_dump >= every {
            *last_dump = now;
            let n = out.iter().filter(|(_, i, _)| *i > 1e-4).count();
            if n > 0 {
                eprintln!(
                    "[daylight] {n} BLEED fixture(s) lit — k {BLEED_K:.2}, fill gain {k_fill:.4}"
                );
                for (r, (hue, i, lit)) in rows
                    .iter()
                    .zip(out.iter())
                    .filter(|(_, (_, i, _))| *i > 1e-4)
                    .take(8)
                {
                    eprintln!(
                        "  BLEED {}  inst {} g{} <-> g{}  p{}  reach {:5.2} (R {:5.2})  lit [{:.5},{:.5},{:.5}] lum {:.5}  hue [{:.3},{:.3},{:.3}]  I {i:.5}",
                        r.e,
                        r.instance,
                        r.sides[0],
                        r.sides[1],
                        r.portal,
                        r.authored,
                        r.r_eff,
                        lit[0],
                        lit[1],
                        lit[2],
                        luminance(*lit),
                        hue[0],
                        hue[1],
                        hue[2],
                    );
                }
            }
        }
    }
}

/// Wire the per-frame aim in. One `add_systems` — the fixtures themselves are spawned by the
/// placement lane and need no resource of their own.
pub(super) fn register(app: &mut App) {
    app.add_systems(
        PostUpdate,
        (
            update_daylight_fixtures,
            // MONKEY (portal bleed): AFTER the daylight aim, so a doorway reading the porch's
            // daylight fixture as its lit-side source reads this frame's sun, not last frame's.
            update_bleed_fixtures,
        )
            .chain()
            .before(super::global_light::classify_light_lanes),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use benilla_formats::{WmoPortalInfo, WmoPortalRef};

    fn group(interior: bool, lo: [f32; 3], hi: [f32; 3]) -> WmoGroupInfo {
        WmoGroupInfo {
            interior,
            show_skybox: false,
            bbox_min: lo,
            bbox_max: hi,
        }
    }

    /// The reach formula: linear in the opening's diagonal, floored at 6 and capped at 20.
    #[test]
    fn reach_is_the_clamped_opening_width() {
        assert_eq!(daylight_reach(0.0), 6.0); // the floor, not 4
        assert_eq!(daylight_reach(2.0), 7.0);
        assert!((daylight_reach(2.6) - 7.9).abs() < 1e-5); // a 1.2 x 2.2 yd door
        assert_eq!(daylight_reach(100.0), 20.0); // the cap
        // Monotone in between, so a wider opening never throws light LESS far.
        let mut prev = 0.0;
        for i in 0..40 {
            let r = daylight_reach(i as f32 * 0.5);
            assert!(r >= prev);
            prev = r;
        }
    }

    /// The class test: exactly one INTERIOR side and an exterior (or absent) other side.
    #[test]
    fn only_exterior_facing_portals_are_doorways() {
        // g0 interior room (west of the doorway plane), g1 exterior shell, g2 a second interior
        // room (east of it) — the two rooms sit on OPPOSITE sides of x = 0, which is what gives the
        // nudge a sign to find.
        let groups = [
            group(true, [-10.0, -5.0, 0.0], [0.0, 5.0, 5.0]),
            group(false, [-20.0, -20.0, 0.0], [20.0, 20.0, 10.0]),
            group(true, [0.0, -5.0, 0.0], [10.0, 5.0, 5.0]),
        ];
        // One 2 x 2 yd quad reused by all three portals (position is irrelevant to the class test).
        let vertices = [
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 2.0],
            [0.0, -1.0, 2.0],
        ];
        let infos = [
            WmoPortalInfo { start_vertex: 0, count: 4, plane: [1.0, 0.0, 0.0, 0.0] },
            WmoPortalInfo { start_vertex: 0, count: 4, plane: [1.0, 0.0, 0.0, 0.0] },
            WmoPortalInfo { start_vertex: 0, count: 4, plane: [1.0, 0.0, 0.0, 0.0] },
        ];
        // p0: g0 <-> g1 (interior to exterior — a doorway).
        // p1: g0 <-> g2 (room to room — not).
        // p2: named by g2 alone (a portal to the outside — a doorway).
        let refs = [
            WmoPortalRef { portal: 0, group: 1, side: 1 },
            WmoPortalRef { portal: 1, group: 2, side: 1 },
            WmoPortalRef { portal: 0, group: 0, side: -1 },
            WmoPortalRef { portal: 1, group: 0, side: -1 },
            WmoPortalRef { portal: 2, group: 2, side: 1 },
        ];
        // g0 owns refs 0..2, g1 owns 2..3, g2 owns 3..5.
        let slices = [(0, 2), (2, 1), (3, 2)];
        let portals = PortalGraph {
            vertices: &vertices,
            infos: &infos,
            refs: &refs,
            slices: &slices,
        };
        let seeds = daylight_seeds(&groups, portals, std::iter::empty());
        let mut found: Vec<(u16, u16)> =
            seeds.iter().map(|s| (s.portal.unwrap(), s.group)).collect();
        found.sort_unstable();
        // p0 seeded into the interior side g0; p2 into g2; p1 (room to room) refused outright.
        assert_eq!(found, vec![(0, 0), (2, 2)]);
        // …and the seed really did step INTO the room: the quad lies on the plane x = 0, so the
        // nudge must have taken it a full NUDGE_YD off that plane.
        let s = seeds.iter().find(|s| s.portal == Some(0)).unwrap();
        assert!((s.pos[0] + NUDGE_YD).abs() < 1e-4, "pos {:?}", s.pos);
        // …and the OTHER room's own doorway steps the other way, off the same plane.
        let s2 = seeds.iter().find(|s| s.portal == Some(2)).unwrap();
        assert!((s2.pos[0] - NUDGE_YD).abs() < 1e-4, "pos {:?}", s2.pos);
    }

    /// The APERTURE seed: an EXT-class batch of an INTERIOR group, and nothing else.
    #[test]
    fn apertures_are_ext_batches_of_interior_groups() {
        let groups = [
            group(true, [-5.0, -5.0, 0.0], [5.0, 5.0, 5.0]),
            group(false, [-20.0, -20.0, 0.0], [20.0, 20.0, 10.0]),
        ];
        // A 2 x 3 yd window slab on g0's south wall; the same slab on the exterior shell (which
        // must not seed — the sky already lights it); and an INT-class batch of g0 (the floor).
        let window = [
            [-1.0, -5.0, 1.0],
            [1.0, -5.0, 1.0],
            [1.0, -4.9, 4.0],
            [-1.0, -4.9, 4.0],
        ];
        let floor = [[-4.0, -4.0, 0.0], [4.0, 4.0, 0.0], [4.0, -4.0, 0.0]];
        let seeds = daylight_seeds(
            &groups,
            PortalGraph::default(),
            [
                (0u16, true, &window[..]),
                (0u16, false, &floor[..]),
                (1u16, true, &window[..]),
            ],
        );
        assert_eq!(seeds.len(), 1);
        assert_eq!(seeds[0].group, 0);
        assert_eq!(seeds[0].how, DaylightHow::Aperture);
        // Nudged toward the room centre, i.e. +Y off the south wall.
        assert!(seeds[0].pos[1] > -5.0);
        // The sill is at z = 1 and the centre at z = 2.5, so the calibration stands 1.5 yd up.
        let fx = seeds[0].fixture(Entity::PLACEHOLDER);
        assert!((fx.cal_d - (0.25f32 + 1.5 * 1.5).sqrt()).abs() < 1e-4);
        assert!(fx.cal_ndl > 0.9); // a floor fragment under the sill looks almost straight up at it
    }

    /// The budget: largest openings first, capped per placement.
    #[test]
    fn the_budget_keeps_the_widest_openings() {
        let groups = [group(true, [-50.0, -50.0, 0.0], [50.0, 50.0, 20.0])];
        // Twelve windows of strictly increasing area, all EXT-class batches of the one room and all
        // inside `APERTURE_MAX_DIAG` (this test is about the BUDGET, not the size filter).
        let quads: Vec<[[f32; 3]; 4]> = (1..=12)
            .map(|i| {
                let w = i as f32 * 0.5;
                [
                    [-w, -50.0, 1.0],
                    [w, -50.0, 1.0],
                    [w, -49.9, 1.0 + w],
                    [-w, -49.9, 1.0 + w],
                ]
            })
            .collect();
        let seeds = daylight_seeds(
            &groups,
            PortalGraph::default(),
            quads.iter().map(|q| (0u16, true, &q[..])),
        );
        assert_eq!(seeds.len(), MAX_DAYLIGHT_PER_PLACEMENT);
        // Descending area, and the smallest survivor is window 5 (2*2.5 wide x 2.5 tall = 12.5
        // yd^2) — i.e. the four narrowest of the twelve were the ones dropped.
        for w in seeds.windows(2) {
            assert!(w[0].area >= w[1].area);
        }
        assert!(
            (seeds.last().unwrap().area - 12.5).abs() < 1e-3,
            "smallest survivor {:?}",
            seeds.last().unwrap().area
        );
    }

    /// MONKEY (daylight fixtures: boundary): the stitched-threshold seed — it finds the shared
    /// vertex run, refuses a vertical party-wall seam and a corner stitch, and steps into the room.
    #[test]
    fn boundary_seeds_are_the_stitched_threshold() {
        // g0 the room (0..10 square, floor at z = 0), g1 the building shell around it.
        let groups = [
            group(true, [0.0, 0.0, 0.0], [10.0, 10.0, 5.0]),
            group(false, [-10.0, -10.0, -1.0], [20.0, 20.0, 10.0]),
        ];
        // THE DOORWAY: a 4 yd run of floor vertices on the y = 0 wall, authored into BOTH groups.
        let door: Vec<[f32; 3]> = (0..=8)
            .map(|i| [2.0 + i as f32 * 0.5, 0.0, 0.0])
            .collect();
        // A PARTY-WALL seam: the same stitch, but running 5 yd UP the x = 10 wall.
        let wall: Vec<[f32; 3]> = (0..=10)
            .map(|i| [10.0, 8.0, i as f32 * 0.5])
            .collect();
        // A CORNER stitch: three coincident vertices where two walls meet the floor.
        let corner = [[0.0, 10.0, 0.0], [0.0, 10.0, 0.2], [0.2, 10.0, 0.0]];
        // …and interior-only floor the shell never sees.
        let inside = [[5.0, 5.0, 0.0], [6.0, 5.0, 0.0], [5.0, 6.0, 0.0]];
        let mut shared: Vec<[f32; 3]> = door.clone();
        shared.extend_from_slice(&wall);
        shared.extend_from_slice(&corner);
        let seeds = daylight_seeds(
            &groups,
            PortalGraph::default(),
            [
                (0u16, false, &shared[..]),
                (0u16, false, &inside[..]),
                (1u16, false, &shared[..]),
            ],
        );
        assert_eq!(seeds.len(), 1, "{seeds:#?}");
        let s = &seeds[0];
        assert_eq!(s.how, DaylightHow::Boundary);
        assert_eq!(s.group, 0);
        // The threshold run is 4 yd wide, so the opening it stands for ranks as ~16 yd^2.
        assert!((s.diag - 4.0).abs() < 0.2, "diag {}", s.diag);
        assert!((s.area - 16.0).abs() < 2.0, "area {}", s.area);
        // Centred on the 4 yd run (x = 4, y = 0), stepped exactly NUDGE_YD HORIZONTALLY toward the
        // room's vertex centroid — so the step is into the room (+y) and its length is the nudge,
        // but its bearing is the centroid's, not the wall normal's. And it stands SEED_MAX_HZ above
        // the threshold rather than on it.
        let step = ((s.pos[0] - 4.0).powi(2) + s.pos[1].powi(2)).sqrt();
        assert!((step - NUDGE_YD).abs() < 1e-3, "pos {:?} step {step}", s.pos);
        assert!(s.pos[1] > 0.0, "did not step into the room: {:?}", s.pos);
        assert!((s.pos[2] - SEED_MAX_HZ).abs() < 1e-3, "pos {:?}", s.pos);
        assert!((s.hz - SEED_MAX_HZ).abs() < 1e-3);
        // A doorway that ALSO has an authored seed is not seeded twice.
        let with_portal = daylight_seeds(
            &groups,
            PortalGraph::default(),
            [
                (0u16, false, &shared[..]),
                (1u16, false, &shared[..]),
                // an EXT-class batch of the room, right at the same doorway
                (0u16, true, &door[..]),
            ],
        );
        assert_eq!(with_portal.len(), 1, "{with_portal:#?}");
        assert_eq!(with_portal[0].how, DaylightHow::Aperture);
    }


    /// MONKEY (portal bleed): the seed rule — interior<->interior portals only, standing IN the
    /// opening, carrying both sides.
    #[test]
    fn bleed_seeds_are_the_room_to_room_doorways() {
        // g0 and g2 interior rooms either side of x = 0; g1 the building's exterior shell.
        let groups = [
            group(true, [-10.0, -5.0, 0.0], [0.0, 5.0, 5.0]),
            group(false, [-20.0, -20.0, 0.0], [20.0, 20.0, 10.0]),
            group(true, [0.0, -5.0, 0.0], [10.0, 5.0, 5.0]),
        ];
        // One 2 x 2 yd quad at x = 0, sill at z = 0, reused by all three portals.
        let vertices = [
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 2.0],
            [0.0, -1.0, 2.0],
        ];
        let infos = [
            WmoPortalInfo { start_vertex: 0, count: 4, plane: [1.0, 0.0, 0.0, 0.0] },
            WmoPortalInfo { start_vertex: 0, count: 4, plane: [1.0, 0.0, 0.0, 0.0] },
            WmoPortalInfo { start_vertex: 0, count: 4, plane: [1.0, 0.0, 0.0, 0.0] },
        ];
        // p0: g0 <-> g1 (a room to the SHELL — the daylight seed's population, not this one).
        // p1: g0 <-> g2 (room to room — the one bleed seed).
        // p2: named by g2 alone (a portal to the outside — not a doorway between two rooms).
        let refs = [
            WmoPortalRef { portal: 0, group: 1, side: 1 },
            WmoPortalRef { portal: 1, group: 2, side: 1 },
            WmoPortalRef { portal: 0, group: 0, side: -1 },
            WmoPortalRef { portal: 1, group: 0, side: -1 },
            WmoPortalRef { portal: 2, group: 2, side: 1 },
        ];
        let slices = [(0, 2), (2, 1), (3, 2)];
        let portals = PortalGraph {
            vertices: &vertices,
            infos: &infos,
            refs: &refs,
            slices: &slices,
        };
        let seeds = bleed_seeds(&groups, portals);
        assert_eq!(seeds.len(), 1, "{seeds:#?}");
        assert_eq!(seeds[0].portal, 1);
        let mut sides = seeds[0].sides;
        sides.sort_unstable();
        assert_eq!(sides, [0, 2]);
        // …and it stands IN the opening: the quad lies on x = 0 and is NOT nudged off it (that is
        // the whole aperture argument — a doorway has no side, so there is no side to step onto).
        assert!(seeds[0].pos[0].abs() < 1e-4, "pos {:?}", seeds[0].pos);
        // The z clamp is `seed_point`'s: the quad's centre is 1 yd up, under `SEED_MAX_HZ`.
        assert!((seeds[0].pos[2] - 1.0).abs() < 1e-4, "pos {:?}", seeds[0].pos);
        // Both rooms are MOLR-seeded, so both are claimed HARD — a doorway lights either side.
        let claims = seeds[0].claims(&groups, portals);
        for g in [0u16, 2] {
            let c = claims.iter().find(|c| c.group == g).expect("claim");
            assert_eq!(c.fade_radius, 0.0, "g{g} claim should be hard: {c:?}");
        }
    }

    /// MONKEY (portal bleed): the calibration really does deliver `BLEED_K x lit` at the portal
    /// plane — fed back through the shader's own profile, which is the claim the whole rule rests on.
    #[test]
    fn a_doorway_transmits_bleed_k_at_its_own_plane() {
        let knobs = DynamicInteriors::default();
        let k_fill = knobs.fill * knobs.interior_gain;
        let r_eff = super::super::interior_reach(daylight_authored_reach(5.91), knobs.atten_scale);
        // A warm candle-coloured irradiance, the Goldshire hall's own hue.
        let lit = [0.0240f32, 0.0240, 0.0158];
        let (hue, i) = bleed_intensity(lit, r_eff, k_fill);
        assert!(i > 0.0 && i <= 1.0, "I {i}");
        // The fixture's own contribution at NUDGE_YD, evaluated exactly as `room_irradiance` would.
        let delivered = luminance(hue)
            * i
            * (core_atten(NUDGE_YD) * interior_window(NUDGE_YD, r_eff, INTERIOR_DIRECT_POW)
                + k_fill * interior_window(NUDGE_YD, INTERIOR_FILL_SPAN * r_eff, INTERIOR_FILL_POW));
        assert!(
            (delivered - BLEED_K * luminance(lit)).abs() < 1e-6,
            "delivered {delivered} vs {}",
            BLEED_K * luminance(lit)
        );
        // …and it can never out-shine what it carries: `BLEED_K < 1` by construction.
        assert!(delivered < luminance(lit));
        // The hue is the lit side's chroma renormalised to peak 1, so the packed colour is `hue * I`.
        assert!((hue[0] - 1.0).abs() < 1e-5 && hue[2] < hue[1]);
        // A dark neighbour is no fixture at all — the packer must not hold a slot for it.
        assert_eq!(bleed_intensity([0.0; 3], r_eff, k_fill).1, 0.0);
        // Brighter in, brighter out, monotone.
        let (_, brighter) = bleed_intensity([0.05, 0.05, 0.033], r_eff, k_fill);
        assert!(brighter > i);
    }

    /// MONKEY (portal bleed): the two lanes share ONE eight-slot budget, ranked by area, and a tie
    /// goes to daylight (the inn's p0 porch door and p1 vestibule doorway are exactly that tie).
    #[test]
    fn the_shared_budget_ranks_both_lanes_by_area() {
        let groups = [
            group(true, [-50.0, -50.0, 0.0], [0.0, 50.0, 20.0]),
            group(true, [0.0, -50.0, 0.0], [50.0, 50.0, 20.0]),
        ];
        // Six room-to-room doorways on the x = 0 wall, of strictly increasing area -- all under
        // `SPLIT_PORTAL_MIN_AREA` (28 yd^2), which the seed rule refuses as a cut room rather than
        // a door: `2w x w` for w in 1..=6 is 2..24 yd^2.
        let quads: Vec<[[f32; 3]; 4]> = (1..=6)
            .map(|i| {
                let w = i as f32;
                let y = i as f32 * 10.0 - 45.0;
                [
                    [0.0, y - w, 0.0],
                    [0.0, y + w, 0.0],
                    [0.0, y + w, w],
                    [0.0, y - w, w],
                ]
            })
            .collect();
        let vertices: Vec<[f32; 3]> = quads.iter().flatten().copied().collect();
        let infos: Vec<WmoPortalInfo> = (0..6)
            .map(|i| WmoPortalInfo {
                start_vertex: i * 4,
                count: 4,
                plane: [1.0, 0.0, 0.0, 0.0],
            })
            .collect();
        let mut refs = Vec::new();
        for i in 0..6u16 {
            refs.push(WmoPortalRef { portal: i, group: 1, side: 1 });
        }
        for i in 0..6u16 {
            refs.push(WmoPortalRef { portal: i, group: 0, side: -1 });
        }
        let slices = [(0, 6), (6, 6)];
        let portals = PortalGraph {
            vertices: &vertices,
            infos: &infos,
            refs: &refs,
            slices: &slices,
        };
        // …plus five WINDOWS (EXT-class batches of g0), all smaller than the widest doorways.
        let windows: Vec<[[f32; 3]; 4]> = (1..=5)
            .map(|i| {
                let y = i as f32 * 8.0 - 40.0;
                [
                    [-50.0, y - 1.0, 1.0],
                    [-50.0, y + 1.0, 1.0],
                    [-49.9, y + 1.0, 3.0],
                    [-49.9, y - 1.0, 3.0],
                ]
            })
            .collect();
        let (day, bleed) = placement_openings(
            &groups,
            portals,
            windows.iter().map(|q| (0u16, true, &q[..])),
        );
        assert_eq!(
            day.len() + bleed.len(),
            MAX_DAYLIGHT_PER_PLACEMENT,
            "day {day:?} bleed {bleed:?}"
        );
        // The widest doorways (2w x w = 2, 8, 18, 32*, 50*, 72* yd^2 -- starred ones refused as
        // SPLIT cuts) outrank the 2 x 2 yd windows, and the narrowest doorway loses to them.
        assert!(bleed.iter().all(|b| b.area < 28.0), "a split cut got seeded: {bleed:?}");
        assert!(bleed.len() >= 2, "{bleed:?}");
        for w in bleed.windows(2) {
            assert!(w[0].area >= w[1].area);
        }
        assert!(bleed.iter().all(|b| b.sides.contains(&0) && b.sides.contains(&1)));
    }

    /// The sun curve: dark at and below the horizon, monotone up to full — and the calibration
    /// really does solve the equation it claims to.
    #[test]
    fn intensity_tracks_the_sun() {
        let knobs = DynamicInteriors {
            // The owner's live settings (`benilla-config/config.toml`), not the defaults.
            ambient: 0.02,
            exposure: 4.0,
            ..DynamicInteriors::default()
        };
        let mut light = WowLighting {
            ambient: [0.25, 0.26, 0.30],
            diffuse: [0.55, 0.53, 0.47],
            ..WowLighting::default()
        };
        // Straight overhead: the floor takes the whole diffuse band.
        light.sun_dir = Vec3::NEG_Y;
        light.celestial_dir = Vec3::Y;
        let (hue, t_day, w_day) = daylight_target(&light);
        assert_eq!(w_day, 1.0);
        assert!(t_day > luminance(light.ambient));
        // …and below the horizon the fixture is off however bright the bands read.
        light.celestial_dir = Vec3::new(0.0, -0.3, 1.0);
        assert_eq!(daylight_target(&light).2, 0.0);
        // Monotone through the dawn knee.
        let mut prev = -1.0;
        for i in 0..=20 {
            light.celestial_dir = Vec3::new(0.0, -0.05 + i as f32 * 0.02, 1.0);
            let w = daylight_target(&light).2;
            assert!(w >= prev, "sun weight went backwards at step {i}");
            prev = w;
        }
        // The calibration: a dimmer target needs less intensity, and a target the ambient floor
        // already exceeds needs none.
        let a = direct_profile(1.21, 0.909, 11.2);
        let w_fill = interior_window(1.21, INTERIOR_FILL_SPAN * 11.2, INTERIOR_FILL_POW);
        let hue_lum = luminance(hue);
        let bright = daylight_intensity(t_day, hue_lum, a, w_fill, &knobs);
        let dim = daylight_intensity(t_day * 0.5, hue_lum, a, w_fill, &knobs);
        assert!(bright > dim && dim > 0.0, "bright {bright} dim {dim}");
        assert!(bright <= 1.0);
        assert_eq!(daylight_intensity(0.0, hue_lum, a, w_fill, &knobs), 0.0);
        // Feed the solved intensity back through the shader's own arithmetic: the rolloff must land
        // on the reference target, which is the whole claim the seam closure rests on.
        let room = hue_lum * bright * (a + knobs.fill * w_fill) + knobs.ambient;
        let illum = 1.0 - (-room * knobs.exposure).exp();
        assert!((illum - t_day).abs() < 1e-3, "illum {illum} vs target {t_day}");
    }
}
