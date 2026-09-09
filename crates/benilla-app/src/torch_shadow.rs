//! MONKEY (torch shadows, Phase 1) — the interior point-light shadow lane, a plug-in on
//! [`super::shadow_core`], independent of [`super::world_shadow`]/[`super::character_shadow`].
//!
//! **Why this is not Bevy's point shadows.** benilla's world camera sets
//! `bevy::light::cluster::ClusterConfig::None`, which disables Bevy's point/spot clustering AND
//! starves the point-light shadow-map prep — so `fetch_point_shadow` / `clusterable_objects` are
//! DEAD here. The old design (promote the nearest fixtures to shadow-casting `PointLight` proxies and
//! sample their cube maps) could never fire, so it is gone. In its place: this lane picks the
//! interior fixtures that actually LIGHT the player, computes a down-looking reverse-Z `view_proj` for each, builds the
//! retained interior caster geometry, and publishes it all as [`TorchShadowViews`] (an
//! [`ExtractResource`]). The render world then renders a depth map per fixture
//! (`benilla_world::static_gx::torch_depth`) and `static_gx.wgsl`'s interior surface lane samples
//! it so a pillar throws a radial shadow.
//!
//! **MONKEY (torch caster selection).** Which fixtures get maps used to be "the four nearest the
//! camera", recomputed and hard-switched every frame. That is the shadow POP: in Northshire Abbey
//! (42 candelabra, ~5 yd apart) the nearest-four set churns every few steps, and every fixture
//! outside it lit with no shadow at all, so a fixture's shadow appeared as you walked up to it and
//! vanished as you moved on. The selection is now (a) ranked by the fixture's own DIRECT
//! CONTRIBUTION at the player using the shaders' profile ([`fixture_score`]), (b) hysteretic
//! (a challenger must beat an incumbent by [`TORCH_SWAP_RATIO`], one swap per
//! [`TORCH_SWAP_COOLDOWN`]), and (c) CROSS-FADED — every slot carries a weight the receivers
//! `mix` with, so a promotion ramps in and a demotion ramps out before the slot is reused. The
//! budget is [`MAX_TORCH_CASTERS`] slots, of which `interiorShadowCasters` (default 12) are filled.
//!
//! Gated on `interiorLight` + `interiorShadows` ([`VideoConfig`]); declares demand so the shared rig
//! stays up. Remove this module + its plugin + the `interiorShadows` cvar and the other lanes are
//! untouched. MONKEY (static torch cache): static_gx and wow_model share the 6416-byte table.
//! Camera distance must be <= ShadowDistance, AND player distance <= 4R+3. Contribution ranking,
//! 1.5x hysteresis and fades choose up to twelve of sixteen resident slots; this is still a finite
//! budget, not a promise that a room with forty eligible candles can give all forty cube maps.
//! Rigid furniture joins the cached static geometry. Only the nearest interiorShadowDynamic
//! promoted fixtures (default four, measured from the player) overlay creatures/animated parts.

use std::hash::{Hash, Hasher};

use bevy::math::{DMat4, DVec3, DVec4};
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResourcePlugin;

use bevy::pbr::MeshMaterial3d;

use benilla_assets::materials::WowModelMaterial;
use benilla_assets::coords::wow_to_bevy;
use benilla_formats::ModelBlend;
use benilla_world::billboard::BillboardCard;
use benilla_world::interact::PickMesh;
use benilla_world::lighting::{
    interior_reach, m2_light_reach, DynamicInteriors, FireLightGain, LightLane, LightLitRooms,
    LightReach, LightRooms, ShadowDistance, SyntheticFireLight,
};
// MONKEY (carried light stability): the settle verdict a carried light earns by standing still.
use crate::entities::CarriedLightMotion;
use benilla_world::model_render::{ModelKind, ModelPart, ShadowOccluder};
use benilla_world::rig_palette::{RigPalettes, RigPart, RigSkin};
use benilla_world::static_gx::{StaticGx, TorchShadowViews};
use benilla_world::view::WorldCamera;

use crate::char_select::ClientState;
use crate::net::SelfPlayer;
use crate::shadow_core::{
    empty_shadow_mesh, restore_mesh_buffers, take_mesh_buffers,
    ShadowDemand, ShadowFrame, ShadowSet,
};
use crate::video::VideoConfig;

/// How many interior fixtures can hold a depth map at once — the ALLOCATION. Matches
/// `torch_depth::MAX_TORCH_MAPS`; the number actually filled is the live `interiorShadowCasters`
/// cvar (default 12), so raising the cap does not by itself cost a frame.
const MAX_TORCH_CASTERS: usize = 16;
/// MONKEY (torch caster reach): promote out to FOUR times the fixture's effective reach. Its
/// pool still needs shadows when viewed across the room, long before it lights the player's body.
const TORCH_ELIGIBLE_REACH_MULT: f32 = 4.0;
/// MONKEY (static torch cache): minimum coarse CAMERA search; the live search is
/// max(TORCH_SEARCH_RADIUS, ShadowDistance), so it never truncates the shadow-distance setting.
const TORCH_SEARCH_RADIUS: f32 = TORCH_ELIGIBLE_REACH_MULT * TORCH_RANGE + TORCH_ELIGIBLE_SLACK;
/// MONKEY (torch caster reach): extra eligibility slack beyond the expanded 4R window (yd).
const TORCH_ELIGIBLE_SLACK: f32 = 3.0;
/// MONKEY (torch caster selection): a challenger must beat the weakest incumbent by this FACTOR
/// before it may take its slot. Pure "best N" thrashes: two candles a step apart trade the lead
/// every few frames as you walk between them, and each trade used to be an instant on/off. 1.5 is
/// wide enough that a real change of room wins immediately and a step sideways never does.
const TORCH_SWAP_RATIO: f32 = 1.5;
/// MONKEY (torch caster selection): and at most ONE contested swap this often (s). The ratio alone
/// still permits a cascade — six slots could each lose their contest in the same frame, which is
/// six shadows changing at once no matter how smooth each fade is.
const TORCH_SWAP_COOLDOWN: f64 = 0.25;
/// MONKEY (torch caster selection): the cross-fade rate (weight per second) — ~1/3 s to ramp a
/// shadow fully in or out. Slow enough to read as "the light reaches you now", fast enough that
/// the geometry is not visibly shadow-less while you stand in front of it.
const TORCH_FADE_RATE: f32 = 3.0;
/// MONKEY (torch caster selection): the interior direct-light profile, MIRRORED from the
/// `interior_room_light` block that both `static_gx.wgsl` and `wow_model.wgsl` carry ("MONKEY (soft
/// falloff)" — keep the three in sync). Ranking by the shader's OWN contribution is the whole point
/// of this change: distance ranked a bright forge behind you below a dead candle at your feet.
const INTERIOR_CORE_FRAC: f32 = 0.26;
const INTERIOR_CORE_GAIN: f32 = 1.5;
const INTERIOR_DIRECT_POW: f32 = 10.0;
/// MONKEY (torch caster reach): unchanged fixture-centred far plane. `interior_reach` caps R at
/// 48 yd; seeing that pool from 4R away needs earlier promotion, not a longer projection.
const TORCH_RANGE: f32 = 48.0;
/// MONKEY (Phase 5): six cube faces per fixture. Keep in sync with `torch_depth::CUBE_FACES`.
const CUBE_FACES: usize = 6;
/// MONKEY (static torch cache): how many resident slots have their source FINGERPRINT recomputed
/// per frame. Hashing one fixture's 48 yd source set walks every resident `GxItem` and every model
/// part; doing that for all twelve residents every frame hands the CPU back the cost the GPU cache
/// just saved (~12 full scene scans per frame, and it is a per-frame cost even in a room where
/// nothing ever changes). Four slots, cursor-rotated, still notices any streaming change within
/// three frames — under a tenth of a second, and the map it then rebuilds is the same one.
const TORCH_SCAN_PER_FRAME: usize = 4;
/// MONKEY (static torch cache): how far (yd²) a fixture may drift from the position its cached map
/// was rendered at before that map is WITHDRAWN. A cached map that is merely STALE (the room
/// streamed a chair in since it was baked) keeps being published — the shadow is a few frames out
/// of date, which is invisible. A map whose fixture MOVED is not stale but WRONG: every projection
/// in it aims from somewhere else, and the published `view_projs` (rebuilt from the live position
/// each frame) no longer address it. Withdrawing a merely-stale map instead was a hard on/off
/// blink: an unready slot's `positions[i].w` is forced to 0 with no fade — precisely the pop the
/// whole cross-fade exists to prevent — for however many frames the rebuild budget takes to reach it.
const TORCH_STALE_DRIFT_SQ: f32 = 0.01;
/// Each cube face's FOV: 90° plus a hair, so a direction exactly on a face border still lands
/// inside the face the shader picks for it (the projector returns "lit" outside its frustum).
const TORCH_FACE_FOV: f32 = std::f32::consts::FRAC_PI_2 + 0.02;

pub(crate) struct TorchShadowPlugin;

impl Plugin for TorchShadowPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TorchLane>()
            .init_resource::<TorchShadowViews>()
            // The render world reads the published fixtures/caster through this.
            .add_plugins(ExtractResourcePlugin::<TorchShadowViews>::default())
            .add_systems(Last, update_torch_shadows.in_set(ShadowSet::Lanes));
    }
}

/// MONKEY (torch caster selection): one OCCUPIED caster slot — a fixture that holds (or is
/// releasing) six depth-map layers. Slots are retained across frames, which is what makes
/// hysteresis and cross-fading possible at all: the old lane recomputed "the nearest four" from
/// scratch every frame and so had no notion of an incumbent to keep.
struct TorchSlot {
    // MONKEY (static torch cache): physical residency survives Vec compaction and rank changes.
    cache_slot: usize,
    mesh: Option<Handle<Mesh>>,
    geometry_key: Option<u64>,
    /// MONKEY (static torch cache): the fixture position `mesh` was actually gathered AT, which is
    /// not necessarily `pos` — the cursor scans only [`TORCH_SCAN_PER_FRAME`] slots a frame, so a
    /// slot can be published several frames after it last rebuilt. `None` until the first build.
    /// See [`TORCH_STALE_DRIFT_SQ`] for why staleness publishes and drift does not.
    built_at: Option<Vec3>,
    /// The fixture entity — the identity the slot is held BY. Matching on entity rather than on
    /// position is what lets a CARRIED torch (which moves every frame) keep its slot.
    fixture: Entity,
    /// Its world position this frame (absolute Bevy), the shader's correlation key.
    pos: Vec3,
    /// Its contribution score at the player this frame (see [`fixture_score`]).
    score: f32,
    /// The cross-fade weight, published as `positions[i].w` and applied as `mix(1, shadow, w)` by
    /// both receivers.
    w: f32,
    /// Set once the slot has lost its contest (or its fixture went out of reach, or the budget
    /// shrank). It keeps rendering while `w` ramps down, and the slot is freed for a challenger
    /// only when `w` reaches 0 — "fade, not switch".
    evicting: bool,
}

/// MONKEY (static torch cache): stable physical slots and per-fixture source meshes.
/// Camera drift never rebuilds static geometry; only a source-set/fixture change does.
#[derive(Resource, Default)]
struct TorchLane {
    // MONKEY (static torch cache): only moving casters are rebuilt every frame. Each slot owns
    // its own static geometry handle; a replacement gets a NEW asset id, so the render world
    // cannot certify an old GPU mesh as the newly streamed geometry while uploads catch up.
    entity_mesh: Option<Handle<Mesh>>,
    rebuild_cursor: usize,
    /// MONKEY (torch caster selection): the occupied slots, at most `interiorShadowCasters` of them
    /// (plus any still fading out after the budget shrank). Published in this order.
    slots: Vec<TorchSlot>,
    /// When the last CONTESTED swap was started (seconds of `Time::elapsed`), for the cooldown.
    last_swap: f64,
}

/// MONKEY (torch caster selection): the interior direct term this fixture delivers at distance `d`
/// — the shaders' own profile (core inverse-square with a soft core, times the C1 window), times
/// the fixture's luminous intensity. This is the ranking key, and it is the answer to the bug: at
/// Northshire's candelabra spacing "nearest 4" and "the 4 that light you" are different sets, and
/// only the second one is stable as you walk, because a fixture's contribution changes smoothly
/// with distance while its RANK by distance changes in steps.
///
/// MONKEY (torch caster reach): retain the intensity and soft inverse-square core at the player's
/// distance, but stretch the selection window to 4R (including eligibility slack). This is a
/// ranking extension, not a lighting change: at 3R the score is small, positive and still falling.
/// No window floor flattens distant candidates; the window reaches zero at the eligibility edge.
fn fixture_score(intensity: f32, d: f32, r: f32) -> f32 {
    let r0 = (INTERIOR_CORE_FRAC * r).max(1e-3);
    let atten = INTERIOR_CORE_GAIN / (1.0 + (d / r0) * (d / r0));
    let extent = TORCH_ELIGIBLE_REACH_MULT * r + TORCH_ELIGIBLE_SLACK;
    let w = (1.0 - (d / extent.max(1e-4)).clamp(0.0, 1.0).powf(INTERIOR_DIRECT_POW)).clamp(0.0, 1.0);
    intensity * atten * w * w
}

/// MONKEY (review fixes): rank only light the interior receivers actually sample. Match the
/// packer's unresolved-lane fallback AND its synthetic-only gain; a disabled flame or a resolved
/// exterior fixture must score zero before it can evict a useful caster. Reach still uses the
/// authored intensity, so dimming a flame changes its contribution without moving its pool.
fn candidate_score(
    intensity: f32, d: f32, r: f32, lane: Option<&LightLane>, has_rooms: bool,
    synthetic: bool, fire_gain: f32,
) -> f32 {
    if !lane.map_or(has_rooms, |l| l.interior) {
        return 0.0;
    }
    fixture_score(intensity, d, r) * if synthetic { fire_gain.max(0.0) } else { 1.0 }
}

/// MONKEY (static torch cache): camera distance uses the world-shadow resource's Bevy-yard
/// yardstick. The player-centred 4R+3 window still limits/ranks useful pools; BOTH tests apply.
/// The coarse camera search grows with shadowDistance so it can never silently shorten that dial.
fn fixture_eligible(player_d: f32, camera_d: f32, r: f32, distance: f32) -> bool {
    camera_d <= TORCH_SEARCH_RADIUS.max(distance)
        && camera_d <= distance
        && player_d <= TORCH_ELIGIBLE_REACH_MULT * r + TORCH_ELIGIBLE_SLACK
}

/// MONKEY (static torch cache): may a slot publish its cached map this frame? Yes while the
/// fixture still stands where the map was baked from (see [`TORCH_STALE_DRIFT_SQ`]) — staleness in
/// the SURROUNDINGS is invisible for the few frames a rebuild takes, whereas withdrawing the map
/// zeroes the slot's weight with no fade. No when the fixture has moved: the six projections are
/// republished from the live position every frame and would no longer address that depth.
fn map_publishable(built_at: Option<Vec3>, pos: Vec3) -> bool {
    built_at.is_some_and(|p| p.distance_squared(pos) <= TORCH_STALE_DRIFT_SQ)
}

/// MONKEY (static torch cache): the nearest promoted fixtures to the player receive entities.
/// Stable physical-slot tie breaking keeps coincident/equidistant candles deterministic; this
/// subset is independent of contribution ranking and never moves the resident static maps.
fn dynamic_set(fixtures: &[(usize, Vec3)], anchor: Vec3, budget: usize) -> u32 {
    let mut nearest = fixtures.to_vec();
    nearest.sort_by(|a, b| a.1.distance_squared(anchor).total_cmp(&b.1.distance_squared(anchor))
        .then(a.0.cmp(&b.0)));
    nearest.iter().take(budget.min(MAX_TORCH_CASTERS)).fold(0, |mask, (i, _)| mask | (1 << i))
}

/// MONKEY (torch caster selection): the entity-caster query trio, bundled as ONE `SystemParam`.
/// Not tidiness — [`update_torch_shadows`] sat at 15 of Bevy's 16 system params before this change
/// added the player anchor and the interior knobs, and three separate members would not compile.
#[derive(bevy::ecs::system::SystemParam)]
struct EntityCasters<'w, 's> {
    /// The SAME query the world lane passes to `collect_entity_geometry` (written inline by that
    /// contract so a lane can pass its own).
    parts: Query<
        'w,
        's,
        (
            Entity,
            &'static PickMesh,
            &'static ModelPart,
            Option<&'static GlobalTransform>,
            Option<&'static RigPart>,
            &'static ShadowOccluder,
            Option<&'static MeshMaterial3d<WowModelMaterial>>,
        ),
        Without<BillboardCard>,
    >,
    rigs: Query<'w, 's, &'static RigSkin>,
    palettes: Res<'w, RigPalettes>,
    distance: Res<'w, ShadowDistance>,
}

/// The lane is active when the dynamic-interior lane is on AND its shadow toggle is on.
fn interior_shadows_on(video: &VideoConfig) -> bool {
    video.interior_light && video.interior_shadows
}

/// A reverse-Z perspective (near → 1, far → 0; wgpu clip, RH, looking down −Z) — matches the whole
/// engine's reverse-Z convention and the torch depth pipeline's `GreaterEqual` compare. Built in f64.
fn reverse_z_perspective(fov_y: f64, aspect: f64, near: f64, far: f64) -> DMat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    // Column-major columns.
    DMat4::from_cols(
        DVec4::new(f / aspect, 0.0, 0.0, 0.0),
        DVec4::new(0.0, f, 0.0, 0.0),
        DVec4::new(0.0, 0.0, near / (far - near), -1.0),
        DVec4::new(0.0, 0.0, (far * near) / (far - near), 0.0),
    )
}

/// MONKEY (Phase 5): the six cube-face `view_proj`s of a fixture — a 90°(+ε) reverse-Z perspective
/// looking down each of ±X, ±Y, ±Z — so the maps cover EVERY direction around it. This replaced the
/// single aimed cone: any single cone has an edge, and both the vertical (walk up to the forge) and
/// lateral (stand beside the player) shadow collapses were that edge being crossed. Face order is
/// the contract with `static_gx.wgsl`'s `torch_face`: 0 +X, 1 −X, 2 +Y, 3 −Y, 4 +Z, 5 −Z. Computed
/// in f64 because fixtures sit at absolute Bevy world coords (~9,300 out) where f32 view-matrix
/// arithmetic loses precision. Nothing here depends on the camera or the player any more.
fn cube_view_projs(fixture: Vec3) -> [Mat4; CUBE_FACES] {
    let eye = DVec3::new(fixture.x as f64, fixture.y as f64, fixture.z as f64);
    let proj = reverse_z_perspective(TORCH_FACE_FOV as f64, 1.0, 0.1, TORCH_RANGE as f64);
    let faces: [(DVec3, DVec3); CUBE_FACES] = [
        (DVec3::X, DVec3::Y),
        (DVec3::NEG_X, DVec3::Y),
        (DVec3::Y, DVec3::NEG_Z),
        (DVec3::NEG_Y, DVec3::NEG_Z),
        (DVec3::Z, DVec3::Y),
        (DVec3::NEG_Z, DVec3::Y),
    ];
    faces.map(|(dir, up)| {
        let vp = proj * DMat4::look_to_rh(eye, dir, up);
        Mat4::from_cols_array(&vp.to_cols_array().map(|v| v as f32))
    })
}

#[allow(clippy::too_many_arguments)]
fn update_torch_shadows(
    video: Res<VideoConfig>,
    state: Res<State<ClientState>>,
    mut demand: ResMut<ShadowDemand>,
    frame: Res<ShadowFrame>,
    mut lane: ResMut<TorchLane>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut views: ResMut<TorchShadowViews>,
    cameras: Query<&GlobalTransform, With<WorldCamera>>,
    // MONKEY (torch caster selection): the RANKING ANCHOR. Shadows are for the player's own
    // surroundings — a free-look camera swung across the room must not re-pick the casters — so
    // scoring is done at the body, and only falls back to the camera when there is none (the glue
    // scene, a capture fixture, the frame before the player streams in).
    player: Query<&GlobalTransform, With<SelfPlayer>>,
    // Interior WMO fixtures: a `PointLight` carrying a room claim. MONKEY (torch caster selection):
    // the intensity/colour and the authored reach ride along now — the score is the light's own
    // contribution, so the lane must read exactly what the packer reads.
    // MONKEY (review fixes): containment is membership evidence even with no MOLR; keep the
    // visibility membership separate so the missing-lane fallback stays identical to the packer.
    // MONKEY (carried light stability): `Has<ChildOf>` + `Option<&CarriedLightMotion>` ride the
    // tuple so the filter below can refuse a MOVING carried light a slot. Both are cheap table
    // reads on a query that already walks these entities.
    torches: Query<
        (Entity, &GlobalTransform, &PointLight, Option<&LightReach>, Option<&LightLane>,
         Has<LightRooms>, Has<SyntheticFireLight>, Has<ChildOf>,
         Option<&crate::entities::CarriedLightMotion>),
        Or<(With<LightRooms>, With<LightLitRooms>)>,
    >,
    // MONKEY (torch caster selection): `atten_scale` — a fixture's EFFECTIVE radius is its authored
    // end times this live cvar, and the score's window is a fraction of that radius. Reading the
    // same resource the packer folds into the shader's `.w` keeps ranking and rendering agreed.
    interiors: Res<DynamicInteriors>,
    fire_gain: Res<FireLightGain>,
    gx: Option<Res<StaticGx>>,
    // MONKEY (Phase 3B): the entity caster inputs, bundled (see [`EntityCasters`]).
    ents: EntityCasters,
    time: Res<Time>,
    mut last_trace: Local<f64>,
) {
    // Declare demand so the shared rig stays up while on.
    let on = interior_shadows_on(&video) && *state.get() == ClientState::InWorld;
    demand.0 = demand.0 || on;

    if !(frame.active && on) {
        teardown(&mut lane, &mut meshes, &mut views);
        return;
    }
    let Some(cam) = cameras.iter().next().map(GlobalTransform::translation) else {
        return;
    };

    // (1) MONKEY (torch caster selection): who casts. Three rules, in this order.
    //
    //   RANK BY CONTRIBUTION, NOT DISTANCE. The old lane took the four fixtures nearest the camera
    //   and hard-switched the set every frame. In a candle-dense room that set churns every few
    //   steps — Northshire Abbey authors 42 candelabra about 5 yd apart — and every fixture NOT
    //   in it lit with no shadow at all, so shadows visibly popped in as you approached a candle
    //   and out again as you passed it. Scoring by the light's own direct term at the PLAYER makes
    //   the ranking a smooth function of position instead of a step function, and makes it right:
    //   a bright forge across the room outranks a guttering candle at your feet, which is what the
    //   eye expects of a shadow.
    //
    //   HYSTERESIS. A smooth score still crosses. An incumbent keeps its slot unless a challenger
    //   beats it by TORCH_SWAP_RATIO, and at most one contested swap starts per
    //   TORCH_SWAP_COOLDOWN — so walking the candle corridor does not trade slots every step.
    //
    //   FADE, NOT SWITCH. A promoted slot ramps its weight 0 to 1; a losing one ramps 1 to 0 and is
    //   only REPLACED at 0 (it keeps rendering while it fades). The receivers apply the weight as
    //   `mix(1, shadow, w)`, so a shadow dissolves rather than blinking.
    let anchor = player
        .iter()
        .next()
        .map(GlobalTransform::translation)
        .unwrap_or(cam);
    let want = (video.interior_shadow_casters as usize).clamp(1, MAX_TORCH_CASTERS);
    let dt = time.delta_secs();
    let now = time.elapsed_secs_f64();

    // (1a) MONKEY (torch caster reach): each pool is eligible from 4R + slack away. Rank with the
    // same soft core/intensity so nearby contributors still win over pools seen across the room.
    let mut cands: Vec<(Entity, Vec3, f32)> = torches
        .iter()
        .filter_map(|(e, gt, pl, reach, light_lane, has_rooms, synthetic, carried, motion)| {
            // MONKEY (carried light stability): a CARRIED light (`ChildOf` — a pet's hand flame,
            // an NPC's torch, a transport's deck brazier) casts only while it is STANDING STILL.
            //
            // This whole lane caches a depth cube per fixture and keeps it only while the fixture
            // stays within `TORCH_STALE_DRIFT_SQ` of where it was baked ([`map_publishable`]); a
            // slot whose map is withdrawn has its cross-fade weight forced to 0 with NO ramp
            // (`static_gx::torch_depth`'s `TorchTable::pack`). A moving fixture therefore toggled
            // a full-strength shadow at frame rate — the reported epileptic pool around a
            // summoned imp — and consumed the lane's global two-rebuilds-a-frame budget re-baking
            // a mesh that was stale again on arrival, so the STATIC candles beside it flickered too.
            //
            // Gated on MOTION rather than excluded outright, because the stationary members of the
            // same `ChildOf` family are exactly what the entity half of this lane was built for: a
            // placed brazier GameObject, a campfire. They settle in well under a second and cast
            // like a MOLT fixture. And because this is a CANDIDACY verdict, an incumbent that
            // starts walking drops out of `cands` and is cross-faded out by (1b)/(1c) rather than
            // cut — "fade, not switch" holds for the eviction as much as for a swap.
            if carried && !motion.is_some_and(CarriedLightMotion::settled) {
                return None;
            }
            let p = gt.translation();
            let d = p.distance(anchor);
            if p.distance(cam) > TORCH_SEARCH_RADIUS.max(ents.distance.0) {
                return None;
            }
            // The packer's own recovery of authored colour x intensity (`spawn_point_light`
            // premultiplied 4*pi), and the packer's own reach: the authored MOLT end where there is
            // one, the intensity bucket otherwise, times the live `interiorAttenScale`.
            let base = pl.intensity / (4.0 * std::f32::consts::PI);
            let authored = reach
                .map(|r| r.0)
                .filter(|r| *r > 0.5)
                .unwrap_or_else(|| m2_light_reach(base));
            let r = interior_reach(authored, interiors.atten_scale);
            if !fixture_eligible(d, p.distance(cam), r, ents.distance.0) {
                return None;
            }
            let c = pl.color.to_linear();
            // Rec.709 luminance of the committed colour: ONE scalar for "how much light is this",
            // so a dim blue magic brazier cannot outrank a bright hearth on channel count.
            let lum = (0.2126 * c.red + 0.7152 * c.green + 0.0722 * c.blue).max(0.0) * base.max(0.0);
            let score = candidate_score(lum, d, r, light_lane, has_rooms, synthetic, fire_gain.0);
            (score > 0.0).then_some((e, p, score))
        })
        .collect();
    cands.sort_by(|a, b| b.2.total_cmp(&a.2));

    // (1b) Refresh the incumbents against this frame's candidates. A fixture that dropped out of
    // the set (it moved away, despawned, its room streamed out) starts fading rather than
    // vanishing. A slot past the live budget does the same, so lowering `interiorShadowCasters`
    // fades its extra shadows out instead of cutting them.
    for (i, slot) in lane.slots.iter_mut().enumerate() {
        match cands.iter().find(|(e, _, _)| *e == slot.fixture) {
            Some((_, p, sc)) => {
                slot.pos = *p;
                slot.score = *sc;
            }
            None => {
                slot.score = 0.0;
                slot.evicting = true;
            }
        }
        if i >= want {
            slot.evicting = true;
        }
    }

    // (1c) Ramp the weights, then release the slots that finished fading out.
    let step = TORCH_FADE_RATE * dt;
    for slot in lane.slots.iter_mut() {
        slot.w = if slot.evicting {
            (slot.w - step).max(0.0)
        } else {
            (slot.w + step).min(1.0)
        };
    }
    lane.slots.retain(|s| {
        let release = s.evicting && s.w <= 0.0;
        if release {
            if let Some(mesh) = &s.mesh { meshes.remove(mesh.id()); }
        }
        !release
    });

    // (1d) Fill free slots from the best unassigned candidates. A newly filled slot starts at
    // w = 0 and fades IN. Note this can only run once a fading slot has fully drained, which is
    // what stops a churn from being visible even when the SELECTION churns.
    // MONKEY (static torch cache): at most two new residents per frame, including startup.
    let mut promotions = 0;
    while lane.slots.len() < want && promotions < 2 {
        let Some(&(e, p, sc)) = cands
            .iter()
            .find(|(e, _, _)| !lane.slots.iter().any(|s| s.fixture == *e))
        else {
            break;
        };
        let cache_slot = (0..MAX_TORCH_CASTERS)
            .find(|i| !lane.slots.iter().any(|s| s.cache_slot == *i)).unwrap();
        promotions += 1;
        lane.slots.push(TorchSlot {
            cache_slot,
            mesh: None,
            geometry_key: None,
            built_at: None,
            fixture: e,
            pos: p,
            score: sc,
            w: 0.0,
            evicting: false,
        });
    }

    // (1e) The contested swap: the best unassigned candidate takes the weakest incumbent's slot
    // only if it beats it by the ratio AND the cooldown has elapsed. Marking `evicting` rather
    // than replacing is the whole of "fade, not switch" — (1c) drains it and (1d) refills it a
    // fade later, by which time the challenger has been re-scored at the player's new position.
    if now - lane.last_swap >= TORCH_SWAP_COOLDOWN {
        let weakest = lane
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.evicting)
            .min_by(|a, b| a.1.score.total_cmp(&b.1.score))
            .map(|(i, s)| (i, s.score));
        let challenger = cands
            .iter()
            .find(|(e, _, _)| !lane.slots.iter().any(|s| s.fixture == *e))
            .map(|(_, _, sc)| *sc);
        if let (Some((idx, weak)), Some(best)) = (weakest, challenger) {
            if best > weak * TORCH_SWAP_RATIO {
                lane.slots[idx].evicting = true;
                lane.last_swap = now;
            }
        }
    }

    // The published set, in slot order: fixture, position, score (trace only) and fade weight,
    // capped at the table's allocation.
    let promoted: Vec<(usize, Entity, Vec3, f32, f32)> = lane
        .slots
        .iter()
        .take(MAX_TORCH_CASTERS)
        .map(|s| (s.cache_slot, s.fixture, s.pos, s.score, s.w))
        .collect();

    // (2) MONKEY (static torch cache): fingerprint resident SOURCE geometry per fixture, not
    // StaticGx's change tick (frame bookkeeping changes it continuously). This catches arrivals,
    // removals and replacements even with a stationary camera. Entity-resident rigid furniture
    // joins the static mesh; creatures and skinned/animated parts belong to the dynamic overlay.
    let fixture_positions: Vec<_> = promoted.iter().map(|(i, _, p, _, _)| (*i, *p)).collect();
    let dynamic = dynamic_set(&fixture_positions, anchor, video.interior_shadow_dynamic as usize);
    let mut rebuilt = 0;
    let mut static_rebuilt = 0u32;
    let mut published = TorchShadowViews {
        count: promoted.iter().map(|(i, ..)| *i as u32 + 1).max().unwrap_or(0),
        soft: video.interior_shadow_soft,
        dynamic_mask: dynamic,
        ..Default::default()
    };
    // MONKEY (static torch cache): rotate service order so two moving/carried fixtures cannot
    // consume the rebuild budget forever and starve a newly resident room's static maps. The
    // rotation bounds the SCAN as well as the rebuild (see TORCH_SCAN_PER_FRAME): fingerprinting a
    // fixture's surroundings is the one part of this lane that is O(resident scene), so doing it
    // for every slot every frame would be the new cost ceiling — and at most two of those slots
    // could act on the answer anyway.
    let slot_count = lane.slots.len();
    let start = lane.rebuild_cursor;
    let scan = slot_count.min(TORCH_SCAN_PER_FRAME);
    for offset in 0..scan {
        let index = (start + offset) % slot_count;
        let slot = &mut lane.slots[index];
        let i = slot.cache_slot;
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        slot.fixture.hash(&mut hash);
        slot.pos.to_array().map(f32::to_bits).hash(&mut hash);
        gx.as_ref().map(|gx| gx.torch_geometry_key(slot.pos, TORCH_RANGE)).hash(&mut hash);
        let mut entity_key = 0u64;
        for (entity, pick, part, global, rig, occluder, _) in &ents.parts {
            if !static_part(part, rig.is_some()) || !occluder.0 { continue; }
            let Some(global) = global else { continue };
            if global.translation().distance_squared(slot.pos) > TORCH_RANGE * TORCH_RANGE { continue; }
            // Arc identity catches geometry replacement, transform bits catch moved furniture.
            let mut part_hash = std::collections::hash_map::DefaultHasher::new();
            entity.hash(&mut part_hash);
            (std::sync::Arc::as_ptr(&pick.0) as usize).hash(&mut part_hash);
            global.to_matrix().to_cols_array().map(f32::to_bits).hash(&mut part_hash);
            entity_key = entity_key.wrapping_add(part_hash.finish());
        }
        entity_key.hash(&mut hash);
        let key = hash.finish();
        let dirty = slot.geometry_key != Some(key);
        if dirty && rebuilt < 2 {
            let mut positions = Vec::new();
            let mut indices = Vec::new();
            if let Some(gx) = &gx {
                gx.append_torch_triangles(slot.pos, TORCH_RANGE, &mut positions, &mut indices);
            }
            collect_torch_entities(&ents, true, &[(i, slot.pos)], &mut positions, &mut indices);
            let mut mesh = empty_shadow_mesh();
            restore_mesh_buffers(&mut mesh, positions, indices);
            if let Some(old) = slot.mesh.replace(meshes.add(mesh)) { meshes.remove(old.id()); }
            slot.geometry_key = Some(key);
            slot.built_at = Some(slot.pos);
            rebuilt += 1;
            static_rebuilt |= 1 << i;
        }
    }
    // Advance past everything scanned, not just what was rebuilt, so the window sweeps the whole
    // resident set at a steady rate instead of re-examining the head of it.
    lane.rebuild_cursor = (start + scan) % slot_count.max(1);

    // MONKEY (static torch cache): publish EVERY slot every frame — position, fade weight and the
    // six projections all move with the fixture and the cross-fade, and only the caster MESH is
    // cached. A slot whose map is stale (dirty, awaiting its rebuild turn) still publishes it; a
    // slot whose fixture has drifted away from where its map was baked publishes none, which forces
    // its weight to zero rather than showing shadows cast from a position the light has left.
    for slot in &lane.slots {
        let i = slot.cache_slot;
        // MONKEY (carried light stability): where this slot casts FROM this frame. Normally the
        // live fixture position. A slot that has drifted off its baked map AND is on its way out
        // publishes from `built_at` instead of withdrawing: withdrawing clears the slot's
        // `ready_mask` bit, and `TorchTable::pack` answers that by forcing the cross-fade weight
        // to 0 with no ramp — a hard shadow-off in one frame, which is the very pop the "fade,
        // not switch" law of this lane exists to remove. Frozen, the shadow dissolves over
        // `TORCH_FADE_RATE` from the last place it was correct; a third of a second later the
        // fixture is a stride away and the shadow is already gone, which reads as a shadow being
        // left behind rather than as a blink. A slot that is NOT evicting still withdraws — a
        // resident fixture mid-rebuild must never show shadows cast from a position it has left.
        let origin = if map_publishable(slot.built_at, slot.pos) {
            Some(slot.pos)
        } else {
            slot.built_at.filter(|_| slot.evicting)
        };
        if origin.is_some() {
            published.caster_meshes[i] = slot.mesh.as_ref().map(Handle::id);
        }
        let at = origin.unwrap_or(slot.pos);
        published.positions[i] = at.extend(slot.w);
        for (f, vp) in cube_view_projs(at).into_iter().enumerate() {
            published.view_projs[i * CUBE_FACES + f] = vp;
        }
    }
    // MONKEY (static torch cache): gather only around the dynamic subset, not every promoted
    // fixture. A single mesh serves those N maps and contains no rigid furniture already cached.
    let dynamic_positions: Vec<_> = fixture_positions.iter().copied()
        .filter(|(i, _)| dynamic & (1 << i) != 0).collect();
    let mut ent_admitted = 0;
    if dynamic != 0 {
        if lane.entity_mesh.is_none() { lane.entity_mesh = Some(meshes.add(empty_shadow_mesh())); }
        if let Some(mesh) = lane.entity_mesh.as_ref().and_then(|h| meshes.get_mut(h)) {
            let (mut positions, mut indices) = take_mesh_buffers(mesh);
            ent_admitted = collect_torch_entities(&ents, false, &dynamic_positions, &mut positions, &mut indices);
            restore_mesh_buffers(mesh, positions, indices);
        }
        published.entity_mesh = lane.entity_mesh.as_ref().map(Handle::id);
    }
    *views = published;

    // `WOW_TORCH_TRACE=1` — once a second, what the lane published: fixture distances + whether a
    // caster mesh exists. The app-side half of the debug (the shader group-3 sample is the GPU half).
    if std::env::var_os("WOW_TORCH_TRACE").is_some() {
        if now - *last_trace >= 1.0 {
            *last_trace = now;
            info!(
                "torch-trace: {}/{} slots of {} ({} candidates, caster {}, {} entity casters)",
                promoted.len(),
                want,
                MAX_TORCH_CASTERS,
                cands.len(),
                if lane.slots.iter().any(|s| s.mesh.is_some()) {
                    "built"
                } else {
                    "none"
                },
                ent_admitted,
            );
            // MONKEY (torch caster selection): per SLOT, the three numbers the selection is made
            // of — which fixture, what it scores at the player, and how far its cross-fade has
            // ramped. A shadow that pops is a `w` that jumped; a shadow that should be there and
            // is not is a fixture missing from this list (then read `cands`, above).
            // MONKEY (torch caster reach): scientific scores expose the small distant tail;
            // distance is still from the player, and eligibility now extends to 4R + 3 yd.
            for (i, e, p, sc, w) in &promoted {
                info!(
                    "  slot {i}: {e} at [{:.1},{:.1},{:.1}] d(player) {:.1} score {:.4e} w {:.2} (camera <= shadowDistance AND player <= 4R+3) static {} dynamic {}",
                    p.x,
                    p.y,
                    p.z,
                    p.distance(anchor),
                    sc,
                    w,
                    if static_rebuilt & (1 << i) != 0 { "rebuilt (GPU pending)" } else { "cached/requested" },
                    if dynamic & (1 << i) != 0 { "yes" } else { "no" },
                );
            }
        }
    }
}

// MONKEY (static torch cache): a rig can deform even while its root stands still. Cache only
// rigid non-creature parts; animated furniture follows the same nearest-N policy as characters.
fn static_part(part: &ModelPart, rigged: bool) -> bool {
    part.kind != ModelKind::Creature && !rigged && part.blend == ModelBlend::Opaque
}

fn collect_torch_entities(
    ents: &EntityCasters, want_static: bool, fixtures: &[(usize, Vec3)],
    positions: &mut Vec<[f32; 3]>, indices: &mut Vec<u32>,
) -> u32 {
    let mut admitted = 0;
    for (_, pick, part, global, rig_part, occluder, _) in &ents.parts {
        if !occluder.0 || static_part(part, rig_part.is_some()) != want_static { continue; }
        let solid = part.blend == ModelBlend::Opaque
            || (part.kind == ModelKind::Creature && part.blend == ModelBlend::AlphaTest);
        if !solid { continue; }
        let rig = rig_part.and_then(|r| ents.rigs.get(r.0).ok());
        let origin = global.map(GlobalTransform::translation)
            .or_else(|| rig.and_then(|r| ents.palettes.slot_origin(r.slot)));
        let Some(origin) = origin else { continue };
        if !fixtures.iter().any(|(_, p)| p.distance_squared(origin) <= TORCH_RANGE * TORCH_RANGE) { continue; }
        let base = positions.len() as u32;
        if let Some(rig) = rig {
            let Some(palette) = ents.palettes.world_palette(rig.slot, rig.bones() as usize) else { continue };
            for (v, p) in pick.0.positions.iter().enumerate() {
                let p = wow_to_bevy(*p);
                let (Some(joints), Some(weights)) = (pick.0.joints.get(v), pick.0.weights.get(v)) else {
                    positions.push(palette.first().map(|m| m.transform_point3(p)).unwrap_or(origin).to_array());
                    continue;
                };
                let mut skinned = Vec3::ZERO;
                for lane in 0..4 {
                    if weights[lane] > 0.0 {
                        if let Some(m) = palette.get(joints[lane] as usize) {
                            skinned += m.transform_point3(p) * weights[lane];
                        }
                    }
                }
                positions.push(skinned.to_array());
            }
        } else if rig_part.is_none() {
            if let Some(global) = global {
                positions.extend(pick.0.positions.iter().map(|p| global.transform_point(wow_to_bevy(*p)).to_array()));
            }
        }
        let added = positions.len() as u32 - base;
        if added == 0 { continue; }
        for tri in pick.0.indices.chunks_exact(3) {
            if tri.iter().all(|i| *i < added) { indices.extend(tri.iter().map(|i| base + *i)); }
        }
        admitted += 1;
    }
    admitted
}

/// Tear down the caster mesh and clear the published views (the lane went off, or we left the world).
fn teardown(lane: &mut TorchLane, meshes: &mut Assets<Mesh>, views: &mut TorchShadowViews) {
    for slot in &mut lane.slots {
        if let Some(handle) = slot.mesh.take() { meshes.remove(handle.id()); }
    }
    if let Some(handle) = lane.entity_mesh.take() { meshes.remove(handle.id()); }
    // MONKEY (torch caster selection): the slot table goes with them — an incumbent kept across a
    // teardown would come back at full weight in a room it is no longer in.
    lane.slots.clear();
    *views = TorchShadowViews::default();
}

// MONKEY (torch caster reach): regression boundaries and the unfloored distant ranking tail.
#[cfg(test)]
mod tests {
    use super::*;

    // MONKEY (review fixes): useful interior casters win regardless of an exterior fixture's
    // brightness; the fire dial scales only invented sources, including its live kill switch.
    #[test]
    fn candidates_follow_the_packed_lane_and_fire_gain() {
        let interior = LightLane::carried(true);
        let exterior = LightLane::carried(false);
        let authored = candidate_score(1.0, 10.0, 10.0, Some(&interior), true, false, 0.0);
        assert!(authored > 0.0);
        assert_eq!(candidate_score(100.0, 1.0, 10.0, Some(&exterior), true, false, 1.0), 0.0);
        for gain in [0.0, -1.0] {
            assert_eq!(candidate_score(100.0, 1.0, 10.0, Some(&interior), true, true, gain), 0.0);
        }
        assert_eq!(candidate_score(1.0, 10.0, 10.0, Some(&interior), true, true, 0.5), authored * 0.5);
        // No MOLR is needed once a containment-only fixture resolves interior; before resolution
        // the old LightRooms fallback is shared with build_light_data, in both directions.
        assert_eq!(candidate_score(1.0, 10.0, 10.0, Some(&interior), false, false, 0.0), authored);
        assert_eq!(candidate_score(1.0, 10.0, 10.0, None, true, false, 0.0), authored);
        assert_eq!(candidate_score(1.0, 10.0, 10.0, None, false, false, 0.0), 0.0);
    }

    #[test]
    fn eligibility_covers_four_reaches_with_slack() {
        for r in [7.0, 15.0, 48.0] {
            assert!(fixture_eligible(3.5 * r, 80.0, r, 80.0));
            assert!(fixture_eligible(4.0 * r + 3.0, 80.0, r, 80.0));
            assert!(!fixture_eligible(4.2 * r + 3.0, 80.0, r, 80.0));
        }
        assert!(!fixture_eligible(1.0, 80.01, 48.0, 80.0));
        assert!(fixture_eligible(1.0, 250.0, 48.0, 300.0));
        assert!(!fixture_eligible(32.0, 1.0, 7.0, 80.0));
        assert!(!fixture_eligible(1.0, f32::NAN, 48.0, 80.0));
    }

    // MONKEY (static torch cache): sparse physical slots, tie stability and live shrink/off.
    #[test]
    fn dynamic_subset_is_nearest_and_deterministic() {
        let fixtures = [(15, Vec3::X), (8, -Vec3::X), (2, Vec3::X * 20.0), (4, Vec3::ZERO)];
        assert_eq!(dynamic_set(&fixtures, Vec3::ZERO, 2), (1 << 4) | (1 << 8));
        assert_eq!(dynamic_set(&fixtures, Vec3::X * 20.0, 1), 1 << 2);
        assert_eq!(dynamic_set(&fixtures, Vec3::ZERO, 0), 0);
        assert_eq!(dynamic_set(&fixtures, Vec3::ZERO, 16).count_ones(), 4);
    }

    // MONKEY (static torch cache): a never-built slot publishes nothing; a merely STALE map keeps
    // being published (its shadow must not blink off while the rebuild cursor reaches it); a map
    // whose fixture has MOVED is withdrawn, because the projections no longer address it.
    #[test]
    fn a_cached_map_survives_staleness_but_not_fixture_drift() {
        assert!(!map_publishable(None, Vec3::ZERO));
        assert!(map_publishable(Some(Vec3::ZERO), Vec3::ZERO));
        assert!(map_publishable(Some(Vec3::ZERO), Vec3::X * 0.09));
        assert!(!map_publishable(Some(Vec3::ZERO), Vec3::X * 0.2));
        assert!(!map_publishable(Some(Vec3::ZERO), Vec3::splat(10.0)));
    }

    #[test]
    fn distant_scores_stay_positive_and_ordered() {
        let r = 10.0;
        let near = fixture_score(1.0, r, r);
        let far = fixture_score(1.0, 3.0 * r, r);
        let farther = fixture_score(1.0, 3.5 * r, r);
        let slack = fixture_score(1.0, 4.0 * r + 2.0, r);
        assert!(near > far && far > farther && farther > slack && slack > 0.0);
        assert!(far < near * 0.2);
        assert_eq!(fixture_score(0.0, 3.0 * r, r), 0.0);
        assert_eq!(fixture_score(1.0, 4.0 * r + 3.0, r), 0.0);
        assert!(fixture_score(2.0, 3.0 * r, r) > far);
    }
}
