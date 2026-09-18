//! The **one shared global light** — the faithful replacement for the per-material light copy.
//!
//! The real 1.12 client has a single scene light every draw reads; updating it is O(1). We used to
//! store the resolved [`super::WowLighting`] *per material* and re-push it into every loaded terrain/
//! model/liquid/wdl material each frame — Bevy then freed+recreated every material's bind group
//! (`bevy_pbr` material.rs has an explicit "no fast path; we delete and recreate" TODO), a confirmed
//! ~40fps tax.
//!
//! Instead: ONE persistent GPU storage buffer, created once from the main-world [`RenderDevice`]; every
//! material references it via `#[storage(90, read_only, buffer)]` (a pre-made `Buffer`, baked into the
//! bind group at prepare, zero per-frame upload — `#[uniform]` can't do this, it always re-allocates). The
//! material assets are **never mutated after creation**, so no bind group is ever rebuilt. Each frame
//! [`build_light_data`] (main world) packs the resolved light into a std430 blob and [`upload_light`]
//! (render world, `PrepareResources`) writes it in place — all bind groups see the new data, zero
//! rebuilds, regardless of how many tiles/models are loaded or how fast the clock moves.

use benilla_formats::LiquidKind;
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_resource::{Buffer, BufferDescriptor, BufferUsages};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::{Render, RenderApp, RenderSystems};

// MONKEY (flame flicker): the wobble's component + evaluator, folded into the packed colour below.
use super::flicker::{FlameFlicker, FlickerMod};
use super::prop_probes::MAX_PROP_PROBES;
use super::{sh, WowLighting};
use crate::dev_state::DebugState;
use crate::view::ViewDistance;
use crate::view::WorldCamera;

/// The shared light, std430-packed as contiguous `vec4<f32>` rows. All-`vec4` so std430 == std140
/// (each row 16-aligned, no stride surprises). The row order is the canonical layout every shader
/// bound at `storage(90)` mirrors as a prefix — the WGSL structs in `wow_model.wgsl`/`terrain.wgsl`/
/// `wow_effect.wgsl`. (`liquid.wgsl`/`wdl.wgsl` reuse the field NAMES but bind their own
/// per-material uniforms fed by `apply_wow_lighting` — editing this layout does NOT reach them.)
///   0 light_ambient (w=Mod2x 1.0) · 1 light_diffuse (w=clamp on) · 2 light_sun (w=dir/SH enable) ·
///   3 light_spec (w=terrain shininess 20) · 4 fog_color (w=enable) · 5 fog_params (x=start y=end
///      w=farclip; MONKEY (moon shadows) `.z` = the SIGNED directional-shadow weight — `+sun_w`
///      by day, `−(moon weight × moonShadowStrength)` at night, 0 when neither body casts. The
///      two are mutually exclusive by [`moon_shadow_weight`], so one float carries both; see
///      [`pack_shadow_lane`]) ·
///   6-8 sh_c10_{r,g,b} · 9-11 sh_c13_{r,g,b} · 12 sh_c16 (.w = the world-shadow flag in the
///      integer part, MONKEY (bake floor) `interiorBakeFloor × interiorGain` in the fraction —
///      [`BAKE_LANE_SCALE`]) ·
///   13-14 water river {shallow,deep} (w=alpha) · 15-16 water ocean {shallow,deep} (w=alpha) ·
///   17 grade — `.x` = the SIDN night fraction (`WowLighting::sidn_night`: 1 overnight, 0 all day;
///      `wow_model.wgsl` scales every WMO SIDN material's authored emissive by it); `.yzw` = the
///      sun's isotropic SH DC term at intensity 1 (the exterior M2 lane scales it per instance) ·
///   18 wmo_fog_color · 19 wmo_fog_params (x=start y=end) — the INTERIOR fog triple (the 4 s
///      camera-in-WMO MFOG crossfade; == the scene fog outdoors). Read only by `wow_model.wgsl`'s
///      interior lanes (round-6 Q-I consumer map); terrain mirrors the rows for layout only ·
///   20 point_count (x = live entries) · 21+ the point-light table, TWO rows per light:
///      `[pos.xyz, range]`, `[rgb, lane]` (decision 0278 — the Gouraud point term reads this in the
///      VERTEX stage; bevy's own clusterable buffer is fragment-only in the view bind-group layout,
///      so the lights ride our buffer instead).
///      MONKEY (light lanes): the colour row's `.w` — free until now — is the **lane discriminator
///      AND the interior EFFECTIVE RADIUS in one float**: `0.0` = an EXTERIOR light, anything
///      `> 0.5` = an INTERIOR fixture whose value IS its radius `R` in yards ([`interior_reach`] —
///      the authored MOLT end × `interiorAttenScale`, always clamped ≥ 1.0, so the two cases can
///      never be confused; the shader derives the whole soft profile from `R`, out to a `2R` fill
///      dome). MONKEY (light lane by position): which side a light falls on is decided by WHERE IT
///      STANDS ([`LightLane`] / [`classify_light_lanes`]), not by whether it claims a room — that
///      is what lets Stormwind's street torches light the cobbles while the Goldshire inn's
///      fixtures still stay off the lawn. The two consumer families are DISJOINT: the exterior
///      `point_light_sum` / `wmo_exterior_point_sum` (terrain/wow_model/static_gx) skip `> 0.5`
///      and the interior `interior_room_light` (wow_model/static_gx) skips `< 0.5`, which is what
///      stops an inn's candles pooling on the grass outside its wall and an outdoor campfire
///      lighting the floor through a closed door. `[pos.xyz, range]`'s `range` is UNTOUCHED
///      (still the 48 yd candidacy constant), so the exterior lanes are byte-identical.
///
/// The GPU buffer is LARGER than this per-frame blob: the interior-prop probe table
/// (`lighting::prop_probes`, 7 rows per slot) lives at its tail — [`light_blob_bytes`] sizes the
/// buffer for both, `upload_light` rewrites only this prefix each frame, and `upload_prop_probes`
/// rewrites the tail only when a prop spawns/despawns. Only `wow_model.wgsl` declares the tail
/// region — the other shaders mirror the PREFIX and bind the same (larger) buffer, which wgpu
/// allows. Keeping the probes out of this struct keeps it stack-cheap: the ExtractResource clone
/// runs every frame, and a ~900 KB by-value blob overflowed a render-thread stack (measured live).
#[repr(C)]
#[derive(Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct LightStd430 {
    rows: [[f32; 4]; LIGHT_HEADER_ROWS],
    points: [[f32; 4]; 2 * MAX_POINT_LIGHTS],
}

/// Header row count of the canonical layout above (rows 0..=20). Every producer of a light blob
/// sizes against this — the portrait booth included.
pub const LIGHT_HEADER_ROWS: usize = 21;

/// Pack the **model-lighting core** into `rows` — every row the model shaders' lit lanes derive
/// from the (ambient, diffuse, sun_dir) triple: rows 0-2 (ambient/diffuse/sun, w = enables), the
/// SH block (rows 6-11 + row 12 `.xyz`; DC = ambient in the `.w` lanes), and the sun's SH DC
/// redistribution (row 17 `.yzw`, at intensity 1). Leaves every other row — and row 12 `.w`
/// (free) / row 17 `.x` (SIDN) — untouched.
///
/// The sun's bands are the `Model2.bls` closed form — the SAME [`sh::prop_probe_coeffs`] fold the
/// interior lane runs, per the disassembly of the shipped ARB program (wow-re
/// `system/models/scratch/model2-bls-vertex-sh.md`): `E(n) = D·(3 + 16μ + 15μ²)/34`, μ = n·u
/// toward-light, EVERY band linear in the committed colour `D` — there is no separate amplitude
/// scalar, and the per-instance intensity lives entirely in that colour (a consumer multiplies
/// ALL sun terms by I; packed here at I = 1). The peak (μ=1) equals the FFP walls' `D·(N·L)`
/// peak by construction (the 16/17 accumulate scale exists for exactly that), and the closed form
/// never goes meaningfully negative — the old trace-fit's ~¼-strength lobe with a negative back
/// side (shadow-side characters turned blue as the warm channels floored at 0) is superseded.
///
/// **The SH block (rows 6-11, row 12 `.xyz`, row 17 `.yzw`) is the live exterior M2 response**
/// (0803). It was dormant for months — 0410 took the lane off this curve onto a hard-cutoff FFP
/// matte on the director's look call and nothing consumed the rows — until 0796 refuted the fidelity
/// premise behind that retirement (the reference's M2 lane IS this SH shader) and 0799 put the two
/// side by side for the call. Anything that stops writing these rows now renders every exterior
/// doodad and creature black. The interior-prop and glue-rig lanes are unaffected — they fold their
/// own probes through the per-instance `prop_probes` table, not these rows.
///
/// This is the ONE packer for the scene light ([`build_light_data`]) AND the portrait booth's
/// studio light (`portrait::setup_booths`): the booth used to hand-copy the layout and rendered
/// black portraits the day 0354 moved the lit lanes onto rows it never wrote. A producer that
/// copies the layout goes stale the day the layout moves — so producers don't copy it, they call
/// this.
pub fn pack_model_core_rows(
    rows: &mut [[f32; 4]; LIGHT_HEADER_ROWS],
    ambient: [f32; 3],
    diffuse: [f32; 3],
    sun_dir: Vec3,
) {
    rows[0] = [ambient[0], ambient[1], ambient[2], 1.0]; // 0 light_ambient (w=Mod2x 1.0)
    rows[1] = [diffuse[0], diffuse[1], diffuse[2], 1.0]; // 1 light_diffuse (w=clamp on)
    rows[2] = [sun_dir.x, sun_dir.y, sun_dir.z, 1.0]; // 2 light_sun (w=dir/SH enable 1.0)
                                                      // The sun lobe folded at intensity 1 with NO ambient — ambient rides the DC lanes directly
                                                      // (it never scales with the per-instance intensity), while the fold's own `.w` output is the
                                                      // sun's DC redistribution, re-homed onto row 17 `.yzw` so the shader can scale it by I.
    let sun = sh::prop_probe_coeffs([0.0; 3], &[(-sun_dir, diffuse)]);
    for (i, row) in sun.iter().enumerate().take(6) {
        rows[6 + i] = row.to_array(); // 6-8 sh_c10_{r,g,b} · 9-11 sh_c13_{r,g,b}
    }
    rows[6][3] = ambient[0]; // the DC lanes carry ambient alone
    rows[7][3] = ambient[1];
    rows[8][3] = ambient[2];
    rows[12][0] = sun[6].x; // 12 sh_c16 xyz — .w is the shared world-shadow/bake-floor lane, written
                            // by `build_light_data` alone (the portrait booth leaves it 0, so a
                            // studio portrait takes neither — the frozen look is deliberate)
    rows[12][1] = sun[6].y;
    rows[12][2] = sun[6].z;
    // 17 `.yzw` — the sun's SH DC redistribution at intensity 1 (`D·(4/17)(0.375+0.9375(uₓ²+u_y²))`
    // per channel): an SH consumer adds it × the per-instance intensity (dormant since 0410 — see
    // the doc above). `.x` (SIDN) is the scene's.
    rows[17][1] = sun[0].w;
    rows[17][2] = sun[1].w;
    rows[17][3] = sun[2].w;
}

/// The reference's committed point-light diffuse is the **RAW** `colour × intensity × modelFade`
/// — over-gamut values included (VERIFIED at the bytes + OBSERVED live, wow-re
/// `models/scratch/trace-forensics-overgamut-point-commit-d3d.md`; compose arithmetic
/// `m2-light-emitter-instances.md` §6b, animate leg `716a67`–`716aa6`).
///
/// `0x71ca80` — which two prior rounds read as a clamp01 and then as a peak-normalize — is
/// actually a lossy **RGBE-style encoder**: it stores a peak-normalized byte colour at
/// `CGxLight+0x14` *and* the raw peak float `m = max(1, r, g, b)` at `+0x20`, and the device copy
/// `0x593040` **decodes them right back** (`byte · m/255 ≈ raw channel`) before the GL light is
/// set. Net effect: identity up to 8-bit peak-relative quantization (≤ ~0.5 %, which we skip). A
/// night terrain draw in the ring capture commits `(1.2, 1.035, 0.805)` verbatim — over-white
/// preserved. So we pack the raw product; the saturation the eye sees comes from the *vertex*
/// clamp of the summed lighting (GL T&L clamps `ambient + sun + Σ points` per vertex BEFORE
/// interpolation — see `terrain.wgsl`), never from the commit.
pub fn commit_raw(rgb: [f32; 3]) -> [f32; 3] {
    rgb.map(|c| c.max(0.0))
}

/// **Slot** capacity of the packed point-light table (fixed-size in the WGSL mirror structs — keep
/// in sync). 256 lights × 2 rows × 16 B = 8 KB — generous for the densest streamed village/city
/// interior set; [`build_light_data`] packs the nearest-to-camera first when over capacity.
///
/// This is the BUFFER shape (`LightStd430` and [`RoomClaimTable`] size against it) and it must not
/// move: the blob is 8528 B and is mirrored by three shaders plus the portrait booth. How many of
/// those slots may actually hold a light is [`MAX_LIVE_POINT_LIGHTS`], which is one less.
pub(super) const MAX_POINT_LIGHTS: usize = 256;

/// MONKEY (ext light k8): how many of [`MAX_POINT_LIGHTS`] slots may hold a LIVE entry — **255**,
/// one short of the buffer.
///
/// The three shaders publish each draw unit's chosen exterior lights from the vertex stage to the
/// fragment stage as a packed index list (`ext_sel`). Widening that list from three lights to eight
/// — the fix for the Darkmoon Faire "scars", where adjacent draw units kept different threes out of
/// 15+ synthesised torch lights and the disagreement drew a straight edge across the grass — cost
/// bits: eight ranks in two u32s is **8 bits each**, so an index runs 0..=254 and **255 is the
/// `EXT_SEL_EMPTY` sentinel** for "this rank is unfilled". A 256th live light would pack as 255 and
/// every shader would read it as "stop here", silently truncating that unit's whole list.
///
/// So the pack truncates one entry earlier. The lights are sorted nearest-camera-first before the
/// truncation, so the one this drops is the farthest of a 256-strong set — sub-pixel and usually
/// fogged at that density. The BUFFER keeps all 256 slots, so nothing about the layout, the 8528 B
/// blob or the room-claim table changes.
pub(super) const MAX_LIVE_POINT_LIGHTS: usize = MAX_POINT_LIGHTS - 1;

/// MONKEY (dynamic interiors): a fixture this close to the camera (yd) is packed regardless of the
/// portal flood — see the room term in [`build_light_data`]. Sized to a WHOLE BUILDING PLUS its
/// approach, not just the room: at 40 yd, walking around the outside of the inn kept crossing the
/// boundary and a fixture would evict, so an interior-classified NPC by the door brightened and
/// dimmed as the camera moved (issue A). 90 yd keeps a building's fixtures admitted the whole time
/// you are near it; the per-fixture 48 yd range check still gates which fragments they actually
/// light, so the wider admit only affects table MEMBERSHIP, never reach. (256-slot cap: Goldshire's
/// buildings stay well under it at this radius.)
const INTERIOR_NEAR_ADMIT: f32 = 90.0;

/// Pack lights only within this camera distance (yd). A point light's whole visible effect lives
/// within its ~48 yd candidacy range (`spawn::POINT_LIGHT_RANGE`, the packed `.w`); a pool farther
/// than ~300 yd is sub-pixel and usually fogged, and the cap keeps the per-vertex selection walk
/// (0285: each unit picks its ≤3 nearest from this table) bounded.
const POINT_PACK_RADIUS: f32 = 300.0;

/// **The rooms a point light belongs to** — a WMO's own MOLT fixture (the groups whose MOLR names
/// it) or one of its props' M2 lights (the groups whose MODR names the prop). Absent on an ADT map
/// doodad's light, a creature's, a GameObject's: nothing claims those.
///
/// The fourth rider of decision 0689's law, after the prop's mesh, its particle clouds and its
/// ribbon trails. The reference never needs it: a WMO's furniture is instantiated out of each
/// **visible** group's MODR list, so a torch in a culled room does not exist and registers no
/// light. Its light-register walk really does have no visibility term of its own — byte-verified,
/// wow-re `m2-light-emitter-instances.md` §4: the gate for a model entering the register walk is
/// the scene update-list activation flag `[model+0x10]`, "not visibility, not distance, not LOD",
/// and the ≤4 cap is purely receiver-side. So the faithful fix is NOT a visibility test bolted onto
/// the gather; it is that the SOURCE should not be there at all, which is what this component says.
/// [`build_light_data`] drops the light while its rooms are culled, exactly as the model-visibility
/// authority drops the prop's own submeshes.
///
/// A newtype rather than a bare [`crate::wmo_portal::WmoGroupVis`] on purpose: that component on a
/// light entity would enlist it in `apply_model_visibility`'s `group_only` query — a `PointLight`
/// carries `Visibility` and `GlobalTransform`, so it matches — making the model-visibility
/// authority a second writer on an entity whose `Visibility` nothing reads (decision 0025).
#[derive(Component)]
pub struct LightRooms(pub(crate) crate::wmo_portal::WmoGroupVis);

impl LightRooms {
    /// MONKEY (fire GO lights): claim a room for a light spawned OUTSIDE this crate — the app-side
    /// carried lights (`benilla_app::entities::carried_light`), whose owner's room comes from the
    /// interior classifier rather than from a WMO's own MODD/MOLR tables. Without a constructor the
    /// tuple field's `pub(crate)` made the component unbuildable in benilla-app, so a GameObject's
    /// torch could never carry a room — and therefore could never be promoted to a cube-map shadow
    /// caster (`torch_shadow's candidate query is `With<WorldPointLight>, With<LightRooms>`).
    pub fn new(rooms: crate::wmo_portal::WmoGroupVis) -> Self {
        Self(rooms)
    }
}

/// MONKEY (room gate): the rooms a fixture may **LIGHT** — deliberately a different question from
/// [`LightRooms`], which answers "which rooms must be VISIBLE for this light to exist at all".
///
/// The gate needs a set that covers every room the fixture actually stands in the middle of, and
/// the authored MOLR relation alone does not supply one. Measured on the shipped 1.12 corpus
/// (`benilla-extract wmolights` + the group flags):
///   * **Goldshire inn** — 12 groups, and only TWO author a MOLR at all (MOGP `0x200`). Nine of its
///     ten fixtures are claimed by `g4 "upstairs"`, which is the EXTERIOR-flagged shell spanning
///     the whole building (`z -0.2 .. 23.3`); the tenth by `g3 "entry"`. Its kitchen, common room,
///     hall, guest rooms and basement claim NOTHING.
///   * **NSabbey** — 42 fixtures over 14 groups, but `Main Hall`, `LftWng`, `RtWng`, `Library`,
///     `Library2`, `Library Wing` and `Stairs2` are named by no MOLR either.
/// Gating purely on MOLR would therefore black out most of both buildings — the abbey's look is
/// the one the owner signed off, so that is a regression, not a fix. MOLR is authored for the
/// reference's own purpose (register GL lights while drawing a VISIBLE group's doodads and units),
/// not to describe which walls a fixture lights, which the reference took from the MOCV bake.
///
/// So the claim set is the UNION of MOLR with the interior groups whose authored MOGI **bounding
/// box contains the fixture** — dense, authored, static, and resolved once at spawn. On the inn
/// that gives every room its own fixtures back while leaving the basement (`z <= 0.45`, no fixture
/// inside) and the far guest room claimed by nobody, which is exactly the reported leak; on the
/// abbey every interior group gets at least one fixture, so its look is preserved.
///
/// MONKEY (portal claims): the set is now built by [`benilla_formats::room_claims`] and is the
/// COMPLETE, ordered claim list — MOLR folded in, plus the containment above, plus the groups one
/// open PORTAL away within the fixture's reach (the "light crosses a doorway" rule that stops a
/// continuous floor changing brightness in a straight line at a group boundary). Because it already
/// contains MOLR, [`RoomClaim::build`] takes it INSTEAD of the MOLR list rather than unioning the
/// two: re-adding MOLR there would re-admit exactly the district-scale shells the rule deliberately
/// marks exterior-lane-ineligible.
///
/// Each entry is `group | `[`LIT_ROOM_EXT_DENY`]: bit 15 marks a claim that gates the fixture but
/// must not be honoured by the exterior batch lane (see the constant).
///
/// The WMO MOLT lane inserts this, and since MONKEY (interior prop lights) so does the WMO MODD
/// PROP lane — a hanging lantern whose flame light we synthesise needs the same room claim an
/// authored fixture gets, or the gate would refuse it in the very room it hangs in. A
/// carried/GameObject light already carries the single room the interior classifier put it in,
/// through [`LightRooms`], and needs no supplement.
#[derive(Component)]
// MONKEY (review fixes): the placement belongs to the lighting claims themselves. A fixture
// contained by a room can have NO MOLR/MODR visibility membership; it still needs a GPU room key.
pub struct LightLitRooms {
    pub(crate) rooms: crate::wmo_portal::WmoGroupVis,
    /// MONKEY (soft portal claims): index-parallel with `rooms.groups` — each claim's softness.
    /// A separate array rather than more bits in the `u16` id because a fade is 4 floats and a
    /// group id has 15 spare BITS; index-parallel rather than a map because the packer walks the
    /// two together and the GPU record interleaves them at a fixed stride, so a length mismatch is
    /// a claim gated by another claim's doorway. Built by the one producer
    /// (`terrain_stream::spawn::fx`'s `pack_claims`), which is why the invariant holds by
    /// construction; a legacy/fallback path that has no fades supplies an EMPTY slice, read as
    /// "every claim is hard", i.e. exactly the binary gate this replaced.
    pub(crate) fades: std::sync::Arc<[ClaimFade]>,
}

/// MONKEY (soft portal claims): the softness of ONE claim — the doorway the fixture's light came
/// through and how far past it the light still reaches.
///
/// `radius == 0` is the HARD claim (containment/MOLR): weight 1 everywhere in the group, which is
/// what the gate did for every claim before this — and it is [`ClaimFade::default`], so every path
/// that has no fade to give (a carried light's raw MOLR rooms) gates exactly as it always did. A
/// portal claim carries the real numbers, and the shader turns them into
/// `w = entry * (1 − smoothstep(0, radius, max(|P − center| − slack, 0)))`.
///
/// **`center` is in BEVY WORLD space**, not the WMO model space `benilla_formats::room_claims`
/// measured it in: the shader has only the fragment's world position, and converting there would
/// need the placement's matrix per fragment. The producer holds that matrix already, so the
/// conversion happens once, at claim time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClaimFade {
    /// The portal's centre, BEVY WORLD space.
    pub center: Vec3,
    /// The portal polygon's bounding-sphere radius (yd) — distance inside it counts as zero, so
    /// the doorway itself is weight 1 on both sides of the group plane.
    pub slack: f32,
    /// The fade length (yd): the fixture's reach still unspent at the doorway. **0 = hard claim.**
    pub radius: f32,
    /// The weight the claim already carries AT `center` — 1 for a first hop, and for a second hop
    /// whatever the first hop's fade had decayed to by that door. Multiplies the smoothstep, which
    /// is what keeps a two-hop chain continuous (and monotone) at every threshold.
    pub entry: f32,
}

/// MONKEY (portal claims): bit 15 of a [`LightLitRooms`] entry — "this claim gates the fixture, but
/// the EXTERIOR batch lane must ignore it". A group index cannot reach it (the record's room key is
/// 12 bits), so it rides for free in the id.
///
/// It exists for one population: a city's district-scale EXTERIOR shell, whose MOGI box swallows
/// the interiors inside it (`Stormwind.wmo`: 470 of its 606 fixtures stand inside one). Such a
/// claim must stay in the set, because a fixture stripped of its only claim is packed UNGATED and
/// the ungated arm fails OPEN — one candle lighting every room of the building. But it must not
/// reach the exterior lane, which would put that candle's pool on the cobbles outside the wall.
/// A building-scale exterior group (the Goldshire inn's shell, `57.7 x 32.2 yd`) carries no deny
/// bit and IS lit — that is bug B's second half.
pub const LIT_ROOM_EXT_DENY: u16 = 0x8000;

/// MONKEY (room gate): u32 per packed light in the claim table — `[instance, count, g0..g5]`,
/// MONKEY (soft portal claims) followed by one 4-word FADE record per claim slot
/// (`[center.x, center.y, center.z, radius|slack]`) at [`ROOM_CLAIM_FADE`].
/// **Must equal `ROOM_CLAIM_STRIDE` in `static_gx.wgsl`** — a mismatch reads every fixture's
/// claims out of another fixture's record, which blanks (or floods) every building at once.
pub const ROOM_CLAIM_STRIDE: usize = 32;
/// MONKEY (soft portal claims): the first FADE word of a claim record, i.e. `2 + ROOM_CLAIM_MAX`.
/// Slot `k`'s fade is `[ROOM_CLAIM_FADE + 4k .. +4]`. **Must equal `ROOM_CLAIM_FADE` in
/// `static_gx.wgsl`.**
pub const ROOM_CLAIM_FADE: usize = 8;
/// MONKEY (soft portal claims): the fixed-point scale of the packed radius/slack pair — both ride
/// one u32 as `u16` yards x 256 (low half radius, high half slack). 1/256 yd is far below anything
/// a smoothstep over 2..48 yd can show, and the range tops out at 255.99 yd, which is five times
/// the widest reach the packer will ever hand out (`CLAIM_REACH_MAX` = 48). Fixed point rather
/// than two f32 lanes because the alternative is a 40-word stride for two numbers that are already
/// coarse; fixed point rather than f16 because both sides are three arithmetic ops with no
/// bit-twiddling to get wrong. **Must equal `CLAIM_FADE_SCALE` in `static_gx.wgsl`.**
pub const CLAIM_FADE_SCALE: f32 = 256.0;
/// MONKEY (room gate): claims a fixture can carry before it is packed UNGATED instead (fail-open —
/// a truncated list would black out the rooms that fell off the end). **Must equal
/// `ROOM_CLAIM_MAX` in `static_gx.wgsl`.**
pub const ROOM_CLAIM_MAX: usize = 6;

/// MONKEY (portal claims): bit 16 of a packed GPU claim word — "the EXTERIOR batch lane may honour
/// this claim". The CPU side of the same fact is [`LIT_ROOM_EXT_DENY`] (inverted, see
/// [`RoomClaim::write`]). **Must equal `CLAIM_EXT_OK` in `static_gx.wgsl`**, and must stay clear of
/// the 12-bit `+1` group id below it.
pub const CLAIM_EXT_OK: u32 = 1 << 16;

/// MONKEY (soft portal claims): bits 17..=24 of a packed GPU claim word — [`ClaimFade::entry`] as
/// a byte (`round(entry * 255)`). It rides the id word rather than the fade record because the id
/// word has 15 bits going spare above [`CLAIM_EXT_OK`] (a group index is 12) while the fade record
/// is four words of exact f32 centre plus a full u16/u16 pair, and widening the stride for one
/// byte would cost 256 x 4 more bytes of buffer for nothing. A HARD claim writes 0 here and the
/// shader never reads it (it returns 1.0 the moment it sees `radius == 0`), so the zero padding
/// cannot be mistaken for "contributes nothing". **Must equal `CLAIM_ENTRY_SHIFT` in
/// `static_gx.wgsl`.**
pub const CLAIM_ENTRY_SHIFT: u32 = 17;

/// MONKEY (room gate): this frame's claim table, index-parallel with [`WowLightData`]'s point
/// entries. Its own resource and its own GPU buffer on purpose: [`LightStd430`] is mirrored by
/// three shaders plus the portrait booth and must never be resized, and only `static_gx` reads
/// this. `static_gx::render` owns the buffer and the binding.
#[derive(Resource, Clone, ExtractResource)]
pub struct RoomClaimTable(pub Box<[u32; ROOM_CLAIM_STRIDE * MAX_POINT_LIGHTS]>);

/// MONKEY (room gate): the claim table's byte size — the one place `static_gx::render` sizes its
/// GPU buffer from, so the table cannot grow here and leave the binding short (a bound storage
/// buffer smaller than the shader's runtime-sized array fails validation at draw time, which
/// vanishes every building).
pub fn room_claim_bytes() -> u64 {
    (ROOM_CLAIM_STRIDE * MAX_POINT_LIGHTS * std::mem::size_of::<u32>()) as u64
}

impl Default for RoomClaimTable {
    fn default() -> Self {
        Self(Box::new([0; ROOM_CLAIM_STRIDE * MAX_POINT_LIGHTS]))
    }
}

/// MONKEY (interior attenuation): a fixture's **authored attenuation end** in yards — the WMO
/// MOLT record's `attenuation_end` (`+0x2c`), carried from the spawn to the packer because
/// `PointLight` has nowhere to put it and `PointLight::range` means something else entirely (the
/// 48 yd *candidacy* constant, which the exterior lanes still read).
///
/// Only [`crate::terrain_stream::spawn`]'s MOLT lane inserts it. Every other source is an **M2**
/// light, whose authored attenuation pair is demonstrably NOT a reach in this corpus — a Karazhan
/// BONFIRE authors `end = 0.97` yd and the whole orc brazier/firepit family `0.33/0.97`, while
/// `2.22/5.56` is a template default stamped on half the weapons and glue models.
/// `benilla_formats::M2Light`'s own doc says it: "the GL curve is fixed and ignores these" — the
/// fields are parsed as a cull hint no consumer reads. So M2 sources get the bucketed
/// [`m2_light_reach`] instead, derived in the packer from the intensity they already carry — which
/// also means the app-side spawners (carried torches, transport props) need no change at all.
///
/// MOLT is the opposite case and the reason this exists: the reference really does fold a MOLT
/// fixture through its authored window (wow-re `trace-forensics-abbey-interior-d3d` §4 fitted the
/// fold at exactly the `+0x28/+0x2c` values), and the vanilla numbers are sane — Goldshire inn
/// 6.97-9.53 yd over 10 fixtures, its blacksmith 6.0 over 3, NSabbey 4.17-5.56 over 42.
#[derive(Component)]
pub struct LightReach(pub f32);

/// MONKEY (light lane by position): which consumer family a point light belongs to — decided by
/// WHERE THE LIGHT PHYSICALLY IS, not by whether it claims a room.
///
/// [`build_light_data`] used `Has<LightRooms>` for this, and that conflated two different
/// questions. `LightRooms` answers "which rooms must be visible for this light to exist" (the
/// portal gate + the torch-shadow eligibility, decision 0689) — a MODR/MOLR *reference*. The lane
/// answers "is this light inside a sealed room, or out on the street", which is a fact about
/// geometry. In vanilla content the two disagree constantly, in both directions:
///
/// - **Stormwind.** 235 of its 606 MOLT fixtures are named ONLY by EXTERIOR-class groups (MOGP
///   `& 0x48`) — the Trade District's street torches and hanging lanterns — and 45 more are named
///   by no group at all. Under the old rule every one of the 235 was filed INTERIOR, so the
///   exterior consumers skipped it and the interior consumer never runs on a street: they lit
///   *nothing at all*. That is bug B's first cause.
/// - **The Goldshire inn.** Its 10 MOLT fixtures claim group 4, which IS exterior-flagged, yet
///   they physically stand inside the inn's rooms. A rule keyed on the claimed group's FLAGS would
///   file them EXTERIOR and put a warm pool on the lawn outside the wall — precisely the leak the
///   lane split was introduced to stop. A rule keyed on POSITION files them interior, because the
///   classifier's down-ray from each fixture lands on the inn's own interior-class floor.
///
/// So the verdict is the client's own light-attach ray ([`crate::wmo_portal::indoor_verdict_at`]
/// with `LightAttach::DownRay` — the same predicate that classifies a standing NPC), run ONCE per
/// static light in [`classify_light_lanes`] and refreshed only when the resident WMO set changes.
/// A CARRIED light takes its lane from its owner's already-resolved room instead
/// (`benilla_app::entities::carried_light`): the classifier has run for the bearer, and re-raying
/// from a swinging torch would be one ray per light per frame for an answer we already hold.
#[derive(Component, Clone, Copy, PartialEq, Eq, Debug)]
pub struct LightLane {
    /// `true` ⇒ the packer writes the interior reach into the colour row's `.w`; `false` ⇒ `0`.
    pub interior: bool,
    /// The [`WmoResidency`](crate::interior::WmoResidency) generation this verdict was taken at —
    /// a light classified before its building streamed in reads `Outdoors`, and must be re-asked
    /// once the building is there. [`SETTLED`](Self::SETTLED) means "never re-ask".
    pub generation: u32,
}

impl LightLane {
    /// A generation that never goes stale — the carried lane's stamp (see the type doc).
    pub const SETTLED: u32 = u32::MAX;

    /// The carried lane's constructor: interior iff the bearer is standing in a room.
    pub fn carried(interior: bool) -> Self {
        Self {
            interior,
            generation: Self::SETTLED,
        }
    }
}

/// MONKEY (interior attenuation): the interior reach (yd) of an **M2** point light of committed
/// intensity `i`, bucketed on the same ladder [`benilla_formats::fire_intensity`] buckets a
/// synthesised flame onto (candle 0.6 · torch 1.5 · brazier 2.0 · bonfire 3.0), so a synthesised
/// source lands on its own bucket exactly and an authored one is filed by how bright it is.
///
/// Bucketing rather than reading the record is the finding, not a shortcut — see [`LightReach`]:
/// the authored M2 attenuation pair does not describe a reach in this corpus. The yard figures are
/// sized so the pool matches the flame you can see: a table candle lights its table, a wall torch
/// its corner, a brazier the end of a hall, a bonfire the clearing around it.
pub fn m2_light_reach(intensity: f32) -> f32 {
    // MONKEY (portal claims): the ladder itself lives in `benilla_formats::room_claim` — the spawner
    // needs it to size a synthesised prop light's PORTAL HOP, and the offline `wmolamps` audit needs
    // the identical rungs to predict what the runtime will claim. One ladder, both readers.
    benilla_formats::room_claim::m2_light_reach(intensity)
}

/// MONKEY (soft falloff): the **effective radius** packed into an interior entry's lane — the
/// authored reach after the live `interiorAttenScale` ([`DynamicInteriors::atten_scale`], default
/// 1.6). The shader derives its whole profile from this one number as a fraction of it (soft core
/// at `0.26 R`, window vanishing at `R`, fill dome out to `1.5 R` — see `interior_room_light`), so
/// the cvar is a pure zoom on every pool at once rather than three separate knobs.
///
/// `scale == 0` is the A/B **off** switch and maps to [`INTERIOR_LEGACY_REACH`] — the 48 yd the
/// lane used before there was a window at all. Under the soft profile that is the widest, flattest
/// pool the lane can make (the core alone is ~12.5 yd wide), i.e. the nearest thing left to the
/// pre-window flat lane; it is no longer the byte-exact restore it was, because the window's SHAPE
/// moved with the change. The `max(…, 1.0)` floor keeps the value clear of the `0.5` lane
/// threshold at any scale.
///
/// MONKEY (torch caster selection): `pub` so `benilla_app::torch_shadow` can rank fixtures by the
/// SAME effective radius the shader windows them with. Recomputing it there would be a second
/// copy of a value the packer already owns, and a drift between them would rank a fixture the
/// shader has already faded to black.
pub fn interior_reach(reach: f32, scale: f32) -> f32 {
    if scale <= 0.0 {
        return INTERIOR_LEGACY_REACH;
    }
    (reach * scale).clamp(1.0, INTERIOR_LEGACY_REACH)
}

/// The reach an interior fixture had before the authored window existed: `spawn::POINT_LIGHT_RANGE`.
/// Mirrored as a plain constant because it is now a *legacy* value — the A/B's off arm — rather
/// than the lane's working range, and also the cap on a widened one (past it the packed table's own
/// membership rules, not the window, decide what a fragment sees).
const INTERIOR_LEGACY_REACH: f32 = 48.0;

/// MONKEY (fire GO lights): marks a `PointLight` whose colour/intensity were SYNTHESISED from a
/// model's flame particle emitter rather than authored ([`benilla_assets::ModelLight::synthetic`]).
///
/// Two lanes read it, both in [`build_light_data`]: the `fireLightGain` cvar multiplies the
/// intensity of exactly these entries (so the dial is live — retuning it does not respawn a single
/// prop, which is the whole point of packing it here rather than folding it in at spawn), and the
/// `WOW_POINTS_DUMP` census counts them, so "is that light in the table one we invented?" is a
/// number rather than a guess.
#[derive(Component)]
pub struct SyntheticFireLight;

/// MONKEY (fire GO lights): the live gain on every SYNTHESISED fire light (`fireLightGain`, default
/// 1.0), bridged from benilla-app's cvars the way [`DynamicInteriors`] is. `0` turns the whole
/// invented-light lane off without touching the authored ones — the kill switch for a heuristic
/// that, unlike everything around it, is not byte-verified against anything.
#[derive(Resource, Clone, Copy, PartialEq, Debug)]
pub struct FireLightGain(pub f32);

impl Default for FireLightGain {
    fn default() -> Self {
        Self(1.0)
    }
}

/// MONKEY (spellLightGain): marks a `PointLight` that a SPELL EFFECT invented — a kit's aura glow,
/// a missile's core, an impact flash, a firework shell's burst. The app's own
/// `entities::spell_fx::SpellLight` carries the envelope, the mode and the budget; this is the
/// two-word shadow of it the PACKER needs.
///
/// A world-side marker rather than moving `SpellLight` down here, deliberately. `SpellLight` is a
/// LIFECYCLE — it reads `FxDecay` off the effect root, it knows `SpellLightMode::{Kit, Missile,
/// Burst}`, it is aged and evicted by the effect lane's own budget — and every one of those
/// concepts belongs to benilla-app's effect layer, which benilla-world knows nothing about and must
/// not learn. What the packer needs is one bit ("scale this row by `spellLightGain`, not by
/// `fireLightGain`"), so one bit is what crosses the crate boundary; the app inserts it at the one
/// spawn site that makes a spell light (`carried_light::spawn_spell_light_child`), beside the other
/// three markers it already stamps there.
///
/// Every spell light also carries [`SyntheticFireLight`] (it IS an invented source, and the census
/// counts it as one). This marker OVERRIDES that one in the gain fold — see [`build_light_data`].
#[derive(Component)]
pub struct SpellFxLight;

/// MONKEY (spellLightGain): the live gain on every spell-effect light (`spellLightGain`, default
/// 1.0), bridged from benilla-app's cvars exactly as [`FireLightGain`] is, and applied at the same
/// place for the same reason (a dial that needed a respawn to retune is not a dial).
///
/// Separate from `fireLightGain` because the two answer different questions. The fire gain tunes a
/// CONTENT HEURISTIC — "how bright should the light we invented for this campfire prop be" — and
/// its `0` is the kill switch for a lane that is not byte-verified against anything. This one tunes
/// a GAMEPLAY lane: spell lights are short, bright and numerous, they are the one light source that
/// can strobe a room during a fight, and a player who wants combat flashes turned down must not
/// have to put out every hearth in the world to get it.
#[derive(Resource, Clone, Copy, PartialEq, Debug)]
pub struct SpellLightGain(pub f32);

impl Default for SpellLightGain {
    fn default() -> Self {
        Self(1.0)
    }
}

/// MONKEY (torch shadows, Stage B): marks a `PointLight` that exists ONLY to render a cube shadow
/// map (`benilla_app::torch_shadow`'s promoted fixtures). Such a proxy must NOT enter the
/// `wow_light` light table — benilla's receivers would treat it as a real fixture and its nominal
/// intensity (~1000, committed far over gamut) would blast light onto everything near the building,
/// blinking as the proxy repositions. [`build_light_data`]'s gather excludes it; only its Bevy cube
/// map + clusterable entry are wanted.
#[derive(Component)]
pub struct ShadowProxyLight;

/// MONKEY (world shadows): whether the realtime WORLD-shadow lane is active this frame (the
/// `worldShadows` cvar). Set by benilla-app's shadow rig; read by [`build_light_data`], which packs
/// it into the free `sh_c16.w` light lane so `terrain.wgsl` can suppress the baked MCSH terrain
/// shadows ONLY when the world is casting realtime — not merely because a shadow sun exists (the
/// sun also exists for character-only shadows, which must leave MCSH alone). A dedicated resource
/// rather than a `WowLighting` field because `update_time_lighting` wholesale-overwrites that.
#[derive(Resource, Clone, Copy, Default)]
pub struct WorldShadowActive(pub bool);

/// MONKEY (distance slider): the realtime-shadow render distance in yards (the `shadowDistance`
/// slider), bridged from benilla-app's shadow rig. [`build_light_data`] packs it into a free light
/// lane so the receivers (`terrain.wgsl`/`wow_model.wgsl`) fade the realtime shadow at THIS distance
/// rather than a fixed one — the fade must track the cascade's actual `maximum_distance`.
#[derive(Resource, Clone, Copy)]
pub struct ShadowDistance(pub f32);

impl Default for ShadowDistance {
    fn default() -> Self {
        Self(80.0)
    }
}

/// MONKEY (moon shadows): how dark a MOON-shadowed fragment gets at night (`moonShadowStrength`,
/// 0..1, default 0.35), bridged from benilla-app's settings registry exactly as [`ShadowDistance`]
/// is. `0.35` means a fully moon-shadowed fragment keeps `1 − 0.35 = 65 %` of the night sky (ambient + directional)
/// term — a hint of a silhouette, not a daylight-hard shadow.
///
/// A dial rather than a constant because the honest answer to "how much light does a clear full moon
/// throw" is *very little* (the reference client casts none at all), and the interesting range —
/// "just enough that the world is not flat" to "stylised moonlight" — is entirely a taste question
/// that wants to be answered with the scene on screen.
///
/// **`0` is the faithful null.** Every consumer guards on it (the packer refuses to hand the moon a
/// non-zero weight, and the receivers keep their existing `< 0.999` early-out arm), so a night at
/// `moonShadowStrength = 0` renders bit-identically to the build before this feature existed.
#[derive(Resource, Clone, Copy, PartialEq, Debug)]
pub struct MoonShadowStrength(pub f32);

impl Default for MoonShadowStrength {
    fn default() -> Self {
        Self(0.35)
    }
}

/// MONKEY (sun shadow perf): the live `shadowFilter` choice, bridged from benilla-app's shadow rig
/// and extracted into the RENDER world.
///
/// Every other realtime-shadow receiver benilla has is a Bevy `Material` (terrain, `wow_model`), so
/// its pipeline picks the PCF branch up automatically from the view's `ShadowFilteringMethod`. The
/// retained `static_gx` pass is specialized BY HAND, so it has to be told: `static_gx::render`
/// folds this into its pipeline key and pushes the matching `SHADOW_FILTER_METHOD_*` shader def.
/// Without it the two halves of the frame would disagree — the ground filtered one way and the
/// buildings standing on it the other.
///
/// `true` = Gaussian (9 taps, the default look); `false` = Hardware2x2 (1 tap).
#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug, ExtractResource)]
pub struct ShadowFilterGaussian(pub bool);

impl Default for ShadowFilterGaussian {
    fn default() -> Self {
        Self(true)
    }
}

/// **An authored WoW point light source** — an M2 light, a WMO MOLT omni, a carried torch —
/// as the packed light table reads it. This used to be Bevy's `PointLight`, kept purely as a
/// data carrier: [`build_light_data`] was its only reader in the engine, every lit surface
/// takes its lights from the shared table (0273/0285), and no shader here consumes Bevy's
/// clustered lights at all. Bevy nonetheless ran its whole light lane over every one of them
/// each frame — `assign_objects_to_clusters` (a `Vec` rebuilt per frame with a `RenderLayers`
/// clone per light, even with the world camera's `ClusterConfig::None`), `extract_lights`,
/// `prepare_lights`, the light visibility check — a city of lamps' worth of work for nothing,
/// ~2 % of the alone frame on the crowd rig's sampled profile (decision 1945). The fields keep
/// the `PointLight` numbers exactly (`intensity` in the same 4π-scaled units), so the packing
/// below and every recipe are unchanged.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct WorldPointLight {
    /// Linear RGB, hue preserved.
    pub color: [f32; 3],
    /// `4π × authored intensity` — the `PointLight` convention, so the packer's `/(4π)` reads
    /// the authored product back.
    pub intensity: f32,
    /// The ≤3-nearest selection-candidacy radius (yd) — see `terrain_stream::point_light`.
    pub range: f32,
}

/// Main-world resource holding the packed light for this frame; extracted into the render world where
/// [`upload_light`] writes it. Rebuilt every frame by [`build_light_data`] (cheap — one std430 pack).
#[derive(Resource, Clone, Copy, ExtractResource)]
struct WowLightData(LightStd430);

impl Default for WowLightData {
    fn default() -> Self {
        Self(LightStd430 {
            rows: [[0.0; 4]; 21],
            points: [[0.0; 4]; 2 * MAX_POINT_LIGHTS],
        })
    }
}

/// The one persistent storage buffer all materials bind. Created once in [`create_shared_light_buffer`]
/// (main world, so material construction can clone it into the `#[storage(90, …)]` field), then cloned
/// into the render world via `ExtractResource` so [`upload_light`] can write it. `Buffer` clone shares
/// the same GPU resource.
#[derive(Resource, Clone, ExtractResource)]
pub struct SharedLightBuffer(pub Buffer);

/// Wire the shared-light infra into the app. `build_light_data` is chained after the lighting resolve
/// in [`super::LightingPlugin`]; this adds the resource, the extract plugins, the startup buffer
/// creation, and the render-world upload.
pub(super) fn register(app: &mut App) {
    app.init_resource::<WowLightData>()
        // MONKEY (room gate): the per-fixture room claims, packed beside the point table.
        .init_resource::<RoomClaimTable>()
        .init_resource::<WorldShadowActive>()
        .init_resource::<ShadowDistance>()
        // MONKEY (moon shadows): the night directional-shadow strength dial (0 = today's render).
        .init_resource::<MoonShadowStrength>()
        .init_resource::<ShadowHandover>()
        // MONKEY (sun shadow perf): the live PCF choice, extracted for `static_gx`'s hand-rolled
        // pipeline (every Bevy-material receiver keys off the view component instead).
        .init_resource::<ShadowFilterGaussian>()
        .add_plugins(ExtractResourcePlugin::<ShadowFilterGaussian>::default())
        .init_resource::<DynamicInteriors>()
        // MONKEY (fire GO lights): the live gain on synthesised fire lights.
        .init_resource::<FireLightGain>()
        // MONKEY (spellLightGain): and the one on spell-effect lights, which overrides it.
        .init_resource::<SpellLightGain>()
        .init_resource::<super::prop_probes::PropProbeExtract>()
        .add_plugins(ExtractResourcePlugin::<WowLightData>::default())
        .add_plugins(ExtractResourcePlugin::<RoomClaimTable>::default())
        .add_plugins(ExtractResourcePlugin::<SharedLightBuffer>::default())
        .add_plugins(ExtractResourcePlugin::<super::prop_probes::PropProbeExtract>::default())
        // PostUpdate, **after transform propagation**: the point table is packed from each light's
        // `GlobalTransform`, and a CARRIED light (0587 — the torch in an NPC's hand) is a child of a
        // moving joint, so its global is only correct once `Propagate` has run. Packed from `Update`
        // it read the PREVIOUS frame's pose — the pool rubber-banded behind a walking bearer, and a
        // freshly spawned light packed one frame at the world origin. A world-baked doodad light
        // never moves, which is why this was invisible until entities started carrying lights.
        .add_systems(
            PostUpdate,
            // MONKEY (light lane by position): the lane classifier is chained BEFORE the packer, so
            // a light that has stood for a frame is packed on this frame's verdict. Its writes go
            // through `Commands`, so a NEWLY spawned light is still packed on the fail-safe
            // fallback for one frame — see that fallback's note in `build_light_data`.
            (classify_light_lanes, build_light_data)
                .chain()
                .after(bevy::transform::TransformSystems::Propagate)
                .after(super::update_time_lighting),
        )
        // After the spawners (PostUpdate): publish the probe table for extraction on change.
        .add_systems(PostUpdate, super::prop_probes::publish_prop_probes);
    // Guarded like every other render-side registration in the tree: a headless build (no GPU,
    // `backends: None`) has no render app, and the schedule tests build the engine that way.
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.add_systems(
            Render,
            (upload_light, super::prop_probes::upload_prop_probes)
                .in_set(RenderSystems::PrepareResources),
        );
    }
}

/// Create the single persistent storage buffer. `RenderDevice` is a main-world resource (inserted in
/// `RenderPlugin::finish`, available from `Startup` on), so the `assets` foundation builds it alongside
/// `WorldAssets` (which stores a clone so `model_material` can hand it to every model) and inserts the
/// returned resource (cloned into the render world for [`upload_light`]). `STORAGE | COPY_DST` (storage
/// binding + per-frame `write_buffer`).
pub fn new_shared_light_buffer(device: &RenderDevice) -> SharedLightBuffer {
    SharedLightBuffer(device.create_buffer(&BufferDescriptor {
        label: Some("wow_shared_light"),
        size: light_blob_bytes(),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }))
}

/// The full byte size of the shared light BUFFER: the per-frame blob ([`LightStd430`] — 19 header
/// rows + the point-light table) PLUS the interior-prop probe region PLUS the skin-palette
/// regions (rig slot table + tint table + rig-origin table + mat-anim table + straddle clip
/// table + palette rows — decisions 0720/0812/0974/1381/2188) at the tail. **Every buffer bound as
/// `wow_light` must be at least this big** — `wow_model.wgsl` declares the whole layout,
/// and wgpu validates bound size against the shader's struct at draw time. The portrait booth's
/// frozen studio-light buffer sizes itself with this (its table regions stay zeroed ⇒ no scene
/// point lights and black probes on portraits — the studio look is deliberately static); a
/// booth's PALETTE and ORIGIN regions are live, kept written by `rig_palette`'s mirror registry.
pub fn light_blob_bytes() -> u64 {
    per_frame_blob_bytes()
        + (7 * MAX_PROP_PROBES * 16) as u64
        + crate::rig_palette::palette_regions_bytes()
}

/// Byte size of the per-frame prefix alone (= the probe region's offset — see
/// `prop_probes::prop_probe_region_offset`).
pub(super) fn per_frame_blob_bytes() -> u64 {
    std::mem::size_of::<LightStd430>() as u64
}

/// Pack the resolved [`WowLighting`] (+ the global fog-disable toggle and the view farclip) into the
/// std430 blob. The `.w` lanes carry the faithful invariants the shaders expect (Mod2x 1.0, clamp on,
/// terrain shininess 20, fog-enable, farclip wall); the model SH coeffs and both water swatches are
/// derived here once per frame (they used to be recomputed + pushed per-material in `apply_wow_lighting`).
/// MONKEY (night fade): realtime-shadow day strength from the celestial sun's height
/// (`sin(elevation)`): 0 at or below the horizon, ramping to 1 by ~12° so shadows fade out at dusk
/// and in at dawn. A smoothstep for a soft knee rather than a hard switch at the horizon.
/// MONKEY (moon shadows): `pub` because the SHADOW RIG needs the same verdict the packer reaches.
/// The rig (`benilla_app::shadow_core::manage_rig`) re-aims the one directional light at the moon
/// exactly when this is 0, and the packer hands the moon a weight exactly when this is 0 — two
/// copies of that threshold would be two chances to disagree about which body the map holds.
/// (`blob_shadow.rs` keeps its own three-line mirror, pinned by its own test; that one predates
/// this export and is deliberately self-contained.)
pub fn sun_shadow_strength(sun_height: f32) -> f32 {
    let t = (sun_height / 0.208).clamp(0.0, 1.0); // 0.208 ≈ sin(12°)
    t * t * (3.0 - 2.0 * t)
}

/// MONKEY (moon shadows): the NIGHT half of the one shadow rig's HAND-OVER LAW, in `[0,1]`.
///
/// `sun_height` / `moon_height` are `celestial_dir.y` / `moon_dir.y` — the SINE of each body's
/// elevation, Bevy space. Returns how much the moon is allowed to cast, BEFORE
/// [`MoonShadowStrength`] scales it.
///
/// **Why a hard gate and not a crossfade.** benilla's receivers sample ONE shadow-mapped
/// directional light (the loop in `shadow_hook.wgsl` ASSIGNS — last light wins), so at any instant
/// the map holds exactly one body's depth. A weight that overlapped the sun's would be a weight
/// applied to the WRONG body's map for the duration of the overlap. So the moon's weight is
/// strictly zero while the sun still has any, and the rig re-aims in the window where BOTH are
/// zero — `sun_shadow_strength` is exactly 0 at and below the horizon, and this is exactly 0 until
/// the moon clears the same 12° ramp.
///
/// **The gate never actually clips anything**, which is what makes it continuous rather than a
/// step: with the shipped `DayNight` tables the celestial sun crosses the horizon at ≈20:30 and the
/// white moon does not clear it until ≈22:17, so `sun_w` and the moon's elevation ramp are never
/// both positive (asserted over the whole game day by
/// `the_two_shadow_weights_never_overlap_and_neither_jumps`). The gate is therefore an INVARIANT
/// guard, not a shaping term: it keeps the single-light rule true even if a zone or a retuned table
/// ever put the two in the sky together.
///
/// The elevation ramp is [`sun_shadow_strength`]'s own curve, evaluated on the moon's height —
/// deliberately the same smoothstep over the same 0..sin(12°) band, so a body low on the horizon
/// casts nothing (which is also what keeps a moonrise shadow from stretching to infinity; the RIG
/// clamps the basis at 18° for the same reason, `shadow_core::MIN_SHADOW_SUN_ELEVATION`).
pub fn moon_shadow_weight(sun_height: f32, moon_height: f32) -> f32 {
    if sun_shadow_strength(sun_height) > 0.0 {
        return 0.0;
    }
    sun_shadow_strength(moon_height)
}

/// MONKEY (moon shadows): one map, one acknowledged body. The app requests the current clock's
/// body before transform propagation, acknowledges an actual Transform write, then holds zero
/// through the next frame. Both receivers and blobs read `weight`, never the desired clock weight.
/// This makes a time jump / midnight enable as safe as the naturally dark dawn/dusk interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShadowBody {
    Sun,
    Moon,
}

#[derive(Resource, Default)]
pub struct ShadowHandover {
    pub aimed: Option<ShadowBody>,
    pub wanted: Option<ShadowBody>,
    pub weight: f32,
    settling: bool,
    ramp: f32,
}

impl ShadowHandover {
    /// Called ONCE per frame, after lighting resolve and the settings bridge. The rig runs before
    /// Propagate; an acknowledgement from its preceding invocation has therefore propagated now.
    pub fn request(&mut self, sun: f32, moon: f32, strength: f32, active: bool, dt: f32) {
        let wanted = active.then_some(if strength > 0.0 && sun <= 0.0 {
            ShadowBody::Moon
        } else {
            ShadowBody::Sun
        });
        if wanted != self.wanted || !active {
            self.wanted = wanted;
            self.ramp = 0.0;
            self.weight = 0.0;
        }
        if !active {
            self.aimed = None;
            self.settling = false;
            return;
        }
        if self.aimed != wanted {
            self.weight = 0.0;
            return;
        }
        if self.settling {
            self.settling = false;
            self.weight = 0.0;
            return;
        }
        // Fixed envelope: 1.5 seconds from zero to full, independent of frame rate/strength.
        self.ramp = (self.ramp + dt.max(0.0) / 1.5).min(1.0);
        self.weight = match wanted {
            Some(ShadowBody::Sun) => pack_shadow_lane(sun * self.ramp, 0.0),
            Some(ShadowBody::Moon) => pack_shadow_lane(0.0, moon * strength.clamp(0.0, 1.0) * self.ramp),
            None => 0.0,
        };
    }

    /// Only the rig may acknowledge; choosing a body without writing a transform is not ready.
    pub fn aim_written(&mut self, body: ShadowBody) {
        if self.aimed != Some(body) {
            self.aimed = Some(body);
            self.settling = true;
            self.ramp = 0.0;
            self.weight = 0.0;
        }
    }
}

/// MONKEY (moon shadows): the three realtime-shadow dials [`build_light_data`] packs, as ONE
/// system param.
///
/// A bundle rather than three `Res` arguments because the packer had reached Bevy's **16-param
/// ceiling** — `MoonShadowStrength` was the seventeenth, and the failure is not a helpful one
/// (`.chain()` reports that a trait bound is unsatisfied on the system TUPLE, naming neither the
/// system nor the limit). Grouping them is also the honest shape: all three are bridged from the
/// app's shadow rig, all three ride packed lanes, and a fourth shadow dial now has somewhere to go
/// that does not cost the next reader an afternoon.
#[derive(SystemParam)]
pub(super) struct ShadowLanes<'w> {
    /// MONKEY (world shadows): the `worldShadows` lane flag, packed into `sh_c16.w` for the MCSH gate.
    world_active: Res<'w, WorldShadowActive>,
    /// MONKEY (distance slider): the realtime-shadow render distance, packed for the edge fade.
    distance: Res<'w, ShadowDistance>,
    /// MONKEY (moon shadows): the acknowledged signed weight, shared with the rig and blob.
    handover: Res<'w, ShadowHandover>,
}

/// MONKEY (moon shadows): the CPU half of the `fog_params.z` pack — ONE SIGNED lane carrying both
/// directional-shadow weights.
///
/// **There is still no free f32** ([`LightStd430`] is 8528 B, pinned by tests and mirrored by three
/// shaders plus the portrait booth — see [`DAYLIGHT_LANE_SCALE`] for the full accounting). The
/// daylight floor took `wmo_fog_params.w`'s fraction and the bake floor took `sh_c16.w`'s, so the
/// two lanes with spare RANGE are spent. This one needs neither: the hand-over law above guarantees
/// the two weights are **never both non-zero**, so they can share a single lane by SIGN — the
/// cheapest possible packing, exact in both directions, with no quantisation and no cliff.
///
/// `+w` = the sun's weight, `-w` = the moon's (already scaled by [`MoonShadowStrength`]), `0` =
/// neither (the ≈20:30-22:17 window when the sun has set and the moon has not risen — and every
/// frame of a build with `moonShadowStrength 0`).
///
/// **The shader end** decodes with `max(z, 0)` / `max(-z, 0)` (`shadow_hook.wgsl`'s `sun_shadow_w`
/// / `moon_shadow_w`). Every pre-existing reader of this lane is either one of those two calls or
/// already clamped — `clamp(1 - z, 0, 1)` (the three `ext_night_w` decodes) returns 1 for any
/// negative `z`, which is the same 1 it returned for the 0 that used to be there, so the exterior
/// torch lane is bit-identical under the new sign. **Keep in sync with those two WGSL functions.**
pub fn pack_shadow_lane(sun_w: f32, moon_w: f32) -> f32 {
    debug_assert!(
        !(sun_w > 0.0 && moon_w > 0.0),
        "the shadow rig holds ONE body's map: sun {sun_w} and moon {moon_w} cannot both cast"
    );
    sun_w - moon_w
}

/// MONKEY (moon shadows): the SHADER's two decodes, transcribed. Only exist for the round-trip test
/// below; the real decodes live in `shadow_hook.wgsl`.
#[cfg(test)]
fn unpack_shadow_lane(w: f32) -> (f32, f32) {
    (w.max(0.0), (-w).max(0.0))
}

/// MONKEY (enclosed day floor): how [`DynamicInteriors::daylight`] rides to the shader — as the
/// FRACTIONAL part of the packed `wmo_fog_params.w` lane, `w = 1 + debug + daylight * this`.
///
/// **There was no free f32 left.** [`LightStd430`] is 8528 B, mirrored by three shaders plus the
/// portrait booth, and must not grow; every `.w` in rows 0..=20 is spoken for (the Mod2x/clamp/SH
/// enables, terrain shininess, fog enable, farclip, the ambient DC lanes, `sh_c13_*.w` which the
/// SH `dot(row, quad)` reads as a real band, the world-shadow flag, the shadow distance, the
/// exposure). The tail regions are declared only by `wow_model.wgsl`, so `static_gx` — the one
/// consumer that needs this number — cannot reach them at all.
///
/// So it rides a lane that has spare RANGE rather than a spare slot. `wmo_fog_params.w` is packed
/// as `0` (interior lane off) or `1 + interiorDebug` (0..=4), and every one of its four decodes
/// across the three shaders is insensitive to a fraction below 0.5:
///   * `w > 0.5` — the on/off test (`static_gx` x2, `wow_model`): adding to `1 + debug` cannot
///     reach it from below, and cannot leave it from above.
///   * `u32(max(w - 1, 0) + 0.5)` — the debug decode (`static_gx` x3, `wow_model`, `terrain`):
///     `debug + f + 0.5` truncates to `debug` for every `f` in `[0, 0.5)`.
/// **0.49** keeps a clear margin under that 0.5 cliff while spending the whole of the rest of the
/// range on the value. `1 + debug` is an exact integer in f32, so the shader's `fract(w)` returns
/// `daylight * 0.49` bit-for-bit up to the f32 ulp at `w <= 5.49` (~5e-7 — four orders of magnitude
/// below anything the eye can see in an ambient floor). **Keep in sync with `static_gx.wgsl`'s
/// `DAYLIGHT_LANE_SCALE`.**
///
/// **The default (0.12) is calibrated, not chosen.** At the owner's live cvars (`interiorExposure 4`,
/// `interiorAmbient 0.02`, `interiorGain 0.7`) the room law's floor is `0.02 x 0.7 = 0.014`, so an
/// unlit interior fragment renders `1 - exp(-0.014 x 4) = 0.055`. Elwynn at 11:28 renders its
/// sunlit threshold at luminance **0.831** (`ambient (103,129,154)/255 + diffuse (255,133,0)/255 x
/// N.L 0.5805`, clamped). With the floor: `1 - exp(-(0.014 + 0.12) x 4) = 0.415`, i.e. **50 % of
/// the threshold** — a soft step across a doorway instead of a black hole. (0.10 reads 44 %, 0.14
/// reads 55 %; the 0.35 the brief suggested reads **92 %**, which is a room with no walls.)
pub const DAYLIGHT_LANE_SCALE: f32 = 0.49;

/// MONKEY (bake floor): how [`DynamicInteriors::bake_floor`] rides to the shader — as the
/// FRACTIONAL part of the packed `sh_c16.w` lane, `w = world_shadow_flag + bake × this`.
///
/// **There is still no free f32.** [`LightStd430`] is 8528 B and mirrored by three shaders plus the
/// portrait booth (see [`DAYLIGHT_LANE_SCALE`] for the full accounting of why it must not grow).
/// `wmo_fog_params.w` — the lane the daylight floor rides — is now spent: its integer part is
/// `1 + interiorDebug` and its fraction is `interiorDaylight × 0.49`, and two fractions cannot
/// share one lane without a second quantisation. So this one takes the OTHER lane with spare range.
///
/// `sh_c16.w` is the world-shadow flag: packed as exactly `0.0` or `1.0`, and it has precisely ONE
/// decode in the whole shader set — `terrain.wgsl`'s `wow_light.sh_c16.w > 0.5` (the MCSH
/// suppression gate). `sh_c16.xyz` is a real SH band that `wow_model`/`static_gx` read; neither
/// read `.w` at all before this, and both now read only its FRACTION. A fraction strictly below
/// 0.5 therefore cannot move terrain’s decode in either direction: `0 + f` stays below the
/// threshold, `1 + f` stays above it.
///
/// **0.49** keeps the same clear margin under the 0.5 cliff that the daylight lane keeps, and the
/// packed value is CLAMPED to `[0, 1]` before scaling — `bake_floor` is 0..1 but `interior_gain`
/// reaches 1.5, so the product alone could reach 1.5 and push the fraction past the cliff (0.735),
/// which would switch MCSH terrain shadows on for anyone running `interiorBakeFloor 1` with the
/// Bright preset. The clamp costs nothing real: the slider's useful range is 0..0.3.
///
/// `world_shadow_flag` is an exact integer in f32, so the shader's `fract(w)` returns
/// `bake × 0.49` bit-for-bit up to the f32 ulp at `w <= 1.49` (~6e-8). **Keep in sync with
/// `static_gx.wgsl` / `wow_model.wgsl`'s `BAKE_LANE_SCALE`.**
pub const BAKE_LANE_SCALE: f32 = 0.49;

/// MONKEY (bake floor): the CPU half of the `sh_c16.w` pack — the lane word for a world-shadow
/// flag and a bake floor. Split out so the round-trip test below exercises the SAME arithmetic the
/// packer runs, not a transcription of it.
pub fn pack_bake_lane(world_shadow: bool, bake_floor: f32, interior_gain: f32) -> f32 {
    let flag = if world_shadow { 1.0 } else { 0.0 };
    flag + (bake_floor * interior_gain).clamp(0.0, 1.0) * BAKE_LANE_SCALE
}

/// MONKEY (bake floor): the SHADER's decode, transcribed — `fract(w) / BAKE_LANE_SCALE`. Only
/// exists for the round-trip test; the real decode lives in the two WGSL files.
#[cfg(test)]
fn unpack_bake_lane(w: f32) -> f32 {
    (w - w.floor()) / BAKE_LANE_SCALE
}

/// MONKEY (dynamic interiors): the live knobs of the interior lane — WMO interior surfaces AND
/// interior props light from the room's live fixtures (`static_gx.wgsl` `interior_room_light`)
/// instead of the MOCV bake / the baked prop probe. Bridged every frame from the app-side cvars
/// (`interiorLight`, `interiorAmbient`, `interiorFill`, `interiorExposure` — benilla-app's
/// `dynamic_interior` module) the way [`ShadowDistance`] is, so the look is tunable in-game.
/// `enabled == false` → the faithful baked path. Packed into the free `wmo_fog_params.w` (on/off)
/// and `point_count.yzw` (the three knobs) lanes.
#[derive(Resource, Clone, Copy, PartialEq, Debug)]
pub struct DynamicInteriors {
    /// The lane on/off (`interiorLight`).
    pub enabled: bool,
    /// Base ambient every interior fragment gets, so a fixture-less nook never goes black.
    pub ambient: f32,
    /// Bounce gain per fixture — the normal-free "lamps everywhere" glow the bake carried.
    pub fill: f32,
    /// Multiplier on the whole light budget before the soft rolloff — the room's brightness.
    pub exposure: f32,
    /// MONKEY (soft falloff): live scale on every interior fixture's authored attenuation window
    /// (`interiorAttenScale`) — the fixture's EFFECTIVE RADIUS is `authored end × this`. Default
    /// **2.5**: the artists' end is where FULL brightness stops, not where light does, so `1` drew
    /// a hard-edged disc at exactly that radius. `>2.5` widens the pools, `<2.5` tightens them,
    /// **`0` disables the window** ([`interior_reach`]). Folded into the packed lane rather than a header lane: the
    /// window is per-fixture anyway, the header's 21 rows are fully claimed, and the fold keeps the
    /// shader from carrying a second knob it would have to combine per light.
    pub atten_scale: f32,
    /// MONKEY (room gate): the per-room fixture gate (`interiorRoomGate`). `false` packs every
    /// fixture UNGATED, i.e. the pre-gate behaviour where every INT fixture lights every interior
    /// surface in range — the live A/B for "is the gate what darkened this room?".
    pub room_gate: bool,
    /// MONKEY (interior debug): diagnostic overlay for the interior lane (`interiorDebug`). 0 = off
    /// (normal shading). 1 = CLASSIFICATION: interior-lit fragments render solid green, so a wrongly
    /// interior-classified OUTDOOR entity shows up. 2 = SHADOW: the torch-shadow factor as greyscale
    /// (black = shadowed, white = lit) — shows whether cast shadows are computed at all. 3 = CASTER
    /// COUNT: how many shadow-casting proxies the fragment sees in `clusterable_objects` (red = 0 —
    /// the proxies never reached the shader; green = 1; blue = 2; white = 3+).
    /// MONKEY (ext-class night law): 4 = WMO LANE MAP — every WMO fragment painted by the lighting
    /// law its GROUP falls under (green interior-class, blue exterior-class at building scale, red
    /// exterior-class shell/district), terrain and models untouched. The instrument for "is this
    /// seam a LAW mismatch (two colours) or a CLAIM mismatch (one colour)?".
    /// Packed with the on/off flag into `wmo_fog_params.w` as `1 + debug` (so `>0.5` still means
    /// "on").
    pub debug: u32,
    /// MONKEY (flame flicker): live gain on every flame's brightness wobble (`fireFlicker`, 0..2).
    /// `1` = the authored per-kind amplitudes ([`FlameKind::amplitude`]), `0` = steady constants
    /// (the pre-feature look, and the escape hatch if a flicker ever reads wrong), `2` = doubled.
    ///
    /// It rides THIS resource rather than a header lane because it is consumed entirely on the CPU:
    /// [`build_light_data`] folds the multiplier into the packed colour, so the shader never learns
    /// the feature exists and `LightStd430` gains no lane. Also why it is not on
    /// [`FireLightGain`]: that one scales the SYNTHESISED lane only, while a flicker belongs to
    /// authored wall torches just as much.
    pub flicker: f32,
    /// MONKEY (darkness gains): the live dim on the EXTERIOR day/night law (`nightGain`, 0.2..1.5,
    /// default **0.8** = nights 20 % darker). Folded into the packed ambient/diffuse/specular rows
    /// by `mix(1, gain, night_w)`, `night_w = 1 - sun_shadow_strength(celestial_dir.y)` — so it is
    /// EXACTLY inert while the sun is up and full strength after dark.
    ///
    /// It rides THIS resource despite being an exterior knob because the bridge is the same one
    /// (a live video cvar → a resource the light packer folds at pack time), and a second resource
    /// plus a second plugin to carry one `f32` buys nothing. `enabled` does NOT gate it: the night
    /// law lights terrain and models whether or not the interior lane is on.
    pub night_gain: f32,
    /// MONKEY (darkness gains): the live dim on the whole INTERIOR room lane (`interiorGain`,
    /// 0.2..1.5, default **0.7** = interiors 30 % darker). Scales the lane's three INPUTS — the
    /// packed [`Self::ambient`], [`Self::fill`], and every interior fixture's committed colour —
    /// and deliberately not [`Self::exposure`], which is the user's own live dial; a gain on the
    /// inputs composes with whatever exposure they have settled on.
    pub interior_gain: f32,
    /// MONKEY (enclosed day floor): the DAYLIGHT a room in a building gets through the doorways
    /// this renderer cannot locate (`interiorDaylight`, 0..1, default **0** — the director wants daylight ONLY at doorways/windows, i.e. from the daylight fixtures; the room-wide floor stays available as a dial).
    ///
    /// It is an ADDITIVE ambient in exactly [`Self::ambient`]'s units — the room law's pre-exposure
    /// illumination — scaled by the sun's own day envelope and tinted by the sky's ambient band, so
    /// it is 0 all night and full at midday. Three seeds now find a building's authored openings
    /// (`lighting::daylight`), but the shipped corpus also contains rooms whose doorway is in NO
    /// table: the Goldshire inn's entry group `g3` has no portal, no EXT-class batch, no vertex
    /// stitched to the shell that owns its threshold planks, and no localized spot in its own MOCV
    /// bake. For those there is nothing to stand a fixture in, and the honest fallback is to say
    /// what IS known — the room is inside a building, the sun is up, so it is not pitch dark.
    ///
    /// Rides the free FRACTION of the packed `wmo_fog_params.w` lane (see
    /// [`DAYLIGHT_LANE_SCALE`]); `0` restores the pre-feature look exactly.
    pub daylight: f32,
    /// MONKEY (bake floor): the share of a WMO interior batch's OWN MOCV bake that every
    /// interior-lane fragment keeps, whether or not a fixture reaches it (`interiorBakeFloor`,
    /// 0..1, default **0.12**).
    ///
    /// The lane's premise — "the live fixtures decide, the bake's LEVEL is wrong" — has one hole
    /// in it: a room the fixture table cannot reach renders at the bare [`Self::ambient`] floor,
    /// i.e. black. The Lion's Pride Inn's east vestibule (group `g0`, box x 14.1..20.5) is the case
    /// that forced it — MOLR 0, ZERO fixture claims (nearest L2 ≈ 16 yd against R 11.2, L3 ≈ 17.6
    /// against R 14.7), one faded portal hop, so the whole budget collapses to `0.0075` and the
    /// door band renders 0.019 × tex between a sky-lit porch and a candle-lit hall. The reference
    /// client has no such hole: it draws every interior batch at its authored MOCV regardless of
    /// fixtures, so a fixture-starved room is DIM there, never black.
    ///
    /// So the honest floor is the bake ITSELF, at a fraction: `vc.rgb × this`, added to the room's
    /// pre-exposure budget INSIDE the `1 − exp(−x·exposure)` rolloff, so it saturates with the
    /// fixtures instead of stacking on top of them (a candle-lit surface barely moves — measured
    /// +7 % at the inn's `g3` floor under `L9`, +9 % at a `g5` wall 3 yd from `L0`) while an unlit
    /// one goes from black to the bake's own relative statement about the room. A FRACTION, not the
    /// bake, precisely because the LEVEL is what this lane rejects: an UNCAPPED bake share on this
    /// same band measured 15-18 × its neighbour and read as a flat grey slab (see the
    /// `static_gx.wgsl` portal-bleed comment); 0.12 restores about an eighth of it.
    ///
    /// Scaled by [`Self::interior_gain`] at PACK time (so the Dim preset dims it with everything
    /// else and the shader carries no second knob), and NOT by `fireLightGain`, NOT flickered — it
    /// is not a fire, it is the room's own authored light. `0` restores the pre-feature look
    /// exactly. Rides the free FRACTION of the packed `sh_c16.w` lane — see [`BAKE_LANE_SCALE`].
    pub bake_floor: f32,
}

impl Default for DynamicInteriors {
    fn default() -> Self {
        Self {
            enabled: true,
            ambient: 0.015,
            fill: 0.08,
            exposure: 2.5,
            // MONKEY (soft falloff): 1.6, matching the `interiorAttenScale` cvar default (2.5 read oversaturated — the soft core `0.26·R` widens with it too).
            atten_scale: 1.6,
            // MONKEY (room gate): on by default — without it a building's fixtures light through
            // its own floors and walls.
            room_gate: true,
            debug: 0,
            // MONKEY (flame flicker): on at the authored amplitudes.
            flicker: 1.0,
            // MONKEY (darkness gains): the director's call — nights 20 % darker, interiors 30 %.
            night_gain: 0.45,
            interior_gain: 0.5,
            // MONKEY (enclosed day floor): calibrated, not chosen — see [`DAYLIGHT_LANE_SCALE`]
            // for the arithmetic that lands the Goldshire inn's entry floor at ~half the sunlit
            // threshold beside it instead of at the near-black `interiorAmbient`.
            daylight: 0.0,
            // MONKEY (bake floor): an eighth of the authored bake — see the field doc for the
            // measurement this is calibrated against (the inn's `g0` door band, and the
            // candle-lit surfaces it must NOT move).
            bake_floor: 0.12,
        }
    }
}

/// MONKEY (room gate): one packed fixture's room claim — the CPU form of the eight u32
/// `static_gx.wgsl` reads (`[instance, count, g0..g5]`). `count == 0` is the UNGATED head every
/// fail-open arm packs: an exterior-lane light, a fixture that names no room, and every light
/// while `interiorRoomGate` is off. MONKEY (GO room claims): an OVERFLOWING list no longer fails
/// open — it keeps its [`ROOM_CLAIM_MAX`] highest-priority claims (the spawner orders them
/// tightest-containment → MOLR → nearest portal hop, so the dropped tail is the outermost shells /
/// farthest rooms). Failing open here lit every room of the building through its walls, and a
/// carried light inside nested MOGI shells can now reach the cap where a MOLT fixture never did.
#[derive(Clone, Copy)]
struct RoomClaim {
    instance: u32,
    n: u8,
    groups: [u16; ROOM_CLAIM_MAX],
    /// MONKEY (soft portal claims): index-parallel with `groups`. A default (`radius == 0`) is the
    /// HARD claim every pre-fade path produces, so a fixture whose claims arrived without fades
    /// gates exactly as it did before.
    fades: [ClaimFade; ROOM_CLAIM_MAX],
}

impl Default for ClaimFade {
    /// The HARD claim: no doorway, no fade, full weight. Spelled out rather than derived because
    /// `entry` must default to **1**, not 0 — a derived zero would mean "this claim contributes
    /// nothing", i.e. every fallback path would silently black out its rooms.
    fn default() -> Self {
        Self {
            center: Vec3::ZERO,
            slack: 0.0,
            radius: 0.0,
            entry: 1.0,
        }
    }
}

impl RoomClaim {
    const UNGATED: Self = Self {
        instance: 0,
        n: 0,
        groups: [0; ROOM_CLAIM_MAX],
        fades: [ClaimFade {
            center: Vec3::ZERO,
            slack: 0.0,
            radius: 0.0,
            entry: 1.0,
        }; ROOM_CLAIM_MAX],
    };

    /// The fixture's claim list. MONKEY (portal claims): when the spawner built one
    /// ([`LightLitRooms`]) it IS the answer — it already folds the MOLR rooms in, in priority order,
    /// with each claim's exterior-lane eligibility resolved. Only a light that carries no such list
    /// (a carried torch, a transport prop) falls back to its raw MOLR rooms. MONKEY (review fixes):
    /// each list carries its own placement, independently of whether MOLR names the fixture.
    fn build(rooms: Option<&LightRooms>, lit: Option<&LightLitRooms>) -> Self {
        // MONKEY (soft portal claims): the fades ride the SAME arm the groups came from — the
        // MOLR fallback has none, and reading them off `lit` while the groups came off `rooms`
        // would fade one room by another room's doorway.
        let lit = lit.filter(|l| !l.rooms.groups.is_empty());
        let fades: &[ClaimFade] = lit.map_or(&[], |l| &l.fades[..]);
        let Some(claims) = lit.map(|l| &l.rooms).or_else(|| rooms.map(|r| &r.0)) else {
            return Self::UNGATED; // nothing claims this light: it lights everything, as before
        };
        let instance = claims.instance.index().index();
        // The identity travels to the shader through an f32 lane (`static_gx::render`'s cell
        // uniform), which is exact only below 2^24 — and index 0 is the "terrain cell / no
        // building" sentinel there. Outside that range the receiving side packs no key at all, so
        // this side must not gate either, or the building would go black.
        if instance == 0 || instance >= (1 << 24) {
            return Self::UNGATED;
        }
        let mut out = Self {
            instance,
            ..Self::UNGATED
        };
        for (i, &g) in claims.groups.iter().enumerate() {
            if out.groups[..usize::from(out.n)].contains(&g) {
                continue;
            }
            if usize::from(out.n) == ROOM_CLAIM_MAX {
                break; // overflow keeps the six highest-priority claims (see the struct doc)
            }
            out.groups[usize::from(out.n)] = g;
            // A missing fade is a HARD claim, not a dropped one: the fallback paths (a carried
            // light's raw MOLR rooms) legitimately have none, and the pre-fade behaviour is
            // exactly "weight 1 wherever the group matches".
            out.fades[usize::from(out.n)] = fades.get(i).copied().unwrap_or_default();
            out.n += 1;
        }
        out
    }

    /// Write the eight-u32 GPU form. Groups are stored `+ 1` so the shader can keep 0 for "empty"
    /// (and for "this fragment names no room") — group 0 is a real, common group id.
    ///
    /// MONKEY (portal claims): [`LIT_ROOM_EXT_DENY`] (bit 15 of the CPU id) is re-encoded as
    /// [`CLAIM_EXT_OK`] (bit 16 of the GPU word), positively: the exterior batch lane takes a claim
    /// only with that bit set, so every legacy/fallback path that never learned about the flag
    /// (a carried light's raw MOLR rooms) reads as "not eligible" rather than as a leak.
    fn write(&self, dst: &mut [u32]) {
        dst[0] = self.instance;
        dst[1] = u32::from(self.n);
        for (i, slot) in dst[2..2 + ROOM_CLAIM_MAX].iter_mut().enumerate() {
            // Past the count the shader never looks, but a `+1` on a padding zero would read as a
            // claim on group 0 to anything that ever did — so the padding stays literally empty.
            *slot = match self.groups.get(i).filter(|_| i < usize::from(self.n)) {
                Some(g) => {
                    // MONKEY (soft portal claims): the entry weight rides the spare high bits.
                    let entry = self.fades[i].entry.clamp(0.0, 1.0);
                    u32::from(*g & !LIT_ROOM_EXT_DENY) + 1
                        | if *g & LIT_ROOM_EXT_DENY == 0 {
                            CLAIM_EXT_OK
                        } else {
                            0
                        }
                        | ((entry * 255.0).round() as u32) << CLAIM_ENTRY_SHIFT
                }
                None => 0,
            };
        }
        // MONKEY (soft portal claims): the fade records. Centre as raw f32 bits (the shader
        // `bitcast`s them straight back — a world coordinate reaches +-17000 yd, which no fixed
        // point or f16 lane could carry); radius and slack as `u16` yards x `CLAIM_FADE_SCALE` in
        // one word. A padded slot writes zeros, and `radius == 0` IS the hard claim, so the padding
        // can never be read as a fade even by a reader that ignores the count.
        let q = |v: f32| ((v.max(0.0) * CLAIM_FADE_SCALE).round() as u32).min(0xffff);
        for i in 0..ROOM_CLAIM_MAX {
            let f = &mut dst[ROOM_CLAIM_FADE + 4 * i..][..4];
            let Some(fade) = self.fades.get(i).filter(|_| i < usize::from(self.n)) else {
                f.fill(0);
                continue;
            };
            f[0] = fade.center.x.to_bits();
            f[1] = fade.center.y.to_bits();
            f[2] = fade.center.z.to_bits();
            f[3] = q(fade.radius) | (q(fade.slack) << 16);
        }
    }
}

impl std::fmt::Display for RoomClaim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.n == 0 {
            return write!(f, "UNGATED");
        }
        write!(f, "inst {} g", self.instance)?;
        for (i, g) in self.groups[..usize::from(self.n)].iter().enumerate() {
            write!(f, "{}{g}", if i == 0 { "" } else { "," })?;
        }
        Ok(())
    }
}

#[allow(clippy::type_complexity)]
fn build_light_data(
    light: Res<WowLighting>,
    debug: Res<DebugState>,
    view: Res<ViewDistance>,
    cam: Query<&GlobalTransform, With<WorldCamera>>,
    // `Without<ShadowProxyLight>`: the torch-shadow proxies are `WorldPointLight`s too, but they
    // exist only to cast a cube map — never to light benilla's receivers (see [`ShadowProxyLight`]).
    // MONKEY (fire GO lights): `Has<SyntheticFireLight>` rides along so the live `fireLightGain`
    // can scale exactly the invented sources at PACK time — no respawn, no per-spawn bake.
    // MONKEY (interior attenuation): `Option<&LightReach>` too — a MOLT fixture's authored
    // attenuation end, `None` on every M2 source (which is bucketed below instead).
    // MONKEY (light lane by position): `Option<&LightLane>` decides the interior/exterior split —
    // `LightRooms` stays in the tuple, but only for the portal ROOM gate below (and as the
    // fail-safe lane for the frame or two before the classifier has answered).
    lights_q: Query<
        (
            &WorldPointLight,
            &GlobalTransform,
            Option<&LightRooms>,
            Has<SyntheticFireLight>,
            Option<&LightReach>,
            Option<&LightLane>,
            // MONKEY (room gate): the bbox-derived supplement to the MOLR claim set.
            Option<&LightLitRooms>,
            // MONKEY (flame flicker): present iff this light BURNS ([`flame_kind_for`] decided so
            // at spawn). Read-only, never written — the wobble is a pure function of it plus the
            // clock, so it adds no `Changed` traffic and no archetype churn to the frame.
            Option<&FlameFlicker>,
            // MONKEY (darkness gains): the daylight-fixture marker — an interior-lane entry that
            // is the SUN standing in a doorway, not a candle, so `interiorGain` must skip it.
            Has<super::DaylightFixture>,
            // MONKEY (spellLightGain): the spell-effect marker ([`SpellFxLight`]). Last in the
            // tuple so every positional destructuring below keeps its index — and read for one
            // thing only: WHICH live gain owns this row.
            Has<SpellFxLight>,
        ),
        Without<ShadowProxyLight>,
    >,
    // The per-frame portal PVS, for the room term below ([`LightRooms`]).
    portals: Query<&crate::wmo_portal::WmoPortalInstance>,
    mut data: ResMut<WowLightData>,
    // MONKEY (room gate): packed in the same walk as the point table, so the two can never
    // disagree about which entry is which.
    mut claims: ResMut<RoomClaimTable>,
    time: Res<Time>,
    // The three realtime-shadow dials, bundled (see [`ShadowLanes`]).
    shadow: ShadowLanes,
    // MONKEY (dynamic interiors): the interior lane's on/off + live knobs, packed for `static_gx.wgsl`.
    dynamic_interiors: Res<DynamicInteriors>,
    // MONKEY (fire GO lights): the live gain on synthesised fire lights (0 = the lane off).
    fire_gain: Res<FireLightGain>,
    // MONKEY (spellLightGain): and the spell lane's own, which overrides it on a spell row.
    spell_gain: Res<SpellLightGain>,
    mut last_dump: Local<f64>,
    mut last_rows_dump: Local<f64>,
) {
    let l = &*light;
    let fog_enable = if debug.lighting.disable_fog { 0.0 } else { 1.0 };
    let farclip = view.farclip;
    // MONKEY (darkness gains): `nightGain` — one live dim over everything the EXTERIOR law lights
    // (terrain, models, WMO exteriors, and the ext-class night blend, all of which derive from the
    // ambient/diffuse/specular rows below). Folded CPU-side into those packed rows rather than
    // added as a shader uniform because `LightStd430` has no free lane left (8528 B, mirrored by
    // three WGSL structs) — and a pack-time fold costs the GPU exactly nothing anyway.
    //
    // The ramp is the DUSK CLOCK every other night feature already fades on: `sun_w` is
    // `sun_shadow_strength(celestial_dir.y)` — 1 in daylight, smoothstepping to 0 as the celestial
    // sun reaches the horizon — so `night_w = 1 - sun_w` is 0 all day and 1 after dark. Written as
    // `1 + (g - 1)·night_w` (== `mix(1, g, night_w)`) so daylight multiplies by an EXACT 1.0 and
    // every daytime frame stays bit-identical to before the feature: a gain that perturbed the day
    // in the last ulp would move the `WOW_LIGHT_DUMP` row hash, which is the instrument three
    // rounds of shading forensics are denominated in.
    //
    // POINT lights are deliberately NOT scaled by it. Dimming the sky law alone is the whole point:
    // a campfire should read BRIGHTER against a darker night, not equally dim.
    let sun_w = sun_shadow_strength(l.celestial_dir.y);
    let night_k = 1.0 + (dynamic_interiors.night_gain - 1.0) * (1.0 - sun_w);
    let night_dim = |c: [f32; 3]| c.map(|v| v * night_k);
    // Per-kind water swatches (shallow/deep rgb + alpha). River/lake use the non-ocean path.
    let (rs, rd, rsa, rda) = l.water_colors(LiquidKind::Still);
    let (os, od, osa, oda) = l.water_colors(LiquidKind::Ocean);
    // Built in a scratch copy and written through `ResMut` only when a row moved: the extract
    // clones this 8.5 KB blob every frame it reads as changed, and a parked frame changes nothing.
    let mut fresh = data.0;
    fresh.rows = [[0.0; 4]; LIGHT_HEADER_ROWS];
    let rows = &mut fresh.rows;
    // MONKEY (darkness gains): the specular row takes the night dim because it IS the sun — the
    // DBC sun-halo colour driving the terrain sheen. The FOG rows below do not (fog colour is the
    // horizon backdrop the world is seen AGAINST, and dimming it would paint a dark world under a
    // bright skyline), and neither does the sky dome (`resolve::apply_sky_backdrop`, which never
    // reads this blob) or the water swatches.
    let spec = night_dim(l.spec);
    rows[3] = [spec[0], spec[1], spec[2], 20.0]; // 3 light_spec (w=terrain shininess 20)
    rows[4] = [l.fog_color[0], l.fog_color[1], l.fog_color[2], fog_enable]; // 4 fog_color (w=enable)
    rows[5] = [l.fog_start, l.fog_end, 0.0, farclip]; // 5 fog_params (z unused; w=farclip)
    rows[13] = [rs[0], rs[1], rs[2], rsa]; // 13 water river shallow (w=alpha)
    rows[14] = [rd[0], rd[1], rd[2], rda]; // 14 water river deep
    rows[15] = [os[0], os[1], os[2], osa]; // 15 water ocean shallow
    rows[16] = [od[0], od[1], od[2], oda]; // 16 water ocean deep
                                           // 17 `.x` — the SIDN night fraction (the windows-glow-at-night ramp: `wow_model.wgsl`
                                           // multiplies each WMO SIDN material's authored emissive colour by it on the lit lanes).
                                           // `.yzw` is the core packer's below.
    rows[17][0] = l.sidn_night;
    // 18/19 — the INTERIOR fog triple (see the layout doc above). 19.zw are free lanes: they
    // carried retired dials (the 0273/0354-era A/Bs, the point gain, the 0750/0751 sun
    // calibration). 12.w was free too until 0796 gave it the response A/B (below).
    rows[18] = [
        l.wmo_fog_color[0],
        l.wmo_fog_color[1],
        l.wmo_fog_color[2],
        fog_enable,
    ];
    rows[19] = [l.wmo_fog_start, l.wmo_fog_end, 0.0, 0.0];
    // Rows 0-2, the SH block 6-12.xyz, and the sun DC (17.yzw) — the shared model-light core
    // (also the portrait booth's packer). Row 20 (point_count) is the point-table pack's below.
    // MONKEY (darkness gains): the night dim goes in HERE, on the (ambient, diffuse) triple the
    // whole model core is derived from, rather than onto the packed rows afterwards — every row
    // this packs (rows 0/1, the SH block, the sun's DC redistribution) is LINEAR in that triple, so
    // one multiply at the input dims all of them consistently and no derived row can be missed.
    pack_model_core_rows(rows, night_dim(l.ambient), night_dim(l.diffuse), l.sun_dir);
    // MONKEY (world shadows): pack the world-shadow lane flag into the free `sh_c16.w` lane. The
    // MCSH terrain-shadow suppression in `terrain.wgsl` keys on THIS — not on the mere presence of
    // a shadow sun — so character-only shadows (sun present, world lane off) keep the baked MCSH.
    // MONKEY (bake floor): ...and `interiorBakeFloor` rides the free FRACTION of that same lane,
    // with `interiorGain` already folded in (the shader must not carry a second knob, and the room
    // lane's other two inputs — the base ambient and the per-fixture fill — take the gain at pack
    // time in exactly this way, three rows down). See [`BAKE_LANE_SCALE`] for why a fraction is
    // invisible to the one `> 0.5` decode this lane has, and why the product is clamped first.
    rows[12][3] = pack_bake_lane(
        shadow.world_active.0,
        dynamic_interiors.bake_floor,
        dynamic_interiors.interior_gain,
    );
    // MONKEY (night fade): realtime-shadow strength by the REAL celestial sun height, packed into
    // the free `fog_params.z` lane. The receivers (terrain/model) lighten their shadow term by it,
    // so shadows soften and vanish at night; the shadow basis is separately clamped to 18° so a low
    // sun still casts the right DIRECTION.
    // MONKEY (darkness gains): computed once above — `nightGain` rides this exact same curve, so
    // the dim and the shadow fade can never drift onto two different dusk clocks.
    // MONKEY (moon shadows): pack the rig's ACKNOWLEDGED weight, not the clock's desired one.
    // The body may have changed abruptly this frame; the shared state holds zero until the new
    // basis has propagated and then ramps. One signed float still keeps the buffer at 8528 bytes.
    rows[5][2] = shadow.handover.weight;
    // MONKEY (distance slider): the realtime-shadow render distance (yd), packed into the free
    // `_wmo_fog[1].z` / `wmo_fog_params.z` lane (row 19). The receivers' edge fade reads it so the
    // shadow fades at the cascade's actual `maximum_distance`, whatever the slider is set to.
    rows[19][2] = shadow.distance.0;
    // MONKEY (dynamic interiors): the lane's on/off into the free `wmo_fog_params.w` lane (1 = WMO
    // interior surfaces + props light from the room's live fixtures, 0 = the faithful baked path).
    // Its three knobs ride `point_count.yzw`, packed with the table below.
    // On/off in the integer part, the debug mode added on top: 0 = off, 1 = on, 1+n = on + debug n.
    // MONKEY (enclosed day floor): …and `interiorDaylight` rides the same lane's FRACTION (see
    // [`DAYLIGHT_LANE_SCALE`] for why there was nowhere else to put it and why every existing
    // decode survives it). Zero when the lane is off, so the packed word is byte-identical to
    // before the feature in that arm.
    rows[19][3] = if dynamic_interiors.enabled {
        1.0 + dynamic_interiors.debug as f32
            + dynamic_interiors.daylight.clamp(0.0, 1.0) * DAYLIGHT_LANE_SCALE
    } else {
        0.0
    };
    // The dynamic point-light table (decision 0278): every spawned point light within
    // [`POINT_PACK_RADIUS`] of the camera, nearest-first when over capacity — the VERTEX stages of
    // `terrain.wgsl`/`wow_model.wgsl` walk it for the Gouraud point term (bevy's clusterable buffer
    // is fragment-only in the view layout, so the lights ride this buffer). Colour = the light's
    // effective linear RGB: bevy stores colour and intensity apart, and the spawn premultiplied 4π
    // (`spawn_point_light`), so `intensity/(4π)` recovers exactly the authored colour × intensity —
    // packed RAW as the reference commits it ([`commit_raw`]: the encode→decode round trip is
    // identity, over-gamut preserved). Entries past `count` stay stale in the blob — the count row
    // guards every reader.
    let cam_pos = cam.single().map(|t| t.translation()).unwrap_or(Vec3::ZERO);
    // MONKEY (flame flicker): the one clock read for the whole table — every flame's phase is a
    // function of this absolute second and its own seed, so the frame is internally consistent and
    // the result does not depend on how long the frame took.
    let now_secs = time.elapsed_secs();
    // MONKEY (fire GO lights): `(…, synthetic)` rides the tuple so the census below can count the
    // invented entries without a second query.
    // MONKEY (light lanes): the tuple's last member is the packed COLOUR-ROW `.w` — 0 for an
    // exterior light, the interior reach in yards for a fixture that claims a room (see the layout
    // doc). It is resolved here, in the one place that already knows both the rooms and the
    // recovered intensity, so neither shader has to reconstruct either.
    // MONKEY (GO room claims): the trailing `usize` is the RAW `LightLitRooms` claim count —
    // what the spawner (or the app's carried-light claimer) built, before the lane/`room_gate`
    // filter below decides whether it is packed at all. Printed by `WOW_POINTS_DUMP` so an
    // EXTERIOR-lane entry that HAS claims is visible as such: the shader reads claims only on
    // the interior lane, so `claims 4` beside `EXT` is the readout of exactly that trade.
    let mut pts: Vec<(f32, Vec3, f32, [f32; 3], bool, f32, RoomClaim, usize)> = lights_q
        .iter()
        .filter(|(_, gt, rooms, _, _, _, _, _, _, _)| {
            // The ROOM term (decision 0689's law, fourth lane — see [`LightRooms`]). Not a
            // visibility test bolted onto a faithful gather: the reference's register walk has no
            // such term either, it simply never has a culled room's torch to register. Ungated for
            // every light that names no rooms, which is every light outside a building.
            //
            // MONKEY (dynamic interiors): a fixture NEAR the camera is admitted regardless of the
            // flood. The PVS follows the camera, so a room's torches dropped out of the table with
            // camera angle — outside its door, or from the next room — and a fixture-LIT interior
            // went dark where the bake never could (the dump flipped 18↔9 packed walking the inn).
            // The wall leak this allows is the reference's own: a drawn room's torches register.
            crate::wmo_portal::room_admits(
                rooms.map(|r| &r.0),
                rooms.and_then(|r| portals.get(r.0.instance).ok()),
            ) || (dynamic_interiors.enabled
                && gt.translation().distance_squared(cam_pos)
                    < INTERIOR_NEAR_ADMIT * INTERIOR_NEAR_ADMIT)
        })
        .filter_map(|(pl, gt, rooms, synthetic, reach, lane_of, lit_rooms, flicker, daylight, spell)| {
            let p = gt.translation();
            let d2 = p.distance_squared(cam_pos);
            (d2 < POINT_PACK_RADIUS * POINT_PACK_RADIUS).then(|| {
                // MONKEY (merge): `WorldPointLight::color` is already linear RGB.
                let c = pl.color;
                // The authored intensity, BEFORE the fire gain: the pool's geometry must not move
                // when the user dims the invented lights, only its brightness.
                let base = pl.intensity / (4.0 * std::f32::consts::PI);
                // MONKEY (fire GO lights): the live `fireLightGain` folds in HERE, over the
                // recovered colour×intensity, and only for a synthesised source. At spawn it
                // would need a world respawn to retune; here the dial moves the frame it changes.
                // MONKEY (spellLightGain): …and the spell lane's own gain INSTEAD of it on a
                // spell row. Every spell light is also tagged synthetic (it is an invented source
                // and the census counts it as one), so the arms must be ordered, not summed: a
                // fireball scaled by both dials would darken when the player turned the hearths
                // down, which is precisely the coupling the second cvar exists to cut.
                let s = base
                    * if spell {
                        spell_gain.0.max(0.0)
                    } else if synthetic {
                        fire_gain.0.max(0.0)
                    } else {
                        1.0
                    };
                // MONKEY (flame flicker): the fire wobble, folded in at the very last moment — over
                // the committed colour and NOWHERE else. Deliberately downstream of `base`, which
                // still feeds the reach/lane below unmodulated: a breathing REACH would move the
                // interior window, re-decide the room gate's portal hop and re-rank the torch
                // shadow casters every frame, i.e. exactly the frame-rate thrash this feature
                // exists to avoid. Only the brightness moves; the pool's geometry is frozen.
                //
                // `elapsed_secs` (absolute, never a delta) is what makes it frame-rate independent:
                // the same instant gives the same brightness whatever the frame took.
                let fm = flicker.map_or(FlickerMod::STEADY, |f| {
                    f.at(now_secs, dynamic_interiors.flicker)
                });
                let s = s * fm.intensity;
                let rgb = commit_raw([c[0] * s, c[1] * s, c[2] * s * fm.blue]);
                // MONKEY (light lane by position): a light is INTERIOR iff it PHYSICALLY STANDS in
                // an interior-class WMO group — the verdict [`classify_light_lanes`] (static) or
                // the carried-light claim (entities) wrote onto it. Claiming a room is no longer
                // the test: Stormwind's street torches claim exterior-class groups and must light
                // the cobbles, while the Goldshire inn's fixtures claim an exterior-flagged group
                // from inside the building and must NOT light the lawn (see [`LightLane`]).
                //
                // No lane yet ⇒ fall back to the OLD `LightRooms` rule rather than to "exterior".
                // A light lives one or two frames before the classifier's first pass (it spawns in
                // the stream stage, the classifier runs the frame after), and the conservative
                // arm of that gap is the pre-change behaviour: a MOLT fixture starts interior and
                // is corrected outward, never the reverse — so the gap can never flash a pool onto
                // an inn's lawn.
                let interior = lane_of.map_or_else(|| rooms.is_some(), |l| l.interior);
                // MONKEY (darkness gains): `interiorGain` dims the room lane's third input — every
                // INTERIOR fixture's committed colour — downstream of the fire gain and the
                // flicker, so those two keep their own meanings ("how bright is this invented
                // source" / "how hard does it breathe") and this one reads purely as "how dark is
                // the room". The EXTERIOR half of the table is untouched on purpose: an outdoor
                // campfire belongs to the night law, which dims the sky around it and not it.
                //
                // And so is the DAYLIGHT FIXTURE ([`super::DaylightFixture`]) — an interior-lane
                // entry that IS the sun standing in a doorway. A sunlit opening must not dim with
                // the room's candles: this dial means "how dark is the CANDLELIGHT". Nor does
                // `nightGain` claim it instead — that one dims the night, and a daylight fixture is
                // already scaled to nothing by its own day envelope (`daylight_target`'s `sun_w`,
                // the same curve) by the time the night dim is at full strength.
                let rgb = if interior && !daylight {
                    rgb.map(|c| c * dynamic_interiors.interior_gain)
                } else {
                    rgb
                };
                // A fixture's reach is its authored MOLT end where there is one and the M2 bucket
                // otherwise, with a fail-open default for a MOLT record whose end is absent or
                // degenerate (a few author 0 with `useAtten` clear).
                let lane = if interior {
                    let r = reach
                        .map(|r| r.0)
                        .filter(|r| *r > 0.5)
                        .unwrap_or_else(|| m2_light_reach(base));
                    interior_reach(r, dynamic_interiors.atten_scale)
                } else {
                    0.0
                };
                // MONKEY (room gate): the fixture's claim set, resolved HERE because this is the
                // one place that already knows both the rooms and the interior verdict. An
                // EXTERIOR-lane light is never read by the gate, so it packs the ungated head.
                let claim = if interior && dynamic_interiors.room_gate {
                    RoomClaim::build(rooms, lit_rooms)
                } else {
                    RoomClaim::UNGATED
                };
                let lit_n = lit_rooms.map_or(0, |l| l.rooms.groups.len());
                (d2, p, pl.range, rgb, synthetic, lane, claim, lit_n)
            })
        })
        .collect();
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    // MONKEY (ext light k8): 255, not 256 — index 255 is the shaders' `EXT_SEL_EMPTY` sentinel now
    // that a draw unit's selection packs eight 8-bit ranks. See [`MAX_LIVE_POINT_LIGHTS`]. The sort
    // above is nearest-camera-first, so the entry this drops is the farthest of the set.
    pts.truncate(MAX_LIVE_POINT_LIGHTS);
    // `.x` = the live entry count; MONKEY (dynamic interiors): `.yzw` = the interior lane's live
    // knobs (base ambient, per-fixture fill gain, exposure) — free lanes until now.
    // MONKEY (darkness gains): `interiorGain` scales two of the room lane's three inputs here (the
    // base ambient floor and the per-fixture fill gain; the fixtures' own colours took it above)
    // and pointedly leaves EXPOSURE alone — that one is the user's live dial (their config sits at
    // 4), and folding a dim into it would have the two knobs fight over the same number. A gain on
    // the INPUTS composes with whatever exposure is set to. The strict exterior-batch `ext_room`
    // term and the ext-class night blend read these same packed lanes, so they follow for free.
    fresh.rows[20] = [
        pts.len() as f32,
        dynamic_interiors.ambient * dynamic_interiors.interior_gain,
        dynamic_interiors.fill * dynamic_interiors.interior_gain,
        dynamic_interiors.exposure,
    ];
    for (i, (_, p, range, rgb, _, lane, claim, _)) in pts.iter().enumerate() {
        fresh.points[2 * i] = [p.x, p.y, p.z, *range];
        fresh.points[2 * i + 1] = [rgb[0], rgb[1], rgb[2], *lane];
        claim.write(&mut claims.0[i * ROOM_CLAIM_STRIDE..][..ROOM_CLAIM_STRIDE]);
    }
    // Entries past the count are stale in the point table by design (the count row guards every
    // reader) — but the claim table is read at the SAME index, so a stale head there would gate a
    // live light with a dead building's identity. Clear the tail instead of trusting the count.
    for slot in claims.0[pts.len() * ROOM_CLAIM_STRIDE..].iter_mut() {
        *slot = 0;
    }
    // `WOW_POINTS_DUMP=1`: print the committed point table once a second — the numeric probe for
    // "what is actually lighting this ground". A pool that reads wrong is one of a small set of
    // measurable causes (a duplicate light stacking, a light at the wrong height, an over-driven
    // colour, a count that shouldn't be there), and every one of them is a number here. Throttled,
    // and capped at the nearest 8 so a torch-lit town doesn't flood the log.
    //
    // `WOW_POINTS_DUMP=frame` drops the throttle. A once-a-second dump can only answer "is the pool
    // right?", never "is it the *same* pool it was last frame?" — and B38's flicker turned out to
    // alternate frame to frame, which a 1 Hz sample cannot see at all. Reading a per-second dump as
    // evidence of per-frame stability is how that light was cleared once already (0665's parked
    // culling test made the same mistake with a different instrument).
    static POINTS_DUMP: std::sync::OnceLock<Option<std::ffi::OsString>> =
        std::sync::OnceLock::new();
    if let Some(mode) = POINTS_DUMP.get_or_init(|| std::env::var_os("WOW_POINTS_DUMP")) {
        let every = if mode.as_os_str() == "frame" {
            0.0
        } else {
            1.0
        };
        let now = time.elapsed_secs_f64();
        if now - *last_dump >= every {
            *last_dump = now;
            // How contested the three slots are for the chunk under the camera — the number that
            // decides whether ground pops as emitters move. Candidacy is the faithful Chebyshev
            // box (`terrain.wgsl`'s `TERRAIN_REACH`); the old 48-yd sphere is printed beside it so
            // the over-gather stays visible rather than being taken on trust.
            let cell = 533.333_3 / 16.0;
            let half = 32.0 * 533.333_3;
            let snap = |v: f32| (((half + v) / cell).floor() + 0.5) * cell - half;
            let anchor = Vec3::new(snap(cam_pos.x), cam_pos.y, snap(cam_pos.z));
            let (mut boxed, mut sphere) = (0usize, 0usize);
            for (_, p, _, _, _, _, _, _) in &pts {
                let dv = *p - anchor;
                boxed += usize::from(dv.x.abs().max(dv.z.abs()) <= 33.570_166);
                sphere += usize::from(dv.length() <= 48.0);
            }
            // MONKEY (fire GO lights): how many of the packed entries we INVENTED. The whole
            // feature is a heuristic over content, so "the inn is too bright" has to be separable
            // into "too many synthetic sources" and "the authored ones changed" without a rebuild.
            let synth = pts.iter().filter(|(.., s, _, _, _)| *s).count();
            // MONKEY (flame flicker): how many of the nearby sources BURN — the census that answers
            // "did the route rule file this room's fixtures as flames at all?" without a rebuild.
            // Counted over the same radius the pack gathers from rather than off `pts`, so it does
            // not have to ride the packing tuple through five destructurings for a diagnostic.
            let flames = lights_q
                .iter()
                .filter(|(_, gt, ..)| {
                    gt.translation().distance_squared(cam_pos)
                        < POINT_PACK_RADIUS * POINT_PACK_RADIUS
                })
                .filter(|t| t.7.is_some())
                .count();
            // MONKEY (light lanes): how the table splits between the two now-disjoint consumer
            // families. "The inn's candles are lighting the lawn" is an INT count on a row the
            // exterior lane should never have seen — a number, printed per row below as INT/EXT.
            let interior = pts.iter().filter(|(.., lane, _, _)| *lane > 0.5).count();
            eprintln!(
                "[points] {} packed ({interior} INT / {} EXT, {synth} synthetic, {flames} flickering, gain {:.2}, atten x{:.2}, flicker x{:.2}, night x{:.2}, interior x{:.2}, spell x{:.2}), cam {cam_pos:.1?} — this chunk's candidates: {boxed} (was {sphere} at the 48 yd sphere), 3 slots",
                pts.len(),
                pts.len() - interior,
                fire_gain.0,
                dynamic_interiors.atten_scale,
                dynamic_interiors.flicker,
                // MONKEY (darkness gains): the two dim dials, beside the gains they compose with —
                // "is this room dark because the gain is 0.2 or because no fixture claims it" is
                // the question this line exists to answer without a rebuild.
                dynamic_interiors.night_gain,
                dynamic_interiors.interior_gain,
                // MONKEY (spellLightGain): beside them for the same reason — "is that fireball
                // dark because the dial is 0 or because its model synthesised no light at all" is
                // one number away, and the two causes look identical on screen.
                spell_gain.0,
            );
            for (d2, p, _, rgb, synthetic, lane, claim, lit_n) in pts.iter().take(8) {
                eprintln!(
                    "  d {:6.2}  at [{:8.2},{:7.2},{:8.2}]  rgb [{:.3},{:.3},{:.3}]  {}{}",
                    d2.sqrt(),
                    p.x,
                    p.y,
                    p.z,
                    rgb[0],
                    rgb[1],
                    rgb[2],
                    if *lane > 0.5 {
                        format!("INT reach {lane:5.2}")
                    } else {
                        "EXT".to_string()
                    },
                    if *synthetic { "  SYNTH" } else { "" },
                );
                // MONKEY (GO room claims): how many rooms this entry CLAIMS, whether or not the
                // lane lets the shader read them. A carried light (a brazier GameObject) now builds
                // its claims from its own world position through the placed lanes' rule, so "why is
                // the shop next door still dark" is `claims 1` vs `claims 3` here rather than a
                // guess — and on an `EXT` row it says the claims exist but are not being read.
                if *lit_n > 0 {
                    eprintln!("      claims {lit_n}");
                }
                // MONKEY (room gate): which rooms this entry is allowed to light — the number
                // behind "why is this room dark" / "why does that wall still glow".
                if *lane > 0.5 {
                    eprintln!("      rooms {claim}");
                }
            }
        }
    }
    // `WOW_LIGHT_DUMP=frame` (or `=1` for 1 Hz): the WHOLE packed header, bit-exact, per frame.
    //
    // The point of dumping every row rather than the interesting ones is that B38 has now eliminated
    // every *per-material* and *per-instance* shading input by measurement — they are bit-identical
    // on bright and dim frames alike — which leaves this buffer and the view as the only things that
    // can still be moving. A dump of selected rows would answer "did ambient move?"; only the full
    // set answers "did ANY shading input move?", and that is the question worth a run. Rows are
    // printed as raw f32 bits, so a change far below a printed decimal cannot hide.
    static LIGHT_DUMP: std::sync::OnceLock<Option<std::ffi::OsString>> = std::sync::OnceLock::new();
    if let Some(mode) = LIGHT_DUMP.get_or_init(|| std::env::var_os("WOW_LIGHT_DUMP")) {
        let every = if mode.as_os_str() == "frame" {
            0.0
        } else {
            1.0
        };
        let now = time.elapsed_secs_f64();
        if now - *last_rows_dump >= every {
            *last_rows_dump = now;
            let hash = fresh
                .rows
                .iter()
                .flatten()
                .fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
                    (h ^ u64::from(v.to_bits())).wrapping_mul(0x1000_0000_01b3)
                });
            eprintln!("[light] rows {hash:#018x}");
            for (i, r) in fresh.rows.iter().enumerate() {
                eprintln!(
                    "  {i:2} {:08x} {:08x} {:08x} {:08x}   {:9.5} {:9.5} {:9.5} {:9.5}",
                    r[0].to_bits(),
                    r[1].to_bits(),
                    r[2].to_bits(),
                    r[3].to_bits(),
                    r[0],
                    r[1],
                    r[2],
                    r[3],
                );
            }
        }
    }
    if data.0 != fresh {
        data.0 = fresh;
    }
}

/// MONKEY (light lane by position): decide each STATIC point light's [`LightLane`] from where it
/// physically stands, using the client's own light-attach down-ray
/// ([`crate::wmo_portal::indoor_verdict_at`], `LightAttach::DownRay` — the predicate that
/// classifies a standing NPC). See [`LightLane`] for WHY the lane cannot be read off `LightRooms`.
///
/// The verdict maps straight across:
/// - [`IndoorVerdict::DayNight`] / [`IndoorVerdict::Baked`] — the nearest face under the light is
///   an interior-class one (MOGP `& 0x48 == 0`). INTERIOR. This is what keeps the Goldshire inn's
///   ten fixtures interior even though the group that NAMES them is exterior-flagged: the ray from
///   each fixture lands on the inn's own room floor, one storey down at most, and the answer is a
///   fact about that floor rather than about the MOLR list.
/// - [`IndoorVerdict::OutdoorsOnWmo`] — a definite claim by an EXTERIOR-class face: a street, a
///   courtyard, a porch, a deck. EXTERIOR. Stormwind's 235 street fixtures land here.
/// - MONKEY (review fixes): [`IndoorVerdict::Outdoors`] with a terrain hit is resolved EXTERIOR,
///   even if MOLR/MODR names the fixture — a courtyard torch must still light its ground. Only
///   an empty ray (no WMO and no terrain) keeps the old `LightRooms` fallback: a streaming gap
///   supplies no evidence that a previously interior fixture should move outdoors.
///
/// **Cost.** One ray per static light per [`WmoResidency`](crate::interior::WmoResidency)
/// generation, and nothing at all once a neighbourhood is settled: a world-baked light never
/// moves, so the verdict is stable, and the generation only ticks when a building streams in or
/// out. `Without<ChildOf>` keeps the carried lights out — they are children of a bone joint and
/// take their lane from their bearer's already-resolved room instead.
#[allow(clippy::type_complexity)]
pub fn classify_light_lanes(
    mut commands: Commands,
    residency: Res<crate::interior::WmoResidency>,
    wmos: Res<Assets<benilla_assets::WmoModel>>,
    instances: Query<&crate::wmo_portal::WmoPortalInstance>,
    streamer: Res<crate::terrain_stream::TerrainStreamer>,
    adt_tiles: Res<Assets<benilla_assets::AdtTile>>,
    lights: Query<
        (Entity, &GlobalTransform, Option<&LightRooms>, Option<&LightLane>),
        (
            With<WorldPointLight>,
            Without<ShadowProxyLight>,
            Without<ChildOf>,
        ),
    >,
) {
    let generation = residency.generation();
    // Collected once per frame that has work; the ray walk borrows it per light. An empty world
    // (no placement yet) still short-circuits below on the `settled` check for every light.
    let mut instance_list: Option<Vec<&crate::wmo_portal::WmoPortalInstance>> = None;
    for (light, gt, rooms, lane) in &lights {
        if lane.is_some_and(|l| l.generation == generation || l.generation == LightLane::SETTLED) {
            continue; // settled — no ray, no archetype write
        }
        let instances = instance_list.get_or_insert_with(|| instances.iter().collect());
        let (verdict, _) = crate::wmo_portal::indoor_verdict_at(
            &wmos,
            instances.iter().map(|i| ((), *i)),
            &streamer,
            &adt_tiles,
            gt.translation(),
            crate::wmo_portal::LightAttach::DownRay,
        );
        // MONKEY (review fixes): `Outdoors` alone conflates terrain winning the ray with no
        // resident surface at all. Reuse its terrain probe only on that ambiguous arm; the
        // winning WMO verdicts already carry all the information the lane needs. Terrain above
        // the anchor is not a DOWN-ray hit (the same buried-terrain rule as down_ray_claim).
        let probe = gt.translation() + Vec3::Y * crate::wmo_portal::POSITION_PROBE_LIFT;
        let terrain_hit = matches!(verdict, crate::wmo_portal::IndoorVerdict::Outdoors)
            && crate::terrain_stream::terrain_height_under(
                &streamer,
                &adt_tiles,
                probe,
            ).is_some_and(|height| height <= probe.y);
        let interior = light_verdict_interior(&verdict, terrain_hit, rooms.is_some());
        let want = LightLane {
            interior,
            generation,
        };
        // Always re-stamped when the generation moved, even if the verdict didn't — the stamp is
        // what stops the next frame re-raying the same light.
        if lane != Some(&want) {
            commands.entity(light).insert(want);
        }
    }
}

// MONKEY (review fixes): room references are a fallback for an unresolved ray, never a veto on
// resolved outdoor geometry. Keep this decision explicit so both outdoor arms stay covered.
fn light_verdict_interior(
    verdict: &crate::wmo_portal::IndoorVerdict,
    terrain_hit: bool,
    has_rooms: bool,
) -> bool {
    use crate::wmo_portal::IndoorVerdict;
    match verdict {
        IndoorVerdict::DayNight | IndoorVerdict::Baked { .. } => true,
        IndoorVerdict::OutdoorsOnWmo => false,
        IndoorVerdict::Outdoors => !terrain_hit && has_rooms,
    }
}

/// Render-world: write the packed light into the shared buffer in place, before any draw reads it
/// (`RenderSystems::PrepareResources`). One small upload per frame, independent of material count.
fn upload_light(
    queue: Res<RenderQueue>,
    buffer: Option<Res<SharedLightBuffer>>,
    data: Option<Res<WowLightData>>,
) {
    let (Some(buffer), Some(data)) = (buffer, data) else {
        return;
    };
    queue.write_buffer(&buffer.0, 0, bytemuck::bytes_of(&data.0));
}

#[cfg(test)]
mod tests {
    use super::*;

    // MONKEY (review fixes): a reference cannot override resolved terrain or an outdoor WMO
    // face. An empty ray still preserves the old rule until residency supplies real evidence.
    #[test]
    fn light_lanes_distinguish_resolved_outdoors_from_an_empty_ray() {
        use crate::wmo_portal::IndoorVerdict;
        for has_rooms in [false, true] {
            assert!(!light_verdict_interior(&IndoorVerdict::Outdoors, true, has_rooms));
            assert!(!light_verdict_interior(&IndoorVerdict::OutdoorsOnWmo, false, has_rooms));
            assert!(light_verdict_interior(&IndoorVerdict::DayNight, true, has_rooms));
            assert!(light_verdict_interior(
                &IndoorVerdict::Baked { mocv: [0; 3], lobes: Vec::new() }, false, has_rooms,
            ));
            assert_eq!(light_verdict_interior(&IndoorVerdict::Outdoors, false, has_rooms), has_rooms);
        }
    }

    // MONKEY (moon shadows): a CPU mirror of the receivers' order, including the exact off arm.
    fn moon_combine(sky: f32, point: f32, strength: f32, shadow: f32) -> f32 {
        let factor = 1.0 - strength * (1.0 - shadow);
        if factor < 1.0 {
            (sky * factor + point).clamp(0.0, 1.0)
        } else {
            (sky + point).clamp(0.0, 1.0)
        }
    }

    #[test]
    fn moon_shadows_preserve_saturated_points_and_attenuate_the_whole_sky() {
        assert_eq!(moon_combine(0.2, 1.0, 0.35, 0.0), 1.0);
        assert_eq!(moon_combine(0.2, 0.0, 0.35, 0.0), 0.2 * (1.0 - 0.35));
        for (sky, point) in [(0.2f32, 0.0f32), (0.2, 1.0), (1.2, 0.1)] {
            assert_eq!(moon_combine(sky, point, 0.0, 0.0).to_bits(),
                (sky + point).clamp(0.0, 1.0).to_bits());
            assert_eq!(moon_combine(sky, point, 0.35, 1.0).to_bits(),
                (sky + point).clamp(0.0, 1.0).to_bits());
        }
    }

    // MONKEY (moon shadows): exercise the REAL state machine, including delayed acknowledgement
    // (new rig's deferred spawn), the propagation hold, and frame-rate-independent monotone ramp.
    fn assert_shadow_transition(state: &mut ShadowHandover, body: ShadowBody, strength: f32) {
        let (sun, moon) = if body == ShadowBody::Sun { (1.0, 0.0) } else { (0.0, 1.0) };
        for _ in 0..3 {
            state.request(sun, moon, strength, true, 1.0 / 60.0);
            assert_eq!(state.weight, 0.0, "no acknowledgement: wrong/absent basis must not cast");
        }
        state.aim_written(body);
        assert_eq!(state.weight, 0.0, "transform-write frame");
        state.request(sun, moon, strength, true, 1.0 / 60.0);
        assert_eq!(state.weight, 0.0, "propagation hold frame");
        let mut previous = 0.0;
        for _ in 0..91 {
            state.request(sun, moon, strength, true, 1.0 / 60.0);
            assert_eq!(state.aimed, state.wanted);
            assert!(state.weight.abs() >= previous, "monotone hand-over ramp");
            assert!((state.weight.abs() - previous) <= 1.0 / 90.0 + 1e-6);
            previous = state.weight.abs();
        }
        assert_eq!(state.weight, if body == ShadowBody::Sun { 1.0 } else { -strength });
    }

    #[test]
    fn moon_shadow_clock_jumps_wait_for_the_right_basis() {
        let mut state = ShadowHandover::default();
        assert_shadow_transition(&mut state, ShadowBody::Sun, 0.35);
        assert_shadow_transition(&mut state, ShadowBody::Moon, 0.35); // noon -> midnight
        assert_shadow_transition(&mut state, ShadowBody::Sun, 0.35); // midnight -> noon
    }

    #[test]
    fn moon_shadow_midnight_login_and_enable_wait_for_the_rig() {
        let mut state = ShadowHandover::default();
        assert_shadow_transition(&mut state, ShadowBody::Moon, 0.35); // login at midnight
        state.request(0.0, 1.0, 0.35, false, 1.0);
        assert_eq!(state.weight, 0.0);
        assert_shadow_transition(&mut state, ShadowBody::Moon, 0.35); // enable rig at midnight
    }

    #[test]
    fn moon_shadow_strength_zero_to_enabled_waits_for_the_moon() {
        let mut state = ShadowHandover::default();
        state.request(0.0, 1.0, 0.0, true, 0.0);
        state.aim_written(ShadowBody::Sun);
        state.request(0.0, 1.0, 0.0, true, 0.0);
        state.request(0.0, 1.0, 0.0, true, 2.0);
        assert_eq!(state.weight, 0.0);
        assert_shadow_transition(&mut state, ShadowBody::Moon, 0.35);
    }

    /// GOLDEN — the **live exterior M2 response** (0803): `wow_model.wgsl`'s doodad/entity lane must
    /// reproduce `E = A + I·D·(4/17)(0.375 + 2μ + 1.875μ²)` off the rows [`pack_model_core_rows`]
    /// writes, with `A` NOT scaling by the per-instance intensity and every sun band scaling by it
    /// exactly once (never I²).
    ///
    /// This exists because the capture harness cannot check it. `visual.sh`-style captures are
    /// bit-deterministic on static scenes (water-noon: MAE 0.000) but NOT on the entity/GameObject
    /// scenarios this lane owns — measured run-to-run at MAE 5.1 (chest-shade-rear) and 8.8
    /// (creature-sun-rear), a noise floor far above the ~0.24 signal the response change produces
    /// (0799 §2). So the lane's correctness is pinned HERE, deterministically, and the captures are
    /// only good for "it compiles and it moves pixels".
    ///
    /// `eval_sh_lane` mirrors the WGSL lane-for-lane on purpose — read the two side by side; a
    /// channel or row swap in the shader is caught by eye against this, not by this test.
    #[test]
    fn the_sh_response_lane_matches_the_closed_form_at_every_intensity() {
        // Stormwind, minute ≈1185 — the bands wow-re independently recovered from the reference's
        // own uploaded shader constants (0796 §1), so the test is anchored on a real committed pair.
        let ambient = [102.0 / 255.0, 97.0 / 255.0, 123.0 / 255.0];
        let diffuse = [255.0 / 255.0, 112.0 / 255.0, 0.0];
        let sun_dir = Vec3::new(0.31, -0.82, 0.48).normalize(); // travel dir; to-light = −this
        let mut rows = [[0.0f32; 4]; LIGHT_HEADER_ROWS];
        pack_model_core_rows(&mut rows, ambient, diffuse, sun_dir);

        /// The SH branch of `wow_model.wgsl`'s exterior doodad/entity lane, verbatim.
        fn eval_sh_lane(rows: &[[f32; 4]; LIGHT_HEADER_ROWS], n: Vec3, intensity: f32) -> [f32; 3] {
            let quad = [n.x * n.y, n.y * n.z, n.z * n.z, n.x * n.z];
            let x2y2 = n.x * n.x - n.y * n.y;
            let dot3 = |r: [f32; 4]| r[0] * n.x + r[1] * n.y + r[2] * n.z;
            let dot4 =
                |r: [f32; 4]| r[0] * quad[0] + r[1] * quad[1] + r[2] * quad[2] + r[3] * quad[3];
            [0usize, 1, 2].map(|ch| {
                // sh_c10_{r,g,b} = rows[6+ch] (.w = ambient) · sh_c13_{r,g,b} = rows[9+ch]
                // sh_c16.xyz = rows[12][ch] · grade.yzw = rows[17][1+ch] (the sun's DC, at I=1)
                rows[6 + ch][3]
                    + rows[17][1 + ch] * intensity
                    + intensity * (dot3(rows[6 + ch]) + dot4(rows[9 + ch]) + rows[12][ch] * x2y2)
            })
        }

        let u = -sun_dir; // toward-light unit
        let f = |mu: f32| (4.0 / 17.0) * (0.375 + 2.0 * mu + 1.875 * mu * mu);
        // A side-on normal (μ = 0) and a mid-back one (μ ≈ −0.53, the lobe's negative dip).
        let side = u.cross(Vec3::Y).normalize();
        let mid_back = (u * -0.5333 + side * (1.0f32 - 0.5333 * 0.5333).sqrt()).normalize();
        for (label, n) in [
            ("facing", u),
            ("away", -u),
            ("side-on", side),
            ("mid-back", mid_back),
        ] {
            for intensity in [0.5f32, 1.0, 2.5] {
                let mu = n.dot(u);
                let got = eval_sh_lane(&rows, n, intensity);
                for ch in 0..3 {
                    let want = ambient[ch] + intensity * diffuse[ch] * f(mu);
                    assert!(
                        (got[ch] - want).abs() < 1e-5,
                        "{label} I={intensity} ch{ch}: got {} want {want}",
                        got[ch]
                    );
                }
            }
        }
        // The peak is calibrated to the FFP peak by construction (the 16/17 accumulate scale) — so
        // moving onto this curve changed NOTHING on a surface square to the sun, and the whole
        // visible difference lives on the shadow side. That is why 0803 read subtle, not dramatic.
        let peak = eval_sh_lane(&rows, u, 1.0);
        for ch in 0..3 {
            let ffp_peak = ambient[ch] + diffuse[ch]; // ambient + D·max(N·L,0) at N·L = 1
            assert!(
                (peak[ch] - ffp_peak).abs() < 1e-5,
                "peak ch{ch}: SH {} vs FFP {}",
                peak[ch],
                ffp_peak
            );
        }
        // And the mid-back dip really is BELOW ambient — the low-order-SH ringing the reference
        // authors. Clamping the sun term per-term instead of the sum would erase it.
        let dip = eval_sh_lane(&rows, mid_back, 1.0);
        assert!(
            dip[0] < ambient[0],
            "mid-back should dip below ambient: {} vs {}",
            dip[0],
            ambient[0]
        );
    }

    /// GOLDEN — the **commit clamp** (wow-re `m2-light-emitter-instances.md` §6a: `0x71ca80` with
    /// `w = 1.0` degenerates to clamp01), driven end to end through the real packer so removing the
    /// clamp from the pack expression fails here rather than in the director's eye.
    ///
    /// The held torch is the case that made it visible: authored `(0.467, 0.290, 0.133) × 3.0`, i.e.
    /// a red channel 40% past white. Unclamped it saturated the MCVT grid far wider than the
    /// reference and the ground pool read white instead of flame-orange.
    #[test]
    fn the_torch_commits_the_raw_authored_product() {
        let mut app = packer_app();
        // The real authored torch light, through the real spawn recipe.
        app.world_mut().spawn((
            crate::terrain_stream::point_light([0.466_666_7, 0.290_196_1, 0.133_333_34], 3.0),
            GlobalTransform::from_translation(Vec3::new(0.0, 1.5, 0.0)),
        ));
        app.update();

        let rows = &app.world().resource::<WowLightData>().0;
        assert_eq!(rows.rows[20][0], 1.0, "the light packed");
        let rgb = rows.points[1];
        // The raw authored product — over-white preserved. Two earlier rounds "fixed" this to a
        // per-channel clamp and then a peak-normalize; the trace-confirmed mechanism is that the
        // `0x71ca80` encode is decoded straight back by `0x593040`, so the GL light receives the
        // raw `colour × intensity` (ring capture: a terrain draw commits (1.2, 1.035, 0.805)
        // verbatim). Saturation belongs to the receiving vertex's lighting clamp, not the commit.
        assert!(
            (rgb[0] - 1.400_000_1).abs() < 1e-4,
            "red commits raw past white: {rgb:?}"
        );
        assert!(
            (rgb[1] - 0.870_588_3).abs() < 1e-4,
            "green commits raw: {rgb:?}"
        );
        assert!((rgb[2] - 0.4).abs() < 1e-4, "blue commits raw: {rgb:?}");
    }

    /// A minimal app around the real [`build_light_data`]: every resource the system takes, plus a
    /// world camera at the origin. Was inline in the torch golden and short three resources — the
    /// system had grown `WorldShadowActive`/`ShadowDistance`/`DynamicInteriors` params since, and
    /// the test had been failing param validation rather than asserting anything. One builder now,
    /// so a fourth param can't silently red the same way twice.
    fn packer_app() -> App {
        let mut app = App::new();
        app.init_resource::<WowLighting>()
            .init_resource::<crate::dev_state::DebugState>()
            .init_resource::<crate::view::ViewDistance>()
            .init_resource::<WowLightData>()
            // MONKEY (room gate): the packer writes the claim table in the same walk.
            .init_resource::<RoomClaimTable>()
            .init_resource::<Time>()
            .init_resource::<WorldShadowActive>()
            .init_resource::<ShadowDistance>()
            // MONKEY (moon shadows): the packer's sixth dial resource.
            .init_resource::<MoonShadowStrength>()
            .init_resource::<ShadowHandover>()
            .init_resource::<DynamicInteriors>()
            .init_resource::<FireLightGain>()
            .init_resource::<SpellLightGain>()
            .add_systems(Update, build_light_data);
        app.world_mut()
            .spawn((crate::view::WorldCamera, GlobalTransform::IDENTITY));
        app
    }

    /// MONKEY (room gate): the claim a fixture is packed with is the SPAWNER'S OWN claim list when
    /// it built one, and it fails OPEN on every ambiguous arm.
    ///
    /// That the set is wider than MOLR is a data fact rather than a preference: the shipped
    /// Goldshire inn authors a MOLR on 2 of its 12 groups (nine of its ten fixtures claimed by the
    /// exterior-flagged whole-building shell) and NSabbey leaves seven of its rooms unnamed, so a
    /// MOLR-only claim set would black most of both buildings out.
    ///
    /// MONKEY (portal claims): the list REPLACES the MOLR union rather than being merged with it —
    /// `benilla_formats::room_claims` already folds MOLR in, in priority order, and re-adding it
    /// here would re-admit the district-scale shells that rule deliberately marks
    /// exterior-lane-ineligible. Only a light with no list (a carried torch) falls back to MOLR.
    #[test]
    fn the_room_claim_takes_the_spawners_list_and_falls_back_to_molr() {
        let mut world = World::new();
        // Entity index 0 is the "terrain cell / no building" sentinel in the shader's identity
        // lane, so the packer fails OPEN on it — burn it, then take a real index.
        let zeroth = world.spawn_empty().id();
        assert_eq!(zeroth.index().index(), 0, "the first index really is the sentinel");
        let inst = world.spawn_empty().id();
        let rooms = LightRooms(crate::wmo_portal::WmoGroupVis {
            instance: inst,
            groups: std::sync::Arc::from([4u16]),
        });
        // The spawner's list: the room it stands in (g5), the MOLR group it re-found (g4), and one
        // more its box holds (g7) — its own order, which the packer must not reshuffle.
        let lit = LightLitRooms {
            rooms: crate::wmo_portal::WmoGroupVis {
            instance: inst,
                groups: std::sync::Arc::from([5u16, 4, 7]),
            },
            // MONKEY (soft portal claims): no fades — every claim is HARD, which is the packer's
            // pre-fade behaviour and what every fallback path still produces.
            fades: std::sync::Arc::from([]),
        };

        let claim = RoomClaim::build(Some(&rooms), Some(&lit));
        assert_eq!(claim.instance, inst.index().index(), "keyed to the placement");
        assert_eq!(claim.n, 3, "the spawner's list, verbatim: {claim}");
        assert_eq!(&claim.groups[..3], &[5, 4, 7], "in the spawner's priority order");

        let mut gpu = [0u32; ROOM_CLAIM_STRIDE];
        claim.write(&mut gpu);
        assert_eq!(gpu[1], 3, "the count the shader loops to");
        assert_eq!(ROOM_CLAIM_FADE, 2 + ROOM_CLAIM_MAX, "the fades start after the id block");
        // `group + 1`, so the shader can keep 0 for "empty" / "this fragment names no room" —
        // group 0 is a real, common group id and could not be its own sentinel. MONKEY (portal
        // claims): plus the positive exterior-lane bit, since none of these carries the deny flag.
        assert_eq!(
            &gpu[2..5],
            &[
                6 | CLAIM_EXT_OK | (255 << CLAIM_ENTRY_SHIFT),
                5 | CLAIM_EXT_OK | (255 << CLAIM_ENTRY_SHIFT),
                8 | CLAIM_EXT_OK | (255 << CLAIM_ENTRY_SHIFT),
            ],
            "claims stored + 1, exterior-lane eligible, at full entry weight"
        );
        assert_eq!(&gpu[5..ROOM_CLAIM_FADE], &[0, 0, 0], "padding stays empty, not a claim on g0");
        // MONKEY (soft portal claims): no fades supplied ⇒ every record is zero, and `radius == 0`
        // is the HARD claim the shader reads as weight 1 everywhere.
        assert!(gpu[ROOM_CLAIM_FADE..].iter().all(|w| *w == 0), "hard claims write no fade");

        // MONKEY (review fixes): containment claims must pack identically with NO MOLR. This
        // used to discard the whole list and let one fixture light every room in the building.
        let mut no_molr_gpu = [0u32; ROOM_CLAIM_STRIDE];
        RoomClaim::build(None, Some(&lit)).write(&mut no_molr_gpu);
        assert_eq!(no_molr_gpu, gpu);

        // MONKEY (portal claims): a district-scale shell still gates the fixture (it is in the
        // count, so the fixture is NOT ungated) but reaches the exterior lane with no eligibility
        // bit — the one thing that keeps a tavern candle off the street outside.
        let shell = LightLitRooms {
            rooms: crate::wmo_portal::WmoGroupVis {
            instance: inst,
                groups: std::sync::Arc::from([5u16, 9 | LIT_ROOM_EXT_DENY]),
            },
            fades: std::sync::Arc::from([]),
        };
        let mut gpu = [0u32; ROOM_CLAIM_STRIDE];
        RoomClaim::build(Some(&rooms), Some(&shell)).write(&mut gpu);
        assert_eq!(gpu[1], 2);
        assert_eq!(
            &gpu[2..4],
            &[
                6 | CLAIM_EXT_OK | (255 << CLAIM_ENTRY_SHIFT),
                10 | (255 << CLAIM_ENTRY_SHIFT),
            ],
            "the shell claims, but not outdoors"
        );

        // No list at all (a carried torch): the MOLR rooms, as before.
        assert_eq!(
            RoomClaim::build(Some(&rooms), None).groups[0],
            4,
            "the fallback is the authored relation"
        );

        // The sentinel index fails OPEN rather than gating against a key no region can carry.
        assert_eq!(
            RoomClaim::build(
                Some(&LightRooms(crate::wmo_portal::WmoGroupVis {
                    instance: zeroth,
                    groups: std::sync::Arc::from([4u16]),
                })),
                None,
            )
            .n,
            0,
            "an instance the identity lane cannot express packs UNGATED"
        );

        // A light nothing claims lights everything, exactly as before the gate existed.
        assert_eq!(RoomClaim::build(None, None).n, 0, "unclaimed = ungated");

        // MONKEY (GO room claims): an overflow keeps the six highest-priority claims, in order —
        // never ungated (that lit the whole building through its walls).
        let many: Vec<u16> = (0..ROOM_CLAIM_MAX as u16 + 1).collect();
        let big = LightLitRooms {
            rooms: crate::wmo_portal::WmoGroupVis {
            instance: inst,
                groups: std::sync::Arc::from(&many[..]),
            },
            fades: std::sync::Arc::from([]),
        };
        let packed = RoomClaim::build(Some(&rooms), Some(&big));
        assert_eq!(usize::from(packed.n), ROOM_CLAIM_MAX, "overflow truncates to the cap");
        assert_eq!(&packed.groups[..], &many[..ROOM_CLAIM_MAX], "keeps the head, drops the tail");
    }

    /// GOLDEN — MONKEY (soft portal claims): the fade half of the GPU record, by byte offset.
    ///
    /// The layout is the one contract the CPU and `static_gx.wgsl` cannot negotiate at runtime: a
    /// stride or offset that disagrees reads every fixture's claims out of a neighbour's record,
    /// which blanks or floods every building at once. So it is asserted as literal indices here,
    /// beside the constants the shader mirrors.
    #[test]
    fn the_fade_record_packs_at_a_fixed_offset_per_claim() {
        let mut world = World::new();
        world.spawn_empty(); // burn the 0 sentinel
        let inst = world.spawn_empty().id();
        // Claim 0 HARD (the room the fixture stands in), claim 1 SOFT (one doorway away).
        let lit = LightLitRooms {
            rooms: crate::wmo_portal::WmoGroupVis {
                instance: inst,
                groups: std::sync::Arc::from([5u16, 6]),
            },
            fades: std::sync::Arc::from([
                ClaimFade::default(),
                ClaimFade {
                    center: Vec3::new(-1234.5, 60.25, 7000.0),
                    slack: 1.5,
                    radius: 6.25,
                    entry: 0.4,
                },
            ]),
        };
        let mut gpu = [0u32; ROOM_CLAIM_STRIDE];
        RoomClaim::build(None, Some(&lit)).write(&mut gpu);
        assert_eq!(gpu[1], 2);
        // Slot 0: hard ⇒ all four words zero, so `radius == 0` and the shader weights it 1.
        assert_eq!(&gpu[ROOM_CLAIM_FADE..ROOM_CLAIM_FADE + 4], &[0, 0, 0, 0]);
        // Slot 1: centre as raw f32 bits (world coordinates reach +-17000 yd — nothing narrower
        // carries them), radius and slack as u16 yards x 256 in one word, radius in the low half.
        let f = ROOM_CLAIM_FADE + 4;
        assert_eq!(f32::from_bits(gpu[f]), -1234.5);
        assert_eq!(f32::from_bits(gpu[f + 1]), 60.25);
        assert_eq!(f32::from_bits(gpu[f + 2]), 7000.0);
        assert_eq!(gpu[f + 3] & 0xffff, (6.25 * CLAIM_FADE_SCALE) as u32, "radius, low half");
        assert_eq!(gpu[f + 3] >> 16, (1.5 * CLAIM_FADE_SCALE) as u32, "slack, high half");
        // The entry weight rides the ID word's spare bits, not the fade record.
        assert_eq!((gpu[3] >> CLAIM_ENTRY_SHIFT) & 0xff, 102, "0.4 x 255, on the soft claim");
        assert_eq!((gpu[2] >> CLAIM_ENTRY_SHIFT) & 0xff, 255, "full, on the hard one");
        // Every slot past the count is literally empty — a stale fade would follow a live light.
        assert!(gpu[ROOM_CLAIM_FADE + 8..].iter().all(|w| *w == 0));
        assert_eq!(gpu.len(), ROOM_CLAIM_STRIDE, "2 head + 6 ids + 6 x 4 fade");
    }

    /// GOLDEN — MONKEY (fire GO lights): `fireLightGain` scales SYNTHESISED sources and **only**
    /// them, at pack time.
    ///
    /// Pack time is the whole design: the dial has to be live (`/script SetCVar("fireLightGain",
    /// 0)` must darken the invented lights on the next frame, with no world respawn), and it has
    /// to be a kill switch for a heuristic that — unlike every mechanism around it — is derived
    /// from content rather than byte-verified. Both properties are exactly this assertion: the
    /// tagged light moves with the gain, the authored one beside it does not.
    #[test]
    fn the_fire_gain_scales_only_synthesised_lights() {
        let mut app = packer_app();
        // Same colour and intensity for both, so the only thing that can separate them is the tag.
        let recipe = || crate::terrain_stream::point_light([1.0, 0.5, 0.25], 2.0);
        app.world_mut()
            .spawn((recipe(), GlobalTransform::from_translation(Vec3::X)));
        app.world_mut().spawn((
            recipe(),
            GlobalTransform::from_translation(Vec3::new(2.0, 0.0, 0.0)),
            SyntheticFireLight,
        ));
        app.world_mut().insert_resource(FireLightGain(0.5));
        app.update();

        let data = app.world().resource::<WowLightData>().0;
        assert_eq!(data.rows[20][0], 2.0, "both packed");
        // Nearest-first: the authored one at x=1 is entry 0, the synthetic at x=2 is entry 1.
        let authored = data.points[1];
        let synthetic = data.points[3];
        assert!(
            (authored[0] - 2.0).abs() < 1e-4,
            "the authored light is untouched by the fire gain: {authored:?}"
        );
        assert!(
            (synthetic[0] - 1.0).abs() < 1e-4,
            "the synthesised light takes the gain: {synthetic:?}"
        );

        // Zero is the kill switch: the invented light commits black, the authored one is unmoved.
        app.world_mut().insert_resource(FireLightGain(0.0));
        app.update();
        let data = app.world().resource::<WowLightData>().0;
        assert_eq!(data.points[3], [0.0, 0.0, 0.0, 0.0], "gain 0 = lane off");
        assert!((data.points[1][0] - 2.0).abs() < 1e-4, "authored unaffected");
    }

    /// GOLDEN — MONKEY (spellLightGain): a SPELL light takes `spellLightGain` **instead of**
    /// `fireLightGain`, not as well as it.
    ///
    /// Every spell light is tagged [`SyntheticFireLight`] too — it IS an invented source, and the
    /// census must keep counting it as one — so the two arms overlap on every row this dial exists
    /// for. If they composed, turning the world's hearths down would darken every fireball in the
    /// game, and `spellLightGain 0` would still leave combat flashes on wherever a player had
    /// raised `fireLightGain`. Both halves are asserted: the spell row moves with ONE dial and is
    /// deaf to the other, and the plain synthetic row beside it is unmoved by the new one.
    #[test]
    fn the_spell_gain_overrides_the_fire_gain_on_a_spell_light() {
        let mut app = packer_app();
        let recipe = || crate::terrain_stream::point_light([1.0, 0.5, 0.25], 2.0);
        // A campfire's invented light…
        app.world_mut().spawn((
            recipe(),
            GlobalTransform::from_translation(Vec3::X),
            SyntheticFireLight,
        ));
        // …and a fireball's, which carries BOTH tags exactly as the spawn site stamps them.
        app.world_mut().spawn((
            recipe(),
            GlobalTransform::from_translation(Vec3::new(2.0, 0.0, 0.0)),
            SyntheticFireLight,
            SpellFxLight,
        ));
        app.world_mut().insert_resource(FireLightGain(0.5));
        app.world_mut().insert_resource(SpellLightGain(2.0));
        app.update();

        let data = app.world().resource::<WowLightData>().0;
        assert_eq!(data.rows[20][0], 2.0, "both packed");
        let fire = data.points[1];
        let spell = data.points[3];
        assert!(
            (fire[0] - 1.0).abs() < 1e-4,
            "the campfire takes the fire gain alone: {fire:?}"
        );
        assert!(
            (spell[0] - 4.0).abs() < 1e-4,
            "the spell light takes 2x, NOT 2x0.5: {spell:?}"
        );

        // Zero is this lane's own kill switch, and it reaches nothing else.
        app.world_mut().insert_resource(SpellLightGain(0.0));
        app.update();
        let data = app.world().resource::<WowLightData>().0;
        assert_eq!(data.points[3], [0.0, 0.0, 0.0, 0.0], "spell gain 0 = spell lights off");
        assert!(
            (data.points[1][0] - 1.0).abs() < 1e-4,
            "the campfire still burns at its own gain"
        );
    }

    /// GOLDEN — MONKEY (flame flicker): the wobble reaches the packed COLOUR and nothing else.
    ///
    /// Three properties, and each one is a bug if it inverts. (1) A flame's committed colour MOVES
    /// with the clock while an identical light without the component beside it does not — the
    /// feature exists at all. (2) The move stays inside the kind's authored amplitude, which is the
    /// anti-strobe guarantee measured where it actually lands (after the `4π` round trip and
    /// `commit_raw`), not just in the waveform's own unit test. (3) The colour row's `.w` — the
    /// interior REACH — is byte-identical across those frames. That is the one thing that must not
    /// breathe: a moving reach would re-window the interior pool, re-decide the room gate's portal
    /// hop and re-rank the torch-shadow casters every single frame, which is the frame-rate thrash
    /// ("the epileptic imp") this feature is built to stay clear of.
    #[test]
    fn the_flicker_moves_the_packed_colour_and_never_the_reach() {
        use super::super::flicker::FlameKind;
        let at = |app: &mut App, ms: u64| {
            let mut t = Time::<()>::default();
            t.advance_by(std::time::Duration::from_millis(ms));
            app.world_mut().insert_resource(t);
            app.update();
            app.world().resource::<WowLightData>().0
        };
        let mut app = packer_app();
        // Same recipe, same lane, same intensity: only the component can separate them.
        let recipe = || crate::terrain_stream::point_light([1.0, 0.5, 0.25], 1.5);
        let lane = LightLane { interior: true, generation: LightLane::SETTLED };
        app.world_mut()
            .spawn((recipe(), GlobalTransform::from_translation(Vec3::X), lane));
        app.world_mut().spawn((
            recipe(),
            GlobalTransform::from_translation(Vec3::new(2.0, 0.0, 0.0)),
            lane,
            FlameFlicker::new(FlameKind::Torch, 0x51ee_d105),
        ));

        let (mut moved, mut steady_moved) = (0.0f32, 0.0f32);
        let first = at(&mut app, 0);
        let (base_steady, base_flame, reach) = (first.points[1][0], first.points[3][0], first.points[3][3]);
        assert!(reach > 0.5, "the flame is on the interior lane, so `.w` carries its reach");
        for ms in (40..4000).step_by(37) {
            let d = at(&mut app, ms);
            steady_moved = steady_moved.max((d.points[1][0] - base_steady).abs());
            moved = moved.max((d.points[3][0] - base_flame).abs() / base_flame.max(1e-6));
            assert_eq!(d.points[3][3], reach, "the packed reach never moves with the flicker");
            assert_eq!(d.points[2], first.points[2], "nor does the position/range row");
        }
        assert_eq!(steady_moved, 0.0, "a light with no FlameFlicker is a constant, as before");
        assert!(moved > 0.02, "the flame barely moved at all: {moved}");
        // Both endpoints of the excursion are inside the torch rung's authored +-10%, doubled by
        // the two-sided base sample (the reference frame is t=0, itself off the mean).
        assert!(moved < 2.0 * FlameKind::Torch.amplitude(), "over amplitude: {moved}");

        // `fireFlicker 0` is the off switch: the flame commits the same bytes on every frame.
        app.world_mut().insert_resource(DynamicInteriors {
            flicker: 0.0,
            ..DynamicInteriors::default()
        });
        let off = at(&mut app, 5000).points[3];
        for ms in [5100u64, 5250, 5600] {
            assert_eq!(at(&mut app, ms).points[3], off, "gain 0 = the pre-feature constant");
        }
    }

    /// GOLDEN — MONKEY (light lanes / interior attenuation): the colour row's `.w` separates the
    /// two consumer families AND carries the interior reach, and `[pos.xyz, range]` is untouched.
    ///
    /// Three properties, one assertion each, because each is a bug that shipped or nearly did:
    /// a light that claims NO room packs lane `0` (so the exterior shaders keep reading it and the
    /// outdoor-fire lane is unchanged); a light that DOES packs its reach (so an inn's candles stop
    /// pooling on the lawn — the exterior loops skip `> 0.5`); and the reach is always ≥ 1 yd, so
    /// the "0 means exterior" test can never be confused by a degenerate authored end.
    ///
    /// MONKEY (light lane by position): none of these lights carries a [`LightLane`], so what this
    /// pins is the packer's **fail-safe fallback** — the pre-classifier rule, which is what a
    /// freshly spawned light is packed on for its first frame. The lane override itself is pinned
    /// by [`the_position_lane_overrides_the_room_claim`].
    ///
    /// `pos_range.w` is asserted at 48 on BOTH: the exterior lanes rank and cut on it, so packing
    /// the (much smaller) authored reach there instead — the obvious first design — would have
    /// silently shrunk every outdoor fire's candidacy radius by 5-7x.
    #[test]
    fn the_lane_flag_splits_interior_from_exterior_and_carries_the_reach() {
        let mut app = packer_app();
        // Same recipe both times: only the room claim can separate them.
        let recipe = || crate::terrain_stream::point_light([1.0, 0.8, 0.5], 1.0);
        app.world_mut()
            .spawn((recipe(), GlobalTransform::from_translation(Vec3::X)));
        let instance = app.world_mut().spawn(()).id();
        app.world_mut().spawn((
            recipe(),
            GlobalTransform::from_translation(Vec3::new(2.0, 0.0, 0.0)),
            LightRooms::new(crate::wmo_portal::WmoGroupVis::single(instance, 4)),
            LightReach(6.972), // the Goldshire inn's own authored MOLT end
        ));
        app.update();

        let data = app.world().resource::<WowLightData>().0;
        assert_eq!(data.rows[20][0], 2.0, "both packed");
        // Nearest-first: the roomless light at x=1 is entry 0, the fixture at x=2 is entry 1.
        assert_eq!(data.points[1][3], 0.0, "no rooms ⇒ the EXTERIOR lane");
        // MONKEY (soft falloff): the packed value is the EFFECTIVE RADIUS — the authored end times
        // the default `interiorAttenScale` (1.6), which is where the pool's soft profile now ends.
        assert!(
            (data.points[3][3] - 6.972 * 1.6).abs() < 1e-3,
            "a fixture packs its authored reach × the default scale: {:?}",
            data.points[3],
        );
        // The candidacy radius the exterior shaders rank on is untouched on both entries.
        assert_eq!(data.points[0][3], 48.0);
        assert_eq!(data.points[2][3], 48.0);

        // `interiorAttenScale` is live and folds in at pack time; 0 restores the legacy reach so
        // the window can be A/B'd from chat without a rebuild.
        for (scale, want) in [(2.0f32, 13.944f32), (0.0, INTERIOR_LEGACY_REACH)] {
            let mut di = *app.world().resource::<DynamicInteriors>();
            di.atten_scale = scale;
            app.world_mut().insert_resource(di);
            app.update();
            let data = app.world().resource::<WowLightData>().0;
            assert!(
                (data.points[3][3] - want).abs() < 1e-3,
                "scale {scale}: got {} want {want}",
                data.points[3][3],
            );
            assert_eq!(data.points[1][3], 0.0, "scale {scale}: exterior stays 0");
        }

        // A degenerate authored end must never land near the 0.5 lane threshold — it would read as
        // an EXTERIOR light and start lighting the hillside the fixture is inside.
        let e = app
            .world_mut()
            .spawn((
                recipe(),
                GlobalTransform::from_translation(Vec3::new(3.0, 0.0, 0.0)),
                LightRooms::new(crate::wmo_portal::WmoGroupVis::single(instance, 4)),
                LightReach(0.001),
            ))
            .id();
        let mut di = *app.world().resource::<DynamicInteriors>();
        di.atten_scale = 1.0;
        app.world_mut().insert_resource(di);
        app.update();
        let data = app.world().resource::<WowLightData>().0;
        assert!(
            data.points[5][3] >= 1.0,
            "a degenerate reach still packs as INTERIOR: {:?}",
            data.points[5],
        );
        app.world_mut().despawn(e);
    }

    /// GOLDEN — MONKEY (light lane by position): a [`LightLane`] OVERRIDES the room claim, in both
    /// directions. This is the whole of bug B's first cause and its regression guard in one test.
    ///
    /// - A fixture that CLAIMS a room but stands outdoors packs lane `0` — Stormwind's 235
    ///   street-only MOLT fixtures, which the old `Has<LightRooms>` rule filed interior and which
    ///   therefore lit nothing at all (no exterior consumer would read them, and the interior
    ///   consumer never runs on a street).
    /// - A light that claims NO room but stands indoors packs a reach — a GM-placed brazier in the
    ///   Lion's Pride Inn, and the M2 bucket (12 yd at intensity 1) × the default scale.
    ///
    /// The room claim itself is untouched by either: it still gates the portal PVS and the
    /// torch-shadow promotion, which is why the two facts have to be separate components.
    #[test]
    fn the_position_lane_overrides_the_room_claim() {
        let mut app = packer_app();
        let recipe = || crate::terrain_stream::point_light([1.0, 0.8, 0.5], 1.0);
        let instance = app.world_mut().spawn(()).id();
        // A street torch: named by an exterior-class group, so it has rooms — and stands outdoors.
        app.world_mut().spawn((
            recipe(),
            GlobalTransform::from_translation(Vec3::X),
            LightRooms::new(crate::wmo_portal::WmoGroupVis::single(instance, 4)),
            LightReach(9.889),
            LightLane {
                interior: false,
                generation: 0,
            },
        ));
        // A brazier carried into a room: no MOLT/MODR claim at all, but physically inside one.
        app.world_mut().spawn((
            recipe(),
            GlobalTransform::from_translation(Vec3::new(2.0, 0.0, 0.0)),
            LightLane::carried(true),
        ));
        app.update();

        let data = app.world().resource::<WowLightData>().0;
        assert_eq!(data.rows[20][0], 2.0, "both packed");
        assert_eq!(
            data.points[1][3], 0.0,
            "a room-claiming fixture that stands OUTDOORS is exterior: {:?}",
            data.points[1],
        );
        assert!(
            (data.points[3][3] - 12.0 * 1.6).abs() < 1e-3,
            "a roomless light that stands INDOORS is interior, at the M2 bucket × scale: {:?}",
            data.points[3],
        );
    }

    /// GOLDEN — MONKEY (interior attenuation): the M2 reach ladder lands each SYNTHESISED bucket on
    /// its own rung. The four intensities are `fire_light::fire_intensity`'s exact outputs, so a
    /// change to either ladder that desynchronises them fails here rather than in a dark tavern.
    #[test]
    fn the_m2_reach_ladder_matches_the_fire_intensity_buckets() {
        assert_eq!(m2_light_reach(0.6), 6.0, "candle");
        assert_eq!(m2_light_reach(1.5), 12.0, "torch / campfire");
        assert_eq!(m2_light_reach(2.0), 16.0, "brazier");
        assert_eq!(m2_light_reach(3.0), 24.0, "bonfire / forge");
        // The authored corpus rides the same ladder: a lantern (1.0) reads as a torch, the held
        // Club_1H_Torch (3.0, committed 1.4 red) as a bonfire.
        assert_eq!(m2_light_reach(1.0), 12.0);
        assert_eq!(m2_light_reach(16.0), 24.0, "the ladder has no upper hole");
    }

    /// GOLDEN — the PACKER's SH block: evaluating the rows this packer writes (DC lane +
    /// grade.yzw × I + I × (linear + quad + x²−y²)) must reproduce the disassembled `Model2.bls`
    /// closed form `clamp01(ambient + D·I·(3 + 16μ + 15μ²)/34)` at every intensity rung (2.5 lit /
    /// 1.0 mid-band / 0.5 MCSH-shadowed). Pins the "every sun band scales by I, never I²" law and
    /// the row homes (ambient in the DC lanes, the sun's DC redistribution on grade.yzw).
    ///
    /// NB (0747): no shader currently READS rows 6-12.xyz / 17.yzw — the live exterior lane in
    /// `wow_model.wgsl` implements the same closed form inline (sun side only, `max(0, f(μ))`;
    /// the full block's back-side wrap stays out per the anti-sun ruling). This golden pins the
    /// packed block itself; the flagged cleanup is either wiring a lane to the rows or retiring
    /// the dead fold together with this test.
    #[test]
    fn exterior_lane_reproduces_the_closed_form_at_every_intensity_rung() {
        let ambient = [0.30, 0.32, 0.38];
        let diffuse = [0.85, 0.70, 0.45];
        let sun_dir = Vec3::new(0.3, -0.8, 0.52).normalize(); // travel dir; to-light = −sun_dir
        let mut rows = [[0.0f32; 4]; LIGHT_HEADER_ROWS];
        pack_model_core_rows(&mut rows, ambient, diffuse, sun_dir);
        // The shader's exterior eval over the packed rows, per channel, at intensity `i`.
        let eval = |n: Vec3, i: f32| -> [f32; 3] {
            let quad = [n.x * n.y, n.y * n.z, n.z * n.z, n.x * n.z];
            [0usize, 1, 2].map(|ch| {
                let c10 = rows[6 + ch];
                let c13 = rows[9 + ch];
                let lin = c10[0] * n.x + c10[1] * n.y + c10[2] * n.z;
                let q: f32 = (0..4).map(|k| c13[k] * quad[k]).sum::<f32>()
                    + rows[12][ch] * (n.x * n.x - n.y * n.y);
                (c10[3] + rows[17][1 + ch] * i + i * (lin + q)).clamp(0.0, 1.0)
            })
        };
        let u = -sun_dir; // toward-light
        let side = u.cross(Vec3::Y).normalize();
        for i in [2.5f32, 1.0, 0.5] {
            for (n, mu, who) in [
                (u, 1.0f32, "facing"),
                (-u, -1.0, "away"),
                (side, 0.0, "side"),
            ] {
                let b = (3.0 + 16.0 * mu + 15.0 * mu * mu) / 34.0;
                let got = eval(n, i);
                for ch in 0..3 {
                    let want = (ambient[ch] + diffuse[ch] * i * b).clamp(0.0, 1.0);
                    assert!(
                        (got[ch] - want).abs() < 1e-5,
                        "I={i} {who}: ch{ch} got {} want {want}",
                        got[ch]
                    );
                }
            }
        }
        // The whole back hemisphere stays non-negative BEFORE ambient — the retired trace-fit's
        // negative lobe (blue shadow-side characters) must never come back. Closed-form minimum is
        // −0.0373·C at μ≈−0.53; with ambient ≥ 0.038·D the sum never floors a channel at 0.
        let zero_amb = {
            let mut r = [[0.0f32; 4]; LIGHT_HEADER_ROWS];
            pack_model_core_rows(&mut r, [0.0; 3], diffuse, sun_dir);
            r
        };
        let eval0 = |n: Vec3, i: f32| -> f32 {
            let quad = [n.x * n.y, n.y * n.z, n.z * n.z, n.x * n.z];
            let c10 = zero_amb[6];
            let c13 = zero_amb[9];
            let lin = c10[0] * n.x + c10[1] * n.y + c10[2] * n.z;
            let q: f32 = (0..4).map(|k| c13[k] * quad[k]).sum::<f32>()
                + zero_amb[12][0] * (n.x * n.x - n.y * n.y);
            zero_amb[6][3] + zero_amb[17][1] * i + i * (lin + q)
        };
        // Sweep μ over the back hemisphere: the dip never exceeds the documented −0.0373·C·I.
        for k in 0..=20 {
            let mu = -1.0 + k as f32 / 20.0;
            let n = (u * mu + side * (1.0 - mu * mu).sqrt()).normalize();
            let floor = -0.0374 * diffuse[0] * 2.5;
            assert!(
                eval0(n, 2.5) >= floor,
                "μ={mu}: ringing {} below the closed-form floor {floor}",
                eval0(n, 2.5)
            );
        }
    }

    /// MONKEY (ext light k8): the LIVE cap is one short of the SLOT cap, and the blob did not grow.
    ///
    /// The two numbers are easy to confuse and the failure mode of confusing them is invisible: a
    /// light seated at index 255 packs identically to `EXT_SEL_EMPTY` in all three shaders, so the
    /// draw units that selected it would read "end of list" and quietly drop every rank after it —
    /// a dark patch with nothing in any log. The second assertion is the other half of the deal
    /// that bought those bits: widening the selection had to cost the buffer NOTHING, because
    /// `LightStd430` is mirrored by three shaders plus the portrait booth's frozen studio blob.
    #[test]
    fn the_live_point_cap_leaves_the_sentinel_index_free() {
        assert_eq!(MAX_LIVE_POINT_LIGHTS, 255, "255 is EXT_SEL_EMPTY in the three shaders");
        assert_eq!(MAX_LIVE_POINT_LIGHTS, MAX_POINT_LIGHTS - 1);
        // 21 header rows + 2 x 256 point rows, 16 B each.
        assert_eq!(per_frame_blob_bytes(), 8528, "the mirrored blob must not change size");
    }

    /// MONKEY (enclosed day floor): `interiorDaylight` rides the FRACTION of the interior lane's
    /// on/off word, and every existing decode of that word must be blind to it.
    ///
    /// This is the test the feature stands on: there was no free `f32` left in an 8528-byte layout
    /// three shaders mirror, so the value shares a lane with two other facts. If the fraction ever
    /// grew past 0.5 — a wider `DAYLIGHT_LANE_SCALE`, a `daylight` that escaped its clamp — the
    /// debug decode `u32(max(w - 1, 0) + 0.5)` would round UP and every building in the frame would
    /// silently switch to a diagnostic overlay. Asserting the two decodes, not just the value, is
    /// what makes that impossible to introduce quietly.
    #[test]
    fn the_daylight_lane_rides_the_fraction_without_disturbing_its_neighbours() {
        // The shader's own two decodes of `wmo_fog_params.w`, transcribed.
        let interiors_on = |w: f32| w > 0.5;
        let idbg = |w: f32| (0.0f32.max(w - 1.0) + 0.5) as u32;
        let daylight_of = |w: f32| (w - w.floor()) / DAYLIGHT_LANE_SCALE;
        for debug in 0..=4u32 {
            for daylight in [0.0f32, 0.01, 0.12, 0.5, 0.999, 1.0] {
                let w = 1.0 + debug as f32 + daylight * DAYLIGHT_LANE_SCALE;
                assert!(interiors_on(w), "lane off at debug {debug} daylight {daylight}");
                assert_eq!(idbg(w), debug, "debug decode moved (w {w})");
                assert!(
                    (daylight_of(w) - daylight).abs() < 1e-4,
                    "daylight {daylight} round-tripped as {} (w {w})",
                    daylight_of(w),
                );
            }
        }
        // The OFF arm is untouched — byte-identical to before the feature, whatever the cvar says.
        assert!(!interiors_on(0.0));
        assert_eq!(daylight_of(0.0), 0.0);
        // …and the packer really writes it. `1 + debug + d*scale`, clamped at both ends.
        let pack = |enabled: bool, debug: u32, d: f32| {
            if enabled {
                1.0 + debug as f32 + d.clamp(0.0, 1.0) * DAYLIGHT_LANE_SCALE
            } else {
                0.0
            }
        };
        assert_eq!(pack(false, 3, 0.5), 0.0);
        assert_eq!(idbg(pack(true, 3, 5.0)), 3, "an out-of-range cvar must still clamp under 0.5");
        assert!((daylight_of(pack(true, 3, 5.0)) - 1.0).abs() < 1e-4);
        assert!((daylight_of(pack(true, 0, -1.0))).abs() < 1e-4);
    }

    /// MONKEY (bake floor): `interiorBakeFloor` rides the FRACTION of the world-shadow lane
    /// (`sh_c16.w`), and that lane's one existing decode must be blind to it.
    ///
    /// Same shape as the daylight-lane test above and load-bearing for the same reason: the layout
    /// has no free `f32`, so the value shares a row with a boolean. The failure this forbids is
    /// quiet and remote — a fraction that reached 0.5 would make `terrain.wgsl` read
    /// `sh_c16.w > 0.5` as TRUE with `worldShadows` OFF, i.e. every ADT in the world would drop its
    /// baked MCSH shadows because someone moved an interior slider. Hence the clamp inside
    /// [`pack_bake_lane`] (the product `bake_floor × interior_gain` reaches 1.5 at the knobs' own
    /// limits, which × 0.49 is 0.735 — over the cliff) and hence asserting the DECODE, not the value.
    #[test]
    fn the_bake_floor_rides_the_fraction_without_disturbing_the_world_shadow_flag() {
        // `terrain.wgsl`'s only decode of this lane, transcribed.
        let world_shadow_lane = |w: f32| w > 0.5;
        for &flag in &[false, true] {
            for &bake in &[0.0f32, 0.01, 0.08, 0.12, 0.2, 0.5, 1.0] {
                for &gain in &[0.2f32, 0.5, 1.0, 1.5] {
                    let w = pack_bake_lane(flag, bake, gain);
                    assert_eq!(
                        world_shadow_lane(w),
                        flag,
                        "world-shadow decode moved (w {w}, bake {bake}, gain {gain})",
                    );
                    let want = (bake * gain).clamp(0.0, 1.0);
                    assert!(
                        (unpack_bake_lane(w) - want).abs() < 1e-4,
                        "bake {bake} x gain {gain} round-tripped as {} (w {w})",
                        unpack_bake_lane(w),
                    );
                }
            }
        }
        // `0` restores the pre-feature look EXACTLY: the packed word is the bare flag, bit for bit.
        assert_eq!(pack_bake_lane(false, 0.0, 0.5), 0.0);
        assert_eq!(pack_bake_lane(true, 0.0, 0.5), 1.0);
        // An out-of-range cvar still clamps under the cliff rather than flipping the flag.
        assert!(!world_shadow_lane(pack_bake_lane(false, 9.0, 1.5)));
        assert!((unpack_bake_lane(pack_bake_lane(false, 9.0, 1.5)) - 1.0).abs() < 1e-4);
        // The gain really is the dimmer: the Dim preset's floor is under the Default's.
        assert!(pack_bake_lane(false, 0.08, 0.5) < pack_bake_lane(false, 0.12, 0.5));
    }

    /// GOLDEN — MONKEY (darkness gains): `nightGain` is EXACTLY inert while the sun is up and
    /// exactly the gain after dark, and it never touches a flame.
    ///
    /// The daylight half is the load-bearing one. The dial folds into rows the entire exterior look
    /// is derived from (ambient, diffuse, the SH block, the sun halo), so a ramp that missed 1.0 by
    /// an ulp at noon would perturb every daytime frame of a renderer whose fidelity work is
    /// measured against bit-exact row hashes — the bug would be invisible on screen and expensive
    /// in the log. `1 + (g - 1)·night_w` is written the way it is to make that endpoint exact.
    ///
    /// The point-table half is the FEATURE, not a detail: dimming the sky law while leaving the
    /// fires alone is what makes a torch read brighter against a darker night. A gain that reached
    /// the point entries would dim the flame by the same 20 % and net out to no change at all.
    #[test]
    fn the_night_gain_is_inert_by_day_and_exact_after_dark() {
        let base = WowLighting {
            ambient: [0.30, 0.32, 0.38],
            diffuse: [0.85, 0.70, 0.45],
            spec: [0.60, 0.55, 0.50],
            sun_dir: Vec3::new(0.3, -0.8, 0.52).normalize(),
            ..default()
        };
        // `sun_shadow_strength` saturates at `sin(12°)`: y = 1 is broad daylight, y = 0 the horizon.
        let pack = |celestial_y: f32| {
            let mut app = packer_app();
            app.world_mut().insert_resource(WowLighting {
                celestial_dir: Vec3::new(0.0, celestial_y, 0.0),
                ..base
            });
            // MONKEY (moon shadows): this gain test models an already-settled sun rig; the
            // packer now consumes its publication instead of manufacturing a clock-only weight.
            app.world_mut().resource_mut::<ShadowHandover>().weight = sun_shadow_strength(celestial_y);
            // An EXTERIOR fire, to prove the dial stops at the sky law.
            app.world_mut().spawn((
                crate::terrain_stream::point_light([1.0, 0.5, 0.25], 2.0),
                GlobalTransform::from_translation(Vec3::X),
            ));
            app.update();
            app.world().resource::<WowLightData>().0
        };

        let day = pack(1.0);
        assert_eq!(day.rows[5][2], 1.0, "the sun is up: night_w is 0");
        assert_eq!([day.rows[0][0], day.rows[0][1], day.rows[0][2]], base.ambient);
        assert_eq!([day.rows[1][0], day.rows[1][1], day.rows[1][2]], base.diffuse);
        assert_eq!([day.rows[3][0], day.rows[3][1], day.rows[3][2]], base.spec);
        // The SH block is derived from the same triple — its DC lanes carry ambient verbatim.
        assert_eq!(day.rows[6][3], base.ambient[0], "the SH DC is undimmed by day too");

        let night = pack(0.0);
        assert_eq!(night.rows[5][2], 0.0, "below the horizon: night_w is 1");
        let g = DynamicInteriors::default().night_gain;
        assert_eq!(g, 0.45, "the shipped default is the director's pick (2026-09-11: 0.45)");
        for c in 0..3 {
            assert_eq!(night.rows[0][c], base.ambient[c] * g, "ambient takes the gain");
            assert_eq!(night.rows[1][c], base.diffuse[c] * g, "diffuse takes the gain");
            assert_eq!(night.rows[3][c], base.spec[c] * g, "the sun halo follows its sun");
        }
        assert_eq!(night.rows[6][3], base.ambient[0] * g, "the SH DC follows the triple");
        // The fire is the same brightness on both frames — which is the point of the feature.
        assert_eq!(day.points[1], night.points[1], "a point light never takes the night dim");
        assert!((night.points[1][0] - 2.0).abs() < 1e-4, "…at its authored value");
    }

    /// GOLDEN — MONKEY (moon shadows): THE HAND-OVER LAW, swept over the whole game day against
    /// the REAL `DayNight` tables.
    ///
    /// Four properties, and each one is a way the feature breaks in a manner a screenshot cannot
    /// diagnose:
    ///
    /// 1. **Never both.** benilla has exactly ONE shadow-mapped directional light (the receivers
    ///    ASSIGN over the light loop — last one wins), so the map holds one body's depth at a time.
    ///    Two non-zero weights at any minute would mean one of them was being applied to the OTHER
    ///    body's map, which renders as a shadow pointing the wrong way rather than as an error.
    /// 2. **The gate never clips.** The moon's weight is hard-gated to zero while the sun has any
    ///    ([`moon_shadow_weight`]), and a hard gate biting on a non-zero value would be a STEP.
    ///    This asserts the gate is inert with the shipped tables — the moon's own elevation ramp is
    ///    already 0 everywhere the sun's is positive — so the gate is an invariant guard, not a
    ///    shaping term.
    /// 3. **Continuity.** No minute-to-minute jump in either weight, at dusk or at dawn. The
    ///    threshold is deliberately loose (0.1 per game minute): the point is "no pop", not a
    ///    derivative bound, and the smoothsteps are far smoother than that.
    /// 4. **The window is real and it is the blob's.** Between the sun's set and the moon's rise
    ///    NEITHER casts. That is not a bug to be crossfaded away — the moon is genuinely under the
    ///    horizon then — and the oval blob covers it at full strength (`blob_shadow::blob_weight`
    ///    of a zero lane is 1.0). The test pins the window's existence so a future table edit that
    ///    closed or inverted it has to say so here.
    #[test]
    fn the_two_shadow_weights_never_overlap_and_neither_jumps() {
        let sample = |minute: f32| {
            let sun_h = super::super::daynight::celestial_sun_direction(minute).y;
            let moon_h = super::super::daynight::moon_direction(minute).y;
            (
                sun_shadow_strength(sun_h),
                moon_shadow_weight(sun_h, moon_h),
                // The moon's ramp BEFORE the gate — property 2 needs to see what the gate hid.
                sun_shadow_strength(moon_h),
            )
        };
        let (mut prev_sun, mut prev_moon, _) = sample(0.0);
        let (mut dark_minutes, mut moon_minutes, mut sun_minutes) = (0u32, 0u32, 0u32);
        for m in 1..=1440 {
            let (sun, moon, ungated) = sample(m as f32);
            assert!(
                !(sun > 0.0 && moon > 0.0),
                "minute {m}: both bodies cast (sun {sun}, moon {moon}) — the rig holds ONE map"
            );
            assert!(
                !(sun > 0.0 && ungated > 0.0),
                "minute {m}: the moon's elevation ramp ({ungated}) is live while the sun's is \
                 ({sun}) — the hand-over gate is now CLIPPING a non-zero value, i.e. it has become \
                 a step in the weight rather than a guard on the invariant"
            );
            assert!(
                (sun - prev_sun).abs() < 0.1,
                "minute {m}: the sun weight jumped {prev_sun} -> {sun}"
            );
            assert!(
                (moon - prev_moon).abs() < 0.1,
                "minute {m}: the moon weight jumped {prev_moon} -> {moon}"
            );
            if sun > 0.0 {
                sun_minutes += 1;
            } else if moon > 0.0 {
                moon_minutes += 1;
            } else {
                dark_minutes += 1;
            }
            prev_sun = sun;
            prev_moon = moon;
        }
        assert!(sun_minutes > 600, "the sun should cast most of the day: {sun_minutes} min");
        assert!(moon_minutes > 200, "the moon should cast most of the night: {moon_minutes} min");
        assert!(
            dark_minutes > 60,
            "the sun sets ~1h45m before the moon rises — the oval blob owns that window, and a \
             zero here would mean the two bodies had been made to overlap: {dark_minutes} min"
        );
    }

    /// MONKEY (moon shadows): the SHADER end of the signed lane, asserted against the shader TEXT.
    ///
    /// [`pack_shadow_lane`] and `shadow_hook.wgsl`'s `sun_shadow_w` / `moon_shadow_w` are two
    /// declarations of ONE encoding, and the failure mode of a drift is not a build error: a
    /// receiver that went back to reading `fog_params.z` raw would take a NEGATIVE sun weight at
    /// night and BRIGHTEN every shadowed fragment (`1 − occ·negative > 1`), which reads as a
    /// glowing patch under a tree, not as a broken decode. `static_gx.wgsl` is this crate's own
    /// receiver, so it is the copy a `benilla-world` test can see; `terrain.wgsl` and
    /// `wow_model.wgsl` are `benilla-assets`' and are covered by the naga probes.
    #[test]
    fn the_static_gx_receiver_decodes_the_signed_shadow_lane() {
        let src = include_str!("../shaders/static_gx.wgsl");
        for call in [
            "shadow_hook::sun_shadow_w(wow_light.fog_params.z)",
            "shadow_hook::moon_shadow_w(wow_light.fog_params.z)",
        ] {
            assert!(
                src.contains(call),
                "static_gx.wgsl no longer decodes the signed shadow lane through `{call}` — a raw \
                 read of `fog_params.z` takes the MOON's negative weight as the sun's and brightens \
                 what it should darken"
            );
        }
        // …and the moon arm is still behind its early-out, which is the whole `moonShadowStrength 0`
        // ⇒ bit-identical-night contract on this receiver.
        assert!(
            src.contains("if (world_moon < 1.0) {"),
            "static_gx.wgsl's moon arm lost its `world_moon < 1.0` guard — every daylight and \
             feature-off fragment now pays for (and may round through) the night subtraction"
        );
    }

    /// GOLDEN — MONKEY (moon shadows): the SIGNED `fog_params.z` pack, both ends.
    ///
    /// The lane is one float carrying two mutually-exclusive weights by sign, so the round trip has
    /// to be exact in both directions — a lossy pack here is a shadow that is silently the wrong
    /// strength, or (worse) a moon weight leaking into the day arm. [`unpack_shadow_lane`] is the
    /// transcription of `shadow_hook.wgsl`'s two decodes; if the WGSL and this drift, the assertion
    /// that a moon frame reads zero on the sun's decode is the one that fires.
    #[test]
    fn the_signed_shadow_lane_round_trips_both_weights() {
        for sun in [0.0f32, 0.25, 0.5, 1.0] {
            assert_eq!(unpack_shadow_lane(pack_shadow_lane(sun, 0.0)), (sun, 0.0));
        }
        for moon in [0.0f32, 0.12, 0.35, 1.0] {
            assert_eq!(unpack_shadow_lane(pack_shadow_lane(0.0, moon)), (0.0, moon));
        }
        // The pre-feature bits: a zero moon weight leaves the lane EXACTLY the sun's own value, so
        // a `moonShadowStrength 0` build packs the float it always packed.
        assert_eq!(pack_shadow_lane(1.0, 0.0), 1.0);
        assert_eq!(pack_shadow_lane(0.0, 0.0), 0.0);
    }

    /// GOLDEN — MONKEY (moon shadows): the dial's two ends, through the REAL packer.
    ///
    /// `moonShadowStrength 0` must pack the byte-identical night frame the build before this
    /// feature packed — that is the whole "guard it like the existing `world_shadow < 0.999` arm"
    /// contract, and it is what stands between a faint moon shadow and a renderer whose night rows
    /// no longer match three rounds of shading forensics. The non-zero arm then proves the strength
    /// is folded CPU-side (so the receivers carry no extra uniform) and lands NEGATIVE, where the
    /// sun's own decode (`max(z, 0)`) reads zero out of it.
    #[test]
    fn the_moon_strength_dial_is_packed_cpu_side_and_zero_restores_the_old_night() {
        // Midnight: the sun is 10 degrees under, the white moon is overhead (+55 degrees).
        let sun_h = super::super::daynight::celestial_sun_direction(0.0).y;
        let moon_h = super::super::daynight::moon_direction(0.0).y;
        assert!(sun_h < 0.0 && moon_h > 0.3, "midnight: sun down, moon high ({sun_h}, {moon_h})");
        let pack = |strength: f32| {
            let mut app = packer_app();
            app.world_mut().insert_resource(WowLighting {
                celestial_dir: Vec3::new(0.0, sun_h, 0.0),
                moon_dir_white: Vec3::new(0.0, moon_h, 0.0),
                ..default()
            });
            app.world_mut().insert_resource(MoonShadowStrength(strength));
            let mut handover = app.world_mut().resource_mut::<ShadowHandover>();
            handover.request(0.0, 1.0, strength, true, 0.0);
            handover.aim_written(if strength > 0.0 { ShadowBody::Moon } else { ShadowBody::Sun });
            handover.request(0.0, 1.0, strength, true, 0.0);
            handover.request(0.0, 1.0, strength, true, 1.5);
            app.update();
            app.world().resource::<WowLightData>().0
        };
        let off = pack(0.0);
        assert_eq!(
            off.rows[5][2], 0.0,
            "moonShadowStrength 0 must pack the pre-feature night lane, to the bit"
        );
        let on = pack(MoonShadowStrength::default().0);
        assert_eq!(
            on.rows[5][2], -0.35,
            "a high moon at the shipped strength packs the FULL weight, negated"
        );
        assert_eq!(
            unpack_shadow_lane(on.rows[5][2]),
            (0.0, 0.35),
            "the sun's decode reads zero out of a moon frame"
        );
        // Every other row is the same night frame either way — the dial reaches ONE lane.
        for row in 0..LIGHT_HEADER_ROWS {
            for c in 0..4 {
                if (row, c) == (5, 2) {
                    continue;
                }
                assert_eq!(
                    on.rows[row][c], off.rows[row][c],
                    "row {row}.{c} moved with moonShadowStrength — the dial owns ONE lane"
                );
            }
        }
    }

    /// GOLDEN — MONKEY (darkness gains): `interiorGain` scales all THREE of the room lane's inputs
    /// and nothing on the exterior lane.
    ///
    /// Three inputs make a room's brightness (`static_gx.wgsl`'s `interior_room_light`): the base
    /// ambient floor, the per-fixture fill gain, and the fixtures' own colours. Scaling two of the
    /// three would change the room's COLOUR BALANCE rather than dim it — a 30 % cut that left the
    /// ambient floor standing reads as a washed-out room, not a darker one. `interiorExposure` is
    /// pointedly NOT in the set: it is the dial the user tunes live, and this must compose with it.
    ///
    /// The exterior assertion is the seam that matters. The two lanes share one packed table, so a
    /// gain applied before the lane verdict would dim every campfire and street torch in the world
    /// with the candles.
    #[test]
    fn the_interior_gain_scales_the_room_lane_and_never_the_exterior_one() {
        let mut app = packer_app();
        // One recipe, one lane component apart: nothing else can separate the two entries.
        let recipe = || crate::terrain_stream::point_light([1.0, 0.5, 0.25], 2.0);
        let lane = |interior| LightLane { interior, generation: LightLane::SETTLED };
        app.world_mut()
            .spawn((recipe(), GlobalTransform::from_translation(Vec3::X), lane(true)));
        app.world_mut().spawn((
            recipe(),
            GlobalTransform::from_translation(Vec3::new(2.0, 0.0, 0.0)),
            lane(false),
        ));
        // MONKEY (darkness gains): a DAYLIGHT fixture — an interior-lane entry that is the sun in
        // a doorway. It rides the room lane but belongs to the exterior law, so the candle dial
        // must step over it; without the marker check it would dim a sunlit doorway by 30 %.
        app.world_mut().spawn((
            recipe(),
            GlobalTransform::from_translation(Vec3::new(3.0, 0.0, 0.0)),
            lane(true),
            crate::lighting::DaylightFixture {
                instance: Entity::PLACEHOLDER,
                group: 0,
                portal: None,
                how: crate::lighting::DaylightHow::Portal,
                reach: 8.0,
                cal_d: 4.0,
                cal_ndl: 0.5,
            },
        ));
        let g = 0.7;
        app.world_mut().insert_resource(DynamicInteriors {
            ambient: 0.2,
            fill: 0.5,
            exposure: 3.0,
            interior_gain: g,
            ..default()
        });
        app.update();

        let data = app.world().resource::<WowLightData>().0;
        assert_eq!(data.rows[20][0], 3.0, "all three packed");
        assert_eq!(data.rows[20][1], 0.2 * g, "the base ambient floor takes the gain");
        assert_eq!(data.rows[20][2], 0.5 * g, "the per-fixture fill takes the gain");
        assert_eq!(data.rows[20][3], 3.0, "exposure stays the user's own dial");
        // Nearest-first: the interior fixture at x=1 is entry 0, the exterior one at x=2 entry 1.
        let (int, ext) = (data.points[1], data.points[3]);
        assert!(int[3] > 0.5 && ext[3] == 0.0, "the lanes packed as expected: {int:?} {ext:?}");
        assert!(
            (int[0] - 2.0 * g).abs() < 1e-4,
            "the interior fixture's colour takes the gain: {int:?}"
        );
        assert!(
            (ext[0] - 2.0).abs() < 1e-4,
            "the exterior light is untouched by it: {ext:?}"
        );
        let sun = data.points[5];
        assert!(sun[3] > 0.5, "the daylight fixture packed on the interior lane: {sun:?}");
        assert!(
            (sun[0] - 2.0).abs() < 1e-4,
            "…and a sunlit doorway does not dim with the candles: {sun:?}"
        );
    }
}
