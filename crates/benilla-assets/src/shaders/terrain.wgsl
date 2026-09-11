// Phase 6 terrain splat shader — extends Bevy's StandardMaterial with a CUSTOM vertex + fragment stage.
//
// Blends up to 4 tiled layer textures by a per-chunk alpha map, then lights the result with WoW's
// faithful terrain lighting (fixed-function GL_LIGHTING, reproduced):
//   • DIFFUSE — `clamp(ambient_row1 + diffuse_row0·max(N·L,0) + Σ point·att·N·L)`, modulated into
//     the texture (MOD 1×). Evaluated AND CLAMPED per VERTEX (the GL T&L locus), then
//     Gouraud-interpolated — the clamp position matters once an over-gamut point light is in the
//     sum (see the `primary` varying note).
//   • SPECULAR (the bare-ground "sheen") — `clamp(row9 · pow(max(N·H,0),20)) · sheen_mask`, added
//     AFTER the texture modulate (separate-specular, LOCAL_VIEWER). The Blinn term is computed
//     PER-VERTEX (clamped per vertex, Gouraud-interpolated); the `sheen_mask` is the `_s` texture's
//     per-texel ALPHA, blended like the colour (so the sheen rides only shiny texels — stone, not
//     grass). See docs/knowledge/lighting.md
//     ⚠️ Two things bound it, BOTH needed (each was a separate blow-out): (a) per-vertex undersampling
//     of the sharp lobe (sampled only at the ~4 yd MCVT vertices, clamped, linearly smeared) vs a
//     per-PIXEL eval that saturates; (b) the `_s` alpha sheen-mask (ref mean ≈ 0.1) vs applying it at
//     a uniform 1.0 — row 9 ≈ near-white, so unmasked it washes the whole lit face to cream.
//
// Because Bevy's `VertexOutput` has no slot for the interpolated specular, we do our own (minimal)
// vertex transform and a custom IO struct, and drop `main_pass_post_lighting_processing` (fog returns
// IN-SHADER at Step 5, per docs/plans/lighting-rebuild.md — not Bevy's DistanceFog).
//
// One material serves a whole ADT tile (see terrain.rs): layer textures live in a `texture_2d_array`
// (binding 100), per-chunk alpha maps in another (104). Each merged-mesh vertex carries its chunk's 4
// layer indices in `color` (vertex COLOR) and its alpha-layer index in `uv_b.x` (UV1) — constant across
// the chunk, so interpolation lands back on the integer index.

#import bevy_pbr::{
    mesh_functions,
    forward_io::Vertex,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::{lights, view},
    shadows,
}
// MONKEY (shadow hook): the realtime directional-shadow term (fetch + edge/night fade) lives here.
#import benilla::shadow_hook

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var layer_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(104) var alpha_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(105) var splat_samp: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(110) var shadow_array: texture_2d_array<f32>;

// A fully covered character shadow retains 45% of the authored terrain colour.
const SHADOW_SUN_FLOOR: f32 = 0.45;

// Per-tile Vec4 uniforms packed into ONE buffer (binding 106) — the field order here MUST match the
// Rust `TerrainExtension` struct. Light + fog live in the shared global-light storage buffer (below);
// what's left is just the per-tile layer tiling factor.
//   params — x = layer tiling factor; yzw unused.
struct TerrainParams {
    params: vec4<f32>,
};
@group(#{MATERIAL_BIND_GROUP}) @binding(106) var<uniform> t: TerrainParams;

// The shared global light (lighting::global_light): ONE storage buffer every material reads, updated
// once/frame in place — replaces the per-material light/fog uniforms `apply_wow_lighting` re-pushed
// each frame. Rows 0-5 (of the 18-row LightStd430) are the light + fog terrain needs, plus the grade
// row at the end (rows 6-16 — model SH + water swatches — are skipped as a pad). Read in BOTH stages
// (the per-vertex sheen needs light_sun/spec in the vertex stage).
struct WowLight {
    light_ambient: vec4<f32>, // rgb = row 1 ambient; w = Mod2x scale (×1).
    light_diffuse: vec4<f32>, // rgb = row 0 sun diffuse; w = clamp-light flag (>0.5 ⇒ saturate).
    light_sun: vec4<f32>,     // xyz = world-space sun TRAVEL direction (to-light = −xyz); w unused.
    light_spec: vec4<f32>,    // rgb = row 9 specular color; w = shininess (20). rgb == 0 disables.
    fog_color: vec4<f32>,     // rgb = row 7 fog (raw, gamma 0..1); w = enable (>0.5 ⇒ blend).
    fog_params: vec4<f32>,    // x = fog_start yd; y = fog_end yd; z unused; w = farclip wall.
    _sh: array<vec4<f32>, 6>, // rows 6-11: the model SH coeffs (live in wow_model.wgsl, 0354) — unread by terrain.
    sh_c16: vec4<f32>,        // row 12: xyz the models' c16 quad band; .w a FREE lane (the 0273 point gain is retired).
    _water: array<vec4<f32>, 4>, // rows 13-16: the liquid swatches — unread by terrain.
    grade: vec4<f32>,         // reserved (the 0282 interior A/B retired; 0163). Layout only.
    _wmo_fog: array<vec4<f32>, 2>, // rows 18-19: the interior fog triple — unread by terrain.
    // The dynamic point-light table (decision 0278), packed by `global_light::build_light_data`:
    // row 20 `.x` = live entry count; then TWO rows per light — `[pos.xyz, range]`, `[rgb, lane]`.
    // MONKEY (light lanes): `lane` = 0 for an EXTERIOR light, > 0.5 (and equal to the fixture's
    // reach in yards) for an INTERIOR one. Terrain consumes only the exterior half.
    point_count: vec4<f32>,
    points: array<vec4<f32>, 512>,
};
@group(#{MATERIAL_BIND_GROUP}) @binding(90) var<storage, read> wow_light: WowLight;

// MONKEY (outdoor torch shadows: terrain): the GROUND receiver's torch bindings - the SAME depth
// array static_gx's group 3 samples and the SAME 6416-byte table, riding this material's own group
// (`TerrainExtension` bindings 91/92/93) because a Bevy material draw sets groups 0/1/2 only.
// ALWAYS bound (the image and buffer exist from startup), so no shader-def guards this block; only
// the fragment stage reads it. The struct text is COPIED VERBATIM from wow_model.wgsl - std430 here,
// std140 in static_gx, identical bytes because every member is 16-aligned - and a drift between the
// three copies would not be a compile error, it would be terrain sampling the wrong matrices.
struct TorchTable {
    // MONKEY (static torch cache): byte-identical in ALL THREE shaders and TorchTableUniform.
    // count@0 (16): x high-water slot count, y soft*100, z dynamic/live-bank mask, w flags.
    // positions@16 (256), view_projs@272 (6144): total 6416 bytes. A mismatch hides buildings.
    // MONKEY (live bank rank): 96 static layers + 48 live layers; matrices stay slot-addressed.
    count: vec4<u32>,
    positions: array<vec4<f32>, 16>,
    view_projs: array<mat4x4<f32>, 96>,
}
@group(#{MATERIAL_BIND_GROUP}) @binding(91) var torch_depth: texture_depth_2d_array;
@group(#{MATERIAL_BIND_GROUP}) @binding(92) var torch_samp: sampler_comparison;
@group(#{MATERIAL_BIND_GROUP}) @binding(93) var<storage, read> torch_table: TorchTable;
// Reverse-Z receiver bias - KEEP IN SYNC with static_gx.wgsl / wow_model.wgsl's TORCH_BIAS (0.001,
// the known-good value; larger detaches every shadow far from the torch).
const TORCH_BIAS: f32 = 0.001;
// NORMAL-OFFSET for the GROUND (yd): sample the map a hand's width above the surface along its own
// normal. Terrain is the one receiver whose casters REST on it - a crate, a fence post, a standing
// NPC - so the contact point is where receiver and caster depths agree to within a texel, i.e.
// exactly where a plain compare produces acne. Offsetting along the terrain normal (not straight
// up) is what keeps a SLOPE clean: on a 30 deg bank a vertical offset buys only `0.15*cos(30)` of
// separation while the depth gradient across a texel has grown by `1/cos(30)`, so the margin
// collapses on precisely the surfaces that need it most. The same 0.15 the entity lane uses, so a
// character and the ground under it are offset alike and their shadows meet without a gap.
const TORCH_NORMAL_OFFSET: f32 = 0.15;
// MONKEY (torch caster selection, MIRRORED from static_gx.wgsl): the live PCF tap-radius scale,
// unpacked from the table's `count.y` (stored x100 - the row is `vec4<u32>`).
fn torch_soft() -> f32 {
    return max(f32(torch_table.count.y) * 0.01, 0.05);
}
// MONKEY (outdoor torch shadows, MIRRORED from static_gx.wgsl - keep in sync): the EXTERIOR lane's
// world-distance fade radius (yd) and the CPU's one-bit lane gate in `count.w`. See the static_gx
// copies for the full rationale (the interior `ndc.z` fade is already down to 5 % at 10 yd, which
// erases a campfire's 15-25 yd pool).
const TORCH_EXT_FADE_YD: f32 = 44.0;
const TORCH_EXT_LANE: u32 = 1u;
// MONKEY (torch lane perf, MIRRORED from static_gx.wgsl): below THIS much unshadowed direct
// contribution a fragment skips the torch table scan and its four comparison taps entirely, because
// the shadow factor multiplies that contribution and can therefore only ever subtract less than this
// from the frame. Terrain's combine is a plain modulate rather than the interior exposure rolloff,
// so the worst possible step across the guard's boundary is 1e-4 of a fully-lit texel - two orders
// under one 8-bit code. Most of the skips are exact zeros anyway (`nl` IS 0 on ground facing away
// from the fire), so the threshold only has to be small enough to be invisible.
const TORCH_SKIP_EPS: f32 = 1e-4;
// The CPU's one-bit verdict: `exteriorShadows` AND night AND at least one promoted exterior
// fixture. EVERYTHING this lane costs terrain hangs off this single uniform read.
fn torch_ext_on() -> bool {
    return (torch_table.count.w & TORCH_EXT_LANE) != 0u;
}

// MONKEY (Phase 5, COPIED VERBATIM from wow_model.wgsl - keep in sync): the cube face that contains
// direction `d` (fixture -> fragment) - the major axis, signed. Face order is the contract with
// `benilla_app::torch_shadow::cube_view_projs`: 0 +X, 1 -X, 2 +Y, 3 -Y, 4 +Z, 5 -Z.
fn torch_face(d: vec3<f32>) -> u32 {
    let a = abs(d);
    if (a.x >= a.y && a.x >= a.z) {
        return select(1u, 0u, d.x > 0.0);
    }
    if (a.y >= a.z) {
        return select(3u, 2u, d.y > 0.0);
    }
    return select(5u, 4u, d.z > 0.0);
}

// MONKEY (outdoor torch shadows: terrain): this fixture's OWN cast shadow on a GROUND fragment -
// `wow_model.wgsl`'s `torch_entity_shadow_at` with the exterior fade already chosen, because terrain
// has no interior lane to share the body with (an MCNK cell is outdoors by definition, and the
// interior half of the light table is skipped before ranking). Correlate the `wow_light` fixture at
// `light_pos` to a promoted torch (position match within 1 yd), pick the cube face facing the
// fragment, and sample that layer through the shared projector. 1.0 (unshadowed) when no map matches
// - the lane is off, or the fixture was not among the promoted slots.
fn torch_terrain_shadow(light_pos: vec3<f32>, P: vec3<f32>, N: vec3<f32>) -> f32 {
    // MONKEY (torch lane perf): the empty table is the common case, so answer it before the normal
    // offset and the loop set-up.
    if (torch_table.count.x == 0u) {
        return 1.0;
    }
    // Normal-offset the sample point off the ground (see TORCH_NORMAL_OFFSET).
    let Ps = P + N * TORCH_NORMAL_OFFSET;
    for (var i = 0u; i < torch_table.count.x; i = i + 1u) {
        // MONKEY (static torch cache): holes and pending uploads never sample stale layers.
        if (torch_table.positions[i].w <= 0.0) { continue; }
        let fixture = torch_table.positions[i].xyz;
        if (distance(fixture, light_pos) < 1.0) {
            let face = torch_face(Ps - fixture);
            let layer = i * 6u + face;
            // MONKEY (live bank rank): count.z is CPU-ready-filtered; holes must not consume live
            // cubes. Keep the projection on the static slot while compacting depth only.
            let rank = countOneBits(torch_table.count.z & ((1u << i) - 1u));
            let depth_layer = select(layer, 96u + 6u * rank + face, (torch_table.count.z & (1u << i)) != 0u);
            // The EXTERIOR world-distance fade, measured from the FIXTURE (not the camera): this is
            // "how far does this fire's shadow carry", a property of the fire. Chosen just inside the
            // cube projection's own 48 yd far plane so the frustum edge is never a hard cut.
            let fade = 1.0 - smoothstep(0.8 * TORCH_EXT_FADE_YD, TORCH_EXT_FADE_YD, distance(fixture, Ps));
            let s = shadow_hook::torch_map_shadow(
                torch_table.view_projs[layer], i32(depth_layer), Ps, torch_depth, torch_samp, TORCH_BIAS,
                torch_soft(), fade);
            // MONKEY (torch caster selection): `.w` is the slot's FADE WEIGHT, so a promoted fixture's
            // shadow ramps in over ~1/3 s and a demoted one ramps out. The ground and the NPC standing
            // on it MUST use the same weight or one shadow would pop while the other faded.
            return mix(1.0, s, torch_table.positions[i].w);
        }
    }
    return 1.0;
}

// MONKEY (torch debug, interiorDebug 2 on TERRAIN): the MIN raw cube-map shadow factor over every
// promoted fixture (ignoring the fixture match) - mirrors static_gx's `torch_debug_factor` and
// wow_model's `torch_entity_debug_factor`, so the GROUND's sampling can be SEEN as greyscale:
// all-white = the table is empty / the projection misses the ground; shaped dark = the depth map
// reaches the terrain and any remaining fault is downstream of sampling (correlation, weighting).
fn torch_terrain_debug_factor(P: vec3<f32>, N: vec3<f32>) -> f32 {
    let Ps = P + N * TORCH_NORMAL_OFFSET;
    var s = 1.0;
    for (var i = 0u; i < torch_table.count.x; i = i + 1u) {
        if (torch_table.positions[i].w <= 0.0) { continue; }
        let face = torch_face(Ps - torch_table.positions[i].xyz);
        let layer = i * 6u + face;
        // MONKEY (live bank rank): debug samples the same compact bank as the lit path.
        let rank = countOneBits(torch_table.count.z & ((1u << i) - 1u));
        let depth_layer = select(layer, 96u + 6u * rank + face, (torch_table.count.z & (1u << i)) != 0u);
        let raw = shadow_hook::torch_map_shadow(
            torch_table.view_projs[layer], i32(depth_layer), Ps, torch_depth, torch_samp, TORCH_BIAS,
            torch_soft(), -1.0);
        // MONKEY (torch caster selection): the WEIGHTED factor, matching the real render.
        s = min(s, mix(1.0, raw, torch_table.positions[i].w));
    }
    return s;
}

// Custom vertex→fragment payload: the standard fields the splat fragment needs, PLUS `specular` — the
// per-vertex-clamped sun glint that must be Gouraud-interpolated (Q14). Same struct on both stages, so
// `@builtin(position)` is clip-position out / frag-coord in (`world_position` is unused until Step 5 fog).
struct TerrainVsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) uv_b: vec2<f32>,
    @location(4) color: vec4<f32>,
    @location(5) specular: vec3<f32>,
    // The full Gouraud diffuse — `clamp01(ambient + sun·max(N·L,0) + Σ point·att·N·L)`, evaluated
    // AND CLAMPED per VERTEX: the GL T&L locus. The clamp position is load-bearing for a hot
    // carried light (the raw committed torch is (1.4, 0.87, 0.40)): GL clamps the summed vertex
    // colour BEFORE interpolation, so an over-gamut torch saturates one vertex and Gouraud falls
    // linearly away from it. Interpolating the raw sum and clamping per pixel — what we did
    // before — keeps pixels pinned at white far from the hot vertex: a wide flat plateau with a
    // hard knee, the director's "hard square". Sun N·L moving per-pixel → per-vertex is the
    // faithful direction (Step 3's own note: indistinguishable on the smooth term).
    @location(6) primary: vec3<f32>,
    // The receiver normal is carried so the real cascaded shadow lookup can apply a slope-aware
    // bias. This is forward-pass data only; the terrain shadow caster is a stock Bevy proxy.
    @location(7) world_normal: vec3<f32>,
    // MONKEY (outdoor torch shadows: terrain): WHICH <=3 exterior table entries `primary`'s point
    // term was summed from, packed 10 bits each (see `EXT_SEL_EMPTY`). FLAT, because it is a CHOICE
    // and not a quantity - interpolating three packed indices across a triangle would produce a
    // fourth, meaningless one. It lets the fragment stage re-shadow the VERTEX's own selection
    // instead of re-ranking, which is what keeps the selection popping at chunk granularity (the
    // authored behaviour) rather than drawing a hard line mid-cell.
    @location(8) @interpolate(flat) ext_sel: u32,
    // MONKEY (outdoor torch shadows: terrain): `primary` WITHOUT the point term and WITHOUT the
    // clamp - just `ambient + sun*max(N.L,0)`. `primary` is clamped per VERTEX (the GL T&L locus,
    // see its note above) and that clamp is lossy: once a hot torch has saturated a vertex there is
    // no way to subtract the point term back out in the fragment stage. So the sun/ambient half
    // rides its own slot and the night lane rebuilds `clamp(base + shadowed_points)` from it. This
    // costs 3 interstage components and is read ONLY inside the night branch; the day path still
    // reads `primary` and is therefore bit-identical. Unclamped is safe to interpolate: it is
    // linear in `N.L` and the clamp that used to bound it is re-applied per fragment.
    @location(9) base_lit: vec3<f32>,
}

// Terrain's point-light **candidacy half-width** (yd): the reference's guaranteed covered box is
// `w + 10` where `w` is the chunk's bounding-sphere radius — `sqrt(2·16.666666² + (zExtent/2)²)`,
// i.e. 23.570166 on flat ground (`0x68df70` hard-codes that same constant at `68dfac`). Relief
// widens a chunk's own `w`; we use the flat value, so on steep ground we gather very slightly
// narrower than the reference — the lights that differ are ≥33 yd out, where `att ≈ 0.02`.
const TERRAIN_REACH: f32 = 33.570166;

// The dynamic point-light term at a world-space point (decisions 0016/0273/0278, selection 0285) —
// the reference commits AT MOST THREE point lights per draw, the nearest to the receiving unit's own
// position (the gather sorts by squared distance from the unit, the commit seats GL slots 1-3 —
// wow-re `wmo-surface-dynamic-light` §4/§6). Terrain's unit is the 33.33-yd MCNK chunk, so the anchor
// is the chunk cell center under the vertex (our tiles merge chunks into one world-space mesh —
// `mcnk_cell_anchor` recovers the chunk analytically). Pass 1 selects; pass 2 evaluates only the
// selected lights at the vertex — the byte-verified falloff `1/(0.7·d + 0.03·d²)`, diffuse-only
// (committed ambient/specular are zero — VERIFIED: the staging struct's ambient/specular slots are
// explicitly zeroed at `71c7e3`-`71c7ec` right before the point loop), on the submitted vertex
// normal, no distance cutoff on the EVALUATION (a committed GL light has none; candidacy alone
// bounds the set).
//
// **Candidacy is the reference's spatial-hash sweep, not a radius** (VERIFIED, wow-re
// `terrain/scratch/terrain-dynamic-light-gather.md`): the gather walks a 20-yd-cell hash over a cell
// span `[floor((c − w − 10)/20), floor((c + w + 10)/20)]` — a **Chebyshev box**, no circular reject —
// where terrain's own `w` is the chunk's bounding-SPHERE radius `chunk+0x68`
// (`sqrt(2·16.666666² + (zExtent/2)²)` = 23.570166 on flat ground, `0x71bc30` copies it at
// `71bc47`-`71bc5f`). That gives a guaranteed covered half-width of `w + 10` = [`TERRAIN_REACH`].
// We used a 48-yd SPHERE here — an INFERRED stand-in from decision 0285, before anyone had read the
// terrain side. It over-gathered: with 29 live point lights measured around Darkshire against three
// committed slots, every extra candidate is another chance for a chunk's ≤3 set to flip as an
// emitter moves, which is chunk-shaped popping the reference never has.
//
// The M2 lane in wow_model.wgsl still uses the packed per-light range — its own `w` is 0 (a ±10 yd
// box, a *different* byte-pinned number), a change that would move the approved interior look, so it
// is the director's to weigh. The two lanes diverge on purpose; do not "fix" them back into one.
// MONKEY (outdoor torch shadows: terrain, MIRRORED from wow_model.wgsl / static_gx.wgsl - keep in
// sync): the <=3-nearest selection packed into ONE u32, three 10-bit indices (rank 0 in the low
// bits) with `EXT_SEL_EMPTY` for an unfilled rank. The point table is capped at 256 entries, so ten
// bits leaves two bits of headroom. It exists so the per-FRAGMENT shadowed term below can
// re-evaluate the VERTEX stage's choice rather than making its own - re-ranking per fragment would
// draw a hard line wherever the ranking flips, and on terrain that line would run straight through
// the middle of an MCNK cell, which is exactly the chunk-shaped popping the 0285 anchor removed.
const EXT_SEL_EMPTY: u32 = 1023u;

// The RANKING half of the old `point_light_sum`, split out verbatim: the same interior-lane skip,
// the same Chebyshev candidacy box, the same tie order, so the set a chunk selects is bit-for-bit
// what it was and the day render cannot move.
fn point_light_pick(anchor: vec3<f32>) -> u32 {
    let count = u32(wow_light.point_count.x);
    var sel = array<u32, 3>(0u, 0u, 0u);
    var sd = array<f32, 3>(1e30, 1e30, 1e30);
    for (var i = 0u; i < count; i = i + 1u) {
        // MONKEY (light lanes): skip INTERIOR fixtures. The colour row's `.w` is `0` on an exterior
        // source and the fixture's reach in yards (always ≥ 1) on one that claims a room, so
        // `> 0.5` is the lane test. Terrain is exterior by definition: an inn's candles reaching
        // the grass at the base of its wall — a warm pool on the lawn at night — is the whole bug,
        // and it is here rather than in the pack because the fixtures MUST stay packed for the
        // interior lane to light the room they belong to. Skipped BEFORE the ≤3 ranking, so an
        // interior fixture cannot even occupy a slot an outdoor fire should have had.
        if (wow_light.points[2u * i + 1u].w > 0.5) {
            continue;
        }
        let pos_range = wow_light.points[2u * i];
        let dv = pos_range.xyz - anchor;
        let d2 = dot(dv, dv);
        // The hash sweep is horizontal (the grid keys on WoW x/y = Bevy z/x) and has no vertical
        // bound; ranking below is the full 3-D distance, as `0x71bf90` does.
        if (max(abs(dv.x), abs(dv.z)) > TERRAIN_REACH) {
            continue;
        }
        if (d2 < sd[0]) {
            sd[2] = sd[1]; sel[2] = sel[1];
            sd[1] = sd[0]; sel[1] = sel[0];
            sd[0] = d2; sel[0] = i;
        } else if (d2 < sd[1]) {
            sd[2] = sd[1]; sel[2] = sel[1];
            sd[1] = d2; sel[1] = i;
        } else if (d2 < sd[2]) {
            sd[2] = d2; sel[2] = i;
        }
    }
    return select(EXT_SEL_EMPTY, sel[0], sd[0] <= 9.9e29)
        | (select(EXT_SEL_EMPTY, sel[1], sd[1] <= 9.9e29) << 10u)
        | (select(EXT_SEL_EMPTY, sel[2], sd[2] <= 9.9e29) << 20u);
}

// The EVALUATION half - the byte-verified falloff `1/(0.7d + 0.03d^2)` x `max(N.L, 0)` x the
// committed colour, in rank order, stopping at the first empty rank exactly as the old
// `sd[s] > 9.9e29` break did (an unfilled rank packs as `EXT_SEL_EMPTY`, and no real index can
// collide with it: the table caps at 256).
fn point_light_eval(sel: u32, P: vec3<f32>, N: vec3<f32>) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    for (var s = 0u; s < 3u; s = s + 1u) {
        let idx = (sel >> (10u * s)) & 1023u;
        if (idx == EXT_SEL_EMPTY) {
            break;
        }
        let to_light = wow_light.points[2u * idx].xyz - P;
        let d = length(to_light);
        let atten = 1.0 / (0.7 * d + 0.03 * d * d);
        let nl = max(dot(N, to_light / max(d, 1e-4)), 0.0);
        sum += wow_light.points[2u * idx + 1u].rgb * (atten * nl);
    }
    return sum;
}

// MONKEY (outdoor torch shadows: terrain): the SHADOWED evaluation - the SAME three entries the
// vertex picked, each multiplied by its OWN fixture's cube-map occlusion, so a crate between the
// campfire and this patch of grass darkens the campfire's term while the brazier across the road is
// untouched. This is what step (a) already did for WMO exteriors and models; without it a
// character's shadow stopped dead at the grass.
//
// Bounded at THREE iterations (the FFP commit limit), each costing a <=16-slot table scan and four
// comparison taps, and reached only under `torch_ext_on()` - so it is night-only and dead with
// `exteriorShadows 0`. `TORCH_SKIP_EPS` drops the scan entirely wherever the unshadowed term is
// already invisible, which on terrain is most of the frame (ground facing away from the fire, or
// far enough out that the falloff has taken the term below a code).
fn point_light_eval_shadowed(sel: u32, P: vec3<f32>, N: vec3<f32>) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    for (var s = 0u; s < 3u; s = s + 1u) {
        let idx = (sel >> (10u * s)) & 1023u;
        if (idx == EXT_SEL_EMPTY) {
            break;
        }
        let fixture = wow_light.points[2u * idx].xyz;
        let to_light = fixture - P;
        let d = length(to_light);
        let atten = 1.0 / (0.7 * d + 0.03 * d * d);
        let nl = max(dot(N, to_light / max(d, 1e-4)), 0.0);
        // MONKEY (torch lane perf): the occlusion is a factor on a term that is already zero on any
        // ground facing away from the fire.
        let ext_w = atten * nl;
        var occ = 1.0;
        if (ext_w > TORCH_SKIP_EPS) {
            occ = torch_terrain_shadow(fixture, P, N);
        }
        sum += wow_light.points[2u * idx + 1u].rgb * (ext_w * occ);
    }
    return sum;
}

// The MCNK chunk cell center under a world point — terrain's light-selection anchor (the terrain
// draw unit; the grid is fixed world-wide, chunk = 533.3333/16 yd, half-extent 32 tiles). WoW x/y
// are Bevy −z/−x and the grid is symmetric, so snapping Bevy x/z directly lands on the same cells.
//
// **The XY snap is VERIFIED** (wow-re `terrain-dynamic-light-gather.md`: `CMapChunk+0x5c..0x64` is
// the chunk's world AABB centre, written by the MCVT world-vert builder `0x6b0e50` at
// `6b106c`/`6b10ec`, and the 33.33-yd chunk really is the draw unit — gather → commit → draw pair
// 1:1 per record at `68478a`/`68479c`/`6847bb`/`6847c4`).
//
// **The height is KNOWN-WRONG and deliberately not fixed here.** The reference's anchor Z is the
// chunk's own mid-height `(minH + maxH)/2` over its 145 verts — one constant per chunk. We keep the
// vertex's own y, which means the anchor VARIES within a chunk, so on relief two vertices of the
// same chunk can select different light sets and seam mid-chunk — the reference cannot do that.
// Fixing it needs a per-chunk constant plumbed to the vertex stage (a custom mesh attribute + a
// `specialize` on the terrain material, which has none today), so it is scoped as its own change;
// on flat ground the two anchors coincide and nothing differs. Mirrored in wow_model.wgsl (clutter
// snaps to the same cells).
fn mcnk_cell_anchor(P: vec3<f32>) -> vec3<f32> {
    let cell = 533.33333 / 16.0;
    let half = 32.0 * 533.33333;
    let ix = floor((half + P.x) / cell);
    let iz = floor((half + P.z) / cell);
    return vec3<f32>((ix + 0.5) * cell - half, P.y, (iz + 0.5) * cell - half);
}

@vertex
fn vertex(in: Vertex) -> TerrainVsOut {
    var out: TerrainVsOut;
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);
    out.world_position =
        mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(in.position, 1.0));
    out.clip_position = position_world_to_clip(out.world_position.xyz);
    let world_normal = mesh_functions::mesh_normal_local_to_world(in.normal, in.instance_index);
    out.world_normal = world_normal;
    out.uv = in.uv;
    out.uv_b = in.uv_b;
    out.color = in.color;

    // STEP 3b: terrain SUN SPECULAR, computed PER-VERTEX (Q14). Material spec = white, shininess =
    // wow_light.light_spec.w (20), spec color = row 9 (wow_light.light_spec.rgb); H = halfway(to-light, to-eye) with
    // the LOCAL viewer (per-vertex eye dir). Clamp HERE, per vertex — rasterizer Gouraud-interpolates.
    let n = normalize(world_normal);
    let l = -normalize(wow_light.light_sun.xyz); // to-light (the sun travels along +light_sun)
    let v = normalize(view.world_position.xyz - out.world_position.xyz); // to-eye (local viewer)
    let h = normalize(l + v);
    let ndoth = max(dot(n, h), 0.0);
    out.specular = clamp(wow_light.light_spec.rgb * pow(ndoth, wow_light.light_spec.w), vec3<f32>(0.0), vec3<f32>(1.0));

    // STEP 3 at the faithful locus — the WHOLE diffuse sum, clamped HERE per vertex (see the
    // struct note). Sun: `ambient + diffuse·max(N·L,0)` on the MCNR normal; points: the ≤3-nearest
    // committed lights of this vertex's MCNK chunk cell — the terrain draw unit (0285) — at the
    // byte-verified `1/(0.7d + 0.03d²)`, raw over-gamut colours in.
    let ndotl = max(dot(n, l), 0.0);
    // MONKEY (outdoor torch shadows: terrain): the selection is PUBLISHED so the fragment stage can
    // shadow these same three entries at night without re-ranking. One table walk still, not two -
    // the sum was always `eval(pick(..))`, it is just no longer inlined.
    let sel = point_light_pick(mcnk_cell_anchor(out.world_position.xyz));
    out.ext_sel = sel;
    let points = point_light_eval(sel, out.world_position.xyz, n);
    let sun_lighting = wow_light.light_diffuse.rgb * ndotl;
    // The sun/ambient half on its own slot, UNCLAMPED (see `base_lit`) - the night lane's only way
    // back to a `primary` that does not already have the unshadowed point term baked into it.
    out.base_lit = wow_light.light_ambient.rgb + sun_lighting;
    out.primary = clamp(
        wow_light.light_ambient.rgb + sun_lighting + points,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    return out;
}

@fragment
fn fragment(in: TerrainVsOut) -> @location(0) vec4<f32> {
    // HARD FAR-CLIP WALL (faithful `farclip`): the reference clips the detailed world at its projection
    // far plane (~777 yd). Geometry beyond is GPU-clipped per-pixel, so a tall object reveals
    // closest-part-first as you approach and the sky/WDL shows behind it. We reproduce that with a
    // per-fragment discard on PLANAR eye-Z (same coordinate as the fog) beyond `fog_params.w` = farclip
    // (0 ⇒ disabled). WDL/sky use other shaders, so they still render to the horizon behind the wall.
    if (wow_light.fog_params.w > 0.0) {
        let clip_z = -(view.view_from_world * vec4<f32>(in.world_position.xyz, 1.0)).z;
        if (clip_z > wow_light.fog_params.w) {
            discard;
        }
    }

    // Per-chunk array indices, baked into the merged mesh: COLOR = 4 layer indices, UV1.x = alpha.
    let li = vec4<i32>(round(in.color));
    let ai = i32(round(in.uv_b.x));

    let tiled = in.uv * t.params.x;
    // Full RGBA: `.rgb` = diffuse, `.a` = the `_s` texture's per-texel SHEEN MASK (see assets.rs /
    // terrain.rs — the base `.blp` has no alpha, so we load the `_s` variant; matte where no mask).
    // `view.mip_bias` (decision 1639): a render scale below 1 doubles every derivative and so
    // picks a coarser mip, and that blur is then stretched back up by the resolve. The bias undoes
    // exactly that shift and is 0.0 at native and above, so this is an identity at scale 1.
    // Only the LAYER array takes it: `alpha_array`/`shadow_array` are one 64x64 grid per chunk,
    // built with a single mip level (terrain.rs `array_image`), so no bias could move them.
    let s0 = textureSampleBias(layer_array, splat_samp, tiled, li.x, view.mip_bias);
    let s1 = textureSampleBias(layer_array, splat_samp, tiled, li.y, view.mip_bias);
    let s2 = textureSampleBias(layer_array, splat_samp, tiled, li.z, view.mip_bias);
    let s3 = textureSampleBias(layer_array, splat_samp, tiled, li.w, view.mip_bias);
    // Alpha + shadow maps are one 64² grid per chunk in 0..1 (NOT tiled), yet they share the layer
    // textures' Repeat sampler. The old "repeat == clamp here" assumption is FALSE under LINEAR
    // filtering: at a chunk edge (uv→0/1) the bilinear footprint WRAPS to the chunk map's opposite
    // edge, blending unrelated weights → a thin seam at every chunk border (the creases; introduced
    // when 228d336 switched Nearest→Linear). Inset by half a texel so the footprint clamps instead.
    let auv = clamp(in.uv, vec2<f32>(0.5 / 64.0), vec2<f32>(1.0 - 0.5 / 64.0));
    let a = textureSample(alpha_array, splat_samp, auv, ai);

    var color = s0.rgb;
    color = mix(color, s1.rgb, a.r);
    color = mix(color, s2.rgb, a.g);
    color = mix(color, s3.rgb, a.b);

    // Sheen mask: blend the layers' `_s` alpha with the SAME splat weights as the colour, so the sun
    // specular rides only the shiny texels (stone/pebbles), not matte grass/dirt. This per-pixel mask
    // is what bounds the highlight — the real client's mask averages ≈ 0.1 (Q14); applying the sheen at
    // a uniform 1.0 is what blew out the floor. (FP: `secondary · texture[0].w`.)
    var specmask = s0.a;
    specmask = mix(specmask, s1.a, a.r);
    specmask = mix(specmask, s2.a, a.g);
    specmask = mix(specmask, s3.a, a.b);

    // STEP 3 (diffuse) arrives from the vertex stage, ALREADY summed (ambient + sun N·L + the
    // chunk's ≤3 committed point lights) and ALREADY clamped — Gouraud of the clamped vertex
    // colour, byte-faithful to GL T&L (see the `primary` struct note; the MCNR normal and the
    // DayNight sun share a space per the Phase-0 validation). The MCSH ×(0.3·s + 0.7) factor below
    // still scales the whole modulate, as the traced combine does.
    // Fetch the character coverage here, but apply it as a NEUTRAL scalar after the terrain's
    // authored lighting below. Subtracting the warm sun RGB left only Elwynn's green ambient and
    // produced a green silhouette on yellow morning ground. The era-style projected shadow is a
    // greyscale attenuation: it darkens the existing hue instead of changing it.
    // Keep the raw fetch for the specular gate below. The realtime map now contains CHARACTERS
    // only; MCSH contains the static world's baked blockers. They are independent occluders, so
    // the character map must never replace (and thereby brighten) the authored terrain bake.
    // MONKEY (shadow hook): the realtime shadow — fetch + edge fade (toward the `shadowDistance`
    // range in `_wmo_fog[1].z`) + night fade (`fog_params.z`) — is computed by `benilla::shadow_hook`.
    // Terrain keeps only its own MCSH suppression + spec gate below; `sun_shadow_strength` feeds the
    // MCSH fade-back.
    let view_z = (view.view_from_world * in.world_position).z;
    let cam_dist = distance(in.world_position.xyz, view.world_position.xyz);
    let sun_shadow_strength = wow_light.fog_params.z;
    // Hoisted out of the `realtime_shadow` call (same expression, same bits) because the torch lane
    // below needs the same normal for its normal-offset sample.
    let n_lit = normalize(in.world_normal);
    let world_shadow = shadow_hook::realtime_shadow(
        in.world_position,
        n_lit,
        view_z,
        cam_dist,
        wow_light._wmo_fog[1].z,
        sun_shadow_strength,
    );
    // MONKEY (outdoor torch shadows: terrain): the exterior point term, CAST-SHADOWED at night.
    //
    // `in.primary` is the Gouraud (per-vertex, per-vertex-CLAMPED) diffuse, and it stays the only
    // thing this lane reads by day. After dark the same three entries the vertex picked
    // (`in.ext_sel`) are re-evaluated PER FRAGMENT with each fixture's own cube-map occlusion folded
    // in, re-summed onto the sun/ambient half (`in.base_lit`) and re-clamped here - the clamp has to
    // move to the fragment because the shadow is a per-fragment quantity, and clamping the sum is
    // still the faithful GL order (saturate the light sum, then modulate the texture).
    //
    // Written as a BLEND rather than a swap, for the same two reasons the model and WMO lanes are:
    //   - `fog_params.z` is EXACTLY 1.0 whenever the sun is above the daylight threshold
    //     (`global_light::sun_shadow_strength` is a smoothstep that saturates), so `ext_night_w` is
    //     exactly 0, the branch is not entered, and DAYTIME IS THE SAME BITS - not "a mix that ought
    //     to round back to `in.primary`". Nothing here is evaluated by day at any cost.
    //   - at dusk the shadow, and the Gouraud->per-fragment change of the term itself, arrive on the
    //     same clock the sun shadows leave on instead of snapping at a threshold.
    // `torch_ext_on()` is the CPU's one-bit verdict (`exteriorShadows` AND night AND at least one
    // promoted exterior fixture), so with the cvar off this is dead too.
    var primary = in.primary;
    let ext_night_w = select(0.0, clamp(1.0 - sun_shadow_strength, 0.0, 1.0), torch_ext_on());
    if (ext_night_w > 0.0) {
        let primary_night = clamp(
            in.base_lit + point_light_eval_shadowed(in.ext_sel, in.world_position.xyz, n_lit),
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
        primary = mix(in.primary, primary_night, ext_night_w);
    }

    // STEP 4: MCSH baked shadow. On the reference path (pixelShaders+specular) terrain is ONE pass
    // through `terrainp_s.bls`, which carries the MCSH bit in the blend texture's alpha (1.0 lit / 0.0
    // shadowed) and uses it TWO ways (Q15) — NOT the Q11 separate ambient-tint overlay (that's the
    // legacy/non-pixelShaders path). Our `shadow_array` R8 is 255=shadowed/0=lit, so `shadow_lit =
    // 1 - mcsh`. Chunks with no MCSH map (`uv_b.y < 0`) are fully lit.
    var shadow_lit = 1.0;
    let si = in.uv_b.y;
    if (si >= 0.0) {
        let mcsh = textureSample(shadow_array, splat_samp, auv, i32(round(si))).r;
        shadow_lit = 1.0 - mcsh;
    }
    // MONKEY (world shadows): when the realtime WORLD-shadow lane is active (`worldShadows` on), the
    // BAKED MCSH terrain shadows are redundant — the realtime map now shadows the static world
    // (trees + buildings) too — and keeping both double-shadows. So drop MCSH then. This keys on the
    // world lane flag (`sh_c16.w`, packed by `global_light::build_light_data`), NOT on the mere
    // presence of a directional light: the shared shadow sun ALSO exists for CHARACTER-only shadows,
    // and those must leave the world's baked MCSH intact. With the world lane off, `shadow_lit_eff`
    // keeps MCSH and only the realtime `character_shadow_term` adds the dynamic character shadow.
    // The world lane suppresses baked MCSH — but only by `sun_shadow_strength`, so as the realtime
    // world shadow fades at night the baked MCSH fades back IN. By day (strength 1) MCSH is fully
    // replaced; at night (strength 0) the authored bake returns; dusk crossfades. Character-only
    // (lane flag off) never suppresses MCSH.
    let world_shadow_lane = wow_light.sh_c16.w > 0.5;
    let mcsh_suppress = select(0.0, sun_shadow_strength, world_shadow_lane);
    let shadow_lit_eff = mix(shadow_lit, 1.0, mcsh_suppress);
    let spec_gate = min(shadow_lit, world_shadow);
    let character_shadow_term = mix(SHADOW_SUN_FLOOR, 1.0, world_shadow);

    // The faithful `terrainp_s` combine (Q15):
    //   diffuse  = tex · primary · (0.3·shadow + 0.7)   → a flat −30% in shadow, NO colour tint
    //   specular = per-vertex sheen · gloss_mask · shadow → gated to ZERO in shadow (no sheen in shade)
    // (`tex·primary` is the MODULATE-1× diffuse; the sheen is added after, separate-specular.) Then
    // LDR-clamp; gamma/byte throughout; raw gamma out (GAMMA LANE, 0161).
    let diffuse_term = color * primary * (0.3 * shadow_lit_eff + 0.7) * character_shadow_term;
    let spec_term = in.specular * specmask * spec_gate;
    var tuned = clamp(diffuse_term + spec_term, vec3<f32>(0.0), vec3<f32>(1.0));

    // MONKEY (torch debug, interiorDebug 2 on TERRAIN): the ground's OWN cube-map sampling as
    // greyscale, the same instrument static_gx and wow_model already paint on walls and bodies. The
    // mode rides `wmo_fog_params.w` packed as `1 + debug` (rows 18-19 here, hence `_wmo_fog[1]`), so
    // a zero - the interior lane off entirely - decodes to mode 0 and this is dead. Only mode 2 is
    // answered: 1/3/4 are questions about the INTERIOR lane and the WMO group classes, and terrain
    // belongs to neither, so leaving it normally shaded is what makes those three views readable.
    // Placed before the fog so a distant reading still fogs like the surfaces beside it.
    let idbg = u32(max(wow_light._wmo_fog[1].w - 1.0, 0.0) + 0.5);
    if (idbg == 2u) {
        tuned = vec3<f32>(torch_terrain_debug_factor(in.world_position.xyz, n_lit));
    }

    // STEP 5: gamma-space linear fog (q6). Applied AFTER tone/curve (so the diagnostic knobs see
    // the unfogged lit pixel) and BEFORE the viz bypass + raw output. The fog coordinate is
    // PLANAR eye-Z (view-space depth along the camera forward axis), NOT radial distance: the
    // reference VS computes `fogcoord` as a linear function of a single axial coordinate (eye-Z) —
    // VERIFIED from the host GLSL fog source in two apitraces (WoW.5/WoW.8). Radial `length()` is
    // ≥ planar everywhere off-axis, so it OVER-fogs the screen edges (worst in short-fog zones like
    // Duskwood, +0.26–0.32 fog factor at the corners). The per-vertex T&L original and this
    // per-pixel form land on the same byte under LINEAR fog (the factor interpolates exactly), and
    // the per-pixel form is what survives Bevy's mesh interpolators without a custom slot.
    if (wow_light.fog_color.w > 0.5) {
        let eye_z = -(view.view_from_world * vec4<f32>(in.world_position.xyz, 1.0)).z;
        let denom = max(wow_light.fog_params.y - wow_light.fog_params.x, 0.001);
        let factor = clamp((wow_light.fog_params.y - eye_z) / denom, 0.0, 1.0);
        tuned = mix(wow_light.fog_color.xyz, tuned, factor);
    }

    // GAMMA LANE (decision 0161): the framebuffer HOLDS gamma bytes — all blending happens in
    // gamma exactly like the reference's byte framebuffer; the frame's ONE decode lives in the
    // FFXGlow combine (the last node). Output raw.
    return vec4<f32>(tuned, 1.0);
}
