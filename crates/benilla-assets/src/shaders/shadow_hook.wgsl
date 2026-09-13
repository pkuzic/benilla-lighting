#define_import_path benilla::shadow_hook

// MONKEY (shadow extension hook): the ONE place the realtime directional-shadow term is computed.
//
// benilla's receivers — `terrain.wgsl`, `wow_model.wgsl`, `static_gx.wgsl` — call `realtime_shadow`
// instead of each hand-rolling the fetch + fades. Adding or changing the realtime shadow now touches
// THIS file only; the three receiver shaders (the ones that blanked the terrain when edited by hand)
// stay untouched. The receivers keep their own downstream gating (terrain's MCSH suppression + spec
// gate, the model's interior/rig-skin exclusion + ambient-preserving apply, static_gx's ambient arm)
// — those are lighting composition, not shadow sampling, so they belong to each receiver.
//
// It is a no-op whenever no shadow sun exists: the only Bevy `DirectionalLight` benilla ever spawns
// is the shadow rig's (`benilla_app::shadow_core`), so `lights.n_directional_lights == 0` exactly
// when shadows are off, and the loop below never runs. No shader-def toggle is needed.

#import bevy_pbr::{
    shadows,
    mesh_view_bindings::lights,
}

// The realtime shadow lightens over the last SHADOW_EDGE_BAND yards of the cascade's max distance so
// it fades in rather than popping at the resolve boundary.
const SHADOW_EDGE_BAND: f32 = 14.0;

// The realtime directional-shadow factor at `sample_pos`: 1.0 = lit, 0.0 = fully shadowed.
//
// - `sample_pos`   world-space point to sample the shadow map at — a per-fragment position, or a
//                  unit's nudged rig anchor (the caller chooses; that is the rig-skin decision).
// - `normal`       receiver normal, for the slope-aware depth bias.
// - `view_z`       view-space z, for cascade selection.
// - `cam_dist`     distance from the camera to the receiver fragment, for the EDGE fade.
// - `shadow_range` the cascade's max distance (the `shadowDistance` slider) — where the edge fade ends.
// - `night`        the sun-height strength (0 at night .. 1 by day) — the NIGHT fade.
//
// Both fades lighten the result toward 1.0 (no shadow). The caller passes `shadow_range`/`night`
// from its own light buffer (the receivers declare that buffer three incompatible ways, so the hook
// takes the scalars as arguments rather than reading it).
fn realtime_shadow(
    sample_pos: vec4<f32>,
    normal: vec3<f32>,
    view_z: f32,
    cam_dist: f32,
    shadow_range: f32,
    night: f32,
) -> f32 {
    var shadow = 1.0;
    if (lights.n_directional_lights > 0u) {
        for (var light_id = 0u; light_id < lights.n_directional_lights; light_id = light_id + 1u) {
            if ((lights.directional_lights[light_id].flags & 1u) != 0u) {
                shadow = shadows::fetch_directional_shadow(light_id, sample_pos, normal, view_z);
                break;
            }
        }
    }
    let edge_fade = smoothstep(shadow_range - SHADOW_EDGE_BAND, shadow_range, cam_dist);
    return 1.0 - (1.0 - shadow) * night * (1.0 - edge_fade);
}

// MONKEY (torch shadows Phase 1): the point/cluster-based torch shadow is DEAD. benilla's world
// camera sets `ClusterConfig::None`, which starves `clusterable_objects` AND the point-shadow prep,
// so `fetch_point_shadow`/the clusterable scan can never fire. The replacement is a from-scratch
// depth map (`benilla_world::static_gx::torch_depth` + `torch_map_shadow` below), sampled by the
// static_gx interior lane through its own group-3 bindings, and (Phase 3A) by `wow_model.wgsl`'s
// entity lane through its material's own bindings 91/92/93 — both call `torch_map_shadow` below.
// These two functions are kept only so `interior_debug_override` (mode 2) compiles; they return
// "unshadowed" and no receiver's lighting path calls them any more.
fn torch_shadow(sample_pos: vec4<f32>, normal: vec3<f32>) -> f32 {
    return 1.0;
}

fn torch_shadow_for(light_pos: vec3<f32>, sample_pos: vec4<f32>, normal: vec3<f32>) -> f32 {
    return 1.0;
}

// MONKEY (torch shadows Phase 1): the old clusterable-caster count diagnostic. The clustering is
// dead (see above), so this is now a constant 0 — kept only so `interior_debug_override` compiles.
fn torch_caster_count() -> u32 {
    return 0u;
}

// MONKEY (pcss): the contact-hardening constants.
//
// `TORCH_DEPTH_B` is the reverse-Z projection's depth OFFSET, mirrored from
// `benilla_app::torch_shadow::{reverse_z_perspective, cube_view_projs}` (near 0.1, far TORCH_RANGE
// 48 — every matrix this function is ever handed is one of those six cube faces). Under reverse-Z
// the stored depth is an AFFINE function of the RECIPROCAL distance along the face axis:
// `z = A/t + B`, `A = near*far/(far-near)`, `B = -near/(far-near)`. So `z - B` is proportional to
// `1/t`, and the one quantity a penumbra needs — the ratio of two distances — is
// `(z_blocker - B) / (z_receiver - B)` with `A` cancelling: no linearisation, no extra uniform, one
// subtraction per term. A constant rather than something decoded out of `view_proj` because the
// w-row of a look-at product carries the eye translation, so `B` cannot be read off a single
// element of that matrix without dividing two of them — and a wrong `B` would silently mis-size
// every penumbra instead of failing.
const TORCH_DEPTH_B: f32 = -0.1 / 47.9;
// How far out (in `soft` texels) the blocker search looks. Two: wide enough to find the caster of
// an edge a texel or two away, narrow enough that a fragment well inside a lit area still reads
// four empty texels and takes the cheap path.
const TORCH_PCSS_SEARCH: f32 = 2.0;
// Penumbra gain — extra `soft` radii per unit of `(d_receiver - d_blocker)/d_blocker`. A receiver
// twice as far from the fixture as its blocker (ratio 1) is filtered three times as wide; the
// Darkmoon tent (post ~2 yd from the torch, grass ~10) saturates the clamp below, and a chair leg
// resting ON the floor (ratio ~0.05) stays within 10 % of the hard kernel — the whole point.
const TORCH_PCSS_K: f32 = 2.0;
// …clamped at four radii. Past that the 8 taps are too sparse for the area they cover and the
// penumbra starts to band, and the shadow there is faint enough that widening it further is
// invisible anyway.
const TORCH_PCSS_MAX: f32 = 4.0;
// Above this filter radius (texels) the 4-tap box is too sparse and earns the second ring.
const TORCH_PCSS_WIDE: f32 = 1.5;

// MONKEY (pcss): one RAW depth texel of a torch map — the blocker search's read. `textureLoad`
// takes no sampler and imposes no uniformity requirement, which is what makes the search legal
// inside the receivers' per-fixture loop. The clamp is the border rule: a search that walks off the
// face re-reads its edge texel, the same answer the comparison sampler's ClampToEdge address mode
// gives the PCF taps beside it.
fn torch_map_depth(
    depth_tex: texture_depth_2d_array,
    layer: i32,
    uv: vec2<f32>,
    dims: vec2<f32>,
) -> f32 {
    let c = clamp(vec2<i32>(floor(uv * dims)), vec2<i32>(0, 0), vec2<i32>(dims) - vec2<i32>(1, 1));
    return textureLoad(depth_tex, c, layer, 0);
}

// MONKEY (torch shadows Phase 1): the depth-map projector. Because `shadow_hook` is imported by
// receivers WITHOUT the group-3 torch bindings (wow_model, terrain), the texture + comparison
// sampler come in AS PARAMETERS — the caller supplies them only where they are bound (static_gx).
//   - `view_proj`  the fixture's down-looking reverse-Z matrix.
//   - `layer`      which array layer (fixture index) to sample.
//   - `world_pos`  the receiving fragment's world position.
//   - `depth_tex`/`comp`  the group-3 depth array + `GreaterEqual` comparison sampler.
//   - `bias`       reverse-Z receiver bias (nudges the compare ref up to kill self-shadow acne).
//   - `soft`       MONKEY (torch caster selection): the BASE PCF tap-radius scale
//                  (`interiorShadowSoft`, 0.5..3, carried in the table's `count.y` LOW half as
//                  `soft x 100`). The four taps sit at ±0.5 texel × this; 1 is the historical
//                  half-texel box. MONKEY (pcss): it is now the radius at a CONTACT (a receiver
//                  touching its blocker) — the contact-hardening search below grows it with the
//                  blocker distance, so `soft` sets how sharp the sharpest edge in the scene is
//                  and every penumbra scales off it.
//   - `fade`       MONKEY (outdoor torch shadows): WHICH far fade ends the shadow.
//                  **Negative = the interior/legacy behaviour, bit-for-bit** — the reverse-Z
//                  far-plane fade computed from `ndc.z` below. Zero-or-positive = the CALLER's own
//                  fade weight, used verbatim (0 = no shadow left, 1 = full strength).
//
//                  Why an override at all: the projection is near/far `0.1/48`
//                  (`torch_shadow::cube_view_projs`), so under reverse-Z `ndc.z` is
//                  `near·(far−t)/((far−near)·t)` = **0.0079 at t = 10 yd**, which
//                  `smoothstep(0, 0.06, ·)` turns into a 5 % shadow. For a candle whose pool is
//                  3-4 yd that is a correct soft end; for a campfire whose pool is 15-25 yd it
//                  erases the shadow exactly where the shadow IS the effect. The fix cannot live
//                  in here — it wants to be a function of WORLD distance from the fixture, and
//                  this function is deliberately a pure projector that never sees the fixture — so
//                  the exterior lane computes `1 − smoothstep(0.8R, R, d)` at its call site.
// Returns 1.0 outside the frustum / behind the light; a PCF factor otherwise (0 = shadowed).
//
// MONKEY (pcss): CONTACT-HARDENING. The old kernel was one fixed ±0.5·soft texel box everywhere,
// which is why a torch beside a Darkmoon tent post threw the canvas onto the grass fifteen yards
// away as a razor-edged cut-out: a real penumbra grows with the receiver's distance FROM ITS
// BLOCKER, and a constant kernel has no way to know that distance. The three steps below recover
// it from the map itself — search for blockers, estimate how far in front of the receiver they
// are, widen the filter by that much — so a chair leg stays crisp where it meets the floor while
// the tent's edge goes soft where the canvas is metres above the grass.
//
// The blocker search reads RAW depth with `textureLoad`, not through the comparison sampler: the
// array is bound as `texture_depth_2d_array` (wgpu sample type Depth), and a Depth binding serves
// BOTH `textureSampleCompare*` (through `comp`) and an unfiltered `textureLoad` that takes no
// sampler at all — so no binding, layout or pipeline changed. That is worth insisting on, because
// the alternative approximation (infer the blocker from the FRACTION of blocked compares at two or
// three radii) only ever answers "how deep inside the shadow am I", never "how far in front of me
// is the caster", and those two differ by exactly the quantity a penumbra is made of.
//
// Cost: 4 unfiltered loads, then 4 comparison taps (the shipped kernel, unchanged) whenever the
// search finds no blocker — which is every fully-lit fragment, i.e. most of a pool — or 8 when the
// widened radius earns them. Worst case 8 compares + 4 loads per fixture per fragment.
fn torch_map_shadow(
    view_proj: mat4x4<f32>,
    layer: i32,
    world_pos: vec3<f32>,
    depth_tex: texture_depth_2d_array,
    comp: sampler_comparison,
    bias: f32,
    soft: f32,
    fade: f32,
) -> f32 {
    let clip = view_proj * vec4<f32>(world_pos, 1.0);
    if (clip.w <= 0.0) {
        return 1.0;
    }
    let ndc = clip.xyz / clip.w;
    if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    let dims = vec2<f32>(textureDimensions(depth_tex).xy);
    let inv_dims = 1.0 / dims;
    // MONKEY (torch caster selection): the base radius, clamped off zero so a table that never got
    // a scale still samples a real (hard) box rather than four copies of one texel.
    let base = max(soft, 0.05);
    // Reverse-Z: the fragment is lit iff its own depth is at least the stored nearest depth, so the
    // compare ref is `ndc.z + bias` against `GreaterEqual`. UNCHANGED bias semantics: the same
    // biased reference decides both the blocker test and every comparison tap, so a surface can no
    // more blocker-detect itself than it could shadow itself.
    let ref_depth = ndc.z + bias;

    // MONKEY (pcss) 1/3 — BLOCKER SEARCH. Four raw reads on the diagonals of a box of half-width
    // `soft * TORCH_PCSS_SEARCH` texels, averaging the depths that lie IN FRONT of the receiver.
    // Four taps at one radius rather than a ring at several: they only have to answer "is there a
    // caster near me, and roughly how far in front", and denser searching costs more than the extra
    // precision buys on a 512² face whose texels are ~10 cm at a candle's range.
    let search = base * TORCH_PCSS_SEARCH * inv_dims;
    let b0 = torch_map_depth(depth_tex, layer, uv + vec2<f32>(-1.0, -1.0) * search, dims);
    let b1 = torch_map_depth(depth_tex, layer, uv + vec2<f32>( 1.0, -1.0) * search, dims);
    let b2 = torch_map_depth(depth_tex, layer, uv + vec2<f32>(-1.0,  1.0) * search, dims);
    let b3 = torch_map_depth(depth_tex, layer, uv + vec2<f32>( 1.0,  1.0) * search, dims);
    var blocker = 0.0;
    var blockers = 0.0;
    if (b0 > ref_depth) { blocker += b0; blockers += 1.0; }
    if (b1 > ref_depth) { blocker += b1; blockers += 1.0; }
    if (b2 > ref_depth) { blocker += b2; blockers += 1.0; }
    if (b3 > ref_depth) { blocker += b3; blockers += 1.0; }

    // MONKEY (pcss) 2/3 — PENUMBRA WIDTH. `radius` stays at `base` when the search found nothing: a
    // caster thinner than the search box (a chair leg, a tent rope, a candlestick) must NOT be
    // widened away to nothing, and such a fragment is then filtered by exactly the kernel this
    // function has always used.
    var radius = base;
    if (blockers > 0.0) {
        // `(d_receiver - d_blocker) / d_blocker`, exact — see TORCH_DEPTH_B: `z - B` is
        // proportional to `1/distance`, so the quotient of the two `z - B` terms IS the quotient of
        // the two distances, and the projection's scale factor cancels. The receiver term uses the
        // UNBIASED `ndc.z` (the bias exists to keep a surface off its own stored depth, not to move
        // the surface), and `max` only guards f32 noise: inside the frustum a detected blocker is
        // always the nearer of the two.
        let ratio = max((blocker / blockers - TORCH_DEPTH_B) / (ndc.z - TORCH_DEPTH_B) - 1.0, 0.0);
        radius = min(base * (1.0 + TORCH_PCSS_K * ratio), base * TORCH_PCSS_MAX);
    }

    // MONKEY (pcss) 3/3 — PCF at that radius. The 4-tap box below `TORCH_PCSS_WIDE` texels is the
    // shipped kernel bit-for-bit (`radius == base` reproduces the old `texel` expression exactly);
    // past it that box would alias into four separate soft bands, so a second ring of four cardinal
    // taps at the full radius fills the area in.
    let texel = inv_dims * radius;
    var sum = 0.0;
    sum += textureSampleCompareLevel(depth_tex, comp, uv + vec2<f32>(-0.5, -0.5) * texel, layer, ref_depth);
    sum += textureSampleCompareLevel(depth_tex, comp, uv + vec2<f32>( 0.5, -0.5) * texel, layer, ref_depth);
    sum += textureSampleCompareLevel(depth_tex, comp, uv + vec2<f32>(-0.5,  0.5) * texel, layer, ref_depth);
    sum += textureSampleCompareLevel(depth_tex, comp, uv + vec2<f32>( 0.5,  0.5) * texel, layer, ref_depth);
    var pcf = sum * 0.25;
    if (radius > TORCH_PCSS_WIDE) {
        sum += textureSampleCompareLevel(depth_tex, comp, uv + vec2<f32>(-1.0,  0.0) * texel, layer, ref_depth);
        sum += textureSampleCompareLevel(depth_tex, comp, uv + vec2<f32>( 1.0,  0.0) * texel, layer, ref_depth);
        sum += textureSampleCompareLevel(depth_tex, comp, uv + vec2<f32>( 0.0, -1.0) * texel, layer, ref_depth);
        sum += textureSampleCompareLevel(depth_tex, comp, uv + vec2<f32>( 0.0,  1.0) * texel, layer, ref_depth);
        pcf = sum * 0.125;
    }
    // Phase 5: the six cube faces tile the whole sphere, so there is no cone edge to soften — a fade
    // there would punch a lit seam along every face border. Only the far plane fades (reverse-Z:
    // ndc.z → 0 at far), so a shadow at the fixture's range limit ends softly.
    // MONKEY (outdoor torch shadows): …unless the caller supplied its own (see `fade`). `select`
    // rather than a branch so both arms cost the same and the interior arm is the identical
    // expression it always was.
    let ndc_fade = smoothstep(0.0, 0.06, ndc.z);
    return mix(1.0, pcf, select(fade, ndc_fade, fade < 0.0));
}

// MONKEY (interior debug): the interior lane's diagnostic overlay. `mode` is decoded from the
// receiver's `wmo_fog_params.w` (`1 + debug`); the receivers call this at the END of their interior
// branch so ONLY interior-lit fragments are recoloured — a wrongly-classified outdoor entity then
// stands out against the normally-shaded world. `mode == 0` returns the real `rgb` unchanged.
//   1 CLASSIFICATION → solid green (this fragment took the interior lane)
//   2 SHADOW         → the torch-shadow factor as greyscale (black shadowed .. white lit)
//   3 CASTER COUNT   → red 0 / green 1 / blue 2 / white 3+ shadow proxies visible to this view
fn interior_debug_override(
    mode: u32,
    rgb: vec3<f32>,
    sample_pos: vec4<f32>,
    normal: vec3<f32>,
) -> vec3<f32> {
    if (mode == 1u) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    if (mode == 2u) {
        return vec3<f32>(torch_shadow(sample_pos, normal));
    }
    if (mode == 3u) {
        let n = torch_caster_count();
        if (n == 0u) { return vec3<f32>(1.0, 0.0, 0.0); }
        if (n == 1u) { return vec3<f32>(0.0, 1.0, 0.0); }
        if (n == 2u) { return vec3<f32>(0.0, 0.0, 1.0); }
        return vec3<f32>(1.0, 1.0, 1.0);
    }
    return rgb;
}
