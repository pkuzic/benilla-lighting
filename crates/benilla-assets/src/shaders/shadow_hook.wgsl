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

// MONKEY (torch shadows Phase 1): the depth-map projector. Because `shadow_hook` is imported by
// receivers WITHOUT the group-3 torch bindings (wow_model, terrain), the texture + comparison
// sampler come in AS PARAMETERS — the caller supplies them only where they are bound (static_gx).
//   - `view_proj`  the fixture's down-looking reverse-Z matrix.
//   - `layer`      which array layer (fixture index) to sample.
//   - `world_pos`  the receiving fragment's world position.
//   - `depth_tex`/`comp`  the group-3 depth array + `GreaterEqual` comparison sampler.
//   - `bias`       reverse-Z receiver bias (nudges the compare ref up to kill self-shadow acne).
//   - `soft`       MONKEY (torch caster selection): the PCF tap-radius scale (`interiorShadowSoft`,
//                  0.5..3, carried in the table's `count.y` as `soft x 100`). The four taps sit at
//                  ±0.5 texel × this; 1 is the historical half-texel box. A candle cluster casts
//                  many overlapping penumbra-less edges, and widening the kernel is the cheapest
//                  softening available here (a real penumbra would need the blocker distance and a
//                  variable kernel — not worth a second sampling pass on a 512² face).
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
// Returns 1.0 outside the frustum / behind the light; a 4-tap PCF factor otherwise (0 = shadowed).
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
    // MONKEY (torch caster selection): the tap radius is `soft` texels, clamped off zero so a
    // table that never got a scale still samples a real (hard) 4-tap box rather than four copies
    // of one texel.
    let texel = (1.0 / vec2<f32>(textureDimensions(depth_tex).xy)) * max(soft, 0.05);
    // Reverse-Z: the fragment is lit iff its own depth is at least the stored nearest depth, so the
    // compare ref is `ndc.z + bias` against `GreaterEqual`.
    let ref_depth = ndc.z + bias;
    var sum = 0.0;
    sum += textureSampleCompareLevel(depth_tex, comp, uv + vec2<f32>(-0.5, -0.5) * texel, layer, ref_depth);
    sum += textureSampleCompareLevel(depth_tex, comp, uv + vec2<f32>( 0.5, -0.5) * texel, layer, ref_depth);
    sum += textureSampleCompareLevel(depth_tex, comp, uv + vec2<f32>(-0.5,  0.5) * texel, layer, ref_depth);
    sum += textureSampleCompareLevel(depth_tex, comp, uv + vec2<f32>( 0.5,  0.5) * texel, layer, ref_depth);
    // Phase 5: the six cube faces tile the whole sphere, so there is no cone edge to soften — a fade
    // there would punch a lit seam along every face border. Only the far plane fades (reverse-Z:
    // ndc.z → 0 at far), so a shadow at the fixture's range limit ends softly.
    // MONKEY (outdoor torch shadows): …unless the caller supplied its own (see `fade`). `select`
    // rather than a branch so both arms cost the same and the interior arm is the identical
    // expression it always was.
    let ndc_fade = smoothstep(0.0, 0.06, ndc.z);
    return mix(1.0, sum * 0.25, select(fade, ndc_fade, fade < 0.0));
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
