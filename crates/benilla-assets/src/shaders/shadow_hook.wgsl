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
    mesh_view_bindings::{lights, clusterable_objects},
    mesh_view_types::POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT,
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

// MONKEY (torch shadows #2): the realtime POINT-light shadow factor at `sample_pos` — 1.0 = lit,
// 0.0 = fully shadowed. Scans the view's clusterable objects for shadow-casting point lights (the
// torch-shadow lane promotes the nearest interior fixture to one, `benilla_app::torch_shadow`) and,
// for any whose radius covers the fragment, samples its cube shadow map. `min` over them so the
// darkest occluder wins. A no-op indoors when no such light exists (returns 1.0). Called by the
// INTERIOR receiver paths (the exterior sun's `realtime_shadow` is gated out of interiors).
fn torch_shadow(sample_pos: vec4<f32>, normal: vec3<f32>) -> f32 {
    var shadow = 1.0;
    let count = arrayLength(&clusterable_objects.data);
    for (var i = 0u; i < count; i = i + 1u) {
        if ((clusterable_objects.data[i].flags & POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u) {
            let pr = clusterable_objects.data[i].position_radius;
            if (distance(pr.xyz, sample_pos.xyz) < pr.w) {
                shadow = min(shadow, shadows::fetch_point_shadow(i, sample_pos, normal));
            }
        }
    }
    return shadow;
}
