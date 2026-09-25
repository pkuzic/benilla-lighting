#define_import_path benilla::fog_hook

// MONKEY (p0 fog hook): the ONE place the distance fog is computed.
//
// Every receiver — terrain, wow_model, static_gx, liquid, enhanced_water, wow_effect, wdl — calls
// in here instead of hand-rolling the fog law, so a new fog model touches this file only (the
// `shadow_hook.wgsl` pattern). Pure functions: the receivers bind the shared light buffer at
// different slots (material binding 90, group 2 binding 0, group 1 binding 2, binding 105), so
// every value arrives as an argument and this file owns no bindings.
//
// The classic law is the reference's GL_LINEAR fog on PLANAR eye-Z in gamma space (terrain.wgsl
// has the full note): `factor = clamp((end - eye_z) / max(end - start, 0.001), 0, 1)`, then
// `mix(fog_rgb, rgb, factor)`. The expressions below are the ones the receivers carried, in the
// same order, so the classic result is bit-identical.
//
// Extension point for the FOG lane: `fog_sample` branches on `mf.fog_d.x` (`MonkeyFrame.fog_model`,
// 0 classic / 1 modern). The modern arm currently returns the classic result.

#import benilla::monkey_frame::MonkeyFrame

// The classic survival factor: 1 = no fog, 0 = fully fogged.
fn fog_linear(eye_z: f32, span: vec2<f32>) -> f32 {
    let denom = max(span.y - span.x, 0.001);
    return clamp((span.y - eye_z) / denom, 0.0, 1.0);
}

// The fog a surface stands in: rgb = the fog colour, w = how much of the surface survives (1 = no
// fog). `fog_rgb` is the colour the caller's policy picked; `scene_colour` is true when that is the
// scene's own fog colour (false for the forced black/white/grey policies of additive, Mod and Mod2x
// batches, which a new model may attenuate but must not recolour). `span` = (start, end) yd,
// `eye_z` = planar eye depth (yd), `world_pos` / `camera_pos` in Bevy world space.
fn fog_sample(
    fog_rgb: vec3<f32>,
    span: vec2<f32>,
    eye_z: f32,
    world_pos: vec3<f32>,
    camera_pos: vec3<f32>,
    scene_colour: bool,
    mf: MonkeyFrame,
) -> vec4<f32> {
    let classic = vec4<f32>(fog_rgb, fog_linear(eye_z, span));
    if (mf.fog_d.x > 0.5) {
        // FOG lane: the modern model (height / sun / end-colour / curve fog) goes here, reading
        // mf.fog_a..fog_d. Until it lands, the modern arm is the classic law.
        return classic;
    }
    return classic;
}

// Fog `rgb` in place: `mix(fog colour, rgb, survival)`. Same arguments as `fog_sample`.
fn apply_fog(
    rgb: vec3<f32>,
    fog_rgb: vec3<f32>,
    span: vec2<f32>,
    eye_z: f32,
    world_pos: vec3<f32>,
    camera_pos: vec3<f32>,
    scene_colour: bool,
    mf: MonkeyFrame,
) -> vec3<f32> {
    let f = fog_sample(fog_rgb, span, eye_z, world_pos, camera_pos, scene_colour, mf);
    return mix(f.rgb, rgb, f.w);
}
