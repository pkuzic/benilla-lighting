#define_import_path benilla::wet_hook

// MONKEY (wet): rain on surfaces — the ONE place the wet look is computed (the `shadow_hook.wgsl`
// pattern). Receivers (terrain, static_gx, wow_model) call it with one-line hooks and pass the
// MonkeyFrame (its `wet_a` row: `rain_rate, wetness, ripple_time_s, snow`, written by
// `benilla-world/src/weather/wetness.rs`). A zero wetness (dry weather, or `rainSurfaces 0`) returns
// the inputs untouched, so the dry image is bit-identical.
//
// Sky exposure, phase 1: the receiver passes `exposure` = 0 for interior-class / unlit / emissive
// surfaces and 1 otherwise; the normal then takes undersides out (dry), walls half wet, tops fully
// wet and glossy. There is no rain-occlusion map yet, so an exterior surface under a porch or a
// bridge still reads wet.

#import benilla::monkey_frame::MonkeyFrame

struct WetSurface {
    albedo: vec3<f32>,
    // Gloss weight for `wet_sheen`: 0 dry, ~1 a soaked top face, up to 2 in a terrain puddle (the
    // part above 1 is standing water: a flat mirror of the sky).
    boost: f32,
}

// How wet a surface is from its facing: none below, half on walls, full on tops.
fn wet_amount(n: vec3<f32>, exposure: f32, wetness: f32) -> f32 {
    let facing = smoothstep(-0.2, 0.25, n.y);
    return clamp(wetness, 0.0, 1.0) * clamp(exposure, 0.0, 1.0) * facing
        * mix(0.5, 1.0, smoothstep(0.3, 0.8, n.y));
}

// Wet albedo: darker (water fills the micro-pores) and a touch more saturated; the gloss weight
// is kept for up-facing faces only.
fn wet_surface(albedo: vec3<f32>, n: vec3<f32>, world_pos: vec3<f32>, exposure: f32,
    mf: MonkeyFrame) -> WetSurface {
    let wet_row = mf.wet_a;
    var o: WetSurface;
    o.albedo = albedo;
    o.boost = 0.0;
    if (wet_row.y <= 0.0 || exposure <= 0.0) {
        return o;
    }
    let w = wet_amount(n, exposure, wet_row.y);
    let luma = dot(albedo, vec3<f32>(0.299, 0.587, 0.114));
    let sat = max(mix(vec3<f32>(luma), albedo, 1.0 + 0.3 * w), vec3<f32>(0.0));
    o.albedo = sat * (1.0 - 0.36 * w);
    o.boost = w * smoothstep(0.35, 0.9, n.y);
    return o;
}

fn wet_hash(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2<f32>(0.1031, 0.1030));
    let r = q + dot(q, q.yx + 33.33);
    return fract((r.x + r.y) * r.x);
}

fn wet_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(wet_hash(i), wet_hash(i + vec2<f32>(1.0, 0.0)), u.x),
        mix(wet_hash(i + vec2<f32>(0.0, 1.0)), wet_hash(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
}

// Terrain bonus: puddle patches on flat ground once it is well soaked — darker, and a stronger
// gloss weight so they read as standing water catching the sky.
fn wet_puddles(s: WetSurface, n: vec3<f32>, world_pos: vec3<f32>, mf: MonkeyFrame) -> WetSurface {
    let wet_row = mf.wet_a;
    var o = s;
    if (wet_row.y <= 0.0) {
        return o;
    }
    let xz = world_pos.xz;
    let field = 0.65 * wet_noise(xz * 0.17) + 0.35 * wet_noise(xz * 0.53 + 17.0);
    let level = smoothstep(0.94, 0.99, n.y);
    // Puddles grow with the wetness: the threshold falls as the ground soaks.
    let edge = mix(0.8, 0.6, smoothstep(0.4, 1.0, wet_row.y));
    let puddle = smoothstep(edge, edge + 0.06, field) * level * smoothstep(0.35, 0.8, wet_row.y);
    o.albedo = o.albedo * (1.0 - 0.45 * puddle);
    o.boost = max(o.boost, 2.0 * puddle);
    return o;
}

// The wet sheen, added to the lit gamma colour: a tight sun highlight (shadowed) and a Fresnel
// reflection of the sky/fog colour. Zero when `boost` is zero.
fn wet_sheen(boost: f32, n: vec3<f32>, to_view: vec3<f32>, to_light: vec3<f32>,
    sun_rgb: vec3<f32>, sky_rgb: vec3<f32>, shadow: f32) -> vec3<f32> {
    if (boost <= 0.0) {
        return vec3<f32>(0.0);
    }
    let h = normalize(to_light + to_view);
    let nl = max(dot(n, to_light), 0.0);
    let spec = pow(max(dot(n, h), 0.0), 80.0) * smoothstep(0.0, 0.15, nl);
    let fres = 0.03 + 0.97 * pow(1.0 - clamp(dot(n, to_view), 0.0, 1.0), 5.0);
    let gloss = min(boost, 1.0);
    let mirror = max(boost - 1.0, 0.0);
    return sun_rgb * ((0.9 * gloss + 1.2 * mirror) * spec * shadow)
        + sky_rgb * (gloss * min(0.25 * fres, 0.2) + mirror * (0.22 + 0.3 * fres));
}
