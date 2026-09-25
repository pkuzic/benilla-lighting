// The cloud dome. Its colour is CPU-side as in the reference: the kernel's `0x6cfb00` port builds
// the image the reference binds as its texture (`0x58ac70`). This stage applies the dome's rim
// fade (`0x6d0530`) and blends the raw gamma texels premultiplied. Depth is the far pin in
// `sky_vertex.wgsl`.
//
// MONKEY (sky): at `skyQuality` 2 the texels get GPU detail and a short sun march.
// Ported from WarcraftXL (https://github.com/WarcraftXL) by iThorgrim — module wxl-retail-clouds, Clouds.cpp.

#import bevy_pbr::forward_io::VertexOutput
#import benilla_world::sky_fx

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var cloud_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var cloud_samp: sampler;

// MONKEY (sky): `fx.x` the sky tier, `.y` the sky clock (s), `.zw` the unit direction toward the
// glow body on the sheet (u = world x, v = world z); `lit.rgb` the Light.dbc cloud sun colour
// (gamma), `.w` the march strength (the body's flatness × the glow envelope).
struct CloudFx {
    fx: vec4<f32>,
    lit: vec4<f32>,
};
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var<uniform> cfx: CloudFx;

// MONKEY (sky): coverage with detail at a sheet point; `oct` detail octaves.
fn detailed_cover(uv: vec2<f32>, oct: i32) -> f32 {
    let a = textureSampleLevel(cloud_tex, cloud_samp, uv, 0.0).a;
    if (a <= 0.0) {
        return 0.0;
    }
    return sky_fx::cloud_erode(a, sky_fx::cloud_detail(uv, cfx.fx.y, oct));
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(cloud_tex, cloud_samp, in.uv);
    var a = texel.a;
    var rgb = texel.rgb;
    // MONKEY (sky): High only; Classic and Enhanced keep the kernel's texels as they are.
    if (cfx.fx.x >= 1.5) {
        let n = sky_fx::cloud_detail(in.uv, cfx.fx.y, 4);
        a = sky_fx::cloud_erode(a, n);
        // A painterly tone inside the mass from the same detail.
        rgb = rgb * (0.93 + 0.14 * n);
        let strength = cfx.lit.w;
        if (a > 0.003 && strength > 0.0) {
            let sd = cfx.fx.zw;
            let step = 0.011;
            var occl = detailed_cover(in.uv + sd * step, 2);
            occl += detailed_cover(in.uv + sd * step * 2.0, 2) * 0.75;
            occl += detailed_cover(in.uv + sd * step * 3.5, 2) * 0.5;
            let transmit = exp(-occl * 1.44 * strength);
            // WarcraftXL's ambient floor 0.55, softened toward the painted texel.
            let light = 0.62 + 0.38 * transmit;
            let ahead = detailed_cover(in.uv + sd * step * 1.5, 2);
            let rim = max(ahead - a, 0.0) * (1.0 - a) * transmit * strength;
            rgb = rgb * mix(1.0, light, strength) + cfx.lit.rgb * rim * 0.9;
            rgb = min(rgb, vec3<f32>(1.0));
        }
    }
#ifdef VERTEX_COLORS
    a *= in.color.a; // the dome's rim fade (ring alphas)
#endif
    // Premultiplied gamma blend; the RGB is already the reference's byte math.
    return vec4<f32>(rgb * a, a);
}
