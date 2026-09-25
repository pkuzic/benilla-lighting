// MONKEY (post): screen-space radial blur through sky pixels in the scene depth.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
#ifdef MULTISAMPLED
@group(0) @binding(2) var depth: texture_depth_multisampled_2d;
#else
@group(0) @binding(2) var depth: texture_depth_2d;
#endif

struct Shafts {
    sun: vec4<f32>,
    color: vec4<f32>,
};
@group(0) @binding(3) var<uniform> shafts: Shafts;

fn scene_depth(uv: vec2<f32>) -> f32 {
    let dims = textureDimensions(depth);
    let p = vec2<i32>(clamp(uv, vec2(0.0), vec2(0.999999)) * vec2<f32>(dims));
#ifdef MULTISAMPLED
    var d = 0.0;
    for (var sample = 0u; sample < textureNumSamples(depth); sample++) {
        // Reversed Z: any covered sample is enough to block the sky ray.
        d = max(d, textureLoad(depth, p, i32(sample)));
    }
    return d;
#else
    return textureLoad(depth, p, 0);
#endif
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let base = textureSample(scene, scene_sampler, in.uv);
    let count = u32(shafts.sun.w);
    let step_uv = (shafts.sun.xy - in.uv) * (0.88 / shafts.sun.w);
    var uv = in.uv;
    var decay = 1.0;
    var light = 0.0;
    var norm = 0.0;
    for (var i = 0u; i < 32u; i++) {
        if (i >= count) { break; }
        uv += step_uv;
        decay *= 0.965;
        let sky = 1.0 - smoothstep(0.000002, 0.00008, scene_depth(uv));
        light += sky * decay;
        norm += decay;
    }
    let rays = light / max(norm, 0.0001);
    return vec4(base.rgb + shafts.color.rgb * rays * shafts.sun.z * 0.16, base.a);
}
