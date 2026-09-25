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
    // MONKEY (fix-post): one sample, not a max over all of them: the march is a blur, and the
    // per-sample loop cost 28 x N loads per pixel (224 at 8x).
    return textureLoad(depth, p, 0);
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
        // MONKEY (fix-post): only the cleared / sky-pinned depth (exactly 0 on infinite reverse-Z)
        // is sky. The old ramp read d = near / z, so it moved with the live `nearclip` cvar and
        // let shafts pour through terrain beyond ~125 yd at nearclip 0.01.
        let sky = select(0.0, 1.0, scene_depth(uv) <= 1.0e-7);
        light += sky * decay;
        norm += decay;
    }
    // MONKEY (fix-post): the path mean is only the occlusion ratio; the shaft itself falls off
    // radially from the sun (aspect-corrected), else every clear sky pixel got the same lift.
    let dims = vec2<f32>(textureDimensions(scene));
    let to_sun = (in.uv - shafts.sun.xy) * vec2<f32>(dims.x / dims.y, 1.0);
    let radial = 1.0 - clamp(length(to_sun) / 0.9, 0.0, 1.0);
    let rays = light / max(norm, 0.0001) * radial * radial;
    return vec4(base.rgb + shafts.color.rgb * rays * shafts.sun.z * 0.16, base.a);
}
