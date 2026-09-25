// Colour grading ported from WarcraftXL wxl-retail-grading.
// Copyright (C) 2026 WarcraftXL. GPL-3.0-or-later; see THIRD-PARTY.md.
// MONKEY (post): real 32³ textures make the original 1024x32 strip's two bilinear taps plus
// lerp(fract(b*31)) one hardware-trilinear lookup, for both the day and night cubes.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
@group(0) @binding(2) var day_lut: texture_3d<f32>;
@group(0) @binding(3) var night_lut: texture_3d<f32>;
@group(0) @binding(4) var lut_sampler: sampler;

struct Grade {
    control: vec4<f32>,
};
@group(0) @binding(5) var<uniform> grade: Grade;

fn grade_one(lut: texture_3d<f32>, color: vec3<f32>) -> vec3<f32> {
    // Half a texel in from each face, with 31 intervals across the 32-texel cube.
    let uvw = color * (31.0 / 32.0) + vec3(0.5 / 32.0);
    return textureSampleLevel(lut, lut_sampler, uvw, 0.0).rgb;
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let source = textureSampleLevel(scene, scene_sampler, in.uv, 0.0);
    let base = clamp(source.rgb, vec3(0.0), vec3(1.0));
    let day = grade_one(day_lut, base);
    let night = grade_one(night_lut, base);
    let authored = mix(day, night, grade.control.y);
    let graded = mix(base, authored, grade.control.x);
    // Bloom already consumed the HDR signal; preserve its remaining excess for the FFX clamp.
    return vec4(graded + max(source.rgb - vec3(1.0), vec3(0.0)), source.a);
}
