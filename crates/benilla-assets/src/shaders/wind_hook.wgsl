// Ported from WarcraftXL (https://github.com/WarcraftXL) by iThorgrim — module
// wxl-experimental-wind, grass/GrassWind.hpp and grass/GrassWind.cpp.
#define_import_path benilla::wind_hook

#import benilla::monkey_frame::MonkeyFrame

const WIND_TAU: f32 = 6.28318530718;

fn wind_cross_dir(dir: vec2<f32>) -> vec2<f32> {
    // WarcraftXL's secondary wave is 35 degrees off the primary.
    return vec2<f32>(
        dir.x * 0.819152 - dir.y * 0.573576,
        dir.x * 0.573576 + dir.y * 0.819152,
    );
}

// MONKEY (wind): WarcraftXL's two travelling waves, gust response, downwind lean, per-tuft
// phase/variance and distance fade, plus the same radial parting extended from the player to the
// eight MonkeyFrame benders. `bend_height` is authored from local mesh height, never texture V.
fn grass_offset(
    world: vec3<f32>,
    bend_height: f32,
    tuft_phase: f32,
    camera: vec3<f32>,
    frame: MonkeyFrame,
) -> vec3<f32> {
    let anchored = clamp((bend_height - 0.30) / 0.70, 0.0, 1.0);
    let weight = anchored * anchored;
    if (weight <= 0.0 || frame.wind1.z <= 0.0) {
        return vec3<f32>(0.0);
    }

    let dir1 = normalize(frame.wind0.xy + vec2<f32>(1.0e-6, 0.0));
    let dir2 = wind_cross_dir(dir1);
    let k1 = WIND_TAU / 18.0;
    let k2 = WIND_TAU / 6.5;
    let travel = frame.wind0.z * 0.33;
    let phase1 = k1 * dot(dir1, world.xz) - k1 * travel * frame.wind1.x + tuft_phase;
    let phase2 = k2 * dot(dir2, world.xz) - k2 * travel * 1.37 * frame.wind1.x + tuft_phase * 1.7;
    let gust = 1.0 + 0.5 * (frame.wind0.w - 0.5) * 2.0;
    let variance = mix(0.7, 1.3, fract(sin(tuft_phase * 12.9898) * 43758.5453));
    let distance_fade = 1.0 / (1.0 + distance(camera, world) * 0.015);
    var xz = (
        dir1 * 0.060 * (sin(phase1) + 0.35)
        + dir2 * 0.020 * sin(phase2)
    ) * gust * variance * weight * distance_fade * frame.wind1.y * frame.wind1.z;

    let count = min(u32(frame.misc.x), 8u);
    for (var i = 0u; i < 8u; i += 1u) {
        if (i < count) {
            let bender = frame.benders[i];
            let away = world.xz - bender.xz;
            let r = length(away);
            let radial = away / max(r, 1.0e-3);
            let edge = 1.0 - smoothstep(0.0, max(bender.w, 0.01), r);
            let cone = smoothstep(0.30, 0.80, world.y - bender.y);
            xz += radial * 0.50 * edge * cone * weight * frame.wind1.z;
        }
    }
    return vec3<f32>(xz.x, 0.0, xz.y);
}
