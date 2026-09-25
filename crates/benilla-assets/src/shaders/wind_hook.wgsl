// Ported from WarcraftXL (https://github.com/WarcraftXL) by iThorgrim — module
// wxl-experimental-wind, grass/GrassWind.hpp and grass/GrassWind.cpp.
#define_import_path benilla::wind_hook

#import benilla::monkey_frame::MonkeyFrame

const WIND_TAU: f32 = 6.28318530718;
// MONKEY (fix-wind): `wind_b.x` is the integrated wind travel in yards, wrapped at 4096 on the CPU
// (`wind::TRAVEL_WRAP`). Each wave rate is a whole number of cycles per wrap, so the wrap is
// seamless; the phase advances at k * speed, never k * speed * age.
const WIND_WRAP: f32 = 4096.0;

// The phase (radians) of a wave doing `cycles` whole cycles per wrap of travel.
fn wind_travel_phase(travel: f32, cycles: f32) -> f32 {
    return fract(cycles * (travel / WIND_WRAP)) * WIND_TAU;
}

// The fixed profile heading: the waves' spatial term uses it, so the veer (which only turns the
// bend direction) never rotates the phase field about the world origin.
fn wind_base_dir(frame: MonkeyFrame) -> vec2<f32> {
    return vec2<f32>(cos(frame.wind_a.z), sin(frame.wind_a.z));
}

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
    if (weight <= 0.0 || frame.wind_b.z <= 0.0) {
        return vec3<f32>(0.0);
    }

    let dir_primary = normalize(frame.wind_a.xy + vec2<f32>(1.0e-6, 0.0));
    let dir_secondary = wind_cross_dir(dir_primary);
    let base_primary = wind_base_dir(frame);
    let base_secondary = wind_cross_dir(base_primary);
    let wave_primary = WIND_TAU / 18.0;
    let wave_secondary = WIND_TAU / 6.5;
    // WarcraftXL's k * 0.33 * speed (primary) and k * 0.33 * 1.37 * speed (secondary), as whole
    // cycles per wrap: 75 and 285.
    let phase_primary = wave_primary * dot(base_primary, world.xz)
        - wind_travel_phase(frame.wind_b.x, 75.0) + tuft_phase;
    let phase_secondary = wave_secondary * dot(base_secondary, world.xz)
        - wind_travel_phase(frame.wind_b.x, 285.0) + tuft_phase * 1.7;
    let gust = 1.0 + 0.5 * (frame.wind_a.w - 0.5) * 2.0;
    let variance = mix(0.7, 1.3, fract(sin(tuft_phase * 12.9898) * 43758.5453));
    let distance_fade = 1.0 / (1.0 + distance(camera, world) * 0.015);
    var xz = (
        dir_primary * 0.060 * (sin(phase_primary) + 0.35)
        + dir_secondary * 0.020 * sin(phase_secondary)
    ) * gust * variance * weight * distance_fade * frame.wind_b.y * frame.wind_b.z;

    let count = min(u32(frame.misc.x), 8u);
    for (var i = 0u; i < 8u; i += 1u) {
        if (i < count) {
            let bender = frame.benders[i];
            let away = world.xz - bender.xz;
            let r = length(away);
            let radial = away / max(r, 1.0e-3);
            let edge = 1.0 - smoothstep(0.0, max(bender.w, 0.01), r);
            let cone = smoothstep(0.30, 0.80, world.y - bender.y);
            xz += radial * 0.50 * edge * cone * weight * frame.wind_b.z;
        }
    }
    return vec3<f32>(xz.x, 0.0, xz.y);
}

// MONKEY (wind): our tree/bush design (WarcraftXL has no tree wind). Only leaf-card batches call
// this: a low-frequency crown sway, small leaf flutter, stable anchor phase and distance fade.
fn tree_offset(
    world: vec3<f32>,
    anchor: vec3<f32>,
    camera: vec3<f32>,
    frame: MonkeyFrame,
) -> vec3<f32> {
    if (frame.wind_b.w <= 0.0) {
        return vec3<f32>(0.0);
    }
    let h = max(world.y - anchor.y, 0.0);
    // A one-yard bush still needs a crown; tree leaf cards generally begin above this range and
    // therefore reach full weight, while the first 0.15 yd remains a planted base.
    let crown = smoothstep(0.15, 1.5, h);
    let weight = crown * crown;
    let dir = normalize(frame.wind_a.xy + vec2<f32>(1.0e-6, 0.0));
    let cross = vec2<f32>(-dir.y, dir.x);
    let base = wind_base_dir(frame);
    let base_cross = vec2<f32>(-base.y, base.x);
    let phase = fract(sin(dot(anchor.xz, vec2<f32>(0.173, 0.317))) * 43758.5453) * WIND_TAU;
    // 0.08 rad/yd sway, 0.61x side wave and a 2.3 rad/s-at-base-speed flutter, as whole cycles
    // per wrap of travel: 52, 32 and 167.
    let slow = sin(dot(base, world.xz) * (WIND_TAU / 34.0)
        - wind_travel_phase(frame.wind_b.x, 52.0) + phase);
    let side = sin(dot(base_cross, world.xz) * (WIND_TAU / 21.0)
        - wind_travel_phase(frame.wind_b.x, 32.0) + phase * 1.7);
    let flutter = sin(wind_travel_phase(frame.wind_b.x, 167.0)
        + dot(world.xz, vec2<f32>(1.7, 2.1)) + phase * 2.0);
    let gust = 0.75 + frame.wind_a.w * 0.5;
    let fade = 1.0 / (1.0 + distance(camera, world) * 0.006);
    let xz = (
        dir * (0.045 + slow * 0.055)
        + cross * side * 0.025
        + dir * flutter * 0.012
    ) * weight * gust * fade * frame.wind_b.y * frame.wind_b.w;
    return vec3<f32>(xz.x, 0.0, xz.y);
}
