// Liquid shader — a port of the reference's **ADT** water path (`ocean0_s.bls`) to WGSL.
//
// ## SCOPE — this file implements ONE of the reference's three liquid renderers
//
// This is the ADT MCLQ river/ocean path and nothing else. It is also, today, what benilla runs on
// WMO-embedded liquid, and that is **wrong** — see `liquid/surface.rs`'s `spawn_wmo_liquids` and the
// `wmo_liquid_arms` census. The reference has three:
//
//   * **ADT MCLQ river/ocean** — passes `0x6851b0`/`0x685010`. Two texture stages: a static depth-ramp
//     on stage 0, the animated sheet on stage 1, combined by the pixel program `ocean0_s.bls`. The
//     only liquid path with a depth ramp, and the only one whose vertex carries NO colour. **This
//     file.**
//   * **WMO MLIQ water** — `0x6b62e0` category 0, split again on the group's `MOGP.flags & 0x48`:
//     exterior `0x6b6630` (binds `MapObjExtWater0.bls`), interior `0x6b6420` (no shader, lighting
//     forced off). One texture stage, no depth ramp at all — zero references to the ADT ramp globals
//     `0xc7fbc0`/`0xc81768`/`0xc7fcd8` anywhere in `[0x6b0000, 0x6c4000)` — and alpha comes from a
//     per-vertex authored byte, not from a depth swatch. 164 groups; 90% of them interior.
//   * **magma/slime** — `0x6b68f0` (WMO) / `0x68dca0` (ADT), arm-blind, the sheet IS the body.
//
// ## The ADT combine (VERIFIED — the program is an asset, extracted and read)
//
// `Shaders\Pixel\ocean0_s.bls` out of `patch.MPQ`, verbatim ARB:
//
//   PARAM c[1] = { { 0.25 } };
//   TEX R0, fragment.texcoord[0], texture[0], 2D;   # colorTex  = the depth ramp
//   TEX R1, fragment.texcoord[1], texture[1], 2D;   # detailTex = the animated sheet
//   MAD R1.xyz, fragment.color.primary, R0, R1;
//   ADD R0.xyz, fragment.color.secondary, c[0].x;
//   MAD result.color.xyz, R0, R1.w, R1;
//   MOV result.color.w, R0;                          # R0.w still holds colorTex.a
//
//   ⇒ rgb = primary*colorTex.rgb + detailTex.rgb + (secondary + 0.25)*detailTex.a
//     alpha = colorTex.a
//
// The `+0.25` is the program's own scalar `PARAM`, not an FFP env colour (`glTexEnvfv` has zero call
// sites image-wide) and not a material. **The formula this file has always carried is right, verbatim
// — but its provenance was fiction**: the header used to cite "apitrace WoW.17 program 159" and a
// `docs/knowledge/terrain.md`, neither of which exists, and wow-re had the program attributed to a
// *character* draw (`Model2.bls` ships 32 ARBfp permutations, none containing `0.25` or
// `fragment.color.secondary`). Corrected and recorded: wow-re `terrain/scratch/water-shading-law.md`.
//
//   * colorTex (unit 0) is the depth swatch — a 2-endpoint linear lerp of the zone's dedicated
//     `Light.dbc` water rows, RAW (no ×0.711): `water_shallow.rgb` = IntBand row 16 (river/lake) / 14
//     (ocean), `water_deep.rgb` = row 17 / 15. Rebuilt **per world frame**, not baked once: the dirty
//     flags `[0xc8117c]`/`[0xc81b70]` clear at `0x680b90`/`0x680b97` and refill via `0x58acd0`, so the
//     colour and the opacity track the zone and the clock. (The earlier "reflected sky × 0.711 via
//     `FUN_0068c250`" model fingered the WRONG builder — a separate grey edge texture never bound on
//     the water unit. Rows 14–17 were right all along.)
//   * detailTex (unit 1) is the animated `lake_a`/`ocean_h` frame: RGB near-black, ALPHA = the ripple.
//     MEASURED off the shipped BLPs (DXT3, 9 authored mips): lake_a RGB mean 0.0140 and achromatic to
//     ±1 LSB, alpha mean 0.21 / p50 51 / p99 255. So it adds a faint flat lift + an achromatic shimmer
//     on the crests — NOT the body. The authored mip chain deliberately kills the shimmer with
//     distance (per-mip alpha max 255, 255, 255, 136, 68, then flat 51), which is why the sampler's
//     mips and 16× aniso are load-bearing rather than a nicety.
//   * primary = the vertex's lit colour `clamp(ambient + N·L·sun)`. The ADT liquid vertex has no
//     colour element, so `glColor` is the device default `(1,1,1,1)` tracked into material
//     ambient+diffuse by `glColorMaterial(FRONT_AND_BACK, AMBIENT_AND_DIFFUSE)`, and `GL_LIGHTING` is
//     ON at both water draws (lava explicitly turns it off; water does not).
//   * secondary = the specular sheen — see `sun_sheen` for what is verified in it and what is not.
//   * alpha = swatch.a over the SAME V as the colour. LightParams endpoints: river 0.5→1.0, ocean
//     0.75→1.0. Deeper water = more opaque. **Open**: wow-re reads the byte-verified ADT water alpha
//     as the `0xc7fbc0` LUT's `1.6·(i/63)^8` curve rather than the linear `127+2·row` this file
//     applies — a much later-breaking ramp. Not changed here; it is a look change to every ADT water
//     surface in the game and it belongs in the same A/B as the WMO arms.
//
// **Both `ocean0_s.bls` and `MapObjExtWater0.bls` are CVar-gated** — `specular` and `pixelShaders`
// (registered `0x6886a0`/`0x688712`) default to `"0"`, and with them off there is no program, no
// specular, the stage-1 combine is a plain ADD, and blend is never set so water draws OPAQUE. Our
// reference install's `Config.wtf` sets both to `"1"`, so every capture and every director comparison
// is against the shader leg — which is the leg we implement. The two do not compose: an active ARB
// program bypasses the texture environment entirely.
//
// A SINGLE swatch row (V) indexes both the colour and the alpha — they track together. Each kind reads
// its OWN verified LUT, built side by side in `FUN_0068c4c0`: `clamp(byte/42)` for river/lake
// (`c81768`, `FUN_0068d790`, saturates ~5 yd → the channel middle hits the deep teal row) and
// `clamp(byte/255)` for ocean (`c7fcd8`, `FUN_0068d690`, saturates ~148 yd). The divisors differ
// because the authored bytes do — the sea ramps 1.72 byte/yd against a river's 8.96, and 83 % of ocean
// vertices are pinned at 255 outright (decision 2069, `examples/liquid_depth_census`). (Earlier cuts:
// ripple-as-colour → black; ×8 over-saturated; FLAT colour killed the gradient; sky×0.711 was the wrong
// builder; `byte/255` on the RIVER was the wrong LUT → river middle never went teal. Corrected to
// rows 14–17 raw lerp + the /42 V on rivers, 2026-05-31.)
//
// Two-sided comes from the material (cull off) and is right for EVERY kind: all four reference liquid
// passes force GL_CULL_FACE off at pass entry against a cull-ON device baseline, and `glFrontFace` is
// not even imported (VERIFIED wow-re `liquid-render-state-sided` §6). Blending is per KIND, decided in
// liquid.rs: water/ocean blend with depth-write off; magma/slime are opaque and depth-write. Fog + gamma
// mirror terrain.wgsl (planar eye-Z GL_LINEAR fog in gamma space; raw gamma out — GAMMA LANE, 0161), fog
// applies to every kind — magma and slime included — and WHICH fog block a surface takes is per-surface:
// a WMO interior group's own pool fogs with the interior block, everything else with the scene block
// (see `apply_fog`). Light, fog and both water swatches all come off the ONE shared global-light buffer.

#import bevy_pbr::{
    mesh_functions,
    forward_io::Vertex,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::{view, globals},
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var frames: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var frames_samp: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(103) var scene_depth: texture_2d<f32>;

struct LiquidParams {
    // x = fullbright (magma/slime); y = ocean swatch; z = interior fog; w = sun-sheen shininess.
    kind: vec4<f32>,
    // x = WHICH RENDERER (see `LiquidPath`): 0 = ADT MCLQ, 1 = WMO exterior, 2 = WMO interior.
    // y = quality: 0 Classic, 1 Enhanced, 2 High (stage 1 shares Enhanced); z = wave energy (0..1); w reserved.
    path: vec4<f32>,
    // x = fixed Enhanced capture time; y = frame count; z = the SCROLL FLAG (1 only on the nibble-6/7
    // WMO magma/slime lane — the reference's animated stage-0 texture matrix; see
    // `liquid/surface.rs`'s `scrolls`, and `apply_scroll` below); w = the clock enable (0 on a
    // deterministic run — the whole animation freezes at frame 0 / scroll 0, the 0600 capture
    // pin, baked at material build).
    anim: vec4<f32>,
    sky_zenith: vec4<f32>, // linear RGB, LightIntBand 2
    sky_horizon: vec4<f32>, // linear RGB, LightIntBand 6
    celestial: vec4<f32>, // xyz toward visible body; w = 0 sun, 1 white moon
};
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var<uniform> w: LiquidParams;

// The shared global light (`lighting::global_light`) — the SAME buffer terrain and the models read,
// mirrored here as its canonical row prefix. Liquid used to carry its own copy of every one of these
// values, re-pushed per material by `apply_wow_lighting`; that copy is what left it with only the
// scene fog and no way to see the interior block (decision 0691).
struct WowLight {
    light_ambient: vec4<f32>,      // 0  rgb = ambient; w = Mod2x scale
    light_diffuse: vec4<f32>,      // 1  rgb = sun diffuse; w = clamp flag
    light_sun: vec4<f32>,          // 2  xyz = sun TRAVEL dir (to-light = −xyz)
    light_spec: vec4<f32>,         // 3  rgb = row-9 specular colour; w = TERRAIN shininess (liquid uses w.kind.w)
    fog_color: vec4<f32>,          // 4  rgb = scene fog (block 1, gamma 0..1); w = enable (>0.5)
    fog_params: vec4<f32>,         // 5  x = start yd; y = end yd; w = the farclip wall
    _sh: array<vec4<f32>, 6>,      // 6-11  model SH coeffs — unread here
    _sh_c16: vec4<f32>,            // 12
    water_river: array<vec4<f32>, 2>, // 13-14 shallow/deep river-lake swatch (IntBand 16/17); w = alpha
    water_ocean: array<vec4<f32>, 2>, // 15-16 shallow/deep ocean swatch    (IntBand 14/15); w = alpha
    _grade: vec4<f32>,             // 17
    wmo_fog_color: vec4<f32>,      // 18 rgb = INTERIOR fog (block 2); w = enable
    wmo_fog_params: vec4<f32>,     // 19 x = start yd; y = end yd
    point_count: vec4<f32>,        // 20 live count + point-light controls
    points: array<vec4<f32>, 512>, // 21+ position/range, colour/lane pairs
};
@group(#{MATERIAL_BIND_GROUP}) @binding(90) var<storage, read> wow_light: WowLight;

struct LiquidVsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) depth: f32,
    // The sun-sheen `secondary` evaluated PER-VERTEX (the faithful 1.12 path: the real client computes
    // the Blinn highlight in its FFP vertex shader and interpolates it across the coarse water mesh).
    // The fragment shader uses this interpolated value directly.
    @location(4) secondary_vtx: vec3<f32>,
    // The mesh's vertex COLOUR, which on a WMO INTERIOR pool carries its `MOMT.diffColor` body
    // colour — the reference's own interior water vertex carries a colour dword for exactly this.
    // White on every other lane (and on any mesh with no colour attribute), where nothing reads it.
    @location(5) vcolor: vec4<f32>,
    // The per-frame INTERIOR FOG lane for this surface's own room (decision 1787): the client's
    // `[0xca7f00]`, which gates the WMO liquid pass's block-2 submit (`0x6b6323`–`0x6b6342`)
    // exactly as it gates the geometry pass — so a pool and the walls around it can never
    // disagree about which fog they wear. Carried on `MeshTag` bit 30, read once in the vertex
    // stage and flat-interpolated (the whole surface is one instance).
    @location(6) @interpolate(flat) room_fog: u32,
}

// Sun sheen (`secondary`): a Blinn highlight of the sun on the flat water surface — the glint that's
// strongest at grazing (sunrise/sunset) sun. `secondary = light_spec.rgb · (N·H)^shininess`. Shared by
// both stages so the per-vertex (faithful) and per-pixel (current) paths run IDENTICAL math — only the
// EVALUATION DOMAIN differs (interpolated vs evaluated per fragment).
fn sun_sheen(world_normal: vec3<f32>, world_pos: vec3<f32>) -> vec3<f32> {
    let n = normalize(world_normal);
    let to_light = -normalize(wow_light.light_sun.xyz);
    // LOCAL viewer, not the infinite-viewer constant: the reference sets
    // `GL_LIGHT_MODEL_LOCAL_VIEWER = 1` at `0x59cf89`, so `H = normalize(L + normalize(eye − vertex))`
    // and the eye vector is recomputed per vertex (VERIFIED wow-re `water-shading-law.md`). It is
    // the difference between a highlight and a flood: with the infinite viewer `N·H` is very nearly
    // CONSTANT over a flat water plane, so at high sun the whole sheet saturates at once instead of
    // carrying a glint that moves with the camera.
    let to_view = normalize(view.world_position.xyz - world_pos);
    let half_v = normalize(to_light + to_view);
    let ndoth = max(dot(n, half_v), 0.0);
    // The reference's specular is additionally gated on `N·L > 0` (the fixed-function rule: a light
    // behind the surface contributes no specular). **Deliberately not ported**, because it cannot
    // fire here: `DayNight::SetDirection` holds the LIGHTING sun's azimuth constant and wobbles its
    // elevation only between +20° and +37° — it is always above the horizon, night being the colours
    // going dark rather than the sun setting (`lighting::daynight::sun_direction`, and the separate
    // *visible* sun is the one that rises and sets). Against a liquid surface's flat up normal `N·L`
    // is therefore positive at every minute of the day, so the gate is a branch that can only ever
    // take one arm. Named rather than added: it is real, it is verified, and it is inapplicable.
    //
    // Shininess is WATER's own (`w.kind.w`), not the shared row-3 terrain exponent. Material
    // specular is white (`SetRenderState(3, 0xffffffff)`) and the exponent is 6.0 (`[0x8102e8]`),
    // both VERIFIED — so `light_spec.rgb` is the whole scale, and it is the one input here that is
    // still INFERRED: wow-re pinned the mechanism (`CGLight+0x48` → `collector+0x6c` → `0x589d80(0)`
    // → `glLightfv(GL_SPECULAR)`) but not the number, and reads it as a warm ≈(1.0, 0.91, 0.76)
    // against the row-9 feed we use. Left on row 9 until that lands rather than swapped for an
    // estimate.
    return wow_light.light_spec.rgb * pow(ndoth, max(w.kind.w, 1.0));
}

// The lava/slime **surface scroll**: the reference's animated stage-0 texture matrix, which for
// liquid-type nibbles 6 and 7 is the identity with element 13 — the **v translate** — set to
// `fmod(uptime_s, 10.0) · 0.1` (VERIFIED wow-re `liquid-uv-scroll-law.md`, six-agent §5, matrix
// built at `0x6b68f0` and pushed at stage 0 by `0x6b6ae3`). A texture matrix times `(s, t, 0, 1)`
// with only element 13 non-identity is exactly `t += phase`, so the whole mechanism is this add —
// no matrix, no second sampler, no cost on the paths that do not scroll (`anim.z` is a hard 0
// there, which the CPU side guarantees rather than the shader branching on it).
//
// A full repeat every 10 s, so `REPEAT` wrapping makes the sawtooth's reset invisible. Only the
// rate and the period are reproducible — the reference's phase comes off `GetTickCount`, i.e. the
// machine's uptime, so its absolute value is not a thing to match.
// The liquid clock: `globals.time` (the same wall-elapsed seconds the CPU cycler used to read)
// under the build-time enable. Both animations below are pure functions of it — the CPU-side
// 24 Hz `Assets::get_mut` tick this replaces mutated ~14 materials a tick and its Modified
// fallout (uniform re-uploads, bind-group rebuilds, whole-population `AssetChanged` arming)
// measured 0.28 cpu_ms/frame at the SW pin (2026-08-18 bracket).
fn anim_time() -> f32 {
    if w.kind.x < 0.5 && w.path.y > 0.5 && w.anim.w == 0.0 { return w.anim.x; }
    return w.anim.w * globals.time;
}

// The 24 fps frame flip — 30 frames over 1.25 s (VERIFIED `FUN_0068aac0`), floor-quantized to
// the tick exactly as the reference's integer frame index is.
fn frame_layer() -> i32 {
    return i32(floor(anim_time() * 24.0) % max(w.anim.y, 1.0));
}

fn apply_scroll(uv: vec2<f32>) -> vec2<f32> {
    // v += (t mod 10) · 0.1 — repeats/s `[0x801620]` = 0.1, period `[0x80e5a0]` = 10.0 (VERIFIED
    // wow-re `liquid-uv-scroll-law.md` §5): a sawtooth over exactly one repeat, invisible under
    // REPEAT wrapping. CONTINUOUS now, where the CPU tick quantized it to 1/24 s — the reference
    // itself rebuilds the matrix per draw off a millisecond clock, so this is the more faithful
    // reading, not a new liberty. anim.z is the flag: hard 0 on every non-scrolling lane.
    return vec2<f32>(uv.x, uv.y + w.anim.z * fract(anim_time() / 10.0));
}

// Distance fog — planar eye-Z, GL_LINEAR, gamma space (mirrors terrain.wgsl). Applied to EVERY liquid
// kind, because the reference never disables fog for a liquid batch: the device default for GL_FOG is
// **ON** (`0x593bf0` writes state id `0x0f` = 1) and all 42 fog-enable setters in the binary are
// Push/Pop-scoped, so what a batch inherits at its draw is that default. The ADT lava pass sets only
// cull/lighting/blend (`0x6855ca`/`0x6855d6`/`0x6855e2`) and the WMO magma/slime arm sets only lighting
// (`0x6b6afe`) — neither touches fog — while the WMO *river* arm goes out of its way to re-assert
// `(0x0f, 1)`, which only makes sense in a world where fog-on is liquid's intended state. (VERIFIED
// wow-re `liquid-render-state-sided` §1–§3, §5.)
//
// WHICH fog is a per-surface choice, and it is the reference's own (VERIFIED wow-re `fog-env-state`
// §5, the complete 6-site submit census). The device holds two fog blocks: **block 1** (`+0x70/74/78`)
// is the scene fog, submitted once a frame from `WorldFrame::Render` (`0x66ff20`), and **block 2**
// (`+0x80/84/88`) is block 1 smoothed toward the MFOG/zone target over ~4 s (`0x6cf054`+) — the
// interior haze. Only two call sites in the whole binary re-submit block 2, and they are the WMO
// *geometry* pass (`0x6b51d9`/`0x6b51ea`) and the WMO *liquid* pass (`0x6b6323`–`0x6b6342`), both under
// the same `[0xca7f00]` gate. So an interior room's pool takes the room's fog, in lockstep with the
// walls around it; ADT liquid submits nothing and draws under the scene block. `w.kind.z` is the STATIC
// half of that gate, resolved at spawn from the group's `MOGI & 0x48`; the per-frame half is the
// room's own `[0xca7f00]` bit on `MeshTag` bit 30 (decision 1787 — the flood decides it, and the
// WMO geometry lane reads the same answer, which is what keeps the two in step).
fn apply_fog(rgb: vec3<f32>, world_pos: vec3<f32>, room_fog: u32) -> vec3<f32> {
    var fog_color = wow_light.fog_color;
    var fog_span = wow_light.fog_params.xy;
    if (w.kind.z > 0.5 && room_fog != 0u) {
        fog_color = wow_light.wmo_fog_color;
        fog_span = wow_light.wmo_fog_params.xy;
    }
    if (fog_color.w <= 0.5) {
        return rgb;
    }
    let eye_z = -(view.view_from_world * vec4<f32>(world_pos, 1.0)).z;
    let denom = max(fog_span.y - fog_span.x, 0.001);
    let factor = clamp((fog_span.y - eye_z) / denom, 0.0, 1.0);
    return mix(fog_color.xyz, rgb, factor);
}

@vertex
fn vertex(in: Vertex) -> LiquidVsOut {
    var out: LiquidVsOut;
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);
    out.world_position =
        mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(in.position, 1.0));
    // Only mesh-resolvable long swell moves ocean vertices. Authored shallow depth pins beaches.
    if w.kind.x < 0.5 && w.kind.y > 0.5 && w.path.x < 0.5 && w.path.y > 0.5 {
        // The swell band is the owner-approved one:
        // the explicit zero keeps this call byte-identical to what it has always been.
        out.world_position.y += water_waves(out.world_position.xz, anim_time(), 0.0, 0.0,
            swell_shore_fade(in.uv_b.x), true).x;
    }
    out.clip_position = position_world_to_clip(out.world_position.xyz);
    out.world_normal = mesh_functions::mesh_normal_local_to_world(in.normal, in.instance_index);
    out.uv = in.uv;
#ifdef VERTEX_COLORS
    out.vcolor = in.color;
#else
    out.vcolor = vec4<f32>(1.0);
#endif
    // Per-vertex MCLQ depth (0..1) packed into UV1.x; drives the opacity ramp.
    out.depth = in.uv_b.x;
    // The faithful per-vertex sun sheen — interpolated across the coarse mesh by the fragment stage.
    out.secondary_vtx = sun_sheen(out.world_normal, out.world_position.xyz);
    // `MeshTag` bit 30 — see `LiquidVsOut::room_fog`. ADT surfaces carry no tag (0 ⇒ scene fog,
    // which is their only lane anyway).
    out.room_fog = mesh_functions::get_tag(in.instance_index) & 0x40000000u;
    return out;
}

// ── The ADT depth swatch, as the reference actually BUILDS and SAMPLES it ────────────────────
//
// `FUN_0068a830` fills an 8×64 texture (U inert — each row is `rep stosd`-replicated across all 8
// columns, matching the vertex fill's pinned `u = 0.5`). Its rows are an exact 32-bit integer
// accumulator in **byte space**, not a float lerp:
//
//     step   = ((c1 - c0) << 8) >> 6     ; == 4*(c1 - c0) EXACTLY (six zero low bits, so the sar
//     acc    = c0 << 8 ; acc += step     ;  cannot truncate -> no accumulator drift over 64 rows)
//     row(i) = (acc >> 8) & 0xff         ; == c0 + floor(i*(c1 - c0) / 64),  i = 0..63
//
// Two things that costs us, both missed until wow-re's ocean §5 round (decision 2074):
//
//   * **the ramp never reaches the deep endpoint.** Row 63 is `c0 + floor(63*d/64)`, ≈98.4 % of the
//     way, not `c1`. Our old `mix(shallow, deep, V)` ran the last 1/64 of the ramp that does not
//     exist.
//   * **the ocean's last row alone is darkened.** The tail `[0x68a9c0, 0x68aa36)` is gated on *last
//     row* AND *selector == 0* (ocean; river is selector 1 and gets neither): RGB→HSV,
//     `0x68aa13 fmul [0x8102ec]` — **V *= 0.9** (`0x3f666666`, the f32 nearest 0.9) — HSV→RGB, then
//     `0x7bbec0`/`0x7bbec8` forcing that row's alpha to 255. `0x7bbd60`'s HSV→RGB writes every
//     channel as a product with V and the tail touches neither H nor S (and `S == 0` returns
//     `(V,V,V)` without reading H, so achromatic input takes no hue shift), so the colour half is
//     exactly `floor(0.9 * byte)` per channel — transcribed at f32 and run over all 2^24 byte
//     triples: 99.03 % bit-exact, **max deviation 1/255**.
//
// It is not a corner case: ~80 % of the ocean vertices in the shipped world carry depth byte 255,
// so `V = 1.0` and this row IS the open sea (decision 2069's census).
//
// Sampling is **LINEAR/LINEAR, no mip, D3DTADDRESS_CLAMP** — flags word `0x201`, `0x5a2a18 and 7`
// -> `0x85c7d8` row 1 `{MAG, MIN, MIP} = {2, 2, 0}`, `0x5a2a62 shr 3` -> `0x80a254[0] = 3`;
// corroborated by a GL capture of this exact texture (8x64, levels 1, CLAMP_TO_EDGE, LINEAR/LINEAR).
// So V maps to the texel coordinate `V*64 - 0.5` and blends across neighbouring rows: the darkening
// **ramps in over the final 1/64 of V**, it does not step. That band is the only place any of this
// is visible on a shore.
//
// The WMO arms do NOT come through here — their opacity is a different, 256-entry ramp
// (`0xca7f10`), whose 1/256 granularity the plain lerp above reproduces.
fn swatch_row(shallow: vec4<f32>, deep: vec4<f32>, i: f32, ocean: bool) -> vec4<f32> {
    // RGB endpoints arrive already BYTES (`0x68a8fb`/`0x68a902` read two packed dwords straight out
    // of DayNight state — there is no quantization step for RGB at all); the ALPHA endpoints are
    // `LightParams` floats the reference quantizes `floor(v*255)` first.
    let c0 = vec4<f32>(round(shallow.rgb * 255.0), floor(shallow.w * 255.0));
    let c1 = vec4<f32>(round(deep.rgb * 255.0), floor(deep.w * 255.0));
    let row = c0 + floor(i * (c1 - c0) / 64.0);
    if ocean && i >= 63.0 {
        return vec4<f32>(floor(row.rgb * 0.9), 255.0) / 255.0;
    }
    return row / 255.0;
}

/// The swatch sampled at depth coord `v`, LINEAR across the two rows it falls between.
fn swatch_at(shallow: vec4<f32>, deep: vec4<f32>, v: f32, ocean: bool) -> vec4<f32> {
    let t = clamp(v * 64.0 - 0.5, 0.0, 63.0);
    let i0 = floor(t);
    return mix(
        swatch_row(shallow, deep, i0, ocean),
        swatch_row(shallow, deep, min(i0 + 1.0, 63.0), ocean),
        t - i0,
    );
}

// Enhanced is entirely analytic: height and its exact x/z derivatives, in yards.
// Direction is radians from world +X toward +Z; phase speed follows deep-water dispersion.
// Two mesh-resolvable long waves sum to at most 0.34 yd before energy/shore attenuation.
const WATER_WAVES: array<vec4<f32>, 8> = array<vec4<f32>, 8>(
    // direction, wavelength, amplitude, phase offset
    vec4<f32>(0.35, 18.0, 0.200, 0.0),
    vec4<f32>(0.80, 12.8, 0.140, 1.7),
    vec4<f32>(-0.18, 9.5, 0.090, 3.1),
    vec4<f32>(0.52, 4.8, 0.055, 0.8),
    vec4<f32>(1.10, 3.3, 0.033, 2.4),
    vec4<f32>(-0.45, 2.6, 0.018, 4.6),
    vec4<f32>(0.15, 1.3, 0.006, 1.2),
    vec4<f32>(0.95, 0.85, 0.003, 3.8),
);

// ── SHORE BREAK (item 1) + FOAM LEVELS: the tuned constants, all in one place ──
//
// The shore break is a SEPARATE band from `WATER_WAVES` and touches neither the long swell that
// moves ocean vertices nor the inland colour profile. It exists only in the shoaling zone —
// roughly 0.35 to 3.5 yd of water over the bed — and it is normals + foam, never displacement.
//
// **Its phase coordinate is the vertical depth itself**, not a world ruler. That is the whole
// trick: an iso-depth line IS the shoreline's own contour, so crests run exactly parallel to the
// beach on a curved coast, a headland and a cove alike, with no per-chunk seam and no phase jump
// when the depth gradient rotates. The world direction only ever enters through the NORMAL, where
// it arrives as the analytic derivative `d(phase)/d(depth) · grad(depth)` — so the picture and the
// lighting agree by construction. Where the bed is flat enough that the gradient is unusable the
// direction falls back to the wind (`SHORE_WIND_DIR`, the primary swell's bearing), which is also
// the only place the fallback can show, because a flat bed has no shoaling band to draw.
//
// `shore_g(d) = (d + A·(1 − e^(−d/B))) / L` is a smooth, strictly increasing crest count. Its
// derivative `(1 + (A/B)·e^(−d/B)) / L` is the local wavenumber in DEPTH space, so with A/B = 1.3
// the crest spacing runs 0.5 yd of depth at the waterline out to ~1.07 yd offshore: the wave
// shortens by better than 2× as it shoals, which is the shoaling law's visible half. Because the
// coordinate is depth, that spacing turns into a WORLD wavelength through the bed slope — so a
// gentle beach gets long rollers and a steep one short ones, and the number of visible lines
// approaching the sand stays about four either way.
const SHORE_LAMBDA_D: f32 = 1.15;      // yards of depth per crest, offshore
const SHORE_COMPRESS: f32 = 1.56;      // A — shoaling compression (A/B = 1.3 ⇒ 2.3× at the edge)
const SHORE_COMPRESS_D: f32 = 1.2;     // B — yards of depth the compression decays over
const SHORE_PERIOD: f32 = 3.6;         // seconds between arrivals; the swash runs at 2×  this
const SHORE_TILT: f32 = 0.09;          // peak crest steepness (tan of the tilt), slope-independent
const SHORE_WARP_A: f32 = 0.22;        // yards of depth — coarse crest wander (never ruler-straight)
const SHORE_WARP_B: f32 = 0.10;        // yards of depth — finer segmentation of the same crests
const SHORE_WIND_DIR: f32 = 0.35;      // = WATER_WAVES[0].x, the primary swell bearing

// Foam alphas. The owner rejected BOTH a thick icing sheet and straight stripes before this, so
// every one of these is gated behind a noise breakup and a depth window; the numbers are the
// ceiling a fully-lit, fully-broken crest can reach, not what a typical pixel gets.
const FOAM_WET_EDGE: f32 = 0.35;       // the faint wet line where water meets anything solid
const FOAM_SWASH: f32 = 0.62;          // the sheet running up the sand and fading
const FOAM_CREST: f32 = 0.88;          // the white front of the last wave or two, and its lace
const FOAM_MAX: f32 = 0.90;            // hard ceiling on the sum

fn swell_shore_fade(depth: f32) -> f32 {
    // Ocean V = byte/255, about 148 yd at 1.0: fade in over ~0.15..3.7 yd.
    return smoothstep(0.001, 0.025, depth);
}

fn water_waves(p: vec2<f32>, t: f32, distance: f32, footprint: f32,
    shore: f32, long_only: bool) -> vec3<f32> {
    let energy = clamp(w.path.z, 0.0, 1.0);
    let tempo = mix(0.4, 1.0, sqrt(energy));
    let inland = w.kind.y < 0.5 || w.path.x > 0.5;
    // Inland ADT water drifts gently (a constant vector: a rigid translation, never a shear).
    // WMO pools have no drift at all.
    var ripple_p = p;
    if inland && w.path.x < 0.5 {
        ripple_p -= t * vec2<f32>(0.06, 0.025);
    }
    var result = vec3<f32>(0.0);
    for (var i = 0u; i < 8u; i += 1u) {
        if long_only && i >= 2u { break; }
        if inland && i < 3u { continue; }
        let wave = WATER_WAVES[i];
        let direction = vec2<f32>(cos(wave.x), sin(wave.x));
        let k = 6.2831853 / wave.y;
        let speed = sqrt(10.72 / k); // gravity in yd/s^2; phase speed in yd/s
        let phase = k * (dot(direction, ripple_p) - speed * tempo * t) + wave.w;
        // Suppress unresolved waves before Nyquist, and remove fine ripples beyond 35 yd.
        var fade = 1.0 - smoothstep(0.10, 0.45, footprint / wave.y);
        if i >= 6u { fade *= 1.0 - smoothstep(10.0, 35.0, distance); }
        if i < 2u { fade *= shore; }
        let amplitude = wave.z * energy * fade;
        result += vec3<f32>(amplitude * sin(phase), amplitude * k * cos(phase) * direction);
    }
    return result;
}

fn foam_hash(p_in: vec2<f32>) -> f32 {
    // MONKEY (foam noise): world coordinates here are ~1e4 yd and are scaled further before they
    // arrive, so `p * 0.1031` sat near 2e3 where an f32 keeps only ~13 fraction bits - the hash
    // degenerated into a visible regular tile grid inside the foam (capture it11-water-beach-top).
    // Wrap the LATTICE CELL to a 512 period first: exact for integers, invisible at 512 cells, and it
    // gives the hash its full precision back.
    let p = p_in - 512.0 * floor(p_in / 512.0);
    let q = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    let r = q + dot(q, q.yzx + 33.33);
    return fract((r.x + r.y) * r.z);
}

fn foam_value_noise(p: vec2<f32>) -> f32 {
    let cell = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    return mix(mix(foam_hash(cell), foam_hash(cell + vec2<f32>(1.0, 0.0)), u.x),
        mix(foam_hash(cell + vec2<f32>(0.0, 1.0)),
            foam_hash(cell + vec2<f32>(1.0, 1.0)), u.x), u.y);
}

// MONKEY (foam noise): thresholded VALUE noise shows its lattice - the foam and crest masks came out
// as boxy slabs with stair-stepped edges aligned to the world axes (capture it9/it10-water-beach-top).
// Two samples on differently ROTATED and scaled lattices, averaged, have no shared axis, so the
// thresholded shapes turn into irregular blobs. Same range (0..1), same call sites, twice the taps.
fn foam_noise(p: vec2<f32>) -> f32 {
    let a = vec2<f32>(0.7986 * p.x - 0.6018 * p.y, 0.6018 * p.x + 0.7986 * p.y);
    let q = p * 1.37 + vec2<f32>(17.3, -9.1);
    let b = vec2<f32>(0.3584 * q.x + 0.9336 * q.y, -0.9336 * q.x + 0.3584 * q.y);
    // Averaging narrows the distribution; re-expand it so the existing thresholds keep their cut.
    return clamp((0.5 * (foam_value_noise(a) + foam_value_noise(b)) - 0.5) * 1.4 + 0.5, 0.0, 1.0);
}
// Reverse-Z perspective: valid for both finite and infinite far planes. For z_view = -distance,
// depth = -P22 + P32 / distance. Do not use a forward-Z near/far approximation.
fn water_view_distance(depth: f32) -> f32 {
    return view.clip_from_view[3][2] / max(depth + view.clip_from_view[2][2], 1e-7);
}

// == ENHANCED WATER (the optional water module; see WATER.md at the repo root) ==================
// Design reference and credit: the WarcraftXL project's `wxl-experimental-water` module for the
// 1.12 client, author iThorgrim - https://github.com/WarcraftXL. Its author permits reuse of that
// code here provided the author and the original project are named; this notice is that
// attribution and must stay with the module. A block that ports WarcraftXL code says so where it
// stands and is listed in WATER.md.
fn enhanced_water(in: LiquidVsOut, shallow: vec4<f32>, deep: vec4<f32>) -> vec4<f32> {
    let pixel = clamp(vec2<i32>(in.clip_position.xy), vec2<i32>(0),
        vec2<i32>(textureDimensions(scene_depth)) - vec2<i32>(1));
    let own_depth = textureLoad(scene_depth, pixel, 0).r;
    // MONKEY (bed clutter): grass blades and reeds standing in a stream are in the opaque depth, so
    // every blade read as "something solid at the surface" - each one grew a white contact-foam
    // outline and a paler, less-absorbed tint than the bed around it (owner screenshot, Stonefield
    // Farm). Anything THIN is not a shore: take the FARTHEST of nine taps (reverse-Z: the smallest)
    // over a ~0.7 % of the view height ring, so a blade a few pixels wide resolves to the bed behind
    // it, while a bank, a rock or a hull - wider than the ring - is untouched bar a few pixels.
    let ring = max(i32(view.viewport.w * 0.0065), 2);
    let top = vec2<i32>(textureDimensions(scene_depth)) - vec2<i32>(1);
    var far_depth = own_depth;
    var offs = array<vec2<i32>, 8>(
        vec2<i32>(1, 1), vec2<i32>(1, 0), vec2<i32>(-1, 1), vec2<i32>(0, 1),
        vec2<i32>(-1, -1), vec2<i32>(-1, 0), vec2<i32>(1, -1), vec2<i32>(0, -1));
    for (var k = 0; k < 8; k += 1) {
        let reach = select(ring, 2 * ring, (k & 1) == 1);
        let tap = textureLoad(scene_depth,
            clamp(pixel + offs[k] * reach, vec2<i32>(0), top), 0).r;
        // Reverse-Z: farther is SMALLER; 0 is the sky, which is not a bed.
        if tap < far_depth && tap > 0.0 { far_depth = tap; }
    }
    // The open sea keeps its own pixel: its shore train's phase IS this depth, and sand has no reeds.
    let sea = w.kind.y > 0.5 && w.path.x < 0.5;
    let bed_depth = select(far_depth, own_depth, sea);
    let to_view = normalize(view.world_position.xyz - in.world_position.xyz);
    let eye_pos = (view.view_from_world * vec4<f32>(in.world_position.xyz, 1.0)).xyz;
    // Convert the two eye-Z distances to a distance ALONG the pixel's view ray.
    let ray_cos = max(abs(normalize(eye_pos).z), 0.001);
    let thickness = min(max(water_view_distance(bed_depth)
        - water_view_distance(in.clip_position.z), 0.0) / ray_cos, 1000.0);
    // Reconstruct the opaque scene point, then measure height below the surface.
    // Ray thickness is only the contact measure: opacity must not follow eye-Z.
    let ndc_xy = ((in.clip_position.xy - view.viewport.xy) / view.viewport.zw)
        * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    let scene_h = view.world_from_clip * vec4<f32>(ndc_xy, bed_depth, 1.0);
    var vertical_depth = 1000.0;
    if bed_depth > 0.0 && abs(scene_h.w) > 1e-7 {
        vertical_depth = clamp(in.world_position.y - scene_h.y / scene_h.w, 0.0, 1000.0);
    }
    let t = anim_time(); // WOW_CAPTURE_WATER_T pins the frozen water phase.
    let p = in.world_position.xz;
    let energy = clamp(w.path.z, 0.0, 1.0);
    let ocean_mesh = w.kind.y > 0.5 && w.path.x < 0.5;

    // ── EVERY screen derivative in this shader is taken HERE ────────────────────────────────
    // Top level of the function, before any branch, loop or early return: WGSL's uniformity rule
    // makes a derivative inside non-uniform control flow invalid, and this function is long enough
    // that the only safe discipline is to have exactly one place they can live. Nothing below may
    // reintroduce a `dpdx`/`dpdy` inside a conditional.
    let ddx_p = dpdx(p);
    let ddy_p = dpdy(p);
    let ddx_depth = dpdx(vertical_depth);
    let ddy_depth = dpdy(vertical_depth);
    let footprint = max(length(ddx_p), length(ddy_p));

    // The BED's slope in world yards, recovered by inverting the pixel→world Jacobian: solve
    // `g·(dp/dx) = dd/dx` and `g·(dp/dy) = dd/dy` for the world-space gradient of the vertical
    // depth field. `offshore` is the unit direction of INCREASING depth, so `−offshore` points at
    // the beach and the shore train travels that way.
    let jac_det = ddx_p.x * ddy_p.y - ddx_p.y * ddy_p.x;
    var depth_grad = vec2<f32>(0.0, 0.0);
    if abs(jac_det) > 1e-12 {
        depth_grad = vec2<f32>(
            (ddx_depth * ddy_p.y - ddy_depth * ddx_p.y) / jac_det,
            (ddy_depth * ddx_p.x - ddx_depth * ddy_p.x) / jac_det,
        );
    }
    let bed_slope = length(depth_grad);
    // Flat bed, or the depth field broke across a silhouette (the ratio blows up there): fall back
    // to the wind. A flat bed has no shoaling band to draw, so the fallback is mostly a guard.
    var offshore = vec2<f32>(cos(SHORE_WIND_DIR), sin(SHORE_WIND_DIR));
    if bed_slope > 0.02 && bed_slope < 12.0 {
        offshore = depth_grad / bed_slope;
    }

    // ── The shore break: shoaling trains riding the bathymetry (see the constants block) ────
    // Domain-warped in DEPTH units, so the crests wander and segment along the beach instead of
    // running as ruler-straight bands.
    let shore_warp =
        SHORE_WARP_A * (2.0 * foam_noise(p * 0.075 + t * vec2<f32>(0.010, -0.007)) - 1.0)
        + SHORE_WARP_B * (2.0 * foam_noise(p * 0.21 - t * vec2<f32>(0.006, 0.009)) - 1.0);
    let dq = max(vertical_depth + shore_warp, 0.0);
    let shore_g = (dq + SHORE_COMPRESS * (1.0 - exp(-dq / SHORE_COMPRESS_D))) / SHORE_LAMBDA_D;
    // `+ t/period` with a crest-count that RISES with depth ⇒ a crest of fixed phase slides to
    // shallower water as time runs: the train travels shoreward.
    let shore_phase = 6.2831853 * (shore_g + t / SHORE_PERIOD);
    let shore_phase2 = 6.2831853 * (1.9 * shore_g + t / (SHORE_PERIOD * 0.62)) + 2.1;
    // Steepness rises as it shoals (3.5 → 0.75 yd), collapses into the break below ~0.35 yd, and
    // is gone past the band's offshore edge.
    let shoal = smoothstep(3.5, 0.75, vertical_depth);
    let shore_collapse = smoothstep(0.12, 0.38, vertical_depth);
    let shore_offshore_fade = 1.0 - smoothstep(2.4, 3.6, vertical_depth);
    // MONKEY (surf on wrecks): the shore train reads the SCENE depth, and a sunken boat, a pier
    // foot or a rock shelf is shallow scene depth in the middle of deep water - so the whole surf
    // (crests, lace, swash) was painted over the hull of a wreck off Longshore. A shore is where
    // the SEA BED is shallow. The mesh carries the authored bed depth (`in.depth`, byte/255 of about
    // 148 yd): where the bed lies well below what the pixel sees, the pixel is an OBJECT, and an
    // object gets the thin wet-edge line only. The byte is a FLOOR (1.72 per yd), so on a real beach the
    // authored bed is never deeper than the scene by more than interpolation error: allow 0.5 yd.
    let bed_yd = clamp(in.depth, 0.0, 1.0) * 148.0;
    let on_bed = 1.0 - smoothstep(0.5, 1.0, bed_yd - vertical_depth);
    let shore_gain = select(0.0, shoal * shore_collapse * shore_offshore_fade * on_bed, ocean_mesh)
        * energy;
    // Steepness is set DIRECTLY — the tangent of the crest tilt — rather than through an amplitude
    // and a wavenumber, so a gentle beach and a steep one roll with the same visible strength
    // instead of one washing out and the other exploding.
    let shore_grad = SHORE_TILT * shore_gain
        * (cos(shore_phase) + 0.45 * cos(shore_phase2)) * offshore;

    let shore = select(1.0, swell_shore_fade(in.depth), ocean_mesh);
    let wave = water_waves(p, t, length(eye_pos), footprint, shore, false);
    // One surface gradient: the procedural bands and the shore break.
    let surf_grad = wave.yz + shore_grad;
    var n = normalize(vec3<f32>(-surf_grad.x, 1.0, -surf_grad.y));
    if dot(n, to_view) < 0.0 { n = -n; }

    // Far water settles into a calm sky sheet.
    n = normalize(mix(n, vec3<f32>(0.0, sign(n.y), 0.0),
        0.6 * smoothstep(60.0, 120.0, length(eye_pos))));
    // Clean translucent teal at the edge; zone tint remains in the deeper body.
    let teal_shallow = mix(vec3<f32>(0.12, 0.42, 0.39), shallow.rgb, 0.18);
    // MONKEY (water body): the zone deep row alone renders the open sea and lake middles a muddy
    // grey-brown (Westfall ocean, Loch Modan) - it was authored to sit UNDER the reference ripple
    // sheet, not to be a lit body colour. Pull it 70 % toward the zone zenith sky (deep water takes
    // its colour from the sky it scatters) and keep it blue-dominant, so the body reads as water by
    // day, follows dusk and night through the same sky row, and still differs zone to zone.
    let zenith_gamma = pow(max(w.sky_zenith.rgb, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
    let deep_sky = mix(deep.rgb, zenith_gamma * 1.1, 0.7);
    let deep_body = vec3<f32>(min(deep_sky.r, deep_sky.g * 0.85), deep_sky.g,
        max(deep_sky.b, deep_sky.g * 1.12));
    var body = mix(teal_shallow, deep_body, smoothstep(0.0, 3.5, vertical_depth));
    if !ocean_mesh {
        let clear_row = mix(shallow.rgb, vec3<f32>(0.20, 0.40, 0.36), 0.35);
        let clear_tint = vec3<f32>(min(clear_row.r, clear_row.g * 0.72), clear_row.g,
            clamp(clear_row.b, clear_row.g * 0.82, clear_row.g * 1.15));
        let inland_row = mix(deep.rgb, zenith_gamma * 1.1, 0.25);
        let inland_deep = vec3<f32>(min(inland_row.r, inland_row.g * 0.72), inland_row.g,
            clamp(inland_row.b, inland_row.g * 0.82, inland_row.g * 1.15));
        body = mix(clear_tint, inland_deep, smoothstep(0.0, 5.0, vertical_depth));
    }
    let to_light = -normalize(wow_light.light_sun.xyz);
    // MONKEY (water body): water is lit by the light it SCATTERS, not only by N.L on its skin, so
    // the lit body keeps a high floor (ambient x 1.35, N.L floor 0.5). With the old 0.25 floor a
    // noon lake rendered near-black navy where the owner reference is a bright teal-blue; night
    // still darkens because both rows do.
    let lighting = clamp(wow_light.light_ambient.rgb * 1.35 + wow_light.light_diffuse.rgb
        * max(dot(n, to_light), 0.5), vec3<f32>(0.0), vec3<f32>(1.0));
    var rgb = body * lighting;
    // MONKEY (shore waves): the shore train is shown mainly as a CONTINUOUS crest brightening, not as
    // a normal tilt. Its direction comes from the screen-space gradient of the bed depth, and the
    // terrain is flat triangles, so that direction jumps at every triangle edge: at SHORE_TILT 0.30
    // the highlights broke into blocky rectangular shards with staircase edges (capture
    // it9-water-beach-top). The PHASE is a function of the depth itself and is continuous, so a term
    // driven by the phase alone cannot facet. The tilt stays at 0.09 for a little specular life.
    let shore_crest = pow(max(sin(shore_phase), 0.0), 2.0)
        + 0.45 * pow(max(sin(shore_phase2), 0.0), 2.0);
    rgb += shore_gain * shore_crest * 0.13 * lighting * vec3<f32>(0.72, 0.95, 0.90);
    if !ocean_mesh {
        // Keep the zone's green-blue absorption under warm dusk illumination.
        rgb = body * dot(lighting, vec3<f32>(0.2126, 0.7152, 0.0722));
    }
    let celestial_dir = normalize(w.celestial.xyz);
    let crest = smoothstep(0.35, 0.95, 0.5 + 0.5 * wave.x / max(0.545 * energy, 0.001));
    let transmission = crest * pow(max(dot(to_view, -celestial_dir), 0.0), 3.0) * energy
        * smoothstep(-0.02, 0.12, celestial_dir.y);
    rgb += vec3<f32>(0.06, 0.30, 0.19) * transmission * lighting;
    let reflection_n = normalize(mix(vec3<f32>(0.0, sign(n.y), 0.0), n, 0.45));
    let reflected = reflect(-to_view, reflection_n);
    let sky_linear = mix(w.sky_horizon.rgb, w.sky_zenith.rgb, clamp(reflected.y * 1.4, 0.0, 1.0));
    // Linear reflection interpolation, then conversion to the world's gamma blend/fog lane.
    let sky = select(12.92 * sky_linear,
        1.055 * pow(max(sky_linear, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055,
        sky_linear > vec3<f32>(0.0031308));
    var reflectivity = mix(0.06, 0.88, pow(1.0 - max(dot(n, to_view), 0.0), 4.0));
    if !ocean_mesh { reflectivity = min(reflectivity, 0.55); }
    rgb = mix(rgb, sky, reflectivity);
    // Smooth crest glints; widen and conserve lobe energy as the footprint grows.
    let half_v = normalize(celestial_dir + to_view);
    let ndoth = max(dot(n, half_v), 0.0);
    let normal_variance = dot(dpdx(n), dpdx(n)) + dot(dpdy(n), dpdy(n));
    let spread = 1.0 + 400.0 * normal_variance + 2.0 * smoothstep(30.0, 160.0, length(eye_pos));
    if w.path.x < 1.5 {
        if w.celestial.w > 0.5 {
            // Own peak intensity, never nightGain or the signed moon-shadow weight.
            // sin(8 degrees): the reflection is exactly zero below the horizon.
            rgb += vec3<f32>(0.80, 0.88, 1.0) * 0.45
                * (pow(ndoth, 400.0 / spread) / sqrt(spread)
                    + 0.25 * pow(ndoth, 60.0 / sqrt(spread)) / sqrt(spread))
                * smoothstep(0.0, 0.1391731, celestial_dir.y);
        } else {
            rgb += wow_light.light_diffuse.rgb
                * (pow(ndoth, 400.0 / spread) / sqrt(spread)
                    + 0.25 * pow(ndoth, 60.0 / sqrt(spread)) / sqrt(spread))
                * smoothstep(-0.02, 0.08, celestial_dir.y);
        }
    }

    // Rank candidates by actual fragment distance, not table order. Only four lights are shaded.
    var nearest = array<u32, 4>(256u, 256u, 256u, 256u);
    var distances = array<f32, 4>(900.0, 900.0, 900.0, 900.0);
    for (var i = 0u; i < min(u32(wow_light.point_count.x), 256u); i += 1u) {
        let colour = wow_light.points[2u * i + 1u];
        if colour.w >= 0.5 && w.path.x < 1.5 { continue; }
        let delta = wow_light.points[2u * i].xyz - in.world_position.xyz;
        let d2 = dot(delta, delta);
        if d2 >= distances[3] { continue; }
        var slot = 3u;
        loop {
            if slot == 0u { break; }
            if d2 >= distances[slot - 1u] { break; }
            distances[slot] = distances[slot - 1u];
            nearest[slot] = nearest[slot - 1u];
            slot -= 1u;
        }
        distances[slot] = d2;
        nearest[slot] = i;
    }
    for (var j = 0u; j < 4u; j += 1u) {
        if nearest[j] == 256u { continue; }
        let pos = wow_light.points[nearest[j] * 2u];
        let colour = wow_light.points[nearest[j] * 2u + 1u];
        let delta = pos.xyz - in.world_position.xyz;
        let distance = sqrt(max(distances[j], 0.0001));
        let light_dir = delta / distance;
        let reach = min(select(pos.w, 2.0 * colour.w, colour.w >= 0.5), 30.0);
        let attenuation = pow(1.0 - smoothstep(0.0, max(reach, 0.01), distance), 2.0);
        rgb += colour.rgb * pow(max(dot(n, normalize(light_dir + to_view)), 0.0), 64.0)
            * max(dot(n, light_dir), 0.0) * attenuation * 0.7;
    }

    // Patchy, low-frequency coverage, never a texture-derived white outline.
    let noise = 0.7 * foam_noise(p * 0.6 + t * vec2<f32>(0.025, -0.018))
        + 0.3 * foam_noise(p * 1.7 - t * vec2<f32>(0.014, 0.021));
    let breakup = smoothstep(0.40, 0.72, noise);
    // The cached derivatives from the top of the function — same expression as before, one tap.
    let depth_gradient = length(vec2<f32>(ddx_depth, ddy_depth))
        / max(length(vec2<f32>(length(ddx_p), length(ddy_p))), 0.001);
    let wall_suppression = mix(1.0, 0.3, smoothstep(0.8, 3.0, depth_gradient));
    let contact = smoothstep(0.0, 0.025, thickness)
        * (1.0 - smoothstep(0.055, 0.15, thickness));
    let contact_alpha = FOAM_WET_EDGE * contact * wall_suppression * breakup;
    // MONKEY (beach foam, third pass). The owner called the previous surf ugly: it was the crest
    // line chopped into DASHES by a breakup noise, plus swash "crescents" stamped from an 8 yd patch
    // grid - rows of white blobs. Surf is not dashes. A breaking wave is (1) a thin, nearly
    // continuous bright FRONT, (2) a LACE of foam left behind it that thins out and dissolves, and
    // (3) a sheet that runs up the sand after each arrival and fizzles. All three are driven by the
    // shore train's own phase, so the foam sits on the wave the normals show.
    //
    // `surf_u` is the fraction of a wave spacing BEHIND the front (the phase rises with depth, so
    // behind = seaward): 0 at the front, which sits just ahead of the crest the normals draw.
    let surf_u = fract(shore_g + t / SHORE_PERIOD - 0.20);
    // Web-like lace: ridged noise (bright along the zero set of two noise fields), two octaves.
    let lace_a = 1.0 - abs(2.0 * foam_noise(p * 1.25 + t * vec2<f32>(0.030, -0.020)) - 1.0);
    let lace_b = 1.0 - abs(2.0 * foam_noise(p * 3.30 - t * vec2<f32>(0.020, 0.035)) - 1.0);
    let lace = 0.62 * lace_a + 0.38 * lace_b;
    // The lace dissolves with age: the threshold climbs from "most of it" to "only the ridges".
    let dissolve = mix(0.42, 0.93, smoothstep(0.02, 0.60, surf_u));
    let lace_mask = smoothstep(dissolve, dissolve + 0.16, lace);
    // The front itself: crisp on its shoreward side, solid for a few percent of a spacing.
    let front = smoothstep(0.0, 0.018, surf_u) * (1.0 - smoothstep(0.035, 0.11, surf_u));
    let trail = smoothstep(0.0, 0.03, surf_u) * (1.0 - smoothstep(0.25, 0.70, surf_u));
    // Strength wanders along the beach, but never to nothing - no gaps, no dashes.
    let along = mix(0.55, 1.0, foam_noise(p * 0.055 + t * vec2<f32>(0.008, 0.005)));
    let break_zone = smoothstep(0.10, 0.30, vertical_depth)
        * (1.0 - smoothstep(1.3, 2.6, vertical_depth)) * on_bed;
    let crest_alpha = select(0.0,
        FOAM_CREST * break_zone * along * max(front, 0.85 * trail * lace_mask), ocean_mesh)
        * wall_suppression * energy;

    // The swash: what is left of each wave runs up the last hand of water and fizzles. Its clock
    // is the front's arrival at the break (0.22 yd of water): 0 when it lands, 1 as the next does.
    let land_g = (0.22 + SHORE_COMPRESS * (1.0 - exp(-0.22 / SHORE_COMPRESS_D))) / SHORE_LAMBDA_D;
    let swash_age = fract(land_g + t / SHORE_PERIOD - 0.20 + 0.10 * (along - 0.75));
    let swash_life = smoothstep(0.0, 0.06, swash_age) * (1.0 - smoothstep(0.30, 0.95, swash_age));
    let swash_band = (1.0 - smoothstep(0.10, 0.34, vertical_depth + 0.5 * shore_warp)) * on_bed;
    let swash_thin = mix(0.30, 0.90, smoothstep(0.05, 0.85, swash_age));
    let swash_lace = smoothstep(swash_thin, swash_thin + 0.18, lace);
    // The very lip of the water keeps a thin bright line while the sheet is alive.
    let lip = 1.0 - smoothstep(0.015, 0.07, vertical_depth);
    let arcs_alpha = select(0.0,
        FOAM_SWASH * swash_life * swash_band * max(swash_lace, 0.8 * lip), ocean_mesh)
        * wall_suppression * energy;

    let foam = min(FOAM_MAX, contact_alpha + arcs_alpha + crest_alpha);
    let illumination = wow_light.light_ambient.rgb + wow_light.light_diffuse.rgb;
    let foam_luma = clamp(dot(illumination, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.22, 1.0);
    // Matte foam composites on top of reflection with its own coverage.
    let soft_edge = smoothstep(0.0, 0.18, vertical_depth);
    var body_alpha = soft_edge * mix(0.55, 0.97, smoothstep(0.0, 1.6, vertical_depth));
    if !ocean_mesh {
        body_alpha = soft_edge * mix(0.30, 0.86, smoothstep(0.0, 4.5, vertical_depth));
    }
    let foam_alpha = smoothstep(0.0, 0.045, vertical_depth) * foam;
    let alpha = foam_alpha + body_alpha * (1.0 - foam_alpha);
    // Foam is white UNDER the light it stands in: half the way to the light's own colour, so a
    // dusk surf is warm and a moonlit one blue-grey, not a neutral paste.
    let foam_tint = mix(vec3<f32>(foam_luma),
        clamp(illumination, vec3<f32>(0.22), vec3<f32>(1.0)), 0.5);
    rgb = (foam_tint * foam_alpha + rgb * body_alpha * (1.0 - foam_alpha))
        / max(alpha, 0.0001);
    return vec4<f32>(apply_fog(rgb, in.world_position.xyz, in.room_fog), alpha);
}

@fragment
fn fragment(in: LiquidVsOut) -> @location(0) vec4<f32> {
    // HARD FAR-CLIP WALL (same as terrain/models, see terrain.wgsl): discard water beyond the
    // projection far plane so lakes/rivers don't render past the wall. `fog_params.w` = farclip
    // (0 ⇒ disabled).
    if (wow_light.fog_params.w > 0.0) {
        let clip_z = -(view.view_from_world * vec4<f32>(in.world_position.xyz, 1.0)).z;
        if (clip_z > wow_light.fog_params.w) {
            discard;
        }
    }

    // Enhanced never samples the classic ripple sheet. The 1x1 image retains the safe fallback.
    if w.kind.x < 0.5 && w.path.y > 0.5 && textureDimensions(scene_depth).x > 1u {
        var shallow_enhanced = wow_light.water_river[0];
        var deep_enhanced = wow_light.water_river[1];
        if w.kind.y > 0.5 && w.path.x < 0.5 {
            shallow_enhanced = wow_light.water_ocean[0];
            deep_enhanced = wow_light.water_ocean[1];
        }
        return enhanced_water(in, shallow_enhanced, deep_enhanced);
    }

    // Animated frame. For water/ocean this is the DETAIL ripple (RGB ≈ near-black, ALPHA = ripple);
    // for magma/slime it is the OPAQUE BODY texture.
    // `view.mip_bias`: the render-scale LOD compensation (1639), 0.0 at native and above.
    let detail = textureSampleBias(
        frames,
        frames_samp,
        apply_scroll(in.uv),
        frame_layer(),
        view.mip_bias,
    );

    // Magma / slime (kind.x > 0.5): the animated texture IS the opaque body colour — no depth swatch
    // (the ADT liquid vertex format carries no colour element at all, and the WMO one is a hard
    // `0xffffffff`, so there is nothing to modulate the sheet by) and no N·L (lighting state 0 on both
    // paths). It IS fogged, like every other liquid batch.
    //
    // The earlier "emissive / no-darken / no fog" reading here was WRONG, and wrong twice over: it came
    // from the ADT-lava row of `rf-water-liquid-type-texture-material`, which read GX state `0x37` as an
    // emissive path when `0x37` is the per-stage TEXTURE-MATRIX enable pushing an identity — a texgen
    // *reset*; and that row is the ADT queue, which never dispatches slime at all (Undercity's slime is
    // WMO liquid). Skipping fog is what made a submerged slime surface a flat unshaded sheet at any
    // depth instead of one that recedes into the murk. (VERIFIED wow-re `liquid-render-state-sided`
    // §3/§3.1/§5, which corrects that row.)
    if (w.kind.x > 0.5) {
        return vec4<f32>(apply_fog(detail.rgb, in.world_position.xyz, in.room_fog), 1.0);
    }

    // Per-vertex swatch coord V (in `in.depth`, computed CPU-side in wow-formats/liquid.rs): river/lake
    // = `clamp(byte/42)` (VERIFIED WoW.exe `c81768` LUT / `FUN_0068d790`, saturating ~5 yd so the channel
    // middle reaches the deep/teal row), ocean = `clamp(byte/255)` (VERIFIED `c7fcd8` / `FUN_0068d690`,
    // its own LUT on its own authored byte scale — decision 2069). The depth swatch
    // is a plain 2-endpoint lerp (`FUN_0068a830`), so a SINGLE V indexes BOTH the colour and the alpha
    // row: colour `shallow→deep` and opacity `shallow_α→deep_α` track together. (Earlier `×4` colour
    // compression + the gentle `byte/255` V were band-aids for a wrong "V tops at 0.31" belief — removed.)
    let depth = clamp(in.depth, 0.0, 1.0);
    // The kind's swatch endpoints, off the shared light: ocean reads rows 15/16 (IntBand 14/15),
    // river/lake rows 13/14 (IntBand 16/17). Both are packed every frame by `build_light_data`.
    var shallow = wow_light.water_river[0];
    var deep = wow_light.water_river[1];
    if (w.kind.y > 0.5) {
        shallow = wow_light.water_ocean[0];
        deep = wow_light.water_ocean[1];
    }
    // ---- The two WMO water arms ------------------------------------------------------------
    //
    // Neither is the ADT combine below. `0x6b62e0`'s category 0 splits on the owning group's
    // `MOGP.flags & 0x48`, and both halves bind ONE texture (the animated sheet) with no depth ramp
    // anywhere — there is not a single reference to the ADT ramp globals `0xc7fbc0`/`0xc81768`/
    // `0xc7fcd8` in all of `[0x6b0000, 0x6c4000)`. Opacity on both is the per-vertex authored byte
    // through the zone's linear alpha ramp, which `in.depth` carries and this lerp reproduces
    // (`wmo_water_alpha_v`). VERIFIED wow-re `terrain/scratch/water-shading-law.md` §11.
    let vtx_alpha = mix(shallow.w, deep.w, depth);
    if (w.path.x > 1.5) {
        // ---- WMO INTERIOR (`0x6b6420`) — 134 of the game's 164 water groups, Blackfathom included.
        //
        // Fixed-function, always: the kernel body contains no `mov ecx,0x3f` at all (a positive
        // finding — the same scan on the exterior kernel finds two), and `[0xc9607c]`, the
        // specular/pixelShaders gate, is never read on this path. So it ignores those CVars, runs
        // with lighting OFF (`0x0e = 0`) and fog ON (`0x0f = 1`), and its whole output is the
        // combine preset `(0x1f, 3)` over one texture stage:
        //
        //     rgb = clamp(Cf + Ct)      alpha = clamp(Af + At)
        //
        // `Cf` is the pool's `MOMT[materialId].diffColor` taken RAW — baked into the mesh's vertex
        // colour, which is where the reference's own 6-float vertex carries it. NO sun term and no
        // sheen: that vertex has no normal to compute one from.
        //
        // The ALPHA op is the part worth stating. Preset 3 is `GL_ADD` on both channels via
        // `GL_COMBINE` (`COMBINE_ALPHA = GL_ADD` @`0x85c2fc`, operands left at the GL default), NOT
        // the legacy `GL_TEXTURE_ENV_MODE = GL_ADD` whose alpha would be `Af · At`. The client takes
        // the COMBINE path because `GL_ARB_texture_env_combine` is GL 1.3 core. So the ripple **adds**
        // to the pool's opacity instead of multiplying it away — a pool is at its authored opacity in
        // the troughs and saturates opaque on the crests, which is the opposite of what the legacy
        // reading would have drawn.
        let body = clamp(in.vcolor.rgb + detail.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
        return vec4<f32>(
            apply_fog(body, in.world_position.xyz, in.room_fog),
            clamp(vtx_alpha + detail.a, 0.0, 1.0),
        );
    }
    if (w.path.x > 0.5) {
        // ---- WMO EXTERIOR (`0x6b6630`) — Stormwind's canals and fountains.
        //
        // This arm DOES bind a pixel program, `Shaders\Pixel\MapObjExtWater0.bls` (bound at
        // `0x6b6654`, unbound `0x6b689c`), under the `[0xc9607c]` specular/pixelShaders gate. Both
        // CVars default to "0", but our reference install's `Config.wtf` sets both to "1", so the
        // shader leg is what every director comparison is against and it is the leg we implement.
        // Decoded verbatim from the asset:
        //
        //     rgb = primary.rgb + detail.rgb + secondary·detail.a      alpha = primary.a
        //
        // **No `+0.25`.** That constant is the ADT program's own `PARAM` and has no counterpart
        // here, so carrying it over — which is what we did — added a flat achromatic lift to every
        // canal pixel at all times, sun or no sun. Against the sheet's real texel distribution that
        // is ~3.5x the ripple contrast off the glint, and it is why the reference's canal shimmers
        // only where the sun is while ours sparkled everywhere.
        //
        // `primary` is FFP-lit over a constant up-normal with the vertex colour tracked into
        // ambient+diffuse (`glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)`), so the band
        // is the MATERIAL colour and `primary = band · clamp(ambient + diffuse·max(N·L, 0))`.
        // Lighting really is on here: neither this kernel nor its dispatch touches render-state id
        // `0x0e`, and the control that such a call would be findable is the interior kernel, which
        // does exactly that at `0x6b65bf`.
        //
        // The band is a SINGLE one — there is no bathymetry to lerp by — and it is the **deep** river
        // row, `LightIntBand` sub-17, i.e. `water_river[1]`. Read as a hard immediate
        // (`0x6b66be add edi, 0xec`), so nibbles 0, 4 and 8 all take it; exterior ocean cannot arise
        // (category 2 falls to a bare epilogue with no draw).
        //
        // **Sub-17, not sub-16, and the distinction cost a round trip.** A DayNight band *slot* is not
        // a `LightIntBand` *sub*: `0x6d64d0` displaces sub-8 out to record `+0x4c`, so `sub = slot + 1`
        // across slots 8–16, and the kernel's slot 16 is sub-17. The shipped data says the same thing
        // twice over — across all 367 LightParams rows carrying river bands, sub-16 is browns, olives
        // and muddy yellows (G > B in 71%: the shallow colour of water over a riverbed) while sub-17 is
        // blues and teals. At Stormwind sub-16 is `(79, 93, 20)`, which renders the canals olive-green;
        // sub-17 is `(51, 82, 85)`. The ADT ramp runs sub-16 → sub-17 across its 64 texels; WMO
        // exterior water takes the deep end alone, flat.
        let n_ext = normalize(in.world_normal);
        let to_light_ext = -normalize(wow_light.light_sun.xyz);
        let primary_ext = clamp(
            wow_light.light_ambient.rgb + wow_light.light_diffuse.rgb
                * max(dot(n_ext, to_light_ext), 0.0),
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        ) * deep.rgb;
        let rgb_ext = primary_ext + detail.rgb + in.secondary_vtx * detail.a;
        // `result.color.w = fragment.color.primary` — the vertex alpha ALONE. The bound program
        // bypasses the texture environment entirely, so the interior arm's `+ At` does not apply here.
        return vec4<f32>(apply_fog(rgb_ext, in.world_position.xyz, in.room_fog), vtx_alpha);
    }

    // Body colour: lit vertex colour × the depth-lerped water-row swatch colour (`primary·colorTex`).
    let n = normalize(in.world_normal);
    let to_light = -normalize(wow_light.light_sun.xyz);
    let ndotl = max(dot(n, to_light), 0.0);
    let primary = clamp(
        wow_light.light_ambient.rgb + wow_light.light_diffuse.rgb * ndotl,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );

    // Sun sheen (`secondary`): the `ocean0_s.bls` Blinn highlight, computed PER-VERTEX in `fn vertex`
    // and interpolated across the coarse ~4 yd MCLQ mesh — the faithful 1.12 path (the real client
    // evaluates it in its FFP vertex stage). Per-pixel evaluation of the sharply-peaked `pow(N·H,6)`
    // would fill its broad lobe at full value (a brighter, denser sheen); interpolating from the
    // vertices flattens the peak to match the reference. (A per-pixel/per-vertex A/B toggle proved the
    // two visually identical on our mesh — we keep per-vertex as the faithful mechanism; RE:
    // `docs/knowledge/scratch/liquid-depth/fleck-deep.md`.)
    let secondary = in.secondary_vtx;

    // The stage-0 depth swatch, built and sampled as the reference does (see `swatch_row`): a
    // byte-space 64-row ramp that stops short of the deep endpoint, LINEAR across rows, with the
    // ocean's last row darkened `floor(0.9*byte)` and its alpha forced opaque.
    let swatch = swatch_at(shallow, deep, depth, w.kind.y > 0.5);

    // primary·colorTex.rgb  +  detail.rgb  +  (secondary + 0.25)·detail.a   (the ocean0_s.bls math)
    var rgb = primary * swatch.rgb + detail.rgb + (secondary + vec3<f32>(0.25)) * detail.a;

    // Opacity: depth ramp between the shallow/deep LightParams water alphas, over the SAME V as the
    // colour. Deeper = more opaque, up to α=1.0 where V saturates (river/lake byte 42 ≈ 5 yd), so the
    // channel middle is opaque + teal while the shore stays semi-transparent (V→0, α≈0.5) and the bottom
    // shows through (faithful — the pale edge band). One steep V drives both colour and opacity together.
    let alpha = swatch.w;

    // Distance fog (see `apply_fog`) — the water fog colour is also teal, so far water converges on the
    // haze.
    rgb = apply_fog(rgb, in.world_position.xyz, in.room_fog);

    // GAMMA LANE (0161): raw gamma out; alpha blends in gamma like the reference's bytes.
    return vec4<f32>(rgb, alpha);
}
