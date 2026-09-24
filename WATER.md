# benilla-water

An optional enhanced water renderer for [benilla](https://github.com/samwhosung/benilla), the Rust +
Bevy reimplementation of the 1.12.1 client. It needs no data changes: everything is derived from
what a 1.12 install already carries (MCLQ / MLIQ liquid grids and their authored depth bytes, the
`Light.dbc` water and sky rows, the opaque scene depth).

It is a **module on top of the reference water, not a replacement for it**. The reference path
(`Classic`) is untouched and stays byte-identical to upstream; the enhanced path only runs when the
player picks it, and every part of it can be switched off.

## Credits

- **Original project: [WarcraftXL](https://github.com/WarcraftXL)** — the `wxl-experimental-water`
  module for the 1.12 client.
- **Author: iThorgrim.**

The WarcraftXL water module is the design reference for this one, and its author has given
permission to reuse its code here on the condition that the author and the original project are
named. This section is that attribution; it must travel with the module (this file, and the
`liquid.wgsl` header) wherever the module is copied or shared.

Rule for contributors: a file that ports code or a technique from WarcraftXL says so in its header
(`Portions derived from WarcraftXL wxl-experimental-water by iThorgrim, used with permission`) and
the technique is listed in the table below. Do not remove either.

| Technique | Status here | From WarcraftXL |
|---|---|---|
| Visual-only wave riding for swimmers (authoritative position never moves) | shipped (`water_fx/bob.rs`) | same principle as `world/Ride.cpp`, implemented independently |
| Shore surf limited by the TERRAIN column, so objects in the water never foam | shipped (authored depth gate, `on_bed`) | principle from `sea/Shore.hpp` |
| Per-channel extinction + scene-copy refraction | planned | `shaders/Surface.ps.hlsl`, `render/Refraction.cpp` |
| Caustics, two layers multiplied | planned | `shaders/Surface.ps.hlsl`, `render/Noise.cpp` |
| One-probe screen-space reflection | planned (`High`) | `shaders/Surface.ps.hlsl` |
| Gerstner trains + breaker index, crest-fold foam | planned | `sea/Spectrum.*`, `sea/Shore.hpp`, `shaders/Wave.hlsli` |
| Specular lobe widened by the normal's screen derivative, distance glints | shipped / planned | `shaders/Surface.ps.hlsl` |

## What it does

| Lane | What you see | Main switch |
|---|---|---|
| Quality tiers | `Classic` = the reference water, `Enhanced`, `High` (currently = Enhanced) | Video options → Water Quality, cvar `waterQuality`, env `WOW_WATER=0\|1\|2` |
| Procedural waves | multi-band analytic waves with exact normals; the long swell moves ocean vertices | tier |
| Ocean / inland profiles | the sea and lakes/rivers have their own colour, energy and reflectivity | tier |
| Depth look | soft shoreline, depth-driven colour and opacity from the real scene depth | tier |
| Reflection and glints | Fresnel sky reflection from the zone's sky rows, sun path, moon path, point-light glints | tier |
| Beach surf | a travelling wave front, a lace of foam that dissolves behind it, a swash sheet on the sand | tier |
| Swimmers | wake foam, treading rings and a visual-only bob with the swell | tier |
| Lava glow | magma surfaces emit warm point lights through the lighting module | Advanced Graphics → Lava Glow, `lavaLightGain` |

Rivers deliberately have NO flow effect: a derived current was tried and removed at the owner's
request. If it ever returns it should read the authored MCLQ flow records, not derive one.

## Where the code lives

- **Shader**: `crates/benilla-assets/src/shaders/liquid.wgsl` — `enhanced_water()` and its helpers;
  the `fragment` entry point branches into it and otherwise runs the untouched reference path.
- **Assets** (`crates/benilla-assets/src`): `water_depth.rs` (the scene-depth binding),
  `materials.rs` (`LiquidExt`: quality, wave energy, sky rows, celestial), `lib.rs` (`WaterQuality`).
- **World** (`crates/benilla-world/src`): `liquid/scene_depth.rs` (resolves the opaque depth after
  the main opaque pass), `liquid/waves.rs` (the CPU mirror of the vertex swell), `liquid/surface.rs`
  (per-kind material parameters), `water_fx/bob.rs` (swimmer bob), `lighting/lava_light.rs`.
- **App** (`crates/benilla-app/src`): the setting in `cvars.rs` / `video.rs`, the options rows in
  `assets/ui/OptionsFrame.xml`, the capture scenes `water-*` and `lava-*` in `capture/scenarios.rs`.

## Invariants worth knowing before editing

- `Classic` must stay byte-identical: capture `water-noon` with `WOW_WATER=0` and compare with the
  baseline before and after any edit.
- The CPU swell (`liquid/waves.rs`) and the shader's `WATER_WAVES` table are one contract.
- Every screen derivative in `enhanced_water()` is taken at the top of the function (WGSL
  uniformity); WGSL only fails at pipeline creation, so verify by running a `water-*` scene.
- Never rotate a noise domain by a per-pixel angle at world coordinates (~1e4 yd): it shears.
- Run one GPU capture at a time, and run `cargo test` in a shell WITHOUT `WOW_DATA` exported.
