# benilla-lighting

A dynamic light and shadow system for [benilla](https://github.com/samwhosung/benilla), the Rust +
Bevy reimplementation of the 1.12.1 client. This branch is upstream benilla (merged at
`fd386e75`) plus the lighting work, the enhanced water module and volumetric fog, and nothing else. It needs no data changes: everything is
derived from what a 1.12 install already carries (WMO `MOLT` lights, `MOCV` vertex colour, portals,
M2 particle emitters, `Light.dbc`).

Everything here is a departure from the reference client's look, so every lane has a live cvar and
most can be switched off. Code is tagged `MONKEY (<topic>)` with the rationale and the measurements
behind each constant.

## What it does

| Lane | What you see | Main switch |
|---|---|---|
| Sun shadows, characters | real silhouettes for units instead of the oval blob | `characterShadows` |
| Sun shadows, world | trees, buildings and alpha-tested foliage cast; baked terrain shadows step aside | `worldShadows` |
| Dynamic interiors | WMO rooms are lit per fragment by their own fixtures with a soft falloff, gated per room so light does not leak through walls | `interiorLight`, `interiorRoomGate` |
| Synthesised lights | torches, braziers, lanterns, campfires and lampposts emit a light in the colour of their flame, with flicker | `fireLightGain`, `fireFlicker` |
| Torch shadows | cube-map shadows from point lights onto buildings, models and terrain, with a contact-hardening penumbra and a shadow floor | `interiorShadows`, `exteriorShadows`, `torchShadowStrength`, `interiorShadowSoft` |
| Daylight and doorways | calibrated fixtures at doors, windows and open boundaries carry daylight into a room, and doorways between rooms bleed light | `interiorDaylight`, env `WOW_DAYLIGHT`, `WOW_BLEED` |
| Terrain in torch shadows | the ground casts into an outdoor fire's cube map, so a hill or bank blocks the fire (settled exterior slots only; `torch_terrain.rs` gathers the resident MCNK chunks inside the cube's 48 yd) | `torchTerrainShadows` (default 0, High 1), env `WOW_TORCH_TERRAIN=0\|1` |
| City day floor | `interiorDaylight`'s room floor also reaches Stormwind/Ironforge rooms that the portal graph connects to the sky (an exterior-facing portal, or one hop from one; `district_sky_rooms` in `daylight.rs`); before this no city room could take it | `interiorDaylight` (default 0, suggested High 0.10) |
| Spell and firework lights | fire, holy and fel effects light their surroundings for their lifetime; frost, nature, arcane and shadow do not | `spellLightGain`, env `WOW_SPELL_LIGHT=0` |
| Moon shadows | at night the same shadow rig re-aims at the moon and casts a faint shadow; dims only the night sky term, never point lights | `moonShadowStrength` |
| Ground-effect spells | Flamestrike, Rain of Fire, Consecration, Flare and fire traps light the ground for their duration; frost and nature areas stay dark | `spellLightGain` |
| Volumetric fog | near-field haze that converges on the zone fog colour (clear within 10 yd, full by 150 yd; mistier at dawn and in bad weather, faint indoors) and sun/moon light shafts through gaps, sampled from the shadow map; own fullscreen pass after the main pass (`benilla-app/src/volumetric_fog.rs`) | Advanced Graphics → Volumetric Fog (Off/Low/High), cvar `volumetricFog`, env `WOW_VOLFOG=0\|1\|2` |
| Sky dithering | a faint screen-space dither in the FFXGlow combine so smooth sky and fog gradients do not band (MONKEY p0; was env-only) | Advanced Graphics → Sky Dithering, cvar `skyDither` 0/1 (default 0, Graphics preset High = 1), env `WOW_DITHER=1` still forces it on; bridge in `benilla-app/src/monkey_gfx.rs` |
| HDR emission and bloom | lit WMO windows, additive models/particles and magma can exceed display white; a soft-capped fullscreen halo is added before the faithful FFX clamp | Advanced Graphics → Bloom (Off/Low/High), cvar `bloom`; Off is the unchanged reference image, suggested High preset value 2 |
| Sky quality | Enhanced: the five Light.dbc sky stops through a smooth monotone curve in linear light (no bands at the rings), a soft sun glow tinted by sun and fog colour (fades at night and under cloud), a procedural star field with twinkle and a faint Milky Way over the stock `Stars.m2`. High adds domain-warped cloud detail and sun-lit clouds (self-shadow, silver lining; technique from WarcraftXL, see `THIRD-PARTY.md`). Classic is the reference sky unchanged | Advanced Graphics → Sky Quality, cvar `skyQuality` 0 Classic / 1 Enhanced / 2 High, env `WOW_SKY_QUALITY` |
| Modern fog | MONKEY (fog): radial distance fog with a gradual exponential curve, a daylight sun lobe, and an end colour shared by the world, WDL hull, sky horizon and volumetric haze. Each zone keeps its authored fog end up to the reference view-distance limit; optional `LightFogBand.dbc` rows provide height, sun and end-fog controls. Interior WMO fog stays classic | Advanced Graphics → Modern Fog, cvar `fogModel` 0 Classic (default) / 1 Modern (High = 1), env `WOW_FOGMODEL=0\|1`; `fog_hook.wgsl`, `lighting/fog_model.rs`, `light/fog_band.rs` |
| Rain on surfaces | MONKEY (wet): rain gradually darkens and saturates exposed terrain and models, adds puddles and a sky/sun sheen, and produces rings on exterior Enhanced water. Wetness rises over roughly 90 seconds and dries over roughly four minutes; phase 1 has no rain-occlusion map, so shelter is classified only by interior/unlit state and surface facing | Advanced Graphics → Wet Surfaces in Rain, cvar `rainSurfaces` 0/1 (default and High 1), env `WOW_RAIN_SURFACES`, `WOW_WETNESS`, `WOW_WET_T`; `weather/wetness.rs`, `wet_hook.wgsl` |
| Foliage wind | one weather-fed gust/veer field drives grass and classified tree/bush foliage; grass also parts around the player and nearby units | Advanced Graphics → Foliage Wind, cvar `foliageWind` 0 Off / 1 Grass / 2 Grass + Trees (High = 2), capture override `WOW_FOLIAGE_WIND`; field and benders in `benilla-world/src/wind/`. The wave phase is the CPU-integrated travel (speed integrated over time, wrapped seamlessly), and the waves' spatial term uses the fixed profile heading; the veer turns only the bend. Tree exile copies sway through material marker bit 14 (`FOLIAGE_WIND_MARKER`). Capture offset `WOW_CAPTURE_WIND_T` (s). Known: shadow casters (world-shadow proxies, `torch_depth.wgsl`) do not apply the sway, so leaves sample a static shadow; a future world-camera DepthPrepass must call the same offsets |
| Night and interior level | global dimming of the night sky term and of interior ambient | `nightGain`, `interiorGain`, `interiorBakeFloor` |

Players reach all of it from **Options -> Advanced Graphics** (a Graphics Preset over everything, a Render Distance slider, a Lighting Quality preset Off / Low / Medium / High / Ultra plus the individual rows; Classic / Off is the original client look). The dev build has a panel for all of it: **Ctrl+Shift+D → Lighting & shadows**, with Dim / Default /
Bright presets.

## Graphics Preset

MONKEY (presets). The top row of Advanced Graphics, cvar `graphicsQuality`: Classic / Low / Medium /
High / Ultra, or Custom when the rows match none of them. The table is `GRAPHICS_PRESETS` in
`benilla-app/src/cvars.rs` (one line per governed row); choosing a rung writes each row through
the ordinary cvar path, and the label is re-derived every frame, so editing any governed row
(or any Lighting Quality member) shows Custom. `lightingQuality` is itself a governed row, so
every row the lighting ladder owns (`LIGHTING_PRESETS`: both sun lanes, shadow resolution, the
room and torch lanes, moon shadows, `waterQuality`, `volumetricFog`, `bloom`, `sunShafts`,
`colorGrading`, `foliageWind`, the fire/spell/lava gains, `nightGain`, `interiorGain`) follows it.

| row | Classic | Low | Medium | High | Ultra |
|---|---|---|---|---|---|
| `lightingQuality` | Off | Low | Medium | High | Ultra |
| `farclip` (Render Distance, yd) | 350 | 350 | 477 | 777 | 1497 |
| `skyQuality` | 0 | 1 | 1 | 2 | 2 |
| `skyDither` | 0 | 1 | 1 | 1 | 1 |
| `fogModel` | 0 | 1 | 1 | 1 | 1 |
| `rainSurfaces` | 0 | 1 | 1 | 1 | 1 |
| `torchTerrainShadows` | 0 | 0 | 0 | 1 | 1 |
| `daylightWindowSplit` | 0 | 1 | 1 | 1 | 1 |
| `ambientOcclusion` (TODO, not yet registered) | 0 | 0 | 1 | 2 | 2 |
| `zoneSkyboxes` (TODO, not yet registered) | 0 | 1 | 1 | 1 | 1 |
| `lampFog` (TODO, not yet registered) | 0 | 0 | 1 | 2 | 2 |

What the lighting rungs set, beyond Off (everything off, gains 1.0) and High (the registered
defaults): Low = character shadows only at a 1024 map, no torch or moon shadows, bloom Low;
Medium = both sun lanes, indoor torch shadows (6 resident / 2 moving), moon shadows, bloom Low,
no sun shafts or colour grading; Ultra = High with a 4096 sun map, 16 / 8 torch maps, High water
and High volumetric fog. No rung decides `shadowDistance` or `interiorShadowSoft`.

**Default: High.** A player whose `config.toml` names no `graphicsQuality` is seeded to the High
column at startup (`Cvars::seed_graphics_preset`) on every row the file does not carry; the rows'
registered defaults stay their per-lane ones (`farclip` the reference's 350), and captures and
tests, which never seed, keep them. The Advanced page's Defaults button lands on High too.

**Render Distance** is `farclip`, the same cvar as the Graphics page's Terrain Distance (which
keeps the stock 177–777 slider). Its clamp (`view::FARCLIP_RANGE`) now reaches `FARCLIP_MAX` =
1497 yd, the 1.12 grid of 60 carried on. Terrain residency (`terrain_stream/window.rs`), the
model/WMO and particle walls (`view::within_farclip`) and the art-cache sweep radius (now 7 tiles,
`art_scope.rs`) all follow it; at 1497 the residency window is at most 7×7 ADT tiles against
777's 5×5, which is the memory bound. Past 777 only Modern fog (`fog_model::modern_fog_end`)
stretches the fog end with it; under Classic fog the zone's fog end still hides the extra reach.

The optional enhanced water (Water Quality, refraction, caustics, High reflections, lava glow)
is its own module, documented in `WATER.md`.

## Where the code lives

- **Shaders**: `crates/benilla-assets/src/shaders/shadow_hook.wgsl` (shared shadow sampling),
  `fog_hook.wgsl` (MONKEY p0: the ONE distance-fog law — `fog_linear`, `fog_sample`, `apply_fog`;
  terrain, wow_model, static_gx, liquid, enhanced_water, wow_effect and wdl all call it; classic is
  bit-identical to the old per-shader copies, and `MonkeyFrame.fog_model` = 1 is the FOG lane's
  extension point), `monkey_frame.wgsl` (the MonkeyFrame struct),
  `crates/benilla-assets/src/shaders/emissive_hook.wgsl` (shared opt-in HDR multipliers),
  `crates/benilla-world/src/shaders/torch_depth.wgsl`, and the lighting lanes inside
  `static_gx.wgsl`, `wow_model.wgsl` and `terrain.wgsl`. The three receivers mirror each other; the
  comments say where.
- **World** (`crates/benilla-world/src`): `lighting/global_light.rs` (light table packer, lanes,
  room claims, gains), `lighting/daylight.rs`, `lighting/flicker.rs`, `static_gx/torch_depth.rs`
  (cube-map cache), `static_gx/shadow.rs`, and the light spawning in `terrain_stream/spawn/`.
- **Data rules** (`crates/benilla-formats/src`): `fire_light.rs` (which models emit light, their
  colour and reach, the spell school rules) and `room_claim.rs` (which rooms a light may light).
- **App** (`crates/benilla-app/src`): `torch_shadow.rs` (caster selection), `shadow_core.rs`,
  `character_shadow.rs`, `world_shadow.rs`, `blob_shadow.rs`, `entities/carried_light.rs`,
  `entities/spell_fx/lifecycle.rs`, `dynamic_interior.rs`, `debug_panel/lighting_controls.rs`, and
  the cvars in `cvars.rs` / `video.rs`.
- **Sky** (`crates/benilla-world/src`): `sky_fx.rs` (tier, clock, glow inputs), `shaders/sky_fx.wgsl` (gradient curve, glow, stars, cloud noise), the tier branches in `shaders/sky.wgsl` and `shaders/cloud.wgsl`, `clouds/layer.rs` (`update_cloud_fx`); the cvar bridge is `benilla-app/src/sky_quality.rs`.
- **Census**: `cargo test -p benilla-world --lib lighting::daylight::census -- --ignored --nocapture`
  (`WOW_CENSUS_WMO`, `WOW_CENSUS_PLACE=map,tx,ty,uid`) prints, per interior room of a city WMO,
  whether a daylight fixture, a bleed fixture or neither reaches it, and the sky-room count.
- **Wind**: `wind/mod.rs` resolves the WXL-derived shared field and nearest-unit benders;
  `clutter.rs` authors blade weights/phases; `static_gx` classifies and marks alpha-tested leaf
  batches. `wind_hook.wgsl` owns the displacement shared by clutter, retained trees and fade twins.
- **Tools**: `benilla-extract <Data> wmolights <wmo> [--verts <group>]`, `wmolamps`, `m2firescan`
  print the inputs the system works from (groups, batch classes, portals, claims, flame emitters).

## Invariants worth knowing before editing

- The shared light buffer's per-frame blob is 8784 bytes and the torch table 6416 bytes; both are
  mirrored in the shaders and pinned by tests. The blob is 21 header rows (336 B), the 256-slot
  point-light table (8192 B), then the MONKEY **MonkeyFrame** block (16 rows, 256 B; nothing before
  it moves). Every `WowLight` mirror (terrain, wow_model, static_gx, liquid, enhanced_water,
  wow_effect, wdl) declares it as `monkey: monkey_frame::MonkeyFrame` after `points`:

  | row | x | y | z | w |
  |---|---|---|---|---|
  | `fog_a` | height_fog_density | height_fog_height | height_fog_falloff | curve_blend |
  | `fog_b` | sun_fog r | g | b | sun_fog_strength |
  | `fog_c` | end_fog r | g | b | end_fog_distance |
  | `fog_d` | fog_model (0 classic; Modern = scene fog end in yd) | sun_fog_angle | sun direction octahedral x | y |
  | `wind_a` | dir_x | dir_y | base_heading (rad) | gust |
  | `wind_b` | travel (yd, wrapped at 4096) | sway_strength | grass_strength | tree_strength |
  | `wet_a` | rain_rate | wetness | ripple_time_s | snow |
  | `misc` | bender_count | time_of_day 0..1 | night 0..1 | 0 |
  | `benders[8]` | world x | world y | world z | radius |

  Positions are Bevy world space (Y up, yards; WoW `(x, y, z)` = Bevy `(-y, z, -x)`), the wind
  direction is a unit vector in world XZ. Lanes write the `MonkeyFrame` resource
  (`benilla-world/src/lighting/monkey_frame.rs`); `global_light::pack_monkey_frame` packs it after
  the point table and fills `time_of_day` / `night` itself. All zero = no visual change.
- World lights use upstream's `WorldPointLight`, never Bevy's `PointLight`.
- Foliage wind is vertex-only. Tree and leaf shadow meshes stay static; the small mismatch is the
  accepted first-stage cost/complexity tradeoff.
- WGSL only fails at pipeline creation, so a shader edit is verified by running a world scene, not
  by `cargo check`.
- Exterior point lights are selected per draw unit (12 slots, ranked against the chunk's box); the
  cube-map occlusion is evaluated for the nearest three.

## Build

```bash
cargo build --release -p benilla
```

Debug tracing: `WOW_TORCH_TRACE=1`, `WOW_POINTS_DUMP=1`, `WOW_SHADOW_TRACE=1`, and the cvar
`interiorDebug` 1..4.

## Licence

Same as upstream: MIT OR Apache-2.0. Code and techniques taken from other projects (WarcraftXL,
by iThorgrim) are credited file by file in [`THIRD-PARTY.md`](THIRD-PARTY.md).
