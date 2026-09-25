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
| Spell and firework lights | fire, holy and fel effects light their surroundings for their lifetime; frost, nature, arcane and shadow do not | `spellLightGain`, env `WOW_SPELL_LIGHT=0` |
| Moon shadows | at night the same shadow rig re-aims at the moon and casts a faint shadow; dims only the night sky term, never point lights | `moonShadowStrength` |
| Ground-effect spells | Flamestrike, Rain of Fire, Consecration, Flare and fire traps light the ground for their duration; frost and nature areas stay dark | `spellLightGain` |
| Volumetric fog | near-field haze that converges on the zone fog colour (clear within 10 yd, full by 150 yd; mistier at dawn and in bad weather, faint indoors) and sun/moon light shafts through gaps, sampled from the shadow map; own fullscreen pass after the main pass (`benilla-app/src/volumetric_fog.rs`) | Advanced Graphics → Volumetric Fog (Off/Low/High), cvar `volumetricFog`, env `WOW_VOLFOG=0\|1\|2` |
| Sky dithering | a faint screen-space dither in the FFXGlow combine so smooth sky and fog gradients do not band (MONKEY p0; was env-only) | Advanced Graphics → Sky Dithering, cvar `skyDither` 0/1 (default 0, Graphics preset High = 1), env `WOW_DITHER=1` still forces it on; bridge in `benilla-app/src/monkey_gfx.rs` |
| Night and interior level | global dimming of the night sky term and of interior ambient | `nightGain`, `interiorGain`, `interiorBakeFloor` |

Players reach all of it from **Options -> Advanced Graphics** (a Lighting Quality preset Off / Low / Medium / High plus the individual rows; Off is the original client look). The dev build has a panel for all of it: **Ctrl+Shift+D → Lighting & shadows**, with Dim / Default /
Bright presets.

The optional enhanced water (Water Quality, refraction, caustics, High reflections, lava glow)
is its own module, documented in `WATER.md`.

## Where the code lives

- **Shaders**: `crates/benilla-assets/src/shaders/shadow_hook.wgsl` (shared shadow sampling),
  `fog_hook.wgsl` (MONKEY p0: the ONE distance-fog law — `fog_linear`, `fog_sample`, `apply_fog`;
  terrain, wow_model, static_gx, liquid, enhanced_water, wow_effect and wdl all call it; classic is
  bit-identical to the old per-shader copies, and `MonkeyFrame.fog_model` = 1 is the FOG lane's
  extension point), `monkey_frame.wgsl` (the MonkeyFrame struct),
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
  | `fog0` | height_fog_density | height_fog_height | height_fog_falloff | curve_blend |
  | `fog1` | sun_fog r | g | b | sun_fog_strength |
  | `fog2` | end_fog r | g | b | end_fog_distance |
  | `fog3` | fog_model (0 classic, 1 modern) | sun_fog_angle | 0 | 0 |
  | `wind0` | dir_x | dir_y | speed | gust |
  | `wind1` | time_s | sway_strength | grass_strength | tree_strength |
  | `wet0` | rain_rate | wetness | ripple_time_s | snow |
  | `misc` | bender_count | time_of_day 0..1 | night 0..1 | 0 |
  | `benders[8]` | world x | world y | world z | radius |

  Positions are Bevy world space (Y up, yards; WoW `(x, y, z)` = Bevy `(-y, z, -x)`), the wind
  direction is a unit vector in world XZ. Lanes write the `MonkeyFrame` resource
  (`benilla-world/src/lighting/monkey_frame.rs`); `global_light::pack_monkey_frame` packs it after
  the point table and fills `time_of_day` / `night` itself. All zero = no visual change.
- World lights use upstream's `WorldPointLight`, never Bevy's `PointLight`.
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

Same as upstream: MIT OR Apache-2.0.
