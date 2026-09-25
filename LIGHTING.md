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
| Night and interior level | global dimming of the night sky term and of interior ambient | `nightGain`, `interiorGain`, `interiorBakeFloor` |

Players reach all of it from **Options -> Advanced Graphics** (a Lighting Quality preset Off / Low / Medium / High plus the individual rows; Off is the original client look). The dev build has a panel for all of it: **Ctrl+Shift+D → Lighting & shadows**, with Dim / Default /
Bright presets.

The optional enhanced water (Water Quality, refraction, caustics, High reflections, lava glow)
is its own module, documented in `WATER.md`.

## Where the code lives

- **Shaders**: `crates/benilla-assets/src/shaders/shadow_hook.wgsl` (shared shadow sampling),
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
- **Census**: `cargo test -p benilla-world --lib lighting::daylight::census -- --ignored --nocapture`
  (`WOW_CENSUS_WMO`, `WOW_CENSUS_PLACE=map,tx,ty,uid`) prints, per interior room of a city WMO,
  whether a daylight fixture, a bleed fixture or neither reaches it, and the sky-room count.
- **Tools**: `benilla-extract <Data> wmolights <wmo> [--verts <group>]`, `wmolamps`, `m2firescan`
  print the inputs the system works from (groups, batch classes, portals, claims, flame emitters).

## Invariants worth knowing before editing

- The shared light buffer stays 8528 bytes and the torch table 6416 bytes; both are mirrored in
  three shaders and pinned by tests.
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
