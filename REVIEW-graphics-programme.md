# Review: `graphics-programme` vs `lighting`

Scope: `git diff origin/lighting...origin/graphics-programme` (97 commits, 108 files, +12091/−565), read
subsystem by subsystem: shared light buffer + WGSL hooks, post (bloom/grading/shafts/SSAO/lamp fog),
sky/skybox/fog model, water/weather/wind, cvars/presets/UI, licence headers. Read-only review; findings
are verified by reading code. No WoW data in this container, so no captures. `cargo check -p benilla`
could not complete here: `libudev-sys` build script fails (`libudev.pc` not installed) — environment,
not the branch.

## Findings, by severity

### 1. HIGH — GPL-3.0 code ported into an MIT OR Apache-2.0 crate
`crates/benilla-app/src/post/grading.rs:3`, `crates/benilla-app/src/post/grading.wgsl:1-2`,
`THIRD-PARTY.md:34`.
Both files and the THIRD-PARTY row say the grading is "ported from the GPL-3.0-or-later source
implementation, Copyright (C) 2026 WarcraftXL". The workspace is `license = "MIT OR Apache-2.0"`
(`Cargo.toml:8`) and METHOD.md makes the licence a provenance claim. A port of GPL code is a
derivative work: it cannot be relicensed MIT/Apache, and shipping it makes the `benilla` binary GPL.
The other WarcraftXL rows are covered by the author's attribution-only permission (WATER.md); this one
is explicitly GPL.
**Fix:** either obtain a written MIT/Apache grant from the author for `wxl-retail-grading` and record
it in THIRD-PARTY.md, or replace the port with a clean-room implementation. A 32³ LUT lookup through a
3D texture is standard and trivial to write independently. Until then, keep the files out of `main`.

### 2. MEDIUM — Licence headers/credits missing or inconsistent (THIRD-PARTY rule)
The rule requires both the exact header
`Ported from WarcraftXL (https://github.com/WarcraftXL) by iThorgrim — module <x>, <files>.`
and a THIRD-PARTY.md row:
- `crates/benilla-app/src/post/grading.rs:3` and `post/grading.wgsl:1`: **no standard header**. The
  author is not named ("by iThorgrim") and there is no URL. They carry only a GPL copyright line (see #1).
- `crates/benilla-world/src/liquid/waves.rs`: **no header at all**, although THIRD-PARTY.md:35 lists it
  in the Gerstner row ("the CPU swimmer sample inverts the same displacement (ported)").
- `crates/benilla-world/src/sky_fx.rs:9`: has the header but **no THIRD-PARTY.md row**. The rows name
  `sky_fx.wgsl`, `cloud.wgsl` and `clouds/layer.rs` only.
- `crates/benilla-world/src/clouds/layer.rs:247`: the credit is on a function doc comment, not in
  the file header. That is acceptable as a block credit, but the file header rule is not met.
  **Fix:** add the missing headers, add a `sky_fx.rs` row, and move or copy the layer.rs line into the
  module header.

### 3. MEDIUM — Skybox animation clock is an unbounded f32
`crates/benilla-world/src/skybox.rs:717`: `let now = time.elapsed_secs();` feeds `band_t` and
`gseq = f64::from(now)`, so the precision is already lost before the f64 widening. After about 3 days
of uptime the f32 ulp is about 0.03 s, and about 0.06 s after a week. Every skybox bone, UV and alpha
track (now applied to all skyboxes, not only the CoT spin) then steps visibly.
**Fix:** use `time.elapsed_secs_f64()` for `gseq`, and compute `band_t` as
`(elapsed_f64 % duration as f64) as f32`.

### 4. MEDIUM — Classic skyboxes are no longer unchanged with `zoneSkyboxes = 0`
`crates/benilla-world/src/skybox.rs`: the new material and animation work also runs for the WMO and
ghost skybox slots, which exist on the old path:
- the colour-alpha × transparency track now scales the layer weight (`~:757`), where the old code
  ignored it;
- a non-white M2 colour now adds `ATTRIBUTE_COLOR` to the mesh (`~:484`);
- deterministic captures now pose at `t = capture_t()` (`~:730`), where the old code used the bind pose;
- main batch order base moved `1..` → `25..` (`MAIN_ORDER_BASE`, `:52`).

This is arguably closer to 1.12, but it lands under the Off setting with no note. Classic baselines
(ghost and WMO skyboxes such as CoT) can shift.
**Fix:** decide per item. Either gate the item on the zone slot, or keep it and write it down as a
fidelity fix (commit or comment), then re-baseline the ghost and WMO skybox captures.

### 5. MEDIUM — SSAO normal can be NaN on isolated pixels against the sky
`crates/benilla-app/src/shaders/ssao.wgsl:94-100`.
`neighbour()` returns `centre + (0,0,-1e4)` for a sky tap. When both horizontal neighbours are sky,
`dx = (0,0,1e4)`, and when both vertical neighbours are sky, `dy = (0,0,1e4)`. Then `cross(dx,dy) = 0`
and `normalize(0)` is NaN.
This happens on 1-px poles, ropes and branch tips against the sky inside the fade range. The NaN then
passes through `max(0, NaN)`, which is undefined in SPIR-V `FMax`, and through the 4×4 blur into the
multiply over the HDR scene. From there Bloom spreads it into a halo block.
**Fix:**
`let c = cross(dx, dy); let l2 = dot(c, c); if (l2 < 1e-12) { return vec4(0.0, dist, protect, 1.0); } var n = c * inverseSqrt(l2);`

### 6. MEDIUM — High seed boots `farclip` at 777 with no `Deviates` row
`crates/benilla-app/src/cvars.rs:345` registers `farclip` with the reference's 350, but
`Cvars::seed_graphics_preset` writes the High column (777) for any player whose `config.toml` lacks
it. The value a player actually boots with therefore differs from the reference. METHOD.md: "Every
option boots at the stock 1.12 value… costs an explicit `Deviates` row". The Reference/Deviates test
checks only the registered default, so it stays green. The other seeded rows are `ours(...)` and are
unaffected.
**Fix:** this is the maintainer's call. Either exclude `farclip` from the seed, or declare the
deviation and add a test that walks the seeded column against every reference row.

### 7. MEDIUM (visual, High water only) — Cracks at the near-mesh LOD boundary on ocean
`crates/benilla-world/src/liquid/lod.rs:98-110` and `subdivide` (`:125-216`), together with the
`liquid.wgsl` vertex swell.
A refined chunk adds 3 vertices per shared edge (T-junctions). The Gerstner swell is evaluated per
vertex, with 4.17 yd spacing against 12.8/18 yd wavelengths. The refined edge's midpoints move about
0.1 yd vertically and up to about 0.3 yd horizontally off the coarse neighbour's straight edge. The
result is a sparkling crack line along the 64–80 yd ring on ADT ocean. The module doc's "shares the
same boundary" holds only before displacement.
**Fix:** tag the non-corner edge vertices of the fine grid (e.g. in `UV_1.y`). In the vertex stage,
displace them by the lerp of the swell at the two coarse endpoints, or stitch the fine chunk's outer
ring to the coarse edge.

### 8. LOW-MEDIUM — Enhanced water fragment shades the swell at the displaced position
`crates/benilla-assets/src/shaders/enhanced_water.wgsl:542`, `:616` and `:619`.
The vertex stage now moves vertices up to about 0.8 yd horizontally. The fragment evaluates
`water_waves` bands 0–1 and `water_gerstner(p).w` (the whitecap fold) at the displaced
`in.world_position.xz` without inverting. Normals and High whitecaps therefore sit up to about 0.8 yd
off their crests, and the band gradient misses the Jacobian that `waves.rs::swell` applies. The CPU
side is correct; the GPU is inconsistent with itself.
**Fix:** pass the pre-displacement xz as a varying and use it for those terms.

### 9. LOW — `m2_batches.rs` stage-1 texture-transform sentinel wraps to 0
`crates/benilla-formats/src/models/m2_batches.rs:495`:
`batch.texture_transform_combo_index.wrapping_add(1)`. For a batch with no animation (0xFFFF), this
wraps to index 0, so stage 1 picks up `texAnimLookup[0]`, which is stage 0's animation, and scrolls.
**Fix:** `checked_add(1)` → `None` means no animation.

### 10. LOW — Skybox sort bands collide on large models
`crates/benilla-world/src/skybox.rs:52-54`.
Main batches take `24 + i + 1`, the fog cone 56, and `SKYBOX_ORDER_CAP` clamps. A skybox with ≥ 32
batches ties with the fog cone, so the cone is no longer drawn last. A celestial model with ≥ 24
batches overlaps the main range.
**Fix:** reserve the top rung for the cone, and derive the bases from the batch count.

### 11. LOW — `skybox_def_by_path` is nondeterministic
`crates/benilla-formats/src/light/skybox.rs:131`: `HashMap::values().find(...)`. If two LightSkybox
rows name the same model with different flags, the row that wins changes from run to run. WMO and
ghost layers look up by path, while zone layers look up by id.
**Fix:** build a path→id map at load, keeping the lowest id.

### 12. LOW — Shared cached sky material may be mutated
`crates/benilla-world/src/skybox_anim.rs:371`: `SkyMatLane::register` writes `anim_slots.z` for an
affine-only batch into a material that `model_material` deduplicates. Two sky models that share texture
and order would then share one affine row.
**Fix:** clone the material first, as the stage-1 path already does.

### 13. LOW — Captures read player cvars when `BENILLA_HOME` is set
`crates/benilla-app/src/cvars.rs:2578-2582` (new in this branch). `config_read_path` reads
`$BENILLA_HOME/config.toml` under `WOW_CAPTURE`. `BENILLA_HOME` is also the general local-state
override (`local_state.rs`, `scripts/cine.sh`), so a capture in such a shell silently loads that
player's settings into a baseline.
**Fix:** use a dedicated opt-in variable (e.g. `WOW_CAPTURE_CVARS=<file>`).

### 14. LOW — An env override persists `graphicsQuality = "Custom"` forever
`seed_graphics_preset` skips session-owned rows. `derive_graphics_quality` still counts them, so under
e.g. `WOW_FARCLIP=500` the label derives "Custom" and gets persisted. The next normal boot never seeds
again, and the player shows Custom with `farclip` 350.
**Fix:** skip session-owned rows when deriving, or treat the label as session-owned while any governed
row is.

### 15. LOW — `/console` help strings broken by lost line continuations
`crates/benilla-app/src/cvars.rs:958` (`skyQuality`: a literal `\n` followed by 9 spaces), `:987`
(`zoneSkyboxes`: 10 spaces mid-sentence) and `~:1142` (`torchTerrainShadows`).
**Fix:** restore the `\` continuations.

### 16. LOW — `WOW_FOLIAGE_WIND` is documented as capture-only but is not gated
`crates/benilla-app/src/monkey_gfx.rs:38` and `crates/benilla-world/src/wind/mod.rs:82`. It overrides
the saved `foliageWind` in a normal player run, so the options row and the renderer disagree.
**Fix:** gate it like `capture_daylight` (dev affordances plus `WOW_CAPTURE`).

### 17. PERF — Per-frame allocations and uploads
- `volumetric_fog.rs` `update_fog`: `lampFog ≥ 1` with `volumetricFog = 0` inserts `FogView` every
  frame, including by day with 0 lamps. The result is a full-screen identity pass, a depth bind and a
  ping-pong flip. **Fix:** skip it when `tier == 0 && lamps.count == 0`.
- A new GPU uniform buffer is created per view per frame in `FogNode`, `ssao.rs:400`,
  `post/sun_shafts.rs:273` and `post/grading.rs:370`. **Fix:** use
  `UniformComponentPlugin`/`DynamicUniformIndex`, as bloom does.
- `skybox.rs` `animate_skyboxes` (`~:787-801`) clones path `String`s into a `HashMap`, allocates a pose
  `Vec`, and rebuilds and re-uploads each skinned skybox mesh every frame even when the pose is
  unchanged. `resolve_camera_skybox` also allocates normalised path strings per frame. At Enhanced, the
  sky clock is written into the sky material every frame, which contradicts the "idle frame writes
  nothing" comment.

### Hardening (not bugs today)
- `enhanced_water.wgsl:56,73`: `MONKEY_ROW = 533` and `SHELTER_ROW = 549`, and the
  `array<u32, 16384>` length in terrain, wow_model and static_gx, are hard-coded. No test ties them to
  `LIGHT_HEADER_ROWS`, `MAX_POINT_LIGHTS` or `shelter::GRID`. Add an `include_str!` pin test.
- `enhanced_water.wgsl:75` reads packed shelter `u32`s through an f32 view plus `bitcast`. A fully
  covered cell is a NaN bit pattern, which is safe on Vulkan, Metal and DX today but not guaranteed.
- `global_light.rs` comments (lines 126, 143, 481, …) still say 8528 B.
- `OptionsFrame.xml:2849` hard-codes the `177, 1497, 60` farclip slider with no test tying it to
  `view::FARCLIP_MAX`.
- `graphics_rows_pass_their_observers_unclamped` drives only `farclip`, not every governed row.
- Pre-existing, not from this branch: the ocean swell and shore break run on Bevy's 3600 s wrapped
  `globals.time` with rates that are not whole cycles per hour, so they jump hourly.

## Looks correct
- **Shared light buffer:** 21 + 512 + 16 rows = 8784 B, pinned by a test. Every `WowLight` mirror
  (terrain, wow_model, static_gx, liquid, enhanced_water, wow_effect, wdl) counts to 21 header rows.
  MonkeyFrame's field order matches Rust `pack`. Shelter sits at row 549 (65568 B) and prop probes at
  74352. The wow_model tail regions all moved by 65568 with 16-byte alignment kept. The buffers are
  allocated at `light_blob_bytes()`.
- **naga_oil:** no identifier ending in a digit in any `#define_import_path` module (the branch renamed
  `t1`/`c1`/`p0`/`d1`/`d2`/`shore_phase2`). Imports match their define paths, and all modules are
  registered.
- **Off/zero paths:** `fog_hook` Classic is the removed per-shader code verbatim, and the WDL reduction
  is exact. Sky and cloud tier 0 is unchanged (additions sit behind shader defs). Wet, wind and shelter
  return their inputs at zero. Classic water adds `vec3(0)`. Bloom, grade, shafts and AO being off
  never runs the node, and grading at strength 0 is skipped.
- **Clock wraps:** wind travel wraps at 4096 with whole cycles (75/285/52/32/167). Ripples wrap at
  1000 s with 800/650 cycles. The sky clock is f64 with whole-cycle twinkle.
- **Render graph:** fog → shafts → bloom → grade → FFX glow → Bloom → Tonemapping, with no cycles. AO
  runs after the opaque pass and before the water depth copy. The MSAA depth and colour variants are
  selected by sample count, and the formats specialise on the view target or HDR.
- **Bindings:** bind layouts match the WGSL declarations, and the uniform sizes are Bloom 16, Grade 16,
  Shafts 32, AO 48 and Fog 1136 B.
- **Lamp fog:** the analytic atan integral is correct, and `h ≥ 1.25` and `len ≥ 1e-4` guard the
  divisions.
- **Swell contract:** the CPU swell and Gerstner inversion match the shader (constants, axes and clock;
  Lipschitz 0.33 < 1). The LOD hysteresis and handle lifecycle do not leak. The shelter grid is
  bounds-checked, its packing matches the shader, and the ray budget is bounded.
- **Wetness:** dt-based, with no NaN at dt = 0. The MonkeyFrame writers are ordered
  (wind → wet → shelter → resolve → pack in PostUpdate), and fog edits only its own rows.
- **Presets:** every `GRAPHICS_PRESETS` and `LIGHTING_PRESETS` row has all 5 rungs. Classic is all off
  with `farclip` 350, High matches LIGHTING.md and is the seeded default, and all values are within
  clamps. Custom derivation round-trips (1e-4 tolerance) with no per-frame churn or oscillation, and
  Defaults lands on High.
- **UI strings:** the EN and RU tables have identical key sets. Every `BENILLA_ADVGFX.*` and tooltip key
  resolves, OptionsFrame.xml is well-formed, and dropdown values match each cvar's enum.
