// The B1 retained static pass (decision 1429; `static_gx/mod.rs`): the never-fade order-free
// static-world law of `wow_model.wgsl`, re-expressed over retained buffers — per-vertex
// texture-array layer + flag bits (+ a per-item record) where the entity path had one
// material per batch. Slice 1 = exterior ADT doodads; slice 2 adds WMO group geometry.
//
// **The law here MIRRORS `benilla_assets/shaders/wow_model.wgsl` and must track it** (the
// terrain.wgsl precedent: naga_oil cannot import functions that reference another module's
// bindings, so the shared pieces are copied with keep-in-sync notes at both ends). Mirrored
// pieces, in order: the `WowLight` per-frame PREFIX (the buffer is bigger — probe/palette
// regions ride the tail; binding a prefix-shaped struct against the same buffer is exactly
// what terrain.wgsl does), `wow_normalize`, `point_light_sum`, the exterior-doodad sun family
// (the Model2.bls order-2 lobe, 0803 — `min(I,1)` cap included, with its recorded caveat),
// the WMO surface lanes (slice 2: MOCV inside the clamp under GL_COLOR_MATERIAL, the
// INT/TRANS/EXT batch-class lanes, the WINDOW midpoint light, SIDN × night on lit lanes,
// ZERO point lights, the authored batch-order clip-z nudge, the interior fog triple), the
// interior-PROP probe lane (B4, decision 1433: the per-item slot rides the record table and
// the probe rows live in the TAIL of the same shared buffer — zero live point lights, the
// group-MOLR lobes are folded into the probe at spawn), the exterior-prop Matte fixed-1.0
// family (B4), and the step-5 fog. Deliberately ABSENT (this lane never carries them): the
// clutter/rig/fade/env-map/highlight/mat-anim lanes and per-instance tint (identity for
// statics).
//
// Population contract (enforced by the collector's divert — `static_gx::StaticGx::divert`):
// never-fade order-free non-animated statics (interior props are never-fade by law),
// Opaque/AlphaTest only, no env-map, no depth flags; cells additionally
// `ShadeSel::Lit`/`Shaded` only (the WMO lane never reads the selector; the prop lane adds
// `Matte`). Output alpha is pinned 1.0 — every draw here is opaque-intent by admission
// (wow_model.wgsl's bit-3 armor, as a constant).

#import bevy_pbr::{
    mesh_view_bindings::{lights, view},
    shadows,
}
// MONKEY (shadow hook): the realtime directional-shadow term (fetch + edge/night fade) lives here.
#import benilla::shadow_hook

// Group 0 is Bevy's standard mesh-view bind group (view matrices, directional-light records and
// the shadow textures the retained pass reads).
const SHADOW_SUN_FLOOR: f32 = 0.45;

// The shared global light's per-frame PREFIX (lighting::global_light; the full layout lives in
// wow_model.wgsl — keep field order in sync with BOTH).
struct WowLight {
    light_ambient: vec4<f32>,
    light_diffuse: vec4<f32>,
    light_sun: vec4<f32>,
    light_spec: vec4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
    sh_c10_r: vec4<f32>,
    sh_c10_g: vec4<f32>,
    sh_c10_b: vec4<f32>,
    sh_c13_r: vec4<f32>,
    sh_c13_g: vec4<f32>,
    sh_c13_b: vec4<f32>,
    sh_c16: vec4<f32>,
    _water: array<vec4<f32>, 4>,
    grade: vec4<f32>,
    wmo_fog_color: vec4<f32>,
    wmo_fog_params: vec4<f32>,
    point_count: vec4<f32>,
    points: array<vec4<f32>, 512>,
    // The interior-prop SH probe table (B4 — lighting::prop_probes, 7 rows per slot, 8192
    // slots; wow_model.wgsl owns the layout note). Sits immediately after the point table in
    // the shared buffer, so extending the mirrored prefix by one region is all it takes; the
    // rig/tint/mat-anim regions beyond it stay unmirrored (statics never read them).
    prop_probes: array<vec4<f32>, 57344>,
}
@group(2) @binding(0) var<storage, read> wow_light: WowLight;

// Per-cell state (static_gx/render.rs `cell_layout`).
struct GxCell {
    origin: vec4<f32>, // xyz = the bake's recentring origin (0974): world = vertex + origin
}
@group(1) @binding(0) var<uniform> cell: GxCell;
// The per-ITEM record table, indexed by the vertex word's low bits:
// x = texture-array layer, y = the WMO authored batch order (the clip-z nudge; 0 on cells —
// an exact no-op), z = the MOMT SIDN colour packed r|g<<8|b<<16 (gamma bytes), w = flags:
// bit 0 the exile kill bit (B2), bits 1..=13 the interior-prop probe slot (B4), bit 14 the
// per-frame INTERIOR FOG lane (1787 — the client's per-group `[0xca7f00]`).
@group(1) @binding(1) var<storage, read> recs: array<vec4<u32>>;
@group(1) @binding(2) var tex_array: texture_2d_array<f32>;
// Repeat and clamp variants of the BLP model-albedo sampler (linear tri-filtered, aniso 8 —
// exact parity with the entity path's per-image samplers; render.rs owns the rationale).
@group(1) @binding(3) var samp_repeat: sampler;
@group(1) @binding(4) var samp_clamp: sampler;

// Vertex word bits — keep in sync with static_gx/mod.rs WORD_*.
const WORD_WRAP_X: u32 = 65536u;    // 1 << 16
const WORD_WRAP_Y: u32 = 131072u;   // 1 << 17
const WORD_UNLIT: u32 = 262144u;    // 1 << 18
const WORD_FOG_OFF: u32 = 524288u;  // 1 << 19
const WORD_SHADE_LIT: u32 = 1048576u; // 1 << 20
const WORD_TEXTURED: u32 = 2097152u;  // 1 << 21
// The WMO lane (slice 2) — the entity path's per-material facts as bits:
const WORD_WMO: u32 = 4194304u;        // 1 << 22 — model_flags.x (the WMO surface laws)
const WORD_INTERIOR: u32 = 8388608u;   // 1 << 23 — model_flags.z (interior group)
const WORD_CLASS_INT: u32 = 16777216u; // 1 << 24 — tint.w == 1 (INT batch)
const WORD_CLASS_TRANS: u32 = 33554432u; // 1 << 25 — tint.w == 2 (TRANS batch)
const WORD_WINDOW: u32 = 67108864u;    // 1 << 26 — sidn.w (WINDOW midpoint light)
const WORD_HAS_VC: u32 = 134217728u;   // 1 << 27 — the batch AUTHORED vertex colours
                                       //           (the entity shader's VERTEX_COLORS def)
// The prop lane (B4). INTERIOR without WMO = an interior M2 prop — the entity shader's own
// `interior_prop = flags.z && !flags.x` split: probe lighting, interior fog, zero live
// point lights.
const WORD_MATTE: u32 = 268435456u;    // 1 << 28 — exterior MODD prop: intensity FIXED 1.0

// Vanilla cutout ref (224/255) — wow_model.wgsl's VANILLA_ALPHA_KEY.
const VANILLA_ALPHA_KEY: f32 = 0.8784314;

// MONKEY (torch shadows Phase 1): group 3 — the interior torch depth map + the ≤4-entry fixture
// table. Declared (and sampled) ONLY under TORCH_SHADOWS, which the static_gx pipeline always sets;
// the wow_model.wgsl copy of interior_room_light has no group 3 and must not see this block.
#ifdef TORCH_SHADOWS
struct TorchTable {
    // .x = live fixture count (a vec4 so the 16-byte-aligned std140 layout matches the Rust
    // `TorchTableUniform`: count@0, positions@16, view_projs@80 — total 1616 bytes).
    count: vec4<u32>,
    positions: array<vec4<f32>, 4>,    // xyz = fixture world pos, w = range
    view_projs: array<mat4x4<f32>, 24>, // 6 cube faces per fixture: [fixture*6 + face]
}
@group(3) @binding(0) var torch_depth: texture_depth_2d_array;
@group(3) @binding(1) var torch_samp: sampler_comparison;
@group(3) @binding(2) var<uniform> torch_table: TorchTable;
// Reverse-Z depth bias: nudges the receiver's own depth up so a surface never shadows ITSELF (the
// striped acne). KEEP SMALL — reverse-Z compresses depth far from the torch, so a large constant
// bias there detaches every shadow (0.004 made the whole room read "lit"). 0.001 is the known-good
// value that gave the clean forge-cast floor shadows; the residual acne only showed in the
// worst-case interiorDebug 2 (min over all 4 maps), not the real per-fixture render.
const TORCH_BIAS: f32 = 0.001;
#endif

// MONKEY (Phase 5): the cube face that contains direction `d` (fixture → fragment) — the major
// axis, signed. The face order is the contract with `benilla_app::torch_shadow::cube_view_projs`:
// 0 +X, 1 −X, 2 +Y, 3 −Y, 4 +Z, 5 −Z. Each face is a 90°(+ε) frustum down that axis, so the face
// holding the major axis always contains the direction.
fn torch_face(d: vec3<f32>) -> u32 {
    let a = abs(d);
    if (a.x >= a.y && a.x >= a.z) {
        return select(1u, 0u, d.x > 0.0);
    }
    if (a.y >= a.z) {
        return select(3u, 2u, d.y > 0.0);
    }
    return select(5u, 4u, d.z > 0.0);
}

// This interior fixture's OWN cast shadow: correlate the `wow_light` fixture at `light_pos` to a
// promoted torch (position match within 1 yd), pick the cube face facing the fragment, and sample
// that layer. 1.0 (unshadowed) when no map matches, or when TORCH_SHADOWS is off (wow_model's copy
// of the caller never sets it).
fn torch_surface_shadow(light_pos: vec3<f32>, P: vec3<f32>) -> f32 {
#ifdef TORCH_SHADOWS
    for (var i = 0u; i < torch_table.count.x; i = i + 1u) {
        let fixture = torch_table.positions[i].xyz;
        if (distance(fixture, light_pos) < 1.0) {
            let layer = i * 6u + torch_face(P - fixture);
            return shadow_hook::torch_map_shadow(
                torch_table.view_projs[layer], i32(layer), P, torch_depth, torch_samp, TORCH_BIAS);
        }
    }
#endif
    return 1.0;
}

// MONKEY (torch debug, interiorDebug 2): the MIN raw depth-map shadow factor over EVERY promoted map
// (ignores the fixture-position match), so the map's actual content is visible as greyscale on the
// floor: all-WHITE = maps empty / projection misses the fragment; uniform GREY = self-shadow/bias;
// SHAPED dark regions = real occlusion is being captured (then the fix is correlation/placement).
fn torch_debug_factor(P: vec3<f32>) -> f32 {
#ifdef TORCH_SHADOWS
    var s = 1.0;
    for (var i = 0u; i < torch_table.count.x; i = i + 1u) {
        let layer = i * 6u + torch_face(P - torch_table.positions[i].xyz);
        s = min(s, shadow_hook::torch_map_shadow(
            torch_table.view_projs[layer], i32(layer), P, torch_depth, torch_samp, TORCH_BIAS));
    }
    return s;
#else
    return 1.0;
#endif
}

// ---- mirrored law (wow_model.wgsl) ----

fn wow_normalize(v: vec3<f32>) -> vec3<f32> {
    let l2 = dot(v, v);
    return select(vec3<f32>(0.0), normalize(v), l2 > 1e-12);
}

fn point_light_sum(P: vec3<f32>, N: vec3<f32>, anchor: vec3<f32>) -> vec3<f32> {
    let count = u32(wow_light.point_count.x);
    var sel = array<u32, 3>(0u, 0u, 0u);
    var sd = array<f32, 3>(1e30, 1e30, 1e30);
    for (var i = 0u; i < count; i = i + 1u) {
        let pos_range = wow_light.points[2u * i];
        let dv = pos_range.xyz - anchor;
        let d2 = dot(dv, dv);
        if (d2 > pos_range.w * pos_range.w) {
            continue;
        }
        if (d2 < sd[0]) {
            sd[2] = sd[1]; sel[2] = sel[1];
            sd[1] = sd[0]; sel[1] = sel[0];
            sd[0] = d2; sel[0] = i;
        } else if (d2 < sd[1]) {
            sd[2] = sd[1]; sel[2] = sel[1];
            sd[1] = d2; sel[1] = i;
        } else if (d2 < sd[2]) {
            sd[2] = d2; sel[2] = i;
        }
    }
    var sum = vec3<f32>(0.0);
    for (var s = 0u; s < 3u; s = s + 1u) {
        if (sd[s] > 9.9e29) {
            break;
        }
        let pos_range = wow_light.points[2u * sel[s]];
        let to_light = pos_range.xyz - P;
        let d = length(to_light);
        let atten = 1.0 / (0.7 * d + 0.03 * d * d);
        let nl = max(dot(N, to_light / max(d, 1e-4)), 0.0);
        sum += wow_light.points[2u * sel[s] + 1u].rgb * (atten * nl);
    }
    return sum;
}

// MONKEY (dynamic interiors): the per-FRAGMENT room light for interior WMO surfaces AND interior
// props — EVERY in-range light of the room-gated table, no nearest-3 selection (a per-fragment
// selection swap draws a seam where the 4th light drops out). Two terms per light:
//  · DIRECT — the reference's falloff × a WRAPPED Lambert, (N·L + wrap)/(1 + wrap): a hearth
//    sitting at floor level is horizontal to the floor around it, and a hard max(N·L, 0) lights
//    nothing there (the smithy only worked because its forges are raised).
//  · FILL — a normal-free bounce, colour × K_FILL × (1 − d/range)²: the "lamps everywhere" glow
//    Blizzard's bake carried, sourced from the room's own fixtures instead of a constant. The
//    colour is normalised for this term (the table commits RAW over-gamut colour × intensity),
//    so one hot forge doesn't wash a whole room.
// `torch_shadow` (the #2 lane, 1.0 when no torch is promoted) shades the direct term only —
// bounce is indirect. A base ambient keeps a fixture-less nook from going black. The knobs are
// LIVE cvars packed into `point_count.yzw` — `.y` base ambient, `.z` fill gain, `.w` exposure
// (the callers' multiplier on the whole budget before the rolloff) — so
// `/script SetCVar("interiorExposure", 2)` retunes a room without a rebuild. The fill is
// half-desaturated: bounce off wood and stone is not candle-orange.
const INTERIOR_WRAP: f32 = 0.5;
fn interior_room_light(P: vec3<f32>, N: vec3<f32>) -> vec3<f32> {
    let count = u32(wow_light.point_count.x);
    let k_fill = wow_light.point_count.z;
    var direct = vec3<f32>(0.0);
    var fill = vec3<f32>(0.0);
    for (var i = 0u; i < count; i = i + 1u) {
        let pos_range = wow_light.points[2u * i];
        let to_light = pos_range.xyz - P;
        let d2 = dot(to_light, to_light);
        if (d2 > pos_range.w * pos_range.w) {
            continue;
        }
        let d = sqrt(d2);
        let c = wow_light.points[2u * i + 1u].rgb;
        let atten = 1.0 / (0.7 * d + 0.03 * d2);
        let nl = max(
            (dot(N, to_light / max(d, 1e-4)) + INTERIOR_WRAP) / (1.0 + INTERIOR_WRAP),
            0.0,
        );
        // Normalised for BOTH terms: the table commits RAW over-gamut colour × intensity, and the
        // smithy's three forges (1.4, 0.87, 0.4) drove the direct term to a washed-out white while
        // the inn's unit candles sat where they should. The hue is what survives.
        let c_norm = c / max(1.0, max(c.r, max(c.g, c.b)));
        // Phase 1: this fixture's OWN cast shadow, sampled from its down-looking depth map
        // (`torch_surface_shadow` correlates the fixture to its promoted map by position; 1.0 when
        // it was not promoted). Per-fixture, so a pillar between the fragment and torch A darkens A's
        // term without touching torch B's — the occlusion that makes an interior read
        // lit-and-shadowed instead of flat. (`static_gx.wgsl` only; the wow_model copy of this
        // function keeps the 1.0 `torch_shadow_for` stub — it has no group 3.)
        let s = torch_surface_shadow(pos_range.xyz, P);
        direct += c_norm * (atten * nl) * s;
        let reach = 1.0 - d / max(pos_range.w, 1e-4);
        let c_fill = mix(c_norm, vec3<f32>(dot(c_norm, vec3<f32>(0.299, 0.587, 0.114))), 0.5);
        // The DOMINANT fixture's fill, not the SUM: an ambient room glow must not scale with the
        // candle count, or a dense room (the inn's ~10 fixtures vs the smithy's 3) piles fill up
        // until the rolloff saturates every surface to a flat white. `max` keeps the nearest/
        // brightest fixture's glow and leaves the direct term to carry the per-fixture relief.
        fill = max(fill, c_fill * (k_fill * reach * reach));
    }
    // Fill is INDIRECT bounce, so it is not shadowed; direct already carries each fixture's shadow.
    return direct + fill + vec3<f32>(wow_light.point_count.y);
}

// ---- the pass ----

struct GxVertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) word: u32,
    @location(4) anchor: vec3<f32>,
    // MOCV / the baked constant tint (white where the batch authors none — WORD_HAS_VC says
    // which; the entity path's ATTRIBUTE_COLOR, interpolated exactly like it).
    @location(5) color: vec4<f32>,
}

struct GxVsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) word: u32,
    @location(4) point_lit: vec3<f32>,
    @location(5) color: vec4<f32>,
}

@vertex
fn vertex(v: GxVertex) -> GxVsOut {
    var out: GxVsOut;
    // The exile kill bit (B2, 1431 — record column w, bit 0): a punched-out item's vertices
    // all collapse to one constant point, so every triangle is zero-area and the rasterizer
    // drops it — no fragments, no state change, one u32 write flipped it. Triangles never
    // span items (the bake is item-contiguous), so a partial collapse cannot exist.
    if ((recs[v.word & 0xffffu].w & 1u) != 0u) {
        out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        out.world_position = vec4<f32>(0.0);
        out.world_normal = vec3<f32>(0.0, 1.0, 0.0);
        out.uv = vec2<f32>(0.0);
        out.word = v.word;
        out.point_lit = vec3<f32>(0.0);
        out.color = vec4<f32>(1.0);
        return out;
    }
    // Camera-relative clip (0974/wow_model.wgsl): both addends are small — the recentred
    // vertex and (cell origin − camera).
    let p_cam = v.position + (cell.origin.xyz - view.world_position);
    let world = v.position + cell.origin.xyz;
    out.world_position = vec4<f32>(world, 1.0);
    let view_rot = mat3x3<f32>(
        view.view_from_world[0].xyz,
        view.view_from_world[1].xyz,
        view.view_from_world[2].xyz,
    );
    out.position = view.clip_from_view * vec4<f32>(view_rot * p_cam, 1.0);
    // WMO authored batch order (the per-item record's y; wow_model.wgsl's `m.sun_scale.y`
    // nudge verbatim): a later coplanar MOBA batch must WIN the reverse-Z GreaterEqual test
    // in any draw order — load-bearing here, where the (bucket, texture, group) sort erases
    // authored order by design. 0 on every cell item ⇒ ×1.0, an exact no-op.
    out.position.z *= 1.0 + f32(recs[v.word & 0xffffu].y) * 1.1920929e-7;
    out.world_normal = v.normal;
    out.uv = v.uv;
    out.word = v.word;
    out.color = v.color;
    // The ≤3-nearest FFP selection, anchored at the PLACEMENT origin exactly like the entity
    // path (the baked per-vertex anchor — 1429's parity note; the blob path coarsened this).
    // WMO surfaces take ZERO point lights — the entity path zeroes them in the vertex stage
    // (wow-re trace-forensics-abbey-interior-d3d §2: zero on every observed WMO surface) —
    // and so do interior M2 props (B4): their group-MOLR point lobes are folded into the
    // per-item SH probe at spawn, the entity path's own vertex-stage zeroing.
    if ((v.word & WORD_WMO) != 0u) {
        // WMO surfaces stay zero here, the reference's own vertex-stage zeroing. MONKEY (dynamic
        // interiors): interior groups light from the live torches PER-FRAGMENT in the fragment
        // stage (`interior_room_light`) — a per-vertex sum on a WMO's huge floor triangles is
        // Gouraud: straight-edged wedges, and a torch mid-triangle lights nothing.
        out.point_lit = vec3<f32>(0.0);
    } else {
        out.point_lit = point_light_sum(world, v.normal, v.anchor);
    }
    return out;
}

@fragment
fn fragment(in: GxVsOut) -> @location(0) vec4<f32> {
    // The hard farclip wall — per-pixel planar eye-Z, same plane as the entity path.
    let eye_z = -(view.view_from_world * vec4<f32>(in.world_position.xyz, 1.0)).z;
    if (wow_light.fog_params.w > 0.0 && eye_z > wow_light.fog_params.w) {
        discard;
    }
    // Sample: the item's array layer through the sampler matching its wrap flags. Both-repeat
    // and both-clamp ride real samplers (exact parity); the rare MIXED batch keeps the repeat
    // sampler plus the half-texel inset clamp on its clamped axis (0763's silhouette clamp,
    // per axis). Both samples are taken unconditionally — `textureSample` needs uniform
    // control flow for its derivatives — and the select is per fragment.
    var base = vec4<f32>(1.0);
    let wrap_x = (in.word & WORD_WRAP_X) != 0u;
    let wrap_y = (in.word & WORD_WRAP_Y) != 0u;
    if ((in.word & WORD_TEXTURED) != 0u) {
        let layer = i32(recs[in.word & 0xffffu].x);
        let dims = vec2<f32>(textureDimensions(tex_array).xy);
        let inset = 0.5 / dims;
        var uv_mixed = in.uv;
        if (!wrap_x) {
            uv_mixed.x = clamp(uv_mixed.x, inset.x, 1.0 - inset.x);
        }
        if (!wrap_y) {
            uv_mixed.y = clamp(uv_mixed.y, inset.y, 1.0 - inset.y);
        }
        // `view.mip_bias` (decision 1639) — the render-scale LOD compensation, 0.0 at native.
        // Carried here as well as in wow_model.wgsl because this lane draws the same statics
        // through its own sampler rather than through `pbr_input_from_standard_material`, which
        // applies the bias for free; a lane that skipped it would blur out of step with the
        // entity path drawing the identical art.
        let c_repeat = textureSampleBias(tex_array, samp_repeat, in.uv, layer, view.mip_bias);
        let c_clamp = textureSampleBias(tex_array, samp_clamp, in.uv, layer, view.mip_bias);
        let c_mixed = textureSampleBias(tex_array, samp_repeat, uv_mixed, layer, view.mip_bias);
        if (wrap_x && wrap_y) {
            base = c_repeat;
        } else if (!wrap_x && !wrap_y) {
            base = c_clamp;
        } else {
            base = c_mixed;
        }
    }
#ifdef GX_CUTOUT
    if (base.a < VANILLA_ALPHA_KEY) {
        discard;
    }
#endif
    // Lighting — the exterior-doodad lane verbatim (wow_model.wgsl steps 7+, 0803's lobe):
    // both faces light from the SAME submitted normal (the FFP never enables
    // GL_LIGHT_MODEL_TWO_SIDE). The entity shader's `select(-n, n, is_front)` exists to UNDO
    // bevy_pbr's double-sided back-face negation; this pipeline never negated, so the raw
    // baked normal IS the submitted normal and any select here would INTRODUCE the flip the
    // entity path removes (the first capture A/B lit every back-facing canopy card from the
    // wrong side — the felwood heatmap's bright clusters).
    let n_lit = wow_normalize(in.world_normal);
    let L = -normalize(wow_light.light_sun.xyz);
    let ndotl = max(dot(n_lit, L), 0.0);
    let lit_nl = clamp(
        wow_light.light_ambient.rgb + wow_light.light_diffuse.rgb * ndotl,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    // The retained path uses the same Bevy directional shadow map as terrain and entity models.
    // `static_gx` used to have no mesh-view shadow bindings, which made every Stormwind WMO act
    // as if shadows were disabled even though the caster map was populated.
    // MONKEY (shadow hook): EXTERIOR surfaces take the directional sun shadow (fetch + edge/night
    // fade, via `benilla::shadow_hook`) — now with the same fades as terrain + models. INTERIOR
    // batches (`WORD_INTERIOR`) take the point-light (torch) shadow instead (#2): no sun reaches a
    // sealed room, but a promoted torch does. Both are no-ops when their light source is absent.
    var world_shadow = 1.0;
    if ((in.word & WORD_INTERIOR) == 0u) {
        let view_z = (view.view_from_world * in.world_position).z;
        let cam_dist = distance(in.world_position.xyz, view.world_position.xyz);
        world_shadow = shadow_hook::realtime_shadow(
            in.world_position,
            n_lit,
            view_z,
            cam_dist,
            wow_light.wmo_fog_params.z,
            wow_light.fog_params.z,
        );
    } else {
        world_shadow = shadow_hook::torch_shadow(in.world_position, n_lit);
    }
    let shadow_term = mix(SHADOW_SUN_FLOOR, 1.0, world_shadow);
    // Keep ambient energy when the realtime map blocks the sun. The retained pass also carries
    // authored fixture/probe lighting, which must not be darkened by a directional shadow. The
    // `world_shadow < 0.999` arm keeps `worldShadows 0` (no shadow-mapped light) and a fully
    // sunlit fragment byte-identical: `ambient + (lit − ambient) × 1.0` is not guaranteed to
    // round back to `lit`.
    let lit_nl_shadowed = select(
        lit_nl,
        clamp(
            wow_light.light_ambient.rgb + (lit_nl - wow_light.light_ambient.rgb) * shadow_term,
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        ),
        world_shadow < 0.999,
    );
    // The order-2 SH basis products over the fragment normal — shared by the exterior
    // doodad lobe and the interior-prop probe lane (wow_model.wgsl computes them once too).
    let quad = vec4<f32>(n_lit.x * n_lit.y, n_lit.y * n_lit.z, n_lit.z * n_lit.z, n_lit.x * n_lit.z);
    let x2y2 = n_lit.x * n_lit.x - n_lit.y * n_lit.y;
    // ATTRIBUTE_COLOR, pre-folded into the sampled base exactly where bevy_pbr folds it
    // (`base_color *= in.color` before the texture multiply — commutative, so tex×vc is the
    // same bits). **A colour-less batch takes the CONSTANT 1.0, never the interpolated
    // white**: the entity path's fold literally does not exist without its VERTEX_COLORS
    // shader-def, and GPU interpolation of a constant attribute is NOT exact — a barycentric
    // sum lands at 1.0±ε, and `base × 0.99999994` re-rounds ~half of all textured pixels one
    // byte down. That was the ±1/255 film over 300k pixels of house-south with the WMO lane
    // OFF — the attribution sweep's cells-only leg is what separated it from the WMO bake's
    // own (f64-fixed) quantization film.
    let has_vc = (in.word & WORD_HAS_VC) != 0u;
    let vc = select(vec4<f32>(1.0), in.color, has_vc);
    let folded = select(base.rgb, base.rgb * vc.rgb, has_vc);
    var rgb: vec3<f32>;
    if ((in.word & WORD_WMO) != 0u) {
        // ---- the WMO surface lanes (slice 2) — wow_model.wgsl's is_wmo branch verbatim ----
        let interior = (in.word & WORD_INTERIOR) != 0u;
        let class_int = (in.word & WORD_CLASS_INT) != 0u;
        let class_trans = (in.word & WORD_CLASS_TRANS) != 0u;
        let trans_a = vc.a; // 1.0 where no MOCV authored — the entity path's default
        // WINDOW (MOMT 0x20) — interior drawer only: GL_LIGHT0 swapped to the brighter
        // Direct/Ambient midpoint pair, ambient +16/255 saturating (0x6d37e0).
        let window_mid = 0.5 * (wow_light.light_ambient.rgb + wow_light.light_diffuse.rgb);
        // The window's directional half rides the sun, so it takes the shadow term (×1.0 exact
        // when the caster is off — no identity guard needed on a pure multiply).
        let lit_window_shadowed = clamp(
            window_mid + vec3<f32>(16.0 / 255.0) + window_mid * ndotl * shadow_term,
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
        let lit_int_base = select(lit_nl_shadowed, lit_window_shadowed, (in.word & WORD_WINDOW) != 0u);
        // The interior BATCH-CLASS lanes (trace-forensics-abbey-interior-d3d §2): INT = unlit
        // (the bake IS the room's light), TRANS = the per-vertex MOCV-alpha lit↔bake lerp,
        // EXT = plain lit_nl. Exterior groups take lit_nl at sun-scale 1 (prog 198/VS 151 —
        // no terrain shade, no SH lobe).
        var lit_wmo_interior = vec3<f32>(1.0);
        if (class_trans) {
            lit_wmo_interior = mix(vec3<f32>(1.0), lit_int_base, trans_a);
        } else if (!class_int) {
            lit_wmo_interior = lit_int_base;
        }
        // EXTERIOR surfaces are the ones the realtime map can shadow (the fetch above is
        // gated on !interior, so `lit_nl_shadowed` collapses to `lit_nl` for interior
        // fragments anyway). Taking plain `lit_nl` here left every Stormwind street and
        // wall unshadowed while the caster map was fully populated.
        let lit_wmo = select(lit_nl_shadowed, lit_wmo_interior, interior);
        // SIDN night glow (MOMT 0x10): the authored emissive × the live night fraction, an
        // EMISSION term — inside the clamped sum, never MOCV-multiplied; dead on the unlit
        // INT lane, TRANS-weighted by the lit-pass alpha (wmo-interior-night-light §4).
        let rec = recs[in.word & 0xffffu];
        let sidn_rgb = vec3<f32>(
            f32(rec.z & 0xffu),
            f32((rec.z >> 8u) & 0xffu),
            f32((rec.z >> 16u) & 0xffu),
        ) / 255.0;
        var sidn_w = 1.0;
        if (interior) {
            if (class_trans) {
                sidn_w = trans_a;
            } else if (class_int) {
                sidn_w = 0.0;
            }
        }
        let sidn_e = sidn_rgb * (wow_light.grade.x * sidn_w);
        // GL_COLOR_MATERIAL: MOCV multiplies the lit terms INSIDE the clamp (point term is
        // zero on WMO surfaces, kept in the sum for exact mirroring), emission adds beside.
        let primary = clamp(
            vc.rgb * (lit_wmo + in.point_lit) + sidn_e,
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
        // The entity algebra EXACTLY: bevy pre-folds vc into base and the shader divides it
        // back out — (tex×vc)/max(vc, 1/255) is not tex in floats, so simplifying would
        // break the pixel-identity bar.
        let tex_rgb = folded / max(vc.rgb, vec3<f32>(1.0 / 255.0));
        rgb = tex_rgb * primary;
        if (interior && wow_light.wmo_fog_params.w > 0.5) {
            // MONKEY (dynamic interiors #A, `interiorLight` on): the WHOLE interior group — INT, TRANS and EXT batch
            // classes alike — lights from the LIVE room torches instead of the MOCV bake. Gating
            // this on `class_int` left every TRANS/EXT batch (most of a floor near a doorway) on
            // the baked `vc.rgb × lit` path above: the "old baked light/shadow still there".
            //
            // Per-FRAGMENT (`interior_room_light`, every room light): a per-vertex sum on a WMO's
            // huge floor triangles is Gouraud — straight-edged wedges, a torch mid-triangle lights
            // nothing — the same artefact shape as Blizzard's per-vertex bake. The torch shadow
            // (#2 lane, 1.0 when no torch is promoted) shades the torch term only.
            //
            // Daylight keeps the AUTHORED batch-class weight, so a doorway still takes the sun:
            // INT = none, TRANS = the per-vertex MOCV alpha lerp, EXT = full `lit_int_base`.
            // MOCV.rgb (the baked light + shadow) is what gets dropped; its alpha is the exterior
            // blend weight, not a shadow, and stays.
            //
            // Soft rolloff instead of a hard clamp: `1 − exp(−x)` maps the ILLUMINATION into [0,1)
            // with derivative 1 at 0 (dim nooks untouched) and a smooth asymptote — a bright forge
            // never draws the iso-brightness contour a hard `clamp` did. `tex × illum` can't clip.
            // SIDN rides inside as the emission it is. Ambient 0.15 fills the dark (cvar/slider
            // once the look is confirmed).
            let room = interior_room_light(in.world_position.xyz, n_lit);
            var day_w = 0.0;
            if (class_trans) {
                day_w = trans_a;
            } else if (!class_int) {
                day_w = 1.0;
            }
            let illum = vec3<f32>(1.0) - exp(-(room + sidn_e) * wow_light.point_count.w);
            // BLEND toward the reference's own lit result (`rgb` above — MOCV × the exterior law)
            // by the authored batch-class weight; never ADD daylight on top of the room light,
            // which left the interior side of every threshold brighter than the exterior floor a
            // step away (the hard line at the smithy door). At the portal (weight 1) this IS the
            // exterior group's law, so the seam closes by construction, day or night; deep inside
            // (weight 0) it is pure room light. EXT-class batches (1) stay on the reference path.
            rgb = mix(tex_rgb * illum, rgb, day_w);
            let idbg = u32(max(wow_light.wmo_fog_params.w - 1.0, 0.0) + 0.5);
            if (idbg == 2u) {
                rgb = vec3<f32>(torch_debug_factor(in.world_position.xyz));
            } else {
                rgb = shadow_hook::interior_debug_override(idbg, rgb, in.world_position, n_lit);
            }
        }
    } else if ((in.word & WORD_INTERIOR) != 0u) {
        // ---- the interior M2-PROP lane (B4, decision 1433) — wow_model.wgsl's
        // interior-prop branch verbatim: the folded per-instance SH probe (MODD ambient +
        // the fixed-axis diffuse lobe + the owning group's MOLR lobes, folded ONCE at
        // spawn), evaluated over the fragment normal. The slot rides the record table
        // (w bits 1..14) where the entity path rode MeshTag bits 6..19. The soft SH wrap
        // is the reference's authored response — deliberately NOT a hard max(N·L,0).
        // point_lit is zero (vertex stage); inst_tint identity, no highlight, SIDN zero on
        // every M2 batch — the combine collapses to folded × clamp(probe eval).
        let probe = 7u * ((recs[in.word & 0xffffu].w >> 1u) & 0x1fffu);
        let n1 = vec4<f32>(n_lit, 1.0);
        let lit_prop = clamp(
            vec3<f32>(
                dot(wow_light.prop_probes[probe + 0u], n1)
                    + dot(wow_light.prop_probes[probe + 3u], quad)
                    + wow_light.prop_probes[probe + 6u].x * x2y2,
                dot(wow_light.prop_probes[probe + 1u], n1)
                    + dot(wow_light.prop_probes[probe + 4u], quad)
                    + wow_light.prop_probes[probe + 6u].y * x2y2,
                dot(wow_light.prop_probes[probe + 2u], n1)
                    + dot(wow_light.prop_probes[probe + 5u], quad)
                    + wow_light.prop_probes[probe + 6u].z * x2y2,
            ),
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
        let primary = clamp(lit_prop + in.point_lit, vec3<f32>(0.0), vec3<f32>(1.0));
        rgb = folded * primary;
        // MONKEY (dynamic interiors #A): interior props take the SAME live room light as the
        // surfaces around them (`interior_room_light`) instead of the baked probe — on the probe a
        // tavern's tables and barrels stayed bake-bright inside a torch-lit room ("the .m2s ignore
        // the light"). Same soft rolloff as the surface lane. UNCONDITIONAL for the prove-out; the
        // probe path above is the faithful baked result the cvar will restore.
        if (wow_light.wmo_fog_params.w > 0.5) {
            let room = interior_room_light(in.world_position.xyz, n_lit);
            rgb = folded * (vec3<f32>(1.0) - exp(-room * wow_light.point_count.w));
            let idbg = u32(max(wow_light.wmo_fog_params.w - 1.0, 0.0) + 0.5);
            if (idbg == 2u) {
                rgb = vec3<f32>(torch_debug_factor(in.world_position.xyz));
            } else {
                rgb = shadow_hook::interior_debug_override(idbg, rgb, in.world_position, n_lit);
            }
        }
    } else {
        // ---- the exterior ADT-doodad lane (slice 1) + the exterior MODD-prop family (B4) --
        // The intensity family: statics are `mat_shade` only (no per-instance ramp byte —
        // that is an entity concern): lit ground ⇒ mix(2.5,0.5,0)=2.5, MCSH-shadowed ⇒ 0.5,
        // and the Matte mid-band (an exterior WMO MODD prop) FIXED 1.0 — the 2.5 site is one
        // a MODD prop never reaches (§8b); then the recorded `min(I,1)` cap (unfaithful,
        // kept in exact sync — 0803 §3; today it makes Matte and Lit read identically, and
        // the distinct bit exists so lifting the cap cannot silently split this lane).
        let shade_t = select(1.0, 0.0, (in.word & WORD_SHADE_LIT) != 0u);
        let intensity = min(
            select(mix(2.5, 0.5, shade_t), 1.0, (in.word & WORD_MATTE) != 0u),
            1.0,
        );
        let sun_dc = wow_light.grade.yzw * intensity;
        let sun_lobe = vec3<f32>(
            wow_light.sh_c10_r.w + sun_dc.x
                + intensity
                    * (dot(wow_light.sh_c10_r.xyz, n_lit) + dot(wow_light.sh_c13_r, quad)
                        + wow_light.sh_c16.x * x2y2),
            wow_light.sh_c10_g.w + sun_dc.y
                + intensity
                    * (dot(wow_light.sh_c10_g.xyz, n_lit) + dot(wow_light.sh_c13_g, quad)
                        + wow_light.sh_c16.y * x2y2),
            wow_light.sh_c10_b.w + sun_dc.z
                + intensity
                    * (dot(wow_light.sh_c10_b.xyz, n_lit) + dot(wow_light.sh_c13_b, quad)
                        + wow_light.sh_c16.z * x2y2),
        );
        // The SH lobe's own ambient rows (the .w column) are the doodad lane's ambient share:
        // subtract-and-restore around the shadow term so only the directional part of the lobe
        // dims. Guarded like `lit_nl_shadowed` so the no-shadow state stays byte-identical.
        let sun_ambient = vec3<f32>(
            wow_light.sh_c10_r.w,
            wow_light.sh_c10_g.w,
            wow_light.sh_c10_b.w,
        );
        let lit_doodad_plain = clamp(sun_lobe, vec3<f32>(0.0), vec3<f32>(1.0));
        let lit_doodad = select(
            lit_doodad_plain,
            clamp(
                sun_ambient + (sun_lobe - sun_ambient) * shadow_term,
                vec3<f32>(0.0),
                vec3<f32>(1.0),
            ),
            world_shadow < 0.999,
        );
        // Sun disabled (light_sun.w) falls back to the FFP matte, like the entity path.
        let lit = select(lit_nl_shadowed, lit_doodad, wow_light.light_sun.w > 0.5);
        // FFP combine: the light sum saturates FIRST, the texture (× the baked constant
        // tint, when authored) modulates the clamped result. Statics: inst_tint identity,
        // no highlight, no SIDN.
        let primary = clamp(lit + in.point_lit, vec3<f32>(0.0), vec3<f32>(1.0));
        rgb = folded * primary;
    }
    // Unlit fullbright (M2 UNLIT 0x01 / WMO UNLIT on an exterior-group batch): the albedo —
    // texture × the folded vertex colour — with lighting off; no emission terms reach it.
    if ((in.word & WORD_UNLIT) != 0u) {
        rgb = folded;
    }
    // Step-5 fog (planar eye-Z; policy: Scene, or Off via the render-flag bit — the
    // Black/White/Grey families belong to blends this lane excludes by admission). Interior
    // WMO surfaces fog with the INTERIOR triple — the room keeps its warm MFOG haze
    // (round-6 Q-I: one flag gates walls, pools and props alike).
    // The interior fog lane is the per-frame record bit, NOT the baked `WORD_INTERIOR`: the
    // client pushes the interior triple per GROUP under `[0xca7f00]` (`0x6b5190` for surfaces,
    // `0x6b62e0` for the group's doodads — wow-re round-6 Q-I), and the flood decides it each
    // frame (decision 1787, `wow_model.wgsl` carries the same law on its tag bit 30). Keyed on
    // the batch's static interior flag instead, every true-interior group of a building wore the
    // building's MFOG the moment the camera stood anywhere in it — B335.
    var fog_color = wow_light.fog_color;
    var fog_span = wow_light.fog_params.xy;
    if ((recs[in.word & 0xffffu].w & 16384u) != 0u) {
        fog_color = wow_light.wmo_fog_color;
        fog_span = wow_light.wmo_fog_params.xy;
    }
    if (fog_color.w > 0.5 && (in.word & WORD_FOG_OFF) == 0u) {
        let denom = max(fog_span.y - fog_span.x, 0.001);
        let factor = clamp((fog_span.y - eye_z) / denom, 0.0, 1.0);
        rgb = mix(fog_color.xyz, rgb, factor);
    }
    // Raw gamma out (0161's lane); alpha pinned 1.0 — opaque-intent by admission.
    return vec4<f32>(rgb, 1.0);
}
