//! Corpus scans over **what lights a model** — the population instruments for the lighting
//! lanes, on both sides of the WMO/M2 boundary.
//!
//! `darkpropscan` asks which placed WMO props the interior lane commits as literal black
//! (decision 0969's census), `m2lightscan` which M2s author dynamic light blocks at all,
//! `m2firescan` which of the ones that DON'T would have a light SYNTHESISED from their flame
//! emitter (MONKEY, fire GO lights), and `shadeat` reads the terrain MCSH shadow bit that decides a
//! doodad's sun gain.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use benilla_formats::{Chain, M2Light};

use crate::model_key;

/// Sweep every WMO **root** (under `prefix`, if given) and list the placed MODD props whose
/// INTERIOR lighting lane commits **literal black**.
///
/// The interior lane's whole base light is the MODD entry's own baked colour field: ambient =
/// `cap96(colour)`, diffuse = `floor112(colour)` (`0x694e90` create → `0x6a77e0`; wow-re
/// `trace-forensics-abbey-interior-d3d` §1.1). The floor leg *raises* a dim colour to max 112 — but
/// it is a hue-preserving scale by `112/max`, so a colour of exactly `#000000` has nothing to raise
/// and both words come out zero. Such a prop is lit by nothing but its owning group's MOLR fixture
/// lights, and a group carrying none (or none within its authored attenuation disk) leaves it a
/// pure black silhouette.
///
/// The colour field is zero on ~3% of shipped MODDs — the baker leaves it unbaked for props it
/// treats as exterior — so what decides the symptom is the **class of the groups that reference the
/// prop**, and EXTERIOR WINS (decision 0969: the reference's def is per (MODD, placement) and
/// `0x695aa0` makes the exterior bit absorbing). A prop any exterior group's MODR names is therefore
/// sky-lit and never listed here; the `RESCUED` tally counts them, because taking the *first*
/// referrer instead is exactly what drew Booty Bay's entrance arch as a black silhouette.
pub fn darkpropscan(chain: &mut Chain, prefix: Option<&str>) -> Result<()> {
    let roots = super::wmo_roots(chain, prefix)?;

    let (mut roots_scanned, mut modds_total, mut zero_colour, mut black, mut dim, mut rescued) =
        (0u32, 0u32, 0u32, 0u32, 0u32, 0u32);
    let mut by_model: BTreeMap<String, u32> = BTreeMap::new();
    for root_path in roots {
        let Ok(bytes) = chain.read_file(&root_path) else {
            continue;
        };
        let Ok(root) = benilla_formats::parse_wmo_root(&bytes) else {
            continue;
        };
        roots_scanned += 1;
        modds_total += root.doodads().len() as u32;
        // Nothing to classify without a zero-colour MODD — skip the group reads entirely.
        if !root.doodads().iter().any(|d| d.color[..3] == [0, 0, 0]) {
            continue;
        }
        let lights = benilla_formats::parse_wmo_lights(&bytes);
        let stem = root_path
            .to_ascii_lowercase()
            .strip_suffix(".wmo")
            .unwrap_or(&root_path)
            .to_string();
        // MODD index -> referring groups, and group -> its MOLR light refs.
        let mut refs: BTreeMap<u16, Vec<u32>> = BTreeMap::new();
        let mut light_refs: BTreeMap<u32, Vec<u16>> = BTreeMap::new();
        for gi in 0..root.group_count() {
            let Ok(gbytes) = chain.read_file(&format!("{stem}_{gi:03}.wmo")) else {
                continue;
            };
            for r in benilla_formats::wmo_group_doodad_refs(&gbytes) {
                let e = refs.entry(r).or_default();
                if e.last() != Some(&gi) {
                    e.push(gi);
                }
            }
            light_refs.insert(gi, benilla_formats::wmo_group_light_refs(&gbytes));
        }

        let infos = root.group_infos();
        let mut printed_header = false;
        for (i, d) in root.doodads().iter().enumerate() {
            if d.color[..3] != [0, 0, 0] {
                continue;
            }
            zero_colour += 1;
            let Some(referrers) = refs.get(&(i as u16)) else {
                continue; // ORPHAN: no group names it — the exterior default
            };
            let owner = referrers[0];
            // EXTERIOR WINS over every interior referrer (decision 0969) — the MODD-colour lane is
            // for props referenced by interior groups ONLY.
            if !referrers
                .iter()
                .all(|g| infos.get(*g as usize).is_some_and(|gi| gi.interior))
            {
                if infos.get(owner as usize).is_some_and(|g| g.interior) {
                    rescued += 1;
                }
                continue;
            }
            // The owning group's MOLR omni lights, gated by their own attenuation disk measured
            // from the prop's origin (the spawn fold uses the loaded M2's bounds reference point;
            // the origin is within a model radius of it, so this is the census approximation).
            let in_range = light_refs
                .get(&owner)
                .map(|ls| {
                    ls.iter()
                        .filter_map(|&li| lights.get(li as usize))
                        .filter(|l| l.is_omni() && l.attenuation_end > l.attenuation_start)
                        .filter(|l| {
                            let dv = [
                                l.position[0] - d.position[0],
                                l.position[1] - d.position[1],
                                l.position[2] - d.position[2],
                            ];
                            dv.iter().map(|c| c * c).sum::<f32>().sqrt() < l.attenuation_end
                        })
                        .count()
                })
                .unwrap_or(0);
            if in_range == 0 {
                black += 1;
                *by_model.entry(model_key(&d.model)).or_default() += 1;
            } else {
                dim += 1;
            }
            if !printed_header {
                println!("{root_path}");
                printed_header = true;
            }
            let ref_cell = referrers
                .iter()
                .map(|g| format!("g{g}"))
                .collect::<Vec<_>>()
                .join(" ");
            println!(
                "  modd {i:>5}  {verdict:<5}  pos ({:>8.2}, {:>8.2}, {:>8.2})  molr {in_range} in range  refs(INT) {ref_cell}  {}",
                d.position[0],
                d.position[1],
                d.position[2],
                model_key(&d.model),
                verdict = if in_range == 0 { "BLACK" } else { "dim" },
            );
        }
    }

    eprintln!(
        "{roots_scanned} root(s), {modds_total} MODD(s): {zero_colour} carry colour #000000; \
         of those, interior-ONLY = {} — {black} BLACK (no MOLR light in range), {dim} dim \
         (a fixture light reaches them). RESCUED by exterior-wins: {rescued} (an interior group \
         names them first, an exterior group also names them — sky-lit, decision 0969).",
        black + dim,
    );
    if !by_model.is_empty() {
        eprintln!("BLACK props by model ({} distinct):", by_model.len());
        let mut rows: Vec<_> = by_model.into_iter().collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        for (m, n) in rows {
            eprintln!("  {n:>4}  {m}");
        }
    }
    Ok(())
}

/// How many rows of the closing colour tally print (the rest are counted, never silently dropped).
const TALLY_ROWS: usize = 20;

/// Cheap warm/cool/neutral hue classification of a `diffuse_color`, used only to eyeball the
/// colour-tally section of `m2lightscan`'s summary — the warm-torch family vs anything unusual.
fn hue_tag(r: f32, g: f32, b: f32) -> &'static str {
    if r >= g && r > b * 1.15 {
        "warm"
    } else if b > r && b >= g {
        "cool"
    } else {
        "neutral"
    }
}

/// Per-family tally for `m2lightscan`'s summary: how many models in this content family carry
/// lights, how many `type==1` point lights they author in total, how many of those are dark
/// (`visibility_off`), and a handful of example paths.
#[derive(Default)]
struct FamilyStats {
    models: u32,
    point_lights: u32,
    dark: u32,
    examples: Vec<String>,
}

/// Sweep every `.m2` (optionally under a path prefix) and report which models author M2 dynamic
/// LIGHT blocks — the population instrument for the mechanism (decision 0016 / wow-re
/// `system/models/scratch/m2-dynamic-lights.md`). Per model (only models with ≥1 light, printed
/// sorted by path): its `type==1` point-light count vs directional (`type==0`, ambient-feed, not
/// a discrete GL light) count, then per POINT light: bone, model-space position, `diffuse_color ×
/// diffuse_intensity` (raw colour, intensity, and the product), authored attenuation start/end,
/// and an `OFF` tag when [`M2Light::visibility_off`] — the one shape (a static `0` visibility
/// key) that keeps a light dark (§9.4). The closing summary is the real deliverable: totals, a
/// breakdown by top-level content family ([`super::family_of`]) — benilla only spawns these
/// lights for ADT-placed doodads and WMO props today, so this answers how much of the entity path
/// (creatures, held items, GameObjects) is actually missing them — and a cheap diffuse
/// colour×intensity tally ([`hue_tag`]).
pub fn m2lightscan(chain: &mut Chain, prefix: Option<&str>) -> Result<()> {
    let names = super::m2_names(chain, prefix)?;

    // Rounded `(r, g, b) × 100` (int-keyed to stay orderable) — a cheap grouping key for the
    // authored diffuse×intensity palette across point lights.
    type ColorKey = (i32, i32, i32);

    let (mut scanned, mut hits, mut total_point, mut total_dark) = (0u32, 0u32, 0u32, 0u32);
    let mut families: BTreeMap<String, FamilyStats> = BTreeMap::new();
    // key -> (hit count, one example model).
    let mut color_tally: BTreeMap<ColorKey, (u32, String)> = BTreeMap::new();
    let mut hit_models: Vec<(String, Vec<M2Light>)> = Vec::new();

    for name in names {
        let Ok(bytes) = chain.read_file(&name) else {
            continue;
        };
        scanned += 1;
        let lights = benilla_formats::parse_m2_lights(&bytes);
        if lights.is_empty() {
            continue;
        }
        hits += 1;
        let point_count = lights.iter().filter(|l| l.is_point()).count() as u32;
        let dark_count = lights
            .iter()
            .filter(|l| l.is_point() && l.visibility_off)
            .count() as u32;
        total_point += point_count;
        total_dark += dark_count;

        let fam = families.entry(super::family_of(&name)).or_default();
        fam.models += 1;
        fam.point_lights += point_count;
        fam.dark += dark_count;
        if fam.examples.len() < 8 {
            fam.examples.push(name.clone());
        }

        for l in lights.iter().filter(|l| l.is_point()) {
            let key = (
                (l.diffuse_color[0] * l.diffuse_intensity * 100.0).round() as i32,
                (l.diffuse_color[1] * l.diffuse_intensity * 100.0).round() as i32,
                (l.diffuse_color[2] * l.diffuse_intensity * 100.0).round() as i32,
            );
            color_tally
                .entry(key)
                .or_insert_with(|| (0, name.clone()))
                .0 += 1;
        }

        hit_models.push((name, lights));
    }

    hit_models.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, lights) in &hit_models {
        let point_count = lights.iter().filter(|l| l.is_point()).count();
        let dir_count = lights.len() - point_count;
        println!("{name}  {point_count} point, {dir_count} directional");
        for (i, l) in lights.iter().enumerate() {
            if !l.is_point() {
                continue;
            }
            let prod = [
                l.diffuse_color[0] * l.diffuse_intensity,
                l.diffuse_color[1] * l.diffuse_intensity,
                l.diffuse_color[2] * l.diffuse_intensity,
            ];
            println!(
                "    L{i}  bone {:>4}  pos ({:>9.3}, {:>9.3}, {:>9.3})  diffuse ({:.3}, {:.3}, {:.3}) x {:.3} = ({:.3}, {:.3}, {:.3})  atten [{:.2}, {:.2}]{}",
                l.bone,
                l.position[0], l.position[1], l.position[2],
                l.diffuse_color[0], l.diffuse_color[1], l.diffuse_color[2],
                l.diffuse_intensity,
                prod[0], prod[1], prod[2],
                l.attenuation_start, l.attenuation_end,
                if l.visibility_off { "  OFF" } else { "" },
            );
        }
    }

    println!();
    println!(
        "=== summary ===  {scanned} models scanned, {hits} with light blocks, {total_point} point lights, {total_dark} dark (visibility_off) point lights"
    );

    println!();
    println!("=== by content family ===");
    for (fam, stats) in &families {
        println!(
            "{fam:<32} {:>4} models  {:>4} point lights  {:>3} dark    e.g. {}",
            stats.models,
            stats.point_lights,
            stats.dark,
            stats.examples.join(" · ")
        );
    }

    println!();
    println!("=== diffuse colour x intensity tally (point lights, rounded to 0.01) ===");
    let mut ranked: Vec<(&ColorKey, &(u32, String))> = color_tally.iter().collect();
    ranked.sort_by_key(|(_, (count, _))| std::cmp::Reverse(*count));
    for (key, (count, example)) in ranked.iter().take(TALLY_ROWS) {
        let (r, g, b) = (
            key.0 as f32 / 100.0,
            key.1 as f32 / 100.0,
            key.2 as f32 / 100.0,
        );
        let tag = hue_tag(r, g, b);
        println!("{count:>4}x  ({r:.2}, {g:.2}, {b:.2})  {tag:<5}  e.g. {example}");
    }
    // Never let the top-20 read as "that's all of them".
    if let Some(rest) = ranked.len().checked_sub(TALLY_ROWS).filter(|n| *n > 0) {
        println!("      … and {rest} rarer colours (top {TALLY_ROWS} shown)");
    }

    Ok(())
}

/// MONKEY (fire GO lights) / MONKEY (lamp lights): sweep every `.m2` (optionally under a prefix)
/// and report which models would take a **synthesised** light, and BY WHICH ROUTE — the offline
/// audit of [`benilla_formats::fire_light`]'s two heuristics, which is the only way to see what a
/// rule applied to ~4000 world props actually does before shipping it.
///
/// The three routes, in the order the bake tries them:
/// - `flame` — a flame PARTICLE EMITTER. Per hit: the winning emitter's index/texture, the colour
///   read off its over-life ramp, the size bucket and the emitter's model-local position.
/// - `name` — the model's basename is lamp/lantern/chandelier/sconce vocabulary.
/// - `emissive` — no name hit, but an UNLIT + transparent + glow-textured render batch.
///
/// The lamp routes print `pos` as the emissive geoset's CENTROID and `+Zbase` as its height above
/// the model's own bounding-box floor — the number the whole lamp feature turns on, because a light
/// at the model origin sits at the foot of the pole. `BBOX` marks a hit that found no unlit
/// geometry and fell back to the bounding-box top centre.
///
/// Models that ALREADY author a casting point light are counted separately and skipped — the
/// authored block always wins, and the populations must never be conflated. The closing summary is
/// the deliverable: per-route totals, buckets, per-family counts, and a colour tally so an
/// over-broad texture key shows up as a flood of one hue.
pub fn m2firescan(chain: &mut Chain, prefix: Option<&str>) -> Result<()> {
    let names = super::m2_names(chain, prefix)?;
    let (mut scanned, mut authored, mut hits) = (0u32, 0u32, 0u32);
    let mut buckets: BTreeMap<&'static str, u32> = BTreeMap::new();
    // MONKEY (lamp lights): which of the three detection routes fired, per model.
    let mut routes: BTreeMap<&'static str, u32> = BTreeMap::new();
    let mut families: BTreeMap<String, u32> = BTreeMap::new();
    // Rounded `(r, g, b) × 100` -> (count, one example).
    let mut color_tally: BTreeMap<(i32, i32, i32), (u32, String)> = BTreeMap::new();
    let mut rows: Vec<String> = Vec::new();

    for name in names {
        let Ok(bytes) = chain.read_file(&name) else {
            continue;
        };
        scanned += 1;
        // The authored block wins entirely — same gate the asset bake applies.
        if benilla_formats::parse_m2_lights(&bytes).iter().any(|l| l.casts()) {
            authored += 1;
            continue;
        }
        let emitters = benilla_formats::parse_m2_particle_emitters(&bytes).unwrap_or_default();
        // MONKEY (lamp lights): route 1 is the flame emitter; routes 2/3 read the model's RENDER
        // BATCHES, so the submesh parse only runs where the flame route found nothing (which is
        // most of the corpus, but the parse is the same one the asset bake does anyway).
        let (route, color, intensity, bucket, detail) = match
            benilla_formats::synthesize_fire_light(&name, &emitters)
        {
            Some(fire) => {
                let e = &emitters[fire.emitter];
                let detail = format!(
                    "E{}  {:<44}  size {:.3} rate {:.1} strength {:.3}  pos ({:.2}, {:.2}, {:.2}) bone {}",
                    fire.emitter,
                    e.texture.as_deref().unwrap_or("-"),
                    benilla_formats::fire_light::peak_size(e),
                    e.timing.peak_rate(),
                    benilla_formats::fire_light::fire_strength(e),
                    e.position[0], e.position[1], e.position[2],
                    e.bone,
                );
                (
                    benilla_formats::LightRoute::Flame,
                    fire.color,
                    fire.intensity,
                    fire.bucket,
                    detail,
                )
            }
            None => {
                let Ok(subs) = benilla_formats::parse_m2_render_submeshes(&bytes, "", &[]) else {
                    continue;
                };
                let batches: Vec<benilla_formats::EmissiveBatch<'_>> =
                    subs.iter().map(benilla_formats::EmissiveBatch::from).collect();
                let bounds = benilla_formats::parse_m2_bounds(&bytes).ok();
                let bbox = bounds.as_ref().map(|b| (b.bbox_min, b.bbox_max));
                let Some(lamp) = benilla_formats::synthesize_lamp_light(&name, &batches, bbox)
                else {
                    continue;
                };
                // How high the light ended up above the model's own floor — the whole point of
                // reading the geoset instead of using the origin. A lamppost must read ~4-5 yd.
                let base = bbox.map_or(0.0, |(lo, _)| lo[2]);
                let detail = format!(
                    "{:<6}  {:<44}  pos ({:.2}, {:.2}, {:.2}) +Zbase {:.2} bone {}",
                    match lamp.batch {
                        Some(i) => format!("B{i}"),
                        None => "BBOX".to_string(),
                    },
                    lamp.batch
                        .and_then(|i| subs[i].texture.as_deref())
                        .unwrap_or("(bbox top centre)"),
                    lamp.position[0], lamp.position[1], lamp.position[2],
                    lamp.position[2] - base,
                    lamp.bone,
                );
                (lamp.route, lamp.color, lamp.intensity, lamp.bucket, detail)
            }
        };
        hits += 1;
        *routes.entry(route.label()).or_default() += 1;
        *buckets.entry(bucket).or_default() += 1;
        *families.entry(super::family_of(&name)).or_default() += 1;
        let key = (
            (color[0] * 100.0).round() as i32,
            (color[1] * 100.0).round() as i32,
            (color[2] * 100.0).round() as i32,
        );
        color_tally.entry(key).or_insert_with(|| (0, name.clone())).0 += 1;
        rows.push(format!(
            "{name}\n    [{:<8}] rgb ({:.3}, {:.3}, {:.3})  {:<11} x{:.2}  {detail}",
            route.label(),
            color[0], color[1], color[2],
            bucket, intensity,
        ));
    }

    rows.sort();
    for r in &rows {
        println!("{r}");
    }
    println!();
    println!(
        "=== summary ===  {scanned} models scanned, {authored} already author a casting point light (skipped), {hits} would SYNTHESISE one"
    );
    println!();
    println!("=== by detection route ===");
    for (r, n) in &routes {
        println!("{r:<10} {n:>4}");
    }
    println!();
    println!("=== by intensity bucket ===");
    for (b, n) in &buckets {
        println!("{b:<12} {n:>4}");
    }
    println!();
    println!("=== by content family ===");
    for (f, n) in &families {
        println!("{f:<32} {n:>4}");
    }
    println!();
    println!("=== derived colour tally (peak-normalised, rounded to 0.01) ===");
    let mut ranked: Vec<_> = color_tally.iter().collect();
    ranked.sort_by_key(|(_, (count, _))| std::cmp::Reverse(*count));
    for (key, (count, example)) in ranked.iter().take(TALLY_ROWS) {
        let (r, g, b) = (
            key.0 as f32 / 100.0,
            key.1 as f32 / 100.0,
            key.2 as f32 / 100.0,
        );
        println!(
            "{count:>4}x  ({r:.2}, {g:.2}, {b:.2})  {:<5}  e.g. {example}",
            hue_tag(r, g, b)
        );
    }
    if let Some(rest) = ranked.len().checked_sub(TALLY_ROWS).filter(|n| *n > 0) {
        println!("      … and {rest} rarer colours (top {TALLY_ROWS} shown)");
    }
    Ok(())
}

/// The terrain MCSH shadow bit at a world position + an ASCII texel neighborhood (`#` shadowed,
/// `.` lit, `?` off-tile/no-chunk). One MCSH texel is `TILE_SIZE/1024` ≈ 0.52 yd; the grid spans
/// ±8 texels so a doodad base sitting one texel from a shadow edge — the 2.5-vs-0.5 intensity
/// cliff — is visible at a glance.
pub fn shadeat(chain: &mut Chain, map: &str, x: f32, y: f32) -> Result<()> {
    let tiles = benilla_formats::load_tiles_around(chain, map, x, y, 0)
        .with_context(|| format!("loading the tile under ({x}, {y}) on {map}"))?;
    let Some((_, tile)) = tiles.first() else {
        anyhow::bail!("no tile exists under ({x}, {y}) on {map}");
    };
    let texel = benilla_formats::TILE_SIZE / 1024.0;
    let word = |s: Option<bool>| match s {
        Some(true) => "SHADOWED (doodad sun intensity 0.5)",
        Some(false) => "lit (doodad sun intensity 2.5)",
        None => "off-tile / no chunk",
    };
    println!(
        "MCSH at ({x:.2}, {y:.2}): {}",
        word(benilla_formats::mcsh_shadowed_at(&tile.chunks, [x, y, 0.0]))
    );
    println!(
        "neighborhood, texel {texel:.3} yd — rows +X (north) up, cols +Y (west) left; center marked:"
    );
    for dx in (-8i32..=8).rev() {
        let mut row = String::new();
        for dy in (-8i32..=8).rev() {
            let p = [x + dx as f32 * texel, y + dy as f32 * texel, 0.0];
            let mut c = match benilla_formats::mcsh_shadowed_at(&tile.chunks, p) {
                Some(true) => '#',
                Some(false) => '.',
                None => '?',
            };
            if dx == 0 && dy == 0 {
                c = if c == '#' { 'S' } else { 'O' };
            }
            row.push(c);
        }
        println!("{row}");
    }
    Ok(())
}

/// MONKEY (interior attenuation): dump ONE WMO root's **MOLT fixture table** — the authored
/// numbers the interior lane is calibrated against, which nothing could read before this.
///
/// Per fixture: type, the `useAtten` byte, the raw colour, the intensity, the product the packer
/// commits (`colour × intensity`, [`benilla_world::lighting::commit_raw`]'s input), the authored
/// attenuation **start/end** in yards (`+0x28`/`+0x2c` — see `read_wmo_light`), model-space
/// position, and the groups whose MOLR names it (its ROOMS: a fixture no group names keeps none
/// and stays ungated). The closing block is the number the flatness question actually needs: for
/// each omni fixture, the value of our fixed falloff `1/(0.7d + 0.03d²)` at its own authored end
/// — i.e. how much reach the curve still has where the artist said the light stops.
/// MONKEY (portal claims): everything ONE WMO root's room-claim rule needs, read off the chain in a
/// single group-file pass — the offline twin of what `benilla_world`'s spawner holds in the loaded
/// `WmoModel`. Both feed [`benilla_formats::room_claims`], so the claim set this tool prints IS the
/// claim set the shader will gate on.
struct RootRooms {
    infos: Vec<benilla_formats::WmoGroupInfo>,
    /// Per absolute group: its MOGP flags and its `(portal_ref_start, portal_ref_count)` slice.
    flags: Vec<u32>,
    slices: Vec<(u16, u16)>,
    portals: benilla_formats::WmoPortals,
    /// Per absolute group: the MOLR light indices it names.
    light_refs: Vec<Vec<u16>>,
    /// MODD index -> the groups whose MODR names it (the prop's rooms).
    doodad_groups: BTreeMap<u16, Vec<u16>>,
}

impl RootRooms {
    fn graph(&self) -> benilla_formats::PortalGraph<'_> {
        benilla_formats::PortalGraph {
            vertices: &self.portals.vertices,
            infos: &self.portals.infos,
            refs: &self.portals.refs,
            slices: &self.slices,
        }
    }

    /// The groups whose MOLR names light `i` — the claim rule's MOLR input.
    fn molr(&self, i: u16) -> Vec<u16> {
        self.light_refs
            .iter()
            .enumerate()
            .filter(|(_, refs)| refs.contains(&i))
            .map(|(g, _)| g as u16)
            .collect()
    }
}

fn root_rooms(
    chain: &mut Chain,
    root: &benilla_formats::WmoRoot,
    root_path: &str,
    bytes: &[u8],
) -> RootRooms {
    let stem = root_path
        .to_ascii_lowercase()
        .strip_suffix(".wmo")
        .unwrap_or(root_path)
        .to_string();
    let n = root.group_count();
    let (mut flags, mut slices, mut light_refs) = (Vec::new(), Vec::new(), Vec::new());
    let mut doodad_groups: BTreeMap<u16, Vec<u16>> = BTreeMap::new();
    for gi in 0..n {
        let gbytes = chain.read_file(&format!("{stem}_{gi:03}.wmo")).ok();
        let h = gbytes
            .as_deref()
            .and_then(benilla_formats::wmo_group_header);
        flags.push(h.as_ref().map_or(0, |h| h.flags));
        slices.push(h.map_or((0, 0), |h| (h.portal_ref_start, h.portal_ref_count)));
        light_refs.push(
            gbytes
                .as_deref()
                .map(benilla_formats::wmo_group_light_refs)
                .unwrap_or_default(),
        );
        for r in gbytes
            .as_deref()
            .map(benilla_formats::wmo_group_doodad_refs)
            .unwrap_or_default()
        {
            let e = doodad_groups.entry(r).or_default();
            if e.last() != Some(&(gi as u16)) {
                e.push(gi as u16);
            }
        }
    }
    RootRooms {
        infos: root.group_infos().to_vec(),
        flags,
        slices,
        portals: benilla_formats::parse_wmo_portals(bytes),
        light_refs,
        doodad_groups,
    }
}

/// One claim set as a compact cell: `g4e:in g7i:portal1 3.2/w5.0`. A trailing `*` marks a claim the
/// EXTERIOR batch lane will NOT honour (a district-scale shell — `room_claim::CLAIM_EXT_SHELL_YD`);
/// it still gates the fixture, it just cannot light a whole city block.
///
/// MONKEY (soft portal claims): a portal claim prints
/// `portal<hops> <distance>/w<fade radius>+<doorway slack>x<entry weight>` — the four numbers the
/// shader's weight is built from (`w = entry * (1 - smoothstep(0, radius, |P - door| - slack))`),
/// so the instrument can be read against a screenshot of the seam. A base claim prints no numbers:
/// it is HARD, weight 1 across the whole group.
fn claim_cell(claims: &[benilla_formats::Claim]) -> String {
    if claims.is_empty() {
        return "-  (UNGATED: no room claims it)".to_string();
    }
    claims
        .iter()
        .map(|c| {
            let (hop, d) = if c.how == benilla_formats::ClaimHow::Portal {
                (
                    c.hops.to_string(),
                    format!(
                        " {:.1}/w{:.1}+{:.1}x{:.2}",
                        c.distance, c.fade_radius, c.fade_slack, c.fade_entry,
                    ),
                )
            } else {
                (String::new(), String::new())
            };
            format!(
                "g{}{}:{}{hop}{d}{}",
                c.group,
                if c.interior { "i" } else { "e" },
                c.how.tag(),
                if c.ext_ok { "" } else { "*" },
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn wmolights(chain: &mut Chain, raw_path: &str) -> Result<()> {
    let root_path = raw_path.replace('/', "\\").to_ascii_lowercase();
    let bytes = chain
        .read_file(&root_path)
        .with_context(|| format!("reading WMO root '{root_path}'"))?;
    let root = benilla_formats::parse_wmo_root(&bytes)
        .with_context(|| format!("parsing WMO root '{root_path}'"))?;
    let lights = benilla_formats::parse_wmo_lights(&bytes);
    // MONKEY (portal claims): the group table + the portal graph, read once — the same inputs the
    // spawner feeds `room_claims`, so what prints below is what the room gate will enforce.
    let rr = root_rooms(chain, &root, &root_path, &bytes);
    let infos = root.group_infos();

    println!(
        "{root_path}  —  {} MOLT light(s), {} group(s), {} portal(s)",
        lights.len(),
        root.group_count(),
        rr.portals.infos.len(),
    );
    // The group table: the class the whole lighting split turns on. An `e` group INSIDE a building
    // (an inn's basement stairwell, a covered porch) is drawn by the EXTERIOR law and takes the
    // night sky unless an interior fixture claims it — that is bug B's second cause, and this is
    // where you read which groups are in that population.
    println!("=== groups (class, MOGP flags, MOGI box, portals, MOLR fixtures) ===");
    for gi in 0..root.group_count() as usize {
        let Some(g) = infos.get(gi) else { continue };
        let (_, pc) = rr.slices.get(gi).copied().unwrap_or((0, 0));
        println!(
            "  g{gi:<3} {}  flags {:#010x}  box ({:>8.2},{:>8.2},{:>7.2})..({:>8.2},{:>8.2},{:>7.2})  portals {pc:<3} molr {:?}",
            if g.interior { "INT" } else { "ext" },
            rr.flags.get(gi).copied().unwrap_or(0),
            g.bbox_min[0], g.bbox_min[1], g.bbox_min[2],
            g.bbox_max[0], g.bbox_max[1], g.bbox_max[2],
            rr.light_refs.get(gi).map(|v| v.len()).unwrap_or(0),
        );
    }
    println!();
    for (i, l) in lights.iter().enumerate() {
        let prod = [
            l.color[0] * l.intensity,
            l.color[1] * l.intensity,
            l.color[2] * l.intensity,
        ];
        let molr = rr.molr(i as u16);
        let reach = benilla_formats::room_claim::claim_reach(l.attenuation_end);
        let claims = benilla_formats::room_claims(&rr.infos, rr.graph(), l.position, reach, &molr);
        println!(
            "  L{i:<3} type {}{}  colour ({:.3}, {:.3}, {:.3}) x {:.3} = ({:.3}, {:.3}, {:.3})  atten [{:.3}, {:.3}]  R {reach:.1}  pos ({:>8.2},{:>8.2},{:>8.2})",
            l.light_type,
            if l.use_atten { " atten" } else { "      " },
            l.color[0], l.color[1], l.color[2],
            l.intensity,
            prod[0], prod[1], prod[2],
            l.attenuation_start, l.attenuation_end,
            l.position[0], l.position[1], l.position[2],
        );
        if l.is_omni() {
            println!("        claims {}", claim_cell(&claims));
        }
    }

    // What our FIXED curve is still worth at the artist's own cutoff — the flatness measurement.
    println!();
    println!("=== the fixed falloff 1/(0.7d + 0.03d²) at each omni fixture's authored end ===");
    let omni: Vec<&benilla_formats::WmoLight> = lights.iter().filter(|l| l.is_omni()).collect();
    for (i, l) in omni.iter().enumerate() {
        let d = l.attenuation_end.max(1e-3);
        println!(
            "  omni {i:<3} end {:>7.3} yd  atten(end) {:.4}  atten(end)/atten(1yd) {:.4}",
            l.attenuation_end,
            1.0 / (0.7 * d + 0.03 * d * d),
            (1.0 / (0.7 * d + 0.03 * d * d)) / (1.0 / 0.73),
        );
    }
    let ends: Vec<f32> = omni.iter().map(|l| l.attenuation_end).collect();
    if !ends.is_empty() {
        let mut sorted = ends.clone();
        sorted.sort_by(f32::total_cmp);
        println!(
            "  {} omni: end min {:.3} median {:.3} max {:.3}",
            sorted.len(),
            sorted[0],
            sorted[sorted.len() / 2],
            sorted[sorted.len() - 1],
        );
    }
    Ok(())
}

/// MONKEY (interior prop lights): what ONE prop model contributes as a light source, resolved once
/// per model path and memoised across the whole sweep (a city references the same lantern hundreds
/// of times).
#[derive(Clone)]
enum PropLightKind {
    /// The M2 authors a casting light block — every lane already spawns it, synthetic or not.
    Authored,
    /// No authored block, but `fire_light`'s flame or lamp route derives one: the model-space
    /// position of the FLAME (or the lamp glass) and its intensity.
    Synth { pos: [f32; 3], intensity: f32 },
    /// Neither — a chair, a barrel, a rug.
    None,
}

/// Rotate `v` by the MODD orientation quaternion `(x, y, z, w)` — `v + 2q×(q×v + wv)`. Written out
/// rather than pulled from a math crate because `benilla-formats` deliberately carries none, and
/// this is the only place the offline audit needs to place a prop.
fn quat_rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let u = [x, y, z];
    let t = cross(u, v);
    let t = [t[0] + w * v[0], t[1] + w * v[1], t[2] + w * v[2]];
    let t = cross(u, t);
    [v[0] + 2.0 * t[0], v[1] + 2.0 * t[1], v[2] + 2.0 * t[2]]
}

/// The prop-light rule resolved for one model path: the SAME chain `benilla_assets::m2` applies at
/// asset load (an authored block wins; else the flame route; else the lamp route), so a prop this
/// sweep counts as a light source is exactly one the runtime will spawn a light for.
fn prop_light_kind(chain: &mut Chain, key: &str) -> PropLightKind {
    let Ok(bytes) = chain.read_file(key) else {
        return PropLightKind::None; // a prop the client doesn't ship
    };
    if benilla_formats::parse_m2_lights(&bytes)
        .iter()
        .any(|l| l.casts())
    {
        return PropLightKind::Authored;
    }
    let emitters = benilla_formats::parse_m2_particle_emitters(&bytes).unwrap_or_default();
    if let Some(fire) = benilla_formats::synthesize_fire_light(key, &emitters) {
        return PropLightKind::Synth {
            pos: emitters[fire.emitter].position,
            intensity: fire.intensity,
        };
    }
    let Ok(subs) = benilla_formats::parse_m2_render_submeshes(&bytes, "", &[]) else {
        return PropLightKind::None;
    };
    let batches: Vec<benilla_formats::EmissiveBatch<'_>> = subs
        .iter()
        .map(benilla_formats::EmissiveBatch::from)
        .collect();
    let bounds = benilla_formats::parse_m2_bounds(&bytes).ok();
    let bbox = bounds.as_ref().map(|b| (b.bbox_min, b.bbox_max));
    match benilla_formats::synthesize_lamp_light(key, &batches, bbox) {
        Some(lamp) => PropLightKind::Synth {
            pos: lamp.position,
            intensity: lamp.intensity,
        },
        None => PropLightKind::None,
    }
}

/// MONKEY (wmo exterior points): how close an authored MOLT fixture has to stand to a prop's flame
/// (yd) for the synthesised light to be dropped as a duplicate. **Keep equal to
/// `benilla_world::terrain_stream::spawn::fx`'s `MODD_SYNTH_DEDUPE`** — the point of this sweep is
/// to predict what that dedupe does across the corpus.
const MODD_SYNTH_DEDUPE: f32 = 2.5;

/// One root's audit numbers.
struct LampRow {
    path: String,
    groups: usize,
    interior_groups: usize,
    molt: usize,
    props: usize,
    authored: usize,
    synth: usize,
    deduped: usize,
    /// Interior groups no MOLT fixture claims (the pre-fix darkness).
    dark_molt: usize,
    /// …and still none after the synthesised prop lights are admitted indoors.
    dark_after: usize,
}

/// MONKEY (interior prop lights): the corpus audit behind "make sure all WMOs with built-in light
/// sources have light working". For every WMO root in the chain: its MOLT omni fixtures, its MODD
/// props that WOULD synthesise a light (`fire_light`'s flame/lamp routes, run on the prop's own M2
/// exactly as the asset bake runs them), how many of those the 2.5 yd MOLT dedupe drops, and how
/// many INTERIOR groups end up with no light source claiming them at all — before and after the
/// prop lane is admitted indoors.
///
/// A group left in the `darkAFTER` column is either the artist's intent (a cellar, a closet, a
/// sealed shaft) or a detection miss; `detail` names them per root so the two can be told apart.
pub fn wmolamps(chain: &mut Chain, prefix: Option<&str>, detail: Option<&str>) -> Result<()> {
    let roots = super::wmo_roots(chain, prefix)?;
    // The buildings the owner actually walks through — always reported, wherever they rank.
    const LANDMARKS: [&str; 16] = [
        "buildings\\stormwind\\stormwind.wmo",
        "buildings\\goldshireinn\\goldshireinn.wmo",
        "buildings\\nsabbey\\nsabbey.wmo",
        "cities\\ironforge\\ironforge.wmo",
        "undercity\\undercity.wmo",
        "ogrimmar.wmo",
        "darnassis.wmo",
        "az_deadmines_a.wmo",
        "az_deadmines_b.wmo",
        "stormwindjail.wmo",
        "stormwindprison.wmo",
        "westfall_inn.wmo",
        "duskwood_inn.wmo",
        "redridge_inn.wmo",
        "human_farm",
        "monestary_cathedral.wmo",
    ];
    let mut cache: BTreeMap<String, PropLightKind> = BTreeMap::new();
    let mut rows: Vec<LampRow> = Vec::new();
    let mut n_roots = 0u32;
    for root_path in roots {
        let Ok(bytes) = chain.read_file(&root_path) else {
            continue;
        };
        // The listfile's own casing varies; every match below (landmarks, `detail`) is on the
        // lowercased path, which is also the casing `root_rooms` builds its group-file names in.
        let root_path = root_path.to_ascii_lowercase();
        let Ok(root) = benilla_formats::parse_wmo_root(&bytes) else {
            continue;
        };
        let lights = benilla_formats::parse_wmo_lights(&bytes);
        let omni: Vec<benilla_formats::WmoLight> =
            lights.iter().filter(|l| l.is_omni()).cloned().collect();
        // Nothing to say about a root with no interior geometry and no fixtures (a bridge, a fence).
        if root.group_infos().iter().all(|g| !g.interior) && omni.is_empty() {
            continue;
        }
        n_roots += 1;
        let rr = root_rooms(chain, &root, &root_path, &bytes);
        let n_groups = rr.infos.len();
        // Which groups anything claims — the MOLT fixtures first, then the props.
        let mut lit_by_molt = vec![false; n_groups];
        let mut lit_by_prop = vec![false; n_groups];
        for (i, l) in lights.iter().enumerate() {
            if !l.is_omni() {
                continue;
            }
            let molr = rr.molr(i as u16);
            let reach = benilla_formats::room_claim::claim_reach(l.attenuation_end);
            for c in benilla_formats::room_claims(&rr.infos, rr.graph(), l.position, reach, &molr) {
                if let Some(slot) = lit_by_molt.get_mut(usize::from(c.group)) {
                    *slot = true;
                }
            }
        }
        let (mut props, mut authored, mut synth, mut deduped) = (0usize, 0usize, 0usize, 0usize);
        for (di, d) in root.doodads().iter().enumerate() {
            if d.model.is_empty() {
                continue;
            }
            props += 1;
            let key = model_key(&d.model);
            if !cache.contains_key(&key) {
                let k = prop_light_kind(chain, &key);
                cache.insert(key.clone(), k);
            }
            let (pos, intensity) = match cache[&key] {
                PropLightKind::Authored => {
                    authored += 1;
                    continue; // an authored prop light is not this fix's population
                }
                PropLightKind::None => continue,
                PropLightKind::Synth { pos, intensity } => (pos, intensity),
            };
            // The flame in WMO model space: the prop's placement applied to the M2-local point.
            let r = quat_rotate(
                d.orientation,
                [pos[0] * d.scale, pos[1] * d.scale, pos[2] * d.scale],
            );
            let world = [
                d.position[0] + r[0],
                d.position[1] + r[1],
                d.position[2] + r[2],
            ];
            let dup = omni.iter().any(|l| {
                let dv = [
                    l.position[0] - world[0],
                    l.position[1] - world[1],
                    l.position[2] - world[2],
                ];
                dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]
                    < MODD_SYNTH_DEDUPE * MODD_SYNTH_DEDUPE
            });
            if dup {
                deduped += 1;
                continue;
            }
            synth += 1;
            let molr = rr
                .doodad_groups
                .get(&(di as u16))
                .cloned()
                .unwrap_or_default();
            let reach = benilla_formats::room_claim::claim_reach(
                benilla_formats::room_claim::m2_light_reach(intensity),
            );
            for c in benilla_formats::room_claims(&rr.infos, rr.graph(), world, reach, &molr) {
                if let Some(slot) = lit_by_prop.get_mut(usize::from(c.group)) {
                    *slot = true;
                }
            }
        }
        let interior: Vec<usize> = (0..n_groups).filter(|&g| rr.infos[g].interior).collect();
        let dark_molt = interior.iter().filter(|&&g| !lit_by_molt[g]).count();
        let dark: Vec<usize> = interior
            .iter()
            .copied()
            .filter(|&g| !lit_by_molt[g] && !lit_by_prop[g])
            .collect();
        if detail.is_some_and(|d| root_path.contains(&d.to_ascii_lowercase())) && !dark.is_empty() {
            let names: Vec<String> = dark.iter().map(|g| format!("g{g}")).collect();
            println!("{root_path}: interior groups still unlit: {}", names.join(" "));
        }
        rows.push(LampRow {
            path: root_path.clone(),
            groups: n_groups,
            interior_groups: interior.len(),
            molt: omni.len(),
            props,
            authored,
            synth,
            deduped,
            dark_molt,
            dark_after: dark.len(),
        });
    }

    let sum = |f: fn(&LampRow) -> usize| rows.iter().map(f).sum::<usize>();
    println!();
    println!("=== wmolamps: {n_roots} WMO root(s) with interior geometry or MOLT fixtures ===");
    println!(
        "  MOLT omni fixtures {}   MODD props {}   props with an AUTHORED light {}",
        sum(|r| r.molt),
        sum(|r| r.props),
        sum(|r| r.authored),
    );
    println!(
        "  props that SYNTHESISE a light {}   of which dropped by the {MODD_SYNTH_DEDUPE} yd MOLT dedupe {}   kept {}",
        sum(|r| r.synth) + sum(|r| r.deduped),
        sum(|r| r.deduped),
        sum(|r| r.synth),
    );
    println!(
        "  INTERIOR groups {}   claimed by no MOLT fixture {}   still unclaimed after the prop lights {}",
        sum(|r| r.interior_groups),
        sum(|r| r.dark_molt),
        sum(|r| r.dark_after),
    );
    let header = "  root                                                        grp  int  MOLT  props  synth  dedup  darkMOLT  darkAFTER";
    println!();
    println!("=== landmarks ===");
    println!("{header}");
    for row in rows
        .iter()
        .filter(|r| LANDMARKS.iter().any(|l| r.path.contains(l)))
    {
        print_lamp_row(row);
    }
    let mut ranked: Vec<&LampRow> = rows.iter().filter(|r| r.dark_molt > 0).collect();
    ranked.sort_by(|a, b| b.dark_molt.cmp(&a.dark_molt).then(b.molt.cmp(&a.molt)));
    println!();
    println!("=== the 25 roots with the most interior rooms no MOLT fixture reaches ===");
    println!("{header}");
    for row in ranked.into_iter().take(25) {
        print_lamp_row(row);
    }
    Ok(())
}

fn print_lamp_row(r: &LampRow) {
    let short = r.path.rsplit('\\').next().unwrap_or(&r.path);
    println!(
        "  {short:<58}  {:>3}  {:>3}  {:>4}  {:>5}  {:>5}  {:>5}  {:>8}  {:>9}",
        r.groups, r.interior_groups, r.molt, r.props, r.synth, r.deduped, r.dark_molt, r.dark_after,
    );
}
