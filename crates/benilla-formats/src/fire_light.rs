//! MONKEY (fire GO lights): **synthesising** a point light for a fire prop that authors none.
//!
//! The mechanism the reference commits — an M2 `type == 1` light block becoming a hardware
//! `GL_LIGHT` at the fixed `1/(0.7d + 0.03d²)` falloff (decision 0016) — is already parsed
//! ([`crate::M2Light`]) and already spawned by every lane. The problem is the CONTENT: a
//! client-wide sweep finds **17 of ~430** fire-ish GameObject display models authoring a light
//! block at all. `ElwynnCampfire`, `DwarvenBrazier01`, `Large/SmallFirePit01` do; every Ogre wall
//! torch, `HumanBrazierMagic`, `StormwindBrazier01`, `Forgebonfire`, every candle and every
//! `OrcBonFire*` does **not**. So a torch-lined hall is drawn by its flames and lit by nothing.
//!
//! The flame is nevertheless fully described by the model — in its PARTICLE EMITTER. The emitter
//! names a flame texture, blends additive, and carries the fire's own colour in its over-life ramp
//! ([`OverLife::color`]). This module reads exactly that and derives the light the artist did not
//! author: which emitter is the fire, what hue it burns, and how bright it should be.
//!
//! It lives in `benilla-formats` beside [`ParticleEmitterDef`] rather than in the asset layer that
//! *calls* it (`benilla_assets::m2`, where the [`crate::M2Light`] is manufactured and pushed into
//! the model's light list) for one reason: `benilla-extract`'s offline sweep — the only instrument
//! that can audit a heuristic across 430 models — is a `benilla-formats` binary and cannot reach
//! upward into the asset crate. One rule, one implementation, auditable offline and shipped live.
//!
//! **This is a synthesis, not a parse.** Nothing here is byte-verified against the reference,
//! because the reference does not do it — a vanilla client leaves those props dark. Everything a
//! synthesised light touches is therefore flagged (`ModelLight::synthetic`) so a consumer can
//! refuse it: the WMO MODD prop lane does, because a building's own MOLT fixtures already light
//! its wall torches and a second source would double them.

use crate::{OverLife, ParticleBlend, ParticleEmitterDef};

/// Texture-basename keys that mark an emitter as FIRE. Matched case-insensitively as substrings of
/// the emitter texture's basename (`ITEM\…\FLAMELICKSMALL.BLP` → `FLAMELICKSMALL`).
///
/// The list is the shipped flame/ember/candle art vocabulary, and its exclusions matter as much as
/// its entries: **`GLOW` is deliberately absent**. Half the runestone/crystal/portal corpus emits
/// `GLOW*.BLP`, and admitting it would light Dalaran-blue everywhere a quest crystal sits — a glow
/// card is a self-illuminated surface, not a fire. `SMOKE`, `SPARK` and `DUST` are absent for the
/// same reason from the other side: they ride real fires as SECOND emitters, and letting one win
/// would put the light at the smoke column's height in ash grey.
pub const FIRE_TEXTURE_KEYS: [&str; 8] = [
    "FLAMELICK", "BONFIRE", "BRAZIER", "CANDLE", "TORCH", "FLAME", "EMBER", "FIRE",
];

/// The fallback hue for a fire whose over-life ramp is all-but-white (an emitter that tints from
/// its texture rather than its ramp): plain warm orange. Peak-normalised like every derived colour.
pub const DEFAULT_WARM: [f32; 3] = [1.0, 0.55, 0.2];

/// Below this saturation (`max−min` of the linear RGB) an over-life key is "near-white" and carries
/// no hue worth taking. Every shipped flame's *characteristic* key clears it by a wide margin
/// (`OgreWallTorchpurple`'s purple 0.79, `HumanBrazierMagic`'s green 0.99, the plain torch's orange
/// 1.00), and the keys it rejects are exactly the white hot-cores and the white burn-outs that both
/// ends of a flame ramp are usually authored as.
const MIN_SATURATION: f32 = 0.15;

/// The smallest authored particle (yards, the over-life size ramp's peak) that counts as a FLAME
/// rather than a SPARK. Measured, not guessed: the corpus's real candles sit at 0.056–0.069
/// (`GeneralCandelabra01`, `CandelabraTallWall01`), while the ember/exhaust trails that share their
/// texture vocabulary — a zeppelin's `ember_grey`, a mount's `ember_offset`, the sparks on a
/// flaming helm, a fire elemental's secondary cinders — sit at 0.019–0.028 and are the whole of the
/// false-positive population. The floor is the one number separating the two, so it goes just under
/// the smallest real candle.
const MIN_FLAME_SIZE: f32 = 0.04;

/// One synthesised fire light: which emitter it was derived from, plus the light itself. The caller
/// turns this into an [`crate::M2Light`] at that emitter's model-local position/bone — **the
/// flame**, never the model origin, or a brazier's light sits inside its own bowl and the prop
/// self-shadows every surface it should be lighting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SyntheticFire {
    /// Index into the emitter slice handed in — the winning (strongest) fire emitter.
    pub emitter: usize,
    /// Peak-normalised linear RGB (see [`flame_color`]).
    pub color: [f32; 3],
    /// The light's `diffuse_intensity` (see [`fire_intensity`]).
    pub intensity: f32,
    /// The size bucket the intensity came from — for the offline audit's readout only.
    pub bucket: &'static str,
}

/// Is this model path a SPELL effect? Spell models emit flame for a second and vanish; giving them
/// a light would strobe the world on every Fireball and blow the 256-entry point table in a raid.
/// Matched on the path segment so `WORLD\…\SPELLSTONE.M2` can't trip it.
pub fn is_spell_path(path: &str) -> bool {
    let p = path.replace('/', "\\").to_ascii_uppercase();
    p.starts_with("SPELLS\\") || p.contains("\\SPELLS\\")
}

/// `Medium/Small/TallBrazierNoOmni01` — emitter-identical twins of the lit `…Brazier01` models
/// WITHOUT the authored M2 omni light. The first reading of the name ("the artist's dark twin,
/// honour it as a veto") was WRONG: the May-2026 server dump places 123 of these in Orgrimmar as
/// the city's PRIMARY street lighting (69 Medium / 32 Small / 22 Tall — `NoOmni` there means
/// "the engine's fixed-function omni is not wanted, the flame IS the light"), and vetoing them
/// left 106 of Orgrimmar's 129 rooms with no light source at all. So `NoOmni` is no longer a
/// veto anywhere: the flame route still demands a real additive flame emitter, which is what
/// separates a burning brazier from a cold one. Kept only as a name test for the offline scans
/// (MONKEY, Orgrimmar finding 2026-09-09).
pub fn is_no_omni(path: &str) -> bool {
    path.rsplit(['\\', '/'])
        .next()
        .unwrap_or(path)
        .to_ascii_uppercase()
        .contains("NOOMNI")
}

/// Does this texture path name flame art? Basename, extension stripped, case-insensitive
/// [`FIRE_TEXTURE_KEYS`] substring.
pub fn fire_texture(texture: Option<&str>) -> bool {
    let Some(t) = texture else { return false };
    let base = t.rsplit(['\\', '/']).next().unwrap_or(t);
    let stem = base.split('.').next().unwrap_or(base).to_ascii_uppercase();
    FIRE_TEXTURE_KEYS.iter().any(|k| stem.contains(k))
}

/// The three-part emitter gate: flame ART, flame BLEND, flame SIZE.
///
/// The blend arm is what keeps the smoke out. Every shipped flame is additive (`3` NoAlphaAdd / `4`
/// Add — the campfire is 4), and the same fire model's smoke column is `2` Alpha; a name gate alone
/// admits `FIRESMOKE.BLP`-class art and puts the light up in the plume. (Additive implies
/// `!def.lit` across nearly all of this corpus, so testing both would reject nothing while reading
/// as if it did.)
///
/// The size arm ([`MIN_FLAME_SIZE`]) is what keeps the SPARKS out, and it is the one the corpus
/// sweep added: every fire texture in the game is also used for ember trails, and a zeppelin's
/// exhaust cinders would otherwise light Booty Bay.
pub fn fire_emitter(def: &ParticleEmitterDef) -> bool {
    fire_texture(def.texture.as_deref())
        && def.blend == ParticleBlend::Add
        && peak_size(def) >= MIN_FLAME_SIZE
        && def.timing.peak_rate() > 0.0
}

/// The flame's colour from its over-life ramp: **the most saturated key**, peak-normalised to 1.0.
///
/// Not an average, and not key 0. A flame ramp is authored as *hue → hue → burnout*, and both the
/// mean and the endpoints wash toward the white the burnout ends on: averaging
/// `OgreWallTorchpurple`'s `(1,1,1) / (0.565,0.212,1.0) / (0.42,0,0.745)` gives a pale lilac, and
/// key 0 gives literal white. The most-saturated key is the one an artist chose to say "this fire
/// is purple", and taking it reproduces all three of the calibration models: purple stays purple,
/// `HumanBrazierMagic` comes out green, the plain torch comes out orange.
///
/// Peak-normalising (divide by the largest channel) separates HUE from BRIGHTNESS, which is
/// [`fire_intensity`]'s job — a dim ember and a roaring bonfire of the same hue must not differ in
/// colour, or the intensity bucket would be applied twice.
pub fn flame_color(over_life: &OverLife) -> [f32; 3] {
    let sat = |k: &[f32; 4]| {
        let (lo, hi) = k[..3]
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        hi - lo
    };
    let best = over_life
        .color
        .iter()
        .filter(|k| sat(k) >= MIN_SATURATION)
        .max_by(|a, b| sat(a).total_cmp(&sat(b)));
    let rgb = match best {
        Some(k) => [k[0], k[1], k[2]],
        None => return DEFAULT_WARM, // an all-white ramp tints from its texture; assume a fire
    };
    let peak = rgb[0].max(rgb[1]).max(rgb[2]);
    if peak <= 1e-4 {
        return DEFAULT_WARM; // a black ramp is not a colour; don't divide by it
    }
    [rgb[0] / peak, rgb[1] / peak, rgb[2] / peak]
}

/// The emitter's peak authored particle half-extent (the over-life size ramp's largest key, yards)
/// — the size discriminator [`fire_intensity`] buckets on.
pub fn peak_size(def: &ParticleEmitterDef) -> f32 {
    def.over_life.scale.iter().copied().fold(0.0, f32::max)
}

/// The fire's "how big is this flame" score: `peak particle size × ⁴√(rate)`.
///
/// Size is the honest signal — a candle authors ~0.1 yd particles, a wall torch ~0.3, a brazier
/// ~0.6, a bonfire ~1.5 — but size alone can't separate a fat, lazy two-particle-a-second ember bed
/// from a real fire. The rate enters under a FOURTH root deliberately: emission rates range over
/// two orders of magnitude across this corpus (2 → 200) and any stronger weighting lets a dense
/// spark shower outrank the flame it belongs to. Used both to pick the winning emitter and to
/// bucket its intensity.
pub fn fire_strength(def: &ParticleEmitterDef) -> f32 {
    peak_size(def) * def.timing.peak_rate().max(0.0).sqrt().sqrt()
}

/// Bucket a fire's [`fire_strength`] onto a light intensity, calibrated against the AUTHORED corpus
/// so a synthesised torch sits in the same range as `MediumBrazier01`/`ElwynnCampfire` rather than
/// out-blazing them (the authored fire family clusters at `diffuse_color (0.467,0.290,0.133)` with
/// intensities 1–3). Returns the intensity and the bucket's name for the offline readout.
///
/// The cut points come off the `m2firescan` sweep, chosen so the named landmarks fall where their
/// names say: `GeneralCandelabra01` 0.09 and `CandelabraTallWall01` 0.07 candle; `OgreWallTorch`
/// 1.06, `StormwindBrazier01` 0.82 and `HumanBrazierMagic` 1.30 in the middle two;
/// `Forgebonfire` 2.03 and `OrcBonFire` 3.25 at the top. Note that the middle pair is not cleanly
/// separable by any function of the record — the ogre WALL TORCH authors a bigger flame than the
/// Stormwind BRAZIER — which is exactly why the two buckets differ by 0.5 and not by more:
/// misfiling between them has to be cosmetically invisible, because it will happen.
pub fn fire_intensity(strength: f32) -> (f32, &'static str) {
    match strength {
        s if s < 0.25 => (0.6, "candle"),
        s if s < 0.70 => (1.5, "torch"),
        s if s < 1.60 => (2.0, "brazier"),
        _ => (3.0, "bonfire"),
    }
}

/// Derive **at most one** light for a model that authors no casting light block: pick the strongest
/// fire emitter ([`fire_strength`]), take its hue ([`flame_color`]) and its bucket
/// ([`fire_intensity`]). `None` when the model is a spell effect or has no fire emitter.
///
/// One light per model, never one per emitter. A campfire authors 3–5 emitters (core flame, licks,
/// embers, smoke) and every one of them that passed the gate would stack a full-strength light on
/// the same spot — five torches' worth of blaze from one campfire, and five of the 256 table slots.
pub fn synthesize_fire_light<'a>(
    path: &str,
    emitters: impl IntoIterator<Item = &'a ParticleEmitterDef>,
) -> Option<SyntheticFire> {
    if is_spell_path(path) {
        return None;
    }
    let (emitter, def, strength) = emitters
        .into_iter()
        .enumerate()
        .filter(|(_, d)| fire_emitter(d))
        .map(|(i, d)| (i, d, fire_strength(d)))
        // A hand-rolled max rather than `max_by`, which keeps the LAST maximum on a tie: the
        // FIRST-wins arm (`>=` on the incumbent) is what makes the pick stable when a model
        // authors two identical flames, and the ties are real (mirrored torch pairs).
        .fold(None, |best: Option<(usize, &ParticleEmitterDef, f32)>, cur| {
            match best {
                Some(b) if b.2 >= cur.2 => Some(b),
                _ => Some(cur),
            }
        })?;
    let (intensity, bucket) = fire_intensity(strength);
    Some(SyntheticFire {
        emitter,
        color: flame_color(&def.over_life),
        intensity,
        bucket,
    })
}

// ---------------------------------------------------------------------------------------------
// MONKEY (lamp lights): the SECOND detection route — an emissive GEOSET, not a particle emitter.
// ---------------------------------------------------------------------------------------------
//
// The flame route above reads a fire's PARTICLE EMITTER, and everything with a visible flame is
// caught by it. A LAMPPOST is not: `StormwindStreetlamp01`, `DuskwoodLamppost`,
// `GeneralHangingLantern01`, every `WestfallLampPost*` — none of them authors a single emitter.
// Their glow is authored as GEOMETRY: a small glass geoset carrying the M2 material render-flag
// `0x01` (UNLIT — "draw me at full texture brightness, ignore the sun"), usually with a second
// additive billboard corona card over it. So the whole street-lighting corpus — the Valley of
// Heroes, Stormwind Park, the Duskwood road — sailed straight past `fire_emitter` and lit nothing.
//
// The lamp route detects that shape. It is deliberately WEAKER evidence than a flame — an unlit
// batch says "this surface is self-illuminated", not "this object emits light" — so it is fenced
// three ways: by the model's NAME (a lamp/lantern/chandelier vocabulary), by the batch's TEXTURE
// (glow/lamp art), and by a content-family gate that keeps it out of characters, creatures, items
// and spell effects, where unlit additive geometry is the *norm* rather than a signal.
//
// The flame route always wins where both fire: it derives a real colour from the artist's own ramp,
// whereas this route can only pick a plausible one from a name (see [`lamp_color`]).

/// Model-basename keys that name a luminous prop, most specific first — the order is load-bearing,
/// because [`lamp_kind`] classifies on the FIRST match (`DuskwoodLamppost` must read as LAMPPOST,
/// not as the `LAMP` substring inside it; `CHANDELIER` before `CANDLE` so a chandelier is never
/// filed on the candle rung; and `CANDELABRA` before `CANDLE` — which costs nothing here because
/// `CANDELABRA` does not contain `CANDLE`, but the Scholomance corpus spells it both ways).
pub const LAMP_NAME_KEYS: [&str; 13] = [
    "CHANDELIER",
    "CANDELABRA",
    "LAMPPOST",
    "LAMP POST",
    "LIGHTPOST",
    "STREETLAMP",
    "STREETLIGHT",
    "LANTERN",
    "SCONCE",
    "LAMP",
    "CANDLE",
    "BRAZIER",
    "TORCH",
];

/// DIRECTORY words that mark a folder as a light-fixture folder. This is route (b)'s real gate.
///
/// The first cut of route (b) — "an unlit, transparent, glow-textured batch on any world model" —
/// fired on 149 models, and reading the list is what wrote this constant: blood-elf BANNERS,
/// Ghostlands TREES, PvP RUNES, Uldaman TELEPORT PADS, Booty Bay HOLOGRAMS, silithid EGGS, a
/// Karazhan owl STATUE. Self-illuminated geometry is simply how vanilla draws "this thing is
/// magic", and it says nothing at all about whether the thing casts light.
///
/// What *does* say so is where the artist filed the model. A prop under `…\Lamps\`, `…\Lanterns\`,
/// `…\Candles\`, `…\Sconces\` or `…\Lights\` was authored as a light FIXTURE, whatever it is named
/// (`Scholme_Candelabra`, `TS_LightPole`, `BFD_Wisp01`, `GnomeHazardLight01`). Route (b) is
/// therefore "a fixture folder, corroborated by unlit glow geometry" — which is what it always
/// meant, stated in the one place the corpus actually records it.
const LAMP_DIR_KEYS: [&str; 7] = [
    "LAMP",
    "LANTERN",
    "CANDLE",
    "SCONCE",
    "CHANDELIER",
    "LIGHTS",
    "TORCH",
];

/// Names that describe LIT AIR rather than a light source — god-rays, sun shafts, spotlight cones.
/// `World\NoDXT\…\VolumetricLights\LightShaftA` sits in a `…Lights\` folder and draws exactly the
/// unlit additive glow card route (b) looks for, but it is the *beam*, and its geoset centroid is
/// halfway down the shaft: a point light there lights the floor from mid-air, in a spot the artist
/// already painted bright. Eleven of them ship. Vetoed by name, on both routes.
const BEAM_KEYS: [&str; 3] = ["SHAFT", "BEAM", "SKYLIGHT"];

/// Texture-basename keys that mark a render batch's art as GLOW art. Wider than
/// [`FIRE_TEXTURE_KEYS`] by exactly the entries that list deliberately excludes — `GLOW`, `LIGHT`,
/// `FLARE` — because here they are read in a much narrower context: an UNLIT batch on a model that
/// already cleared the family gate. `Glow32.blp` is the shipped corona card on half the lampposts
/// in the game and the single most diagnostic texture in this route; on a runestone it would be a
/// false positive, which is what [`lamp_family_ok`] is for.
pub const GLOW_TEXTURE_KEYS: [&str; 9] = [
    "GLOW", "LIGHT", "LAMP", "LANTERN", "FLARE", "FIRE", "FLAME", "CANDLE", "TORCH",
];

/// The BROKEN veto. The corpus ships a dark twin of nearly every lamp — `DuskwoodLamppostBroken`,
/// `UldamanLampAndPostBusted01`, `DarnassusWreckedStreetLamp01`, `BE_lantern_busted_001`,
/// `UldamanLampFallen`, `WesternPlaguelandsLampostStraightOff` — and a level designer places one
/// exactly where a lit lamp is wrong (a ruined road, a sacked village). Several still carry the
/// unlit glass geoset they inherited from the whole model, so nothing but the name tells them
/// apart. Same logic as [`is_no_omni`]: it is the artist's own opt-out, and it is honoured.
const BROKEN_KEYS: [&str; 7] = [
    "BROKEN", "BUSTED", "WRECKED", "SMASHED", "DESTROY", "FALLEN", "UNLIT",
];

/// The warm-lamp default hue: an oil/candle lamp behind amber glass. Deliberately paler and less
/// orange than [`DEFAULT_WARM`] (the open-flame default) — a lamp's light has passed through a
/// glass shade, and a street lit at the flame's own `(1, 0.55, 0.2)` reads as a fire alarm.
pub const DEFAULT_LAMP: [f32; 3] = [1.0, 0.72, 0.40];

/// Which luminous-prop family a name lands in — the classification both [`lamp_color`] and
/// [`lamp_intensity`] read. `None` when no [`LAMP_NAME_KEYS`] entry matches (the model then reaches
/// the route only through its geometry, and is treated as a generic lamp).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LampKind {
    /// A hanging multi-candle fixture: a hall's whole light source.
    Chandelier,
    /// A single small candle / candelabra — the dimmest rung.
    Candle,
    /// A lamppost, street lamp, hanging lantern or wall sconce — the population this exists for.
    Lamp,
    /// An open-fire BRAZIER — the authored fire family's own rung.
    Brazier,
    /// An open-fire TORCH — the torch rung, the same one a lamp takes.
    Torch,
}

impl LampKind {
    /// Is this an OPEN-FIRE container (a brazier or a torch) rather than a fixture?
    ///
    /// The distinction exists for exactly one rule: [`lamp_position`]'s bounding-box fallback is
    /// refused to the fire kinds. A lamp whose glass isn't flagged unlit is still a lamp
    /// (`UldamanLamp`, `IronForgeHangingLantern01`, `AhnQirajSconce01` — all real fixtures the
    /// fallback rescues), but a brazier with neither a flame emitter nor one glowing polygon is a
    /// brazier with nothing burning in it: `SummerFest_Brazier_02`, `LordaeronBrazier01`,
    /// `UldamanBrazier01` and `IT_Brazier22` are cold props, and lighting them was the first cut of
    /// this rule's most visible error.
    pub fn is_fire(self) -> bool {
        matches!(self, LampKind::Brazier | LampKind::Torch)
    }
}

/// Which of the two synthesis routes produced a light — reported by the offline sweep, and the
/// first thing to read when a model glows that shouldn't.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightRoute {
    /// The model authors a flame PARTICLE EMITTER ([`synthesize_fire_light`]).
    Flame,
    /// The model's NAME says lamp/lantern/chandelier ([`synthesize_lamp_light`], route a).
    Name,
    /// No name hit — an UNLIT render batch carrying glow art gave it away (route b).
    Emissive,
}

impl LightRoute {
    /// Short label for the offline readout.
    pub fn label(self) -> &'static str {
        match self {
            LightRoute::Flame => "flame",
            LightRoute::Name => "name",
            LightRoute::Emissive => "emissive",
        }
    }
}

/// The model basename, extension stripped, uppercased — the string every name rule below matches.
fn basename_stem(path: &str) -> String {
    let base = path.rsplit(['\\', '/']).next().unwrap_or(path);
    base.split('.').next().unwrap_or(base).to_ascii_uppercase()
}

/// Classify a model path by [`LAMP_NAME_KEYS`] (first match wins — the list is ordered).
pub fn lamp_kind(path: &str) -> Option<LampKind> {
    let stem = basename_stem(path);
    let key = LAMP_NAME_KEYS.iter().find(|k| stem.contains(**k))?;
    Some(match *key {
        "CHANDELIER" => LampKind::Chandelier,
        "CANDLE" | "CANDELABRA" => LampKind::Candle,
        "BRAZIER" => LampKind::Brazier,
        "TORCH" => LampKind::Torch,
        _ => LampKind::Lamp,
    })
}

/// The artist's dark twin ([`BROKEN_KEYS`]), plus the `…Off` suffix that means the same thing.
///
/// The suffix is tested as a SUFFIX **after the trailing serial number is stripped**, never as a
/// substring: `Offset`, `Office` and `_offhand` would take half the corpus with them, while
/// `CandleOff01`/`02`/`03` — which Stormwind's WMO places **65 times**, more than any other candle —
/// only reads as extinguished once `01` comes off the end. `WesternPlaguelandsLampostStraightOff`
/// has no serial and is caught either way. Getting this wrong put 65 lights inside one city.
pub fn is_broken_or_off(path: &str) -> bool {
    let stem = basename_stem(path);
    if BROKEN_KEYS.iter().any(|k| stem.contains(k)) {
        return true;
    }
    stem.trim_end_matches(|c: char| c.is_ascii_digit())
        .trim_end_matches(['_', '-'])
        .ends_with("OFF")
}

/// A model that draws a LIGHT BEAM rather than a light source ([`BEAM_KEYS`]).
pub fn is_light_beam(path: &str) -> bool {
    let stem = basename_stem(path);
    BEAM_KEYS.iter().any(|k| stem.contains(k))
}

/// Does any DIRECTORY segment of this path name a light-fixture folder ([`LAMP_DIR_KEYS`])? The
/// basename is excluded deliberately — that is route (a)'s job, and reading it here would make the
/// two routes indistinguishable in the audit.
pub fn in_lamp_dir(path: &str) -> bool {
    let p = path.replace('/', "\\").to_ascii_uppercase();
    let Some(dir) = p.rfind('\\').map(|i| &p[..i]) else {
        return false;
    };
    LAMP_DIR_KEYS.iter().any(|k| dir.contains(k))
}

/// The content-family gate for the LAMP route: which trees may synthesise from mere unlit geometry.
///
/// Unlit + additive is the ordinary way to draw a highlight on a sword, a ghost's shroud, an
/// elemental's core, a spell's flash — thousands of batches whose models emit no light in any
/// client. Admitting them would turn every equipped weapon and every caster into a lamp. So the
/// route is confined to placed WORLD scenery, which is the entire population it exists for:
/// lampposts, lanterns, chandelier and sconce props. `ITEM\ObjectComponents\` is named explicitly
/// in the exclusion because that is where the two hand-held lanterns live
/// (`Misc_1H_Lantern_A_01`) — a carried lamp is a real light, but the FLAME route already handles
/// the ones whose artist authored a flame, and lighting a lantern *in a bank slot* is not worth it.
pub fn lamp_family_ok(path: &str) -> bool {
    let p = path.replace('/', "\\").to_ascii_uppercase();
    let world = p.starts_with("WORLD\\") || p.starts_with("DUNGEONS\\");
    world && !p.contains("\\SPELLS\\") && !p.starts_with("ITEM\\OBJECTCOMPONENTS\\")
}

/// Does this texture path name glow art? Basename, extension stripped, case-insensitive
/// [`GLOW_TEXTURE_KEYS`] substring — **minus** the [`BEAM_KEYS`] art, for the same reason
/// [`is_light_beam`] exists one level up: `HazardLightBeam.blp` and `TJ_LightShaft.blp` paint the
/// CONE a lamp throws, not the lamp. Their geometry is a long card whose centroid sits a yard or
/// two out along the beam, so taking it as the lamp head puts the point light in mid-air beside the
/// fixture, in the exact spot the artist already painted bright. The lamp-route twin of
/// [`fire_texture`].
pub fn glow_texture(texture: Option<&str>) -> bool {
    let Some(t) = texture else { return false };
    let base = t.rsplit(['\\', '/']).next().unwrap_or(t);
    let stem = base.split('.').next().unwrap_or(base).to_ascii_uppercase();
    !BEAM_KEYS.iter().any(|k| stem.contains(k))
        && GLOW_TEXTURE_KEYS.iter().any(|k| stem.contains(k))
}

/// One render batch as the lamp route reads it — the minimum a caller must expose. Built from a
/// [`crate::RenderSubmesh`] by the [`From`] impl below; hand-built in the tests, which is why it is
/// a borrowed view of five fields rather than the 30-field submesh itself.
#[derive(Debug, Clone, Copy)]
pub struct EmissiveBatch<'a> {
    /// The batch's resolved `.blp` path, if it carries one.
    pub texture: Option<&'a str>,
    /// M2 material render-flag `0x01` — **UNLIT**. The signal this route is built on.
    pub unlit: bool,
    /// M2 blend mode `2` (alpha) / `3` (NoAlphaAdd) / `4` (Add) — the transparent family a glow
    /// card is drawn in. NOT required by route (a): the most common lamp shape in the corpus
    /// (`StormwindStreetlamp01`, `DuskwoodLamppost`) authors its glass as **blend 0, unlit** —
    /// opaque coloured glass — and gating the name route on the blend would have missed exactly the
    /// models this was written for. Route (b) does require it, as corroboration for a nameless hit.
    pub blended: bool,
    /// Model-space vertices (WoW axes, Z up) — the centroid is where the light goes.
    pub positions: &'a [[f32; 3]],
    /// The batch's dominant bone (the mode of its per-vertex primary joint), or 0 for a boneless
    /// batch — so an animating lamp's light can ride the joint its glass rides.
    pub bone: u16,
}

impl<'a> From<&'a crate::RenderSubmesh> for EmissiveBatch<'a> {
    fn from(s: &'a crate::RenderSubmesh) -> Self {
        // The dominant bone: the most frequent PRIMARY joint across the batch's vertices. Lamp
        // glass is a rigid 16-vertex box skinned 100 % to one bone, so this is normally unanimous;
        // the histogram is there for the swinging hanging-lantern shapes, whose glass and chain sit
        // on different bones and whose batches can straddle both.
        //
        // Skipped entirely for a LIT batch, and that is not an optimisation detail: this conversion
        // runs at asset-load time over EVERY batch of every model that authors no light and no
        // flame — the whole character/creature corpus included — and only an unlit batch's bone is
        // ever read back ([`lamp_position`] returns a bone from the batch it picked, and it only
        // picks unlit ones). A per-vertex map insert across a 5000-vertex creature, for a field
        // nothing will look at, is load-time cost for nothing.
        let mut best = (0u16, 0usize);
        if s.emissive {
            let mut counts: std::collections::BTreeMap<u16, usize> =
                std::collections::BTreeMap::new();
            for j in &s.joints {
                let c = counts.entry(j[0]).or_default();
                *c += 1;
                if *c > best.1 {
                    best = (j[0], *c);
                }
            }
        }
        EmissiveBatch {
            texture: s.texture.as_deref(),
            unlit: s.emissive,
            blended: s.additive || matches!(s.blend, crate::ModelBlend::Blend),
            positions: &s.positions,
            bone: best.0,
        }
    }
}

/// One synthesised lamp light. The caller turns this into an [`crate::M2Light`] exactly as it does
/// a [`SyntheticFire`] — same `position`/`bone` convention, same `synthetic: true` flag.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SyntheticLamp {
    /// Which route fired ([`LightRoute::Name`] / [`LightRoute::Emissive`]) — audit only.
    pub route: LightRoute,
    /// The batch the position came from, or `None` when it fell back to the bounding box.
    pub batch: Option<usize>,
    /// Model-space position of the light (WoW axes, Z up) — see [`lamp_position`].
    pub position: [f32; 3],
    /// The position batch's dominant bone (`0` on the bbox fallback).
    pub bone: u16,
    /// Peak-normalised linear RGB (see [`lamp_color`]).
    pub color: [f32; 3],
    /// The light's `diffuse_intensity` (see [`lamp_intensity`]).
    pub intensity: f32,
    /// The intensity bucket's name — for the offline readout, and so the two routes' readouts stay
    /// directly comparable.
    pub bucket: &'static str,
}

/// The lamp's colour. **No texture is sampled** — deliberately.
///
/// Decoding the glass BLP would mean IO inside a rule that has to run both in the offline sweep and
/// in the asset loader's bake, and the answer would nearly always be the same anyway: shipped lamp
/// glass is authored amber. So the rule is a documented table, in priority order:
///
/// 1. **A magic hue in the model name wins** — `NightElfLanternBlue`, `BE_lantern_red_001`,
///    `NE_LanternBlue01`. These are the models where the default would be visibly, obviously wrong
///    (a moonwell-blue night-elf lantern glowing orange), and the artist wrote the colour into the
///    filename precisely because it is the point of the model.
/// 2. **A lamp/lantern/post/sconce/chandelier/candle name** ⇒ [`DEFAULT_LAMP`] `(1.0, 0.72, 0.40)`:
///    warm amber, softened by the glass shade it shines through.
/// 3. **Anything else** (the nameless emissive route (b), and the brazier/torch end of the
///    vocabulary) ⇒ [`DEFAULT_WARM`] `(1.0, 0.55, 0.2)`: open flame — the same default the particle
///    route falls back to, so the two routes agree wherever they describe the same thing.
pub fn lamp_color(path: &str) -> [f32; 3] {
    let stem = basename_stem(path);
    for (key, rgb) in [
        ("BLUE", [0.45, 0.65, 1.0]),
        ("GREEN", [0.40, 1.0, 0.45]),
        ("PURPLE", [0.70, 0.40, 1.0]),
        ("ARCANE", [0.70, 0.40, 1.0]),
        ("RED", [1.0, 0.35, 0.25]),
    ] {
        if stem.contains(key) {
            return rgb;
        }
    }
    // The lamp family gets the shaded amber; a brazier/torch name gets the open-flame default.
    if ["LAMP", "LANTERN", "POST", "SCONCE", "CANDLE", "CHANDELIER"]
        .iter()
        .any(|k| stem.contains(k))
    {
        DEFAULT_LAMP
    } else {
        DEFAULT_WARM
    }
}

/// Bucket a lamp onto the SAME intensity rungs the flame route uses (0.6 / 1.5 / 2.0 / 3.0), so the
/// downstream reach table (`benilla_world::lighting::m2_light_reach`, which buckets 0.6 → 6 yd,
/// 1.5 → 12, 2.0 → 16, 3.0 → 24) needs no new rung and a synthesised lamp sits in the same range as
/// an authored `MediumBrazier01`.
///
/// - A **chandelier** is a hall's whole light source (a dozen candles on one fixture) and a
///   **brazier** is the authored corpus's own 2.0: the brazier rung, 2.0 / 16 yd.
/// - A **candle** is one flame in a glass: the candle rung, 0.6 / 6 yd.
/// - Everything else — lamppost, street lamp, hanging lantern, sconce, torch — is the torch rung,
///   1.5 / 12 yd. That is the number the whole feature turns on: a Valley of Heroes lamppost throws
///   12 yd, which reaches the road under it and the wall beside it without crossing the canal.
///
/// Nothing here reaches the 3.0 bonfire rung. A lamp is not a bonfire, and a synthesised light that
/// out-blazes `Forgebonfire` would be visibly wrong in exactly the place it is most looked at.
pub fn lamp_intensity(kind: Option<LampKind>) -> (f32, &'static str) {
    match kind {
        Some(LampKind::Chandelier) => (2.0, "chandelier"),
        Some(LampKind::Candle) => (0.6, "candle"),
        // A brazier lands on the authored fire family's own rung; a torch on the torch rung. Both
        // reach here only WITH glowing geometry (see [`LampKind::is_fire`]), so this one is lit.
        Some(LampKind::Brazier) => (2.0, "brazier"),
        Some(LampKind::Torch) => (1.5, "torch"),
        _ => (1.5, "lamp"),
    }
}

/// Pick the batch whose centroid is the light's position, and the position itself.
///
/// **Not the model origin.** A lamppost's origin is at its BASE, on the ground: a light there is
/// buried inside the post, lights the cobbles in a ring around the foot of the pole, and (on the
/// torch-shadow lane, which includes the prop's own caster mesh) shadows everything the lamp is
/// supposed to be illuminating. The glass is 4–5 yd up. That vertical offset is the whole feature.
///
/// The pick, in order:
/// 1. The UNLIT batch carrying GLOW art with the most vertices — the glass box (16 verts) rather
///    than the additive corona card (4 verts, a billboard whose two triangles say much less about
///    where the lamp head actually is).
/// 2. Failing that, any unlit batch — a name-route hit whose glass carries unrecognised art.
/// 3. Failing that, the model's bounding-box TOP CENTRE `(cx, cy, zmax)` — but ONLY when `bbox_ok`
///    (see [`LampKind::Fire`]: a cold brazier is refused it). A name-route hit with no unlit
///    geometry is a lamp whose head cannot be located from the geometry; the top of its box is the
///    least-wrong guess for a pole-shaped prop, and the sweep reports it as a fallback. It is the
///    right answer more often than it looks — `StormwindCanalLamp01` is one 21 yd batch whose lamp
///    head is its own bbox top — and merely *high* on a hanging lantern, whose box top is the
///    ceiling hook a yard above the glass.
pub fn lamp_position(
    batches: &[EmissiveBatch<'_>],
    bounds: Option<([f32; 3], [f32; 3])>,
    bbox_ok: bool,
) -> Option<(Option<usize>, [f32; 3], u16)> {
    let centroid = |b: &EmissiveBatch| {
        let n = b.positions.len() as f32;
        let s = b
            .positions
            .iter()
            .fold([0.0f32; 3], |a, p| [a[0] + p[0], a[1] + p[1], a[2] + p[2]]);
        [s[0] / n, s[1] / n, s[2] / n]
    };
    let pick = |glow_only: bool| {
        batches
            .iter()
            .enumerate()
            .filter(|(_, b)| {
                b.unlit && !b.positions.is_empty() && (!glow_only || glow_texture(b.texture))
            })
            .max_by_key(|(_, b)| b.positions.len())
    };
    if let Some((i, b)) = pick(true).or_else(|| pick(false)) {
        return Some((Some(i), centroid(b), b.bone));
    }
    if !bbox_ok {
        return None;
    }
    let (lo, hi) = bounds?;
    Some((None, [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, hi[2]], 0))
}

/// Derive **at most one** light for a model that authors neither a casting light block nor a flame
/// emitter, from its NAME and/or its emissive geometry. `None` for everything else.
///
/// Two routes, evaluated in that order:
///
/// - **(a) NAME.** The basename matches [`LAMP_NAME_KEYS`]. This is the route the whole street-lamp
///   corpus takes, and it does not require the geometry to prove anything — `StormwindStreetlamp01`
///   authors its glass as an *opaque* unlit batch, which no blend-mode test would admit.
/// - **(b) EMISSIVE.** No name hit, but the model sits in a light-FIXTURE FOLDER ([`in_lamp_dir`])
///   AND carries an UNLIT batch that is also in the transparent family (blend 2/3/4) AND whose
///   texture is glow art ([`GLOW_TEXTURE_KEYS`]). This is the "lamp glass without a lamp name"
///   catcher — `Scholme_Candelabra`, `TS_LightPole`, `GnomeHazardLight01`, `BFD_Wisp01`: fixtures
///   named after their building or their species rather than after the thing. It is much the
///   STRICTER of the two, because unlit glow geometry on its own means nothing at all (the sweep's
///   first cut lit blood-elf banners, Ghostlands trees, PvP runes and a Karazhan owl statue).
///
/// Vetoed throughout: spell effects ([`is_spell_path`]), light BEAMS ([`is_light_beam`]), the
/// artist's dark twins ([`is_broken_or_off`]; `NoOmni` is NOT one — see [`is_no_omni`]) and everything outside placed world
/// scenery ([`lamp_family_ok`]).
pub fn synthesize_lamp_light(
    path: &str,
    batches: &[EmissiveBatch<'_>],
    bounds: Option<([f32; 3], [f32; 3])>,
) -> Option<SyntheticLamp> {
    if is_spell_path(path)
        || is_broken_or_off(path)
        || is_light_beam(path)
        || !lamp_family_ok(path)
    {
        return None;
    }
    let kind = lamp_kind(path);
    let route = if kind.is_some() {
        LightRoute::Name
    } else if in_lamp_dir(path)
        && batches
            .iter()
            .any(|b| b.unlit && b.blended && glow_texture(b.texture))
    {
        LightRoute::Emissive
    } else {
        return None;
    };
    // The bounding-box fallback is for FIXTURES, not fire containers — see [`LampKind::is_fire`].
    let bbox_ok = !kind.is_some_and(LampKind::is_fire);
    let (batch, position, bone) = lamp_position(batches, bounds, bbox_ok)?;
    let (intensity, bucket) = lamp_intensity(kind);
    Some(SyntheticLamp {
        route,
        batch,
        position,
        bone,
        color: lamp_color(path),
        intensity,
        bucket,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CellRamp, EmitParams, EmitTiming, ParamsNow, ParticleShape};

    /// A minimal emitter def — only the fields the fire rule reads carry meaning.
    fn def(
        texture: Option<&str>,
        blend: ParticleBlend,
        keys: [[f32; 4]; 3],
        size: f32,
        rate: f32,
    ) -> ParticleEmitterDef {
        ParticleEmitterDef {
            flags: 0,
            position: [0.0, 0.0, 1.0],
            bone: 0,
            shape: ParticleShape::Plane,
            blend,
            lit: false,
            texture: texture.map(str::to_string),
            tile_rows: 1,
            tile_cols: 1,
            head_tail: 0,
            timing: EmitTiming::constant(rate),
            params: EmitParams::constant(ParamsNow::default()),
            drag: 0.0,
            tail_time: 0.0,
            spline: None,
            geometry_model: None,
            recursion_model: None,
            angular_velocity_min: [0.0; 3],
            angular_velocity_max: [0.0; 3],
            inherit_scale: 0.0,
            follow_speed1: 0.0,
            follow_scale1: 0.0,
            follow_speed2: 0.0,
            follow_scale2: 0.0,
            twinkle_speed: 0.0,
            twinkle_percent: 1.0,
            twinkle_min: 0.0,
            twinkle_max: 0.0,
            spin: 0.0,
            over_life: OverLife {
                mid: 0.5,
                color: keys,
                scale: [size; 3],
                head_cells: [CellRamp::new(0, 0); 2],
                tail_cells: [CellRamp::new(0, 0); 2],
                repeat: [1.0; 2],
            },
        }
    }

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 2e-3)
    }

    /// GOLDEN — the colour rule on the three REAL ramps the heuristic was calibrated against
    /// (dumped with `benilla-extract m2part`). Each is authored white-cored, so key 0 or a mean
    /// would wash all three to the same pale nothing; the most-saturated key recovers the hue the
    /// artist painted, and peak-normalising strips the brightness out of it.
    #[test]
    fn flame_color_takes_the_most_saturated_key() {
        // OgreWallTorchpurple — FLAMELICKSMALLMAGICBLUE: white → violet → deep violet.
        let purple = def(
            Some("WORLD\\GENERIC\\OGRE\\TORCHES\\FLAMELICKSMALLMAGICBLUE.BLP"),
            ParticleBlend::Add,
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.565, 0.212, 1.0, 1.0],
                [0.42, 0.0, 0.745, 1.0],
            ],
            0.3,
            20.0,
        );
        assert!(close(flame_color(&purple.over_life), [0.565, 0.212, 1.0]));

        // OgreWallTorch — FLAMELICKSMALL: orange → tan → white burnout.
        let warm = def(
            Some("ITEM\\OBJECTCOMPONENTS\\FLAMELICKSMALL.BLP"),
            ParticleBlend::Add,
            [
                [1.0, 0.282, 0.0, 1.0],
                [0.875, 0.541, 0.184, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            0.3,
            20.0,
        );
        assert!(close(flame_color(&warm.over_life), [1.0, 0.282, 0.0]));

        // HumanBrazierMagic — FLAMELICK: green → pale green → near white.
        let green = def(
            Some("INTERFACE\\GLUES\\FLAMELICK.BLP"),
            ParticleBlend::Add,
            [
                [0.0, 0.988, 0.0, 1.0],
                [0.486, 0.976, 0.51, 1.0],
                [0.851, 1.0, 0.843, 1.0],
            ],
            0.6,
            30.0,
        );
        assert!(close(flame_color(&green.over_life), [0.0, 1.0, 0.0]));

        // An all-white ramp carries no hue at all — the warm default, not white light.
        let white = def(
            Some("FIRE.BLP"),
            ParticleBlend::Add,
            [[1.0, 1.0, 1.0, 1.0]; 3],
            0.3,
            20.0,
        );
        assert_eq!(flame_color(&white.over_life), DEFAULT_WARM);
    }

    /// GOLDEN — the DETECTOR: flame art plus additive blend, and every near-miss that must stay
    /// out. `GLOW` is the load-bearing exclusion (crystals/runestones), smoke is caught by blend.
    #[test]
    fn detector_admits_flames_and_refuses_the_near_misses() {
        let keys = [[1.0, 0.3, 0.0, 1.0]; 3];
        let fire = |t: Option<&str>, b| def(t, b, keys, 0.3, 20.0);

        assert!(fire_emitter(&fire(Some("X\\FLAMELICKSMALL.BLP"), ParticleBlend::Add)));
        assert!(fire_emitter(&fire(Some("X\\CANDLEFLAME.BLP"), ParticleBlend::Add)));
        assert!(fire_emitter(&fire(Some("X\\TORCHFIRE.BLP"), ParticleBlend::Add)));
        // Art that is not fire.
        assert!(!fire_emitter(&fire(Some("X\\GLOW.BLP"), ParticleBlend::Add)));
        assert!(!fire_emitter(&fire(Some("X\\SPELLGLOW32.BLP"), ParticleBlend::Add)));
        assert!(!fire_emitter(&fire(Some("X\\SMOKE01.BLP"), ParticleBlend::Add)));
        assert!(!fire_emitter(&fire(None, ParticleBlend::Add)));
        // Fire art on a NON-additive emitter is the same model's smoke plume, not its flame.
        assert!(!fire_emitter(&fire(
            Some("X\\FIRESMOKE.BLP"),
            ParticleBlend::Alpha
        )));
        // A dead emitter (zero rate) is not a light source.
        assert!(!fire_emitter(&def(
            Some("X\\FLAMELICK.BLP"),
            ParticleBlend::Add,
            keys,
            0.3,
            0.0
        )));
        // A SPARK — fire art, additive, but a 0.025 yd particle: a zeppelin's exhaust cinders, a
        // flaming helm's glitter, a fire elemental's secondary embers. Below the flame floor.
        assert!(!fire_emitter(&def(
            Some("X\\EMBER_OFFSET.BLP"),
            ParticleBlend::Add,
            keys,
            0.025,
            200.0
        )));
        // A real candle clears it by a hair — the floor's whole calibration.
        assert!(fire_emitter(&def(
            Some("X\\FLAMELICKSMALL.BLP"),
            ParticleBlend::Add,
            keys,
            0.056,
            6.0
        )));
    }

    /// GOLDEN — one light per model, from the STRONGEST fire emitter, and never for a spell.
    #[test]
    fn one_light_from_the_strongest_emitter() {
        let embers = def(
            Some("X\\EMBER.BLP"),
            ParticleBlend::Add,
            [[1.0, 0.2, 0.0, 1.0]; 3],
            0.08,
            40.0,
        );
        let flame = def(
            Some("X\\FLAMELICKLARGE.BLP"),
            ParticleBlend::Add,
            [[0.2, 0.4, 1.0, 1.0]; 3],
            1.4,
            60.0,
        );
        let smoke = def(
            Some("X\\FIRESMOKE.BLP"),
            ParticleBlend::Alpha,
            [[0.5, 0.5, 0.5, 1.0]; 3],
            2.0,
            10.0,
        );
        let got = synthesize_fire_light("WORLD\\GOBER\\BONFIRE.M2", [&embers, &flame, &smoke])
            .expect("a fire");
        assert_eq!(got.emitter, 1, "the big flame wins, not the ember shower");
        assert!(close(got.color, [0.2, 0.4, 1.0]));
        assert_eq!(got.bucket, "bonfire");
        assert_eq!(got.intensity, 3.0);

        // The candle end of the scale lands in its own bucket.
        let candle = def(
            Some("X\\CANDLEFLAME.BLP"),
            ParticleBlend::Add,
            [[1.0, 0.5, 0.1, 1.0]; 3],
            0.06,
            8.0,
        );
        let got = synthesize_fire_light("WORLD\\GOBER\\CANDLE.M2", [&candle]).expect("a fire");
        assert_eq!(got.bucket, "candle");

        // A spell effect never synthesises, however fiery — it would strobe the point table.
        assert_eq!(
            synthesize_fire_light("SPELLS\\FIREBALL_MISSILE.M2", [&flame]),
            None
        );
        // The `NoOmni` twin DOES synthesise — it is Orgrimmar's street lighting (123 placed);
        // only the absence of a flame emitter keeps a brazier cold.
        assert!(
            synthesize_fire_light("World\\Generic\\Orc\\Braziers\\TallBrazierNoOmni01.m2", [&flame])
                .is_some()
        );
        assert_eq!(synthesize_fire_light("WORLD\\X.M2", [&smoke]), None);
    }

    // -----------------------------------------------------------------------------------------
    // MONKEY (lamp lights): the emissive-geoset route.
    // -----------------------------------------------------------------------------------------

    /// A minimal render-batch view. `positions` is handed in by the caller because
    /// [`EmissiveBatch`] borrows it — a lamp's glass is a 16-vertex box, and only its centroid and
    /// its vertex count are ever read.
    fn batch<'a>(
        texture: Option<&'a str>,
        unlit: bool,
        blended: bool,
        positions: &'a [[f32; 3]],
    ) -> EmissiveBatch<'a> {
        EmissiveBatch {
            texture,
            unlit,
            blended,
            positions,
            bone: 0,
        }
    }

    /// GOLDEN — route (a), the NAME hit, on the model the whole feature was written for.
    /// `StormwindStreetlamp01` (40 placements inside the Stormwind WMO) authors NO particle
    /// emitter at all and draws its glass **opaque + unlit** — blend 0 — so neither the flame rule
    /// nor any blend-mode test sees it. The light must land on the glass at z ≈ 4.35, four yards
    /// above the post's own origin at the base: that vertical offset IS the fix.
    #[test]
    fn name_route_lights_a_lamppost_at_its_glass() {
        let post: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [0.5, 0.0, 4.0], [-0.5, 0.0, 2.0]];
        let glass: Vec<[f32; 3]> = vec![
            [0.5, 0.5, 4.1],
            [-0.5, 0.5, 4.1],
            [0.5, -0.5, 4.6],
            [-0.5, -0.5, 4.6],
        ];
        let batches = [
            batch(Some("X\\STREETLAMP.BLP"), false, false, &post),
            batch(Some("DUNGEONS\\STORMWINDLAMPGLASS.BLP"), true, false, &glass),
        ];
        let got = synthesize_lamp_light(
            "World\\Generic\\Human\\Passive Doodads\\Lamps\\StormwindStreetlamp01.m2",
            &batches,
            Some(([-0.6, -0.6, 0.0], [0.6, 0.6, 4.9])),
        )
        .expect("a lamppost");
        assert_eq!(got.route, LightRoute::Name);
        assert_eq!(got.batch, Some(1), "the glass, not the post");
        assert!(
            (got.position[2] - 4.35).abs() < 1e-3,
            "the light rides the glass, not the origin: {:?}",
            got.position
        );
        assert_eq!(got.color, DEFAULT_LAMP);
        assert_eq!((got.intensity, got.bucket), (1.5, "lamp"));
    }

    /// GOLDEN — route (b), the NAMELESS hit: a fixture-folder model whose name says nothing
    /// (`Scholme_Candelabra`-shaped), carried entirely by its unlit + transparent + glow-textured
    /// batch. All three signals are required, and the folder is the load-bearing one.
    #[test]
    fn emissive_route_needs_a_fixture_folder_and_all_three_signals() {
        let body: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]];
        let flame: Vec<[f32; 3]> = vec![[0.0, 0.0, 1.0], [0.2, 0.0, 1.4]];
        let path = "World\\Lordaeron\\Scholomance\\PassiveDoodads\\Candles\\Scholme_Wax03.m2";

        let hit = [
            batch(Some("X\\WAX.BLP"), false, false, &body),
            batch(Some("X\\GENERICGLOW64.BLP"), true, true, &flame),
        ];
        let got = synthesize_lamp_light(path, &hit, None).expect("a fixture");
        assert_eq!(got.route, LightRoute::Emissive);
        assert_eq!(got.batch, Some(1));
        assert!((got.position[2] - 1.2).abs() < 1e-3, "{:?}", got.position);
        // No name to classify on ⇒ the generic lamp rung and the open-flame default hue.
        assert_eq!((got.intensity, got.bucket), (1.5, "lamp"));
        assert_eq!(got.color, DEFAULT_WARM);

        // The same geometry filed OUTSIDE a fixture folder is a blood-elf banner / a Ghostlands
        // tree / a PvP rune — the 149-model false-positive population the folder gate removed.
        assert_eq!(
            synthesize_lamp_light(
                "World\\Expansion01\\Doodads\\Generic\\BloodElf\\Banners\\BE_Banner02.m2",
                &hit,
                None
            ),
            None
        );
        // In the folder, but the batch is LIT (no 0x01) — ordinary painted geometry.
        let lit = [batch(Some("X\\GENERICGLOW64.BLP"), false, true, &flame)];
        assert_eq!(synthesize_lamp_light(path, &lit, None), None);
        // In the folder and unlit, but OPAQUE (blend 0): no corroboration for a nameless hit.
        let opaque = [batch(Some("X\\GENERICGLOW64.BLP"), true, false, &flame)];
        assert_eq!(synthesize_lamp_light(path, &opaque, None), None);
        // In the folder, unlit and transparent, but the art is not glow art.
        let plain = [batch(Some("X\\WOOD01.BLP"), true, true, &flame)];
        assert_eq!(synthesize_lamp_light(path, &plain, None), None);
    }

    /// GOLDEN — the NEAR-MISSES that must stay dark. Each one shipped, each one reached the first
    /// cut of the rule, and each one would have been visible as a wrongly glowing object.
    #[test]
    fn lamp_route_refuses_the_near_misses() {
        let verts: Vec<[f32; 3]> = vec![[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]];
        let glow = [batch(Some("X\\GLOW32.BLP"), true, true, &verts)];
        let bbox = Some(([-1.0, -1.0, 0.0], [1.0, 1.0, 3.0]));

        // A GLOW card on a model that is not a fixture and not in a fixture folder: the whole
        // magic-prop corpus (runestones, jewels, holograms, teleport pads, banners, trees).
        assert_eq!(
            synthesize_lamp_light(
                "World\\Expansion01\\Doodads\\Generic\\BloodElf\\RuneStone\\BE_Runestone01.m2",
                &glow,
                bbox
            ),
            None
        );
        // A light BEAM in a `…Lights\` folder — lit air, not a source. Its centroid is mid-shaft.
        assert_eq!(
            synthesize_lamp_light(
                "World\\NoDXT\\Generic\\PassiveDoodads\\VolumetricLights\\LightShaftA.m2",
                &glow,
                bbox
            ),
            None
        );
        // The artist's extinguished twin — `CandleOff01`, which Stormwind's WMO places 65 times.
        // The serial number has to come off before the `OFF` suffix is read.
        assert_eq!(
            synthesize_lamp_light(
                "World\\Generic\\PassiveDoodads\\Lights\\CandleOff01.m2",
                &glow,
                bbox
            ),
            None
        );
        assert_eq!(
            synthesize_lamp_light(
                "World\\Generic\\Human\\Passive Doodads\\LampPosts\\DuskwoodLamppostBroken.m2",
                &glow,
                bbox
            ),
            None
        );
        assert_eq!(
            synthesize_lamp_light(
                "world\\expansion07\\doodads\\dungeon\\doodads\\8du_waycrest_candlestand01_unlit.m2",
                &glow,
                bbox
            ),
            None
        );
        // A COLD brazier: a fire container with no flame emitter and not one glowing polygon. The
        // bounding-box fallback is refused to it, so it stays dark.
        let cold: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.5]];
        let plain = [batch(Some("X\\METAL01.BLP"), false, false, &cold)];
        assert_eq!(
            synthesize_lamp_light(
                "World\\KhazModan\\Uldaman\\PassiveDoodads\\Braziers\\UldamanBrazier01.m2",
                &plain,
                bbox
            ),
            None
        );
        // …while a LAMP with the same unflagged geometry keeps the fallback: `UldamanLamp`,
        // `IronForgeHangingLantern01`, `AhnQirajSconce01` are real fixtures whose glass simply
        // isn't flagged, and the box top is the least-wrong guess for a pole-shaped prop.
        let lamp = synthesize_lamp_light(
            "World\\KhazModan\\Uldaman\\PassiveDoodads\\Lamps\\UldamanLamp.m2",
            &plain,
            bbox,
        )
        .expect("a lamp");
        assert_eq!(lamp.batch, None, "the bbox fallback");
        assert_eq!(lamp.position, [0.0, 0.0, 3.0], "box top centre");
        // A fixture folder whose only unlit card is the BEAM it throws (`HazardLightBeam.blp`):
        // the cone, not the source, and its centroid is a yard out along the beam.
        let beam = [batch(Some("X\\HAZARDLIGHTBEAM.BLP"), true, true, &verts)];
        assert!(!glow_texture(Some("X\\HAZARDLIGHTBEAM.BLP")));
        assert_eq!(
            synthesize_lamp_light(
                "World\\Generic\\Gnome\\Passive Doodads\\HazardLights\\GnomeHazardLight01.m2",
                &beam,
                bbox
            ),
            None
        );
        // Outside placed world scenery entirely — a carried lantern, a spell, an item.
        assert_eq!(
            synthesize_lamp_light(
                "Item\\ObjectComponents\\Weapon\\Misc_1H_Lantern_A_01.m2",
                &glow,
                bbox
            ),
            None
        );
        assert_eq!(
            synthesize_lamp_light("SPELLS\\Lantern_Impact.m2", &glow, bbox),
            None
        );
    }

    /// GOLDEN — the kind/colour/intensity table: the three rungs, and the magic-hue override that
    /// the artist wrote into the filename precisely because the warm default would be wrong there.
    #[test]
    fn lamp_kind_colour_and_intensity() {
        assert_eq!(lamp_kind("x\\KarazanChandelier_01.m2"), Some(LampKind::Chandelier));
        assert_eq!(lamp_kind("x\\Scholme_Candelabra.m2"), Some(LampKind::Candle));
        assert_eq!(lamp_kind("x\\SkullCandle01.m2"), Some(LampKind::Candle));
        assert_eq!(lamp_kind("x\\DuskwoodLamppost.m2"), Some(LampKind::Lamp));
        assert_eq!(lamp_kind("x\\GeneralHangingLantern01.m2"), Some(LampKind::Lamp));
        assert_eq!(lamp_kind("x\\UldamanBrazier01.m2"), Some(LampKind::Brazier));
        assert_eq!(lamp_kind("x\\NA_Torch01.m2"), Some(LampKind::Torch));
        assert_eq!(lamp_kind("x\\BE_Banner02.m2"), None);

        assert_eq!(lamp_intensity(Some(LampKind::Chandelier)), (2.0, "chandelier"));
        assert_eq!(lamp_intensity(Some(LampKind::Candle)), (0.6, "candle"));
        assert_eq!(lamp_intensity(Some(LampKind::Lamp)), (1.5, "lamp"));
        assert_eq!(lamp_intensity(None), (1.5, "lamp"));

        // Every rung must be one the reach table already knows (6 / 12 / 16 / 24 yd).
        for k in [
            None,
            Some(LampKind::Chandelier),
            Some(LampKind::Candle),
            Some(LampKind::Brazier),
            Some(LampKind::Torch),
        ] {
            let i = lamp_intensity(k).0;
            assert!([0.6f32, 1.5, 2.0, 3.0].contains(&i), "off-rung intensity {i}");
        }

        assert_eq!(lamp_color("x\\StormwindStreetlamp01.m2"), DEFAULT_LAMP);
        assert_eq!(lamp_color("x\\NE_LanternBlue01.m2"), [0.45, 0.65, 1.0]);
        assert_eq!(lamp_color("x\\BE_lantern_red_001.m2"), [1.0, 0.35, 0.25]);
        assert_eq!(lamp_color("x\\Scholme_GreenCandelabra.m2"), [0.40, 1.0, 0.45]);
        // No lamp word at all ⇒ the open-flame default, shared with the particle route.
        assert_eq!(lamp_color("x\\Scholme_Wax03.m2"), DEFAULT_WARM);
    }
}
