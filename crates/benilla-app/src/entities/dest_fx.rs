//! **Dest-anchored spell effects** (decision 0797, B132's second half): what a ground cast shows
//! at the point. Two producers, one spawn/attach lane:
//!
//! - **The DynamicObject machine** ([`arm_ground_effects`]): a TYPEID-6 create is the anchor a
//!   persistent area effect hangs on (Blizzard's storm, Flamestrike's burn). The reference
//!   builds **two disjoint visuals** (wow-re `dynobject-visual-machine.md` — never through the
//!   unit-kit pipeline; all 15 `PlaySpellVisualKit` call sites censused, none on this class):
//!   - **Visual A** — the object's own `.mdx`: SPELLID → `Spell.dbc` SpellVisual →
//!     `SpellVisual` field 11 ≠ 0 gate → field 12 → `SpellVisualEffectName` field 2 path
//!     (`0x5d57c0`), instanced at the object's position **verbatim** (no terrain projection),
//!     Z-rotated by `DYNAMICOBJECT_FACING` (`0x613ef0` → `0x7bdd60`), looping its sequence
//!     (`0x5d5580` re-fires on completion).
//!   - **Visual B** — the shard emitter (`AUBlizzardObject`, `0x5d55c0` → `0x6ece30`):
//!     `SpellVisual` field 13 → `SpellVisualKit` → the first `CharProcType == 9` block; its
//!     `CharParamZero` small-int-decodes (`bits(p0 + 512.0) >> 14 & 0xff`) into a **hardcoded
//!     7-entry model table** (`0x870e24` — [`SHARD_MODELS`]), `CharParamOne` is the emit rate
//!     (× the graphics-quality factor — we run max quality, ×1.0). Each emitted shard spawns at
//!     a random offset inside `DYNAMICOBJECT_RADIUS` — the wire radius IS the visual footprint
//!     (`0x6ebad0`). The kit's field-13 `SoundEntries` is the looping area sound.
//! - **The GO dest one-shot** ([`spawn_ground_bursts`] ← the router's [`GroundBurst`]):
//!   `SMSG_SPELL_GO` itself plays the field-12 model ONCE at the packet's dest point when
//!   `SpellVisual` field 6 == 0 (no missile owns the arrival) — `0x6e8088`–`0x6e8143`, a
//!   self-terminating CEffect. Flamestrike's initial burst; it fires **before** the dynobj
//!   create arrives and must not wait for it (wow-re trap #6).
//!
//! **Teardown is a tail, not a snap** (trap #3): `SMSG_DESTROY_OBJECT` despawns the anchor —
//! visual A and the emitter die with it (the ref zeroes the emit rate at `0x6ecf20`) — but the
//! already-emitted shards are FREE entities that run out their own one-pass lifetimes, exactly
//! the ref's "spawned particles finish". The looping sound dies via the sound module's
//! `RemovedComponents<NetEntity>` hold-loop reaper.
//!
//! Named approximations: the ref's 3 s sound fade at destroy is a hard stop here (no channel
//! fade affordance yet); the hardcoded loop sequence id `0x9e` (fired when `obj+0x190` bit 1 is
//! set — the bit's source is unpinned) is approximated by looping the model's first clip; the
//! quality-tier emit factor is pinned at ×1.0 (max); Flamestrike's kit sound plays even though
//! its kit has no type-9 proc (the ref couples sound to the emitter object — whether a
//! procless kit still sounds is unread; the burn crackle missing would be the wronger look).

use bevy::prelude::*;

use crate::creature_anim::{SpellKitSound, SpellVisuals};
use crate::net::{NetEntity, ObjectStore};
use benilla_protocol::EntityKind;

use super::spell_fx::{attach_effect_visuals, ensure_model, FxMaterials, SpellFx};

/// The client's hardcoded shard-model table (`0x870e24`, 7 entries — wow-re
/// `dynobject-visual-machine.md` Q-A1). `CharParamZero`'s decoded small int indexes it.
const SHARD_MODELS: [&str; 7] = [
    "Spells\\Blizzard_Impact_Base.mdx",
    "Spells\\RainOfFire_Impact_Base.mdx",
    "Spells\\CallLightning_Impact.mdx",
    "Spells\\FlamestrikeSmall_Impact_Base.mdx",
    "Spells\\DeathAndDecay_Area_Base.mdx",
    "Spells\\ArcaneShot_Area.mdx",
    "Spells\\StarShards_Impact_Base.mdx",
];

/// The `CharProcType` the dynobj emitter chain scans for (`0x5d55c0`). One key out of the same
/// dispatch space the aura-state body procs come from (`crate::aura_visual`, decision 0806) — the
/// kit column is shared, the consumers are not.
const PROC_TYPE_SHARD_EMITTER: i32 = 9;

/// The exact small-int decode the client applies to `CharParamZero`
/// ([`benilla_formats::char_proc_small_int`] — the one idiom every integer-in-a-float-column proc
/// uses, the chain proc included; decision 0955 lifted it into the format crate) — clamped into
/// [`SHARD_MODELS`] because the client itself has **no bounds check** (`mov cl,al` — data ≥ 7
/// reads past the table; wow-re trap #2).
fn shard_model_index(param0: f32) -> usize {
    let idx = benilla_formats::char_proc_small_int(param0) as usize;
    idx.min(SHARD_MODELS.len() - 1)
}

/// A free-standing dest-anchored effect-model instance (visual A, a shard, a GO burst) —
/// [`attach_ground_fx_models`] hangs the model parts once the M2 builds.
#[derive(Component)]
pub(super) struct GroundFx {
    /// The [`SpellFx`] model-cache key.
    path: String,
    /// `true` = loop the clip for the instance's whole life (visual A — the ref re-fires its
    /// sequence on completion); `false` = one pass of sequence 0 then despawn (a shard, a
    /// burst — the kit pipeline's self-termination clock).
    looping: bool,
    /// Parts attached (the model was ready).
    spawned: bool,
    /// The repeat override landed on the live `AnimationPlayer` (loop instances only).
    loop_armed: bool,
    /// One-shot self-termination deadline (`time.elapsed_secs()`), set at attach.
    expires: Option<f32>,
}

impl GroundFx {
    fn new(path: String, looping: bool) -> Self {
        Self {
            path,
            looping,
            spawned: false,
            loop_armed: false,
            expires: None,
        }
    }
}

/// MONKEY (area spell light): what an armed DynamicObject anchor knows about the light its area
/// visual should throw, stamped at arm time and read once the model finishes building.
///
/// It exists because the two halves of the answer arrive at different moments, in different
/// systems. `radius` (the wire AoE) and `school` (the spell's `Spell.dbc` column) are only in hand
/// at [`arm_ground_effects`], where the create's fields and the spell catalog are; the model's own
/// verdict — whether its emitters synthesised a light, and in what hue — is only in hand at
/// [`attach_ground_fx_models`], however many frames later the M2 finishes building. Carrying the
/// first pair forward on the anchor is cheaper and far less fragile than re-resolving the spell id
/// there.
///
/// Its PRESENCE is load-bearing too: it marks the instance as a dynobj AREA visual, so the looping
/// branch of the attach takes the area verdict *including its refusals*. Without that a frost
/// area's visual A would fall through to the impact-flash light this lane used to spawn for it, and
/// Blizzard would light after all.
#[derive(Component)]
pub(super) struct AreaLightPlan {
    /// `DYNAMICOBJECT_RADIUS` — the effect's real footprint (the shard emitter spreads over exactly
    /// it), which sizes the pool and is the impact dedupe's test radius.
    radius: f32,
    /// The spell's `Spell.dbc` **School** index — the locale-proof half of the classification
    /// ([`benilla_formats::area_light_kind`]).
    school: u32,
}

/// The shard emitter riding a DynamicObject anchor (visual B — the ref's `AUBlizzardObject`).
/// Dies with the anchor (emission stops at destroy); its spawned shards are free entities and
/// finish on their own — the ref's emitter tail.
#[derive(Component)]
pub(super) struct ShardEmitter {
    /// The [`SHARD_MODELS`] path.
    path: String,
    /// `DYNAMICOBJECT_RADIUS` — the per-shard random spawn spread (the AoE footprint).
    radius: f32,
    /// Shards per second (`CharParamOne` × the quality factor, ×1.0 here).
    rate: f32,
    /// Fractional emissions carried between frames.
    accum: f32,
    /// A tiny LCG for the spawn offsets (the ref rolls real randoms; seeded per emitter).
    rng: u64,
}

/// The router's dest one-shot order (`SMSG_SPELL_GO` + `TARGET_FLAG_DEST_LOCATION`, gate
/// `SpellVisual` field 6 == 0 ∧ field 12 ≠ 0 — resolved router-side where the catalogs live).
#[derive(Message, Clone)]
pub(crate) struct GroundBurst {
    /// The field-12 `SpellVisualEffectName` model path.
    pub(crate) path: String,
    /// The dest point, bevy coords (converted at the apply seam).
    pub(crate) pos: Vec3,
}

/// Arm a freshly-created DynamicObject anchor: resolve the spell's visual row and hang the two
/// visuals + the area sound (module doc). Runs on `Added<ObjectStore>` so the create's fields
/// are the trigger; a dynobj is created once and never re-created in place (a re-cast is a new
/// guid — captured live, 0797).
pub(super) fn arm_ground_effects(
    mut commands: Commands,
    created: Query<(Entity, &NetEntity, &ObjectStore), Added<ObjectStore>>,
    visuals: Option<Res<SpellVisuals>>,
    spells: Option<Res<crate::ui_action::Spells>>,
    fx: Option<ResMut<SpellFx>>,
    asset_server: Res<AssetServer>,
    mut sounds: MessageWriter<SpellKitSound>,
) {
    let (Some(visuals), Some(spells), Some(mut fx)) = (visuals, spells, fx) else {
        return;
    };
    for (anchor, net, store) in &created {
        if net.kind != EntityKind::DynamicObject {
            continue;
        }
        let Some(spell_id) = store.0.dynamicobject_spell_id() else {
            continue;
        };
        let Some(stages) = spells
            .catalog
            .get(spell_id)
            .and_then(|d| visuals.0.stages(d.visual))
        else {
            debug!("dest_fx: dynobj spell {spell_id} has no visual row — invisible area");
            continue;
        };
        let facing = store
            .0
            .dynamicobject_position()
            .map(|(_, f)| f)
            .unwrap_or(0.0);
        // MONKEY (area spell light): stamp the two facts only this system holds, for the attach to
        // finish the verdict with. Unconditional — a plan whose school ends up dark is exactly how
        // Blizzard's visual A is told to throw NO light (see [`AreaLightPlan`]).
        let radius = store.0.dynamicobject_radius().unwrap_or(0.0);
        commands.entity(anchor).insert(AreaLightPlan {
            radius,
            school: spells.catalog.get(spell_id).map_or(0, |d| d.school),
        });
        // Visual A — the object's own model, gated on field 11, Z-rotated by FACING. A child of
        // the anchor: the destroy pop takes it exactly like the ref's scene-node teardown.
        if stages.area_gate != 0 && stages.area_effect != 0 {
            if let Some(path) = visuals.0.effect_path(stages.area_effect) {
                let path = path.to_string();
                ensure_model(&mut fx, &asset_server, &path);
                let child = commands
                    .spawn((
                        GroundFx::new(path, true),
                        Transform::from_rotation(Quat::from_rotation_y(facing)),
                        Visibility::default(),
                    ))
                    .id();
                commands.entity(anchor).add_child(child);
            }
        }
        // Visual B — the shard emitter, from the area kit's first type-9 CharProc block; the
        // kit's own sound column is the looping area sound (a tracked hold loop the sound
        // module reaps when the anchor's NetEntity is removed).
        if let Some(kit) = visuals.0.kit(stages.area_kit) {
            if let Some(proc) = kit.char_procs().find(|p| p.ty == PROC_TYPE_SHARD_EMITTER) {
                let path = SHARD_MODELS[shard_model_index(proc.params[0])].to_string();
                ensure_model(&mut fx, &asset_server, &path);
                commands.entity(anchor).insert(ShardEmitter {
                    path,
                    radius,
                    rate: proc.params[1],
                    accum: 0.0,
                    rng: 0x9e3779b97f4a7c15 ^ anchor.to_bits(),
                });
            }
            if let Some(kit_sound) = kit.sound {
                sounds.write(SpellKitSound::Play {
                    entity: anchor,
                    kit_sound,
                });
            }
        }
        debug!(
            "dest_fx: dynobj armed — spell {spell_id}, gateA={} effect={} kit={} radius {:?}",
            stages.area_gate,
            stages.area_effect,
            stages.area_kit,
            store.0.dynamicobject_radius(),
        );
    }
}

/// Spawn the router's GO dest one-shots — free entities at the packet's point, one sequence
/// pass, then gone. Fired at the GO, never waiting on the dynobj create (trap #6).
pub(super) fn spawn_ground_bursts(
    mut commands: Commands,
    mut bursts: MessageReader<GroundBurst>,
    fx: Option<ResMut<SpellFx>>,
    asset_server: Res<AssetServer>,
) {
    let Some(mut fx) = fx else { return };
    for burst in bursts.read() {
        ensure_model(&mut fx, &asset_server, &burst.path);
        commands.spawn((
            GroundFx::new(burst.path.clone(), false),
            Transform::from_translation(burst.pos),
            Visibility::default(),
        ));
    }
}

/// Emit shards: `rate` per second, each a free one-shot instance at a random offset inside the
/// wire radius (uniform over the disc, horizontal plane — the ref's per-particle spread scaled
/// by `+0x11c`).
pub(super) fn tick_shard_emitters(
    mut commands: Commands,
    time: Res<Time>,
    mut emitters: Query<(&mut ShardEmitter, &GlobalTransform)>,
) {
    for (mut em, tf) in &mut emitters {
        em.accum += em.rate * time.delta_secs();
        while em.accum >= 1.0 {
            em.accum -= 1.0;
            // xorshift* — cheap, per-emitter deterministic, no rand dependency.
            let mut next = || {
                em.rng ^= em.rng >> 12;
                em.rng ^= em.rng << 25;
                em.rng ^= em.rng >> 27;
                em.rng.wrapping_mul(0x2545F4914F6CDD1D)
            };
            let u1 = (next() >> 40) as f32 / (1u64 << 24) as f32;
            let u2 = (next() >> 40) as f32 / (1u64 << 24) as f32;
            let r = em.radius * u1.sqrt();
            let theta = u2 * std::f32::consts::TAU;
            let offset = Vec3::new(r * theta.cos(), 0.0, r * theta.sin());
            commands.spawn((
                GroundFx::new(em.path.clone(), false),
                Transform::from_translation(tf.translation() + offset),
                Visibility::default(),
            ));
        }
    }
}

/// MONKEY (area spell light): how far ABOVE OR BELOW a live area pool's own height an impact may
/// still be deduped against it (yd).
///
/// The footprint test itself is HORIZONTAL, because that is what `DYNAMICOBJECT_RADIUS` describes —
/// a disc of ground. A vertical window is needed anyway so the disc does not become an infinite
/// column: a Rain of Fire on the bridge above must not silence the impacts under it. Generous
/// enough to cover the lift the pool hangs at plus a shard landing on a slope, tight enough to be a
/// storey.
const AREA_DEDUPE_HEIGHT: f32 = 10.0;

/// Is `at` (Bevy world space) inside a live area pool's footprint? See [`AREA_DEDUPE_HEIGHT`].
fn inside_a_live_area(
    at: Vec3,
    areas: &Query<(&GlobalTransform, &super::spell_fx::AreaSpellLight)>,
) -> bool {
    areas.iter().any(|(gt, area)| {
        let d = at - gt.translation();
        d.y.abs() <= AREA_DEDUPE_HEIGHT && d.x * d.x + d.z * d.z <= area.radius * area.radius
    })
}

/// Attach model parts to pending instances whose M2 finished building (the missile pattern —
/// free world models, ground-anchored so authored flat quads decal to the terrain), start the
/// one-shot clocks, and run both reapers (one-shot expiry; the loop-repeat override).
#[allow(clippy::type_complexity)]
pub(super) fn attach_ground_fx_models(
    mut commands: Commands,
    time: Res<Time>,
    // MONKEY (area spell light): `ChildOf` joins a VISUAL A instance to the DynamicObject anchor
    // that owns it — where the plan is, and the entity the light must hang on so it dies with the
    // area object. `Transform` is the instance's world point for a FREE one (a shard, a dest
    // one-shot), which is the dedupe's input.
    mut instances: Query<(
        Entity,
        &mut GroundFx,
        Option<&mut AnimationPlayer>,
        Option<&ChildOf>,
        &Transform,
    )>,
    // MONKEY (area spell light): the arm-time half of the area verdict, read through the anchor.
    plans: Query<&AreaLightPlan>,
    // MONKEY (area spell light): every live area pool, for the impact dedupe.
    areas: Query<(&GlobalTransform, &super::spell_fx::AreaSpellLight)>,
    fx: Option<ResMut<SpellFx>>,
    asset_server: Res<AssetServer>,
    mut wow_materials: ResMut<Assets<benilla_assets::materials::WowModelMaterial>>,
    mut tint_reg: ResMut<super::spell_fx::FxTintAnims>,
    mut uv_reg: ResMut<benilla_world::doodad_anim::UvAnimMaterials>,
    mut anim_table: ResMut<benilla_world::mat_anim_table::MatAnimTable>,
    ibps: Res<Assets<bevy::mesh::skinning::SkinnedMeshInverseBindposes>>,
    mut palettes: ResMut<benilla_world::rig_palette::RigPalettes>,
) {
    let Some(mut fx) = fx else { return };
    let now = time.elapsed_secs();
    for (entity, mut inst, player, parent, at) in &mut instances {
        if !inst.spawned {
            ensure_model(&mut fx, &asset_server, &inst.path);
            let Some(dm) = fx.models.get(&inst.path) else {
                continue; // unreachable — just inserted
            };
            if !attach_effect_visuals(
                &mut commands,
                entity,
                dm,
                now,
                true, // a dest-anchored area model's flat quads ARE ground decals
                // A free world model standing at the point, chained to nothing: its trail stays
                // world-frozen and its pool finishes in place when the effect ends.
                super::spell_fx::EffectHost::default(),
                // The dest one-shot is not a `CEffect` on a unit: it plants at the packet's point and
                // runs its own span clock, so it keeps the plain single-clip arm.
                None,
                &mut FxMaterials {
                    store: &mut wow_materials,
                    tint: &mut tint_reg,
                    uv: &mut uv_reg,
                    table: &mut anim_table,
                },
                &ibps,
                &mut palettes,
                None,
            ) {
                continue; // model still building — attach on a later pass
            }
            inst.spawned = true;
            // MONKEY (area spell light): a VISUAL A on an armed anchor is a persistent ground
            // effect, and takes the AREA lane instead of the flash below — one light for the
            // DynamicObject's whole life, hung on the ANCHOR (not on this instance) so the
            // server's destroy is what ends it. `area_light_kind` finishes the verdict the plan
            // started, and its refusals are honoured: when it says dark, this instance gets no
            // light at all rather than falling through to the flash.
            let plan = parent.map(|c| c.parent()).and_then(|anchor| {
                plans
                    .get(anchor)
                    .ok()
                    .map(|plan| (anchor, plan))
                    .filter(|_| inst.looping)
            });
            if let Some((anchor, plan)) = plan {
                let kind = benilla_formats::area_light_kind(
                    &inst.path,
                    plan.school,
                    dm.lights.iter().find_map(|l| l.spell.map(|fx| fx.kind)),
                );
                super::spawn_area_spell_light(
                    &mut commands,
                    &dm.lights,
                    kind,
                    anchor,
                    plan.radius,
                );
                continue; // no one-shot clock and no flash — the area object owns both ends
            }
            // MONKEY (spell light): the IMPACT flash. A dest-anchored effect is the one lane whose
            // light must not hold — a Fire Nova lights the ground it lands on and is gone. The
            // burst span is the model's own first-sequence duration where it authors one (so the
            // light fades with the effect rather than on a guess), the shared default otherwise;
            // a LOOPING plant (a persistent ground aura) takes the default too, because "as long
            // as the aura lasts" is exactly what a burst must not be.
            //
            // MONKEY (area spell light): …unless it lands INSIDE a live area pool, in which case it
            // throws nothing. Rain of Fire emits a shard 5× a second from this very model, each of
            // which used to spawn its own 0.6 s flash: 5 lights/s stacking on one patch of ground,
            // strobing, and — before the budget's area exemption — evicting the standing pool they
            // were landing in. The area light already IS the light of that ground; a second one per
            // impact adds nothing but the flicker.
            if !inside_a_live_area(at.translation, &areas) {
                let span = dm
                    .first_seq_span
                    .filter(|_| !inst.looping)
                    .unwrap_or(super::spell_fx::SPELL_BURST_SPAN);
                super::spawn_spell_light(
                    &mut commands,
                    &dm.lights,
                    entity,
                    super::spell_fx::SpellLightMode::Burst { span },
                );
            }
            if !inst.looping {
                // One pass of the first sequence — the kit pipeline's completion-callback
                // stand-in (`spell_fx`'s span clock, same law).
                let span = dm.first_seq_span.unwrap_or(super::spell_fx::FALLBACK_SPAN);
                inst.expires = Some(now + span);
            }
            continue; // the AnimationPlayer lands next frame — the loop override waits
        }
        // The loop override: the ref re-fires the sequence on completion (`0x5d5580`); ours
        // marks the live player's animations repeat-forever, once.
        if inst.looping && !inst.loop_armed {
            if let Some(mut player) = player {
                for (_, anim) in player.playing_animations_mut() {
                    anim.repeat();
                }
                inst.loop_armed = true;
            }
        }
        if let Some(expires) = inst.expires {
            if now >= expires {
                commands.entity(entity).despawn();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `0x5d55c0` decode: `bits(f32(param0 + 512.0)) >> 14 & 0xff` recovers the small int
    /// exactly (the real rows carry 0.0 → Blizzard, 1.0 → Rain of Fire), and out-of-table data
    /// clamps instead of reading past the 7 entries (the client's own missing bounds check —
    /// wow-re trap #2).
    #[test]
    fn shard_model_index_decodes_and_clamps() {
        assert_eq!(shard_model_index(0.0), 0);
        assert_eq!(shard_model_index(1.0), 1);
        assert_eq!(shard_model_index(6.0), 6);
        assert_eq!(shard_model_index(7.0), 6, "clamped, not read past");
        assert_eq!(shard_model_index(200.0), 6);
        assert_eq!(SHARD_MODELS[0], "Spells\\Blizzard_Impact_Base.mdx");
        assert_eq!(SHARD_MODELS[1], "Spells\\RainOfFire_Impact_Base.mdx");
    }
}
