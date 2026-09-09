//! **Carried M2 lights** — the dynamic point lights an *entity* brings into the world, as opposed to
//! the ones a placed ADT doodad / WMO prop brings (`benilla_world::terrain_stream`'s `spawn_lights_for`).
//!
//! The law is one law. `0x718960` runs per frame over **every** CM2Model the scene draws — a placed
//! doodad, a creature, a GameObject, and (recursing at `7191b9`/`719286`) each attached child model —
//! gathers that model's own `type==1` light blocks, transforms each def position by its **live bone
//! matrix**, and registers the result into the world scene's light DB (`0x71b650` → `0x71bb60`). Every
//! lit surface then selects its ≤3 nearest from that same DB (decisions 0016/0273/0285). Nothing in the
//! chain distinguishes "prop" from "unit": a torch is a torch whether it is staked in the ground or
//! held in a hand.
//!
//! benilla had implemented only the placed half, so a torch-bearing NPC carried a flame that lit
//! nothing — the director's report from Westfall (Remy "Two Times", whose `Club_1H_Torch_A_01.m2`
//! authors exactly one warm point light) is the reference doing the other half: the fence rails and
//! the grass around him light up.
//!
//! **The bone ride is the whole reason this isn't just the placed spawner again.** A placed prop's
//! light bone never moves, so the rest pose is exact and the light can be baked to a world point. An
//! entity's does move — the hand swings — so each light is spawned as a **child of its host bone's
//! joint entity** with the def position rebased into that bone's frame (`position − bone_pivot`),
//! exactly as the emitters and ribbons ride (0130 phase 4). Bevy's transform propagation then walks
//! the light through the animation for free, and the per-frame light packer
//! ([`benilla_world::lighting`]) reads its `GlobalTransform` like any other point light.

use benilla_assets::coords::wow_to_bevy;
use benilla_assets::ModelLight;
use bevy::prelude::*;

use benilla_world::interior::{InteriorAnchor, WmoResidency};
use benilla_world::lighting::{
    LightLane, LightLitRooms, LightRooms, ShadowProxyLight, SyntheticFireLight,
};
use benilla_world::terrain_stream::{carried_light_claims, point_light, CarriedClaimSet};
use benilla_world::wmo_portal::{WmoGroupVis, WmoPortalInstance, WmoRoom};

/// Spawn a `PointLight` child for each **casting** (`type==1`, not visibility-gated dark) M2 light of
/// an entity's model.
///
/// `joint` resolves a light's host bone index to the instance's live joint entity — `None` for a
/// boneless/skeleton-less instance (a held item spawns no skeleton; its `root` already *is* the
/// item's model frame), a `-1` bone, or a bone the instance doesn't carry. A light with a joint rides
/// it in bone-local space; a light without one hangs off `frame` in plain model space, which is the
/// exact rest-pose special case.
///
/// Children, not free entities: the light's lifecycle and its frame both come from the hierarchy, so
/// a gear change, a despawn, or a mount transition takes its lights with it.
pub(super) fn spawn_carried_lights(
    commands: &mut Commands,
    lights: &[ModelLight],
    frame: Entity,
    joint: impl Fn(i16) -> Option<Entity>,
) {
    for l in lights {
        if !l.def.casts() {
            continue; // directionals feed an ambient term; a static `0` visibility key is dark
        }
        // MONKEY (carried light stability): a SYNTHESISED light (`fire_light` — the imp's hand
        // flames, a brazier GameObject, a fire elemental) does NOT ride its host bone; an
        // AUTHORED one still does.
        //
        // The distinction is what the position MEANS. An authored light block is a fixture an
        // artist put on a bone, and `0x718960` re-registering it through the live bone matrix is
        // the reference's own behaviour — the held torch that tracks the swinging hand, and that
        // case takes the boneless arm below anyway (a held item spawns no skeleton). A synthesised
        // one is a stand-in for a whole flame VOLUME whose position we GUESSED off the emitter
        // record: it is already approximate to within the flame's own size, and riding a hand bone
        // through an idle animation adds high-frequency motion that every consumer downstream
        // reads as an EVENT rather than as a wobble. The per-vertex nearest-3 in `point_light_sum`
        // re-ranks; and the cube-shadow cache tripped its 0.1 yd staleness test
        // (`torch_shadow::map_publishable`) every single frame and withdrew its map, which
        // `TorchTable::pack` answers by forcing that slot's cross-fade weight to 0 with no ramp —
        // a full-strength shadow toggling at frame rate. That is the reported "the imp produces a
        // DISCO with its light".
        //
        // The rest pose is identity about the pivot ([`ModelLight::bone_pivot`]), so hanging the
        // light off `frame` at the raw model-space `def.position` puts it EXACTLY where the bone
        // ride puts it in the rest pose — and `frame` is the model root, so the light still
        // follows the unit's own motion with zero lag. Only the animation is dropped.
        let host = if l.synthetic { None } else { joint(l.def.bone) };
        let (parent, local) = match host {
            Some(j) => (
                j,
                [
                    l.def.position[0] - l.bone_pivot[0],
                    l.def.position[1] - l.bone_pivot[1],
                    l.def.position[2] - l.bone_pivot[2],
                ],
            ),
            None => (frame, l.def.position),
        };
        let mut glow = commands.spawn((
            point_light(l.def.diffuse_color, l.def.diffuse_intensity),
            Transform::from_translation(wow_to_bevy(local)),
            Visibility::default(),
        ));
        // MONKEY (fire GO lights): a light we DERIVED from the model's flame emitter is tagged, so
        // the packer can ride the live `fireLightGain` over it (and only over it). The entity lanes
        // take synthetic lights: a placed brazier GameObject, a fire elemental, a flaming helm have
        // no MOLT fixture standing beside them the way a WMO's own wall torches do.
        if l.synthetic {
            glow.insert(SyntheticFireLight);
        }
        let glow = glow.id();
        commands.entity(parent).add_child(glow);
    }
}

/// MONKEY (fire GO lights): how far up the hierarchy the room walk looks before giving up. A
/// carried light's anchor is at most `light → joint → … → net entity`, and a rig's joint chain is
/// the deep part; 64 clears every shipped skeleton with room to spare, and the cap is there so a
/// malformed cycle can never hang the frame.
const ANCHOR_WALK_DEPTH: usize = 64;

/// MONKEY (carried light stability): the bearer crossfade weight at which a carried light COMMITS
/// to its owner's room. [`InteriorAnchor::lane`] ramps at 2/s, so crossing this midpoint needs
/// ~0.25 s of a HELD verdict; one frame of a wrong law moves the weight by `dt * 2` and can never
/// cross it. That IS the hysteresis — no second timer, and because it is the very signal the
/// bearer's own meshes crossfade their lighting on, the light changes lane with the body.
const LANE_INTERIOR: f32 = 0.5;

/// MONKEY (carried light stability): how far a carried light may drift and still count as standing
/// still. Deliberately HALF `torch_shadow`'s 0.1 yd cache-staleness radius: the settle test has to
/// be the stricter of the two, or a light could qualify as a shadow caster while already sitting
/// far enough from its baked map to have that map withdrawn again on the next step.
const STILL_DRIFT: f32 = 0.05;

/// MONKEY (carried light stability): how long a carried light must hold still before it may cast.
/// Long enough that a pet fidgeting between waypoints never qualifies, short enough that a placed
/// brazier GameObject is eligible well inside the first second it exists.
const STILL_HOLD: f32 = 0.75;

/// MONKEY (carried light stability): how long a carried light has been standing still, so the
/// torch-shadow lane can refuse a MOVING one.
///
/// The cube-shadow cache is built for fixtures that do not move: a slot keeps its baked depth only
/// while its fixture stays within 0.1 yd of where the map was baked
/// (`torch_shadow::map_publishable`), and a slot whose map is withdrawn has its cross-fade weight
/// forced to 0 with no ramp (`static_gx::torch_depth`'s `TorchTable::pack`). A walking bearer
/// crossed that threshold every frame, which is a full-strength shadow blinking at frame rate —
/// and worse, it burned the lane's global two-rebuilds-a-frame budget re-baking a mesh that was
/// stale again immediately, starving the STATIC candles around it so those flickered too. One
/// pet's shadow is not worth that.
///
/// A carried light is not excluded outright, because the same `ChildOf` spawn path carries the
/// stationary cases the shadow lane was written for: a GM-placed brazier GameObject in the Lion's
/// Pride Inn, a campfire. Those settle within [`STILL_HOLD`] and cast normally; only things that
/// actually move are refused. Refusal is a *candidacy* verdict, so an incumbent that starts
/// walking is cross-faded out by the lane's own eviction path rather than cut.
#[derive(Component)]
pub(crate) struct CarriedLightMotion {
    /// World position at the last frame the light was judged to have MOVED. Not re-anchored while
    /// it is still, so a slow creep accumulates and eventually trips the test — which is right,
    /// because the shadow cache's own staleness test is likewise absolute from the bake point.
    last: Vec3,
    /// Seconds held within [`STILL_DRIFT`] of `last`, capped at [`STILL_HOLD`]. The cap is what
    /// keeps a settled light a pure READ: no field write, so no change tick, so nothing
    /// downstream is woken by a light that is doing nothing.
    still_for: f32,
}

impl CarriedLightMotion {
    /// Has this light held still long enough to be trusted with a cached cube shadow?
    pub(crate) fn settled(&self) -> bool {
        self.still_for >= STILL_HOLD
    }
}

/// MONKEY (carried light stability): maintain [`CarriedLightMotion`] for every carried light.
///
/// `PostUpdate`, **after transform propagation**, for the same reason the packer is there: a
/// carried light is a child of a moving parent, so its `GlobalTransform` is only this frame's once
/// `Propagate` has run. `update_torch_shadows` reads the verdict in `Last`, i.e. later in the same
/// frame — never one stale.
pub(crate) fn track_carried_light_motion(
    mut commands: Commands,
    time: Res<Time>,
    mut lights: Query<
        (Entity, &GlobalTransform, Option<&mut CarriedLightMotion>),
        (With<PointLight>, With<ChildOf>, Without<ShadowProxyLight>),
    >,
) {
    let dt = time.delta_secs();
    for (light, gt, motion) in &mut lights {
        let p = gt.translation();
        match motion {
            Some(mut m) => {
                if p.distance_squared(m.last) > STILL_DRIFT * STILL_DRIFT {
                    m.last = p;
                    m.still_for = 0.0;
                } else if m.still_for < STILL_HOLD {
                    m.still_for += dt;
                }
            }
            // Born moving: a light that spawns mid-stride must not be eligible on its first frame
            // merely because it has no history yet.
            None => {
                commands.entity(light).try_insert(CarriedLightMotion {
                    last: p,
                    still_for: 0.0,
                });
            }
        }
    }
}

/// The bearer state a carried light's claim is derived from — the room its [`InteriorAnchor`]
/// names and that anchor's crossfade WEIGHT — walked up the hierarchy from the light itself,
/// because the light hangs off a BONE JOINT (or the model frame) and the anchor is the model's
/// net-entity root several links above it (`interior::part_interior_lit`: "`anchor` is the model's
/// NET ENTITY root for every caller").
fn owner_lane(
    light: Entity,
    parents: &Query<&ChildOf>,
    anchors: &Query<&InteriorAnchor>,
) -> (Option<WmoRoom>, f32) {
    let mut e = light;
    for _ in 0..ANCHOR_WALK_DEPTH {
        if let Ok(anchor) = anchors.get(e) {
            return (anchor.room(), anchor.lane());
        }
        match parents.get(e) {
            Ok(c) => e = c.parent(),
            Err(_) => break,
        }
    }
    // No anchor above us at all (a bearer with neither a lit part nor a lit emitter): exterior,
    // and settled there — the same answer the old walk's `None` produced.
    (None, 0.0)
}

/// The room a light's [`LightRooms`] claim currently stands for — our own record of it, because
/// `LightRooms` wraps a `WmoGroupVis` whose fields are benilla-world-private and cannot be read
/// back for a comparison. Absent = never claimed.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ClaimedRoom {
    /// What the light's `LightRooms` / `LightLane` currently stand for. `None` = exterior lane.
    applied: Option<WmoRoom>,
    /// MONKEY (carried light stability): the last room the bearer's anchor actually NAMED. Held
    /// across the exterior ramp: the frame a bearer's law flips to `Exterior` its `room` goes
    /// `None` at once, while the lane weight the claim is judged on takes ~0.25 s to fall past
    /// [`LANE_INTERIOR`]. Without this there would be no room left to keep claiming during that
    /// window, and the light would drop to the exterior lane on the raw flip — exactly the
    /// one-frame verdict the weight exists to smooth.
    seen: Option<WmoRoom>,
    /// MONKEY (GO room claims): the [`LightLane`] currently on the light. The lane used to be a
    /// pure function of `applied` (`is_some()`), so it needed no record; it is now also a function
    /// of the CLAIM SET (a light standing in an exterior-class group that claims a real room moves
    /// onto the interior lane — see [`CarriedClaims`]), and those two move independently. Recorded
    /// so the `LightLane` insert stays change-gated: an insert is an archetype write, and the
    /// render extraction behind it wakes on every one.
    lane: bool,
}

/// MONKEY (GO room claims): how far a carried light may drift (yd) before its claim set is
/// rebuilt. Ten times [`STILL_DRIFT`] on purpose — the settle test is about whether a CACHE of the
/// light's surroundings is still valid frame to frame, this one is about whether a set of ROOMS is
/// still the right set, and a room is yards across. Half a yard is well inside the smallest shipped
/// doorway, so a light cannot cross a portal without tripping it.
const CLAIM_DRIFT: f32 = 0.5;

/// MONKEY (GO room claims): a carried light's [`LightLitRooms`] claim set, plus the three inputs
/// the rebuild is gated on. Our own record, for the same reason [`ClaimedRoom`] is one:
/// `LightLitRooms` wraps a crate-private `WmoGroupVis` and cannot be read back for a comparison.
#[derive(Component)]
pub(crate) struct CarriedClaims {
    /// The set currently applied — `None` when the light carries no claims at all (outdoors, or
    /// moving, or standing in no building's authored box).
    set: Option<CarriedClaimSet>,
    /// The world position the set was decided at. Absolute, not re-anchored: a light that creeps
    /// out of the room it claimed must eventually rebuild, and the drift budget is the room's.
    at: Vec3,
    /// The [`WmoResidency`] generation it was decided under — a light that claimed nothing because
    /// its building had not streamed in yet has to be re-asked once it has, and the reverse (the
    /// placement entity a claim is keyed to is gone) is a stale instance id in the GPU claim head.
    generation: u32,
    /// The bearer room it was decided under: the light walked through a door, so the claim seed
    /// (and the preferred placement) moved even if the light itself barely did.
    room: Option<WmoRoom>,
}

impl CarriedClaims {
    /// MONKEY (GO room claims): **the recompute gate.** `true` = the applied set still stands for
    /// this position, this residency and this bearer room, so the whole claim walk is skipped.
    ///
    /// Split out as a pure function so it can be tested without a world: the walk it guards is the
    /// expensive half of this system (an O(groups) box scan per resident placement), and a gate
    /// that silently stopped firing would show up not as a wrong picture but as a stale one.
    fn fresh(&self, at: Vec3, generation: u32, room: Option<WmoRoom>) -> bool {
        self.generation == generation
            && self.room == room
            && self.at.distance_squared(at) <= CLAIM_DRIFT * CLAIM_DRIFT
    }
}

/// MONKEY (GO room claims): does this claim set justify moving the light onto the INTERIOR lane?
///
/// Two conditions, and the second is the safety half.
///
/// 1. **It must name an interior-class group.** Claims are read by exactly one function,
///    `static_gx.wgsl`'s `interior_room_light`, and that function skips every table entry whose
///    colour-row `.w < 0.5` — i.e. every EXTERIOR-lane light. So on the exterior lane a claim set
///    is inert: it is packed as the UNGATED head (`build_light_data` only calls `RoomClaim::build`
///    on the interior arm) and no shader ever looks at it. A light that claims a real room and
///    stays exterior has therefore gained nothing, which is the whole reason the lane moves.
/// 2. **Every claim must be exterior-lane eligible** (no `LIT_ROOM_EXT_DENY` bit). Moving onto the
///    interior lane is a TRADE: the light leaves `point_light_sum` (terrain, doodads, NPCs) and
///    `wmo_exterior_point_sum` (a WMO's exterior-class walls and cobbles) and joins
///    `interior_room_light`, which reaches an exterior-class surface only through the strict
///    exterior-batch room term — and that term admits a claim only when it carries `CLAIM_EXT_OK`.
///    A district-scale shell claim (Orgrimmar's whole valley, Stormwind's districts) carries the
///    deny bit, so a light standing in one would be taken off the very cobbles it lights today and
///    given nothing back that reached them. Those keep the exterior lane and behave exactly as
///    before; a BUILDING-scale exterior group (the Goldshire inn's shell, the canyon nook a shop
///    door opens onto) is eligible, so its floor keeps the light through the strict term while the
///    interior rooms one portal hop away gain it.
///
/// The trade the rule still accepts, stated plainly: a brazier that moves onto the interior lane
/// stops feeding the nearest-3 exterior term, so ADT TERRAIN under it and the NPCs standing beside
/// it lose that light and gain the room lane instead. That is the correct side for a fixture
/// genuinely inside a building's claimable volume — it is what a MOLT fixture in the same spot
/// already does — and condition 2 is what keeps it from happening out on a district-scale street.
fn lane_worthy(set: &CarriedClaimSet) -> bool {
    set.any_interior && set.all_ext_ok()
}

/// MONKEY (GO room claims): the claim set of a light at `at`, from the first resident WMO placement
/// whose authored MOGI boxes hold it.
///
/// `prefer` is tried in order before the scan — the bearer's own anchor placement and the one the
/// light claimed last time. In the steady state one of those two always wins on the first probe,
/// which is what keeps a residency-generation tick (a building streaming in three streets away
/// re-gates every carried light in the world) from turning into a full scan per light. The scan is
/// the cold path: a GameObject standing in an EXTERIOR-class group has no anchor room at all, so
/// its very first claim has nowhere else to come from.
fn claims_at(
    wmos: &Assets<benilla_assets::WmoModel>,
    placements: &Query<(Entity, &WmoPortalInstance)>,
    prefer: [Option<Entity>; 2],
    at: Vec3,
    reach: f32,
) -> Option<CarriedClaimSet> {
    let one = |e: Entity| {
        let (e, inst) = placements.get(e).ok()?;
        carried_light_claims(wmos.get(&inst.handle)?, e, inst.world_from_local, at, reach)
    };
    for e in prefer.into_iter().flatten() {
        if let Some(set) = one(e) {
            return Some(set);
        }
    }
    placements.iter().find_map(|(e, inst)| {
        if prefer.contains(&Some(e)) {
            return None; // already probed above
        }
        carried_light_claims(wmos.get(&inst.handle)?, e, inst.world_from_local, at, reach)
    })
}

/// MONKEY (fire GO lights): give a CARRIED light the room its owner was classified into.
///
/// A world-baked light gets its rooms at spawn from the building's own MODR/MOLR tables. An
/// entity's light cannot: a GameObject walks (or is placed) into a building the tables know nothing
/// about, and its room is only ever known from the interior classifier's down-ray. So this system
/// copies that verdict onto the light, and two things follow from it:
///
/// - **The faithful gate.** A torch in a portal-culled room stops lighting the hillside outside it
///   ([`LightRooms`], decision 0689) — until now a placed brazier GameObject lit through walls
///   because it claimed no room at all and `room_admits` admits `None` unconditionally.
/// - **Shadow eligibility.** `torch_shadow`'s caster query is `With<PointLight>, With<LightRooms>`,
///   so a roomless light can never be promoted to a cube-map caster however deep indoors it stands.
///   A GM-placed brazier in the Lion's Pride Inn now throws real shadows like a MOLT fixture.
///
/// Applies to synthetic and authored carried lights alike — the room is a fact about the OWNER,
/// not about where the light's colour came from.
///
/// MONKEY (carried light stability): the verdict is taken from the anchor's crossfade WEIGHT, not
/// from its raw per-frame law. The law alternates: `WOW_INTERIOR_LOG` from the 2026-09-09 run has
/// anchor `320v2` resolving `matte`↔`bake` on ten consecutive frames (t 73.75 … t 74.14), and
/// the same threshold noise puts an `Exterior` between two interior verdicts at a doorway. A light
/// that re-decided on that raw answer would hand itself back and forth between two DIFFERENT
/// consumer families every frame — the exterior `point_light_sum` and the interior room lane —
/// which is an on/off strobe of the pool, not a subtle difference. Reading `lane >= LANE_INTERIOR`
/// costs nothing and cannot be crossed by a blip (see [`LANE_INTERIOR`]).
///
/// MONKEY (GO room claims): **and it now builds the light's [`LightLitRooms`] too** — the rooms it
/// may LIGHT, as opposed to the one it must be VISIBLE from. The bearer's anchor answers "which
/// group is this standing in", singular, and one group was the whole gate: a brazier inside the
/// Lion's Pride Inn lit its own room and stopped dead at the doorway, and a brazier in one of
/// Orgrimmar's 123 canyon placements named no group at all (`MOGP & 0x48` ⇒ the anchor is `None`)
/// so it could never reach the shops the canyon opens onto. The fix is not a second classifier: it
/// is the SAME claim rule the WMO's own MOLT fixtures and MODD props run
/// (`benilla_world::terrain_stream::carried_light_claims` → `benilla_formats::room_claims`:
/// containment tightest-first, then one portal hop within reach), asked at the light's world
/// position. Reach is the M2 intensity bucket scaled by the DEFAULT `interiorAttenScale`, exactly
/// as the prop lane sizes its own hop — the live cvar must not be able to open a doorway that was
/// gated when the set was built, or every claim in the world would rebuild on a slider drag.
///
/// **Gated on four things**, because the walk is an O(groups) box scan per resident placement and
/// Orgrimmar has ~150 of these lights: the light must be SETTLED
/// ([`CarriedLightMotion::settled`] — a walking pet's light re-claims no more often than the settle
/// cadence, and while it is unsettled and off its build point it carries NO claims, i.e. exactly
/// the single-anchor-room behaviour it had before this change), and the set is rebuilt only when
/// the light drifts past [`CLAIM_DRIFT`], when [`WmoResidency`] ticks, or when the bearer's room
/// changes ([`CarriedClaims::fresh`]).
///
/// Every component write below is change-gated on [`ClaimedRoom`]/[`CarriedClaims`]: a settled
/// light is one hierarchy walk and three compares per frame, and nothing is written, so the light's
/// archetype (and the render extraction behind it) stays quiet while an NPC stands still in a room.
#[allow(clippy::type_complexity)]
pub(crate) fn claim_carried_light_rooms(
    mut commands: Commands,
    // Carried lights only — `With<ChildOf>` IS the distinction: a world-baked doodad/MOLT light
    // spawns as a free root with its rooms already correct, and re-deciding them from a hierarchy
    // it isn't in would clear them.
    mut lights: Query<
        (
            Entity,
            // MONKEY (GO room claims): the light's own WORLD position (the claim rule's input) and
            // its intensity (which sizes the portal hop). `Update` reads last frame's propagation,
            // which is this frame's answer for anything settled — and only settled lights claim.
            &GlobalTransform,
            &PointLight,
            Option<&CarriedLightMotion>,
            Option<&mut ClaimedRoom>,
            Option<&mut CarriedClaims>,
        ),
        (With<ChildOf>, Without<ShadowProxyLight>),
    >,
    parents: Query<&ChildOf>,
    anchors: Query<&InteriorAnchor>,
    // MONKEY (GO room claims): the resident placements + their models — the claim rule's tables.
    placements: Query<(Entity, &WmoPortalInstance)>,
    wmos: Res<Assets<benilla_assets::WmoModel>>,
    residency: Res<WmoResidency>,
) {
    let generation = residency.generation();
    for (light, gt, pl, motion, mut claimed, mut claims) in &mut lights {
        let (room, lane) = owner_lane(light, &parents, &anchors);
        // The room to claim IF we are on the interior lane: the one the anchor names this frame,
        // else the last one it named (the ramp-down window described on `ClaimedRoom::seen`).
        let seen = room.or_else(|| claimed.as_deref().and_then(|c| c.seen));
        let want = seen.filter(|_| lane >= LANE_INTERIOR);

        // MONKEY (GO room claims): the LIT-room claim set, rebuilt only on the gates above.
        // `None` below means "keep what is already applied"; `Some(x)` means "apply x".
        let at = gt.translation();
        let fresh = claims
            .as_deref()
            .is_some_and(|c| c.fresh(at, generation, want));
        let set = if fresh {
            None
        } else if motion.is_some_and(CarriedLightMotion::settled) {
            // The prop lane's own sizing: the intensity bucket (`PointLight` premultiplied 4π at
            // spawn, so this inverts it exactly) through the DEFAULT atten scale.
            let reach = benilla_formats::room_claim::claim_reach(
                benilla_formats::room_claim::m2_light_reach(
                    pl.intensity / (4.0 * std::f32::consts::PI),
                ),
            );
            let prefer = [
                want.map(|r| r.instance),
                claims
                    .as_deref()
                    .and_then(|c| c.set.as_ref())
                    .map(|s| s.instance),
            ];
            Some(claims_at(&wmos, &placements, prefer, at, reach))
        } else {
            // Moved off its build point and not standing still: drop to the pre-claims behaviour
            // rather than assert a set for a position it was not measured at.
            Some(None)
        };
        // The set that stands THIS frame: the one just decided, else the one already applied. Read
        // from the local rather than back off the component, because on a light's first frame the
        // component is still a queued command and re-reading it would see `None` and remove the
        // very `LightLitRooms` this pass is inserting.
        let had = claims.as_deref().and_then(|c| c.set.as_ref());
        let next_set: Option<&CarriedClaimSet> = match &set {
            Some(next) => next.as_ref(),
            None => had,
        };
        let set_changed = match (next_set, had) {
            (None, None) => false,
            // MONKEY (soft portal claims): the FADES are part of the set. A light that drifts
            // across a room keeps claiming the same groups while the reach it has left past each
            // doorway shrinks — comparing ids alone would leave it faded by the distances it had
            // at its first build point, for as long as it never changed rooms.
            (Some(next), Some(had)) => {
                next.instance != had.instance
                    || next.groups != had.groups
                    || next.fades != had.fades
            }
            _ => true,
        };
        // MONKEY (GO room claims): the lane is the OR of the two halves — the bearer's own room,
        // and (new) a claim set that earns the interior lane on its own (see [`lane_worthy`]).
        let want_lane = want.is_some() || next_set.is_some_and(lane_worthy);

        // What the light already carries, so each insert below is earned.
        let (had_room, had_lane) = claimed
            .as_deref()
            .map_or((None, None), |c| (Some(c.applied), Some(c.lane)));

        let mut e = commands.entity(light);
        // MONKEY (light lane by position): the carried lane's own lane verdict. A bearer's room
        // came from the interior classifier's ray at the bearer's own feet, so "my owner is in a
        // room" IS "I am physically inside one" — no second ray, and the static classifier
        // (`classify_light_lanes`) deliberately skips `With<ChildOf>` so the two never fight.
        // MONKEY (GO room claims): plus the claim half — a light standing in an EXTERIOR-class
        // group (so: no anchor room) that nonetheless claims a real interior room through a portal
        // has to be on the interior lane for those claims to be read at all (see [`lane_worthy`]).
        if had_lane != Some(want_lane) {
            e.insert(LightLane::carried(want_lane));
        }
        if had_room != Some(want) {
            match want {
                Some(room) => {
                    e.insert(LightRooms::new(WmoGroupVis::single(room.instance, room.group)));
                }
                // Back outdoors (or the anchor went away): drop the claim rather than leave a
                // stale room, which would gate the light on a building it has walked out of.
                None => {
                    e.remove::<LightRooms>();
                }
            }
        }
        if set_changed {
            match next_set {
                Some(next) => {
                    e.insert(next.component());
                }
                None => {
                    e.remove::<LightLitRooms>();
                }
            }
        }

        // Bookkeeping LAST — every verdict above borrows it. `seen` is a record, not a component:
        // updating it must not touch the light's COMPONENTS, or a bearer walking a corridor would
        // churn the archetype for nothing.
        match claimed.as_mut() {
            Some(c) => {
                let (applied, lane) = (c.applied, c.lane);
                c.set_if_neq(ClaimedRoom { applied, seen, lane });
                c.applied = want;
                c.lane = want_lane;
            }
            None => {
                e.try_insert(ClaimedRoom {
                    applied: want,
                    seen,
                    lane: want_lane,
                });
            }
        }
        if let Some(next) = set {
            let next = CarriedClaims {
                set: next,
                at,
                generation,
                room: want,
            };
            match claims.as_mut() {
                Some(c) => **c = next,
                None => {
                    e.try_insert(next);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use benilla_formats::M2Light;
    use std::time::Duration;

    fn light(light_type: u16, bone: i16, position: [f32; 3], visibility_off: bool) -> ModelLight {
        ModelLight {
            def: M2Light {
                light_type,
                bone,
                position,
                bone_z: [0.0, 0.0, 1.0],
                ambient_color: [1.0; 3],
                ambient_intensity: 0.0,
                diffuse_color: [0.466_666_7, 0.290_196_1, 0.133_333_34], // the real torch's warm orange
                diffuse_intensity: 3.0,
                attenuation_start: 1.388_889,
                attenuation_end: 2.222_222_3,
                visibility_off,
            },
            bone_pivot: [1.0, 0.0, 0.5],
            synthetic: false,
        }
    }

    /// GOLDEN — the carried-light spawn law. Only a **casting** light spawns (`type==1`, not held
    /// dark by a static `0` visibility key — wow-re `m2-dynamic-lights.md` §9.4, the shape 11 of
    /// the corpus's 85 point lights actually ship), it lands as a CHILD of its host bone's joint
    /// so the animation carries it, and its offset is the def position rebased into that bone's
    /// frame (`position − bone_pivot`, wow→bevy). Colour × intensity survives the `PointLight`
    /// round trip the packer inverts (`intensity/4π`).
    #[test]
    fn only_casting_lights_spawn_and_they_ride_their_bone() {
        let mut app = App::new();
        let frame = app.world_mut().spawn(Transform::IDENTITY).id();
        let joint = app.world_mut().spawn(Transform::IDENTITY).id();
        let lights = [
            light(1, 3, [2.0, 0.0, 1.5], false),  // casts, on bone 3
            light(0, 3, [2.0, 0.0, 1.5], false),  // directional → ambient term, never a GL light
            light(1, 3, [2.0, 0.0, 1.5], true),   // point but authored dark
            light(1, -1, [0.0, 0.0, 4.0], false), // casts, boneless → the frame itself
        ];
        app.world_mut().commands().queue(move |world: &mut World| {
            let mut q = world.commands();
            spawn_carried_lights(&mut q, &lights, frame, move |bone| {
                (bone == 3).then_some(joint)
            });
        });
        app.world_mut().flush();

        let mut spawned: Vec<(Entity, Vec3, Entity)> = app
            .world_mut()
            .query::<(Entity, &PointLight, &Transform, &ChildOf)>()
            .iter(app.world())
            .map(|(e, _, t, c)| (e, t.translation, c.parent()))
            .collect();
        // Spawn order, i.e. light-table order — `Entity`'s own `Ord` is not index-ascending.
        spawned.sort_by_key(|(e, ..)| e.index());
        assert_eq!(
            spawned.len(),
            2,
            "the directional and the dark one stay out"
        );

        // Bone-ridden: parented to the joint, offset rebased into the bone frame.
        assert_eq!(spawned[0].2, joint);
        assert_eq!(spawned[0].1, wow_to_bevy([1.0, 0.0, 1.0]));
        // Boneless (`-1`): hangs off the frame at plain model-space position — the rest-pose case
        // a held item always takes (it spawns no skeleton).
        assert_eq!(spawned[1].2, frame);
        assert_eq!(spawned[1].1, wow_to_bevy([0.0, 0.0, 4.0]));

        let pl = app
            .world()
            .entity(spawned[0].0)
            .get::<PointLight>()
            .unwrap();
        let lin = pl.color.to_linear();
        let recovered = pl.intensity / (4.0 * std::f32::consts::PI);
        assert!(
            (lin.red * recovered - 1.4).abs() < 1e-3,
            "colour × intensity survives the packing"
        );
    }

    /// GOLDEN — MONKEY (carried light stability). A SYNTHESISED light does NOT ride its host
    /// bone even when the instance carries that joint: it hangs off the model FRAME at the raw
    /// model-space position, i.e. exactly where the bone ride would put it in the REST pose. The
    /// frame is the model root, so the light still follows the unit with zero lag; what is dropped
    /// is the per-frame animation swing that made the imp's green pool strobe (it re-ranked the
    /// shaders' per-vertex nearest-3 and invalidated the cube-shadow cache every frame).
    #[test]
    fn a_synthetic_light_stays_on_the_rest_pose_pivot() {
        let mut app = App::new();
        let frame = app.world_mut().spawn(Transform::IDENTITY).id();
        let joint = app.world_mut().spawn(Transform::IDENTITY).id();
        let mut synth = light(1, 3, [2.0, 0.0, 1.5], false);
        synth.synthetic = true;
        let lights = [synth];
        app.world_mut().commands().queue(move |world: &mut World| {
            let mut q = world.commands();
            spawn_carried_lights(&mut q, &lights, frame, move |bone| {
                (bone == 3).then_some(joint)
            });
        });
        app.world_mut().flush();

        let spawned: Vec<(Vec3, Entity)> = app
            .world_mut()
            .query::<(&PointLight, &Transform, &ChildOf)>()
            .iter(app.world())
            .map(|(_, t, c)| (t.translation, c.parent()))
            .collect();
        assert_eq!(spawned.len(), 1);
        assert_eq!(spawned[0].1, frame, "the frame, never the joint");
        assert_eq!(
            spawned[0].0,
            wow_to_bevy([2.0, 0.0, 1.5]),
            "raw model-space position, NOT rebased by the bone pivot"
        );
        // The tag still rides it: the packer must keep folding `fireLightGain` over this one.
        assert!(app
            .world_mut()
            .query::<&SyntheticFireLight>()
            .iter(app.world())
            .next()
            .is_some());
    }

    /// GOLDEN — MONKEY (carried light stability). A carried light earns its cube-shadow
    /// eligibility by HOLDING STILL for [`STILL_HOLD`], and loses it the instant it moves further
    /// than [`STILL_DRIFT`]. This is what keeps a walking bearer out of `torch_shadow`'s candidate
    /// set: a moving fixture invalidates its cached depth map every frame, and a withdrawn map
    /// zeroes that slot's cross-fade weight with no ramp — a shadow blinking at frame rate.
    #[test]
    fn a_carried_light_casts_only_after_it_holds_still() {
        let mut app = App::new();
        app.insert_resource(Time::<()>::default());
        app.add_systems(Update, track_carried_light_motion);
        let bearer = app.world_mut().spawn(Transform::IDENTITY).id();
        let light = app
            .world_mut()
            .spawn((
                PointLight::default(),
                Transform::IDENTITY,
                GlobalTransform::IDENTITY,
                ChildOf(bearer),
            ))
            .id();

        // First sight of it: history exists, but nothing has been HELD yet.
        app.update();
        let settled = |app: &App, e: Entity| {
            app.world()
                .entity(e)
                .get::<CarriedLightMotion>()
                .is_some_and(CarriedLightMotion::settled)
        };
        assert!(!settled(&app, light), "born moving is not settled");

        // A second of standing still clears the hold.
        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(Duration::from_secs_f32(1.0));
        app.update();
        assert!(settled(&app, light), "held still past STILL_HOLD");

        // One stride and it is disqualified again, with no credit for the second it banked.
        *app.world_mut().entity_mut(light).get_mut::<GlobalTransform>().unwrap() =
            GlobalTransform::from_translation(Vec3::new(1.0, 0.0, 0.0));
        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(Duration::from_secs_f32(1.0 / 60.0));
        app.update();
        assert!(!settled(&app, light), "moved further than STILL_DRIFT");
    }

    /// GOLDEN — MONKEY (GO room claims). The claim-set RECOMPUTE GATE. Building a carried light's
    /// lit-room set is an O(groups) authored-box scan per resident WMO placement, and Orgrimmar
    /// alone places ~150 of these lights, so the walk must run on a change and never on a frame.
    /// All three inputs invalidate, and none of them alone is enough to be left out:
    /// residency (a building streamed in under a light that had claimed nothing), the bearer's room
    /// (it walked through a door, so the claim seed moved even though the light barely did), and
    /// drift past [`CLAIM_DRIFT`] (it is no longer where the set was measured).
    #[test]
    fn a_claim_set_is_rebuilt_only_on_residency_room_or_drift() {
        let room = |group| {
            Some(WmoRoom {
                instance: Entity::PLACEHOLDER,
                group,
            })
        };
        let held = CarriedClaims {
            set: None,
            at: Vec3::new(10.0, 2.0, -4.0),
            generation: 7,
            room: room(3),
        };
        assert!(
            held.fresh(held.at, 7, room(3)),
            "nothing moved: no walk this frame"
        );
        assert!(
            held.fresh(held.at + Vec3::X * (CLAIM_DRIFT * 0.9), 7, room(3)),
            "a wobble inside the drift budget still stands"
        );
        assert!(
            !held.fresh(held.at + Vec3::X * (CLAIM_DRIFT * 1.1), 7, room(3)),
            "past CLAIM_DRIFT the set was measured somewhere else"
        );
        assert!(
            !held.fresh(held.at, 8, room(3)),
            "a placement streamed in or out: the claim's instance key may be stale"
        );
        assert!(
            !held.fresh(held.at, 7, room(4)),
            "the bearer changed room"
        );
        assert!(
            !held.fresh(held.at, 7, None),
            "the bearer left the building"
        );
    }
}
