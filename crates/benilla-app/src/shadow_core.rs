//! The shared **shadow rig** — the framework both shadow lanes plug into.
//!
//! A realtime shadow needs three things: something to CAST (a lane's caster geometry), a LIGHT +
//! shadow map to cast INTO, and RECEIVERS that read the map and darken. This module owns the middle
//! part — one directional "sun" on a private render layer, that layer's membership on the world
//! camera, the shadow-map resource, the invisible proxy caster material, and the sun aiming — plus
//! the geometry-collection helpers the lanes share. The receivers live in benilla's own shaders.
//!
//! The two lanes ([`super::character_shadow`], [`super::world_shadow`]) are EQUAL and independent:
//! each is its own plugin that depends only on this core, never on the other. They coordinate with
//! the rig through two resources, with a deliberate one-frame handshake:
//! - [`ShadowDemand`] — each active lane ORs `true` in (in the [`ShadowSet::Lanes`] set); the rig
//!   reads it next frame to decide whether the sun should exist. A lane that is removed simply stops
//!   contributing demand — the core needs no edit.
//! - [`ShadowFrame`] — the rig publishes the shared per-frame facts (is it active, the caster
//!   material, the camera/light position, the sun direction, the two collection reaches) for the
//!   lanes to read when they build their casters.
//!
//! Why ONE shared light and not one per lane: benilla's custom receiver shaders sample a SINGLE
//! directional light (they assign, not accumulate, over the light loop). Two lights would double the
//! shadow-pass cost and the last would win the receiver. So the lanes share this rig and each merely
//! contributes its own caster geometry into the one map.
//!
//! The rig deliberately does NOT put a directional light on the normal world layer: that would make
//! every existing WoW material enter Bevy's shadow-prepass path, the source of the pipeline
//! corruption seen during the first experiment. Only the private-layer proxy casts.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::{NoFrustumCulling, RenderLayers};
use bevy::light::{
    CascadeShadowConfigBuilder, DirectionalLight, DirectionalLightShadowMap, NotShadowReceiver,
};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::pbr::{Material, MaterialPipeline, MaterialPlugin, MeshMaterial3d};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use benilla_assets::coords::wow_to_bevy;
use benilla_assets::materials::WowModelMaterial;
use benilla_formats::ModelBlend;
use benilla_world::billboard::BillboardCard;
use benilla_world::interact::PickMesh;
use benilla_world::lighting::WowLighting;
use benilla_world::model_render::{ModelKind, ModelPart, ShadowOccluder};
use benilla_world::rig_palette::{RigPalettes, RigPart, RigSkin};
use benilla_world::view::WorldCamera;

use crate::char_select::ClientState;

/// Layer 31 is deliberately private to this feature. Bevy supports 32 render layers; the normal
/// world is layer 0 and the UI/portrait layers occupy the low numbered slots.
pub(crate) const PLAYER_SHADOW_LAYER: usize = 31;

/// How far from the camera the shadow map RESOLVES (the single cascade's `maximum_distance`).
/// Collection reaches farther than this: a caster's own body may stand outside the cascade
/// while its shadow lands inside it — so an out-of-frustum tree still darkens ground in view.
const CASTER_RANGE: f32 = 80.0;

/// How far the camera may drift before a cached caster-collection pass must refresh. The collection
/// reach carries this as margin so coverage holds everywhere between refreshes.
pub(crate) const STATIC_REBUILD_STEP: f32 = 16.0;

/// The tallest COMMON caster the reach law budgets for (the big Elwynn/Duskwood tree class, in world
/// units). Not a clamp on what casts — only on how far past the resolve range collection hunts.
const MAX_CASTER_HEIGHT: f32 = 35.0;

/// Ceiling on the shadow-reach extension: a degenerate near-horizontal sun must not explode the
/// collection radius (and with it the rebuild cost) unbounded.
const SHADOW_REACH_CAP: f32 = 120.0;

/// How far the LIVE sun may drift from the last-written direction before the shadow light snaps to
/// it — about one game minute of travel. The basis is quantised because a light that rotates EVERY
/// frame defeats the cascade fit's texel snap and crawls every shadow edge at WoW map coordinates
/// (~9,300 units out). Held between snaps the basis is bit-stable; only the shadow basis steps.
const SUN_SNAP_RADIANS: f32 = 8.0e-4;

/// The daytime elevation band (radians above the horizon) the shadow sun is clamped to. The visible
/// celestial sun sweeps roughly −10°..+85° and dips below the horizon at night; unclamped that would
/// INVERT the shadow (a set sun's travel points upward) or DEGENERATE the `looking_to` basis at high
/// noon (travel nearly ∥ up). Pinning to ~18°..65° keeps shadows visible, right-way-up and bounded
/// while the AZIMUTH still tracks the real sun — so shadows sweep through the day ("moving sun").
const MIN_SHADOW_SUN_ELEVATION: f32 = 0.314; // ~18°
const MAX_SHADOW_SUN_ELEVATION: f32 = 1.134; // ~65°

fn shadow_reach_extension(sun_travel: Vec3) -> f32 {
    // `sun_travel` is the direction the light TRAVELS (downward): -y is the vertical drop.
    let down = -sun_travel.y;
    if down <= 1e-3 {
        return SHADOW_REACH_CAP;
    }
    let horizontal = Vec3::new(sun_travel.x, 0.0, sun_travel.z).length();
    (MAX_CASTER_HEIGHT * horizontal / down).min(SHADOW_REACH_CAP)
}

/// The collection law for a lane that can contribute a TALL caster (a doodad/WMO exile, or the whole
/// static world): resolve range + rebuild margin + the sun-dependent shadow reach.
pub(crate) fn static_collection_reach(sun_travel: Vec3) -> f32 {
    CASTER_RANGE + STATIC_REBUILD_STEP + shadow_reach_extension(sun_travel)
}

/// `WOW_SHADOW_TRACE=1` — log the caster population when it CHANGES, to attribute a vanished shadow
/// to a lane rather than the map/receiver.
pub(crate) fn shadow_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("WOW_SHADOW_TRACE").is_ok_and(|v| v != "0"))
}

fn sun_snap_due(held: Option<Vec3>, live: Vec3) -> bool {
    match held {
        None => true,
        Some(held) => held.angle_between(live) > SUN_SNAP_RADIANS,
    }
}

/// The shadow sun's TRAVEL direction (light propagation — points AWAY from the sun) from the visible
/// celestial TO-sun direction, elevation clamped to [`MIN_SHADOW_SUN_ELEVATION`,
/// `MAX_SHADOW_SUN_ELEVATION`]. Azimuth is preserved so shadows sweep with the real sun; only the
/// height is tamed so a set/zenith sun can't invert or degenerate the basis.
fn shadow_sun_travel(to_sun: Vec3) -> Vec3 {
    let s = to_sun.normalize_or_zero();
    if s == Vec3::ZERO {
        return Vec3::NEG_Z;
    }
    let horizontal = Vec3::new(s.x, 0.0, s.z);
    let bearing = if horizontal.length() > 1e-4 {
        horizontal.normalize()
    } else {
        Vec3::NEG_Z
    };
    let elevation = s
        .y
        .clamp(-1.0, 1.0)
        .asin()
        .clamp(MIN_SHADOW_SUN_ELEVATION, MAX_SHADOW_SUN_ELEVATION);
    let to_sun_clamped = bearing * elevation.cos() + Vec3::Y * elevation.sin();
    -to_sun_clamped
}

/// Marks a layer-31 caster mesh (either lane's). Kept for teardown/inspection symmetry.
#[derive(Component)]
pub(crate) struct ShadowCaster;

#[derive(Component)]
struct ShadowSun;

#[derive(Component)]
struct ShadowCameraLayer {
    had_layers: bool,
}

/// The two ordered phases of the shadow feature within `Last`: the rig runs FIRST (spawns/aims the
/// sun, publishes [`ShadowFrame`]), then every lane runs in [`ShadowSet::Lanes`] — reading the frame
/// to build its casters and ORing its demand back in for the rig to read next frame.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum ShadowSet {
    Rig,
    Lanes,
}

/// Each active lane ORs `true` here (in [`ShadowSet::Lanes`]); the rig reads it the next frame to
/// decide whether the sun should exist, then resets it. A removed lane just stops contributing —
/// the core needs no edit to notice a lane is gone.
#[derive(Resource, Default)]
pub(crate) struct ShadowDemand(pub bool);

/// The rig's per-frame publication for the lanes: whether the rig is live, the shared proxy caster
/// material, and the shared collection inputs (camera/light position, sun direction, and the two
/// reaches). A lane builds nothing until `active` and only with the `material` here.
#[derive(Resource, Default)]
pub(crate) struct ShadowFrame {
    pub active: bool,
    pub light_position: Vec3,
    pub sun_direction: Vec3,
    /// Reach for short casters (creatures/gameobjects — at most a body tall).
    pub entity_reach: f32,
    /// Reach for tall casters (doodads/WMO/static world) — carries the sun-dependent shadow reach.
    pub tall_reach: f32,
    pub material: Option<Handle<ShadowCasterMaterial>>,
}

/// The rig's own retained state — private to the core.
#[derive(Resource, Default)]
struct ShadowRigState {
    sun: Option<Entity>,
    /// The direction the sun was LAST GIVEN — the held shadow basis (see [`SUN_SNAP_RADIANS`]).
    sun_written: Option<Vec3>,
    /// The invisible proxy material shared by every lane's SOLID caster.
    material: Option<Handle<ShadowCasterMaterial>>,
}

pub(crate) struct ShadowCorePlugin;

/// Opaque to Bevy's shadow pass, discarded by the forward pass — the private-layer proxy that lets a
/// caster participate in the real shadow map without drawing a second copy of the world. Shared by
/// every lane's SOLID caster.
#[derive(Asset, AsBindGroup, TypePath, Debug, Clone, Default)]
pub(crate) struct ShadowCasterMaterial {}

impl Material for ShadowCasterMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://benilla_app/shaders/shadow_caster.wgsl".into()
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &bevy::mesh::MeshVertexBufferLayoutRef,
        _key: bevy::pbr::MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // WoW batches are frequently two-sided; keep both faces so a mixed-winding batch's
        // projected silhouette has no holes.
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

impl Plugin for ShadowCorePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(DirectionalLightShadowMap { size: 4096 }) // MONKEY: crisper shadow edges
            .add_plugins(MaterialPlugin::<ShadowCasterMaterial>::default())
            .init_resource::<ShadowRigState>()
            .init_resource::<ShadowDemand>()
            .init_resource::<ShadowFrame>()
            // The rig runs before the lanes every frame; the lanes read the frame it publishes.
            .configure_sets(Last, (ShadowSet::Rig, ShadowSet::Lanes).chain())
            .add_systems(Last, manage_rig.in_set(ShadowSet::Rig));
    }
}

#[allow(clippy::too_many_arguments)]
fn manage_rig(
    state: Res<State<ClientState>>,
    lighting: Res<WowLighting>,
    mut demand: ResMut<ShadowDemand>,
    mut frame: ResMut<ShadowFrame>,
    mut rig: ResMut<ShadowRigState>,
    mut commands: Commands,
    mut materials: ResMut<Assets<ShadowCasterMaterial>>,
    mut cameras: Query<
        (Entity, Option<&mut RenderLayers>, Option<&ShadowCameraLayer>),
        With<WorldCamera>,
    >,
    mut suns: Query<&mut Transform, With<ShadowSun>>,
    cameras_for_position: Query<&GlobalTransform, With<WorldCamera>>,
) {
    let in_world = *state.get() == ClientState::InWorld;
    // Read the demand the lanes accumulated LAST frame, then reset so they re-accumulate this frame
    // (they run after us, in ShadowSet::Lanes). One-frame lag on rig spawn — imperceptible.
    let wanted = demand.0 && in_world;
    demand.0 = false;

    if !wanted {
        if let Some(entity) = rig.sun.take() {
            commands.entity(entity).despawn();
        }
        // Forget the held basis with the light, so a re-enable aims the fresh sun immediately.
        rig.sun_written = None;
        if let Some(handle) = rig.material.take() {
            materials.remove(handle.id());
        }
        for (entity, layers, marker) in &mut cameras {
            let Some(marker) = marker else { continue };
            if marker.had_layers {
                if let Some(mut layers) = layers {
                    *layers = layers.clone().without(PLAYER_SHADOW_LAYER);
                }
            } else {
                commands.entity(entity).remove::<RenderLayers>();
            }
            commands.entity(entity).remove::<ShadowCameraLayer>();
        }
        frame.active = false;
        frame.material = None;
        return;
    }

    if rig.material.is_none() {
        rig.material = Some(materials.add(ShadowCasterMaterial {}));
    }
    if rig.sun.is_none() {
        let sun = commands
            .spawn((
                DirectionalLight {
                    // The light exists to populate a shadow map; WoW's shader lighting stays
                    // authoritative, so it must not add a second diffuse term.
                    illuminance: 0.0,
                    shadows_enabled: true,
                    shadow_depth_bias: 0.02,
                    shadow_normal_bias: 0.8,
                    ..default()
                },
                CascadeShadowConfigBuilder {
                    num_cascades: 1,
                    minimum_distance: 0.1,
                    maximum_distance: CASTER_RANGE,
                    ..default()
                }
                .build(),
                Transform::IDENTITY,
                Visibility::Visible,
                RenderLayers::layer(PLAYER_SHADOW_LAYER),
                ShadowSun,
            ))
            .id();
        rig.sun = Some(sun);
        rig.sun_written = None;
    }

    for (entity, layers, marker) in &mut cameras {
        if marker.is_some() {
            if let Some(mut layers) = layers {
                *layers = layers.clone().with(PLAYER_SHADOW_LAYER);
            }
        } else if let Some(mut layers) = layers {
            *layers = layers.clone().with(PLAYER_SHADOW_LAYER);
            commands
                .entity(entity)
                .insert(ShadowCameraLayer { had_layers: true });
        } else {
            commands.entity(entity).insert((
                RenderLayers::default().with(PLAYER_SHADOW_LAYER),
                ShadowCameraLayer { had_layers: false },
            ));
        }
    }

    let light_position = cameras_for_position
        .iter()
        .next()
        .map(GlobalTransform::translation)
        .unwrap_or(Vec3::ZERO);

    // MONKEY (moving sun): aim at the VISIBLE celestial sun (rises/sets), clamped in elevation.
    let sun_direction = shadow_sun_travel(lighting.celestial_dir());
    let entity_reach = CASTER_RANGE + STATIC_REBUILD_STEP;
    let tall_reach = static_collection_reach(sun_direction);

    if let Some(sun) = rig.sun {
        // Only the rotation matters; QUANTISED to hold the cascade texel snap (see SUN_SNAP_RADIANS).
        if sun_snap_due(rig.sun_written, sun_direction) {
            if let Ok(mut current) = suns.get_mut(sun) {
                *current = Transform::IDENTITY.looking_to(sun_direction, Vec3::Y);
                rig.sun_written = Some(sun_direction);
            }
        }
    }

    frame.active = true;
    frame.light_position = light_position;
    frame.sun_direction = sun_direction;
    frame.entity_reach = entity_reach;
    frame.tall_reach = tall_reach;
    frame.material = rig.material.clone();
}

// ---------------------------------------------------------------------------------------------
// Shared caster-building helpers, used by both lanes.
// ---------------------------------------------------------------------------------------------

/// Spawn a SOLID caster entity on the private shadow layer with the shared proxy material.
pub(crate) fn spawn_solid_caster(
    commands: &mut Commands,
    mesh: Handle<Mesh>,
    material: Handle<ShadowCasterMaterial>,
) -> Entity {
    commands
        .spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            Transform::IDENTITY,
            Visibility::Visible,
            RenderLayers::layer(PLAYER_SHADOW_LAYER),
            NoFrustumCulling,
            NotShadowReceiver,
            ShadowCaster,
        ))
        .id()
}

/// Append a triangle list, dropping WHOLE triangles whose indices exceed the vertex range (filtering
/// single indices out of a `TriangleList` rewires the rest of the submesh into garbage).
pub(crate) fn append_triangles(source: &[u32], vertex_count: u32, base: u32, indices: &mut Vec<u32>) {
    for triangle in source.chunks_exact(3) {
        if triangle.iter().all(|index| *index < vertex_count) {
            indices.extend(triangle.iter().map(|index| base + *index));
        }
    }
}

/// Take a caster mesh's own buffers so a rebuild reuses their allocations. The mesh is left
/// attribute-less until [`restore_mesh_buffers`] puts them back.
pub(crate) fn take_mesh_buffers(mesh: &mut Mesh) -> (Vec<[f32; 3]>, Vec<u32>) {
    let mut positions = match mesh.remove_attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(bevy::mesh::VertexAttributeValues::Float32x3(values)) => values,
        _ => Vec::new(),
    };
    let mut indices = match mesh.remove_indices() {
        Some(Indices::U32(values)) => values,
        _ => Vec::new(),
    };
    positions.clear();
    indices.clear();
    (positions, indices)
}

pub(crate) fn restore_mesh_buffers(mesh: &mut Mesh, positions: Vec<[f32; 3]>, indices: Vec<u32>) {
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_indices(Indices::U32(indices));
}

pub(crate) fn empty_shadow_mesh() -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new());
    mesh.insert_indices(Indices::U32(Vec::new()));
    mesh
}

/// A cutout caster mesh: like [`empty_shadow_mesh`] but carrying a `UV_0` lane, because the
/// alpha-tested prepass fragment samples the leaf sheet at these UVs. The mesh MUST declare UV_0
/// (even empty) so Bevy's prepass specializer sets `VERTEX_UVS_A` and the fragment's `in.uv` exists.
pub(crate) fn empty_cutout_mesh() -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new());
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, Vec::<[f32; 2]>::new());
    mesh.insert_indices(Indices::U32(Vec::new()));
    mesh
}

/// Collect the per-frame ENTITY caster geometry a lane wants into its buffers. `want_creatures` /
/// `want_environment` select which kinds this caster admits — the CHARACTER lane wants creatures,
/// the WORLD lane wants entity-resident environment (gameobjects, faded doodads, WMO props). Returns
/// `(admitted, occluder-rejected)` for the trace. Range-gates before any vertex work. The `parts`
/// query is written inline (not aliased) so a lane can pass its own identically-typed query.
#[allow(clippy::too_many_arguments)]
pub(crate) fn collect_entity_geometry(
    parts: &Query<
        (
            &PickMesh,
            &ModelPart,
            Option<&GlobalTransform>,
            Option<&RigPart>,
            &ShadowOccluder,
            Option<&MeshMaterial3d<WowModelMaterial>>,
        ),
        Without<BillboardCard>,
    >,
    rigs: &Query<&RigSkin>,
    palettes: &RigPalettes,
    want_creatures: bool,
    want_environment: bool,
    light_position: Vec3,
    creature_reach: f32,
    doodad_reach: f32,
    positions: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) -> (u32, u32) {
    let (mut admitted, mut rejected) = (0u32, 0u32);
    for (pick, part, global, rig_part, occluder, _material) in parts.iter() {
        if !casts_realtime_shadow(part.kind, part.blend, want_creatures, want_environment) {
            continue;
        }
        // The content half of the visibility law (toggles, far clip, distance fade, material
        // alpha) — NOT `InheritedVisibility`, which folds camera-orientation verdicts a shadow
        // caster must ignore (a tree behind the camera still blocks the sun).
        if !occluder.0 {
            rejected += 1;
            continue;
        }
        let reach = match part.kind {
            ModelKind::Creature | ModelKind::GameObject => creature_reach,
            ModelKind::Doodad | ModelKind::Wmo => doodad_reach,
        };
        let anchor = global.map(GlobalTransform::translation).or_else(|| {
            rig_part
                .and_then(|rig_part| rigs.get(rig_part.0).ok())
                .and_then(|rig| palettes.slot_origin(rig.slot))
        });
        let Some(anchor) = anchor else { continue };
        if anchor.distance_squared(light_position) > reach * reach {
            continue;
        }
        admitted += 1;
        let base = positions.len() as u32;
        if let Some(rig_part) = rig_part {
            append_skinned(pick, rig_part, rigs, palettes, positions);
        } else if let Some(global) = global {
            positions.extend(
                pick.0
                    .positions
                    .iter()
                    .map(|p| global.transform_point(wow_to_bevy(*p)).to_array()),
            );
        }
        let added = positions.len() as u32 - base;
        if added != 0 {
            append_triangles(&pick.0.indices, added, base, indices);
        }
    }
    (admitted, rejected)
}

/// The per-frame entity caster's admit law. CHARACTERS (creatures/players/mounts) are opaque +
/// alpha-test (hair/fringe cards are small, so casting them solid doesn't box out). Entity-resident
/// ENVIRONMENT (gameobjects, faded doodads, WMO props) is OPAQUE only — a foliage leaf-card cast
/// solid becomes a box; retained-world cutout is cast leaf-shaped by the world lane's alpha-tested
/// material instead. Additive/transparent geometry never casts a solid shadow.
fn casts_realtime_shadow(
    kind: ModelKind,
    blend: ModelBlend,
    want_creatures: bool,
    want_environment: bool,
) -> bool {
    match kind {
        ModelKind::Creature => {
            want_creatures && matches!(blend, ModelBlend::Opaque | ModelBlend::AlphaTest)
        }
        _ => want_environment && matches!(blend, ModelBlend::Opaque),
    }
}

fn append_skinned(
    pick: &PickMesh,
    rig_part: &RigPart,
    rigs: &Query<&RigSkin>,
    palettes: &RigPalettes,
    positions: &mut Vec<[f32; 3]>,
) {
    let Ok(rig) = rigs.get(rig_part.0) else {
        return;
    };
    let Some(palette) = palettes.world_palette(rig.slot, rig.bones() as usize) else {
        return;
    };
    for (index, position) in pick.0.positions.iter().enumerate() {
        // A vertex without skinning data still gets an entry, or `positions` shortens against the
        // index buffer and rewires later triangles. Seat it on the root matrix.
        let (Some(joints), Some(weights)) = (pick.0.joints.get(index), pick.0.weights.get(index))
        else {
            let fallback = palette
                .get(0)
                .map(|matrix| matrix.transform_point3(wow_to_bevy(*position)))
                .unwrap_or(Vec3::ZERO);
            positions.push(fallback.to_array());
            continue;
        };
        let mut skinned = Vec3::ZERO;
        for lane in 0..4 {
            if weights[lane] > 0.0 {
                if let Some(matrix) = palette.get(joints[lane] as usize) {
                    skinned += matrix.transform_point3(wow_to_bevy(*position)) * weights[lane];
                }
            }
        }
        positions.push(skinned.to_array());
    }
}

#[cfg(test)]
mod tests {
    use super::{
        append_triangles, casts_realtime_shadow, shadow_reach_extension, shadow_sun_travel,
        static_collection_reach, sun_snap_due, CASTER_RANGE, MAX_CASTER_HEIGHT,
        MAX_SHADOW_SUN_ELEVATION, MIN_SHADOW_SUN_ELEVATION, SHADOW_REACH_CAP, STATIC_REBUILD_STEP,
        SUN_SNAP_RADIANS,
    };
    use benilla_formats::ModelBlend;
    use benilla_world::model_render::ModelKind;
    use bevy::prelude::{Quat, Vec3};

    #[test]
    fn each_lane_admits_only_its_own_kinds() {
        // The character lane wants creatures (opaque + alpha-test), not environment.
        assert!(casts_realtime_shadow(ModelKind::Creature, ModelBlend::Opaque, true, false));
        assert!(casts_realtime_shadow(ModelKind::Creature, ModelBlend::AlphaTest, true, false));
        assert!(!casts_realtime_shadow(ModelKind::Creature, ModelBlend::Blend, true, false));
        assert!(!casts_realtime_shadow(ModelKind::Doodad, ModelBlend::Opaque, true, false));
        // The world lane wants environment (opaque only), not creatures.
        for kind in [ModelKind::GameObject, ModelKind::Doodad, ModelKind::Wmo] {
            assert!(casts_realtime_shadow(kind, ModelBlend::Opaque, false, true));
            assert!(!casts_realtime_shadow(kind, ModelBlend::AlphaTest, false, true));
        }
        assert!(!casts_realtime_shadow(ModelKind::Creature, ModelBlend::Opaque, false, true));
    }

    #[test]
    fn the_moving_sun_is_clamped_in_elevation_but_keeps_its_bearing() {
        // A SET sun (to-sun below the horizon) is lifted to the floor; travel still points DOWN.
        let set = Vec3::new(0.6, -0.5, 0.6).normalize();
        let t = shadow_sun_travel(set);
        assert!(t.y < 0.0, "travel points down even for a set sun");
        assert!(
            ((-t.y).asin() - MIN_SHADOW_SUN_ELEVATION).abs() < 1e-3,
            "a set sun is pinned to the floor elevation"
        );
        // A near-ZENITH sun is capped so `looking_to(travel, Y)` never degenerates.
        let high = Vec3::new(0.02, 0.999, 0.02).normalize();
        assert!(
            ((-shadow_sun_travel(high).y).asin() - MAX_SHADOW_SUN_ELEVATION).abs() < 1e-2,
            "a near-zenith sun is capped at the ceiling elevation"
        );
        // AZIMUTH preserved: a sun bearing due +X throws its travel toward -X.
        let east = Vec3::new(1.0, 0.4, 0.0).normalize();
        let te = shadow_sun_travel(east);
        assert!(
            te.x < 0.0 && te.z.abs() < 1e-3,
            "travel points away from the sun's horizontal bearing"
        );
        assert_eq!(shadow_sun_travel(Vec3::ZERO), Vec3::NEG_Z);
    }

    #[test]
    fn the_shadow_sun_snaps_on_first_write_holds_under_drift_and_rearms() {
        let held = Vec3::new(0.3, -0.8, 0.52).normalize();
        assert!(sun_snap_due(None, held), "first write always fires");
        let axis = Vec3::Y;
        let small = Quat::from_axis_angle(axis, SUN_SNAP_RADIANS * 0.5) * held;
        assert!(!sun_snap_due(Some(held), small), "sub-threshold drift holds");
        let large = Quat::from_axis_angle(axis, SUN_SNAP_RADIANS * 2.0) * held;
        assert!(sun_snap_due(Some(held), large), "over-threshold snaps");
        assert!(!sun_snap_due(Some(large), large), "identical direction never rewrites");
    }

    #[test]
    fn shadow_reach_follows_the_sun_and_is_capped() {
        let diagonal = Vec3::new(1.0, -1.0, 0.0).normalize();
        assert!((shadow_reach_extension(diagonal) - MAX_CASTER_HEIGHT).abs() < 1e-3);
        assert_eq!(shadow_reach_extension(Vec3::NEG_Y), 0.0);
        assert_eq!(shadow_reach_extension(Vec3::NEG_Z), SHADOW_REACH_CAP);
        assert!(static_collection_reach(Vec3::NEG_Y) >= CASTER_RANGE + STATIC_REBUILD_STEP);
    }

    #[test]
    fn an_out_of_range_index_drops_its_whole_triangle_and_no_others() {
        let mut out = Vec::new();
        append_triangles(&[0, 1, 2, 1, 2, 9, 2, 0, 1], 3, 10, &mut out);
        assert_eq!(out, vec![10, 11, 12, 12, 10, 11]);
    }
}
