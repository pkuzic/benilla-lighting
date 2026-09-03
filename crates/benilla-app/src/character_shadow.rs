//! Real shadow-map support for benilla, split into two INDEPENDENT lanes over ONE shared rig.
//!
//! This module owns the shared **rig** — a single directional light (the "sun") on a private render
//! layer, that layer's membership on the world camera, and the `DirectionalLightShadowMap` — plus
//! the **character lane**: the per-frame CPU-built caster carrying entity geometry (players, NPCs,
//! creatures, mounts). The sibling [`super::world_shadow`] module owns the **world lane** (the
//! static `static_gx` trees/buildings + alpha-tested foliage). The two lanes are gated by SEPARATE
//! cvars — `characterShadows` and `worldShadows` — and either one alone lights up the shared rig.
//!
//! The rig deliberately does NOT put a directional light on the normal world layer: that would make
//! every existing custom WoW material enter Bevy's shadow-prepass path, the source of the pipeline
//! validation/corruption failure seen during the first experiment. Instead a private layer holds
//! CPU-built copies of the caster triangles; only those copies cast, and the normal world shaders
//! merely RECEIVE the resulting shadow lookup.
//!
//! Why one shared light and not one per lane: the custom receiver in `terrain.wgsl`/`wow_model.wgsl`
//! samples a SINGLE directional light (it assigns, not accumulates, over the light loop). Two lights
//! would double the shadow-pass cost and the last one would win the receiver. So the lanes share the
//! rig and each simply contributes its own caster geometry into the one map.

use bevy::asset::{AssetId, RenderAssetUsages};
use bevy::camera::visibility::{NoFrustumCulling, RenderLayers};
use bevy::image::Image;
use bevy::light::{
    CascadeShadowConfigBuilder, DirectionalLight, DirectionalLightShadowMap, NotShadowReceiver,
};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::pbr::{Material, MaterialPipeline, MaterialPlugin, MeshMaterial3d};
use bevy::platform::collections::HashMap;
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
use benilla_world::lighting::{WorldShadowActive, WowLighting};
use benilla_world::model_render::{ModelKind, ModelPart, ShadowOccluder};
use benilla_world::rig_palette::{RigPalettes, RigPart, RigSkin};
use benilla_world::view::WorldCamera;

use crate::char_select::ClientState;
use crate::video::VideoConfig;
use crate::world_shadow;

/// Layer 31 is deliberately private to this feature. Bevy supports 32 render layers; the normal
/// world is layer 0 and the UI/portrait layers occupy the low numbered slots.
pub(crate) const PLAYER_SHADOW_LAYER: usize = 31;

/// How far from the camera the shadow map RESOLVES (the single cascade's `maximum_distance`).
/// Collection reaches farther than this: a caster's own body may stand outside the cascade
/// while its shadow lands inside it — in light space a caster's footprint sits exactly where
/// its shadow falls, so an out-of-frustum tree still darkens ground the player is looking at.
/// See [`static_collection_reach`] for the collection law.
const CASTER_RANGE: f32 = 80.0;

/// How far the camera may drift before a caster-collection pass must refresh. The collection
/// reach carries this as margin so coverage holds everywhere between refreshes.
pub(crate) const STATIC_REBUILD_STEP: f32 = 16.0;

/// The tallest COMMON caster the reach law budgets for (the big Elwynn/Duskwood tree class,
/// in world units). Not a clamp on what casts — only on how far past the resolve range the
/// collection hunts for shadow SOURCES.
const MAX_CASTER_HEIGHT: f32 = 35.0;

/// Ceiling on the shadow-reach extension: the lighting sun's elevation only spans ~20°–37°
/// over the whole day, but a degenerate near-horizontal direction must not explode the
/// collection radius (and with it the rebuild cost) unbounded.
const SHADOW_REACH_CAP: f32 = 120.0;

/// How far past a caster's own position its shadow can land: `height × horizontal/vertical`
/// of the lighting sun's travel direction. At the night elevation (~20°) a 35-unit tree throws
/// a ~96-unit shadow — collecting only to `CASTER_RANGE + STATIC_REBUILD_STEP` made exactly
/// those trees pop in as the camera drifted a rebuild step, their long shadows sweeping across
/// ground the player was already looking at.
fn shadow_reach_extension(sun_travel: Vec3) -> f32 {
    // `sun_travel` is the direction the light TRAVELS (downward): -y is the vertical drop.
    let down = -sun_travel.y;
    if down <= 1e-3 {
        return SHADOW_REACH_CAP;
    }
    let horizontal = Vec3::new(sun_travel.x, 0.0, sun_travel.z).length();
    (MAX_CASTER_HEIGHT * horizontal / down).min(SHADOW_REACH_CAP)
}

/// The one collection law for a lane that can contribute a TALL caster (a doodad/WMO entity
/// exile, or the whole static world): resolve range + rebuild margin + the sun-dependent shadow
/// reach.
fn static_collection_reach(sun_travel: Vec3) -> f32 {
    CASTER_RANGE + STATIC_REBUILD_STEP + shadow_reach_extension(sun_travel)
}

/// `WOW_SHADOW_TRACE=1` — log the caster population whenever it CHANGES, so "the shadow
/// vanished when I turned" can be attributed to a lane: a dynamic-mesh count that dips with
/// camera rotation means the entity lane is still being view-culled somewhere; a static
/// rebuild logged at the same moment means the cached half swapped content; neither changing
/// while the shadow still pops on screen exonerates the caster and indicts the map/receiver.
pub(crate) fn shadow_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("WOW_SHADOW_TRACE").is_ok_and(|v| v != "0"))
}

/// How far the LIVE sun may drift from the direction the shadow light was last given before
/// the light snaps to it — about one game minute of lighting-sun travel (the DayNight phi
/// table sweeps ~0.297 rad over 360 game minutes ≈ 8.25e-4 rad/min).
///
/// The shadow basis is quantised because a light that rotates EVERY frame defeats the cascade
/// fit's texel snap: `calculate_cascade` (bevy_light) snaps the ortho origin to texel
/// multiples in LIGHT space, which pins shadow edges to world positions under camera
/// translation — but rotating the basis re-maps every world position, and at WoW map
/// coordinates (Elwynn sits ~9,300 units from the world origin) even the lighting sun's tiny
/// ~1.4e-5 rad/s drift slides light-space positions ~0.13 units/s ≈ 1.6 shadow texels/s: a
/// continuous sub-texel re-rasterisation crawl on every shadow edge (the inn-roof field
/// report). Held between snaps, the basis is bit-stable and the map only re-rasterises when
/// the sun has honestly moved — a discrete 1–2 texel tick roughly once a game minute, the
/// same physical reasoning that lets MCSH ship as a static bake. LIGHTING stays continuous;
/// only the shadow basis steps.
const SUN_SNAP_RADIANS: f32 = 8.0e-4;

/// Whether the live sun direction has drifted far enough from the held (last-written) one to
/// re-aim the shadow light. `None` = never written (first frame after spawn, or re-enable
/// after teardown): always due.
fn sun_snap_due(held: Option<Vec3>, live: Vec3) -> bool {
    match held {
        None => true,
        Some(held) => held.angle_between(live) > SUN_SNAP_RADIANS,
    }
}

#[derive(Component)]
pub(crate) struct WorldShadowCaster;

#[derive(Component)]
struct WorldShadowSun;

#[derive(Component)]
struct WorldShadowCameraLayer {
    had_layers: bool,
}

/// Shared state for both shadow lanes. The rig + character fields are private to this module; the
/// world-lane fields are `pub(crate)` because [`super::world_shadow`] owns their lifecycle.
#[derive(Resource, Default)]
pub(crate) struct ShadowRuntime {
    // --- shared rig ---
    sun: Option<Entity>,
    /// The direction the sun entity was LAST GIVEN — the held shadow basis. The live direction is
    /// compared against this each frame and written through only past [`SUN_SNAP_RADIANS`] (see
    /// there for why); `None` until the first write, and again after teardown, so spawn and
    /// re-enable always aim the light immediately.
    sun_written: Option<Vec3>,
    /// The invisible proxy material shared by BOTH lanes' solid casters (opaque to the shadow pass,
    /// discarded in the forward pass).
    material: Option<Handle<ShadowCasterMaterial>>,
    // --- character lane (per-frame entity caster) ---
    /// The PER-FRAME caster: entity parts (animated rigs, creatures, and — when `worldShadows` is
    /// on — entity-resident environment doodads/gameobjects) collected fresh every frame.
    dynamic_caster: Option<Entity>,
    dynamic_mesh: Option<Handle<Mesh>>,
    /// [`shadow_trace`]'s change detector: (dynamic tris, admitted parts, rejected parts).
    traced: (u32, u32, u32),
    // --- world lane (owned by `super::world_shadow`) ---
    /// The CACHED static caster — the retained `static_gx` world (trees + buildings), rebuilt only
    /// on camera drift.
    pub(crate) static_caster: Option<Entity>,
    pub(crate) static_mesh: Option<Handle<Mesh>>,
    pub(crate) static_rebuilt_at: Option<Vec3>,
    /// The alpha-tested foliage casters — one per distinct leaf texture, keyed by its `AssetId`.
    pub(crate) static_cutout: HashMap<AssetId<Image>, CutoutCaster>,
}

pub(crate) struct ShadowPlugin;

/// Opaque to Bevy's shadow pass, but discarded by the normal forward pass. This lets the
/// private-layer proxy participate in the real shadow map without drawing a second copy of the
/// world in the main camera. Shared by both lanes' SOLID casters.
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
        // WoW's model batches are frequently authored as two-sided geometry. Keeping both faces
        // in the caster avoids holes in the projected silhouette when a batch has mixed winding.
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

/// The alpha-tested twin of [`ShadowCasterMaterial`]: the same invisible-in-forward proxy, but its
/// prepass fragment samples a leaf sheet and discards transparent texels so cutout FOLIAGE (tree
/// canopies) casts a leaf-shaped silhouette instead of the solid box a positions-only proxy throws.
/// One instance per distinct leaf texture — a merged caster mesh carries only one sheet. Used only
/// by the WORLD lane ([`super::world_shadow`]).
///
/// `AlphaMode::Mask` sets Bevy's `MAY_DISCARD` pipeline key, and the EXPLICIT (non-`Default`)
/// `prepass_fragment_shader` is what makes the otherwise depth-only directional-light SHADOW pass
/// run a fragment at all — together they are the exact pair Bevy's prepass specializer requires to
/// alpha-test in the shadow map (`prepass/mod.rs`: `MAY_DISCARD && get_shader(PrepassFragmentShader)`).
#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
pub(crate) struct CutoutShadowCasterMaterial {
    /// The leaf sheet, bound at material binding 0/1. Reusing the SAME image handle the forward
    /// pass draws makes the sampler inherit that image's clamp/repeat address mode, so the shadow
    /// silhouette matches the visible foliage (a cutout card's out-of-range UVs must clamp, not
    /// wrap, or the transparent margin folds back into the opaque middle and it casts solid).
    #[texture(0)]
    #[sampler(1)]
    pub(crate) leaf: Handle<Image>,
}

impl Material for CutoutShadowCasterMaterial {
    fn fragment_shader() -> ShaderRef {
        // Forward pass: still fully invisible — the shared proxy fragment discards every fragment.
        "embedded://benilla_app/shaders/shadow_caster.wgsl".into()
    }

    fn prepass_fragment_shader() -> ShaderRef {
        // EXPLICIT → the shadow pass runs this fragment for the MAY_DISCARD material and alpha-tests.
        "embedded://benilla_app/shaders/shadow_caster_cutout_prepass.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Mask(benilla_assets::materials::VANILLA_ALPHA_KEY_REF)
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &bevy::mesh::MeshVertexBufferLayoutRef,
        _key: bevy::pbr::MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Two-sided like the solid caster — a canopy card seen edge-on must still occlude. Applies
        // to the shadow (prepass) pipeline too: Bevy runs `specialize` for the prepass specializer.
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

/// One live alpha-tested foliage caster: its layer-31 entity, its per-texture caster mesh, and the
/// [`CutoutShadowCasterMaterial`] carrying that one leaf sheet. Keyed by the leaf texture's
/// `AssetId` in [`ShadowRuntime::static_cutout`] so it persists across rebuilds.
pub(crate) struct CutoutCaster {
    pub(crate) entity: Entity,
    pub(crate) mesh: Handle<Mesh>,
    pub(crate) material: Handle<CutoutShadowCasterMaterial>,
}

impl Plugin for ShadowPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(DirectionalLightShadowMap { size: 4096 }) // MONKEY: crisper shadow edges
            .add_plugins(MaterialPlugin::<ShadowCasterMaterial>::default())
            // MONKEY (world shadows): the alpha-tested foliage caster material (leaf-shaped canopy shadows).
            .add_plugins(MaterialPlugin::<CutoutShadowCasterMaterial>::default())
            .init_resource::<ShadowRuntime>()
            .add_systems(Last, update_shadows);
    }
}

#[allow(clippy::too_many_arguments)]
fn update_shadows(
    state: Res<State<ClientState>>,
    video: Res<VideoConfig>,
    lighting: Res<WowLighting>,
    mut runtime: ResMut<ShadowRuntime>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ShadowCasterMaterial>>,
    // MONKEY (world shadows): the per-leaf-texture alpha-tested foliage caster materials.
    mut cutout_materials: ResMut<Assets<CutoutShadowCasterMaterial>>,
    parts: Query<
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
    rigs: Query<&RigSkin>,
    palettes: Res<RigPalettes>,
    mut cameras: Query<
        (
            Entity,
            Option<&mut RenderLayers>,
            Option<&WorldShadowCameraLayer>,
        ),
        With<WorldCamera>,
    >,
    mut suns: Query<&mut Transform, With<WorldShadowSun>>,
    cameras_for_position: Query<&GlobalTransform, With<WorldCamera>>,
    // MONKEY (world shadows): the retained static world — the source of the cached static caster.
    // `Option` because the resource exists only when the retained pass is armed (`WOW_STATIC_GX`).
    gx: Option<Res<benilla_world::static_gx::StaticGx>>,
    // MONKEY (world shadows): the terrain MCSH-suppression flag — driven by the WORLD lane only.
    mut world_active: ResMut<WorldShadowActive>,
) {
    let in_world = *state.get() == ClientState::InWorld;
    let world_on = video.world_shadows && in_world;
    let char_on = video.character_shadows && in_world;

    // Baked MCSH terrain shadows switch off ONLY for the world lane (a character-only shadow sun
    // must leave the world's baked shadows intact). `terrain.wgsl` reads this via `sh_c16.w`.
    if world_active.0 != world_on {
        world_active.0 = world_on;
    }

    if !world_on && !char_on {
        // Both lanes dark: tear the whole rig down.
        if let Some(entity) = runtime.dynamic_caster.take() {
            commands.entity(entity).despawn();
        }
        if let Some(entity) = runtime.sun.take() {
            commands.entity(entity).despawn();
        }
        // Forget the held shadow basis with its light, so a re-enable aims the fresh sun on
        // its first frame rather than waiting out a snap threshold.
        runtime.sun_written = None;
        if let Some(handle) = runtime.dynamic_mesh.take() {
            meshes.remove(handle.id());
        }
        world_shadow::teardown(&mut runtime, &mut commands, &mut meshes, &mut cutout_materials);
        if let Some(handle) = runtime.material.take() {
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
            commands.entity(entity).remove::<WorldShadowCameraLayer>();
        }
        return;
    }

    // --- Shared rig: the proxy material + the sun, alive whenever EITHER lane is on ---
    if runtime.material.is_none() {
        runtime.material = Some(materials.add(ShadowCasterMaterial {}));
    }
    if runtime.sun.is_none() {
        let sun = commands
            .spawn((
                DirectionalLight {
                    // The light exists to populate a shadow map. WoW's existing shader lighting
                    // remains authoritative, so this light must not add a second diffuse term.
                    illuminance: 0.0,
                    shadows_enabled: true,
                    // Keep enough separation from the receiver to avoid shadow-map acne on WMO
                    // floors, while staying small enough that the silhouette remains attached.
                    shadow_depth_bias: 0.02,
                    shadow_normal_bias: 0.8,
                    ..default()
                },
                CascadeShadowConfigBuilder {
                    // A short-range shadow feature, not a whole-map sun-shadow system. With ONE
                    // cascade, Bevy ignores `first_cascade_far_bound`/`overlap_proportion` and the
                    // single map spans the full `maximum_distance`.
                    num_cascades: 1,
                    minimum_distance: 0.1,
                    maximum_distance: CASTER_RANGE,
                    ..default()
                }
                .build(),
                Transform::IDENTITY,
                Visibility::Visible,
                RenderLayers::layer(PLAYER_SHADOW_LAYER),
                WorldShadowSun,
            ))
            .id();
        runtime.sun = Some(sun);
        runtime.sun_written = None; // aim it on the first frame
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
                .insert(WorldShadowCameraLayer { had_layers: true });
        } else {
            commands.entity(entity).insert((
                RenderLayers::default().with(PLAYER_SHADOW_LAYER),
                WorldShadowCameraLayer { had_layers: false },
            ));
        }
    }

    let light_position = cameras_for_position
        .iter()
        .next()
        .map(GlobalTransform::translation)
        .unwrap_or(Vec3::ZERO);

    let sun_direction = if lighting.sun_dir.length_squared() > 1e-6 {
        lighting.sun_dir.normalize()
    } else {
        Vec3::NEG_Z
    };
    // The two gating radii: short casters (creatures/gameobjects) stop mattering just past the
    // resolve range; a tall-caster lane (doodads/WMO/static world) carries the sun-dependent
    // shadow reach on top, so a tree whose shadow lands in view casts before its trunk enters the
    // resolve range.
    let entity_reach = CASTER_RANGE + STATIC_REBUILD_STEP;
    let tall_reach = static_collection_reach(sun_direction);

    // --- CHARACTER lane (+ entity-resident environment when the world lane is on) ---
    // The per-frame entity caster always exists past the early-out; the ADMIT filter decides what
    // rides it: Creature parts when `characterShadows`, entity-resident doodads/gameobjects when
    // `worldShadows`. Animated rigs re-skin every frame, and entity content spawns/despawns under
    // lifecycles this module cannot cheaply observe — so it is collected fresh every frame.
    if runtime.dynamic_caster.is_none() {
        let mesh = meshes.add(empty_shadow_mesh());
        let material = runtime
            .material
            .clone()
            .expect("rig material ensured above");
        let entity = spawn_solid_caster(&mut commands, mesh.clone(), material);
        runtime.dynamic_caster = Some(entity);
        runtime.dynamic_mesh = Some(mesh);
    }
    if let Some(handle) = runtime.dynamic_mesh.clone() {
        if let Some(mesh) = meshes.get_mut(&handle) {
            let (mut positions, mut indices) = take_mesh_buffers(mesh);
            let (admitted, rejected) = collect_entity_geometry(
                &parts,
                &rigs,
                &palettes,
                char_on,
                world_on,
                light_position,
                entity_reach,
                tall_reach,
                &mut positions,
                &mut indices,
            );
            let dynamic_tris = (indices.len() / 3) as u32;
            restore_mesh_buffers(mesh, positions, indices);
            if shadow_trace() {
                let now = (dynamic_tris, admitted, rejected);
                if runtime.traced != now {
                    info!(
                        "shadow-trace: dynamic {} tris | parts admitted {} / occluder-rejected {} (was {:?})",
                        dynamic_tris, admitted, rejected, runtime.traced
                    );
                    runtime.traced = now;
                }
            }
        }
    }

    // --- WORLD lane: the cached static world (trees + buildings) + alpha-tested foliage ---
    if world_on {
        let solid = runtime.material.clone();
        if let (Some(gx), Some(solid)) = (gx.as_ref(), solid) {
            world_shadow::update(
                &mut runtime,
                &mut commands,
                &mut meshes,
                &solid,
                &mut cutout_materials,
                gx,
                light_position,
                tall_reach,
            );
        }
    } else {
        world_shadow::teardown(&mut runtime, &mut commands, &mut meshes, &mut cutout_materials);
    }

    // --- Aim the shared sun (quantised — see SUN_SNAP_RADIANS) ---
    if let Some(sun) = runtime.sun {
        if sun_snap_due(runtime.sun_written, sun_direction) {
            if let Ok(mut current) = suns.get_mut(sun) {
                *current = Transform::IDENTITY.looking_to(sun_direction, Vec3::Y);
                runtime.sun_written = Some(sun_direction);
            }
        }
    }
}

/// Spawn a SOLID caster entity on the private shadow layer with the shared proxy material. Used by
/// both the character lane (its per-frame caster) and the world lane (its cached static caster).
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
            WorldShadowCaster,
        ))
        .id()
}

/// Append a triangle list, dropping WHOLE triangles whose indices exceed the vertex range.
/// Filtering single indices out of a `TriangleList` shifts every subsequent index by one and
/// rewires the rest of the submesh into garbage triangles.
fn append_triangles(source: &[u32], vertex_count: u32, base: u32, indices: &mut Vec<u32>) {
    for triangle in source.chunks_exact(3) {
        if triangle.iter().all(|index| *index < vertex_count) {
            indices.extend(triangle.iter().map(|index| base + *index));
        }
    }
}

/// Take a caster mesh's own buffers so a rebuild reuses their allocations rather than paying
/// a fresh `Vec` growth curve every time. The mesh is left attribute-less until
/// [`restore_mesh_buffers`] puts them back.
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
/// (even empty) so Bevy's prepass specializer sets the `VERTEX_UVS_A` shader-def and the fragment's
/// `in.uv` exists — without it the cutout discard is compiled out and foliage casts solid again.
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

#[allow(clippy::too_many_arguments)]
fn collect_entity_geometry(
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
    char_on: bool,
    world_on: bool,
    light_position: Vec3,
    creature_reach: f32,
    doodad_reach: f32,
    positions: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) -> (u32, u32) {
    let (mut admitted, mut rejected) = (0u32, 0u32);
    for (pick, part, global, rig_part, occluder, _material) in parts.iter() {
        if !casts_realtime_shadow(part.kind, part.blend, char_on, world_on) {
            continue;
        }
        // NOT `InheritedVisibility`: that folds the exterior window gate, the portal PVS and
        // the exterior-scene cull — all camera-orientation verdicts — and a tree behind the
        // camera still blocks the sun. `ShadowOccluder` is the content half of the same law
        // (toggles, far clip, distance fade, material alpha), written by the one visibility
        // authority.
        if !occluder.0 {
            rejected += 1;
            continue;
        }
        // The distance gate, BEFORE any vertex work: the cascade resolves to CASTER_RANGE, so
        // a part beyond its lane's reach contributes nothing. Creatures/gameobjects are at most a
        // body tall; doodad/WMO exiles can be whole trees and take the tall-caster reach.
        let reach = match part.kind {
            ModelKind::Creature | ModelKind::GameObject => creature_reach,
            ModelKind::Doodad | ModelKind::Wmo => doodad_reach,
        };
        // A rig part's placement lives in its palette (world-space rows), not necessarily a
        // propagated `GlobalTransform`; prefer the transform when present, else the rig's
        // slot origin — both are one lookup, never a skin.
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

/// The per-frame entity caster's admit law. CHARACTERS (creatures/players/mounts) ride the
/// `characterShadows` lane; entity-resident ENVIRONMENT (gameobjects, distance-faded doodads, WMO
/// props) rides the `worldShadows` lane — so a chest or a fading tree casts with the world, not
/// with characters. Only the BLEND gate additionally excludes additive/transparent geometry.
fn casts_realtime_shadow(kind: ModelKind, blend: ModelBlend, char_on: bool, world_on: bool) -> bool {
    match kind {
        // Characters: opaque + alpha-test — hair/fringe cards are small, so casting them solid
        // doesn't produce the ugly canopy-box artifact (and dropping them would lose hair shadows).
        ModelKind::Creature => char_on && matches!(blend, ModelBlend::Opaque | ModelBlend::AlphaTest),
        // Entity-resident environment: OPAQUE only. A foliage leaf-card cast solid becomes a box on
        // the ground (this per-frame caster has no texture to alpha-test); cutout foliage in the
        // retained world is cast leaf-shaped by the WORLD lane's `CutoutShadowCasterMaterial`, and
        // entity-resident cutout (a doodad mid distance-fade) waits for that path to extend here.
        _ => world_on && matches!(blend, ModelBlend::Opaque),
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
        // A vertex without skinning data still gets an entry: skipping it would shorten
        // `positions` against the index buffer and rewire every later triangle in the
        // submesh. Seat it on the root matrix so it at least stays near the model.
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
        append_triangles, casts_realtime_shadow, shadow_reach_extension, static_collection_reach,
        sun_snap_due, CASTER_RANGE, MAX_CASTER_HEIGHT, SHADOW_REACH_CAP, STATIC_REBUILD_STEP,
        SUN_SNAP_RADIANS,
    };
    use benilla_formats::ModelBlend;
    use benilla_world::model_render::ModelKind;
    use bevy::prelude::Vec3;

    #[test]
    fn the_two_lanes_admit_independently() {
        // Characters ride the CHARACTER lane (opaque + alpha-test); entity environment rides the
        // WORLD lane (opaque only). Nothing casts Blend on either lane, and a lane that is off
        // admits nothing of its kind.
        // Both lanes on:
        assert!(casts_realtime_shadow(ModelKind::Creature, ModelBlend::Opaque, true, true));
        assert!(casts_realtime_shadow(ModelKind::Creature, ModelBlend::AlphaTest, true, true));
        assert!(!casts_realtime_shadow(ModelKind::Creature, ModelBlend::Blend, true, true));
        for kind in [ModelKind::GameObject, ModelKind::Doodad, ModelKind::Wmo] {
            assert!(casts_realtime_shadow(kind, ModelBlend::Opaque, true, true));
            assert!(!casts_realtime_shadow(kind, ModelBlend::AlphaTest, true, true));
            assert!(!casts_realtime_shadow(kind, ModelBlend::Blend, true, true));
        }
        // Character lane OFF → creatures cast nothing; the world lane still casts environment.
        assert!(!casts_realtime_shadow(ModelKind::Creature, ModelBlend::Opaque, false, true));
        assert!(casts_realtime_shadow(ModelKind::Doodad, ModelBlend::Opaque, false, true));
        // World lane OFF → environment casts nothing; the character lane still casts creatures.
        assert!(!casts_realtime_shadow(ModelKind::Doodad, ModelBlend::Opaque, true, false));
        assert!(casts_realtime_shadow(ModelKind::Creature, ModelBlend::Opaque, true, false));
    }

    #[test]
    fn the_shadow_sun_snaps_on_first_write_holds_under_drift_and_rearms() {
        let held = Vec3::new(0.3, -0.8, 0.52).normalize();
        // Never written (spawn, or re-enable after teardown): always due.
        assert!(sun_snap_due(None, held), "first write always fires");
        // Sub-threshold drift holds the basis bit-stable — this is the whole point.
        let axis = Vec3::Y;
        let small = bevy::prelude::Quat::from_axis_angle(axis, SUN_SNAP_RADIANS * 0.5) * held;
        assert!(
            !sun_snap_due(Some(held), small),
            "sub-threshold drift holds"
        );
        // Past the threshold the light snaps…
        let large = bevy::prelude::Quat::from_axis_angle(axis, SUN_SNAP_RADIANS * 2.0) * held;
        assert!(sun_snap_due(Some(held), large), "over-threshold snaps");
        // …and the new held value re-arms the window around the NEW direction.
        assert!(
            !sun_snap_due(Some(large), large),
            "identical direction never rewrites"
        );
        let drift_from_new =
            bevy::prelude::Quat::from_axis_angle(axis, SUN_SNAP_RADIANS * 0.5) * large;
        assert!(
            !sun_snap_due(Some(large), drift_from_new),
            "window re-arms around the snapped direction"
        );
    }

    #[test]
    fn shadow_reach_follows_the_sun_and_is_capped() {
        // 45° sun: horizontal == vertical, a shadow reaches exactly one caster height.
        let diagonal = Vec3::new(1.0, -1.0, 0.0).normalize();
        assert!((shadow_reach_extension(diagonal) - MAX_CASTER_HEIGHT).abs() < 1e-3);
        // The night lighting sun (~20° elevation): ~2.7× height, still under the cap.
        let e = 20.0f32.to_radians();
        let night = Vec3::new(e.cos(), -e.sin(), 0.0);
        let expected = MAX_CASTER_HEIGHT * e.cos() / e.sin();
        assert!((shadow_reach_extension(night) - expected).abs() < 1e-2);
        assert!(expected < SHADOW_REACH_CAP);
        // Straight down sheds no shadow beyond the caster; a degenerate horizontal direction
        // (the NEG_Z fallback for a zeroed sun) takes the cap, never unbounded.
        assert_eq!(shadow_reach_extension(Vec3::NEG_Y), 0.0);
        assert_eq!(shadow_reach_extension(Vec3::NEG_Z), SHADOW_REACH_CAP);
        // The collection law always contains the resolve range plus the rebuild margin.
        assert!(static_collection_reach(Vec3::NEG_Y) >= CASTER_RANGE + STATIC_REBUILD_STEP);
    }

    #[test]
    fn an_out_of_range_index_drops_its_whole_triangle_and_no_others() {
        let mut out = Vec::new();
        // Middle triangle references vertex 9 of 3 — the whole triple must go, and the last
        // triangle must keep its own vertices (a single dropped index would shift it).
        append_triangles(&[0, 1, 2, 1, 2, 9, 2, 0, 1], 3, 10, &mut out);
        assert_eq!(out, vec![10, 11, 12, 12, 10, 11]);
    }

    #[test]
    fn a_trailing_partial_triple_is_ignored() {
        let mut out = Vec::new();
        append_triangles(&[0, 1, 2, 0, 1], 3, 0, &mut out);
        assert_eq!(out, vec![0, 1, 2]);
    }
}
