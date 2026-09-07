//! MONKEY (torch shadows, Phase 1) — the interior point-light shadow lane, a plug-in on
//! [`super::shadow_core`], independent of [`super::world_shadow`]/[`super::character_shadow`].
//!
//! **Why this is not Bevy's point shadows.** benilla's world camera sets
//! `bevy::light::cluster::ClusterConfig::None`, which disables Bevy's point/spot clustering AND
//! starves the point-light shadow-map prep — so `fetch_point_shadow` / `clusterable_objects` are
//! DEAD here. The old design (promote the nearest fixtures to shadow-casting `PointLight` proxies and
//! sample their cube maps) could never fire, so it is gone. In its place: this lane picks the nearest
//! ≤4 interior fixtures, computes a down-looking reverse-Z `view_proj` for each, builds the retained
//! interior caster geometry, and publishes it all as [`TorchShadowViews`] (an [`ExtractResource`]).
//! The render world then renders a depth map per fixture (`benilla_world::static_gx::torch_depth`)
//! and `static_gx.wgsl`'s interior surface lane samples it so a pillar throws a radial shadow.
//!
//! Gated on `interiorLight` + `interiorShadows` ([`VideoConfig`]); declares demand so the shared rig
//! stays up. Remove this module + its plugin + the `interiorShadows` cvar and the other lanes are
//! untouched. (Phase 1 receiver = static_gx WMO surfaces only; wow_model/entities/terrain are later.)

use bevy::math::{DMat4, DVec3, DVec4};
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResourcePlugin;

use bevy::pbr::MeshMaterial3d;

use benilla_assets::materials::WowModelMaterial;
use benilla_world::billboard::BillboardCard;
use benilla_world::interact::PickMesh;
use benilla_world::lighting::LightRooms;
use benilla_world::model_render::{ModelPart, ShadowOccluder};
use benilla_world::rig_palette::{RigPalettes, RigPart, RigSkin};
use benilla_world::static_gx::{StaticGx, TorchShadowViews};
use benilla_world::view::WorldCamera;

use crate::char_select::ClientState;
use crate::shadow_core::{
    collect_entity_geometry, empty_shadow_mesh, restore_mesh_buffers, take_mesh_buffers,
    ShadowDemand, ShadowFrame, ShadowSet, STATIC_REBUILD_STEP,
};
use crate::video::VideoConfig;

/// How many interior fixtures cast a depth map at once. The nearest N to the camera win, so the room
/// you are in casts. Matches `torch_depth::MAX_TORCH_MAPS`.
const MAX_TORCH_CASTERS: usize = 4;
/// How far from the camera to look for an interior fixture to promote (yd).
const TORCH_SEARCH_RADIUS: f32 = 55.0;
/// The map's shadow-cast range / perspective far plane (yd) — matches the WMO fixture range.
const TORCH_RANGE: f32 = 48.0;
/// MONKEY (Phase 3B): how far from the camera (yd) an ENTITY — furniture, NPC, the player — is
/// gathered into the per-frame torch caster. Covers the farthest promoted fixture's range.
const ENTITY_REACH: f32 = 60.0;
/// MONKEY (Phase 5): six cube faces per fixture. Keep in sync with `torch_depth::CUBE_FACES`.
const CUBE_FACES: usize = 6;
/// Each cube face's FOV: 90° plus a hair, so a direction exactly on a face border still lands
/// inside the face the shader picks for it (the projector returns "lit" outside its frustum).
const TORCH_FACE_FOV: f32 = std::f32::consts::FRAC_PI_2 + 0.02;

pub(crate) struct TorchShadowPlugin;

impl Plugin for TorchShadowPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TorchLane>()
            .init_resource::<TorchShadowViews>()
            // The render world reads the published fixtures/caster through this.
            .add_plugins(ExtractResourcePlugin::<TorchShadowViews>::default())
            .add_systems(Last, update_torch_shadows.in_set(ShadowSet::Lanes));
    }
}

/// The torch lane's retained state: the interior caster mesh (rebuilt on camera drift) and the
/// camera position it was last built at.
#[derive(Resource, Default)]
struct TorchLane {
    caster_mesh: Option<Handle<Mesh>>,
    caster_at: Option<Vec3>,
    /// MONKEY (Phase 3B): the per-frame ENTITY caster (furniture, NPCs, the player) — rebuilt every
    /// frame because entities move; the static `caster_mesh` only rebuilds on camera drift.
    entity_mesh: Option<Handle<Mesh>>,
}

/// The lane is active when the dynamic-interior lane is on AND its shadow toggle is on.
fn interior_shadows_on(video: &VideoConfig) -> bool {
    video.interior_light && video.interior_shadows
}

/// A reverse-Z perspective (near → 1, far → 0; wgpu clip, RH, looking down −Z) — matches the whole
/// engine's reverse-Z convention and the torch depth pipeline's `GreaterEqual` compare. Built in f64.
fn reverse_z_perspective(fov_y: f64, aspect: f64, near: f64, far: f64) -> DMat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    // Column-major columns.
    DMat4::from_cols(
        DVec4::new(f / aspect, 0.0, 0.0, 0.0),
        DVec4::new(0.0, f, 0.0, 0.0),
        DVec4::new(0.0, 0.0, near / (far - near), -1.0),
        DVec4::new(0.0, 0.0, (far * near) / (far - near), 0.0),
    )
}

/// MONKEY (Phase 5): the six cube-face `view_proj`s of a fixture — a 90°(+ε) reverse-Z perspective
/// looking down each of ±X, ±Y, ±Z — so the maps cover EVERY direction around it. This replaced the
/// single aimed cone: any single cone has an edge, and both the vertical (walk up to the forge) and
/// lateral (stand beside the player) shadow collapses were that edge being crossed. Face order is
/// the contract with `static_gx.wgsl`'s `torch_face`: 0 +X, 1 −X, 2 +Y, 3 −Y, 4 +Z, 5 −Z. Computed
/// in f64 because fixtures sit at absolute Bevy world coords (~9,300 out) where f32 view-matrix
/// arithmetic loses precision. Nothing here depends on the camera or the player any more.
fn cube_view_projs(fixture: Vec3) -> [Mat4; CUBE_FACES] {
    let eye = DVec3::new(fixture.x as f64, fixture.y as f64, fixture.z as f64);
    let proj = reverse_z_perspective(TORCH_FACE_FOV as f64, 1.0, 0.1, TORCH_RANGE as f64);
    let faces: [(DVec3, DVec3); CUBE_FACES] = [
        (DVec3::X, DVec3::Y),
        (DVec3::NEG_X, DVec3::Y),
        (DVec3::Y, DVec3::NEG_Z),
        (DVec3::NEG_Y, DVec3::NEG_Z),
        (DVec3::Z, DVec3::Y),
        (DVec3::NEG_Z, DVec3::Y),
    ];
    faces.map(|(dir, up)| {
        let vp = proj * DMat4::look_to_rh(eye, dir, up);
        Mat4::from_cols_array(&vp.to_cols_array().map(|v| v as f32))
    })
}

#[allow(clippy::too_many_arguments)]
fn update_torch_shadows(
    video: Res<VideoConfig>,
    state: Res<State<ClientState>>,
    mut demand: ResMut<ShadowDemand>,
    frame: Res<ShadowFrame>,
    mut lane: ResMut<TorchLane>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut views: ResMut<TorchShadowViews>,
    cameras: Query<&GlobalTransform, With<WorldCamera>>,
    // Interior WMO fixtures: a `PointLight` carrying a room claim.
    torches: Query<&GlobalTransform, (With<PointLight>, With<LightRooms>)>,
    gx: Option<Res<StaticGx>>,
    // MONKEY (Phase 3B): the entity caster inputs — the SAME query the world lane passes to
    // `collect_entity_geometry` (written inline by that contract so a lane can pass its own).
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
    time: Res<Time>,
    mut last_trace: Local<f64>,
) {
    // Declare demand so the shared rig stays up while on.
    let on = interior_shadows_on(&video) && *state.get() == ClientState::InWorld;
    demand.0 = demand.0 || on;

    if !(frame.active && on) {
        teardown(&mut lane, &mut meshes, &mut views);
        return;
    }
    let Some(cam) = cameras.iter().next().map(GlobalTransform::translation) else {
        return;
    };

    // (1) The N nearest interior fixtures to the camera, nearest first.
    let mut scored: Vec<(f32, Vec3)> = torches
        .iter()
        .filter_map(|t| {
            let p = t.translation();
            let d2 = p.distance_squared(cam);
            (d2 <= TORCH_SEARCH_RADIUS * TORCH_SEARCH_RADIUS).then_some((d2, p))
        })
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));
    scored.truncate(MAX_TORCH_CASTERS);
    let nearest: Vec<Vec3> = scored.into_iter().map(|(_, p)| p).collect();

    // (2) Build/refresh the interior caster mesh (ALWAYS when the lane is on — the depth node needs
    // it regardless of `worldShadows`). Rebuilt from the same retained `StaticGx` geometry the world
    // lane uses, on camera drift.
    if let Some(gx) = gx.as_ref() {
        if lane.caster_mesh.is_none() {
            lane.caster_mesh = Some(meshes.add(empty_shadow_mesh()));
            lane.caster_at = None;
        }
        let due = lane
            .caster_at
            .is_none_or(|at| at.distance(cam) > STATIC_REBUILD_STEP);
        if due {
            if let Some(handle) = lane.caster_mesh.clone() {
                if let Some(mesh) = meshes.get_mut(&handle) {
                    let (mut positions, mut indices) = take_mesh_buffers(mesh);
                    gx.append_shadow_triangles(cam, TORCH_RANGE, &mut positions, &mut indices);
                    restore_mesh_buffers(mesh, positions, indices);
                    lane.caster_at = Some(cam);
                }
            }
        }
    }

    // (2b) MONKEY (Phase 3B): the per-frame ENTITY caster — every frame (entities move), the same
    // collector the world lane feeds the directional map: creatures + gameobjects AND environment
    // (furniture, props) within ENTITY_REACH of the camera, in absolute Bevy world space like the
    // static caster, so the same torch `view_proj` projects both. A table or a character now throws
    // a torch shadow onto the floor the static_gx lane already receives.
    let mut ent_admitted = 0u32;
    if lane.entity_mesh.is_none() {
        lane.entity_mesh = Some(meshes.add(empty_shadow_mesh()));
    }
    if let Some(handle) = lane.entity_mesh.clone() {
        if let Some(mesh) = meshes.get_mut(&handle) {
            let (mut positions, mut indices) = take_mesh_buffers(mesh);
            let (admitted, _rejected) = collect_entity_geometry(
                &parts,
                &rigs,
                &palettes,
                true, // creatures + gameobjects (NPCs, the player, placed chairs/tables)
                true, // environment (doodads / WMO props)
                cam,
                ENTITY_REACH,
                ENTITY_REACH,
                &mut positions,
                &mut indices,
            );
            ent_admitted = admitted;
            restore_mesh_buffers(mesh, positions, indices);
        }
    }

    // (3) Publish the fixtures + their down-looking matrices + both caster mesh ids for the render world.
    let mut published = TorchShadowViews {
        count: nearest.len() as u32,
        caster_mesh: lane.caster_mesh.as_ref().map(Handle::id),
        entity_mesh: lane.entity_mesh.as_ref().map(Handle::id),
        ..Default::default()
    };
    // Six cube faces per fixture, independent of camera and player: layer = fixture * 6 + face.
    for (i, p) in nearest.iter().enumerate() {
        published.positions[i] = p.extend(TORCH_RANGE);
        for (f, vp) in cube_view_projs(*p).into_iter().enumerate() {
            published.view_projs[i * CUBE_FACES + f] = vp;
        }
    }
    *views = published;

    // `WOW_TORCH_TRACE=1` — once a second, what the lane published: fixture distances + whether a
    // caster mesh exists. The app-side half of the debug (the shader group-3 sample is the GPU half).
    if std::env::var_os("WOW_TORCH_TRACE").is_some() {
        let now = time.elapsed_secs_f64();
        if now - *last_trace >= 1.0 {
            *last_trace = now;
            let dists: Vec<f32> = nearest.iter().map(|p| p.distance(cam)).collect();
            info!(
                "torch-trace: {} maps (caster {}, {} entity casters) — fixture dists {:?}",
                nearest.len(),
                if lane.caster_mesh.is_some() { "built" } else { "none" },
                ent_admitted,
                dists,
            );
        }
    }
}

/// Tear down the caster mesh and clear the published views (the lane went off, or we left the world).
fn teardown(lane: &mut TorchLane, meshes: &mut Assets<Mesh>, views: &mut TorchShadowViews) {
    if let Some(handle) = lane.caster_mesh.take() {
        meshes.remove(handle.id());
    }
    if let Some(handle) = lane.entity_mesh.take() {
        meshes.remove(handle.id());
    }
    lane.caster_at = None;
    *views = TorchShadowViews::default();
}
