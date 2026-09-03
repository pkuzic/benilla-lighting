//! MONKEY (world shadows): the WORLD shadow lane — a self-contained, plug-and-play module.
//!
//! This owns the realtime shadows cast by the STATIC world: the retained `static_gx` geometry
//! (trees + buildings, collected solid) and the alpha-tested foliage (leaf canopies, one caster per
//! leaf texture). It is deliberately independent of [`super::character_shadow`]'s character lane:
//! the two share ONE rig (the sun + private render layer + shadow map that `character_shadow` owns)
//! but neither lane's casters depend on the other's. Enabling/disabling the world lane
//! (`worldShadows`) spawns/tears down only the entities below; removing the feature entirely means
//! deleting this module + its two calls in `character_shadow::update_shadows` + the cvar.
//!
//! Cadence: the static world doesn't move, so both casters rebuild only when the camera has drifted
//! [`STATIC_REBUILD_STEP`] (the character lane, by contrast, rebuilds every frame). Between rebuilds
//! the cached meshes stay valid.

use bevy::asset::AssetId;
use bevy::camera::visibility::{NoFrustumCulling, RenderLayers};
use bevy::image::Image;
use bevy::light::NotShadowReceiver;
use bevy::mesh::Indices;
use bevy::pbr::MeshMaterial3d;
use bevy::platform::collections::HashSet;
use bevy::prelude::*;

use benilla_world::static_gx::StaticGx;

use super::character_shadow::{
    empty_cutout_mesh, empty_shadow_mesh, restore_mesh_buffers, shadow_trace, spawn_solid_caster,
    take_mesh_buffers, CutoutCaster, CutoutShadowCasterMaterial, ShadowCasterMaterial,
    ShadowRuntime, WorldShadowCaster, PLAYER_SHADOW_LAYER, STATIC_REBUILD_STEP,
};

/// Ensure the world lane's casters exist and, on camera drift, refresh them from the retained
/// static world. The solid caster shares the rig's invisible proxy material (`solid_material`); the
/// cutout casters each carry their own leaf sheet. Called only while `worldShadows` is on.
pub(crate) fn update(
    runtime: &mut ShadowRuntime,
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    solid_material: &Handle<ShadowCasterMaterial>,
    cutout_materials: &mut Assets<CutoutShadowCasterMaterial>,
    gx: &StaticGx,
    light_position: Vec3,
    tall_reach: f32,
) {
    // Ensure the static SOLID caster (trunks, buildings, fences — the opaque static geometry).
    if runtime.static_caster.is_none() {
        let mesh = meshes.add(empty_shadow_mesh());
        let entity = spawn_solid_caster(commands, mesh.clone(), solid_material.clone());
        runtime.static_caster = Some(entity);
        runtime.static_mesh = Some(mesh);
        runtime.static_rebuilt_at = None; // force a rebuild on the first frame
    }

    // Static geometry doesn't move: rebuild only when the camera has drifted a step. Between
    // rebuilds the cached meshes stay valid, so this is a cheap early-out most frames.
    let due = runtime
        .static_rebuilt_at
        .map_or(true, |at| at.distance(light_position) > STATIC_REBUILD_STEP);
    if !due {
        return;
    }

    // The SOLID pass: the retained static world's opaque geometry, minus the cutout foliage it skips.
    if let Some(handle) = runtime.static_mesh.clone() {
        if let Some(mesh) = meshes.get_mut(&handle) {
            let (mut positions, mut indices) = take_mesh_buffers(mesh);
            gx.append_shadow_triangles(light_position, tall_reach, &mut positions, &mut indices);
            let static_tris = (indices.len() / 3) as u32;
            restore_mesh_buffers(mesh, positions, indices);
            runtime.static_rebuilt_at = Some(light_position);
            if shadow_trace() {
                info!(
                    "shadow-trace: static rebuild — {} tris within {:.0}yd",
                    static_tris, tall_reach
                );
            }
        }
    }

    // The ALPHA-TESTED foliage pass: the cutout leaf cards the solid pass skipped, grouped one
    // caster mesh + material per leaf texture so the shadow pass can sample the sheet and cast a
    // leaf-SHAPED silhouette (see `CutoutShadowCasterMaterial`).
    let buckets = gx.collect_cutout_shadow_triangles(light_position, tall_reach);
    let mut seen: HashSet<AssetId<Image>> = HashSet::new();
    let mut cutout_tris = 0u32;
    for bucket in buckets {
        seen.insert(bucket.texture_id);
        cutout_tris += (bucket.indices.len() / 3) as u32;
        // Spawn a persistent caster the first time a leaf texture appears.
        if !runtime.static_cutout.contains_key(&bucket.texture_id) {
            let mesh = meshes.add(empty_cutout_mesh());
            let material = cutout_materials.add(CutoutShadowCasterMaterial {
                leaf: bucket.texture.clone(),
            });
            let entity = commands
                .spawn((
                    Mesh3d(mesh.clone()),
                    MeshMaterial3d(material.clone()),
                    Transform::IDENTITY,
                    Visibility::Visible,
                    RenderLayers::layer(PLAYER_SHADOW_LAYER),
                    NoFrustumCulling,
                    NotShadowReceiver,
                    WorldShadowCaster,
                ))
                .id();
            runtime
                .static_cutout
                .insert(bucket.texture_id, CutoutCaster { entity, mesh, material });
        }
        // Clone the handle out so the `runtime` borrow ends before `meshes.get_mut`.
        let mesh_handle = runtime.static_cutout[&bucket.texture_id].mesh.clone();
        if let Some(mesh) = meshes.get_mut(&mesh_handle) {
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, bucket.positions);
            mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, bucket.uvs);
            mesh.insert_indices(Indices::U32(bucket.indices));
        }
    }
    // Despawn casters whose leaf texture left the reach, so the set stays bounded as the camera
    // crosses zones (a texture no longer present casts nothing anyway).
    let stale: Vec<AssetId<Image>> = runtime
        .static_cutout
        .keys()
        .filter(|id| !seen.contains(*id))
        .copied()
        .collect();
    for id in stale {
        if let Some(caster) = runtime.static_cutout.remove(&id) {
            commands.entity(caster.entity).despawn();
            meshes.remove(caster.mesh.id());
            cutout_materials.remove(caster.material.id());
        }
    }
    if shadow_trace() {
        info!(
            "shadow-trace: cutout rebuild — {} tris across {} leaf textures",
            cutout_tris,
            runtime.static_cutout.len()
        );
    }
}

/// Tear down every world-lane caster (the solid static caster + all per-texture foliage casters).
/// Called when `worldShadows` is off, or by the rig teardown when both lanes go dark. Leaves the
/// shared rig (sun/layer/material) alone — that is `character_shadow`'s to manage.
pub(crate) fn teardown(
    runtime: &mut ShadowRuntime,
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    cutout_materials: &mut Assets<CutoutShadowCasterMaterial>,
) {
    if let Some(entity) = runtime.static_caster.take() {
        commands.entity(entity).despawn();
    }
    if let Some(handle) = runtime.static_mesh.take() {
        meshes.remove(handle.id());
    }
    runtime.static_rebuilt_at = None;
    for (_texture, caster) in runtime.static_cutout.drain() {
        commands.entity(caster.entity).despawn();
        meshes.remove(caster.mesh.id());
        cutout_materials.remove(caster.material.id());
    }
}
