//! The CHARACTER shadow lane — a thin plug-in on top of [`super::shadow_core`].
//!
//! It owns nothing but its own per-frame caster (players, NPCs, creatures, mounts). When
//! `characterShadows` is on it declares demand to the shared rig ([`ShadowDemand`]); while the rig
//! is live it collects the CREATURE entity parts into its caster using the rig's shared material and
//! the per-frame facts the rig publishes ([`ShadowFrame`]). It does not know [`super::world_shadow`]
//! exists — remove either lane and the other is untouched.

use bevy::pbr::MeshMaterial3d;
use bevy::prelude::*;

use benilla_assets::materials::WowModelMaterial;
use benilla_world::billboard::BillboardCard;
use benilla_world::interact::PickMesh;
use benilla_world::model_render::{ModelPart, ShadowOccluder};
use benilla_world::rig_palette::{RigPalettes, RigPart, RigSkin};

use crate::shadow_core::{
    collect_entity_geometry, empty_shadow_mesh, restore_mesh_buffers, shadow_trace,
    spawn_solid_caster, take_mesh_buffers, ShadowDemand, ShadowFrame, ShadowSet,
};
use crate::video::VideoConfig;

pub(crate) struct CharacterShadowPlugin;

impl Plugin for CharacterShadowPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CharacterLane>()
            .add_systems(Last, collect_character_shadows.in_set(ShadowSet::Lanes));
    }
}

/// The character lane's retained per-frame caster.
#[derive(Resource, Default)]
struct CharacterLane {
    caster: Option<Entity>,
    mesh: Option<Handle<Mesh>>,
    /// [`shadow_trace`]'s change detector: (tris, admitted, rejected).
    traced: (u32, u32, u32),
}

#[allow(clippy::too_many_arguments)]
fn collect_character_shadows(
    video: Res<VideoConfig>,
    mut demand: ResMut<ShadowDemand>,
    frame: Res<ShadowFrame>,
    mut lane: ResMut<CharacterLane>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
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
) {
    let on = video.character_shadows;
    // Declare demand so the shared rig stays up while this lane is on (the rig reads it next frame).
    demand.0 = demand.0 || on;

    // Rig down or lane off → drop our caster so it stops casting.
    if !(frame.active && on) {
        if let Some(entity) = lane.caster.take() {
            commands.entity(entity).despawn();
        }
        if let Some(handle) = lane.mesh.take() {
            meshes.remove(handle.id());
        }
        return;
    }
    let Some(material) = frame.material.clone() else {
        return;
    };

    // Animated rigs re-skin every frame and creatures spawn/despawn under lifecycles this lane
    // can't cheaply observe — so the caster is collected fresh every frame.
    if lane.caster.is_none() {
        let mesh = meshes.add(empty_shadow_mesh());
        let entity = spawn_solid_caster(&mut commands, mesh.clone(), material);
        lane.caster = Some(entity);
        lane.mesh = Some(mesh);
    }
    if let Some(handle) = lane.mesh.clone() {
        if let Some(mesh) = meshes.get_mut(&handle) {
            let (mut positions, mut indices) = take_mesh_buffers(mesh);
            let (admitted, rejected) = collect_entity_geometry(
                &parts,
                &rigs,
                &palettes,
                true,  // want_creatures
                false, // want_environment (the world lane's job)
                frame.light_position,
                frame.entity_reach,
                frame.tall_reach,
                &mut positions,
                &mut indices,
            );
            let tris = (indices.len() / 3) as u32;
            restore_mesh_buffers(mesh, positions, indices);
            if shadow_trace() {
                let now = (tris, admitted, rejected);
                if lane.traced != now {
                    info!(
                        "shadow-trace: character {} tris | creatures admitted {} / rejected {} (was {:?})",
                        tris, admitted, rejected, lane.traced
                    );
                    lane.traced = now;
                }
            }
        }
    }
}
