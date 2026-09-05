//! MONKEY (torch shadows, #2 spike): a point-light shadow lane over the shared rig.
//!
//! The nearest WMO interior fixture (a `PointLight` with `LightRooms`) is promoted to a PRIVATE-layer
//! (layer 31) shadow-casting `PointLight` PROXY at its position. Bevy renders that proxy's cube
//! shadow map from the layer-31 casters the world lane already builds (walls, columns), so the
//! interior architecture casts into it for free. The receivers sample it via
//! `shadow_hook::torch_shadow` (which scans `clusterable_objects` for the shadow-enabled point light
//! and calls `fetch_point_shadow`). The proxy's own Bevy light is IGNORED by benilla's custom
//! receivers (they light from the `wow_light` Gouraud term, not Bevy clustering) — only its cube map
//! + clusterable entry are used.
//!
//! SPIKE SCOPE: env-gated (`WOW_TORCH_SHADOWS`), ONE torch, reuses the world lane's layer-31 casters
//! — so `worldShadows` must be on for the walls/columns to exist as casters. Milestone: a column
//! throws a shadow on the interior floor that tracks as you move. Not yet a cvar/slider feature.

use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;

use benilla_world::lighting::LightRooms;
use benilla_world::view::WorldCamera;

use crate::char_select::ClientState;
use crate::shadow_core::{ShadowSet, PLAYER_SHADOW_LAYER};

/// How far from the camera to look for an interior fixture to promote (yd).
const TORCH_SEARCH_RADIUS: f32 = 60.0;
/// The proxy light's shadow-cast range (yd) — matches the WMO fixture range (`fx::POINT_LIGHT_RANGE`).
const TORCH_RANGE: f32 = 48.0;

pub(crate) struct TorchShadowPlugin;

impl Plugin for TorchShadowPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TorchProxy>()
            .add_systems(Last, update_torch_shadow.in_set(ShadowSet::Lanes));
    }
}

/// Marks the proxy point light so it is never itself a promotion candidate.
#[derive(Component)]
struct TorchShadowProxy;

#[derive(Resource, Default)]
struct TorchProxy(Option<Entity>);

/// `WOW_TORCH_SHADOWS=1` — the spike gate (no cvar yet).
fn torch_shadows_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("WOW_TORCH_SHADOWS").is_ok_and(|v| v != "0"))
}

fn update_torch_shadow(
    state: Res<State<ClientState>>,
    mut proxy: ResMut<TorchProxy>,
    mut commands: Commands,
    cameras: Query<&GlobalTransform, With<WorldCamera>>,
    // Interior WMO fixtures: a `PointLight` that carries a room claim (`LightRooms`). The proxy has
    // no `LightRooms`, so it is naturally excluded from the candidate set.
    torches: Query<&GlobalTransform, (With<PointLight>, With<LightRooms>)>,
    mut proxies: Query<&mut Transform, With<TorchShadowProxy>>,
) {
    let active = torch_shadows_enabled() && *state.get() == ClientState::InWorld;
    if !active {
        if let Some(entity) = proxy.0.take() {
            commands.entity(entity).despawn();
        }
        return;
    }
    let Some(cam) = cameras.iter().next().map(GlobalTransform::translation) else {
        return;
    };
    // The nearest interior fixture within the search radius.
    let mut nearest: Option<(f32, Vec3)> = None;
    for fixture in torches.iter() {
        let pos = fixture.translation();
        let d2 = pos.distance_squared(cam);
        if d2 <= TORCH_SEARCH_RADIUS * TORCH_SEARCH_RADIUS
            && nearest.is_none_or(|(best, _)| d2 < best)
        {
            nearest = Some((d2, pos));
        }
    }
    let Some((_, pos)) = nearest else {
        if let Some(entity) = proxy.0.take() {
            commands.entity(entity).despawn();
        }
        return;
    };
    if let Some(entity) = proxy.0 {
        if let Ok(mut transform) = proxies.get_mut(entity) {
            transform.translation = pos;
        }
    } else {
        let entity = commands
            .spawn((
                PointLight {
                    shadows_enabled: true,
                    range: TORCH_RANGE,
                    // Ignored by the custom receivers; only Bevy's cube-map render + clustering read
                    // the light. A nominal value keeps it from being culled.
                    intensity: 1000.0,
                    shadow_depth_bias: 0.06,
                    shadow_normal_bias: 0.8,
                    ..default()
                },
                Transform::from_translation(pos),
                Visibility::Visible,
                RenderLayers::layer(PLAYER_SHADOW_LAYER),
                TorchShadowProxy,
            ))
            .id();
        proxy.0 = Some(entity);
    }
}
