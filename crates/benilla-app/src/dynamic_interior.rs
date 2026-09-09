//! MONKEY (dynamic interiors): the app-side half of the fixture-lit WMO interior lane — a
//! plug-and-play bridge from the `interiorLight` / `interiorAmbient` / `interiorFill` /
//! `interiorExposure` cvars ([`VideoConfig`]) to benilla-world's [`DynamicInteriors`] resource,
//! which `global_light` packs for `static_gx.wgsl`'s `interior_room_light` (WMO interior surfaces
//! AND interior props light from the room's live fixtures instead of the MOCV bake / the baked prop
//! probe). Independent of the shadow lanes: removing this plugin leaves the resource at its
//! shipped defaults; `interiorLight 0` restores the faithful baked interior path.
//!
//! The knobs are LIVE — `/script SetCVar("interiorExposure", 2)` in chat retunes the room without
//! a rebuild, which is how the defaults were found.
//!
//! MONKEY (fire GO lights): the same bridge also carries `fireLightGain` →
//! [`benilla_world::lighting::FireLightGain`]. Same shape, same guard, same liveness requirement —
//! and `fireLightGain 0` is the kill switch for the synthesised-fire lane the way `interiorLight 0`
//! is for this one.
use benilla_world::lighting::{DynamicInteriors, FireLightGain};
use bevy::prelude::*;

use crate::video::VideoConfig;

pub(crate) struct DynamicInteriorPlugin;

impl Plugin for DynamicInteriorPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, bridge);
    }
}

/// Publish the cvars to benilla-world when they change. Guarded so the resource's change ticks
/// stay quiet on idle frames.
///
/// MONKEY (fire GO lights): `fireLightGain` rides along here rather than in a module of its own —
/// it is the same shape (a live video cvar → a benilla-world resource the light packer reads) and
/// the same one-line guard, and a second plugin to carry one `f32` is not worth its wiring.
fn bridge(
    video: Res<VideoConfig>,
    mut out: ResMut<DynamicInteriors>,
    mut fire: ResMut<FireLightGain>,
) {
    if fire.0 != video.fire_light_gain {
        fire.0 = video.fire_light_gain;
    }
    let want = DynamicInteriors {
        enabled: video.interior_light,
        ambient: video.interior_ambient,
        fill: video.interior_fill,
        exposure: video.interior_exposure,
        // MONKEY (interior attenuation): the authored-window scale rides the same bridge — the
        // light packer folds it into every interior entry's packed reach, so
        // `SetCVar("interiorAttenScale", 2)` widens every pool on the very next frame.
        atten_scale: video.interior_atten_scale,
        // MONKEY (room gate): the packer turns this into a per-light claim head, so
        // `SetCVar("interiorRoomGate", 0)` restores the ungated look on the very next frame.
        room_gate: video.interior_room_gate,
        debug: video.interior_debug,
    };
    if *out != want {
        *out = want;
    }
}
