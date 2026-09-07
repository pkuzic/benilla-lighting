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
use benilla_world::lighting::DynamicInteriors;
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
fn bridge(video: Res<VideoConfig>, mut out: ResMut<DynamicInteriors>) {
    let want = DynamicInteriors {
        enabled: video.interior_light,
        ambient: video.interior_ambient,
        fill: video.interior_fill,
        exposure: video.interior_exposure,
        debug: video.interior_debug,
    };
    if *out != want {
        *out = want;
    }
}
