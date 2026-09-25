//! MONKEY (p0 graphics programme): the cvar bridges of the graphics programme's own settings, each
//! landing on the benilla-world resource its renderer reads. One observer, one arm per cvar;
//! lanes add their rows here rather than to `video.rs`'s reference block.
//!
//! | cvar | values | default | Graphics preset High | resource |
//! |---|---|---|---|---|
//! | `skyDither` | 0 Off / 1 On | 0 | 1 | `benilla_world::ffx_glow::SkyDither` |
//! | `foliageWind` | 0 Off / 1 Grass / 2 Grass + trees | 2 | 2 | `benilla_world::wind::FoliageWind` |
//! | `fogModel` | 0 Classic / 1 Modern | 0 | 1 | `benilla_world::lighting::FogModelSetting` |

use benilla_world::ffx_glow::SkyDither;
use benilla_world::lighting::FogModelSetting;
use benilla_world::wind::FoliageWind;
use bevy::prelude::*;

/// The programme's cvar observer (registered by [`MonkeyGfxPlugin`]).
pub(crate) fn on_cvar(
    ev: On<crate::cvars::CvarChanged>,
    mut dither: ResMut<SkyDither>,
    mut foliage_wind: ResMut<FoliageWind>,
    // MONKEY (fog)
    mut fog_model: ResMut<FogModelSetting>,
) {
    match ev.key().as_str() {
        "skydither" => {
            let want = ev.num() >= 0.5;
            if dither.0 != want {
                dither.0 = want;
            }
        }
        // MONKEY (wind): numeric tier, live; 0 is the exact no-displacement path.
        "foliagewind" => {
            // Capture-only A/B override. Normal runs have no such environment variable and obey
            // the live CVar exactly; the instrument can force Off without rewriting Config.wtf.
            let requested = std::env::var("WOW_FOLIAGE_WIND")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or_else(|| ev.num());
            let want = requested.clamp(0.0, 2.0) as u8;
            if foliage_wind.0 != want {
                foliage_wind.0 = want;
            }
        }
        // MONKEY (fog): `WOW_FOGMODEL` still wins for the session (captures).
        "fogmodel" => fog_model.set_from_cvar(ev.num()),
        _ => {}
    }
}

/// Registers the programme's cvar bridges. Must be added before `CvarPlugin` so the saved values
/// applied at Startup reach it (the edge `game_plugins.rs` documents for every knob plugin).
pub(crate) struct MonkeyGfxPlugin;

impl Plugin for MonkeyGfxPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SkyDither>()
            .init_resource::<FoliageWind>()
            .init_resource::<FogModelSetting>()
            .add_observer(on_cvar);
    }
}
