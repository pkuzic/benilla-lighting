//! MONKEY (p0 graphics programme): the cvar bridges of the graphics programme's own settings, each
//! landing on the benilla-world resource its renderer reads. One observer, one arm per cvar;
//! lanes add their rows here rather than to `video.rs`'s reference block.
//!
//! | cvar | values | default | Graphics preset High | resource |
//! |---|---|---|---|---|
//! | `skyDither` | 0 Off / 1 On | 0 | 1 | `benilla_world::ffx_glow::SkyDither` |
//! | `rainSurfaces` | 0 Off / 1 On | 1 | 1 | `benilla_world::weather::RainSurfaces` |

use bevy::prelude::*;
use benilla_world::ffx_glow::SkyDither;
use benilla_world::weather::RainSurfaces; // MONKEY (wet)

/// The programme's cvar observer (registered by [`MonkeyGfxPlugin`]).
pub(crate) fn on_cvar(
    ev: On<crate::cvars::CvarChanged>,
    mut dither: ResMut<SkyDither>,
    mut rain: ResMut<RainSurfaces>, // MONKEY (wet)
) {
    match ev.key().as_str() {
        "skydither" => {
            let want = ev.num() >= 0.5;
            if dither.0 != want {
                dither.0 = want;
            }
        }
        // MONKEY (wet): wet surfaces + rain ripples.
        "rainsurfaces" => {
            let want = ev.num() >= 0.5;
            if rain.0 != want {
                rain.0 = want;
            }
        }
        _ => {}
    }
}

/// Registers the programme's cvar bridges. Must be added before `CvarPlugin` so the saved values
/// applied at Startup reach it (the edge `game_plugins.rs` documents for every knob plugin).
pub(crate) struct MonkeyGfxPlugin;

impl Plugin for MonkeyGfxPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SkyDither>()
            .init_resource::<RainSurfaces>() // MONKEY (wet)
            .add_observer(on_cvar);
    }
}
