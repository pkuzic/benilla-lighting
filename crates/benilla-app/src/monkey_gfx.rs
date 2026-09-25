//! MONKEY (p0 graphics programme): the cvar bridges of the graphics programme's own settings, each
//! landing on the benilla-world resource its renderer reads. One observer, one arm per cvar;
//! lanes add their rows here rather than to `video.rs`'s reference block.
//!
//! | cvar | values | default | Graphics preset High | resource |
//! |---|---|---|---|---|
//! | `skyDither` | 0 Off / 1 On | 0 | 1 | `benilla_world::ffx_glow::SkyDither` |
//! | `fogModel` | 0 Classic / 1 Modern | 0 | 1 | `benilla_world::lighting::FogModelSetting` |

use bevy::prelude::*;
use benilla_world::ffx_glow::SkyDither;
use benilla_world::lighting::FogModelSetting;

/// The programme's cvar observer (registered by [`MonkeyGfxPlugin`]).
pub(crate) fn on_cvar(
    ev: On<crate::cvars::CvarChanged>,
    mut dither: ResMut<SkyDither>,
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
            .init_resource::<FogModelSetting>()
            .add_observer(on_cvar);
    }
}
