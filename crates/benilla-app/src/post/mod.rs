//! MONKEY (post): optional world-camera effects ahead of FFXGlow's gamma clamp.

mod bloom;

pub(crate) struct PostPlugin;

impl bevy::prelude::Plugin for PostPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_plugins(bloom::BloomPlugin);
    }
}
