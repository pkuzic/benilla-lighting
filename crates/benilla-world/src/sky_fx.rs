//! MONKEY (sky): the sky quality tier and what the Enhanced/High sky reads each frame.
//!
//! `skyQuality` 0 is Classic, the reference's image byte for byte: the dome's piecewise-linear
//! gradient and the kernel's cloud texels, untouched. 1 (Enhanced) turns on the smooth gradient,
//! the sun glow and the procedural night sky in `sky.wgsl`; 2 (High) adds the cloud detail and the
//! sun-lit cloud shading in `cloud.wgsl`. The WGSL lives in `shaders/sky_fx.wgsl`.
//!
//! The High cloud detail and sun march (`sky_fx.wgsl`, `cloud.wgsl`, `clouds/layer.rs`):
//! Ported from WarcraftXL (https://github.com/WarcraftXL) by iThorgrim — module wxl-retail-clouds, Clouds.cpp, Clouds.hpp.

use bevy::prelude::*;

/// The sky tier: 0 Classic, 1 Enhanced, 2 High. Written by the app from the `skyQuality` cvar;
/// `$WOW_SKY_QUALITY` pins it for the session (captures).
#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct SkyQuality(pub u8);

impl SkyQuality {
    /// The highest tier.
    pub const MAX: u8 = 2;

    /// The session override, `$WOW_SKY_QUALITY` clamped to the tiers.
    pub fn env_override() -> Option<u8> {
        std::env::var("WOW_SKY_QUALITY")
            .ok()
            .and_then(|v| v.trim().parse::<u8>().ok())
            .map(|v| v.min(Self::MAX))
    }

    pub fn enhanced(self) -> bool {
        self.0 >= 1
    }

    pub fn high(self) -> bool {
        self.0 >= 2
    }
}

/// The sky's own animation clock (star twinkle, cloud detail drift), seconds wrapped to an hour so
/// the shader's `sin` keeps its precision. Frozen in capture mode, at `$WOW_CAPTURE_SKY_T` or 0.
#[derive(Resource, Default)]
pub struct SkyClock {
    pub secs: f32,
    frozen: Option<f32>,
}

/// How strongly the glow shows: a broad halo at `GLOW_GAIN` × the sun colour at the sun itself.
pub(crate) const GLOW_GAIN: f32 = 0.32;

/// The glow's day fade: full once the sun clears the horizon, gone a few degrees under it.
pub(crate) fn glow_day_fade(sun_y: f32) -> f32 {
    let t = ((sun_y + 0.10) / 0.16).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The glow strength: the day fade, dimmed by the cloud cover over the sun (the same coverage the
/// glare's occlusion reads) and by storm weather.
pub(crate) fn glow_strength(sun_y: f32, cover_at_sun: f32, bcc: f32) -> f32 {
    GLOW_GAIN * glow_day_fade(sun_y) * (1.0 - 0.8 * cover_at_sun.clamp(0.0, 1.0))
        * (1.0 - 0.85 * bcc.clamp(0.0, 1.0))
}

pub(crate) struct SkyFxPlugin;

impl Plugin for SkyFxPlugin {
    fn build(&self, app: &mut App) {
        // The library `sky.wgsl` and `cloud.wgsl` import; its embedded path is served by
        // `shaders::plugin`, this keeps it loaded so the import resolves.
        let lib: Handle<Shader> =
            bevy::asset::load_embedded_asset!(app, "shaders/sky_fx.wgsl");
        std::mem::forget(lib);
        let frozen = std::env::var_os("WOW_CAPTURE").map(|_| {
            std::env::var("WOW_CAPTURE_SKY_T")
                .ok()
                .and_then(|v| v.trim().parse::<f32>().ok())
                .unwrap_or(0.0)
        });
        app.insert_resource(SkyClock {
            secs: frozen.unwrap_or(0.0),
            frozen,
        });
        let quality = SkyQuality(SkyQuality::env_override().unwrap_or(0));
        app.insert_resource(quality)
            .add_systems(Update, tick_sky_clock);
    }
}

fn tick_sky_clock(time: Res<Time>, mut clock: ResMut<SkyClock>) {
    if let Some(t) = clock.frozen {
        if clock.secs != t {
            clock.secs = t;
        }
        return;
    }
    clock.secs = (clock.secs + time.delta_secs()) % 3600.0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glow_fades_out_below_the_horizon_and_under_cloud() {
        assert_eq!(glow_strength(-0.2, 0.0, 0.0), 0.0);
        assert!((glow_strength(0.5, 0.0, 0.0) - GLOW_GAIN).abs() < 1e-6);
        assert!(glow_strength(0.5, 1.0, 0.0) < 0.25 * GLOW_GAIN);
        assert!(glow_strength(0.5, 0.0, 1.0) < 0.2 * GLOW_GAIN);
        // Monotone across dusk.
        let mut prev = 0.0;
        for i in 0..40 {
            let y = -0.15 + i as f32 * 0.01;
            let g = glow_day_fade(y);
            assert!(g >= prev);
            prev = g;
        }
    }
}
