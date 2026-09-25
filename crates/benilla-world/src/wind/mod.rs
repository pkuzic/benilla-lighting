//! Ported from WarcraftXL (https://github.com/WarcraftXL) by iThorgrim — module
//! `wxl-experimental-wind`, `field/Wind.hpp` and `field/Wind.cpp`.
//!
//! MONKEY (wind): one stateless wind field shared by every visual system. The field is a pure
//! function of its profile, weather and time; consumers read [`WindField`] rather than inventing
//! their own heading or gust clock. [`FoliageWind`] only gates the foliage receivers.

use bevy::prelude::*;

use crate::lighting::MonkeyFrame;
use crate::weather::{storm_blend, WeatherState, WeatherTick};

const TAU: f64 = std::f64::consts::TAU;
const RATIOS: [f64; 3] = [1.0, 0.437, 0.1913];
const WEIGHTS: [f64; 3] = [0.60, 0.28, 0.12];

/// The WarcraftXL wind profile. Units are yards, seconds and degrees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindProfile {
    pub base_speed: f32,
    pub heading_deg: f32,
    pub gust: f32,
    pub gust_period: f32,
    pub veer_deg: f32,
    pub veer_period: f32,
    pub weather_coupling: f32,
    pub seed: u32,
}

impl Default for WindProfile {
    fn default() -> Self {
        Self {
            base_speed: 9.0,
            heading_deg: 45.0,
            gust: 0.55,
            gust_period: 17.0,
            veer_deg: 22.0,
            veer_period: 53.0,
            weather_coupling: 1.0,
            seed: 0x5749_4E44,
        }
    }
}

/// One resolved instant of the shared field, in Bevy's horizontal XZ plane.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WindSample {
    pub dir: Vec2,
    pub speed: f32,
    pub heading_deg: f32,
    /// The shaped gust envelope, 0 at a lull and 1 at a peak.
    pub gust: f32,
}

/// The public shared wind resource. Future water/cloud consumers read [`Self::sample`].
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct WindField {
    pub profile: WindProfile,
    pub sample: WindSample,
}

impl Default for WindField {
    fn default() -> Self {
        let profile = WindProfile::default();
        Self {
            sample: sample(profile, 0.0, 0.0),
            profile,
        }
    }
}

/// Foliage wind quality: 0 Off, 1 grass, 2 grass + trees.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoliageWind(pub u8);

impl Default for FoliageWind {
    fn default() -> Self {
        // The resource exists before the CVar host. Its ordinary default therefore has to equal
        // the registered default; a default-valued CVar emits no change event at boot. The env
        // arm is capture-only and lets the A/B harness force the exact zero-displacement path.
        let tier = std::env::var("WOW_FOLIAGE_WIND")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(2.0)
            .clamp(0.0, 2.0) as u8;
        Self(tier)
    }
}

fn seeded_unit(seed: u32, index: i32) -> f64 {
    let mut h = seed ^ (index as u32).wrapping_mul(0x9E37_79B9);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    f64::from(h & 0x00FF_FFFF) / f64::from(0x0100_0000u32)
}

/// WarcraftXL's three-sine band-limited wander in `[-1, 1]`.
fn wander(seed: u32, lane: i32, period: f32, seconds: f64) -> f32 {
    let period = f64::from(period.max(0.05));
    RATIOS
        .into_iter()
        .zip(WEIGHTS)
        .enumerate()
        .map(|(i, (ratio, weight))| {
            let phase = seeded_unit(seed, lane * 97 + i as i32) * TAU;
            let turns = seconds / (period * ratio);
            weight * (turns.fract() * TAU + phase).sin()
        })
        .sum::<f64>() as f32
}

/// Resolves the pure wind field. `storm` is 0 for clear weather and 1 for a full storm.
pub fn sample(profile: WindProfile, seconds: f64, storm: f32) -> WindSample {
    let raw = wander(profile.seed, 0, profile.gust_period, seconds);
    let gust = ((raw + 1.0) * 0.5).powf(1.7);
    let gust_gain = 1.0 + profile.gust * (gust - 0.5) * 2.0;
    let weather_gain = 1.0 + storm.clamp(0.0, 1.0) * 1.1 * profile.weather_coupling;
    let speed = (profile.base_speed * gust_gain * weather_gain).max(0.0);
    let heading_deg = profile.heading_deg
        + profile.veer_deg * wander(profile.seed, 1, profile.veer_period, seconds);
    let heading = heading_deg.to_radians();
    WindSample {
        dir: Vec2::new(heading.cos(), heading.sin()),
        speed,
        heading_deg,
        gust,
    }
}

/// MONKEY (wind): resolve once at the frame boundary, then publish the same sample and clock to
/// Rust consumers and the appended MonkeyFrame block. A zero tier clears every visual strength.
fn update_wind(
    time: Res<Time>,
    weather: Res<WeatherState>,
    quality: Res<FoliageWind>,
    mut wind: ResMut<WindField>,
    mut frame: ResMut<MonkeyFrame>,
) {
    let seconds = time.elapsed_secs_f64();
    wind.sample = sample(wind.profile, seconds, storm_blend(weather.sky_density));

    frame.wind_dir = wind.sample.dir.to_array();
    frame.wind_speed = wind.sample.speed;
    frame.wind_gust = wind.sample.gust;
    frame.wind_time_s = seconds as f32;
    frame.sway_strength = if quality.0 > 0 { 1.0 } else { 0.0 };
    frame.grass_strength = if quality.0 >= 1 { 1.0 } else { 0.0 };
    frame.tree_strength = if quality.0 >= 2 { 1.0 } else { 0.0 };
}

/// Installs the shared field after this frame's weather ramp has resolved.
pub struct WindPlugin;

impl Plugin for WindPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WindField>()
            .init_resource::<FoliageWind>()
            .add_systems(Update, update_wind.after(WeatherTick));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_the_warcraft_xl_profile() {
        let p = WindProfile::default();
        assert_eq!(p.base_speed, 9.0);
        assert_eq!(p.heading_deg, 45.0);
        assert_eq!(p.gust, 0.55);
        assert_eq!(p.gust_period, 17.0);
        assert_eq!(p.veer_deg, 22.0);
        assert_eq!(p.veer_period, 53.0);
    }

    #[test]
    fn field_is_stateless_and_weather_only_scales_speed() {
        let p = WindProfile::default();
        let clear = sample(p, 123.456, 0.0);
        assert_eq!(clear, sample(p, 123.456, 0.0));
        let storm = sample(p, 123.456, 1.0);
        assert_eq!(clear.dir, storm.dir);
        assert_eq!(clear.gust, storm.gust);
        assert!((storm.speed / clear.speed - 2.1).abs() < 1.0e-5);
    }

    #[test]
    fn direction_is_unit_length_and_gust_is_bounded() {
        let p = WindProfile::default();
        for t in [0.0, 1.0, 17.0, 53.0, 86_400.0] {
            let s = sample(p, t, 0.0);
            assert!((s.dir.length() - 1.0).abs() < 1.0e-5);
            assert!((0.0..=1.0).contains(&s.gust));
            assert!(s.speed >= 0.0);
        }
    }
}
