//! Player-facing **video settings** — the knobs that reach the window and the presentation path.
//!
//! Two, both of them 1.12's own CVars over the Graphics page: **Display Mode** (`gxWindow`, the
//! *Display Mode* row) and **VSync** (`gxVSync`, the *Vertical Sync* row).
//!
//! # Display mode — the reference's CVar, deliberately not the reference's meaning (decision 1627)
//!
//! 1.12 ships a **two-state** display model: `gxWindow "0"` takes the display with an exclusive
//! mode-set at `gxResolution`, `gxWindow "1"` runs in a window (with `gxMaximize` for a maximized
//! one). We keep the CVar, the row, and the default; we redefine `"0"`.
//!
//! **`"0"` here means BORDERLESS fullscreen** — a normal window sized to the monitor, flagged
//! fullscreen to the compositor (`xdg_toplevel.set_fullscreen` on Wayland,
//! `_NET_WM_STATE_FULLSCREEN` on X11, a native fullscreen window on macOS). We ship no exclusive
//! mode at all, and that is not a shortcut — it is unavailable or pointless on all three targets:
//!
//! - **Wayland has no client-side mode-setting**, so there is nothing to implement: winit 0.30.13
//!   answers `Fullscreen::Exclusive` with ``warn!("`Fullscreen::Exclusive` is ignored on Wayland")``
//!   and leaves the window as it was (`platform_impl/linux/wayland/window/mod.rs:147,488`).
//! - **On X11 it is XRandR mode-setting**, which changes the *desktop's* mode and, as winit's own
//!   comment says, "does not provide a mechanism to … restore this to the desktop video mode as
//!   macOS and Windows do" — a crash leaves the player's desktop at our resolution.
//! - **On macOS** there is no exclusive mode to take.
//!
//! And the industry moved: SDL3 deleted `SDL_WINDOW_FULLSCREEN_DESKTOP` and made borderless-desktop
//! what a fullscreen window *is* unless you opt into a mode with `SDL_SetWindowFullscreenMode`;
//! WoW itself removed exclusive fullscreen in **8.0.1**, leaving Windowed and Windowed
//! (Fullscreen) — VERIFIED at the source, not from the patch notes: Blizzard's own
//! `Blizzard_SettingsDefinitions_Shared/Graphics.lua` (live, `classic` and `classic_era`, all
//! identical) registers Display Mode as a **boolean** proxy and builds its dropdown from exactly
//! two entries, `VIDEO_OPTIONS_WINDOWED_FULLSCREEN` and `VIDEO_OPTIONS_WINDOWED`. **Our two states
//! are modern Classic's two states**, which is why 1650 wears them as that client's dropdown
//! rather than 1.12's checkbox. (Modern hangs the boolean on `gxMaximize`, having deleted
//! `gxWindow` outright; we keep `gxWindow`, which is the CVar our configs already persist.) Bevy's own `WindowMode::Fullscreen` arm is a liability besides — it `expect`s a
//! monitor at creation and `panic!`s on a live change that cannot resolve one
//! (`bevy_winit::winit_windows:91`, `system.rs:333`).
//!
//! **What this is worth, concretely.** Before 1627 the window was born `WindowMode::Windowed` at a
//! hard-coded 1600×900 with nothing able to change it — larger than a Steam Deck's 1280×800 panel,
//! and never flagged fullscreen. A window that does not fill gamescope's nested output is a
//! documented input break upstream (ValveSoftware/gamescope#1086 — pointer trapped in the nested
//! rect, clicks outside it dead; #1209 — tapping the letterbox warps the cursor to centre forever),
//! whose recorded workaround is "set the game to fullscreen".
//!
//! **No `gxRestart`.** 1.12 flags its whole video block restart-required; ours takes effect on the
//! click, the same stated departure [`apply_present_mode`] already makes for `gxVSync`.
//!
//! # VSync
//!
//! **Why it is a player setting and not a dev toggle.** It briefly lived as a checkbox on the perf
//! HUD, where it existed to answer one instrument question — "is the GPU keeping up?" — because a
//! synced frame reads as the display's grant whether it needed 3 ms or 16 (0717). That is the wrong
//! home twice over: the HUD is `#[cfg(feature = "dev")]`, so a player build could never reach it,
//! and vsync is not a diagnostic in the first place. It is the same option 1.12 shipped —
//! `OptionsFrameCheckButtons["VERTICAL_SYNC"] = { index = 5, cvar = "gxVSync", gxRestart = 1 }`,
//! Video Options, checkbox 5 — and the same one every engine since has kept. The instrument
//! question keeps `$WOW_NOVSYNC=1`, which is where a measurement knob belongs.
//!
//! **We do not require the restart 1.12 did.** The reference's row carries `gxRestart = 1` because
//! its device could not swap the presentation interval live; wgpu reconfigures the surface on the
//! next frame, so the checkbox takes effect as you click it. A deliberate, stated departure.
//!
//! **`AutoNoVsync`, never `Immediate`.** On macOS/Metal, explicit `Immediate` both rails *and*
//! takes ~1 s `nextDrawable` stalls — measured, and pinned at [`crate::capture::probe_uncap_mode`].

use benilla_ui::script::ScreenResolution;
use bevy::prelude::*;
use bevy::window::{MonitorSelection, PresentMode, PrimaryWindow, WindowMode, WindowResolution};

/// `$WOW_NOVSYNC=1` — the session-only measurement override. It wins over the config for the run
/// and never reaches `config.toml` (registered in [`crate::cvars`]'s `session_owned` set), so a
/// headless FPS-journal run can uncap without making the player's setting sticky.
pub(crate) fn novsync_env() -> bool {
    std::env::var("WOW_NOVSYNC").as_deref() == Ok("1")
}

/// Does this run **size its own window**, and therefore stay windowed whatever the config says?
///
/// Three sources, and the first two are named explicitly rather than left to the third because a
/// capture that silently went fullscreen would render at the display's resolution instead of the
/// scenario's — every visual regression diff in the tree is denominated in the window the scenario
/// asks for, so this one must not depend on the machine's panel. The third is the general rule: an
/// instrumented run's window is plumbing ([`benilla_world::bgwin`]), and the probe fleet sizes and
/// parks it deliberately (decisions 0703/0709/1148).
///
/// Session-only, exactly like [`novsync_env`]: `gxWindow`/`gxResolution` are registered
/// env-overridden while it holds, so the file's value neither reaches the window nor is saved over.
pub(crate) fn windowed_env() -> bool {
    std::env::var_os("WOW_WIN").is_some()
        || std::env::var_os("WOW_CAPTURE").is_some()
        || std::env::var_os("WOW_CAPTURE_UI").is_some()
        || benilla_world::bgwin::background_run()
}

/// `$WOW_WIN=WxH` in **logical** px — the one parser, so the window that is *asked for* and the
/// window that is *checked* can never drift. It was spelled twice inline in `lib.rs` (once for the
/// UI-fixture arm, once for the world arm) and is now spelled here once and read there.
pub(crate) fn requested_window_size() -> Option<UVec2> {
    let v = std::env::var("WOW_WIN").ok()?;
    let (w, h) = v.split_once('x')?;
    Some(UVec2::new(w.parse().ok()?, h.parse().ok()?))
}

/// `$WOW_DPI=<f32>` — **render at a player's pixel grid, not this machine's.**
///
/// Every session here runs on a 2× panel; nearly every text-under-scale report the channel files
/// comes from a 1080p/1440p one at 1×. That gap is not cosmetic. The two places our text meets the
/// grid — the raster size (`TextEngine::ppem`, `round(logical × dpi)`) and the per-block vertical
/// snap (`ui_text::layout::snap_block_top`) — are both *quantizers*, and a quantizer's error is
/// denominated in device pixels: at 1× the same layout rounds twice as coarsely as it does here.
/// A defect that is half a pixel at 2× is a whole one at 1×, which is the difference between
/// invisible and reported (B209, B231, B232 — all from 1×, all reproduced here only by forcing
/// this).
///
/// The override goes on the *window*, so `Window::scale_factor()` — the one number the text engine,
/// the vplates raster and the world backdrop all read — answers with it, and `WOW_WIN` then means
/// physical pixels one-to-one. The image a capture writes is byte-for-byte the framebuffer that
/// player's GPU would scan out; on this display it is simply drawn at half the size.
///
/// Absent, nothing changes: the window keeps whatever the display reports.
pub(crate) fn requested_dpi() -> Option<f32> {
    let v: f32 = std::env::var("WOW_DPI").ok()?.parse().ok()?;
    (v.is_finite() && v > 0.0).then_some(v)
}

/// Apply [`requested_dpi`] to a window resolution — the one place the knob is spent, so the size
/// that is asked for and the grid it is asked for on cannot drift apart.
pub(crate) fn at_requested_dpi(res: WindowResolution) -> WindowResolution {
    match requested_dpi() {
        Some(dpi) => res.with_scale_factor_override(dpi),
        None => res,
    }
}

/// The display modes benilla ships. **Two, and neither is the reference's mode-setting
/// fullscreen** — the module doc says why.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum DisplayMode {
    /// Borderless, filling the monitor. `gxWindow "0"` — the reference's own default value, and
    /// what every shipped game defaults to.
    #[default]
    Fullscreen,
    /// A window at [`VideoConfig::windowed`]. `gxWindow "1"`.
    Windowed,
}

/// The windowed size a fresh config gets: the 1600×900 that was the client's only size before
/// 1627, so nothing about a windowed run moved when the mode setting landed.
pub(crate) const DEFAULT_WINDOWED: UVec2 = UVec2::new(1600, 900);

/// `gxWindow`'s value → the mode. The reference's own parse for its 0/1 CVars is int + `!= 0`, and
/// one function is the whole law so [`crate::cvars`]'s arm and the boot read cannot drift.
pub(crate) fn display_from_flag(v: f32) -> DisplayMode {
    if v != 0.0 {
        DisplayMode::Windowed
    } else {
        DisplayMode::Fullscreen
    }
}

/// `gxResolution`'s value → a size. The reference's spelling (`"1280x800"`), whose own parse is
/// `sscanf("%d%c%d")`; ours is stricter by one thing only — a zero extent is refused rather than
/// handed to the windowing system.
pub(crate) fn parse_resolution(value: &str) -> Option<UVec2> {
    let (w, h) = value.split_once(['x', 'X'])?;
    let size = UVec2::new(w.trim().parse().ok()?, h.trim().parse().ok()?);
    (size.x > 0 && size.y > 0).then_some(size)
}

/// The `WindowMode` a display mode means, on a given monitor.
pub(crate) fn window_mode(display: DisplayMode, monitor: MonitorSelection) -> WindowMode {
    match display {
        DisplayMode::Fullscreen => WindowMode::BorderlessFullscreen(monitor),
        DisplayMode::Windowed => WindowMode::Windowed,
    }
}

/// The mode the primary window is **born** in — resolved before the `App` exists, because the
/// alternative is worse than the frame it costs. Booting windowed and flipping at `Startup` (where
/// [`crate::cvars`]'s `load_config` runs) is a visible flash on every launch, and under a
/// compositor that only maps a fullscreen surface 1:1 it is a first second spent in exactly the
/// broken input state 1627 exists to end.
///
/// `MonitorSelection::Primary`, not `Current`: at creation there is no current monitor —
/// `bevy_winit::select_monitor` warns and answers `None` for `Current` — and "the monitor the
/// window is on" is not a question with an answer before the window exists. A live toggle uses
/// `Current`; [`apply_window_mode`] carries why the two never fight.
pub(crate) fn boot_window_mode() -> WindowMode {
    let display = if windowed_env() {
        DisplayMode::Windowed
    } else {
        crate::cvars::boot_cvar("gxWindow")
            .and_then(|v| v.parse::<f32>().ok())
            .map_or_else(DisplayMode::default, display_from_flag)
    };
    window_mode(display, MonitorSelection::Primary)
}

/// The windowed size the primary window is **born** at — `gxResolution`, read at the same pre-`App`
/// moment and for the same reason as [`boot_window_mode`]. Ignored by winit while the mode is
/// fullscreen (`bevy_winit` applies an inner size only on the `Windowed` arm), and the value
/// [`apply_window_mode`] hands back on the way out of it.
pub(crate) fn boot_windowed_size() -> UVec2 {
    crate::cvars::boot_cvar("gxResolution")
        .and_then(|v| parse_resolution(&v))
        .unwrap_or(DEFAULT_WINDOWED)
}

/// The video knobs a CVar write can land on. Default = what the client ships: fullscreen, synced,
/// matching the primary window's own boot values — see `lib.rs`'s window literal.
///
/// The **file** is deliberately not read here, only the environment (the posture `vsync` has had
/// since 0294): the window literal resolves env-then-file for itself, `load_config` applies the
/// file to this resource at `Startup`, and because both read the same key the reconcile is a no-op
/// rather than a mode change one frame into the run.
// NB: no `Eq` — `shadow_distance` is an f32 (only `PartialEq` is needed, for `!=` change detection).
#[derive(Resource, Clone, Copy, PartialEq, Debug)]
pub(crate) struct VideoConfig {
    pub(crate) vsync: bool,
    /// Whether the STATIC WORLD (trees, buildings, foliage) casts realtime shadows and baked MCSH
    /// terrain shadows are suppressed. Independent of [`Self::character_shadows`] — either drives
    /// the shared shadow rig (`character_shadow` / `world_shadow`).
    pub(crate) world_shadows: bool,
    /// Whether CHARACTERS (players, NPCs, creatures, mounts) cast realtime silhouettes instead of
    /// the legacy oval blob. Independent of [`Self::world_shadows`].
    pub(crate) character_shadows: bool,
    /// Realtime-shadow render distance in yards (the `shadowDistance` slider) — the shadow-map
    /// cascade range + caster reach. Clamped to `shadow_core::SHADOW_DISTANCE_RANGE`.
    pub(crate) shadow_distance: f32,
    /// MONKEY (sun shadow perf): the directional shadow map's edge in texels (`shadowMapSize`;
    /// 1024/2048/4096, default 2048). The rig shipped a 4096 literal, and at 1080p the two sun
    /// lanes measured ~5 ms/frame together — a shadow map is quadratic in this number, so halving
    /// the edge quarters the pass's fill AND the depth texture (4096² D32 = 64 MB, 2048² = 16 MB).
    /// 2048 over one 80 yd cascade is ~26 texels/yd, still finer than the receivers' Gaussian
    /// kernel resolves. Applied to Bevy's `DirectionalLightShadowMap` resource, which
    /// `extract_lights` re-publishes on change and `prepare_lights` re-keys the texture cache with
    /// — so the map is genuinely re-created, live, with no restart.
    pub(crate) shadow_map_size: u32,
    /// MONKEY (sun shadow perf): the receivers' PCF kernel (`shadowFilter`; 0 = Hardware2x2, one
    /// hardware comparison sample, 1 = Gaussian, nine). Default **1** — the current look — because
    /// this one is a taste call the user A/Bs, not a free win: `0` is 9× fewer shadow-map fetches
    /// per lit fragment (the RECEIVER half of the cost, where `shadowMapSize` is the caster half)
    /// at the price of visibly stair-stepped edges. Applied BOTH to the world camera's
    /// `ShadowFilteringMethod` (which keys every Bevy-material receiver — terrain, `wow_model`)
    /// AND to the retained `static_gx` pipeline's shader def, which is specialized by hand and
    /// would otherwise keep whichever branch was compiled in.
    pub(crate) shadow_filter: u32,
    /// MONKEY (sun shadow perf): Hz cap on the CHARACTER lane's proxy re-skin
    /// (`characterShadowRate`, 0..120, default 30; `0` = every frame, the pre-cvar behaviour). The
    /// lane CPU-skins every admitted unit and mutates a `Mesh` asset, which costs a full
    /// vertex+index re-upload — the character lane's ~3 ms. The shadow MAP is still rendered every
    /// frame from the last proxy, so a capped rate does not flicker; it only lets a running NPC's
    /// silhouette lag by up to 1/rate s.
    pub(crate) character_shadow_rate: u32,
    /// MONKEY (sun shadow perf): the same cap for the WORLD lane's per-frame ENTITY caster
    /// (`worldShadowRate`, 0..120, default 30) — gameobjects, distance-faded doodads, WMO props.
    /// A SEPARATE row from [`Self::character_shadow_rate`] on purpose: its population is nearly
    /// static (a swinging lamp, a fading doodad) where the character lane's is animated every
    /// frame, so it tolerates a much lower rate — and one dial named for characters silently
    /// governing the world lane is the kind of thing nobody finds again. The world lane's STATIC
    /// casters are untouched: they already rebuild only on 16 yd camera drift.
    pub(crate) world_shadow_rate: u32,
    /// MONKEY (sun shadow perf): multiplier on the CASTER-COLLECTION reach (`shadowCasterReach`,
    /// 0.25..2, default 1 = unchanged). Collection reaches past the resolve range on purpose (a
    /// tree standing outside the cascade still throws a shadow into it), which at `shadowDistance`
    /// 80 admits 112 yd of entities and up to 204 yd of statics, with `NoFrustumCulling` on the
    /// proxies. Trimming it is the direct dial on caster POPULATION — the input to both lanes'
    /// per-rebuild cost — at the risk of a tall caster's shadow popping in as you approach.
    pub(crate) shadow_caster_reach: f32,
    /// MONKEY (dynamic interiors): WMO interiors + their props light from the room's live fixtures
    /// (`interiorLight`) instead of the baked path. The three knobs are `interiorAmbient` (base
    /// ambient, 0..1), `interiorFill` (per-fixture bounce gain, 0..2) and `interiorExposure`
    /// (light-budget multiplier, 0.25..8) — bridged to benilla-world by `dynamic_interior`.
    pub(crate) interior_light: bool,
    pub(crate) interior_ambient: f32,
    pub(crate) interior_fill: f32,
    pub(crate) interior_exposure: f32,
    /// MONKEY (soft falloff): live scale on every interior fixture's AUTHORED attenuation window
    /// (`interiorAttenScale`, 0..8) — the fixture's EFFECTIVE RADIUS is `authored end × this`. A
    /// WMO MOLT record's `+0x2c` (an M2 source buckets by intensity, its authored pair being a
    /// template default rather than a reach) is a "full brightness ends here" number, not a
    /// "nothing past here" one, so `1` gave a hard-edged disc at exactly the authored end. The
    /// default is **2.5**: a 5 yd candle now tails smoothly out to 12.5, reading ~⅓ of its 1 yd
    /// brightness at the authored 5 and ~8 % at 10. `0` still means "no window" (the 48 yd lane).
    /// Bridged to benilla-world's `DynamicInteriors::atten_scale`, which the light packer folds
    /// into each interior entry's packed radius, so the dial moves the frame it changes.
    pub(crate) interior_atten_scale: f32,
    /// MONKEY (torch shadows, Stage B): whether interior fixtures cast real shadows (the nearest few
    /// promoted to cube-map casters — `torch_shadow`). Only meaningful with `interior_light` on.
    pub(crate) interior_shadows: bool,
    /// MONKEY (outdoor torch shadows): whether EXTERIOR fire lights (campfires, braziers,
    /// lampposts, bonfires — the point table's exterior half, colour row `.w == 0`) cast real
    /// cube-map shadows onto WMO outdoor surfaces, doodads and models AT NIGHT (`exteriorShadows`,
    /// default on). Deliberately NOT gated on `interior_light`: the exterior receivers were never
    /// part of the dynamic-interior feature and draw identically with it off. By day the lane is
    /// inert on both sides — no candidates, no maps, and the receivers' own `night_w` is exactly 0
    /// — so this dial has no daylight effect to have.
    ///
    /// It shares `interior_shadow_casters`' sixteen resident cube slots, capped at half of them
    /// (`torch_shadow::exterior_budget`) so a village square cannot evict an inn's candles.
    pub(crate) exterior_shadows: bool,
    /// MONKEY (static torch cache): resident fixture budget (1..16, default 12). Static
    /// geometry renders only on promotion/residency changes; lowering this fades extra slots out.
    pub(crate) interior_shadow_casters: u32,
    /// MONKEY (static torch cache): nearest promoted fixtures with per-frame entity overlays
    /// (0..16, default 4). Zero keeps all static shadows and disables only the moving casters.
    pub(crate) interior_shadow_dynamic: u32,
    /// MONKEY (torch lane perf): how often (Hz) the moving-caster mesh is REGATHERED
    /// (`interiorShadowEntityRate`, 0..240, default 30; `0` = every frame, the pre-feature
    /// behaviour). The gather CPU-skins every admitted unit inside the dynamic fixtures' reach and
    /// then MUTATES the aggregate `Mesh` asset, which costs a full vertex+index re-extraction and
    /// GPU re-upload plus an `AssetChanged<Mesh3d>` fan-out through material specialisation - a
    /// fixed per-frame charge that neither `interiorShadowCasters` nor `interiorShadowDynamic`
    /// could reduce (both were measured to change nothing). The six overlay passes still run EVERY
    /// frame from the LAST mesh, so lowering this cannot blink a shadow off; it only ages the pose
    /// the mesh was gathered at. At 30 Hz on a 46 fps frame that is "regather about two frames in
    /// three", and a walking NPC's shadow lags its body by at most one frame's stride.
    pub(crate) interior_shadow_entity_rate: u32,
    /// MONKEY (torch caster selection): the PCF tap-radius scale for the torch maps
    /// (`interiorShadowSoft`, 0.5..3, default **1.5**). A candle cluster casts many hard-edged
    /// overlapping shadows; widening the 4-tap kernel is the cheap softening. Rides the torch
    /// table's `count.y` (as `x100`, LOW half) rather than a `DynamicInteriors` field, because it
    /// belongs to the shadow table's own bytes.
    ///
    /// MONKEY (pcss): it is now the CONTACT radius, not the radius everywhere — the projector's
    /// blocker search grows the kernel with the receiver's distance from its caster and clamps at
    /// 4x this. So this dial sets how sharp the sharpest edge in the scene is, and 1 (the old
    /// default) now reads sharper at a contact than it used to read anywhere; 1.5 restores the
    /// shipped softness at a contact and lets the penumbra open up from there.
    pub(crate) interior_shadow_soft: f32,
    /// MONKEY (shadow floor): how much of the DIRECT term a torch shadow removes
    /// (`torchShadowStrength`, 0..1, default **0.7**). A torch map is the only occlusion the
    /// direct arm has, so a blocked fragment used to lose all of it — the pitch-black razor-edged
    /// "scars" the Darkmoon tents printed on the grass and the Darkshire chairs printed on the inn
    /// floor. Nothing in this renderer bounces light, so the 30 % left standing at the default IS
    /// the bounce. Rides the torch table's `count.y` HIGH half beside `interior_shadow_soft`, and
    /// the receivers fold it into the slot's cross-fade weight (one multiply, no extra tap), so it
    /// touches the direct arm only — fill and ambient never saw this factor. `1` restores the
    /// shipped look exactly; `0` turns torch shadows off without disturbing the lane behind them.
    pub(crate) torch_shadow_strength: f32,
    /// MONKEY (room gate): whether an interior fixture may only light the ROOMS IT CLAIMS
    /// (`interiorRoomGate`, default on). Off = the pre-gate behaviour, where every interior fixture
    /// in range lights every interior surface in range and the only occlusion is the handful of
    /// promoted cube-shadow casters — the live A/B for "did the gate darken this room, or was it
    /// always unlit?". Bridged to benilla-world's `DynamicInteriors::room_gate`, which the light
    /// packer applies at PACK time (an ungated pack is one `count = 0` head per light), so it moves
    /// the frame it changes and costs the shader nothing.
    pub(crate) interior_room_gate: bool,
    /// MONKEY (interior debug): the interior-lane diagnostic overlay (`interiorDebug`, 0..4). See
    /// [`benilla_world::lighting::DynamicInteriors::debug`].
    pub(crate) interior_debug: u32,
    /// MONKEY (darkness gains): the exterior night dim (`nightGain`, 0.2..1.5, default **0.8** =
    /// nights 20 % darker). Bridged to `DynamicInteriors::night_gain`, which the light packer folds
    /// into the packed ambient/diffuse/specular rows on a `mix(1, gain, night_w)` ramp — so it is
    /// exactly inert while the sun is up and live the frame it changes after dark.
    pub(crate) night_gain: f32,
    /// MONKEY (lighting debug panel): the interior dim (`interiorGain`, 0.2..1.5, default **0.5** =
    /// room inputs 50 % weaker). Bridged to `DynamicInteriors::interior_gain`, which scales the room
    /// lane's INPUTS (base ambient, per-fixture fill, every interior fixture's colour) and not
    /// `interiorExposure` — that stays the user's own dial, and this composes with it.
    pub(crate) interior_gain: f32,
    /// MONKEY (enclosed day floor): the DAYLIGHT floor a room inside a building gets by day
    /// (`interiorDaylight`, 0..1, default **0.12**). Bridged to
    /// [`benilla_world::lighting::DynamicInteriors::daylight`], packed into the free fraction of
    /// the interior lane's on/off word, and added to the room law's ambient budget for batches the
    /// record table flags as enclosed. `0` restores the pre-feature look exactly; the night look is
    /// unaffected at any value (the term is scaled by the sun's own day envelope).
    pub(crate) interior_daylight: f32,
    /// MONKEY (bake floor): the share of a WMO interior batch's OWN MOCV bake every interior-lane
    /// fragment keeps whether or not a fixture reaches it (`interiorBakeFloor`, 0..1, default
    /// **0.12**). Bridged to [`benilla_world::lighting::DynamicInteriors::bake_floor`], packed
    /// (times `interiorGain`) into the free fraction of the world-shadow lane, and added to the
    /// room law's budget inside its rolloff. It is what stops a room the fixture table cannot
    /// reach — the Lion's Pride Inn's east vestibule — rendering black; `0` restores the
    /// pre-feature look exactly.
    pub(crate) interior_bake_floor: f32,
    /// MONKEY (fire GO lights): gain on every light SYNTHESISED from a fire prop's flame emitter
    /// (`fireLightGain`, 0..4; `0` = the invented-light lane off). Bridged to benilla-world's
    /// [`benilla_world::lighting::FireLightGain`] by `dynamic_interior`, and applied at PACK time
    /// so it is live.
    pub(crate) fire_light_gain: f32,
    /// MONKEY (spellLightGain): gain on every light a SPELL EFFECT invented — a kit's aura glow, a
    /// missile's core, an impact flash, a firework shell's burst (`spellLightGain`, 0..4; `0` = the
    /// spell-light lane off). Bridged to benilla-world's
    /// [`benilla_world::lighting::SpellLightGain`] by `dynamic_interior` and applied at PACK time,
    /// so it is live. Deliberately NOT folded into `fireLightGain`: a spell light is tagged
    /// synthetic too, and one dial over both would mean turning the world's hearths down darkened
    /// every fireball in the game.
    pub(crate) spell_light_gain: f32,
    /// MONKEY (flame flicker): how strongly every FLAME's brightness wobbles (`fireFlicker`, 0..2;
    /// `1` = the authored per-kind amplitudes, `0` = the pre-feature steady constants, `2` =
    /// doubled). Bridged to `DynamicInteriors::flicker` and applied at PACK time, so it is live —
    /// and separate from `fireLightGain`, which scales only the SYNTHESISED lane while a flicker
    /// belongs to authored wall torches too.
    pub(crate) fire_flicker: f32,
    pub(crate) display: DisplayMode,
    /// The windowed size, `gxResolution`. Kept while fullscreen so leaving it can restore it.
    pub(crate) windowed: UVec2,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            vsync: !novsync_env(),
            world_shadows: true,
            character_shadows: true,
            shadow_distance: crate::shadow_core::DEFAULT_SHADOW_DISTANCE,
            // MONKEY (sun shadow perf): 2048, not the rig's old 4096 literal — see the field docs.
            shadow_map_size: crate::shadow_core::DEFAULT_SHADOW_MAP_SIZE,
            shadow_filter: crate::shadow_core::DEFAULT_SHADOW_FILTER,
            character_shadow_rate: crate::shadow_core::DEFAULT_SHADOW_RATE,
            world_shadow_rate: crate::shadow_core::DEFAULT_SHADOW_RATE,
            shadow_caster_reach: 1.0,
            // The cvar defaults are the source of truth at load; these only stand in until then.
            interior_light: true,
            interior_ambient: 0.015,
            interior_fill: 0.08,
            interior_exposure: 2.5,
            // MONKEY (soft falloff): 2.5, not 1 — see the field doc.
            interior_atten_scale: 1.6,
            interior_shadows: true,
            // MONKEY (outdoor torch shadows): on — a night campfire with no shadow is the thing
            // this lane exists to fix, and it costs nothing whenever the sun is up.
            exterior_shadows: true,
            // MONKEY (static torch cache): 12 resident maps, four moving-caster overlays.
            interior_shadow_casters: 12,
            interior_shadow_dynamic: 4,
            // MONKEY (torch lane perf): 30 Hz - see the field doc.
            interior_shadow_entity_rate: 30,
            // MONKEY (pcss): 1.5 — see the field doc; `soft` is now the CONTACT radius.
            interior_shadow_soft: 1.5,
            // MONKEY (shadow floor): 0.7 — a shadow takes 70 % of the direct term, not all of it.
            torch_shadow_strength: 0.7,
            // MONKEY (room gate): on — without it a building's fixtures light through its own
            // floors and walls.
            interior_room_gate: true,
            interior_debug: 0,
            // MONKEY (lighting debug panel): nights 20 % darker, interior inputs 50 % weaker.
            night_gain: 0.45,
            interior_gain: 0.5,
            // MONKEY (enclosed day floor): calibrated so the Goldshire inn's entry floor reads
            // ~50 % of the sunlit threshold beside it — see `lighting::DAYLIGHT_LANE_SCALE`.
            interior_daylight: 0.0,
            // MONKEY (bake floor): an eighth of the authored bake — measured to lift the inn's
            // black door band from 0.019 to 0.108 x tex while moving candle-lit surfaces by
            // under 10 % (see `lighting::DynamicInteriors::bake_floor`).
            interior_bake_floor: 0.12,
            fire_light_gain: 1.0,
            spell_light_gain: 1.0,
            fire_flicker: 1.0,
            display: if windowed_env() {
                DisplayMode::Windowed
            } else {
                DisplayMode::default()
            },
            windowed: DEFAULT_WINDOWED,
        }
    }
}

/// The present mode a vsync setting means.
///
/// **On is `PresentMode::default()` — Bevy's default, deliberately, not `AutoVsync`.** 0294 recorded
/// running the engine default and that is `Fifo`: strict, never tears. `AutoVsync` is *not* a
/// synonym — it resolves to `FifoRelaxed` where available, which permits a tear on a late frame.
/// The dev HUD's old checkbox wrote `AutoVsync` on the way back on, so toggling it off and on
/// silently left the client on a different mode than it booted with; naming one function the single
/// mapping is what closes that.
pub(crate) fn present_mode(vsync: bool) -> PresentMode {
    if vsync {
        PresentMode::default()
    } else {
        PresentMode::AutoNoVsync
    }
}

pub(crate) struct VideoPlugin;

impl Plugin for VideoPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VideoConfig>()
            .init_resource::<GxRestarts>()
            .add_systems(Startup, (log_display_session, check_window_pinned).chain())
            .add_systems(
                Update,
                (
                    (drain_restart_gx, (apply_present_mode, apply_window_mode)).chain(),
                    publish_display_modes,
                ),
            );
    }
}

/// How many `RestartGx()` calls the interface has made — the video window's "apply the staged
/// settings now" (decision 2177).
///
/// A generation counter rather than a flag: [`apply_present_mode`] and [`apply_window_mode`] each
/// keep their own `Local` of the last value they acted on, so one bump forces exactly one
/// re-assertion in each, whichever order they run in and however many frames apart.
#[derive(Resource, Default, Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct GxRestarts(u32);

/// Move the interface's `RestartGx()` calls into [`GxRestarts`].
///
/// benilla applies its video settings live (this module's doc says so for `gxVSync` and the
/// display mode), so what a restart means HERE is "re-assert them against the window now" rather
/// than "tear the device down and rebuild it". The two systems below do the asserting; this one
/// only carries the request across the VM boundary.
fn drain_restart_gx(
    script: Option<NonSendMut<benilla_ui::script::UiScript>>,
    mut restarts: ResMut<GxRestarts>,
) {
    let Some(mut script) = script else {
        return;
    };
    let asks = script.take_restart_gx_asks();
    if asks == 0 {
        // Never touch the resource on a quiet frame: a `ResMut` deref-mut is a change signal, and
        // both consumers below are gated on the value moving.
        return;
    }
    restarts.0 = restarts.0.wrapping_add(asks);
    info!("video: RestartGx — re-asserting the display mode and present mode");
}

/// **The reference's own three filters on the resolution list** (wow-re
/// `ui/scratch/video-options-verbs.md` §1.1, all VERIFIED at `0x48bcfa`–`0x48bd18`), in its own
/// order: keep iff `w/h >= 1.248`, `w >= 800`, `h >= 600`.
///
/// The aspect constant is `[0x804570]`, the f32 `1.2480000257492065` — chosen just under 5:4 so it
/// admits 5:4, 4:3, 16:10 and 16:9 and rejects square and portrait modes. It is not a
/// widescreen-only gate; the `widescreen` CVar is a separate switch [`SCREEN_FALLBACK`] describes.
fn offerable(r: ScreenResolution) -> bool {
    r.width >= 800 && r.height >= 600 && f64::from(r.width) / f64::from(r.height) >= 1.248
}

/// **What the list is when enumeration produces nothing** — the reference's four hardcoded modes,
/// in its own append order (`0x48bda2`, `0x48bddf`, `0x48be43`, `0x48bea7`).
///
/// In the reference this is a "produced nothing" path, not an else: it is taken when the display
/// enumeration survives no mode, when the `widescreen` CVar's record is missing, or when
/// `widescreen == 0`. **benilla does not register `widescreen`** (`0x63a747`, registered default
/// `"1"`), so only the first of the three reaches here — a headless run, or a monitor whose modes
/// all fail [`offerable`]. Registering it would be a knob whose whole effect is to shrink this
/// dropdown to these four, which is a setting on its own merits and not this list's business.
const SCREEN_FALLBACK: [ScreenResolution; 4] = [
    ScreenResolution {
        width: 800,
        height: 600,
    },
    ScreenResolution {
        width: 1024,
        height: 768,
    },
    ScreenResolution {
        width: 1280,
        height: 1024,
    },
    ScreenResolution {
        width: 1600,
        height: 1200,
    },
];

/// The pair [`publish_display_modes`] remembers between frames: the list it last pushed and the
/// index into it. One name because they are one fact — a list without its index cannot be read —
/// and because a `Local` spelling it inline is what `-D clippy::type-complexity` refuses.
type PublishedModes = Option<(Vec<ScreenResolution>, Option<ScreenResolution>)>;

/// **What the Video options window's resolution dropdown offers, and where the client is in it** —
/// the host half of `GetScreenResolutions` / `GetCurrentResolution` / `SetScreenResolution`
/// (decision 2177).
///
/// The reference enumerates the graphics device's display modes, because picking one is a
/// mode-set. benilla ships no exclusive mode at all (this module's doc walks why, per target), so
/// what a pick here really changes is the **windowed size** — `gxResolution`, which
/// `SetScreenResolution` writes and [`apply_window_mode`] applies on the next frame.
///
/// **The unit is LOGICAL pixels, not the monitor's physical mode table, and that is deliberate.**
/// `gxResolution` already means a logical inner size everywhere else in this client
/// ([`boot_windowed_size`] hands it straight to `WindowResolution`), so offering physical sizes
/// would make the dropdown's rows and the CVar they write mean two different things on every
/// HiDPI display — a 1600×900 pick landing a 3200×1800 window. One unit end to end beats matching
/// the reference's spelling into a variable that means something else here.
///
/// The rows are the monitor's own distinct mode sizes plus its full size, in logical units, put
/// through [`offerable`] — device data and the reference's own filters, not a ladder we invented —
/// and [`SCREEN_FALLBACK`] when that survives nothing. The live window size is added by
/// [`UiScript::set_screen_resolutions`] if it is not already among them, which is the common case
/// (a 1600×900 window on a 4K panel, or anything below the 800×600 floor) and the one
/// `CT_Viewport.lua:201` depends on: it reads its own screen size as `arg[GetCurrentResolution()]`
/// and silently falls back to 4:3 on a miss.
///
/// **Recomputed on change, where the reference builds it once and never invalidates it** — its own
/// list survives a `gxRestart`, a `widescreen` toggle and a monitor change (a 25-hit dword census
/// of the count global says so). That is a cache bug to leave behind, not a mechanism: a client
/// that can be dragged between monitors has to answer for the one it is on.
fn publish_display_modes(
    script: Option<NonSendMut<benilla_ui::script::UiScript>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    monitors: Query<&bevy::window::Monitor>,
    mut last: Local<crate::ui_script::VmMemo<PublishedModes>>,
) {
    let Some(mut script) = script else {
        return;
    };
    let Ok(window) = windows.single() else {
        return;
    };
    let res = &window.resolution;
    let current = Some(ScreenResolution {
        width: res.width() as u32,
        height: res.height() as u32,
    })
    .filter(|r| r.width > 0 && r.height > 0);
    let mut offered: Vec<ScreenResolution> = Vec::new();
    for m in &monitors {
        // The monitor's own scale — not the window's. A window straddling two displays reports
        // whichever winit last gave it, and these rows describe the PANEL.
        let scale = if m.scale_factor > 0.0 {
            m.scale_factor
        } else {
            1.0
        };
        let logical = |size: UVec2| ScreenResolution {
            width: (size.x as f64 / scale).round() as u32,
            height: (size.y as f64 / scale).round() as u32,
        };
        offered.push(logical(m.physical_size()));
        offered.extend(m.video_modes.iter().map(|v| logical(v.physical_size)));
    }
    offered.retain(|r| offerable(*r));
    if offered.is_empty() {
        offered.extend(SCREEN_FALLBACK);
    }
    offered.sort_by_key(|r| (u64::from(r.width) * u64::from(r.height), r.width, r.height));
    offered.dedup();
    // **VM-keyed** (decision 1290): a `ReloadUI` replaces the VM, and the fresh one has been
    // pushed nothing. A plain `Local` here would remember the OLD VM's list and skip the push
    // that the new VM needs, leaving `GetScreenResolutions` empty for the rest of the session.
    let memo = last.get(&script);
    if memo.as_ref() == Some(&(offered.clone(), current)) {
        return;
    }
    *memo = Some((offered.clone(), current));
    script.set_screen_resolutions(offered, current);
}

/// **Did the window actually get the size `$WOW_WIN` asked for?** Refuse the run if not.
///
/// The window manager is free to clamp a requested inner size to the display, and macOS does. The
/// failure is silent and it invalidates comparisons: on 2026-08-26 an MSAA A/B produced one leg at
/// 3200x1800 and the other at 3024x1800, because the director had the laptop panel on overnight
/// instead of the external monitor, and the window landed on a smaller screen. Nothing said so.
/// `benilla-visual` refused the pair with `image size mismatch` — the right outcome by luck, from a
/// tool three steps downstream that could only report the symptom, and a full A/B cycle was spent
/// getting there.
///
/// Measured, not assumed: `WOW_WIN=4000x3000` on this machine yields `1920x1048 logical` — clamped
/// to the display the window happened to open on, minus its chrome.
///
/// **Fatal under a capture, a warning otherwise**, and the split is the point. Every visual
/// regression diff in the tree is denominated in the window the scenario asks for
/// ([`windowed_env`]'s doc says so), so a capture at the wrong size is not a worse capture, it is
/// an invalid one that will be diffed against a valid one. A non-capture run with `$WOW_WIN` is
/// somebody looking at something; a clamp there is surprising, not wrong, so it says so and
/// carries on.
fn check_window_pinned(
    windows: Query<&Window, With<PrimaryWindow>>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(want) = requested_window_size() else {
        return;
    };
    let Ok(window) = windows.single() else {
        return;
    };
    // `$WOW_DPI` re-denominates `$WOW_WIN` in PHYSICAL px (that knob's doc says so: the capture is
    // the framebuffer that player's GPU would scan out), so the size this compares has to follow
    // it. Comparing the logical size under an override refused every non-1× run outright — it read
    // `want` against `want/dpi`.
    let res = &window.resolution;
    let got = match requested_dpi() {
        Some(_) => UVec2::new(res.physical_width(), res.physical_height()),
        None => UVec2::new(res.width() as u32, res.height() as u32),
    };
    if got == want {
        return;
    }
    let unit = if requested_dpi().is_some() {
        "physical"
    } else {
        "logical"
    };
    let capturing =
        std::env::var_os("WOW_CAPTURE").is_some() || std::env::var_os("WOW_CAPTURE_UI").is_some();
    if !capturing {
        warn!(
            "window: asked for {}x{} {unit}, got {}x{} — the window manager clamped it to the \
             display. Harmless here; it would invalidate a capture.",
            want.x, want.y, got.x, got.y
        );
        return;
    }
    error!(
        "window: REFUSING this capture — asked for {}x{} {unit}, got {}x{}. The window manager \
         clamped the request to the display this window opened on, so the image would not be the \
         size the scenario is denominated in and any diff against it would be meaningless. Use a \
         size that fits the current display (or move the window to a bigger one) and re-run.",
        want.x, want.y, got.x, got.y
    );
    exit.write(AppExit::error());
}

/// One line at boot naming what the window actually got — and, on a Linux session, what it is
/// talking to.
///
/// **The instrument half of 1627**, and it is not decoration. Every display and input report this
/// client has had from a Linux player came from a machine nobody here can run: the Steam Deck
/// perf reports behind 1624 and 1626, and the gamescope input report 1627 answers, were all
/// diagnosed from prose. `bevy_winit` already logs the monitor, its scale factor and its refresh
/// rate at creation; the two things it never says are which **backend** winit picked and whether a
/// nested compositor is in the way — and a surprising amount hangs on the first of those.
/// `CursorGrabMode::Locked`, which is what mouse-look asks for, is a real pointer lock on Wayland
/// and is **rejected outright on X11** (`winit`'s x11 `set_cursor_grab` returns `NotSupported`),
/// where `bevy_winit::attempt_grab` quietly falls back to `Confined`. "Which one am I on" should
/// never again be a thing we reason about instead of read.
///
/// Not `#[cfg(feature = "dev")]`: the whole point is that it is in the log a *player* pastes.
fn log_display_session(windows: Query<&Window, With<PrimaryWindow>>) {
    let Ok(window) = windows.single() else {
        return;
    };
    let res = &window.resolution;
    info!(
        "video: {:?}, {}x{} logical / {}x{} physical (scale {}){}",
        window.mode,
        res.width(),
        res.height(),
        res.physical_width(),
        res.physical_height(),
        res.scale_factor(),
        display_session(),
    );
}

/// The display-server facts worth naming, as a trailing clause. Empty everywhere but a Linux/BSD
/// session, where the same `cfg` the Wayland clipboard uses (decision 0702) marks the one platform
/// whose windowing backend is a runtime choice rather than the only one there is.
fn display_session() -> String {
    #[cfg(all(unix, not(any(target_os = "macos", target_os = "android"))))]
    {
        let set = |k: &str| std::env::var_os(k).is_some();
        // Which backend winit took: it prefers Wayland when `WAYLAND_DISPLAY` names a socket and
        // falls back to X11 (which, under a Wayland compositor, means XWayland). Both features are
        // on — `bevy`'s `default_platform` carries `x11` *and* `wayland`, so this really is a
        // runtime pick and not a build-time one.
        let backend = match (set("WAYLAND_DISPLAY"), set("DISPLAY")) {
            (true, _) => "wayland",
            (false, true) => "x11",
            (false, false) => "none",
        };
        // gamescope exports its own socket name to children; the SteamOS session also stamps the
        // Deck. Named because a nested compositor is the difference between "the window is wrong"
        // and "the window is right and the compositor is scaling it".
        let nested = if set("GAMESCOPE_WAYLAND_DISPLAY") {
            ", gamescope"
        } else {
            ""
        };
        let deck = if set("SteamDeck") { ", steamdeck" } else { "" };
        format!(
            " [{backend}{nested}{deck}, XDG_SESSION_TYPE={}]",
            std::env::var("XDG_SESSION_TYPE").unwrap_or_else(|_| "unset".into()),
        )
    }
    #[cfg(not(all(unix, not(any(target_os = "macos", target_os = "android")))))]
    String::new()
}

/// Push the setting to the window **when the setting moves**, and only then.
///
/// The value compare is the `rescatter_clutter` pattern (0992), and it is load-bearing for two
/// separate reasons. `Res::is_changed()` over-fires: the cvar sync builds its `Knobs` bundle by
/// deref-mutting *every* knob resource, so any CVar write anywhere flags this one. And re-asserting
/// the setting every frame would fight the capture probes, which write `present_mode` on the window
/// directly mid-run (`capture::mod`, `probes::live_fps`) to uncap a measurement — their override
/// has to stick.
///
/// First sight deliberately does **not** just arm: `load_config` applies the saved value at
/// `Startup`, after the window already exists at its boot mode, so the first run is the one that
/// reconciles them.
fn apply_present_mode(
    cfg: Res<VideoConfig>,
    restarts: Res<GxRestarts>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
    mut last: Local<Option<bool>>,
    mut last_restart: Local<u32>,
) {
    // A `RestartGx()` re-asserts even when nothing moved — that is what the caller asked for.
    let forced = std::mem::replace(&mut *last_restart, restarts.0) != restarts.0;
    if last.replace(cfg.vsync) == Some(cfg.vsync) && !forced {
        return;
    }
    let want = present_mode(cfg.vsync);
    let Ok(mut window) = windows.single_mut() else {
        return;
    };
    // Guarded so an unchanged mode does not deref-mut `Window` — a spurious change there is a
    // surface reconfigure.
    if window.present_mode != want {
        window.present_mode = want;
        info!(
            "video: vsync {} ({want:?})",
            if cfg.vsync { "on" } else { "off" }
        );
    }
}

/// The display-mode half, on the same change-gated shape as [`apply_present_mode`] and for the same
/// two reasons — with one wrinkle of its own.
///
/// **The guard asks whether the window is fullscreen, never on which monitor.** Birth uses
/// `MonitorSelection::Primary` (nothing else has an answer yet) and a live toggle uses `Current`
/// (fill the monitor the window is actually on), so the two `WindowMode`s that both mean
/// "fullscreen" are not equal values, and a `!=` compare would re-assert fullscreen on the first
/// frame of every launch. Matching on the variant is the honest question.
fn apply_window_mode(
    cfg: Res<VideoConfig>,
    restarts: Res<GxRestarts>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
    mut last: Local<Option<DisplayMode>>,
    mut last_restart: Local<u32>,
) {
    // A `RestartGx()` re-asserts even when nothing moved, and that includes re-applying
    // `gxResolution` to a window already in windowed mode — the one video setting whose value can
    // have moved without the MODE moving, and therefore the one this verb is most useful for.
    let forced = std::mem::replace(&mut *last_restart, restarts.0) != restarts.0;
    if last.replace(cfg.display) == Some(cfg.display) && !forced {
        return;
    }
    let Ok(mut window) = windows.single_mut() else {
        return;
    };
    let want = window_mode(cfg.display, MonitorSelection::Current);
    let already = matches!(
        (&window.mode, &want),
        (WindowMode::Windowed, WindowMode::Windowed)
            | (
                WindowMode::BorderlessFullscreen(_),
                WindowMode::BorderlessFullscreen(_)
            )
    );
    if already && !forced {
        return;
    }
    // Leaving fullscreen has to hand the size back, because entering it **overwrote**
    // `window.resolution` with the monitor's — `bevy_window`'s own documented behaviour for
    // `BorderlessFullscreen` — so a bare mode flip returns a monitor-sized "window". `gxResolution`
    // is the record of what windowed means, which is exactly why it persists.
    //
    // Both writes land in one frame on purpose: `bevy_winit::changed_windows` applies `mode` before
    // `resolution` in the same pass, so the inner size is set after the window has left fullscreen.
    if cfg.display == DisplayMode::Windowed {
        window
            .resolution
            .set(cfg.windowed.x as f32, cfg.windowed.y as f32);
    }
    window.mode = want;
    info!("video: display mode {:?} ({want:?})", cfg.display);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped default is synced, and "synced" is the engine default 0294 chose — `Fifo`, not
    /// `AutoVsync`. Written as an assertion because the two are easy to conflate and are not the
    /// same: `AutoVsync` resolves to `FifoRelaxed` where available, which tears on a late frame.
    #[test]
    fn the_default_is_synced_like_the_window_literal() {
        assert!(VideoConfig::default().vsync);
        assert_eq!(present_mode(true), PresentMode::default());
        assert_eq!(present_mode(true), PresentMode::Fifo);
    }

    /// Uncapped is `AutoNoVsync` and never `Immediate` — the Metal finding this module exists to
    /// hold (`Immediate` rails and takes ~1 s `nextDrawable` stalls).
    #[test]
    fn uncapped_is_autonovsync_not_immediate() {
        assert_eq!(present_mode(false), PresentMode::AutoNoVsync);
    }

    /// The shipped display mode is fullscreen, and fullscreen is **borderless** — the whole point
    /// of 1627. Welded as an assertion because `WindowMode::Fullscreen` is one word away and is the
    /// mode that is ignored on Wayland and panics without a monitor.
    #[test]
    fn the_default_is_borderless_fullscreen_not_exclusive() {
        assert_eq!(DisplayMode::default(), DisplayMode::Fullscreen);
        assert!(matches!(
            window_mode(DisplayMode::Fullscreen, MonitorSelection::Primary),
            WindowMode::BorderlessFullscreen(_)
        ));
        assert_eq!(
            window_mode(DisplayMode::Windowed, MonitorSelection::Primary),
            WindowMode::Windowed
        );
    }

    /// `gxWindow` reads like every other 0/1 CVar in the tree: int + `!= 0`, with `"1"` the
    /// *windowed* state — the reference's polarity, which is why the row is "Windowed Mode" and
    /// not "Fullscreen".
    #[test]
    fn gxwindow_one_is_windowed() {
        assert_eq!(display_from_flag(0.0), DisplayMode::Fullscreen);
        assert_eq!(display_from_flag(1.0), DisplayMode::Windowed);
    }

    /// The mode round-trip, driven through the **real system** rather than around it — because the
    /// two things that are easy to get wrong here are both invisible to a single live run.
    ///
    /// 1. A window born fullscreen must not be re-asserted on the first frame. Birth carries
    ///    `MonitorSelection::Primary` and a live toggle carries `Current`, so the two `WindowMode`s
    ///    that both mean "fullscreen" are unequal *values*; a `!=` guard would fire every launch.
    /// 2. Leaving fullscreen must hand `gxResolution` back. Entering it overwrote
    ///    `window.resolution` with the monitor's, so a bare mode flip returns a monitor-sized
    ///    "window" — and on the 3440-wide panel B242/1619 came from, that is not a subtle miss.
    #[test]
    fn the_fullscreen_round_trip_keeps_the_monitor_out_of_the_windowed_size() {
        let mut app = App::new();
        app.insert_resource(VideoConfig {
            vsync: true,
            world_shadows: false,
            character_shadows: false,
            shadow_distance: 80.0,
            display: DisplayMode::Fullscreen,
            windowed: UVec2::new(1024, 768),
            // MONKEY (review fixes): this window test inherits unrelated lighting defaults.
            ..Default::default()
        })
        // The plugin's resource, seated by hand because this test runs the one system rather than
        // the plugin — `apply_window_mode` reads it to know a `RestartGx()` asked for a re-assert.
        .init_resource::<GxRestarts>()
        .add_systems(Update, apply_window_mode);
        let win = app
            .world_mut()
            .spawn((Window::default(), PrimaryWindow))
            .id();
        // Born the way `lib.rs` builds it: already fullscreen, on the boot monitor selection.
        app.world_mut()
            .entity_mut(win)
            .get_mut::<Window>()
            .unwrap()
            .mode = window_mode(DisplayMode::Fullscreen, MonitorSelection::Primary);

        app.update();
        assert!(
            matches!(
                app.world().entity(win).get::<Window>().unwrap().mode,
                WindowMode::BorderlessFullscreen(MonitorSelection::Primary)
            ),
            "an already-fullscreen window must not be re-asserted onto another monitor selection"
        );

        // Stand in for the compositor's half of being fullscreen — `bevy_window` documents that
        // the resolution becomes the monitor's — then leave.
        app.world_mut()
            .entity_mut(win)
            .get_mut::<Window>()
            .unwrap()
            .resolution
            .set(3440.0, 1440.0);
        app.world_mut().resource_mut::<VideoConfig>().display = DisplayMode::Windowed;
        app.update();
        let w = app.world().entity(win).get::<Window>().unwrap();
        assert_eq!(w.mode, WindowMode::Windowed);
        assert_eq!(
            (w.resolution.width(), w.resolution.height()),
            (1024.0, 768.0),
            "leaving fullscreen restores gxResolution, never the monitor's size"
        );

        // And back in, which must reach `Current` — the monitor the window is on by then, not the
        // one it was born on.
        app.world_mut().resource_mut::<VideoConfig>().display = DisplayMode::Fullscreen;
        app.update();
        assert!(matches!(
            app.world().entity(win).get::<Window>().unwrap().mode,
            WindowMode::BorderlessFullscreen(MonitorSelection::Current)
        ));
    }

    /// `gxResolution` is the reference's `"WxH"` string, and a value that cannot be a window is
    /// refused rather than passed on.
    #[test]
    fn gxresolution_parses_the_reference_spelling() {
        assert_eq!(parse_resolution("1280x800"), Some(UVec2::new(1280, 800)));
        assert_eq!(parse_resolution("1600X900"), Some(UVec2::new(1600, 900)));
        assert_eq!(parse_resolution("1024 x 768"), Some(UVec2::new(1024, 768)));
        assert_eq!(parse_resolution("0x600"), None);
        assert_eq!(parse_resolution("1280x"), None);
        assert_eq!(parse_resolution("fullscreen"), None);
    }
}
