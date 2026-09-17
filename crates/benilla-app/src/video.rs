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

/// **The Video Options block's change callbacks** (decision 2303) — the rows the reference
/// registers from its one video-options registration block (`0x688470`, wow-re
/// `cvar/scratch/graphics-cost-cvar-census.md` §2), landing on the resources they drive. Each
/// arm writes only its own resource, so a `ViewDistance` change is `farclip` moving and nothing
/// else; the clamps are each row's own, stated beside it.
pub(crate) fn on_cvar(
    ev: On<crate::cvars::CvarChanged>,
    mut cfg: ResMut<VideoConfig>,
    mut view: ResMut<benilla_world::view::ViewDistance>,
    mut msaa: ResMut<benilla_world::view::MsaaSetting>,
    msaa_formats: Res<benilla_world::view::MsaaFormats>,
    mut tex_filter: ResMut<benilla_assets::TexFilterSetting>,
    mut clutter: ResMut<benilla_world::clutter::ClutterConfig>,
    mut weather: ResMut<benilla_world::weather::WeatherState>,
    mut cvars: ResMut<crate::cvars::Cvars>,
) {
    use benilla_world::view::{FARCLIP_RANGE, MSAA_RANGE};
    let v = ev.num();
    match ev.key().as_str() {
        // The one string row here (1627), composed in the reference's own `WxH` spelling. A
        // value that is not a size is consumed with a warn and the window keeps its truth.
        "gxresolution" => match parse_resolution(&ev.new) {
            Some(size) => cfg.windowed = size,
            None => warn!("cvar gxResolution: unparseable value '{}' ignored", ev.new),
        },
        // Vertical Sync — a flag like every other checkbox. `apply_present_mode` pushes it to
        // the window when this resource moves; nothing else reads it.
        "gxvsync" => cfg.vsync = ev.flag(),
        // Display mode (1627) — the reference's own polarity: `1` is WINDOWED (the row is
        // "Windowed Mode"). `apply_window_mode` pushes it to the window when this moves.
        "gxwindow" => cfg.display = display_from_flag(v),
        // ── MONKEY (lighting): the dynamic light + shadow system's 29 rows ────────────────────
        // They live in THIS observer, and not in one of their own beside `shadow_core` /
        // `dynamic_interior`, because of the law the arm above states: *each arm writes only its
        // own resource*. Every one of these knobs IS a field of [`VideoConfig`] — the lanes read
        // that resource per frame (`shadow_core::update_shadows`, `dynamic_interior::bridge`,
        // `torch_shadow`), none of them owns a resource of its own — so a second observer beside
        // them would be a second writer of this one resource for no gain, splitting one match
        // over two files while dirtying exactly the same thing.
        //
        // What it DOES cost is the precision of `Res<VideoConfig>::is_changed()`: on upstream's
        // struct that signal means "a Video Options row moved", and here it means "a video OR a
        // lighting row moved". The one consumer that cares is `apply_present_mode`, which keeps
        // 2303's retired value compare for exactly this reason — see its doc.
        //
        // Clamps are each row's own, stated beside it, exactly as for the reference rows above;
        // the `ours(...)` entries in `cvars::REGISTERED` carry the matching defaults, and
        // `cvars::tests::registered_defaults_mirror_the_code_truths` welds all 29 pairs.
        "worldshadows" => cfg.world_shadows = ev.flag(),
        "charactershadows" => cfg.character_shadows = ev.flag(),
        "shadowdistance" => {
            cfg.shadow_distance = v.clamp(
                *crate::shadow_core::SHADOW_DISTANCE_RANGE.start(),
                *crate::shadow_core::SHADOW_DISTANCE_RANGE.end(),
            )
        }
        // MONKEY (sun shadow perf): the five cost dials, clamped at the edge like every numeric row
        // here. `shadowMapSize` SNAPS onto the power-of-two ladder rather than clamping into a
        // range — an off-ladder value is not a weaker setting, it is one Bevy silently rounds UP
        // into a bigger and slower map than the one that was typed.
        "shadowmapsize" => {
            cfg.shadow_map_size = crate::shadow_core::clamp_shadow_map_size(v.max(0.0) as u32)
        }
        "shadowfilter" => {
            cfg.shadow_filter = (v.max(0.0) as u32).min(crate::shadow_core::MAX_SHADOW_FILTER)
        }
        // `0` is MEANINGFUL on both rate rows (the pre-cvar every-frame rebuild), so they floor at
        // 0 rather than at 1 — the shadow off-switches are `characterShadows` / `worldShadows`.
        "charactershadowrate" => {
            cfg.character_shadow_rate =
                (v.max(0.0) as u32).min(crate::shadow_core::MAX_SHADOW_RATE)
        }
        "worldshadowrate" => {
            cfg.world_shadow_rate = (v.max(0.0) as u32).min(crate::shadow_core::MAX_SHADOW_RATE)
        }
        "shadowcasterreach" => {
            cfg.shadow_caster_reach = v.clamp(
                *crate::shadow_core::CASTER_REACH_RANGE.start(),
                *crate::shadow_core::CASTER_REACH_RANGE.end(),
            )
        }
        // MONKEY (dynamic interiors): the interior lane's on/off + knobs, clamped at the edge like
        // every other numeric row. `dynamic_interior::bridge` publishes them to benilla-world.
        "interiorlight" => cfg.interior_light = ev.flag(),
        "interiorambient" => cfg.interior_ambient = v.clamp(0.0, 1.0),
        "interiorfill" => cfg.interior_fill = v.clamp(0.0, 2.0),
        "interiorexposure" => cfg.interior_exposure = v.clamp(0.25, 8.0),
        // MONKEY (interior attenuation): the authored-window scale. `0` is a MEANINGFUL value here
        // (the window off), so the range floors at 0 rather than at a small positive.
        "interiorattenscale" => cfg.interior_atten_scale = v.clamp(0.0, 8.0),
        "interiorroomgate" => cfg.interior_room_gate = ev.flag(),
        "interiorshadows" => cfg.interior_shadows = ev.flag(),
        // MONKEY (outdoor torch shadows): a flag like every other checkbox here. Live — the lane
        // reads `VideoConfig` every frame, so `0` fades the outdoor shadows out (the slots evict
        // through the same cross-fade a walked-away fixture does) and `1` fades them back in.
        "exteriorshadows" => cfg.exterior_shadows = ev.flag(),
        // MONKEY (torch caster selection): the working-set size and the PCF radius, clamped at the
        // edge like every other numeric row. `casters` floors at 1, not 0 — `interiorShadows 0` is
        // already the off switch, and a 0 here would be a second, confusing one.
        "interiorshadowcasters" => cfg.interior_shadow_casters = (v.max(1.0) as u32).clamp(1, 16),
        // MONKEY (static torch cache): the live bank rank, 1..`MAX_TORCH_DYNAMIC`.
        "interiorshadowdynamic" => {
            cfg.interior_shadow_dynamic =
                (v.max(1.0) as u32).clamp(1, crate::torch_shadow::MAX_TORCH_DYNAMIC as u32)
        }
        // MONKEY (torch lane perf): the moving-caster regather cadence in Hz. `0` is MEANINGFUL
        // here (every frame -- the behaviour before the gate), so unlike `casters` this floors at
        // 0 rather than at 1. Ceiling 240 so a typo cannot ask for a per-frame rebuild AND a
        // divide by a huge number; anything at or above the frame rate is already "every frame".
        "interiorshadowentityrate" => {
            cfg.interior_shadow_entity_rate = (v.max(0.0) as u32).min(240)
        }
        "interiorshadowsoft" => cfg.interior_shadow_soft = v.clamp(0.5, 3.0),
        // MONKEY (shadow floor): 0 IS meaningful (shadows off), so this floors at 0, not at a
        // minimum-useful value; 1 is the pre-feature pitch black.
        "torchshadowstrength" => cfg.torch_shadow_strength = v.clamp(0.0, 1.0),
        "interiordebug" => cfg.interior_debug = (v.max(0.0) as u32).min(4),
        // MONKEY (darkness gains): the two dim dials, clamped at the edge like every numeric row
        // here. The floor is 0.2 rather than 0: a true 0 would be indistinguishable from a broken
        // light pack (black world / black room), and the off switch people actually want is `1`.
        "nightgain" => cfg.night_gain = v.clamp(0.2, 1.5),
        "interiorgain" => cfg.interior_gain = v.clamp(0.2, 1.5),
        // MONKEY (enclosed day floor): 0 IS meaningful here (it restores the pre-feature look
        // exactly), unlike the two dim dials above whose 0 would be a broken-looking world.
        "interiordaylight" => cfg.interior_daylight = v.clamp(0.0, 1.0),
        // MONKEY (bake floor): 0 IS meaningful here too (it restores the pre-feature look exactly).
        // The upper clamp matters more than usual: the packer multiplies this by `interiorGain`
        // (up to 1.5) and rides the product in a lane fraction that must stay under 0.5 after
        // scaling, so a value that escaped this clamp would reach the world-shadow flag it shares
        // a lane with. `pack_bake_lane` clamps the product too — belt and braces, one at each end.
        "interiorbakefloor" => cfg.interior_bake_floor = v.clamp(0.0, 1.0),
        // MONKEY (fire GO lights): the synthesised-fire gain, clamped at the edge like the rest.
        "firelightgain" => cfg.fire_light_gain = v.clamp(0.0, 4.0),
        // MONKEY (spellLightGain): the spell lane's gain, same range and same edge clamp — and `0`
        // is meaningful here (the lane off) exactly as it is for the fire gain above.
        "spelllightgain" => cfg.spell_light_gain = v.clamp(0.0, 4.0),
        // MONKEY (flame flicker): 0..2 — the amplitudes are authored at 1, and 2 is the deliberate
        // over-drive for judging the shape. Clamped at the edge like every knob here.
        "fireflicker" => cfg.fire_flicker = v.clamp(0.0, 2.0),
        // ── end MONKEY (lighting) ─────────────────────────────────────────────────────────────
        "farclip" => view.farclip = v.clamp(*FARCLIP_RANGE.start(), *FARCLIP_RANGE.end()),
        // The reference REFUSES an out-of-range write here rather than clamping (`0x688d90`
        // echoes "NearClip must be in range 0.01 - 0.33" and returns 0). We clamp, which is the
        // table's standing posture for every range — the consumer clamps at its own edge.
        "nearclip" => view.set_nearclip(v),
        // Multisampling (1629) — the reference's own `atoi`-then-clamp `[1, 16]` at `0x63b250`,
        // then the DEVICE's ceiling (1643): a count this GPU does not offer is not a setting
        // that degrades, it is a wgpu validation error that kills the render thread on frame
        // one. Clamping at the write covers every writer there is: the file, a Lua `SetCVar`,
        // the dropdown, and the Defaults button. Latched: the camera reads this once, at spawn.
        "gxmultisample" => {
            let asked = (v as u32).clamp(*MSAA_RANGE.start(), *MSAA_RANGE.end());
            let granted = msaa_formats.clamp(asked);
            if granted != asked {
                // At `warn`, the same posture as the seed clamp: the player asked for something
                // and did not get it, and this is the only place that fact exists.
                warn!("cvar gxMultisample: this GPU does not offer {asked}x multisampling — using {granted}x");
            }
            msaa.samples = granted;
        }
        // The filter policy's two halves (1642). Both write a value nothing reads until the
        // next launch — the process policy is published once at the end of `CvarLoad` — which
        // is benilla's limitation, not a latch: the reference registers both with `flags = 1`
        // and applies them live. `anisotropic` takes the reference's own parse-then-clamp
        // `[1, 16]` (`0x689110`); `trilinear` is a flag like every other.
        "trilinear" => tex_filter.trilinear = ev.flag(),
        "anisotropic" => {
            tex_filter.aniso = (v as u32).clamp(
                *benilla_assets::ANISO_RANGE.start(),
                *benilla_assets::ANISO_RANGE.end(),
            )
        }
        // The panel's 0/1/2 lands as the density multiplier ×1/×2/×3; the clamp is the 1.12
        // slider's own range (an off-grid hand-edit rides between stops, like every slider).
        // The SAME knob has a second registered spelling (2151), MIRRORED below — the knob is
        // already written, so the sibling row follows without a callback — so `GetCVar` never
        // answers two detail levels for one ground cover.
        "worlddetail" => {
            clutter.density = v.clamp(0.0, 2.0) + 1.0;
            cvars.mirror(
                benilla_ui::script::CVAR_FRILL_DENSITY,
                &clutter.frill_density().to_string(),
            );
        }
        // …and in the reference's own cells-per-chunk, with the reference's own `[1, 256]`
        // clamp rather than the stop's — `ClutterConfig::set_frill_density` carries both, and
        // `terrain_stream::rescatter_clutter` re-scatters the loaded tiles off the resulting
        // density change exactly as it does for the row above (0992's setter law, which is the
        // callback's own chunk rebuild).
        "frilldensity" => {
            clutter.set_frill_density(v);
            cvars.mirror(
                benilla_ui::script::CVAR_WORLD_DETAIL,
                &(clutter.density - 1.0).to_string(),
            );
        }
        // Weather Intensity, the panel's 0..3 step 1 (2181). The reference's callback is
        // `0x67b870`, a jump table (`0x67b8e8`) mapping 0/1/2/3 onto the quality cells
        // {0.1, 0.33, 0.66, 1.0} in `[0x8680ec]` (wow-re `graphics-cost-cvar-census.md` §4).
        // What that table does with an off-grid int is NOT carved, so the clamp here is the
        // table's own standing posture rather than a fidelity claim — and it costs nothing
        // either way, because `WeatherState::density_gain` already `.min(3)`s its own index.
        "weatherdensity" => weather.weather_density = v.trunc().clamp(0.0, 3.0) as u8,
        _ => {}
    }
}

/// `/console detailDoodadAlpha [0..255]` — the reference's own console command (`0x6739a0`;
/// registrar `0x63f9e0`, a command table and not `CVar::Register`, so it never persists — 1804
/// does not apply to it). It is the **ground-clutter cutout reference**, the dial that decides
/// where grass first appears: the detail-doodad draw alpha-tests `texel.a x distance_ramp`
/// against it, so at the default 128 nothing survives past ~61 yd of the 70 yd horizon, and
/// lowering it walks that onset out toward the horizon. The reference rejects an out-of-range
/// value rather than saturating (`0x6739b9: cmp eax,0xff; jbe`), so out-of-range and
/// unparseable are the same here: a readout with the usage. Bare = the readout (the reference
/// reads an uninitialised stack slot there; a readout is the useful reading of "no argument").
fn detail_doodad_alpha(world: &mut World, args: &str) -> Vec<String> {
    let Some(mut clutter) = world.get_resource_mut::<benilla_world::clutter::ClutterConfig>()
    else {
        return vec!["detailDoodadAlpha: this run has no ground clutter".to_string()];
    };
    match args
        .split_whitespace()
        .next()
        .and_then(|v| v.parse::<u8>().ok())
    {
        Some(v) => {
            clutter.alpha_ref = f32::from(v) / 255.0;
            vec![format!("detailDoodadAlpha set to {v}")]
        }
        None => vec![format!(
            "detailDoodadAlpha is {} (usage: /console detailDoodadAlpha 0-255)",
            (clutter.alpha_ref * 255.0).round() as u32
        )],
    }
}

impl Plugin for VideoPlugin {
    fn build(&self, app: &mut App) {
        use crate::console::ConsoleCommandApp;
        app.add_observer(on_cvar);
        app.console_command(
            "detailDoodadAlpha",
            "The ground-clutter cutout reference, 0-255 (128 = the default).",
            detail_doodad_alpha,
        );
        app.init_resource::<VideoConfig>()
            .init_resource::<GxRestarts>()
            .add_systems(Startup, (log_display_session, check_window_pinned).chain())
            .add_systems(
                Update,
                (
                    // After the tick, and after the CVar sync: the stock video window's Okay is
                    // `SetCVar` per changed row and then `RestartGx()`, in one handler, so the
                    // staged rows have to be in the registry before the commit reads it
                    // (decision 2304).
                    (drain_restart_gx, (apply_present_mode, apply_window_mode))
                        .chain()
                        .after(crate::ui_script::UiInput)
                        .after(crate::cvars::sync_cvars),
                    // A push the tick may read (`GetScreenResolutions`): the feed phase.
                    publish_display_modes.in_set(crate::ui_script::UiFeed),
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

/// Move the interface's `RestartGx()` calls into [`GxRestarts`] — and **commit the latch**.
///
/// The `gx*` rows are latched, as the reference registers them (flags `3`, decision 2303): a
/// `SetCVar("gxVSync", 0)` or a `/console gxWindow 1` is staged, `GetCVar` keeps answering the
/// applied value, and nothing moves until `RestartGx()` — which in the reference re-creates the
/// device and calls `CVar::Update 0x63e060` on each row from inside it. Here the commit is
/// [`crate::cvars::Cvars::commit_latched`], its observers run at the sync point before the two
/// appliers below, and what a restart means beyond that is "re-assert the settings against the
/// window now" rather than "tear the device down": wgpu reconfigures the surface on the next
/// present. The two systems below do the asserting; this one carries the request across the VM
/// boundary and fires the commit.
fn drain_restart_gx(
    script: Option<NonSendMut<benilla_ui::script::UiScript>>,
    mut restarts: ResMut<GxRestarts>,
    mut cvars: ResMut<crate::cvars::Cvars>,
    mut commands: Commands,
) {
    let Some(mut script) = script else {
        return;
    };
    let asks = script.take_restart_gx_asks();
    if asks == 0 {
        // Never touch the resources on a quiet frame: a `ResMut` deref-mut is a change signal, and
        // both consumers below are gated on the value moving.
        return;
    }
    restarts.0 = restarts.0.wrapping_add(asks);
    let committed = cvars.commit_latched();
    if cvars.has_events() {
        for event in cvars.take_events() {
            commands.trigger(event);
        }
    }
    info!("video: RestartGx — {committed} staged setting(s) committed; re-asserting the display mode and present mode");
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
/// `Res::is_changed()` is the honest signal since 2303: [`on_cvar`] writes this resource only
/// when one of its own rows moved (before the registry, the CVar host's knob bundle deref-mutted
/// every knob on every write, and this carried a value compare to work around it). The gate is
/// load-bearing: re-asserting the setting every frame would fight the capture probes, which
/// write `present_mode` on the window directly mid-run (`capture::mod`, `probes::live_fps`) to
/// uncap a measurement — their override has to stick.
///
/// First sight deliberately does **not** just arm: an inserted resource reads as changed on its
/// first frame, and `load_config` applies the saved value at `Startup`, after the window already
/// exists at its boot mode — so the first run is the one that reconciles them.
///
/// **MONKEY (lighting): the value compare 2303 retired is kept here, on top of `is_changed()`.**
/// Upstream could drop it because on ITS `VideoConfig` every row is a Video Options row, so
/// "the resource moved" and "a display row moved" are the same fact. This branch hangs 29
/// lighting knobs off the same resource (`interiorGain`, `torchShadowStrength`, …), each written
/// by its own arm of [`on_cvar`] — so `is_changed()` alone would let a `SetCVar("interiorGain")`
/// re-assert the present mode, and the inner `!=` guard below would then happily undo exactly the
/// probe override this gate exists to protect (a probe writes `AutoNoVsync` on the window while
/// `cfg.vsync` still says on). `is_changed()` stays as the cheap pre-filter.
fn apply_present_mode(
    cfg: Res<VideoConfig>,
    restarts: Res<GxRestarts>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
    mut last: Local<Option<bool>>,
    mut last_restart: Local<u32>,
) {
    // A `RestartGx()` re-asserts even when nothing moved — that is what the caller asked for.
    let forced = std::mem::replace(&mut *last_restart, restarts.0) != restarts.0;
    if (!cfg.is_changed() || last.replace(cfg.vsync) == Some(cfg.vsync)) && !forced {
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
    mut last_restart: Local<u32>,
) {
    // A `RestartGx()` re-asserts even when nothing moved, and that includes re-applying
    // `gxResolution` to a window already in windowed mode — the one video setting whose value can
    // have moved without the MODE moving, and therefore the one this verb is most useful for.
    let forced = std::mem::replace(&mut *last_restart, restarts.0) != restarts.0;
    if !cfg.is_changed() && !forced {
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
