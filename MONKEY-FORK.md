# MONKEY-FORK — benilla-everwood

Our fork of **[samwhosung/benilla](https://github.com/samwhosung/benilla)** (a from-scratch Rust +
Bevy reimplementation of the 1.12.1 / 5875 client). Goal: run the Everwood / "Monkey WoW" custom
content — the features currently delivered on the stock Blizzard client via **injected `mapi.dll` /
`phys.dll`** — as **native code in an open client** instead.

Upstream is a solo project, published as **squashed snapshots from a private tree, with issues and
PRs closed**. Nothing we do here can ever land upstream — we carry our changes forever. So this file
is the contract that keeps re-applying them across upstream drops cheap.

## Branch topology

| branch | role | rule |
|---|---|---|
| `vendor` | pristine mirror of `upstream/main` | **never commit here** — only `reset --hard` |
| `everwood` | our work (this fork's default) | the `benilla-monkey` crate + the catalogued edits below |

Remotes: `upstream` = samwhosung/benilla · `origin` = pkuzic/benilla-everwood.

## Updating to a new upstream drop

```bash
git fetch upstream
git checkout vendor && git reset --hard upstream/main   # works even if upstream re-squashed / force-pushed
git checkout everwood && git rebase vendor              # replays our small commit series on top
```

Conflicts can only occur in the few upstream files we edit (see catalogue). `Cargo.lock` conflict →
`git checkout --theirs Cargo.lock` then `cargo build` to regenerate. After every rebase, re-run the
addon harness as a regression gate (below).

## Design rule: keep our surface tiny

All custom logic lives in a **new crate `crates/benilla-monkey/`** (a Bevy plugin). The workspace is
`members = ["crates/*"]`, so a new crate is auto-included and **cannot merge-conflict**. benilla
already registers Lua globals from Bevy plugins (its `ProbeLuaPlugin` does
`script.lua().create_function()` + `globals().set(...)`); `UiScript` exposes `.lua()` /
`.register_bindings()`; rig/pose data is public (`RigPose`, `RigPalettes`, `RigSkin`, `PosePost`).
When a refactor breaks us it surfaces as a **compile error in our crate** pointing at the renamed
symbol — not a silent wipe of edits scattered through theirs.

## Catalogue of edits to UPSTREAM files

Every edit outside `crates/benilla-monkey/` goes here and is marked in-source with `// MONKEY:`.
Keep this list short; prefer moving logic into the crate.

| file | edit | why | status |
|---|---|---|---|
| `crates/benilla-app/src/lib.rs` (`run`) | `app.add_plugins(MonkeyPlugin)` | wire our plugin in (no public plugin-group entry yet) | **planned** |
| `crates/benilla-ui` / `assets/ui` | publish `TargetLevelText` / `PlayerLevelText` regions | MonkeyCharsheet hooks these stock globals | **planned** |
| (register `PlayerModel` frame type) | `CreateFrame("PlayerModel", …)` | BuildUI + ObjectBrowser 3D previews | **planned** |

## The ~10 functions to reimplement natively

Currently injected by `mapi.dll` (`custom/client/mapi/dllmain.cpp` `g_api[]`). Only 4 addons use
them; the other 10 Monkey addons are pure FrameXML.

- **Projection / camera** (easy): `WorldToScreen`, `MonkeyProj`, `MonkeyCamLive`
- **Model spawn / transform** (cleaner than the DLL hacks — we own the loader): `MonkeyCreateModel`,
  `MonkeySetModelTransform`, `MonkeyDestroyModel`; `MonkeyPathByte` becomes unnecessary
- **GO model reuse** (obsolete — just spawn our own entity): `MonkeyHijackGO`, `MonkeyRestoreGO`,
  `MonkeyDriveGO`
- **Bone override** (deepest — via `RigPose` around `PosePost`): `MonkeySetScratchBone`, `MonkeyBendBone`

## De-risk plan (staged)

1. Plugin PoC — register `WorldToScreen` from `benilla-monkey`, confirm a Lua call returns real coords.
2. FrameXML fixes — `PlayerModel` + `TargetLevelText`/`PlayerLevelText`; re-harness (10/14 → ~13/14).
3. Live UI smoke test vs the Everwood server (real pixels + interaction).
4. Native `MonkeySetScratchBone` via `RigPose` — unblocks MonkeyBodyShape end-to-end.

## Regression harness

benilla ships a headless addon corpus harness. Point it at a folder of our addons:

```bash
WOW_DATA=G:/twmoa_1181/Data cargo run -q -p benilla-app --example addon_harness -- <AddOns folder> --verbose
```

Baseline (2026-08-23, before any fix): **12/14 load, 10/14 survive a session**; read the custom
MPQ/DBC patch chain without crashing.
