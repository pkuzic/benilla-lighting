# Third-party code and techniques

## WarcraftXL

[WarcraftXL](https://github.com/WarcraftXL) by **iThorgrim** is a 3.3.5 client extension. The files
below port code or a technique from it; each carries the header line
`Ported from WarcraftXL (https://github.com/WarcraftXL) by iThorgrim — module <module>, <source files>.`

| File | WarcraftXL module | Source files | What was taken |
|---|---|---|---|
| `crates/benilla-world/src/shaders/sky_fx.wgsl` | wxl-retail-clouds | `Clouds.cpp`, `Clouds.hpp` | Domain-warped billow fbm (quintic value noise, 0.6 cotton blend, 0.55 octave gain) for the cloud detail |
| `crates/benilla-world/src/shaders/cloud.wgsl` | wxl-retail-clouds | `Clouds.cpp` | The 3-tap march toward the sun (self-shadow, ambient floor) and the one-tap silver lining |
| `crates/benilla-world/src/clouds/layer.rs` (`update_cloud_fx`) | wxl-retail-clouds | `Clouds.cpp` | March direction = the sun projected on the sheet, strength folded by its flatness (noon clouds evenly lit) |
