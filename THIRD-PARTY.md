# Third-party notices

## WarcraftXL colour grading

The colour-grading pass in `crates/benilla-app/src/post/grading.rs` and `grading.wgsl` is ported
from WarcraftXL's `wxl-retail-grading` implementation, Copyright (C) 2026 WarcraftXL. The source
implementation is licensed under GPL-3.0-or-later. The port preserves its 32³ LUT convention,
strength control, and strip lookup mathematics while uploading the strip as a native 3D texture.
