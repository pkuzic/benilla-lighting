// MONKEY (sky): the Enhanced/High sky library (`skyQuality` ≥ 1), imported by `sky.wgsl` and
// `cloud.wgsl`. Classic never calls into it. Colour math is in linear light; the callers convert
// the gamma stops in and the result back out.
//
// The cloud detail (`cloud_billow`, `cloud_detail`):
// Ported from WarcraftXL (https://github.com/WarcraftXL) by iThorgrim — module wxl-retail-clouds, Clouds.cpp, Clouds.hpp.
#define_import_path benilla_world::sky_fx

const TAU: f32 = 6.2831853;

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((max(c, vec3<f32>(0.0)) + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

// ---- hashes and noise -------------------------------------------------------------------------

// pcg3d (Jarzynski & Olano 2020): three well-mixed u32 from three.
fn pcg3(v_in: vec3<u32>) -> vec3<u32> {
    var v = v_in * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v = v ^ (v >> vec3<u32>(16u));
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

fn hash33(p: vec3<i32>) -> vec3<f32> {
    return vec3<f32>(pcg3(bitcast<vec3<u32>>(p)) >> vec3<u32>(8u)) * (1.0 / 16777216.0);
}

fn hash_cell2(p: vec2<i32>, salt: u32) -> f32 {
    return f32(pcg3(vec3<u32>(bitcast<vec2<u32>>(p), salt)).x >> 8u) * (1.0 / 16777216.0);
}

// Value noise with the quintic fade: the cubic's second-derivative seam shows as square plateaus
// on a slowly drifting sky (the WarcraftXL note).
fn value_noise2(p: vec2<f32>, salt: u32) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a = hash_cell2(i, salt);
    let b = hash_cell2(i + vec2<i32>(1, 0), salt);
    let c = hash_cell2(i + vec2<i32>(0, 1), salt);
    let d = hash_cell2(i + vec2<i32>(1, 1), salt);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn value_noise3(p: vec3<f32>) -> f32 {
    let i = vec3<i32>(floor(p));
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = hash33(i).x;
    let n100 = hash33(i + vec3<i32>(1, 0, 0)).x;
    let n010 = hash33(i + vec3<i32>(0, 1, 0)).x;
    let n110 = hash33(i + vec3<i32>(1, 1, 0)).x;
    let n001 = hash33(i + vec3<i32>(0, 0, 1)).x;
    let n101 = hash33(i + vec3<i32>(1, 0, 1)).x;
    let n011 = hash33(i + vec3<i32>(0, 1, 1)).x;
    let n111 = hash33(i + vec3<i32>(1, 1, 1)).x;
    let x00 = mix(n000, n100, u.x);
    let x10 = mix(n010, n110, u.x);
    let x01 = mix(n001, n101, u.x);
    let x11 = mix(n011, n111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

fn fbm3(p: vec3<f32>) -> f32 {
    var s = 0.0;
    var a = 0.5;
    var q = p;
    for (var o = 0; o < 3; o++) {
        s += a * value_noise3(q);
        q = q * 2.03 + vec3<f32>(1.7, 9.2, 3.4);
        a *= 0.5;
    }
    return s / 0.875;
}

// ---- S1: the smooth gradient ------------------------------------------------------------------

// The reference's ring elevations (degrees), horizon to zenith.
const STOP_ELEV = array<f32, 6>(0.0, 1.8, 3.7, 9.8, 16.8, 90.0);

// Fritsch–Carlson tangent at an interior stop: 0 at a local extremum, else the weighted harmonic
// mean of the two secants, so the curve never overshoots a stop (no halos, no new colours).
fn pchip_tangent(d0: vec3<f32>, d1: vec3<f32>, h0: f32, h1: f32) -> vec3<f32> {
    let w1 = 2.0 * h1 + h0;
    let w2 = h1 + 2.0 * h0;
    let safe0 = select(d0, vec3<f32>(1.0), d0 == vec3<f32>(0.0));
    let safe1 = select(d1, vec3<f32>(1.0), d1 == vec3<f32>(0.0));
    let m = (w1 + w2) / (w1 / safe0 + w2 / safe1);
    return select(vec3<f32>(0.0), m, d0 * d1 > vec3<f32>(0.0));
}

// Monotone cubic through the six stops (`y[0]` = fog at 0°, `y[5]` = zenith), linear light. The
// end tangents are 0: flat into the fog band below the horizon and flat over the zenith, so the
// curve is C1 everywhere and the rings leave no Mach bands.
fn smooth_gradient(elev: f32, y: array<vec3<f32>, 6>) -> vec3<f32> {
    if (elev <= 0.0) {
        return y[0];
    }
    if (elev >= 90.0) {
        return y[5];
    }
    var yy = y;
    var se = STOP_ELEV;
    var d: array<vec3<f32>, 5>;
    var h: array<f32, 5>;
    for (var i = 0; i < 5; i++) {
        h[i] = se[i + 1] - se[i];
        d[i] = (yy[i + 1] - yy[i]) / h[i];
    }
    var k = 0;
    for (var i = 1; i < 5; i++) {
        if (elev >= se[i]) {
            k = i;
        }
    }
    var m0 = vec3<f32>(0.0);
    var m1 = vec3<f32>(0.0);
    if (k > 0) {
        m0 = pchip_tangent(d[k - 1], d[k], h[k - 1], h[k]);
    }
    if (k < 4) {
        m1 = pchip_tangent(d[k], d[k + 1], h[k], h[k + 1]);
    }
    let hk = h[k];
    let t = (elev - se[k]) / hk;
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    return h00 * yy[k] + h10 * hk * m0 + h01 * yy[k + 1] + h11 * hk * m1;
}

// Triangular dither of ±1 LSB at 8 bits, fixed per pixel (no temporal shimmer).
fn dither_tri(frag: vec2<f32>) -> f32 {
    let h = hash33(vec3<i32>(vec2<i32>(frag), 7));
    return (h.x + h.y - 1.0) / 255.0;
}

// ---- S4: the sun glow -------------------------------------------------------------------------

// Henyey–Greenstein phase normalised to 1 toward the sun.
fn hg_norm(c: f32, g: f32) -> f32 {
    let g2 = g * g;
    let a = 1.0 + g2 - 2.0 * g * c;
    let a1 = (1.0 - g) * (1.0 - g);
    return pow(a1 / a, 1.5);
}

// A two-lobe Mie-like halo: a broad aureole plus a tighter core round the disc.
fn sun_glow(dir: vec3<f32>, sun: vec3<f32>) -> f32 {
    let c = dot(dir, sun);
    return 0.55 * hg_norm(c, 0.70) + 0.45 * hg_norm(c, 0.93);
}

// ---- S5: the night sky ------------------------------------------------------------------------

// Grid cells per unit radius: one cell spans ~0.3°, so a candidate star every few pixels.
const STAR_GRID: f32 = 180.0;

// A procedural star field fixed to world directions. Each grid cell holds at most one star (a
// density draw), jittered inside the cell; the eight cells round the sample are tested so no star
// is cut at a cell face. Brightness is a steep power law (few bright, many faint), the footprint
// is at least a pixel wide so a star does not vanish between pixels, and a slow twinkle is
// stronger low in the sky. `px` is the pixel's size in grid units.
fn star_field(dir: vec3<f32>, t: f32, px: f32, band: f32) -> vec3<f32> {
    let p = dir * STAR_GRID;
    let base = floor(p);
    let f = p - base;
    let o = select(vec3<f32>(-1.0), vec3<f32>(1.0), f > vec3<f32>(0.5));
    let density = 0.035 * (1.0 + 2.5 * band);
    var acc = vec3<f32>(0.0);
    for (var i = 0u; i < 8u; i++) {
        let off = vec3<f32>(f32(i & 1u), f32((i >> 1u) & 1u), f32((i >> 2u) & 1u)) * o;
        let cell = vec3<i32>(base + off);
        let h = hash33(cell);
        if (h.x > density) {
            continue;
        }
        let j = hash33(cell + vec3<i32>(7919, 104729, 1299709));
        let centre = normalize(vec3<f32>(cell) + 0.5 + (j - 0.5) * 0.9) * STAR_GRID;
        let d = length(p - centre);
        let mag = pow(h.y, 22.0);
        let b = 0.035 + 0.965 * mag;
        let sigma = px * (0.55 + 0.7 * mag);
        let spot = exp(-d * d / (2.0 * sigma * sigma));
        let tw_amp = 0.18 + 0.3 * (1.0 - clamp(dir.y, 0.0, 1.0));
        let tw = 1.0 + tw_amp * sin(t * (1.3 + 3.1 * j.x) + TAU * j.y) * sin(t * (0.6 + 1.7 * j.z) + TAU * h.z);
        // Star colour: mostly white, a few warm and a few blue-white.
        let tint = mix(vec3<f32>(0.78, 0.86, 1.0), vec3<f32>(1.0, 0.88, 0.72), h.z);
        acc += tint * (b * spot * tw);
    }
    return acc;
}

// The Milky Way's plane in Bevy space (y up): tilted so the band arcs across the dome.
const GALAXY_N: vec3<f32> = vec3<f32>(0.40, 0.52, -0.754);
const GALAXY_CORE: vec3<f32> = vec3<f32>(-0.62, 0.36, -0.08);

// Band weight in [0, 1] toward `dir`.
fn galaxy_band(dir: vec3<f32>) -> f32 {
    let b = dot(dir, normalize(GALAXY_N));
    return exp(-b * b / (2.0 * 0.15 * 0.15));
}

// A faint, mottled band with a dark dust lane down its middle and a brighter core.
fn milky_way(dir: vec3<f32>, band: f32) -> vec3<f32> {
    if (band < 0.01) {
        return vec3<f32>(0.0);
    }
    let b = dot(dir, normalize(GALAXY_N));
    let cloud = fbm3(dir * 5.0);
    let lane = exp(-(b - 0.02) * (b - 0.02) / (2.0 * 0.03 * 0.03)) * smoothstep(0.35, 0.7, fbm3(dir * 9.0 + 11.0));
    let core = 1.0 + 1.2 * pow(max(dot(dir, normalize(GALAXY_CORE)), 0.0), 4.0);
    let v = band * (0.3 + 0.7 * cloud) * (1.0 - 0.75 * lane) * core;
    return vec3<f32>(0.72, 0.8, 1.0) * v;
}

// ---- S3: cloud detail -------------------------------------------------------------------------

// WarcraftXL's billow blend: fold each octave toward rounded puffs by `cotton` (0.6).
fn billow(v: f32) -> f32 {
    return v + (1.0 - abs(2.0 * v - 1.0) - v) * 0.6;
}

// Billowed fbm, amplitude 0.55 per octave as in WarcraftXL.
fn cloud_billow(p: vec2<f32>, octaves: i32) -> f32 {
    var s = 0.0;
    var a = 1.0;
    var tot = 0.0;
    var q = p;
    for (var o = 0; o < octaves; o++) {
        s += a * billow(value_noise2(q, u32(o) * 57u));
        tot += a;
        a *= 0.55;
        q = q * 2.0 + vec2<f32>(3.7, 1.9);
    }
    return s / tot;
}

// The detail field in [0, 1] on the sheet: a broad warp twists the domain before the billowed fbm
// is read, which turns thresholded blobs into cauliflower edges.
fn cloud_detail(uv: vec2<f32>, t: f32, octaves: i32) -> f32 {
    let p = uv * 36.0 + vec2<f32>(0.011, 0.0043) * t;
    let w = vec2<f32>(value_noise2(p * 0.5, 173u), value_noise2(p * 0.5 + 5.3, 219u)) - 0.5;
    return cloud_billow(p + w * 2.6, octaves);
}

// Apply the detail to a base coverage: the edges (mid coverage) are pushed in and out by the
// detail, solid cores and clear sky stay put, so the CPU tile still says where cloud is.
fn cloud_erode(a: f32, n: f32) -> f32 {
    let edge = 4.0 * a * (1.0 - a);
    return clamp(a + (n - 0.5) * 1.2 * edge, 0.0, 1.0);
}
