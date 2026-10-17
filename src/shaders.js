// ============================================================================
// shaders.js — WGSL kernels.
//
// WGSL has no float atomics, so the CUDA atomicAdd(float) splat is replaced
// by a 64-bit fixed-point accumulator built from two u32 atomics with carry.
// ============================================================================

// The optics per surface are identical to trace.cpp / optics.js traceGhost,
// but the work is organised differently from ghost_cuda.cu: one invocation
// handles one (source, pupil sample, wavelength) and walks every ghost pair
// sorted by (b, a). The forward segment through surfaces 0..b-1 is shared by
// all pairs with the same or larger b, so it is traced once and extended
// incrementally; once it is vignetted, every remaining pair is dead too.
export const ghostWGSL = /* wgsl */ `
struct U {
  nSurf: u32, nGrid: u32, nSpec: u32, srcBase: u32,
  pairBase: u32, pairCount: u32, width: u32, height: u32,
  sensorZ: f32, frontR: f32, startZ: f32, shw: f32,
  shh: f32, rayWeight: f32, fixedScale: f32, srcCount: u32,
};
// surf[s] = (radius, semi_aperture, z, coating); ior[] = IOR after surface s
// at spectral sample k, flattened as k * nSurf + s (precomputed on the CPU).
struct LensU { surf: array<vec4f, 64>, ior: array<vec4f, 192> };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<uniform> L: LensU;
@group(0) @binding(2) var<storage, read> pairs: array<vec4f>;   // a, b, area_boost, - (sorted by b, a)
@group(0) @binding(3) var<storage, read> srcs: array<vec4f>;    // [2i]=(ax, ay, -, -) [2i+1]=(r, g, b, -)
@group(0) @binding(4) var<storage, read> grid: array<vec2f>;
@group(0) @binding(5) var<storage, read> spec: array<vec4f>;    // lambda, rw, gw, bw
@group(0) @binding(6) var<storage, read_write> acc: array<atomic<u32>>;

fn iorAfter(s: u32, k: u32) -> f32 { let i = k * u.nSurf + s; return L.ior[i / 4u][i % 4u]; }
fn iorBefore(s: u32, k: u32) -> f32 { if (s == 0u) { return 1.0; } return iorAfter(s - 1u, k); }

fn fresnel(ci: f32, n1: f32, n2: f32) -> f32 {
  let c = abs(ci);
  let eta = n1 / n2;
  let s2t = eta * eta * (1.0 - c * c);
  if (s2t >= 1.0) { return 1.0; }
  let ct = sqrt(1.0 - s2t);
  let rs = (n1 * c - n2 * ct) / (n1 * c + n2 * ct);
  let rp = (n2 * c - n1 * ct) / (n2 * c + n1 * ct);
  return 0.5 * (rs * rs + rp * rp);
}
fn coatingR(ci: f32, n1: f32, n2: f32, cn: f32, d: f32, l: f32) -> f32 {
  let s2c = (n1 / cn) * (n1 / cn) * (1.0 - ci * ci);
  if (s2c >= 1.0) { return fresnel(ci, n1, n2); }
  let cc = sqrt(1.0 - s2c);
  let delta = 6.283185307 * cn * d * cc / l;
  let r01 = (n1 * ci - cn * cc) / (n1 * ci + cn * cc);
  let s22 = (cn / n2) * (cn / n2) * (1.0 - cc * cc);
  if (s22 >= 1.0) { return fresnel(ci, n1, n2); }
  let c2 = sqrt(1.0 - s22);
  let r12 = (cn * cc - n2 * c2) / (cn * cc + n2 * c2);
  let c2d = cos(2.0 * delta);
  let num = r01 * r01 + r12 * r12 + 2.0 * r01 * r12 * c2d;
  let den = 1.0 + r01 * r01 * r12 * r12 + 2.0 * r01 * r12 * c2d;
  return clamp(num / den, 0.0, 1.0);
}
fn surfR(ci: f32, n1: f32, n2: f32, layers: i32, l: f32) -> f32 {
  if (layers <= 0) { return fresnel(ci, n1, n2); }
  var R = coatingR(ci, n1, n2, 1.38, 550.0 / (4.0 * 1.38), l);
  for (var i = 1; i < layers; i++) { R *= 0.25; }
  return clamp(R, 0.0, 1.0);
}

struct Hit { p: vec3f, n: vec3f, ok: bool };
fn intersect(o: vec3f, d: vec3f, s: u32) -> Hit {
  var h: Hit; h.ok = false;
  let sf = L.surf[s];
  let R = sf.x; let semi = sf.y; let z = sf.z;
  if (abs(R) < 1e-6) {
    if (abs(d.z) < 1e-12) { return h; }
    let t = (z - o.z) / d.z;
    if (!(t > 1e-6)) { return h; }
    h.p = o + d * t;
    if (!(dot(h.p.xy, h.p.xy) <= semi * semi)) { return h; }
    h.n = vec3f(0.0, 0.0, select(1.0, -1.0, d.z > 0.0));
    h.ok = true; return h;
  }
  let c = vec3f(0.0, 0.0, z + R);
  let oc = o - c;
  let a = dot(d, d);
  let b = 2.0 * dot(oc, d);
  let cc = dot(oc, oc) - R * R;
  let disc = b * b - 4.0 * a * cc;
  if (disc < 0.0) { return h; }
  let sd = sqrt(disc);
  let inv2a = 0.5 / a;
  let t1 = (-b - sd) * inv2a;
  let t2 = (-b + sd) * inv2a;
  var t: f32;
  if (t1 > 1e-6 && t2 > 1e-6) {
    let z1 = o.z + t1 * d.z; let z2 = o.z + t2 * d.z;
    t = select(t2, t1, abs(z1 - z) < abs(z2 - z));
  } else if (t1 > 1e-6) { t = t1; }
  else if (t2 > 1e-6) { t = t2; }
  else { return h; }
  h.p = o + d * t;
  if (!(dot(h.p.xy, h.p.xy) <= semi * semi)) { return h; }
  var n = (h.p - c) / abs(R);
  if (dot(n, d) > 0.0) { n = -n; }
  h.n = n; h.ok = true; return h;
}

struct Dir { d: vec3f, ok: bool };
fn safeNorm(v: vec3f) -> Dir {
  var r: Dir;
  let sq = dot(v, v);
  r.ok = sq >= 1e-18 && sq < 3.0e38;
  r.d = v * inverseSqrt(max(sq, 1e-30));
  return r;
}
fn refractD(d: vec3f, n: vec3f, eta: f32) -> Dir {
  var r: Dir; r.ok = false;
  let ci = -dot(n, d);
  let s2t = eta * eta * (1.0 - ci * ci);
  if (s2t >= 1.0) { return r; }
  return safeNorm(d * eta + n * (eta * ci - sqrt(1.0 - s2t)));
}
fn reflectD(d: vec3f, n: vec3f) -> Dir { return safeNorm(d - n * (2.0 * dot(d, n))); }

struct Ray { o: vec3f, d: vec3f, ior: f32, w: f32 };
// Intersect surface s, then reflect (weight *= R) or refract into medium n2 (weight *= 1 - R).
fn interact(r: ptr<function, Ray>, s: u32, n2: f32, l: f32, refl: bool) -> bool {
  let h = intersect((*r).o, (*r).d, s);
  if (!h.ok) { return false; }
  (*r).o = h.p;
  let R = surfR(abs(dot(h.n, (*r).d)), (*r).ior, n2, i32(L.surf[s].w), l);
  if (refl) {
    let q = reflectD((*r).d, h.n);
    if (!q.ok) { return false; }
    (*r).d = q.d; (*r).w *= R;
  } else {
    let q = refractD((*r).d, h.n, (*r).ior / n2);
    if (!q.ok) { return false; }
    (*r).d = q.d; (*r).w *= 1.0 - R; (*r).ior = n2;
  }
  return true;
}

// 64-bit fixed-point add: acc[2i] = integer part, acc[2i+1] = fraction * 2^32
fn addFixed(i: u32, v: f32) {
  if (!(v > 0.0)) { return; }
  let s = v * u.fixedScale;
  let fl = floor(s);
  let hiAdd = u32(min(fl, 4294967040.0));
  let loAdd = u32(min((s - fl) * 4294967296.0, 4294967040.0));
  let old = atomicAdd(&acc[i * 2u + 1u], loAdd);
  let carry = select(0u, 1u, old + loAdd < old);
  if (hiAdd + carry > 0u) { atomicAdd(&acc[i * 2u], hiAdd + carry); }
}
fn splat(x: i32, y: i32, c: vec3f, wt: f32) {
  if (x < 0 || y < 0 || x >= i32(u.width) || y >= i32(u.height)) { return; }
  let p = (u32(y) * u.width + u32(x)) * 3u;
  addFixed(p, c.r * wt); addFixed(p + 1u, c.g * wt); addFixed(p + 2u, c.b * wt);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  if (gid.x >= u.nGrid * u.nSpec || wid.y >= u.srcCount) { return; }
  let k = gid.x / u.nGrid;
  let gs = grid[gid.x % u.nGrid];
  let sp = spec[k];
  let si = u.srcBase + wid.y;
  let sa = srcs[si * 2u];
  let col = srcs[si * 2u + 1u].rgb * sp.yzw;
  var fwd: Ray;
  fwd.o = vec3f(gs.x * u.frontR, gs.y * u.frontR, u.startZ);
  fwd.d = normalize(vec3f(tan(sa.x), tan(sa.y), 1.0));
  fwd.ior = 1.0;
  fwd.w = 1.0;
  var fidx = 0u;   // surfaces 0..fidx-1 already transmitted by fwd

  for (var p = u.pairBase; p < u.pairBase + u.pairCount; p++) {
    let pr = pairs[p];
    let a = u32(pr.x); let b = u32(pr.y);
    // Phase 1 (shared between pairs): transmit up to surface b-1.
    var dead = false;
    loop {
      if (fidx >= b) { break; }
      if (!interact(&fwd, fidx, iorAfter(fidx, k), sp.x, false)) { dead = true; break; }
      fidx++;
    }
    if (dead) { break; }   // sorted by b: every remaining pair needs this segment
    var r = fwd;
    // Reflect at b.
    if (!interact(&r, b, iorAfter(b, k), sp.x, true)) { continue; }
    // Phase 2: backward b-1..a, reflect at a.
    var ok = true;
    for (var s2 = i32(b) - 1; s2 >= i32(a); s2--) {
      let s = u32(s2);
      if (!interact(&r, s, iorBefore(s, k), sp.x, s == a)) { ok = false; break; }
      if (s == a) { r.ior = iorAfter(a, k); }
    }
    if (!ok) { continue; }
    // Phase 3: forward a+1..N-1.
    for (var s = a + 1u; s < u.nSurf; s++) {
      if (!interact(&r, s, iorAfter(s, k), sp.x, false)) { ok = false; break; }
    }
    if (!ok || abs(r.d.z) < 1e-12) { continue; }
    let t = (u.sensorZ - r.o.z) / r.d.z;
    if (!(t > 0.0)) { continue; }
    let pos = (r.o + r.d * t).xy;
    let base = r.w * u.rayWeight * pr.z;
    if (base < 1e-20) { continue; }
    let px = (pos.x / (2.0 * u.shw) + 0.5) * f32(u.width);
    let py = (pos.y / (2.0 * u.shh) + 0.5) * f32(u.height);
    if (!(abs(px) < 1e8 && abs(py) < 1e8)) { continue; }   // also rejects NaN
    let x0 = i32(floor(px - 0.5)); let y0 = i32(floor(py - 0.5));
    let fx = (px - 0.5) - f32(x0); let fy = (py - 0.5) - f32(y0);
    let c = col * base;
    splat(x0, y0, c, (1.0 - fx) * (1.0 - fy));
    splat(x0 + 1, y0, c, fx * (1.0 - fy));
    splat(x0, y0 + 1, c, (1.0 - fx) * fy);
    splat(x0 + 1, y0 + 1, c, fx * fy);
  }
}
`;

export const resolveWGSL = /* wgsl */ `
struct U { n: u32, pad0: u32, invScale: f32, pad1: f32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> acc: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4f>;
fn val(i: u32) -> f32 { return (f32(acc[i * 2u]) + f32(acc[i * 2u + 1u]) * 2.3283064365386963e-10) * u.invScale; }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + g.y * 65535u * 256u;
  if (i >= u.n) { return; }
  dst[i] = vec4f(val(i * 3u), val(i * 3u + 1u), val(i * 3u + 2u), 0.0);
}
`;

// Separable box blur, one invocation per line, sliding window with an exact
// re-sum every 32 samples to bound float drift next to very bright pixels.
// Edge samples are renormalised by the clipped window size (as box_blur()).
export const blurWGSL = /* wgsl */ `
struct U { len: u32, lines: u32, lineStride: u32, step: u32, radius: i32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> src: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= u.lines) { return; }
  let base = g.x * u.lineStride;
  let n = i32(u.len);
  let r = u.radius;
  var sum = vec4f(0.0);
  for (var k = 0; k <= min(r, n - 1); k++) { sum += src[base + u32(k) * u.step]; }
  for (var x = 0; x < n; x++) {
    let lo = max(0, x - r); let hi = min(n - 1, x + r);
    if (x > 0) {
      if ((x & 31) == 0) {
        sum = vec4f(0.0);
        for (var k = lo; k <= hi; k++) { sum += src[base + u32(k) * u.step]; }
      } else {
        if (x + r < n) { sum += src[base + u32(x + r) * u.step]; }
        if (x - r - 1 >= 0) { sum -= src[base + u32(x - r - 1) * u.step]; }
      }
    }
    dst[base + u32(x) * u.step] = max(sum / f32(hi - lo + 1), vec4f(0.0));
  }
}
`;

// Haze seed: each detected source block is filled with its colour.
export const hazeSeedWGSL = /* wgsl */ `
struct U { w: u32, h: u32, dw: u32, dh: u32, ds: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> blocks: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4f>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= u.w || g.y >= u.h) { return; }
  let bx = g.x / u.ds; let by = g.y / u.ds;
  var c = vec4f(0.0);
  if (bx < u.dw && by < u.dh) { c = vec4f(blocks[by * u.dw + bx].rgb, 0.0); }
  dst[g.y * u.w + g.x] = c;
}
`;

// Starburst gather (render_starburst): per output pixel, sum every source's
// PSF sampled at a wavelength-scaled radius (R 650 / G 550 / B 450 nm).
export const starburstWGSL = /* wgsl */ `
struct U { w: u32, h: u32, N: u32, nSrc: u32, rRef: f32, pad0: f32, pad1: f32, pad2: f32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> psf: array<f32>;
@group(0) @binding(2) var<storage, read> srcs: array<vec4f>;   // [2i]=(px, py, -, -) [2i+1]=(r, g, b, -)
@group(0) @binding(3) var<storage, read_write> dst: array<vec4f>;
fn samplePSF(fx: f32, fy: f32) -> f32 {
  let n1 = f32(u.N - 1u);
  if (fx < 0.0 || fy < 0.0 || fx >= n1 || fy >= n1) { return 0.0; }
  let ix = u32(fx); let iy = u32(fy);
  let tx = fx - f32(ix); let ty = fy - f32(iy);
  let i = iy * u.N + ix;
  return mix(mix(psf[i], psf[i + 1u], tx), mix(psf[i + u.N], psf[i + u.N + 1u], tx), ty);
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= u.w || g.y >= u.h) { return; }
  let halfN = f32(u.N) * 0.5;
  let sc = vec3f(650.0, 550.0, 450.0) / 550.0;
  let rMax = u.rRef * sc.x;
  var acc = vec3f(0.0);
  for (var s = 0u; s < u.nSrc; s++) {
    let p = srcs[s * 2u];
    let dx = f32(g.x) - p.x; let dy = f32(g.y) - p.y;
    if (abs(dx) > rMax || abs(dy) > rMax) { continue; }
    let c = srcs[s * 2u + 1u].rgb;
    let inv = vec3f(halfN) / (u.rRef * sc);
    acc.r += c.r * samplePSF(dx * inv.r + halfN, dy * inv.r + halfN);
    acc.g += c.g * samplePSF(dx * inv.g + halfN, dy * inv.g + halfN);
    acc.b += c.b * samplePSF(dx * inv.b + halfN, dy * inv.b + halfN);
  }
  dst[g.y * u.w + g.x] = vec4f(acc, 0.0);
}
`;

// Viewer: composite, exposure, SDR tone map or HDR extended-range output.
export const displayWGSL = /* wgsl */ `
struct U {
  imgW: f32, imgH: f32, offX: f32, offY: f32,
  zoom: f32, exposure: f32, inputGain: f32, ghostGain: f32,
  hazeGain: f32, starGain: f32, hdrScale: f32, pad0: f32,
  mode: u32, tonemap: u32, hdr: u32, pad1: u32,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> inputBuf: array<vec4f>;
@group(0) @binding(2) var<storage, read> ghost: array<vec4f>;
@group(0) @binding(3) var<storage, read> haze: array<vec4f>;
@group(0) @binding(4) var<storage, read> star: array<vec4f>;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  var p = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[i], 0.0, 1.0);
}
fn oetf(x: f32) -> f32 {
  let a = abs(x);
  let v = select(1.055 * pow(a, 1.0 / 2.4) - 0.055, a * 12.92, a <= 0.0031308);
  return sign(x) * v;
}
fn aces(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}
@fragment fn fs(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  let ip = (fc.xy - vec2f(u.offX, u.offY)) / u.zoom;
  if (ip.x < 0.0 || ip.y < 0.0 || ip.x >= u.imgW || ip.y >= u.imgH) {
    let ck = (u32(fc.x / 12.0) + u32(fc.y / 12.0)) % 2u;
    let bg = select(0.018, 0.026, ck == 1u);
    return vec4f(vec3f(oetf(bg)), 1.0);
  }
  let i = u32(ip.y) * u32(u.imgW) + u32(ip.x);
  let inp = inputBuf[i].rgb * u.inputGain;
  let gh = ghost[i].rgb * u.ghostGain;
  let hz = haze[i].rgb * u.hazeGain;
  let sb = star[i].rgb * u.starGain;
  var c: vec3f;
  switch (u.mode) {
    case 0u: { c = inp + gh + hz + sb; }
    case 1u: { c = gh + hz + sb; }
    case 2u: { c = gh; }
    case 3u: { c = hz; }
    case 4u: { c = sb; }
    default: { c = inp; }
  }
  c = max(c * u.exposure, vec3f(0.0));
  if (u.hdr == 1u) {
    c = c * u.hdrScale;
  } else {
    switch (u.tonemap) {
      case 1u: { c = c / (1.0 + c); }
      case 2u: { c = aces(c * 0.6); }
      default: { c = min(c, vec3f(1.0)); }
    }
  }
  return vec4f(oetf(c.r), oetf(c.g), oetf(c.b), 1.0);
}
`;
