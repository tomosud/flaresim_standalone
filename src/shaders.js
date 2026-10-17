// ============================================================================
// shaders.js — WGSL kernels.
//
// ghostWGSL mirrors ghost_cuda.cu (and optics.js traceGhost) line for line.
// WGSL has no float atomics, so the CUDA atomicAdd(float) splat is replaced
// by a 64-bit fixed-point accumulator built from two u32 atomics with carry.
// ============================================================================

export const ghostWGSL = /* wgsl */ `
struct Surf { radius: f32, thick: f32, ior: f32, abbe: f32, semi: f32, coating: f32, z: f32, pad: f32 };
struct U {
  nSurf: u32, nGrid: u32, nSpec: u32, nSrc: u32,
  comboBase: u32, comboCount: u32, width: u32, height: u32,
  sensorZ: f32, frontR: f32, startZ: f32, shw: f32,
  shh: f32, rayWeight: f32, fixedScale: f32, pad: f32,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> surfs: array<Surf>;
@group(0) @binding(2) var<storage, read> pairs: array<vec4f>;   // a, b, area_boost, -
@group(0) @binding(3) var<storage, read> srcs: array<vec4f>;    // [2i]=(ax, ay, -, -) [2i+1]=(r, g, b, -)
@group(0) @binding(4) var<storage, read> grid: array<vec2f>;
@group(0) @binding(5) var<storage, read> spec: array<vec4f>;    // lambda, rw, gw, bw
@group(0) @binding(6) var<storage, read_write> acc: array<atomic<u32>>;

fn dispersion(nd: f32, vd: f32, l: f32) -> f32 {
  if (vd < 0.1 || nd <= 1.0001) { return nd; }
  let lF = 486.13; let lC = 656.27; let ld = 587.56;
  let dn = (nd - 1.0) / vd;
  let B = dn / (1.0 / (lF * lF) - 1.0 / (lC * lC));
  let A = nd - B / (ld * ld);
  return A + B / (l * l);
}
fn iorAt(s: u32, l: f32) -> f32 { return dispersion(surfs[s].ior, surfs[s].abbe, l); }
fn iorBefore(s: u32, l: f32) -> f32 { if (s == 0u) { return 1.0; } return iorAt(s - 1u, l); }

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
  let sf = surfs[s];
  if (abs(sf.radius) < 1e-6) {
    if (abs(d.z) < 1e-12) { return h; }
    let t = (sf.z - o.z) / d.z;
    if (!(t > 1e-6)) { return h; }
    h.p = o + d * t;
    if (!(dot(h.p.xy, h.p.xy) <= sf.semi * sf.semi)) { return h; }
    h.n = vec3f(0.0, 0.0, select(1.0, -1.0, d.z > 0.0));
    h.ok = true; return h;
  }
  let R = sf.radius;
  let c = vec3f(0.0, 0.0, sf.z + R);
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
    t = select(t2, t1, abs(z1 - sf.z) < abs(z2 - sf.z));
  } else if (t1 > 1e-6) { t = t1; }
  else if (t2 > 1e-6) { t = t2; }
  else { return h; }
  h.p = o + d * t;
  if (!(dot(h.p.xy, h.p.xy) <= sf.semi * sf.semi)) { return h; }
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

struct Trace { pos: vec2f, w: f32, ok: bool };
fn traceGhost(o0: vec3f, d0: vec3f, a: u32, b: u32, l: f32) -> Trace {
  var tr: Trace; tr.ok = false;
  var o = o0; var d = d0; var ior = 1.0; var w = 1.0;
  // Phase 1: forward 0..b, reflect at b
  for (var s = 0u; s <= b; s++) {
    let h = intersect(o, d, s);
    if (!h.ok) { return tr; }
    o = h.p;
    let n2 = iorAt(s, l);
    let R = surfR(abs(dot(h.n, d)), ior, n2, i32(surfs[s].coating), l);
    if (s == b) {
      let r = reflectD(d, h.n); if (!r.ok) { return tr; }
      d = r.d; w *= R;
    } else {
      let r = refractD(d, h.n, ior / n2); if (!r.ok) { return tr; }
      d = r.d; w *= 1.0 - R; ior = n2;
    }
  }
  // Phase 2: backward b-1..a, reflect at a
  for (var si = i32(b) - 1; si >= i32(a); si--) {
    let s = u32(si);
    let h = intersect(o, d, s);
    if (!h.ok) { return tr; }
    o = h.p;
    let n2 = iorBefore(s, l);
    let R = surfR(abs(dot(h.n, d)), ior, n2, i32(surfs[s].coating), l);
    if (s == a) {
      let r = reflectD(d, h.n); if (!r.ok) { return tr; }
      d = r.d; w *= R; ior = iorAt(a, l);
    } else {
      let r = refractD(d, h.n, ior / n2); if (!r.ok) { return tr; }
      d = r.d; w *= 1.0 - R; ior = n2;
    }
  }
  // Phase 3: forward a+1..N-1
  for (var s = a + 1u; s < u.nSurf; s++) {
    let h = intersect(o, d, s);
    if (!h.ok) { return tr; }
    o = h.p;
    let n2 = iorAt(s, l);
    let R = surfR(abs(dot(h.n, d)), ior, n2, i32(surfs[s].coating), l);
    let r = refractD(d, h.n, ior / n2); if (!r.ok) { return tr; }
    d = r.d; w *= 1.0 - R; ior = n2;
  }
  if (abs(d.z) < 1e-12) { return tr; }
  let t = (u.sensorZ - o.z) / d.z;
  if (!(t > 0.0)) { return tr; }
  tr.pos = (o + d * t).xy; tr.w = w; tr.ok = true;
  return tr;
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
  let gi = gid.x;
  if (gi >= u.nGrid || wid.y >= u.comboCount) { return; }
  let combo = u.comboBase + wid.y;
  let pairIdx = combo / u.nSrc;
  let srcIdx = combo % u.nSrc;
  let pr = pairs[pairIdx];
  let a = u32(pr.x); let b = u32(pr.y);
  if (b <= a || b >= u.nSurf) { return; }
  let sa = srcs[srcIdx * 2u];
  let sc = srcs[srcIdx * 2u + 1u].rgb;
  let dir = normalize(vec3f(tan(sa.x), tan(sa.y), 1.0));
  let g = grid[gi];
  let org = vec3f(g.x * u.frontR, g.y * u.frontR, u.startZ);
  for (var k = 0u; k < u.nSpec; k++) {
    let sp = spec[k];
    let r = traceGhost(org, dir, a, b, sp.x);
    if (!r.ok) { continue; }
    let base = r.w * u.rayWeight * pr.z;
    if (base < 1e-20) { continue; }
    let px = (r.pos.x / (2.0 * u.shw) + 0.5) * f32(u.width);
    let py = (r.pos.y / (2.0 * u.shh) + 0.5) * f32(u.height);
    if (!(abs(px) < 1e8 && abs(py) < 1e8)) { continue; }   // also rejects NaN
    let x0 = i32(floor(px - 0.5)); let y0 = i32(floor(py - 0.5));
    let fx = (px - 0.5) - f32(x0); let fy = (py - 0.5) - f32(y0);
    let c = sc * sp.yzw * base;
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
