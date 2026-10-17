// ============================================================================
// optics.js — lens model, Fresnel/coating physics and CPU ghost trace.
//
// Straight port of flaresim_nuke/src/{lens,fresnel,trace,ghost}.{h,cpp}.
// The GPU kernel in shaders.js mirrors traceGhost() exactly — keep the two
// in sync if either changes.
// ============================================================================

const LF = 486.13, LC = 656.27, LD = 587.56;

// Cauchy dispersion from d-line IOR and Abbe number.
export function dispersionIor(nd, vd, lambda) {
  if (vd < 0.1 || nd <= 1.0001) return nd;
  const dn = (nd - 1) / vd;
  const B = dn / (1 / (LF * LF) - 1 / (LC * LC));
  const A = nd - B / (LD * LD);
  return A + B / (lambda * lambda);
}

export function fresnelReflectance(cosI, n1, n2) {
  cosI = Math.abs(cosI);
  const eta = n1 / n2;
  const sin2t = eta * eta * (1 - cosI * cosI);
  if (sin2t >= 1) return 1;
  const cosT = Math.sqrt(1 - sin2t);
  const rs = (n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT);
  const rp = (n2 * cosI - n1 * cosT) / (n2 * cosI + n1 * cosT);
  return 0.5 * (rs * rs + rp * rp);
}

// Single-layer thin-film coating (Airy formula).
export function coatingReflectance(cosI, n1, n2, cn, dNm, lambda) {
  const sin2c = (n1 / cn) * (n1 / cn) * (1 - cosI * cosI);
  if (sin2c >= 1) return fresnelReflectance(cosI, n1, n2);
  const cosC = Math.sqrt(1 - sin2c);
  const delta = 2 * Math.PI * cn * dNm * cosC / lambda;
  const r01 = (n1 * cosI - cn * cosC) / (n1 * cosI + cn * cosC);
  const sin22 = (cn / n2) * (cn / n2) * (1 - cosC * cosC);
  if (sin22 >= 1) return fresnelReflectance(cosI, n1, n2);
  const cos2 = Math.sqrt(1 - sin22);
  const r12 = (cn * cosC - n2 * cos2) / (cn * cosC + n2 * cos2);
  const c2d = Math.cos(2 * delta);
  const num = r01 * r01 + r12 * r12 + 2 * r01 * r12 * c2d;
  const den = 1 + r01 * r01 * r12 * r12 + 2 * r01 * r12 * c2d;
  return Math.min(Math.max(num / den, 0), 1);
}

// MgF2 quarter-wave @550nm; each extra layer quarters the reflectance.
export function surfaceReflectance(cosI, n1, n2, layers, lambda) {
  if (layers <= 0) return fresnelReflectance(cosI, n1, n2);
  const mgf2 = 1.38;
  let R = coatingReflectance(cosI, n1, n2, mgf2, 550 / (4 * mgf2), lambda);
  for (let i = 1; i < layers; i++) R *= 0.25;
  return Math.min(Math.max(R, 0), 1);
}

// ---------------------------------------------------------------------------
// Lens system
// ---------------------------------------------------------------------------

export class Lens {
  // rows: [radius, thickness, ior, abbe, semi_aperture, coating, is_stop]
  constructor(name, focal, rows, file = '') {
    this.name = name;
    this.file = file;
    this.focal = focal > 0 ? focal : 50;
    this.baseRows = rows.map(r => r.slice());
    this.stopScale = 1;
    this.surfaces = [];
    this.build();
  }

  static fromCatalog(e) { return new Lens(e.n, e.f, e.s, e.file); }

  static parse(text, fallbackName = 'Custom lens') {
    let name = '', focal = 0, inSurf = false;
    const rows = [];
    for (const raw of text.split(/\r?\n/)) {
      const line = raw.trim();
      if (!line || line[0] === '#') continue;
      if (!inSurf) {
        if (line.startsWith('name:')) name = line.slice(5).trim();
        else if (line.startsWith('focal_length:')) focal = parseFloat(line.slice(13));
        else if (line.startsWith('surfaces:')) inSurf = true;
        continue;
      }
      const t = line.split(/\s+/);
      let radius = 0, stop = 0;
      if (t[0] === 'stop' || t[0] === 'STOP') stop = 1;
      else if (t[0] !== 'inf' && t[0] !== 'INF') radius = parseFloat(t[0]);
      const [th, ior, abbe, semi, coat] = t.slice(1, 6).map(Number);
      if (!(semi > 0)) continue;
      rows.push([radius, th || 0, ior || 1, abbe || 0, semi, coat | 0, stop]);
    }
    if (!rows.length) throw new Error('no surfaces found in lens file');
    return new Lens(name || fallbackName, focal, rows);
  }

  // Extension over the Nuke node: the stop's clear semi-diameter can be
  // scaled to emulate stopping down (stopScale < 1).
  setStopScale(s) { this.stopScale = s; this.build(); }

  build() {
    let z = 0;
    this.surfaces = this.baseRows.map(r => {
      const s = {
        radius: r[0], thickness: r[1], ior: r[2], abbe: r[3],
        semi: r[4] * (r[6] ? this.stopScale : 1), coating: r[5], stop: !!r[6], z,
      };
      z += s.thickness;
      return s;
    });
    this.sensorZ = z;
  }

  get n() { return this.surfaces.length; }
  iorAt(i, lambda) { const s = this.surfaces[i]; return dispersionIor(s.ior, s.abbe, lambda); }
  iorBefore(i, lambda) { return i <= 0 ? 1 : this.iorAt(i - 1, lambda); }

  // Packed for the GPU: 8 floats per surface.
  packSurfaces() {
    const a = new Float32Array(this.n * 8);
    this.surfaces.forEach((s, i) => {
      a.set([s.radius, s.thickness, s.ior, s.abbe, s.semi, s.coating, s.z, 0], i * 8);
    });
    return a;
  }

  toText() {
    const lines = [`name: ${this.name}`, `focal_length: ${this.focal}`, '', 'surfaces:',
      '# radius    thickness   ior      abbe    semi_ap   coating'];
    for (const r of this.baseRows) {
      const rad = r[6] ? 'stop' : String(r[0]);
      lines.push(`  ${rad.padEnd(10)} ${String(r[1]).padEnd(10)} ${String(r[2]).padEnd(8)} ${String(r[3]).padEnd(7)} ${String(r[4]).padEnd(9)} ${r[5]}`);
    }
    return lines.join('\n') + '\n';
  }
}

// ---------------------------------------------------------------------------
// Ray tracing
// ---------------------------------------------------------------------------

function intersect(o, d, s, hit, nrm) {
  if (Math.abs(s.radius) < 1e-6) {
    if (Math.abs(d[2]) < 1e-12) return false;
    const t = (s.z - o[2]) / d[2];
    if (!(t > 1e-6)) return false;
    hit[0] = o[0] + d[0] * t; hit[1] = o[1] + d[1] * t; hit[2] = o[2] + d[2] * t;
    if (!(hit[0] * hit[0] + hit[1] * hit[1] <= s.semi * s.semi)) return false;
    nrm[0] = 0; nrm[1] = 0; nrm[2] = d[2] > 0 ? -1 : 1;
    return true;
  }
  const R = s.radius, cz = s.z + R;
  const ox = o[0], oy = o[1], oz = o[2] - cz;
  const a = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
  const b = 2 * (ox * d[0] + oy * d[1] + oz * d[2]);
  const c = ox * ox + oy * oy + oz * oz - R * R;
  const disc = b * b - 4 * a * c;
  if (disc < 0) return false;
  const sd = Math.sqrt(disc), inv2a = 0.5 / a;
  const t1 = (-b - sd) * inv2a, t2 = (-b + sd) * inv2a;
  let t;
  if (t1 > 1e-6 && t2 > 1e-6) {
    const z1 = o[2] + t1 * d[2], z2 = o[2] + t2 * d[2];
    t = Math.abs(z1 - s.z) < Math.abs(z2 - s.z) ? t1 : t2;
  } else if (t1 > 1e-6) t = t1;
  else if (t2 > 1e-6) t = t2;
  else return false;
  hit[0] = o[0] + d[0] * t; hit[1] = o[1] + d[1] * t; hit[2] = o[2] + d[2] * t;
  if (!(hit[0] * hit[0] + hit[1] * hit[1] <= s.semi * s.semi)) return false;
  const iR = 1 / Math.abs(R);
  nrm[0] = hit[0] * iR; nrm[1] = hit[1] * iR; nrm[2] = (hit[2] - cz) * iR;
  if (nrm[0] * d[0] + nrm[1] * d[1] + nrm[2] * d[2] > 0) { nrm[0] = -nrm[0]; nrm[1] = -nrm[1]; nrm[2] = -nrm[2]; }
  return true;
}

function normalizeInto(v, x, y, z) {
  const sq = x * x + y * y + z * z;
  if (sq < 1e-18 || !isFinite(sq)) return false;
  const inv = 1 / Math.sqrt(sq);
  v[0] = x * inv; v[1] = y * inv; v[2] = z * inv;
  return true;
}

function refract(d, n, eta) {
  const cosI = -(n[0] * d[0] + n[1] * d[1] + n[2] * d[2]);
  const sin2t = eta * eta * (1 - cosI * cosI);
  if (sin2t >= 1) return false;
  const k = eta * cosI - Math.sqrt(1 - sin2t);
  return normalizeInto(d, d[0] * eta + n[0] * k, d[1] * eta + n[1] * k, d[2] * eta + n[2] * k);
}

function reflect(d, n) {
  const d2 = 2 * (d[0] * n[0] + d[1] * n[1] + d[2] * n[2]);
  return normalizeInto(d, d[0] - n[0] * d2, d[1] - n[1] * d2, d[2] - n[2] * d2);
}

// Three-phase ghost trace (forward to b, back to a, forward to sensor).
// If `path` is an array, hit points are appended (for the lens diagram).
// Returns {x, y, weight} or null when vignetted / TIR.
export function traceGhost(origin, dir, lens, a, b, lambda, path = null) {
  const o = origin.slice(), d = dir.slice(), hit = [0, 0, 0], nrm = [0, 0, 0];
  const S = lens.surfaces, N = S.length;
  let ior = 1, w = 1;
  if (path) path.push(o.slice());
  const step = (s, n2, mode) => {           // mode: 0 transmit, 1 reflect
    if (!intersect(o, d, S[s], hit, nrm)) return false;
    o[0] = hit[0]; o[1] = hit[1]; o[2] = hit[2];
    if (path) path.push(o.slice());
    const cosI = Math.abs(nrm[0] * d[0] + nrm[1] * d[1] + nrm[2] * d[2]);
    const R = surfaceReflectance(cosI, ior, n2, S[s].coating, lambda);
    if (mode === 1) { if (!reflect(d, nrm)) return false; w *= R; return true; }
    if (!refract(d, nrm, ior / n2)) return false;
    w *= 1 - R; ior = n2;
    return true;
  };
  for (let s = 0; s <= b; s++) if (!step(s, lens.iorAt(s, lambda), s === b ? 1 : 0)) return null;
  for (let s = b - 1; s >= a; s--) {
    if (!step(s, lens.iorBefore(s, lambda), s === a ? 1 : 0)) return null;
    if (s === a) ior = lens.iorAt(a, lambda);
  }
  for (let s = a + 1; s < N; s++) if (!step(s, lens.iorAt(s, lambda), 0)) return null;
  if (Math.abs(d[2]) < 1e-12) return null;
  const t = (lens.sensorZ - o[2]) / d[2];
  if (!(t > 0)) return null;
  const x = o[0] + d[0] * t, y = o[1] + d[1] * t;
  if (path) path.push([x, y, lens.sensorZ]);
  return { x, y, weight: w };
}

// Ordinary (non-ghost) trace to the sensor, for the diagram's image-forming rays.
export function traceImaging(origin, dir, lens, lambda, path) {
  const o = origin.slice(), d = dir.slice(), hit = [0, 0, 0], nrm = [0, 0, 0];
  let ior = 1;
  path.push(o.slice());
  for (let s = 0; s < lens.n; s++) {
    if (!intersect(o, d, lens.surfaces[s], hit, nrm)) return false;
    o[0] = hit[0]; o[1] = hit[1]; o[2] = hit[2];
    path.push(o.slice());
    const n2 = lens.iorAt(s, lambda);
    if (!refract(d, nrm, ior / n2)) return false;
    ior = n2;
  }
  const t = (lens.sensorZ - o[2]) / d[2];
  if (!(t > 0)) return false;
  path.push([o[0] + d[0] * t, o[1] + d[1] * t, lens.sensorZ]);
  return true;
}

// ---------------------------------------------------------------------------
// Ghost pair enumeration and pre-filter (ghost.cpp)
// ---------------------------------------------------------------------------

const RGB_NM = [650, 550, 450];

function estimateIntensity(lens, a, b) {
  const z0 = lens.surfaces[0].z - 20;
  let total = 0;
  for (const l of RGB_NM) {
    const r = traceGhost([0, 0, z0], [0, 0, 1], lens, a, b, l);
    if (r) total += r.weight;
  }
  return total / 3;
}

function estimateSpread(lens, a, b, shw, shh, maxBoost) {
  const G = 8, R = lens.surfaces[0].semi, z0 = lens.surfaces[0].z - 20;
  let x0 = 1e30, x1 = -1e30, y0 = 1e30, y1 = -1e30, n = 0;
  for (let gy = 0; gy < G; gy++) for (let gx = 0; gx < G; gx++) {
    const u = ((gx + 0.5) / G) * 2 - 1, v = ((gy + 0.5) / G) * 2 - 1;
    if (u * u + v * v > 1) continue;
    const r = traceGhost([u * R, v * R, z0], [0, 0, 1], lens, a, b, 550);
    if (!r) continue;
    x0 = Math.min(x0, r.x); x1 = Math.max(x1, r.x);
    y0 = Math.min(y0, r.y); y1 = Math.max(y1, r.y);
    n++;
  }
  if (n < 2) return 1;
  const ratio = (Math.max(x1 - x0, 0.01) * Math.max(y1 - y0, 0.01)) / (4 * shw * shh);
  return Math.min(Math.max(ratio, 1), maxBoost);
}

// Returns [{a, b, est, boost}] for every pair above minIntensity.
export function filterGhostPairs(lens, shw, shh, { minIntensity = 1e-7, normalize = true, maxAreaBoost = 100 } = {}) {
  const out = [];
  const S = lens.surfaces;
  const iorB = i => (i <= 0 ? 1 : S[i - 1].ior);
  for (let a = 0; a < lens.n; a++) {
    for (let b = a + 1; b < lens.n; b++) {
      if (Math.abs(iorB(a) - S[a].ior) < 0.001 || Math.abs(iorB(b) - S[b].ior) < 0.001) continue;
      const est = estimateIntensity(lens, a, b);
      if (est < minIntensity) continue;
      const boost = normalize ? estimateSpread(lens, a, b, shw, shh, maxAreaBoost) : 1;
      out.push({ a, b, est, boost });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Pupil sampling and spectral table (ghost_cuda.cu launcher, FlareLensOptics.cpp)
// ---------------------------------------------------------------------------

function wangHash(s) {
  s = ((s ^ 61) ^ (s >>> 16)) >>> 0;
  s = Math.imul(s, 9) >>> 0;
  s = (s ^ (s >>> 4)) >>> 0;
  s = Math.imul(s, 0x27d4eb2d) >>> 0;
  s = (s ^ (s >>> 15)) >>> 0;
  return s;
}
function halton2(n) {
  n = ((n << 16) | (n >>> 16)) >>> 0;
  n = (((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >>> 8)) >>> 0;
  n = (((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >>> 4)) >>> 0;
  n = (((n & 0x33333333) << 2) | ((n & 0xcccccccc) >>> 2)) >>> 0;
  n = (((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >>> 1)) >>> 0;
  return n / 4294967296;
}
function halton3(n) {
  let r = 0, f = 1 / 3;
  while (n > 0) { r += (n % 3) * f; n = Math.floor(n / 3); f /= 3; }
  return r;
}

export function insideAperture(u, v, blades, rotDeg) {
  const r2 = u * u + v * v;
  if (r2 > 1) return false;
  if (blades < 3) return true;
  const sector = 2 * Math.PI / blades;
  let t = (Math.atan2(v, u) - rotDeg * Math.PI / 180) % sector;
  if (t < 0) t += sector;
  return Math.sqrt(r2) * Math.cos(t - sector * 0.5) <= Math.cos(Math.PI / blades);
}

// mode: 0 grid, 1 stratified, 2 Halton, 3 golden-angle (Vogel spiral, from FlareSimAE)
export function buildPupil(N, blades, rotDeg, mode, seed) {
  const out = [];
  const seedOff = Math.imul(seed >>> 0, 1000003) >>> 0;
  for (let k = 0; k < N * N; k++) {
    const gx = k % N, gy = (k / N) | 0;
    let u, v;
    if (mode === 3) {
      const r = Math.sqrt((k + 0.5) / (N * N)), ang = k * 2.39996322972865332;
      u = r * Math.cos(ang); v = r * Math.sin(ang);
    } else if (mode === 2) {
      u = halton2(k) * 2 - 1; v = halton3(k) * 2 - 1;
    } else {
      const ju = mode === 1 ? wangHash((k + seedOff) >>> 0) / 4294967296 : 0.5;
      const jv = mode === 1 ? wangHash((k + N * N + seedOff) >>> 0) / 4294967296 : 0.5;
      u = ((gx + ju) / N) * 2 - 1; v = ((gy + jv) / N) * 2 - 1;
    }
    if (insideAperture(u, v, blades, rotDeg)) out.push(u, v);
  }
  return new Float32Array(out);
}

// [lambda, rw, gw, bw] per sample; Gaussian approximation of CIE matching.
export function buildSpectrum(n) {
  if (n <= 3) return new Float32Array([650, 1, 0, 0, 550, 0, 1, 0, 450, 0, 0, 1]);
  const g = x => Math.exp(-0.5 * x * x);
  const rows = [];
  let sr = 0, sg = 0, sb = 0;
  for (let i = 0; i < n; i++) {
    const l = 400 + 300 * i / (n - 1);
    const r = Math.max(0, 0.63 * g((l - 600) / 70) + 0.22 * g((l - 450) / 30));
    const gg = Math.max(0, g((l - 545) / 55));
    const b = Math.max(0, g((l - 445) / 45));
    rows.push([l, r, gg, b]); sr += r; sg += gg; sb += b;
  }
  return new Float32Array(rows.flatMap(([l, r, g2, b]) => [l, r / sr, g2 / sg, b / sb]));
}
