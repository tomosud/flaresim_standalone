// ============================================================================
// diagram.js — 2D meridional cross-section of the lens with ghost ray paths.
// ============================================================================
import { traceGhost, traceImaging } from './optics.js';

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

// pair: {a, b} or null; angle: field angle (rad) along y
export function drawLensDiagram(canvas, lens, pair, angle = 0) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  if (!lens) return;
  const S = lens.surfaces;
  const z0 = S[0].z - 20, z1 = lens.sensorZ;
  const maxSemi = Math.max(...S.map(s => s.semi));
  const pad = 10;
  const k = Math.min((W - 2 * pad) / (z1 - z0), (H - 2 * pad) / (2 * maxSemi * 1.05));
  const X = z => pad + (z - z0) * k;
  const Y = y => H / 2 - y * k;

  // optical axis + sensor
  g.strokeStyle = css('--line'); g.lineWidth = 1; g.setLineDash([3, 3]);
  g.beginPath(); g.moveTo(X(z0), Y(0)); g.lineTo(X(z1), Y(0)); g.stroke(); g.setLineDash([]);
  g.strokeStyle = css('--muted'); g.lineWidth = 2;
  g.beginPath(); g.moveTo(X(z1), Y(maxSemi)); g.lineTo(X(z1), Y(-maxSemi)); g.stroke();

  const sag = (s, y) => {
    if (Math.abs(s.radius) < 1e-6) return 0;
    const R = s.radius, v = R * R - y * y;
    return v <= 0 ? R : R - Math.sign(R) * Math.sqrt(v);
  };
  // glass fill between surfaces with ior > 1
  g.fillStyle = css('--glass');
  for (let i = 0; i < S.length - 1; i++) {
    if (S[i].ior <= 1.0001) continue;
    const a = S[i], b = S[i + 1], h = Math.min(a.semi, b.semi);
    g.beginPath();
    for (let t = -1; t <= 1.0001; t += 0.05) g.lineTo(X(a.z + sag(a, t * h)), Y(t * h));
    for (let t = 1; t >= -1.0001; t -= 0.05) g.lineTo(X(b.z + sag(b, t * h)), Y(t * h));
    g.closePath(); g.fill();
  }
  // surfaces
  S.forEach((s, i) => {
    const hl = pair && (i === pair.a || i === pair.b);
    g.strokeStyle = s.stop ? css('--warn') : hl ? css('--accent') : css('--ink2');
    g.lineWidth = hl ? 2 : 1;
    g.beginPath();
    if (s.stop) {
      g.moveTo(X(s.z), Y(s.semi)); g.lineTo(X(s.z), Y(s.semi + 6 / k));
      g.moveTo(X(s.z), Y(-s.semi)); g.lineTo(X(s.z), Y(-s.semi - 6 / k));
    } else {
      for (let t = -1; t <= 1.0001; t += 0.05) g.lineTo(X(s.z + sag(s, t * s.semi)), Y(t * s.semi));
    }
    g.stroke();
  });

  const R0 = S[0].semi, dir = [0, Math.sin(angle), Math.cos(angle)];
  const fan = 11;
  // image-forming rays (faint)
  g.strokeStyle = css('--ray'); g.lineWidth = 0.7; g.globalAlpha = 0.5;
  for (let i = 0; i < fan; i++) {
    const y = (i / (fan - 1) * 2 - 1) * R0 * 0.98, path = [];
    if (traceImaging([0, y, z0], dir, lens, 550, path) || path.length > 2) polyline(g, path, X, Y);
  }
  g.globalAlpha = 1;
  if (pair) {
    g.strokeStyle = css('--accent'); g.lineWidth = 1;
    for (let i = 0; i < fan; i++) {
      const y = (i / (fan - 1) * 2 - 1) * R0 * 0.98, path = [];
      const r = traceGhost([0, y, z0], dir, lens, pair.a, pair.b, 550, path);
      g.globalAlpha = r ? 0.9 : 0.25;
      polyline(g, path, X, Y);
    }
    g.globalAlpha = 1;
  }
}

function polyline(g, path, X, Y) {
  g.beginPath();
  path.forEach((p, i) => (i ? g.lineTo(X(p[2]), Y(p[1])) : g.moveTo(X(p[2]), Y(p[1]))));
  g.stroke();
}
