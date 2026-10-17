// ============================================================================
// sources.js — bright-source extraction (FlareSim.cpp do_compute), source
// clustering, max-source cap and the FlareSimAE manual light.
// ============================================================================

const luma = s => 0.2126 * s.r + 0.7152 * s.g + 0.0722 * s.b;

// img: {width, height, data: Float32Array RGBA (top-down rows)}
// Returns {sources, hazeGrid, dw, dh, ds}
//   sources: [{ax, ay, px, py, r, g, b}]  (angles in radians, px/py in pixels)
//   hazeGrid: Float32Array(dw*dh*4) block colours above threshold (for haze seed)
export function detectSources(img, p, tanH, tanV) {
  const { width: w, height: h, data } = img;
  const ds = Math.max(1, p.downsample | 0);
  const dw = Math.max(1, Math.floor(w / ds)), dh = Math.max(1, Math.floor(h / ds));
  const acc = new Float32Array(dw * dh * 4);   // r, g, b, mask
  const cnt = new Uint32Array(dw * dh);
  const gain = p.inputGain;
  for (let y = 0; y < h; y++) {
    const dy = Math.min((y / ds) | 0, dh - 1);
    for (let x = 0; x < w; x++) {
      const dx = Math.min((x / ds) | 0, dw - 1);
      const bi = dy * dw + dx, i = (y * w + x) * 4;
      acc[bi * 4] += data[i];
      acc[bi * 4 + 1] += data[i + 1];
      acc[bi * 4 + 2] += data[i + 2];
      acc[bi * 4 + 3] += p.alphaGate ? data[i + 3] : 1;
      cnt[bi]++;
    }
  }

  const sources = [];
  const hazeGrid = new Float32Array(dw * dh * 4);
  if (p.detect) {
    for (let dy = 0; dy < dh; dy++) {
      for (let dx = 0; dx < dw; dx++) {
        const bi = dy * dw + dx, c = cnt[bi];
        if (!c) continue;
        let r = acc[bi * 4] / c * gain, g = acc[bi * 4 + 1] / c * gain, b = acc[bi * 4 + 2] / c * gain;
        let L = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        if (p.sourceCap > 0 && L > p.sourceCap) {
          const k = p.sourceCap / L; r *= k; g *= k; b *= k; L = p.sourceCap;
        }
        const m = acc[bi * 4 + 3] / c;
        L *= m; r *= m; g *= m; b *= m;
        if (L < p.threshold) continue;
        const bx0 = dx * ds, bx1 = Math.min(bx0 + ds, w);
        const by0 = dy * ds, by1 = Math.min(by0 + ds, h);
        const cx = (bx0 + bx1) * 0.5, cy = (by0 + by1) * 0.5;
        sources.push(makeSource(cx, cy, w, h, tanH, tanV, r, g, b));
        hazeGrid.set([r, g, b, 1], bi * 4);
      }
    }
  }

  if (p.manual) {
    const m = p.manual;
    // FlareSimAE: a finite disk of D px equals ~pi(D/2)^2/ds^2 Nuke blocks.
    let rad = m.radiance;
    if (m.diameter > 0) {
      const blocks = Math.min(Math.max(Math.PI * (m.diameter / 2) ** 2 / (ds * ds), 1), Math.max(1, p.maxSources || 1e9));
      rad *= blocks;
    }
    const s = makeSource(m.x, m.y, w, h, tanH, tanV, rad * m.tint[0], rad * m.tint[1], rad * m.tint[2]);
    s.manual = true;
    sources.push(s);
    const bx = Math.floor(m.x / ds), by = Math.floor(m.y / ds);
    if (bx >= 0 && by >= 0 && bx < dw && by < dh) {
      const bi = (by * dw + bx) * 4;
      hazeGrid[bi] += m.radiance * m.tint[0];
      hazeGrid[bi + 1] += m.radiance * m.tint[1];
      hazeGrid[bi + 2] += m.radiance * m.tint[2];
    }
  }

  let out = sources;
  if (p.clusterRadius > 0 && out.length > 1) out = clusterSources(out, p.clusterRadius, w, h, tanH, tanV);
  const total = out.length;
  if (p.maxSources > 0 && out.length > p.maxSources) {
    out.sort((a, b) => luma(b) - luma(a));
    out = out.slice(0, p.maxSources);
  }
  return { sources: out, totalBeforeCap: total, hazeGrid, dw, dh, ds };
}

function makeSource(cx, cy, w, h, tanH, tanV, r, g, b) {
  const nx = (cx - w * 0.5) / w, ny = (cy - h * 0.5) / h;
  return { ax: Math.atan(nx * 2 * tanH), ay: Math.atan(ny * 2 * tanV), px: cx, py: cy, r, g, b };
}

// Greedy luma-ordered clustering (cluster_sources in FlareSim.cpp).
function clusterSources(src, radiusPx, w, h, tanH, tanV) {
  const thresh = radiusPx * (2 * tanH / w);
  const t2 = thresh * thresh;
  src = src.slice().sort((a, b) => luma(b) - luma(a));
  const used = new Uint8Array(src.length);
  const out = [];
  for (let i = 0; i < src.length; i++) {
    if (used[i]) continue;
    const s = src[i];
    let sw = luma(s), ax = s.ax * sw, ay = s.ay * sw, r = s.r, g = s.g, b = s.b;
    for (let j = i + 1; j < src.length; j++) {
      if (used[j]) continue;
      const t = src[j], dx = s.ax - t.ax, dy = s.ay - t.ay;
      if (dx * dx + dy * dy <= t2) {
        const lj = luma(t);
        r += t.r; g += t.g; b += t.b; ax += t.ax * lj; ay += t.ay * lj; sw += lj;
        used[j] = 1;
      }
    }
    const mx = sw > 0 ? ax / sw : s.ax, my = sw > 0 ? ay / sw : s.ay;
    out.push({
      ax: mx, ay: my, r, g, b, manual: s.manual,
      px: w * 0.5 + Math.tan(mx) / (2 * tanH) * w,
      py: h * 0.5 + Math.tan(my) / (2 * tanV) * h,
    });
  }
  return out;
}
