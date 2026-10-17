// ============================================================================
// imageio.js — load EXR / HDR / SDR images into linear float RGBA,
// and build a procedural HDR test scene.
// ============================================================================

const srgbToLinear = c => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));

let threeMods = null;
async function three() {
  if (!threeMods) {
    const [T, E, R] = await Promise.all([
      import('three'),
      import('three/addons/loaders/EXRLoader.js'),
      import('three/addons/loaders/RGBELoader.js'),
    ]);
    threeMods = { T, EXRLoader: E.EXRLoader, RGBELoader: R.RGBELoader };
  }
  return threeMods;
}

function toRGBA(src, w, h, channels, flipY) {
  const out = new Float32Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    const sy = flipY ? h - 1 - y : y;
    for (let x = 0; x < w; x++) {
      const i = (sy * w + x) * channels, o = (y * w + x) * 4;
      if (channels === 1) { out[o] = out[o + 1] = out[o + 2] = src[i]; out[o + 3] = 1; }
      else {
        out[o] = src[i]; out[o + 1] = src[i + 1]; out[o + 2] = src[i + 2];
        out[o + 3] = channels === 4 ? src[i + 3] : 1;
      }
    }
  }
  // Guard against NaN / Inf in source files.
  for (let i = 0; i < out.length; i++) if (!Number.isFinite(out[i])) out[i] = 0;
  return out;
}

// opts.sdrExpand: inverse tone curve for 8-bit images so highlights exceed 1.0
export async function loadImageFile(file, opts = {}) {
  const name = file.name || 'image';
  const ext = name.split('.').pop().toLowerCase();
  if (ext === 'exr') {
    const { T, EXRLoader } = await three();
    const r = new EXRLoader().setDataType(T.FloatType).parse(await file.arrayBuffer());
    const ch = r.format === T.RedFormat ? 1 : 4;
    // three's EXRLoader stores rows bottom-up (GL convention) — flip back.
    return { name, width: r.width, height: r.height, hdr: true, data: toRGBA(r.data, r.width, r.height, ch, true) };
  }
  if (ext === 'hdr' || ext === 'pic') {
    const { T, RGBELoader } = await three();
    const r = new RGBELoader().setDataType(T.FloatType).parse(await file.arrayBuffer());
    return { name, width: r.width, height: r.height, hdr: true, data: toRGBA(r.data, r.width, r.height, 4, false) };
  }
  const bmp = await createImageBitmap(file, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  const cv = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = cv.getContext('2d');
  ctx.drawImage(bmp, 0, 0);
  const px = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
  const lut = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    let v = srgbToLinear(i / 255);
    if (opts.sdrExpand) v = v / Math.max(1 - 0.98 * v, 0.02);   // 1.0 -> 50
    lut[i] = v;
  }
  const data = new Float32Array(bmp.width * bmp.height * 4);
  for (let i = 0; i < px.length; i += 4) {
    data[i] = lut[px[i]]; data[i + 1] = lut[px[i + 1]]; data[i + 2] = lut[px[i + 2]];
    data[i + 3] = px[i + 3] / 255;
  }
  return { name, width: bmp.width, height: bmp.height, hdr: false, data };
}

// Dusk sky + sun + a row of street lights. Values are scene-linear HDR.
export function makeTestScene(w = 1920, h = 1080) {
  const data = new Float32Array(w * h * 4);
  const lights = [
    { x: 0.72, y: 0.30, r: 0.018, c: [1.0, 0.92, 0.78], e: 400 },    // sun
  ];
  for (let i = 0; i < 6; i++) lights.push({ x: 0.08 + i * 0.07, y: 0.62 + i * 0.012, r: 0.0035, c: [1.0, 0.75, 0.45], e: 80 });
  const horizon = 0.66;
  for (let y = 0; y < h; y++) {
    const v = y / h;
    for (let x = 0; x < w; x++) {
      const u = x / w, o = (y * w + x) * 4;
      let r, g, b;
      if (v < horizon) {
        const t = v / horizon;
        r = 0.02 + 0.30 * t * t; g = 0.03 + 0.13 * t * t; b = 0.10 + 0.05 * t;
      } else {
        const t = (v - horizon) / (1 - horizon);
        r = 0.012 * (1 - t) + 0.004; g = 0.010 * (1 - t) + 0.004; b = 0.012 * (1 - t) + 0.005;
        if (((u * 40) | 0) % 7 === 0 && t < 0.4) { r += 0.03; g += 0.025; b += 0.02; }
      }
      for (const L of lights) {
        const dx = (u - L.x) * (w / h), dy = v - L.y, d2 = dx * dx + dy * dy;
        if (d2 < L.r * L.r) { r += L.c[0] * L.e; g += L.c[1] * L.e; b += L.c[2] * L.e; }
        const glow = 0.004 * L.e * L.r * L.r / (d2 + 1e-5);
        r += L.c[0] * glow * 0.02; g += L.c[1] * glow * 0.02; b += L.c[2] * glow * 0.02;
      }
      data[o] = r; data[o + 1] = g; data[o + 2] = b; data[o + 3] = 1;
    }
  }
  return { name: 'test_scene', width: w, height: h, hdr: true, data };
}
