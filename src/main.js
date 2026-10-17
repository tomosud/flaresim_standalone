// ============================================================================
// main.js — UI, parameter state, render scheduling, viewer and export.
// ============================================================================
import { Lens, filterGhostPairs, buildPupil, buildSpectrum } from './optics.js';
import { detectSources } from './sources.js';
import { computeStarburstPSF } from './starburst.js';
import { FlareGPU } from './gpu.js';
import { writeEXR } from './exr.js';
import { loadImageFile, makeTestScene } from './imageio.js';
import { drawLensDiagram } from './diagram.js';

const $ = id => document.getElementById(id);

// ---------------------------------------------------------------------------
// Parameters (names follow the Nuke / AE controls)
// ---------------------------------------------------------------------------
const SENSORS = [
  ['Custom', 0, 0], ['Full Frame', 36, 24], ['Super 35', 24.89, 18.66],
  ['APS-C Canon', 22.3, 14.9], ['APS-C Nikon/Sony', 23.5, 15.6], ['Micro Four Thirds', 17.3, 13],
];
const SPECTRAL = [['3 (R/G/B)', 3], ['5', 5], ['7', 7], ['9', 9], ['11', 11]];

const P = {
  stopScale: { label: 'Stop scale', type: 'range', min: 0.05, max: 1, step: 0.01, def: 1, hint: '絞り面の有効径の倍率（拡張機能）。小さくすると絞り込んだ状態に近づきます' },
  flareGain: { label: 'Flare Gain', type: 'range', min: 0, max: 20000, step: 1, def: 1000, log: true, live: true, hint: 'ゴーストの強度倍率。2 回反射なので数百〜数千が普通' },
  rayGrid: { label: 'Ray Grid (NxN)', type: 'int', min: 4, max: 256, step: 1, def: 64, hint: '入射瞳のサンプル数（一辺）。N² 本の光線' },
  pupilJitter: { label: 'Pupil Sampling', type: 'select', options: [['Grid', 0], ['Stratified', 1], ['Halton', 2], ['Golden angle', 3]], def: 2, hint: 'Grid は低レイ数でドット模様が出やすい。Halton / Golden angle 推奨' },
  jitterSeed: { label: 'Jitter Seed', type: 'int', min: 0, max: 9999, step: 1, def: 0 },
  spectral: { label: 'Spectral Samples', type: 'select', options: SPECTRAL, def: 3 },
  detect: { label: 'Detect bright sources', type: 'check', def: true },
  threshold: { label: 'Threshold', type: 'range', min: 0.1, max: 1000, step: 0.01, def: 2, log: true, hint: 'シーンリニア輝度のしきい値' },
  inputEV: { label: 'Input EV', type: 'range', min: -8, max: 8, step: 0.1, def: 0, hint: '入力画像の露出補正（検出と表示の両方）' },
  sourceCap: { label: 'Source Cap (0=off)', type: 'range', min: 0, max: 100000, step: 1, def: 0, log: true },
  maxSources: { label: 'Max Sources', type: 'int', min: 1, max: 2048, step: 1, def: 512 },
  downsample: { label: 'Downsample', type: 'int', min: 1, max: 64, step: 1, def: 4 },
  clusterRadius: { label: 'Cluster Radius (px)', type: 'int', min: 0, max: 200, step: 1, def: 0 },
  alphaGate: { label: 'Gate with input alpha', type: 'check', def: false },
  sdrExpand: { label: 'SDR highlight expand', type: 'check', def: true, hint: 'PNG/JPG を読み込むとき白飛び付近を HDR に持ち上げる (1.0 → 50)' },
  manualOn: { label: 'Enable manual light', type: 'check', def: false },
  manualX: { label: 'Light X (px)', type: 'range', min: -4000, max: 8000, step: 1, def: 1300 },
  manualY: { label: 'Light Y (px)', type: 'range', min: -4000, max: 8000, step: 1, def: 330 },
  manualRadiance: { label: 'Radiance', type: 'range', min: 0, max: 100000, step: 1, def: 2000, log: true },
  manualDiameter: { label: 'Diameter (px)', type: 'range', min: 0, max: 400, step: 1, def: 30 },
  manualTint: { label: 'Tint', type: 'color', def: '#fff3e0' },
  useSensor: { label: 'Use sensor size', type: 'check', def: false },
  fovH: { label: 'FOV H (deg)', type: 'range', min: 1, max: 150, step: 0.1, def: 40 },
  autoFovV: { label: 'Auto FOV V', type: 'check', def: true },
  fovV: { label: 'FOV V (deg)', type: 'range', min: 1, max: 150, step: 0.1, def: 24 },
  sensorPreset: { label: 'Sensor preset', type: 'select', options: SENSORS.map((s, i) => [s[0], i]), def: 1 },
  sensorW: { label: 'Sensor W (mm)', type: 'range', min: 1, max: 80, step: 0.01, def: 36 },
  sensorH: { label: 'Sensor H (mm)', type: 'range', min: 1, max: 80, step: 0.01, def: 24 },
  focal: { label: 'Focal length (mm)', type: 'range', min: 4, max: 800, step: 0.1, def: 50, log: true },
  blades: { label: 'Aperture Blades (0=circle)', type: 'int', min: 0, max: 16, step: 1, def: 0 },
  rotation: { label: 'Aperture Rotation', type: 'range', min: 0, max: 360, step: 0.1, def: 0 },
  ghostBlur: { label: 'Ghost Blur', type: 'range', min: 0, max: 0.05, step: 0.0005, def: 0.003, hint: '画像対角に対する比率' },
  ghostBlurPasses: { label: 'Ghost Blur Passes', type: 'int', min: 0, max: 8, step: 1, def: 3 },
  hazeGain: { label: 'Haze Gain', type: 'range', min: 0, max: 2, step: 0.001, def: 0, live: true },
  hazeRadius: { label: 'Haze Radius', type: 'range', min: 0, max: 0.5, step: 0.005, def: 0.15 },
  hazePasses: { label: 'Haze Blur Passes', type: 'int', min: 0, max: 8, step: 1, def: 3 },
  starGain: { label: 'Starburst Gain', type: 'range', min: 0, max: 2000, step: 0.1, def: 0, log: true, live: true },
  starScale: { label: 'Starburst Scale', type: 'range', min: 0.01, max: 0.6, step: 0.005, def: 0.15 },
  preview: { label: 'Enable preview mode', type: 'check', def: false },
  pvRayGrid: { label: 'Ray Grid', type: 'int', min: 4, max: 128, step: 1, def: 16 },
  pvMaxSources: { label: 'Max Sources', type: 'int', min: 1, max: 2048, step: 1, def: 100 },
  pvDownsample: { label: 'Downsample', type: 'int', min: 1, max: 64, step: 1, def: 8 },
  pvSpectral: { label: 'Spectral', type: 'select', options: SPECTRAL, def: 3 },
  exrHalf: { label: 'Half float (16-bit)', type: 'check', def: true },
  exrZip: { label: 'ZIP compression', type: 'check', def: true },
  exrLayers: { label: 'Write flare / haze / starburst layers', type: 'check', def: true },
};

const STORE = 'flaresim-web-v1';
const state = {};
for (const [k, d] of Object.entries(P)) state[k] = d.def;
const view = { exposure: 0, mode: 0, hdr: false, tonemap: 2, zoom: 1, offX: 0, offY: 0 };
try {
  const saved = JSON.parse(localStorage.getItem(STORE) || '{}');
  for (const k of Object.keys(P)) if (k in (saved.p || {})) state[k] = saved.p[k];
  Object.assign(view, saved.v || {});
  state.lensFile = saved.lensFile;
} catch { /* storage unavailable */ }
function persist() {
  try {
    localStorage.setItem(STORE, JSON.stringify({ p: Object.fromEntries(Object.keys(P).map(k => [k, state[k]])), v: { exposure: view.exposure, mode: view.mode, hdr: view.hdr, tonemap: view.tonemap }, lensFile: lens?.file }));
  } catch { /* ignore */ }
}

// ---------------------------------------------------------------------------
// Parameter widgets
// ---------------------------------------------------------------------------
const widgets = {};
const fmt = (d, v) => (d.type === 'int' ? String(v) : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(4));
const toSlider = (d, v) => (d.log ? Math.log10(Math.max(v, d.min > 0 ? d.min : 0.01) + (d.min > 0 ? 0 : 1)) : v);
const fromSlider = (d, s) => (d.log ? 10 ** s - (d.min > 0 ? 0 : 1) : s);

function buildWidgets() {
  for (const el of document.querySelectorAll('.param')) {
    const key = el.dataset.key, d = P[key];
    if (d.hint) el.title = d.hint;
    if (d.type === 'check') {
      el.innerHTML = `<label class="check"><input type="checkbox"> ${d.label}</label>`;
      const cb = el.querySelector('input');
      cb.checked = !!state[key];
      cb.onchange = () => setParam(key, cb.checked);
      widgets[key] = { set: v => (cb.checked = !!v), el };
    } else if (d.type === 'select') {
      el.innerHTML = `<label>${d.label}</label><select>${d.options.map(([n, v]) => `<option value="${v}">${n}</option>`).join('')}</select>`;
      const s = el.querySelector('select');
      s.value = state[key];
      s.onchange = () => setParam(key, Number(s.value));
      widgets[key] = { set: v => (s.value = v), el };
    } else if (d.type === 'color') {
      el.innerHTML = `<label>${d.label}</label><input type="color">`;
      const c = el.querySelector('input');
      c.value = state[key];
      c.oninput = () => setParam(key, c.value);
      widgets[key] = { set: v => (c.value = v), el };
    } else {
      const lo = toSlider(d, d.min), hi = toSlider(d, d.max);
      const step = d.log ? (hi - lo) / 1000 : d.step;
      el.innerHTML = `<label>${d.label}</label><div class="slider"><input type="range" min="${lo}" max="${hi}" step="${step}"><input type="number" step="${d.step}"></div>`;
      const [r, n] = el.querySelectorAll('input');
      const sync = v => { r.value = toSlider(d, v); n.value = fmt(d, v); };
      sync(state[key]);
      r.oninput = () => {
        let v = fromSlider(d, Number(r.value));
        if (d.type === 'int') v = Math.round(v);
        else v = Math.round(v / d.step) * d.step;
        n.value = fmt(d, v);
        setParam(key, v, true);
      };
      r.onchange = () => setParam(key, state[key]);
      n.onchange = () => {
        let v = Number(n.value);
        if (!Number.isFinite(v)) v = d.def;
        if (d.type === 'int') v = Math.round(v);
        // manual light position may be off-frame, so it is not clamped
        if (key !== 'manualX' && key !== 'manualY') v = Math.min(Math.max(v, d.min), d.max);
        sync(v);
        setParam(key, v);
      };
      widgets[key] = { set: sync, el };
    }
  }
  refreshEnables();
}

function refreshEnables() {
  const dis = (k, off) => widgets[k]?.el.classList.toggle('disabled', off);
  dis('fovH', state.useSensor); dis('autoFovV', state.useSensor);
  dis('fovV', state.useSensor || state.autoFovV);
  dis('sensorPreset', !state.useSensor);
  const custom = state.sensorPreset === 0;
  dis('sensorW', !state.useSensor || !custom); dis('sensorH', !state.useSensor || !custom);
  dis('focal', !state.useSensor);
  dis('jitterSeed', state.pupilJitter !== 1);
  for (const k of ['pvRayGrid', 'pvMaxSources', 'pvDownsample', 'pvSpectral']) dis(k, !state.preview);
  for (const k of ['manualX', 'manualY', 'manualRadiance', 'manualDiameter', 'manualTint']) dis(k, !state.manualOn);
  for (const k of ['threshold', 'sourceCap', 'downsample', 'clusterRadius', 'alphaGate']) dis(k, !state.detect);
}

function setParam(key, v, dragging = false) {
  state[key] = v;
  if (key === 'sensorPreset' && v > 0) {
    state.sensorW = SENSORS[v][1]; state.sensorH = SENSORS[v][2];
    widgets.sensorW.set(state.sensorW); widgets.sensorH.set(state.sensorH);
  }
  if (key === 'sdrExpand' && image && !image.hdr && image.file) { loadFile(image.file); return; }
  if (key === 'stopScale' && lens) { lens.setStopScale(v); updateLensInfo(); }
  refreshEnables();
  persist();
  // Gains only rescale cached buffers; the render pass is a no-op unless a
  // component (e.g. haze going from 0 to >0) has never been computed.
  if (P[key].live) { draw(); scheduleRender(dragging ? 250 : 60); return; }
  if (key.startsWith('exr')) return;
  if (key.startsWith('manual')) drawOverlay();
  scheduleRender(dragging ? 250 : 60);
}

// ---------------------------------------------------------------------------
// Lens catalog
// ---------------------------------------------------------------------------
let catalog = [];
let lens = null;
let pairEnabled = new Map();
let pairsCache = { key: '', pairs: [] };
let selectedPair = null;

async function loadCatalog() {
  const res = await fetch('lenses/lenses.json');
  catalog = await res.json();
  fillLensList('');
  const initial = catalog.find(l => l.file === state.lensFile) || catalog.find(l => /Master Prime T1\.3 50mm/.test(l.n)) || catalog[0];
  selectLens(Lens.fromCatalog(initial));
}

function fillLensList(q) {
  const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
  const list = $('lensList');
  const items = catalog.filter(l => terms.every(t => (l.n + ' ' + l.file).toLowerCase().includes(t)));
  list.innerHTML = items.map(l => `<option value="${catalog.indexOf(l)}">${escapeHtml(l.n)}</option>`).join('');
  if (lens) list.value = String(catalog.findIndex(l => l.file === lens.file));
}
const escapeHtml = s => s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function selectLens(l) {
  lens = l;
  lens.setStopScale(state.stopScale);
  pairEnabled = new Map();
  selectedPair = null;
  pairsCache.key = '';
  updateLensInfo();
  persist();
  scheduleRender(0);
}

function updateLensInfo() {
  const stop = lens.surfaces.find(s => s.stop);
  $('lensInfo').innerHTML = `<b>${escapeHtml(lens.name)}</b><br>f = ${lens.focal} mm · ${lens.n} surfaces · ${stop ? 'stop ⌀' + (stop.semi * 2).toFixed(1) + ' mm' : 'no stop'} · track ${lens.sensorZ.toFixed(1)} mm`;
  drawDiagram();
}

function drawDiagram() {
  const src = lastSources?.[0];
  drawLensDiagram($('diagram'), lens, selectedPair, src ? src.ay : 0);
}

// ---------------------------------------------------------------------------
// Camera / effective settings
// ---------------------------------------------------------------------------
function camera(w, h) {
  let fh, fv;
  if (state.useSensor) {
    const fl = Math.max(state.focal, 0.1);
    fh = 2 * Math.atan(state.sensorW / (2 * fl));
    fv = 2 * Math.atan(state.sensorH / (2 * fl));
  } else {
    fh = state.fovH * Math.PI / 180;
    fv = state.autoFovV ? 2 * Math.atan(Math.tan(fh / 2) / (w / h)) : state.fovV * Math.PI / 180;
  }
  return { fh, fv, tanH: Math.tan(fh / 2), tanV: Math.tan(fv / 2) };
}
function eff() {
  const pv = state.preview;
  return {
    rayGrid: pv ? state.pvRayGrid : state.rayGrid,
    maxSources: pv ? state.pvMaxSources : state.maxSources,
    downsample: pv ? state.pvDownsample : state.downsample,
    spectral: pv ? state.pvSpectral : state.spectral,
    dsRatio: pv ? state.pvDownsample / Math.max(state.downsample, 1) : 1,
  };
}
const hexRGB = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16) / 255).map(c => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));

// ---------------------------------------------------------------------------
// Render scheduling
// ---------------------------------------------------------------------------
let gpu = null, image = null;
let lastSources = [], sourceInfo = { total: 0 };
const keys = { src: '', ghost: '', haze: '', star: '' };
let psfCache = { key: '', psf: null };
let renderTimer = 0, rendering = false, rerender = false, generation = 0;

function scheduleRender(delay = 60) {
  clearTimeout(renderTimer);
  generation++;
  renderTimer = setTimeout(() => runRender(), delay);
}

async function runRender() {
  if (rendering) { rerender = true; return; }
  rendering = true;
  try {
    do { rerender = false; await renderOnce(generation); } while (rerender);
  } catch (e) {
    console.error(e);
    setStatus('エラー: ' + e.message, true);
  }
  rendering = false;
}

async function renderOnce(gen) {
  if (!gpu || !image || !lens) return;
  const cancelled = () => gen !== generation;
  const { width: w, height: h } = image;
  const cam = camera(w, h);
  const e = eff();
  const t0 = performance.now();

  // 1. sources
  const manual = state.manualOn ? { x: state.manualX, y: state.manualY, radiance: state.manualRadiance, diameter: state.manualDiameter, tint: hexRGB(state.manualTint) } : null;
  const srcParams = {
    detect: state.detect, threshold: state.threshold, inputGain: 2 ** state.inputEV, sourceCap: state.sourceCap,
    maxSources: e.maxSources, downsample: e.downsample, clusterRadius: state.clusterRadius, alphaGate: state.alphaGate, manual,
  };
  const srcKey = JSON.stringify([image.id, srcParams, cam.tanH, cam.tanV]);
  let det = null;
  if (srcKey !== keys.src) {
    setStatus('光源を検出中…');
    det = detectSources(image, srcParams, cam.tanH, cam.tanV);
    lastSources = det.sources; sourceInfo = { total: det.totalBeforeCap, det };
    keys.src = srcKey;
    drawOverlay();
  } else det = sourceInfo.det;
  if (cancelled()) return;

  // 2. pairs
  const shw = lens.focal * cam.tanH, shh = lens.focal * cam.tanV;
  const pairKey = JSON.stringify([lens.file, lens.name, lens.stopScale, shw, shh]);
  if (pairKey !== pairsCache.key) {
    pairsCache = { key: pairKey, pairs: filterGhostPairs(lens, shw, shh) };
    buildPairList();
  }
  const active = pairsCache.pairs.filter(p => pairEnabled.get(p.a + '-' + p.b) !== false);

  // 3. ghosts
  const ghostKey = JSON.stringify([srcKey, pairKey, active.map(p => p.a * 1000 + p.b), e.rayGrid, e.spectral, state.pupilJitter, state.jitterSeed, state.blades, state.rotation, state.ghostBlur, state.ghostBlurPasses]);
  let ghostStats = null;
  if (ghostKey !== keys.ghost) {
    const grid = buildPupil(e.rayGrid, state.blades, state.rotation, state.pupilJitter, state.jitterSeed);
    const spec = buildSpectrum(e.spectral);
    const diag = Math.hypot(w, h);
    setStatus(`ゴースト追跡中… ${active.length} pairs × ${lastSources.length} sources × ${grid.length / 2} rays × ${spec.length / 4} λ`);
    setProgress(0);
    ghostStats = await gpu.renderGhosts({
      lens, pairs: active, sources: lastSources, grid, spec, shw, shh,
      blurRadius: state.ghostBlur > 0 ? Math.max(1, Math.round(state.ghostBlur * diag)) : 0, blurPasses: state.ghostBlurPasses,
    }, { onProgress: setProgress, isCancelled: cancelled });
    setProgress(null);
    if (ghostStats.cancelled || cancelled()) return;
    keys.ghost = ghostKey;
  }

  // 4. haze
  if (state.hazeGain > 0) {
    const hazeKey = JSON.stringify([srcKey, state.hazeRadius, state.hazePasses]);
    if (hazeKey !== keys.haze) {
      setStatus('ヘイズを計算中…');
      const radius = state.hazeRadius > 0 ? Math.max(1, Math.round(state.hazeRadius * Math.hypot(w, h))) : 0;
      await gpu.renderHaze(det.hazeGrid, det.dw, det.dh, det.ds, radius, state.hazePasses);
      keys.haze = hazeKey;
    }
  }
  if (cancelled()) return;

  // 5. starburst
  if (state.starGain > 0) {
    const starKey = JSON.stringify([srcKey, state.blades, state.rotation, state.starScale]);
    if (starKey !== keys.star) {
      setStatus('スターバーストを計算中…');
      const pk = state.blades + ':' + state.rotation;
      if (psfCache.key !== pk) psfCache = { key: pk, psf: computeStarburstPSF(state.blades, state.rotation, 512) };
      await gpu.renderStarburst(psfCache.psf, 512, lastSources, state.starScale);
      keys.star = starKey;
    }
  }
  if (cancelled()) return;

  draw();
  drawDiagram();
  const ms = performance.now() - t0;
  const capped = sourceInfo.total > lastSources.length ? ` (cap ${sourceInfo.total}→${lastSources.length})` : '';
  const rays = ghostStats ? ` · ${(ghostStats.rays / 1e6).toFixed(1)} M traces` : '';
  setStatus(`${w}×${h} · ${lastSources.length} sources${capped} · ${active.length}/${pairsCache.pairs.length} pairs${rays} · ${ms.toFixed(0)} ms${state.preview ? ' · PREVIEW' : ''}`);
}

// ---------------------------------------------------------------------------
// Ghost pair list
// ---------------------------------------------------------------------------
function buildPairList() {
  const list = $('pairList');
  const pairs = pairsCache.pairs.slice().sort((x, y) => y.est * y.boost - x.est * x.boost);
  list.innerHTML = pairs.map(p => {
    const id = p.a + '-' + p.b, on = pairEnabled.get(id) !== false;
    return `<label class="pair${selectedPair && selectedPair.a === p.a && selectedPair.b === p.b ? ' sel' : ''}" data-id="${id}"><input type="checkbox" ${on ? 'checked' : ''}><span>(${p.a}, ${p.b})</span><small>${p.est.toExponential(1)} ×${p.boost.toFixed(1)}</small></label>`;
  }).join('');
  updatePairCount();
}
function updatePairCount() {
  const n = pairsCache.pairs.filter(p => pairEnabled.get(p.a + '-' + p.b) !== false).length;
  $('pairCount').textContent = `${n}/${pairsCache.pairs.length}`;
}
$('pairList').addEventListener('change', ev => {
  const lab = ev.target.closest('.pair');
  pairEnabled.set(lab.dataset.id, ev.target.checked);
  updatePairCount();
  scheduleRender(150);
});
$('pairList').addEventListener('click', ev => {
  const lab = ev.target.closest('.pair');
  if (!lab || ev.target.tagName === 'INPUT') return;
  ev.preventDefault();
  const [a, b] = lab.dataset.id.split('-').map(Number);
  selectedPair = selectedPair && selectedPair.a === a && selectedPair.b === b ? null : { a, b };
  for (const el of document.querySelectorAll('.pair')) el.classList.toggle('sel', !!selectedPair && el.dataset.id === lab.dataset.id);
  drawDiagram();
});
function setAllPairs(fn) {
  for (const p of pairsCache.pairs) pairEnabled.set(p.a + '-' + p.b, fn(p));
  buildPairList();
  scheduleRender(0);
}
$('pairsAll').onclick = () => setAllPairs(() => true);
$('pairsNone').onclick = () => setAllPairs(() => false);
$('pairsTop').onclick = () => {
  const top = new Set(pairsCache.pairs.slice().sort((x, y) => y.est * y.boost - x.est * x.boost).slice(0, 20));
  setAllPairs(p => top.has(p));
};

// ---------------------------------------------------------------------------
// Viewer
// ---------------------------------------------------------------------------
const canvas = $('gpuCanvas'), overlay = $('overlay'), viewer = $('viewer');
let dpr = window.devicePixelRatio || 1;

function resizeCanvas() {
  dpr = window.devicePixelRatio || 1;
  const r = viewer.getBoundingClientRect();
  for (const c of [canvas, overlay]) {
    c.width = Math.max(1, Math.round(r.width * dpr));
    c.height = Math.max(1, Math.round(r.height * dpr));
  }
  if (autoFit) fitView(); else draw();
}
let autoFit = true;   // keep fitting on resize until the user pans or zooms
function fitView() {
  autoFit = true;
  if (!image) return;
  const z = Math.min(canvas.width / image.width, canvas.height / image.height) * 0.96;
  view.zoom = z;
  view.offX = (canvas.width - image.width * z) / 2;
  view.offY = (canvas.height - image.height * z) / 2;
  draw();
}
function zoomTo(z, cx = canvas.width / 2, cy = canvas.height / 2) {
  autoFit = false;
  const ix = (cx - view.offX) / view.zoom, iy = (cy - view.offY) / view.zoom;
  view.zoom = Math.min(Math.max(z, 0.02), 64);
  view.offX = cx - ix * view.zoom; view.offY = cy - iy * view.zoom;
  draw();
}

let drawQueued = false;
function draw() {
  if (drawQueued) return;
  drawQueued = true;
  requestAnimationFrame(() => {
    drawQueued = false;
    if (!gpu || !image) return;
    const e = eff();
    gpu.draw({
      offX: view.offX, offY: view.offY, zoom: view.zoom, exposure: 2 ** view.exposure,
      inputGain: 2 ** state.inputEV, ghostGain: state.flareGain * e.dsRatio * e.dsRatio,
      hazeGain: state.hazeGain, starGain: state.starGain, hdrScale: 1,
      mode: view.mode, tonemap: view.tonemap, hdr: view.hdr,
    });
    drawOverlay();
  });
}

function drawOverlay() {
  const g = overlay.getContext('2d');
  g.clearRect(0, 0, overlay.width, overlay.height);
  if (!image) return;
  const P2 = (x, y) => [view.offX + x * view.zoom, view.offY + y * view.zoom];
  if (view.mode === 5) {
    const r = Math.max(2 * dpr, eff().downsample * view.zoom / 2);
    g.strokeStyle = '#4fd1c5'; g.lineWidth = 1 * dpr;
    for (const s of lastSources) {
      const [x, y] = P2(s.px, s.py);
      g.strokeRect(x - r, y - r, 2 * r, 2 * r);
    }
  }
  if (state.manualOn) {
    const [x, y] = P2(state.manualX, state.manualY);
    g.strokeStyle = '#ffb547'; g.lineWidth = 1.5 * dpr;
    g.beginPath(); g.arc(x, y, 10 * dpr, 0, Math.PI * 2); g.stroke();
    g.beginPath(); g.moveTo(x - 16 * dpr, y); g.lineTo(x + 16 * dpr, y); g.moveTo(x, y - 16 * dpr); g.lineTo(x, y + 16 * dpr); g.stroke();
  }
}

let drag = null;
viewer.addEventListener('pointerdown', ev => {
  if (!image || ev.target.closest('.vbar')) return;
  viewer.setPointerCapture(ev.pointerId);
  const cx = ev.offsetX * dpr, cy = ev.offsetY * dpr;
  const lx = view.offX + state.manualX * view.zoom, ly = view.offY + state.manualY * view.zoom;
  const onLight = state.manualOn && ev.button === 0 && Math.hypot(cx - lx, cy - ly) < 22 * dpr;
  drag = { mode: onLight ? 'light' : 'pan', x: cx, y: cy, ox: view.offX, oy: view.offY };
});
viewer.addEventListener('pointermove', ev => {
  const cx = ev.offsetX * dpr, cy = ev.offsetY * dpr;
  probe(cx, cy);
  if (!drag) return;
  if (drag.mode === 'pan') {
    view.offX = drag.ox + cx - drag.x; view.offY = drag.oy + cy - drag.y;
    autoFit = false;
    draw();
  } else {
    moveLight(cx, cy, true);
  }
});
viewer.addEventListener('pointerup', ev => {
  if (drag?.mode === 'light') moveLight(ev.offsetX * dpr, ev.offsetY * dpr, false);
  drag = null;
});
viewer.addEventListener('dblclick', ev => {
  if (!image || ev.target.closest('.vbar')) return;
  if (!state.manualOn) { state.manualOn = true; widgets.manualOn.set(true); }
  moveLight(ev.offsetX * dpr, ev.offsetY * dpr, false);
});
viewer.addEventListener('wheel', ev => {
  ev.preventDefault();
  zoomTo(view.zoom * Math.exp(-ev.deltaY * 0.0015), ev.offsetX * dpr, ev.offsetY * dpr);
}, { passive: false });

function moveLight(cx, cy, dragging) {
  state.manualX = Math.round((cx - view.offX) / view.zoom);
  state.manualY = Math.round((cy - view.offY) / view.zoom);
  widgets.manualX.set(state.manualX); widgets.manualY.set(state.manualY);
  refreshEnables(); persist(); drawOverlay();
  scheduleRender(dragging ? 120 : 0);
}

function probe(cx, cy) {
  if (!image) return;
  const x = Math.floor((cx - view.offX) / view.zoom), y = Math.floor((cy - view.offY) / view.zoom);
  if (x < 0 || y < 0 || x >= image.width || y >= image.height) { $('probe').textContent = ''; return; }
  const i = (y * image.width + x) * 4, k = 2 ** state.inputEV, d = image.data;
  $('probe').textContent = `${x}, ${y}  in ${(d[i] * k).toFixed(3)} ${(d[i + 1] * k).toFixed(3)} ${(d[i + 2] * k).toFixed(3)}`;
}

// ---------------------------------------------------------------------------
// Image loading
// ---------------------------------------------------------------------------
let imageSeq = 0;
function setImage(img) {
  const max = gpu.maxPixels;
  while (img.width * img.height > max) img = halve(img);
  img.id = ++imageSeq;
  image = img;
  gpu.setImage(img);
  keys.src = keys.ghost = keys.haze = keys.star = '';
  $('dropzone').classList.add('hidden');
  fitView();
  scheduleRender(0);
}
function halve(img) {
  const w = img.width >> 1, h = img.height >> 1, d = new Float32Array(w * h * 4), s = img.data;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 4; c++) {
    const i = ((2 * y) * img.width + 2 * x) * 4 + c, rw = img.width * 4;
    d[(y * w + x) * 4 + c] = (s[i] + s[i + 4] + s[i + rw] + s[i + rw + 4]) / 4;
  }
  setStatus(`画像が GPU の上限を超えたため ${w}×${h} に縮小しました`);
  return { ...img, width: w, height: h, data: d };
}
async function loadFile(file) {
  try {
    setStatus(`${file.name} を読み込み中…`);
    const img = await loadImageFile(file, { sdrExpand: state.sdrExpand });
    img.file = file;
    setImage(img);
  } catch (e) {
    console.error(e);
    setStatus('読み込みに失敗しました: ' + e.message, true);
  }
}
$('fileImage').onchange = ev => ev.target.files[0] && loadFile(ev.target.files[0]);
$('btnTest').onclick = () => setImage(makeTestScene());
viewer.addEventListener('dragover', ev => { ev.preventDefault(); viewer.classList.add('drag'); });
viewer.addEventListener('dragleave', () => viewer.classList.remove('drag'));
viewer.addEventListener('drop', ev => {
  ev.preventDefault(); viewer.classList.remove('drag');
  const f = ev.dataTransfer.files[0];
  if (!f) return;
  if (/\.lens$/i.test(f.name)) loadLensFile(f); else loadFile(f);
});
async function loadLensFile(f) {
  try { selectLens(Lens.parse(await f.text(), f.name.replace(/\.\w+$/, ''))); }
  catch (e) { setStatus('.lens の読み込みに失敗: ' + e.message, true); }
}
$('fileLens').onchange = ev => ev.target.files[0] && loadLensFile(ev.target.files[0]);
$('btnLensDl').onclick = () => lens && download(new Blob([lens.toText()], { type: 'text/plain' }), (lens.file || lens.name.replace(/[^\w.-]+/g, '_') + '.lens'));
$('lensSearch').oninput = ev => fillLensList(ev.target.value);
$('lensList').onchange = ev => selectLens(Lens.fromCatalog(catalog[Number(ev.target.value)]));

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------
async function composeLinear() {
  const n = image.width * image.height, e = eff();
  const gg = state.flareGain * e.dsRatio * e.dsRatio, hg = state.hazeGain, sg = state.starGain, ig = 2 ** state.inputEV;
  const [ghost, haze, star] = await Promise.all([gpu.read('ghost'), gpu.read('haze'), gpu.read('star')]);
  const out = new Float32Array(n * 4), inp = image.data, m = view.mode;
  for (let i = 0; i < n * 4; i += 4) {
    for (let c = 0; c < 3; c++) {
      const G = ghost[i + c] * gg, H = haze[i + c] * hg, Sb = star[i + c] * sg, I = inp[i + c] * ig;
      out[i + c] = m === 0 ? I + G + H + Sb : m === 1 ? G + H + Sb : m === 2 ? G : m === 3 ? H : m === 4 ? Sb : I;
    }
    out[i + 3] = m === 0 || m === 5 ? inp[i + 3] : 1;
  }
  return { out, ghost, haze, star, gg, hg, sg };
}

$('btnExr').onclick = async () => {
  if (!image) return;
  try {
    setStatus('EXR を書き出し中…'); setProgress(0);
    const { out, ghost, haze, star, gg, hg, sg } = await composeLinear();
    const n = image.width * image.height;
    const chans = ['R', 'G', 'B', 'A'].map((name, c) => ({ name, data: out, stride: 4, offset: c }));
    if (state.exrLayers) {
      const layer = (pre, buf, k) => {
        const d = new Float32Array(n * 4);
        for (let i = 0; i < d.length; i++) d[i] = buf[i] * k;
        return ['red', 'green', 'blue'].map((nm, c) => ({ name: `${pre}.${nm}`, data: d, stride: 4, offset: c }));
      };
      chans.push(...layer('flare', ghost, gg));
      if (hg > 0) chans.push(...layer('haze', haze, hg));
      if (sg > 0) chans.push(...layer('starburst', star, sg));
    }
    const bytes = await writeEXR({
      width: image.width, height: image.height, channels: chans,
      half: state.exrHalf, compression: state.exrZip ? 'zip' : 'none', onProgress: setProgress,
    });
    setProgress(null);
    download(new Blob([bytes], { type: 'image/x-exr' }), baseName() + '_flare.exr');
    setStatus(`EXR を書き出しました (${(bytes.length / 1048576).toFixed(1)} MB, ${chans.length} ch)`);
  } catch (e) { console.error(e); setProgress(null); setStatus('EXR 書き出しに失敗: ' + e.message, true); }
};

$('btnPng').onclick = async () => {
  if (!image) return;
  const { out } = await composeLinear();
  const w = image.width, h = image.height, px = new Uint8ClampedArray(w * h * 4), k = 2 ** view.exposure;
  const tm = v => { v = Math.max(v * k, 0); if (view.tonemap === 1) return v / (1 + v); if (view.tonemap === 2) { v *= 0.6; return Math.min(1, (v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14)); } return Math.min(v, 1); };
  const enc = v => (v <= 0.0031308 ? v * 12.92 : 1.055 * v ** (1 / 2.4) - 0.055) * 255;
  for (let i = 0; i < px.length; i += 4) { px[i] = enc(tm(out[i])); px[i + 1] = enc(tm(out[i + 1])); px[i + 2] = enc(tm(out[i + 2])); px[i + 3] = 255; }
  const cv = new OffscreenCanvas(w, h);
  cv.getContext('2d').putImageData(new ImageData(px, w, h), 0, 0);
  download(await cv.convertToBlob({ type: 'image/png' }), baseName() + '_flare.png');
};
const baseName = () => (image.name || 'image').replace(/\.\w+$/, '');
function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

// ---------------------------------------------------------------------------
// Status / view controls
// ---------------------------------------------------------------------------
function setStatus(t, err = false) { const s = $('status'); s.textContent = t; s.classList.toggle('err', err); }
function setProgress(p) {
  $('progress').style.display = p == null ? 'none' : 'block';
  if (p != null) $('progressBar').style.width = (p * 100).toFixed(1) + '%';
}
function initViewControls() {
  const ex = $('exposure'), exo = $('exposureOut');
  ex.value = view.exposure; exo.textContent = Number(view.exposure).toFixed(1);
  ex.oninput = () => { view.exposure = Number(ex.value); exo.textContent = view.exposure.toFixed(1); persist(); draw(); };
  ex.ondblclick = () => { ex.value = 0; ex.oninput(); };
  $('outputMode').value = view.mode;
  $('outputMode').onchange = ev => { view.mode = Number(ev.target.value); persist(); draw(); };
  $('tonemap').value = view.tonemap;
  $('tonemap').onchange = ev => { view.tonemap = Number(ev.target.value); persist(); draw(); };
  const mq = matchMedia('(dynamic-range: high)');
  const hdrCb = $('hdrOn');
  const updHdr = () => {
    $('hdrStat').textContent = mq.matches ? 'HDR display' : 'SDR display';
    $('hdrStat').classList.toggle('on', mq.matches && view.hdr);
    $('tonemap').disabled = view.hdr;
  };
  if (localStorage.getItem(STORE) === null) view.hdr = mq.matches;
  hdrCb.checked = view.hdr;
  hdrCb.onchange = () => { view.hdr = hdrCb.checked; updHdr(); persist(); draw(); };
  mq.addEventListener('change', updHdr);
  updHdr();
  $('btnFit').onclick = fitView;
  $('btn100').onclick = () => zoomTo(1);
  window.addEventListener('keydown', ev => {
    if (ev.target.matches('input, select, textarea')) return;
    if (ev.key === 'f') fitView();
    if (ev.key === 'p') { setParam('preview', !state.preview); widgets.preview.set(state.preview); }
    if (ev.key >= '1' && ev.key <= '6') { view.mode = Number(ev.key) - 1; $('outputMode').value = view.mode; draw(); }
  });
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
async function boot() {
  buildWidgets();
  initViewControls();
  new ResizeObserver(resizeCanvas).observe(viewer);
  try {
    gpu = await FlareGPU.create(canvas);
    gpu.lost.then(info => setStatus('GPU デバイスが失われました: ' + info.message + '（ページを再読み込みしてください）', true));
  } catch (e) {
    setStatus('WebGPU を初期化できません: ' + e.message + ' — Chrome / Edge 113 以降が必要です。', true);
    return;
  }
  resizeCanvas();
  window.flaresim = { state, view, get gpu() { return gpu; }, get image() { return image; }, get sources() { return lastSources; } };
  await loadCatalog();
  setImage(makeTestScene());
}
boot();
