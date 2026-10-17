// Parity check of src/optics.js against the original flaresim_nuke C++ code.
//
// 1. Build the reference printer (tests/ref.cpp) against flaresim_nuke/src
//    and run it on a .lens file:  ref.exe lens.lens > ref.txt
// 2. node tests/parity.mjs lens.lens ref.txt
//
// Compares the filtered ghost-pair list, per-pair area boosts and the sensor
// landing position / Fresnel weight of 5 fixed rays per pair.
import fs from 'node:fs';
import { Lens, filterGhostPairs, traceGhost } from '../src/optics.js';

const [lensPath, refPath] = process.argv.slice(2);
if (!lensPath || !refPath) { console.error('usage: node tests/parity.mjs <lens> <ref.txt>'); process.exit(1); }
const lens = Lens.parse(fs.readFileSync(lensPath, 'utf8'));
const ref = fs.readFileSync(refPath, 'utf8').trim().split(/\r?\n/);

const tanH = Math.tan(40 * Math.PI / 360), tanV = tanH / (1920 / 1080);
const shw = lens.focal * tanH, shh = lens.focal * tanV;
const pairs = filterGhostPairs(lens, shw, shh);
const R0 = lens.surfaces[0].semi, z0 = lens.surfaces[0].z - 20;
const d = [Math.tan(0.12), Math.tan(-0.08), 1];
const L = Math.hypot(...d); d.forEach((v, i) => (d[i] = v / L));

let li = 0, maxPos = 0, maxW = 0, maxBoost = 0, mism = 0;
const refCount = Number(ref[li++]);
if (refCount !== pairs.length) { console.log(`pair count differs: ref ${refCount} js ${pairs.length}`); mism++; }
for (const p of pairs) {
  const [, a, b, boost] = ref[li++].split(' ');
  if (Number(a) !== p.a || Number(b) !== p.b) { console.log(`pair order differs at (${p.a},${p.b}) vs (${a},${b})`); mism++; }
  maxBoost = Math.max(maxBoost, Math.abs(Number(boost) - p.boost) / Math.max(1, p.boost));
  for (let k = 0; k < 5; k++) {
    const t = ref[li++].split(' ');
    const u = -0.8 + 0.4 * k, v = 0.3 - 0.15 * k;
    const r = traceGhost([u * R0, v * R0, z0], d, lens, p.a, p.b, 550);
    if ((t[2] === 'x') !== !r) { console.log(`validity differs pair (${p.a},${p.b}) ray ${k}`); mism++; continue; }
    if (!r) continue;
    maxPos = Math.max(maxPos, Math.abs(r.x - Number(t[2])), Math.abs(r.y - Number(t[3])));
    maxW = Math.max(maxW, Math.abs(r.weight - Number(t[4])) / Math.max(1e-12, Number(t[4])));
  }
}
console.log(`${lens.name}: ${pairs.length} pairs, max |dpos| ${maxPos.toExponential(2)} mm, max rel dweight ${maxW.toExponential(2)}, max rel dboost ${maxBoost.toExponential(2)}, mismatches ${mism}`);
process.exit(mism ? 1 : 0);
