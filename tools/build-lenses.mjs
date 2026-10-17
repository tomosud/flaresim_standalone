// Build lenses/lenses.json from a directory of FlareSim .lens prescriptions.
//
// Usage: node tools/build-lenses.mjs <lens_files_dir> [out.json]
//
// The .lens files come from LocalStarlight/flaresim_nuke (MIT). The parser
// mirrors LensSystem::load() in flaresim_nuke/src/lens.cpp.
import fs from 'node:fs';
import path from 'node:path';

export function parseLens(text) {
  let name = '', focal = 0, inSurf = false;
  const surfaces = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line[0] === '#') continue;
    if (!inSurf) {
      if (line.startsWith('name:')) name = line.slice(5).trim();
      else if (line.startsWith('focal_length:')) focal = parseFloat(line.slice(13));
      else if (line.startsWith('surfaces:')) inSurf = true;
      continue;
    }
    const tok = line.split(/\s+/);
    let radius = 0, stop = 0;
    if (tok[0] === 'stop' || tok[0] === 'STOP') stop = 1;
    else if (tok[0] !== 'inf' && tok[0] !== 'INF') radius = parseFloat(tok[0]);
    const [thick, ior, abbe, semi, coat] = tok.slice(1, 6).map(Number);
    if (!(semi > 0)) continue;
    surfaces.push([radius, thick, ior, abbe, semi, coat | 0, stop]);
  }
  return { name, focal, surfaces };
}

const dir = process.argv[2];
const out = process.argv[3] || 'lenses/lenses.json';
if (!dir) {
  console.error('usage: node tools/build-lenses.mjs <lens_files_dir> [out.json]');
  process.exit(1);
}

const files = fs.readdirSync(dir).filter(f => f.endsWith('.lens')).sort((a, b) => a.localeCompare(b));
const lenses = [];
for (const f of files) {
  const l = parseLens(fs.readFileSync(path.join(dir, f), 'utf8'));
  if (!l.surfaces.length) { console.warn('skip (no surfaces):', f); continue; }
  lenses.push({ n: l.name || f.replace(/\.lens$/, ''), file: f, f: l.focal, s: l.surfaces });
}
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, JSON.stringify(lenses));
console.log(`wrote ${lenses.length} lenses -> ${out} (${(fs.statSync(out).size / 1024).toFixed(0)} KB)`);
