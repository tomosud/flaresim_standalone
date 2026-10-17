// ============================================================================
// exr.js — minimal OpenEXR scanline writer (NONE / ZIP, HALF / FLOAT).
//
// Writes single-part scanline images with any number of named channels, so
// layered output such as R,G,B,A + flare.R/G/B + haze.R/G/B is possible.
// ZIP blocks (16 lines) use the OpenEXR byte-split + delta predictor and
// zlib from the browser's CompressionStream('deflate').
// ============================================================================

const PIXEL_HALF = 1, PIXEL_FLOAT = 2;
const COMP_NONE = 0, COMP_ZIP = 3;

// float32 -> float16 bits (round to nearest even)
const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
export function toHalf(v) {
  f32[0] = v;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  let e = (x >>> 23) & 0xff, m = x & 0x7fffff;
  if (e === 0xff) return sign | 0x7c00 | (m ? 0x200 : 0);       // inf / nan
  e = e - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;                          // overflow -> inf
  if (e <= 0) {
    if (e < -10) return sign;
    m |= 0x800000;
    const shift = 14 - e;
    let hm = m >>> shift;
    const rem = m & ((1 << shift) - 1), halfway = 1 << (shift - 1);
    if (rem > halfway || (rem === halfway && (hm & 1))) hm++;
    return sign | hm;
  }
  let h = sign | (e << 10) | (m >>> 13);
  const rem = m & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (h & 1))) h++;
  return h;
}

class ByteWriter {
  constructor() { this.parts = []; this.len = 0; }
  bytes(u8) { this.parts.push(u8); this.len += u8.length; }
  u8(v) { this.bytes(new Uint8Array([v])); }
  i32(v) { const b = new Uint8Array(4); new DataView(b.buffer).setInt32(0, v, true); this.bytes(b); }
  u32(v) { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, v, true); this.bytes(b); }
  f32(v) { const b = new Uint8Array(4); new DataView(b.buffer).setFloat32(0, v, true); this.bytes(b); }
  u64(v) { const b = new Uint8Array(8); const d = new DataView(b.buffer); d.setUint32(0, v % 4294967296, true); d.setUint32(4, Math.floor(v / 4294967296), true); this.bytes(b); }
  str0(s) { this.bytes(new TextEncoder().encode(s)); this.u8(0); }
  attr(name, type, payload) { this.str0(name); this.str0(type); this.i32(payload.length); this.bytes(payload); }
  concat() {
    const out = new Uint8Array(this.len);
    let o = 0;
    for (const p of this.parts) { out.set(p, o); o += p.length; }
    return out;
  }
}

function payload(fn) { const w = new ByteWriter(); fn(w); return w.concat(); }

async function zlib(u8) {
  const cs = new CompressionStream('deflate');
  const stream = new Blob([u8]).stream().pipeThrough(cs);
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function zipPredict(raw) {
  const n = raw.length, tmp = new Uint8Array(n);
  let t1 = 0, t2 = (n + 1) >> 1;
  for (let i = 0; i < n; i += 2) {
    tmp[t1++] = raw[i];
    if (i + 1 < n) tmp[t2++] = raw[i + 1];
  }
  let p = tmp[0];
  for (let i = 1; i < n; i++) {
    const d = (tmp[i] - p + 128 + 256) & 0xff;
    p = tmp[i];
    tmp[i] = d;
  }
  return tmp;
}

/**
 * @param {object} o
 * @param {number} o.width
 * @param {number} o.height
 * @param {{name:string, data:Float32Array, stride?:number, offset?:number}[]} o.channels
 *        data is read as data[(y*width+x)*stride + offset] (top-down rows)
 * @param {boolean} [o.half=true]
 * @param {'zip'|'none'} [o.compression='zip']
 * @returns {Promise<Uint8Array>}
 */
export async function writeEXR({ width, height, channels, half = true, compression = 'zip', onProgress }) {
  const chans = channels.slice().sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  const ptype = half ? PIXEL_HALF : PIXEL_FLOAT;
  const bpp = half ? 2 : 4;
  const comp = compression === 'zip' ? COMP_ZIP : COMP_NONE;
  const linesPerBlock = comp === COMP_ZIP ? 16 : 1;

  const hdr = new ByteWriter();
  hdr.u32(20000630);          // magic
  hdr.u32(2);                 // version 2, single-part scanline
  hdr.attr('channels', 'chlist', payload(w => {
    for (const c of chans) {
      w.str0(c.name); w.i32(ptype); w.u8(0); w.u8(0); w.u8(0); w.u8(0); w.i32(1); w.i32(1);
    }
    w.u8(0);
  }));
  hdr.attr('compression', 'compression', new Uint8Array([comp]));
  const box = payload(w => { w.i32(0); w.i32(0); w.i32(width - 1); w.i32(height - 1); });
  hdr.attr('dataWindow', 'box2i', box);
  hdr.attr('displayWindow', 'box2i', box);
  hdr.attr('lineOrder', 'lineOrder', new Uint8Array([0]));
  hdr.attr('pixelAspectRatio', 'float', payload(w => w.f32(1)));
  hdr.attr('screenWindowCenter', 'v2f', payload(w => { w.f32(0); w.f32(0); }));
  hdr.attr('screenWindowWidth', 'float', payload(w => w.f32(1)));
  hdr.u8(0);                  // end of header
  const header = hdr.concat();

  const nBlocks = Math.ceil(height / linesPerBlock);
  const blocks = [];
  for (let bi = 0; bi < nBlocks; bi++) {
    const y0 = bi * linesPerBlock, y1 = Math.min(height, y0 + linesPerBlock);
    const raw = new Uint8Array((y1 - y0) * width * chans.length * bpp);
    const dv = new DataView(raw.buffer);
    let o = 0;
    for (let y = y0; y < y1; y++) {
      for (const c of chans) {
        const stride = c.stride ?? 1, off = c.offset ?? 0, d = c.data;
        let i = y * width * stride + off;
        if (half) for (let x = 0; x < width; x++, i += stride, o += 2) dv.setUint16(o, toHalf(d[i]), true);
        else for (let x = 0; x < width; x++, i += stride, o += 4) dv.setFloat32(o, d[i], true);
      }
    }
    let data = raw;
    if (comp === COMP_ZIP) {
      const z = await zlib(zipPredict(raw));
      if (z.length < raw.length) data = z;   // decoders treat size==raw as uncompressed
    }
    blocks.push({ y: y0, data });
    if (onProgress && bi % 8 === 0) onProgress(bi / nBlocks);
  }

  const out = new ByteWriter();
  out.bytes(header);
  let offset = header.length + nBlocks * 8;
  for (const b of blocks) { out.u64(offset); offset += 8 + b.data.length; }
  for (const b of blocks) { out.i32(b.y); out.i32(b.data.length); out.bytes(b.data); }
  return out.concat();
}
