// ============================================================================
// starburst.js — Fraunhofer diffraction PSF = |FFT(aperture)|^2
// (port of flaresim_nuke/src/starburst.cpp compute_starburst_psf)
// ============================================================================
import { insideAperture } from './optics.js';

function fft1d(re, im, n, off, stride) {
  // bit reversal
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const a = off + i * stride, b = off + j * stride;
      let t = re[a]; re[a] = re[b]; re[b] = t;
      t = im[a]; im[a] = im[b]; im[b] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang), half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let j = 0; j < half; j++) {
        const a = off + (i + j) * stride, b = off + (i + j + half) * stride;
        const vr = re[b] * cr - im[b] * ci, vi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - vr; im[b] = im[a] - vi;
        re[a] += vr; im[a] += vi;
        const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}

// Returns Float32Array(N*N), DC at (N/2, N/2), peak normalised to 1.
export function computeStarburstPSF(blades, rotDeg, N = 512) {
  const re = new Float64Array(N * N), im = new Float64Array(N * N);
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
    const u = ((c + 0.5) / N) * 2 - 1, v = ((r + 0.5) / N) * 2 - 1;
    if (insideAperture(u, v, blades, rotDeg)) re[r * N + c] = 1;
  }
  for (let r = 0; r < N; r++) fft1d(re, im, N, r * N, 1);
  for (let c = 0; c < N; c++) fft1d(re, im, N, c, N);
  const psf = new Float32Array(N * N);
  let peak = 0;
  const h = N >> 1;
  for (let r = 0; r < N; r++) {
    const rs = (r + h) % N;
    for (let c = 0; c < N; c++) {
      const k = rs * N + (c + h) % N;
      const v = re[k] * re[k] + im[k] * im[k];
      psf[r * N + c] = v;
      if (v > peak) peak = v;
    }
  }
  if (peak > 0) for (let i = 0; i < psf.length; i++) psf[i] /= peak;
  return psf;
}
