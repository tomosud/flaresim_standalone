// ============================================================================
// gpu.js — WebGPU renderer: ghost trace, blur, haze, starburst and viewer.
// ============================================================================
import { ghostWGSL, resolveWGSL, blurWGSL, hazeSeedWGSL, starburstWGSL, displayWGSL } from './shaders.js';

const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
const UNI = GPUBufferUsage.UNIFORM | CD;

export class FlareGPU {
  static async create(canvas) {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter found.');
    const L = adapter.limits;
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
        maxBufferSize: L.maxBufferSize,
        maxStorageBuffersPerShaderStage: Math.min(8, L.maxStorageBuffersPerShaderStage),
      },
    });
    const g = new FlareGPU();
    g.adapter = adapter;
    g.device = device;
    g.canvas = canvas;
    g.ctx = canvas.getContext('webgpu');
    g.lost = device.lost;
    g.init();
    return g;
  }

  init() {
    const d = this.device;
    const comp = (code, label) => d.createComputePipeline({
      label, layout: 'auto', compute: { module: d.createShaderModule({ code, label }), entryPoint: 'main' },
    });
    this.pGhost = comp(ghostWGSL, 'ghost');
    this.pResolve = comp(resolveWGSL, 'resolve');
    this.pBlur = comp(blurWGSL, 'blur');
    this.pHaze = comp(hazeSeedWGSL, 'hazeSeed');
    this.pStar = comp(starburstWGSL, 'starburst');
    const dm = d.createShaderModule({ code: displayWGSL, label: 'display' });
    this.pDisplay = d.createRenderPipeline({
      layout: 'auto',
      vertex: { module: dm, entryPoint: 'vs' },
      fragment: { module: dm, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list' },
    });
    this.uDisplay = d.createBuffer({ size: 64, usage: UNI });
    this.hdr = null;
    this.configure(false);
  }

  configure(hdr) {
    if (this.hdr === hdr) return;
    this.hdr = hdr;
    this.ctx.configure({
      device: this.device, format: 'rgba16float', alphaMode: 'opaque',
      colorSpace: 'srgb', toneMapping: { mode: hdr ? 'extended' : 'standard' },
    });
  }

  get maxPixels() {
    return Math.floor(Math.min(this.device.limits.maxStorageBufferBindingSize, this.device.limits.maxBufferSize) / 24);
  }

  buf(size, usage, label) { return this.device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage, label }); }

  setImage(img) {
    for (const b of [this.bInput, this.bAcc, this.bGhost, this.bHaze, this.bStar, this.bTmp]) b?.destroy();
    const n = img.width * img.height;
    this.w = img.width; this.h = img.height; this.n = n;
    this.bInput = this.buf(n * 16, S | CD, 'input');
    this.bAcc = this.buf(n * 24, S | CD, 'ghostAcc');
    this.bGhost = this.buf(n * 16, S | CD | CS, 'ghost');
    this.bHaze = this.buf(n * 16, S | CD | CS, 'haze');
    this.bStar = this.buf(n * 16, S | CD | CS, 'starburst');
    this.bTmp = this.buf(n * 16, S | CD, 'tmp');
    this.device.queue.writeBuffer(this.bInput, 0, img.data);
    const enc = this.device.createCommandEncoder();
    for (const b of [this.bGhost, this.bHaze, this.bStar]) enc.clearBuffer(b);
    this.device.queue.submit([enc.finish()]);
    this.displayBG = this.device.createBindGroup({
      layout: this.pDisplay.getBindGroupLayout(0),
      entries: [this.uDisplay, this.bInput, this.bGhost, this.bHaze, this.bStar].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
  }

  storage(data, label) {
    const b = this.buf(data.byteLength, S | CD, label);
    this.device.queue.writeBuffer(b, 0, data);
    return b;
  }

  uniform(data) {
    const b = this.buf(data.byteLength, UNI);
    this.device.queue.writeBuffer(b, 0, data);
    return b;
  }

  bind(pipeline, buffers) {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
  }

  // In-place separable box blur of `target` (vec4 per pixel), `passes` times.
  encodeBlur(enc, target, radius, passes) {
    if (radius < 1 || passes < 1) return;
    const { w, h } = this;
    const uH = this.uniform(new Uint32Array([w, h, w, 1, radius, 0, 0, 0]));
    const uV = this.uniform(new Uint32Array([h, w, 1, w, radius, 0, 0, 0]));
    const bgH = this.bind(this.pBlur, [uH, target, this.bTmp]);
    const bgV = this.bind(this.pBlur, [uV, this.bTmp, target]);
    for (let p = 0; p < passes; p++) {
      let pass = enc.beginComputePass();
      pass.setPipeline(this.pBlur); pass.setBindGroup(0, bgH); pass.dispatchWorkgroups(Math.ceil(h / 64)); pass.end();
      pass = enc.beginComputePass();
      pass.setPipeline(this.pBlur); pass.setBindGroup(0, bgV); pass.dispatchWorkgroups(Math.ceil(w / 64)); pass.end();
    }
  }

  /**
   * Trace all (pair, source, pupil sample, wavelength) ghost rays.
   * job: {lens, pairs:[{a,b,boost,est}], sources, grid:Float32Array(u,v..), spec:Float32Array, shw, shh,
   *       blurRadius, blurPasses}
   * Resolves the ghost buffer at gain 1 (the viewer applies Flare Gain).
   */
  async renderGhosts(job, { onProgress, isCancelled } = {}) {
    const d = this.device, q = d.queue;
    const { lens, pairs, sources, grid, spec } = job;
    const nGrid = grid.length / 2, nSpec = spec.length / 4, nSrc = sources.length, nPairs = pairs.length;
    const enc0 = d.createCommandEncoder();
    enc0.clearBuffer(this.bAcc);
    enc0.clearBuffer(this.bGhost);
    q.submit([enc0.finish()]);
    if (!nGrid || !nSrc || !nPairs) return { rays: 0 };

    // Choose a power-of-two fixed-point scale so the worst case (every ray of
    // every pair and source landing on one pixel) cannot overflow 2^32.
    let srcSum = 0;
    for (const s of sources) srcSum += Math.max(s.r, s.g, s.b);
    let pairSum = 0;
    for (const p of pairs) pairSum += p.boost * Math.min(1, 1000 * p.est);
    const bound = Math.max(srcSum * pairSum, 1e-30);
    const fixedScale = 2 ** Math.max(-60, Math.min(60, Math.floor(Math.log2(2e9 / bound))));

    const surfs = this.storage(lens.packSurfaces(), 'surfs');
    const pairArr = new Float32Array(nPairs * 4);
    pairs.forEach((p, i) => pairArr.set([p.a, p.b, p.boost, 0], i * 4));
    const srcArr = new Float32Array(nSrc * 8);
    sources.forEach((s, i) => srcArr.set([s.ax, s.ay, 0, 0, s.r, s.g, s.b, 0], i * 8));
    const bPairs = this.storage(pairArr, 'pairs'), bSrc = this.storage(srcArr, 'sources');
    const bGrid = this.storage(grid, 'grid'), bSpec = this.storage(spec, 'spec');
    const uBuf = this.buf(64, UNI, 'ghostU');
    const bg = this.bind(this.pGhost, [uBuf, surfs, bPairs, bSrc, bGrid, bSpec, this.bAcc]);

    const uData = new ArrayBuffer(64), uU = new Uint32Array(uData), uF = new Float32Array(uData);
    uU[0] = lens.n; uU[1] = nGrid; uU[2] = nSpec; uU[3] = nSrc;
    uU[6] = this.w; uU[7] = this.h;
    uF[8] = lens.sensorZ; uF[9] = lens.surfaces[0].semi; uF[10] = lens.surfaces[0].z - 20;
    uF[11] = job.shw; uF[12] = job.shh; uF[13] = 1 / nGrid; uF[14] = fixedScale;

    const totalCombos = nPairs * nSrc;
    const perBatch = Math.max(1, Math.min(65535, Math.floor(3e6 / (nGrid * nSpec))));
    const wgx = Math.ceil(nGrid / 64);
    let done = 0, sinceSync = 0;
    const t0 = performance.now();
    while (done < totalCombos) {
      if (isCancelled?.()) break;
      const count = Math.min(perBatch, totalCombos - done);
      uU[4] = done; uU[5] = count;
      q.writeBuffer(uBuf, 0, uData);
      const enc = d.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pGhost); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(wgx, count); pass.end();
      q.submit([enc.finish()]);
      done += count;
      if (++sinceSync >= 2 || done >= totalCombos) {
        sinceSync = 0;
        await q.onSubmittedWorkDone();
        onProgress?.(done / totalCombos);
      }
    }
    const cancelled = done < totalCombos;
    if (!cancelled) {
      const enc = d.createCommandEncoder();
      const ur = new ArrayBuffer(16);
      new Uint32Array(ur, 0, 1)[0] = this.n;
      new Float32Array(ur, 8, 1)[0] = 1 / fixedScale;
      const uR = this.uniform(new Uint8Array(ur));
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pResolve); pass.setBindGroup(0, this.bind(this.pResolve, [uR, this.bAcc, this.bGhost]));
      pass.dispatchWorkgroups(Math.ceil(this.n / 256)); pass.end();
      this.encodeBlur(enc, this.bGhost, job.blurRadius, job.blurPasses);
      q.submit([enc.finish()]);
      await q.onSubmittedWorkDone();
    }
    for (const b of [surfs, bPairs, bSrc, bGrid, bSpec, uBuf]) b.destroy();
    return { cancelled, rays: done * nGrid * nSpec, ms: performance.now() - t0 };
  }

  // Haze seed from the source-block grid, then a wide blur (gain applied by viewer).
  async renderHaze(grid, dw, dh, ds, radius, passes) {
    const d = this.device, enc = d.createCommandEncoder();
    const bBlocks = this.storage(grid, 'hazeBlocks');
    const u = this.uniform(new Uint32Array([this.w, this.h, dw, dh, ds, 0, 0, 0]));
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pHaze); pass.setBindGroup(0, this.bind(this.pHaze, [u, bBlocks, this.bHaze]));
    pass.dispatchWorkgroups(Math.ceil(this.w / 16), Math.ceil(this.h / 16)); pass.end();
    this.encodeBlur(enc, this.bHaze, radius, passes);
    d.queue.submit([enc.finish()]);
    await d.queue.onSubmittedWorkDone();
    bBlocks.destroy();
  }

  async renderStarburst(psf, N, sources, scale) {
    const d = this.device, enc = d.createCommandEncoder();
    if (!sources.length) { enc.clearBuffer(this.bStar); d.queue.submit([enc.finish()]); return; }
    const bPsf = this.storage(psf, 'psf');
    const arr = new Float32Array(sources.length * 8);
    sources.forEach((s, i) => arr.set([s.px, s.py, 0, 0, s.r, s.g, s.b, 0], i * 8));
    const bSrc = this.storage(arr, 'sbSources');
    const ub = new ArrayBuffer(32);
    new Uint32Array(ub, 0, 4).set([this.w, this.h, N, sources.length]);
    new Float32Array(ub, 16, 4).set([scale * Math.hypot(this.w, this.h), 0, 0, 0]);
    const u = this.uniform(new Uint8Array(ub));
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pStar); pass.setBindGroup(0, this.bind(this.pStar, [u, bPsf, bSrc, this.bStar]));
    pass.dispatchWorkgroups(Math.ceil(this.w / 16), Math.ceil(this.h / 16)); pass.end();
    d.queue.submit([enc.finish()]);
    await d.queue.onSubmittedWorkDone();
    bPsf.destroy(); bSrc.destroy();
  }

  clear(which) {
    const enc = this.device.createCommandEncoder();
    enc.clearBuffer({ ghost: this.bGhost, haze: this.bHaze, star: this.bStar }[which]);
    this.device.queue.submit([enc.finish()]);
  }

  // v: {offX, offY, zoom, exposure, inputGain, ghostGain, hazeGain, starGain, hdrScale, mode, tonemap, hdr}
  draw(v) {
    if (!this.displayBG) return;
    this.configure(!!v.hdr);
    const b = new ArrayBuffer(64), f = new Float32Array(b), u = new Uint32Array(b);
    f.set([this.w, this.h, v.offX, v.offY, v.zoom, v.exposure, v.inputGain, v.ghostGain, v.hazeGain, v.starGain, v.hdrScale, 0]);
    u[12] = v.mode; u[13] = v.tonemap; u[14] = v.hdr ? 1 : 0;
    this.device.queue.writeBuffer(this.uDisplay, 0, b);
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    });
    pass.setPipeline(this.pDisplay); pass.setBindGroup(0, this.displayBG); pass.draw(3); pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  async read(which) {
    const src = { ghost: this.bGhost, haze: this.bHaze, star: this.bStar }[which];
    const rb = this.device.createBuffer({ size: src.size, usage: GPUBufferUsage.MAP_READ | CD });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, rb, 0, src.size);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0, this.n * 16));
    rb.unmap(); rb.destroy();
    return out;
  }
}
