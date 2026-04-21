# Real Spectral Dispersion Demo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a realtime WebGPU demo that renders Apple "Liquid Glass"-style 3D floating pills with physically accurate wavelength-dependent refraction over a random Picsum photo background.

**Architecture:** One fullscreen fragment pass. Pills are rendered via sphere-traced 3D SDFs (stadium from top, rounded slab from side). Refraction samples the background photo per wavelength, sums against the Wyman-Sloan-Shirley CIE XYZ approximation, and converts to sRGB. Tweakpane drives live parameter tuning; dragging pills updates a GPU uniform buffer each frame.

**Tech Stack:** WebGPU + WGSL, TypeScript (strict), Vite + bun, Tweakpane v4, Vitest (unit tests for math modules).

**Spec:** [docs/superpowers/specs/2026-04-21-spectral-dispersion-design.md](../specs/2026-04-21-spectral-dispersion-design.md)

**Commit policy:** User requires explicit approval before any `git commit`. This plan marks logical commit points with `[COMMIT POINT]`. At each one, the implementer must stop, show what changed, and ask the user before committing. Do not commit autonomously.

---

## File Structure

```
RealRefraction/
├── index.html                          # Canvas + no-WebGPU fallback
├── package.json
├── vite.config.ts                      # Includes ?raw import for *.wgsl
├── tsconfig.json
├── vitest.config.ts
├── src/
│   ├── main.ts                         # Entry: init, frame loop
│   ├── webgpu/
│   │   ├── device.ts                   # Adapter + device + canvas config
│   │   ├── pipeline.ts                 # Render pipeline + bind groups
│   │   ├── uniforms.ts                 # Typed uniform buffer writers
│   │   └── history.ts                  # Ping-pong history texture for TAA
│   ├── math/
│   │   ├── cauchy.ts                   # Cauchy IOR + Abbe number
│   │   ├── wyman.ts                    # CIE XYZ matching functions
│   │   ├── srgb.ts                     # XYZ → sRGB conversion
│   │   └── sdfPill.ts                  # 3D pill SDF (TS mirror of WGSL)
│   ├── pills.ts                        # Pill state + drag handling
│   ├── photo.ts                        # Picsum fetch + texture upload
│   ├── ui.ts                           # Tweakpane wiring
│   └── shaders/
│       ├── fullscreen.wgsl             # Vertex shader (triangle pair)
│       └── dispersion.wgsl             # Fragment shader with SDF + spectral loop
└── tests/
    ├── cauchy.test.ts
    ├── wyman.test.ts
    ├── srgb.test.ts
    └── sdfPill.test.ts
```

Each math module is a small, independently-tested unit. Shader code mirrors the TS math modules so behavior stays aligned.

---

## Task 1: Project scaffold

**Files:**
- Create: `package.json`, `tsconfig.json`, `vite.config.ts`, `vitest.config.ts`, `index.html`, `.gitignore`, `src/main.ts`

- [ ] **Step 1: Initialize project with bun**

```bash
cd /Users/hiko/Documents/repos/Personal/RealRefraction
bun init -y
```

- [ ] **Step 2: Overwrite `package.json`**

Replace the generated file with:

```json
{
  "name": "real-refraction",
  "version": "0.1.0",
  "type": "module",
  "private": true,
  "scripts": {
    "dev": "vite",
    "build": "tsc --noEmit && vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "devDependencies": {
    "@types/node": "^20.11.0",
    "@webgpu/types": "^0.1.49",
    "typescript": "^5.4.0",
    "vite": "^5.2.0",
    "vitest": "^1.5.0"
  },
  "dependencies": {
    "tweakpane": "^4.0.3"
  }
}
```

- [ ] **Step 3: Install**

```bash
bun install
```

- [ ] **Step 4: Write `tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "noImplicitAny": true,
    "noUncheckedIndexedAccess": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "isolatedModules": true,
    "resolveJsonModule": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "types": ["vite/client", "@webgpu/types"]
  },
  "include": ["src/**/*", "tests/**/*"]
}
```

- [ ] **Step 5: Write `vite.config.ts`**

```ts
import { defineConfig } from 'vite';

export default defineConfig({
  server: { port: 5173, open: true },
  assetsInclude: ['**/*.wgsl'],
});
```

Vite already handles `?raw` imports for arbitrary files, so `*.wgsl` will import as a string with `import source from './foo.wgsl?raw'`.

- [ ] **Step 6: Write `vitest.config.ts`**

```ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
  },
});
```

- [ ] **Step 7: Write `index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Real Refraction</title>
    <style>
      html, body { margin: 0; padding: 0; background: #000; color: #fff; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
      canvas { display: block; width: 100vw; height: 100vh; }
      #fallback { position: fixed; inset: 0; display: none; align-items: center; justify-content: center; text-align: center; padding: 2rem; font-size: 1.1rem; line-height: 1.6; }
      #fallback.visible { display: flex; }
    </style>
  </head>
  <body>
    <canvas id="gpu"></canvas>
    <div id="fallback">
      This demo needs a WebGPU-capable browser.<br />
      Try Chrome/Edge 120+ on macOS/Windows/Linux, or Safari 18+ on macOS/iOS.
    </div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

- [ ] **Step 8: Write `.gitignore`**

```
node_modules
dist
.DS_Store
.vscode
.superpowers
*.log
```

- [ ] **Step 9: Write placeholder `src/main.ts`**

```ts
console.log('Real Refraction — boot');
```

- [ ] **Step 10: Verify dev server boots**

```bash
bun run dev
```

Open http://localhost:5173. Expected: black page, console shows `Real Refraction — boot`. Stop the server with Ctrl+C.

- [ ] **Step 11: `[COMMIT POINT]` Stop and ask user before committing.**

Suggested message: `chore: scaffold Vite + bun + TypeScript project`.

---

## Task 2: WebGPU device bootstrap

**Files:**
- Create: `src/webgpu/device.ts`
- Modify: `src/main.ts`

- [ ] **Step 1: Write `src/webgpu/device.ts`**

```ts
export type GpuContext = {
  device: GPUDevice;
  canvas: HTMLCanvasElement;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
  dpr: number;
};

export async function initGpu(canvasId: string): Promise<GpuContext | null> {
  if (!('gpu' in navigator)) return null;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return null;
  const device = await adapter.requestDevice();

  const canvas = document.getElementById(canvasId);
  if (!(canvas instanceof HTMLCanvasElement)) {
    throw new Error(`Canvas #${canvasId} not found`);
  }
  const context = canvas.getContext('webgpu');
  if (!context) return null;

  const format = navigator.gpu.getPreferredCanvasFormat();
  const dpr = Math.min(window.devicePixelRatio ?? 1, 2);
  resizeCanvas(canvas, dpr);
  context.configure({ device, format, alphaMode: 'opaque' });

  return { device, canvas, context, format, dpr };
}

export function resizeCanvas(canvas: HTMLCanvasElement, dpr: number): { width: number; height: number } {
  const width = Math.floor(canvas.clientWidth * dpr);
  const height = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { width, height };
}
```

- [ ] **Step 2: Rewrite `src/main.ts`**

```ts
import { initGpu } from './webgpu/device';

async function main() {
  const ctx = await initGpu('gpu');
  if (!ctx) {
    document.getElementById('fallback')?.classList.add('visible');
    document.getElementById('gpu')?.setAttribute('style', 'display:none');
    return;
  }
  console.log('WebGPU ready:', ctx.format, ctx.canvas.width, ctx.canvas.height);
}

main().catch((err) => {
  console.error(err);
  document.getElementById('fallback')?.classList.add('visible');
});
```

- [ ] **Step 3: Run dev server and confirm**

```bash
bun run dev
```

Expected in a WebGPU browser: console prints `WebGPU ready: bgra8unorm <w> <h>`. In a non-WebGPU browser: the fallback message is visible. Stop server.

- [ ] **Step 4: `[COMMIT POINT]` Ask user before committing.**

Suggested message: `feat: initialize WebGPU device and fallback`.

---

## Task 3: Fullscreen triangle pipeline (solid color)

**Files:**
- Create: `src/shaders/fullscreen.wgsl`, `src/shaders/dispersion.wgsl`, `src/webgpu/pipeline.ts`
- Modify: `src/main.ts`

- [ ] **Step 1: Write `src/shaders/fullscreen.wgsl`**

```wgsl
// Fullscreen triangle — single triangle covering NDC, no vertex buffers.
struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0)       uv : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
  // Covering triangle: (-1,-3), (-1,1), (3,1) in NDC -> UVs in [0,1]
  let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
  let y = f32(vi & 2u) * 2.0 - 1.0;
  var out: VsOut;
  out.pos = vec4<f32>(x, y, 0.0, 1.0);
  out.uv  = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
  return out;
}
```

- [ ] **Step 2: Write `src/shaders/dispersion.wgsl` (solid color placeholder)**

```wgsl
@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  return vec4<f32>(uv.x, uv.y, 0.5, 1.0);
}
```

- [ ] **Step 3: Write `src/webgpu/pipeline.ts`**

```ts
import type { GpuContext } from './device';
import vsSrc from '../shaders/fullscreen.wgsl?raw';
import fsSrc from '../shaders/dispersion.wgsl?raw';

export type Pipeline = {
  pipeline: GPURenderPipeline;
};

export function createPipeline(ctx: GpuContext): Pipeline {
  const { device, format } = ctx;
  const module = device.createShaderModule({
    label: 'dispersion',
    code: vsSrc + '\n' + fsSrc,
  });
  const pipeline = device.createRenderPipeline({
    label: 'dispersion-pipeline',
    layout: 'auto',
    vertex:   { module, entryPoint: 'vs_main' },
    fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  return { pipeline };
}

export function draw(ctx: GpuContext, pl: Pipeline): void {
  const encoder = ctx.device.createCommandEncoder({ label: 'draw' });
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view:      ctx.context.getCurrentTexture().createView(),
      loadOp:    'clear',
      storeOp:   'store',
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
    }],
  });
  pass.setPipeline(pl.pipeline);
  pass.draw(3, 1, 0, 0);
  pass.end();
  ctx.device.queue.submit([encoder.finish()]);
}
```

- [ ] **Step 4: Modify `src/main.ts` to draw every frame**

Replace the body of `main` (after the `ctx` null-check) with:

```ts
  const pl = createPipeline(ctx);
  const loop = () => {
    draw(ctx, pl);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
```

And add imports at the top:

```ts
import { initGpu } from './webgpu/device';
import { createPipeline, draw } from './webgpu/pipeline';
```

- [ ] **Step 5: Verify a gradient renders**

```bash
bun run dev
```

Expected: full-viewport gradient (red/green across UV). No console errors. Stop server.

- [ ] **Step 6: `[COMMIT POINT]` Ask user before committing.**

Suggested message: `feat: fullscreen pipeline rendering UV gradient`.

---

## Task 4: Load Picsum photo as texture

**Files:**
- Create: `src/photo.ts`
- Modify: `src/shaders/dispersion.wgsl`, `src/webgpu/pipeline.ts`, `src/main.ts`

- [ ] **Step 1: Write `src/photo.ts`**

```ts
export type PhotoTex = {
  texture: GPUTexture;
  sampler: GPUSampler;
  width: number;
  height: number;
};

export async function loadPhoto(device: GPUDevice, seed = Date.now()): Promise<PhotoTex> {
  const url = `https://picsum.photos/seed/${seed}/1920/1080`;
  const res = await fetch(url, { mode: 'cors' });
  if (!res.ok) throw new Error(`Photo fetch failed: ${res.status}`);
  const blob = await res.blob();
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none' });

  const texture = device.createTexture({
    label:  'photo',
    size:   [bitmap.width, bitmap.height, 1],
    format: 'rgba8unorm-srgb',
    usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture(
    { source: bitmap },
    { texture },
    [bitmap.width, bitmap.height, 1],
  );
  bitmap.close();

  const sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });
  return { texture, sampler, width: bitmap.width, height: bitmap.height };
}
```

- [ ] **Step 2: Update `src/shaders/dispersion.wgsl` to sample the photo**

Replace the whole file with:

```wgsl
struct Frame {
  resolution: vec2<f32>,
  photoSize:  vec2<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var photoTex: texture_2d<f32>;
@group(0) @binding(2) var photoSmp: sampler;

// Cover-fit UV for the photo so it fills the viewport without distortion.
fn coverUv(uv: vec2<f32>) -> vec2<f32> {
  let screenAspect = frame.resolution.x / frame.resolution.y;
  let photoAspect  = frame.photoSize.x  / frame.photoSize.y;
  var s = vec2<f32>(1.0, 1.0);
  if (screenAspect > photoAspect) {
    s = vec2<f32>(1.0, photoAspect / screenAspect);
  } else {
    s = vec2<f32>(screenAspect / photoAspect, 1.0);
  }
  return (uv - 0.5) * s + 0.5;
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  let bg = textureSample(photoTex, photoSmp, coverUv(uv)).rgb;
  return vec4<f32>(bg, 1.0);
}
```

- [ ] **Step 3: Add a uniform buffer writer — create `src/webgpu/uniforms.ts`**

```ts
export type FrameParams = {
  resolution: [number, number];
  photoSize:  [number, number];
};

export function createFrameBuffer(device: GPUDevice): GPUBuffer {
  return device.createBuffer({
    label: 'frame',
    size:  16, // vec2 + vec2, 16 bytes (two vec2<f32>)
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

export function writeFrame(device: GPUDevice, buf: GPUBuffer, p: FrameParams): void {
  const data = new Float32Array([p.resolution[0], p.resolution[1], p.photoSize[0], p.photoSize[1]]);
  device.queue.writeBuffer(buf, 0, data);
}
```

- [ ] **Step 4: Rewrite `src/webgpu/pipeline.ts` to bind photo + frame**

```ts
import type { GpuContext } from './device';
import type { PhotoTex } from '../photo';
import vsSrc from '../shaders/fullscreen.wgsl?raw';
import fsSrc from '../shaders/dispersion.wgsl?raw';

export type Pipeline = {
  pipeline:  GPURenderPipeline;
  bindGroup: GPUBindGroup;
};

export function createPipeline(
  ctx: GpuContext,
  frameBuf: GPUBuffer,
  photo: PhotoTex,
): Pipeline {
  const { device, format } = ctx;
  const module = device.createShaderModule({ label: 'dispersion', code: vsSrc + '\n' + fsSrc });

  const pipeline = device.createRenderPipeline({
    label: 'dispersion-pipeline',
    layout: 'auto',
    vertex:   { module, entryPoint: 'vs_main' },
    fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: frameBuf } },
      { binding: 1, resource: photo.texture.createView() },
      { binding: 2, resource: photo.sampler },
    ],
  });

  return { pipeline, bindGroup };
}

export function draw(ctx: GpuContext, pl: Pipeline): void {
  const encoder = ctx.device.createCommandEncoder({ label: 'draw' });
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view:      ctx.context.getCurrentTexture().createView(),
      loadOp:    'clear',
      storeOp:   'store',
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
    }],
  });
  pass.setPipeline(pl.pipeline);
  pass.setBindGroup(0, pl.bindGroup);
  pass.draw(3, 1, 0, 0);
  pass.end();
  ctx.device.queue.submit([encoder.finish()]);
}
```

- [ ] **Step 5: Wire up in `src/main.ts`**

Rewrite `main()` body:

```ts
import { initGpu, resizeCanvas } from './webgpu/device';
import { createPipeline, draw } from './webgpu/pipeline';
import { createFrameBuffer, writeFrame } from './webgpu/uniforms';
import { loadPhoto } from './photo';

async function main() {
  const ctx = await initGpu('gpu');
  if (!ctx) {
    document.getElementById('fallback')?.classList.add('visible');
    document.getElementById('gpu')?.setAttribute('style', 'display:none');
    return;
  }
  const photo    = await loadPhoto(ctx.device);
  const frameBuf = createFrameBuffer(ctx.device);
  const pl       = createPipeline(ctx, frameBuf, photo);

  const loop = () => {
    const { width, height } = resizeCanvas(ctx.canvas, ctx.dpr);
    writeFrame(ctx.device, frameBuf, {
      resolution: [width, height],
      photoSize:  [photo.width, photo.height],
    });
    draw(ctx, pl);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

main().catch((err) => {
  console.error(err);
  document.getElementById('fallback')?.classList.add('visible');
});
```

- [ ] **Step 6: Verify photo renders**

```bash
bun run dev
```

Expected: a random landscape photo covering the viewport. Refresh for a different photo. Stop server.

- [ ] **Step 7: `[COMMIT POINT]` Ask user before committing.**

Suggested message: `feat: load and display random Picsum photo as background`.

---

## Task 5: Cauchy IOR math module + tests

**Files:**
- Create: `src/math/cauchy.ts`, `tests/cauchy.test.ts`

- [ ] **Step 1: Write the failing test — `tests/cauchy.test.ts`**

```ts
import { describe, it, expect } from 'vitest';
import { cauchyIor } from '../src/math/cauchy';

describe('cauchyIor', () => {
  it('returns n_d at the d-line wavelength (587.56 nm)', () => {
    // At λ = 587.56 nm, the formula should return ~n_d regardless of V_d.
    // glTF dispersion formula: n(λ) = n_d + (n_d - 1)/V_d * (523655/λ² − 1.5168)
    // 523655 / 587.56² ≈ 1.5168, so the offset term ≈ 0.
    expect(cauchyIor(587.56, 1.5168, 64.2)).toBeCloseTo(1.5168, 3);
  });

  it('returns a larger IOR for shorter (blue) wavelengths', () => {
    const blue = cauchyIor(440, 1.5168, 40);
    const d    = cauchyIor(587.56, 1.5168, 40);
    const red  = cauchyIor(660, 1.5168, 40);
    expect(blue).toBeGreaterThan(d);
    expect(red).toBeLessThan(d);
  });

  it('increases dispersion (blue − red gap) as V_d decreases', () => {
    const lowVd  = cauchyIor(440, 1.5168, 20) - cauchyIor(660, 1.5168, 20);
    const highVd = cauchyIor(440, 1.5168, 80) - cauchyIor(660, 1.5168, 80);
    expect(lowVd).toBeGreaterThan(highVd);
  });

  it('never returns an IOR below 1.0', () => {
    expect(cauchyIor(1000, 1.01, 10)).toBeGreaterThanOrEqual(1.0);
  });
});
```

- [ ] **Step 2: Run the test — expect FAIL**

```bash
bun run test
```

Expected: `Cannot find module '../src/math/cauchy'` or similar — module not created yet.

- [ ] **Step 3: Write `src/math/cauchy.ts`**

```ts
/**
 * Wavelength-dependent IOR using the glTF KHR_materials_dispersion formulation:
 *   n(λ) = n_d + (n_d - 1) / V_d * (523655 / λ² − 1.5168)
 *
 * - λ in nm (visible: 380–700)
 * - n_d: refractive index at the sodium d-line (≈587.56 nm)
 * - V_d: Abbe number (lower = more dispersion)
 */
export function cauchyIor(lambdaNm: number, n_d: number, V_d: number): number {
  const offset = (n_d - 1) / V_d * (523655 / (lambdaNm * lambdaNm) - 1.5168);
  return Math.max(n_d + offset, 1.0);
}
```

- [ ] **Step 4: Run the test — expect PASS**

```bash
bun run test
```

Expected: 4/4 tests pass.

- [ ] **Step 5: `[COMMIT POINT]` Ask user before committing.**

Suggested message: `feat: Cauchy IOR with Abbe number (glTF formulation)`.

---

## Task 6: Wyman CIE XYZ module + tests

**Files:**
- Create: `src/math/wyman.ts`, `tests/wyman.test.ts`

Reference coefficients (Wyman-Sloan-Shirley, JCGT 2013, Table 1, "multi-Gaussian fit"):

| lobe | amp | μ (nm) | σ₁ (below μ) | σ₂ (above μ) |
|---|---|---|---|---|
| x₁ |  0.362 | 442.0 | 16.0 | 26.7 |
| x₂ |  1.056 | 599.8 | 37.9 | 31.0 |
| x₃ | −0.065 | 501.1 | 20.4 | 26.2 |
| y₁ |  0.821 | 568.8 | 46.9 | 40.5 |
| y₂ |  0.286 | 530.9 | 16.3 | 31.1 |
| z₁ |  1.217 | 437.0 | 11.8 | 36.0 |
| z₂ |  0.681 | 459.0 | 26.0 | 13.8 |

Asymmetric Gaussian: `g(λ, μ, σ₁, σ₂) = exp(−0.5 · ((λ−μ)/σ)²)` with `σ = σ₁ if λ < μ else σ₂`.

- [ ] **Step 1: Write the failing test — `tests/wyman.test.ts`**

```ts
import { describe, it, expect } from 'vitest';
import { cieXyz } from '../src/math/wyman';

describe('cieXyz (Wyman-Sloan-Shirley approximation)', () => {
  it('peaks of ȳ are near 555 nm (photopic luminosity)', () => {
    let peakLambda = 555;
    let peakY = 0;
    for (let l = 400; l <= 700; l += 1) {
      const y = cieXyz(l)[1];
      if (y > peakY) { peakY = y; peakLambda = l; }
    }
    expect(peakLambda).toBeGreaterThan(545);
    expect(peakLambda).toBeLessThan(570);
  });

  it('x̄ is higher than ȳ and z̄ in the long-wavelength red (650 nm)', () => {
    const [x, y, z] = cieXyz(650);
    expect(x).toBeGreaterThan(y);
    expect(x).toBeGreaterThan(z);
  });

  it('z̄ dominates in the short-wavelength blue (450 nm)', () => {
    const [x, y, z] = cieXyz(450);
    expect(z).toBeGreaterThan(x);
    expect(z).toBeGreaterThan(y);
  });

  it('returns near-zero for far-UV and far-IR', () => {
    const uv = cieXyz(350);
    const ir = cieXyz(780);
    for (const v of [...uv, ...ir]) expect(Math.abs(v)).toBeLessThan(0.05);
  });
});
```

- [ ] **Step 2: Run tests — expect FAIL (module missing)**

```bash
bun run test
```

- [ ] **Step 3: Write `src/math/wyman.ts`**

```ts
type Lobe = readonly [amp: number, mu: number, s1: number, s2: number];

const X_LOBES: readonly Lobe[] = [
  [ 0.362, 442.0, 16.0, 26.7],
  [ 1.056, 599.8, 37.9, 31.0],
  [-0.065, 501.1, 20.4, 26.2],
];
const Y_LOBES: readonly Lobe[] = [
  [0.821, 568.8, 46.9, 40.5],
  [0.286, 530.9, 16.3, 31.1],
];
const Z_LOBES: readonly Lobe[] = [
  [1.217, 437.0, 11.8, 36.0],
  [0.681, 459.0, 26.0, 13.8],
];

function sumLobes(lobes: readonly Lobe[], lambda: number): number {
  let acc = 0;
  for (const [amp, mu, s1, s2] of lobes) {
    const sigma = lambda < mu ? s1 : s2;
    const t = (lambda - mu) / sigma;
    acc += amp * Math.exp(-0.5 * t * t);
  }
  return acc;
}

/** Wyman-Sloan-Shirley (JCGT 2013) analytic CIE 1931 2° XYZ matching functions. */
export function cieXyz(lambdaNm: number): [number, number, number] {
  return [
    sumLobes(X_LOBES, lambdaNm),
    sumLobes(Y_LOBES, lambdaNm),
    sumLobes(Z_LOBES, lambdaNm),
  ];
}
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
bun run test
```

- [ ] **Step 5: `[COMMIT POINT]` Ask user before committing.**

Suggested message: `feat: CIE XYZ Wyman-Sloan-Shirley approximation`.

---

## Task 7: XYZ → sRGB conversion + tests

**Files:**
- Create: `src/math/srgb.ts`, `tests/srgb.test.ts`

- [ ] **Step 1: Write the failing test — `tests/srgb.test.ts`**

```ts
import { describe, it, expect } from 'vitest';
import { xyzToLinearSrgb, linearToGamma } from '../src/math/srgb';

describe('xyzToLinearSrgb', () => {
  it('maps D65 white (X=0.95047, Y=1.0, Z=1.08883) to approximately (1,1,1)', () => {
    const [r, g, b] = xyzToLinearSrgb([0.95047, 1.0, 1.08883]);
    expect(r).toBeCloseTo(1.0, 2);
    expect(g).toBeCloseTo(1.0, 2);
    expect(b).toBeCloseTo(1.0, 2);
  });

  it('maps pure luminance (Y only) to a neutral-ish gray', () => {
    const [r, g, b] = xyzToLinearSrgb([0, 0.5, 0]);
    // Y-only has non-zero R/G/B but G dominates.
    expect(g).toBeGreaterThan(0);
    expect(g).toBeGreaterThan(r);
    expect(g).toBeGreaterThan(b);
  });
});

describe('linearToGamma', () => {
  it('identity at 0 and 1', () => {
    expect(linearToGamma(0)).toBeCloseTo(0);
    expect(linearToGamma(1)).toBeCloseTo(1);
  });

  it('linear segment for dark values (< 0.0031308)', () => {
    expect(linearToGamma(0.001)).toBeCloseTo(0.001 * 12.92, 6);
  });

  it('gamma-power segment for brighter values', () => {
    const v = 0.5;
    const expected = 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
    expect(linearToGamma(v)).toBeCloseTo(expected, 6);
  });
});
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
bun run test
```

- [ ] **Step 3: Write `src/math/srgb.ts`**

```ts
/** D65 XYZ (normalized, Y=1) → linear sRGB matrix. */
const M = [
  [ 3.2404542, -1.5371385, -0.4985314],
  [-0.9692660,  1.8760108,  0.0415560],
  [ 0.0556434, -0.2040259,  1.0572252],
] as const;

export function xyzToLinearSrgb(xyz: readonly [number, number, number]): [number, number, number] {
  const [x, y, z] = xyz;
  return [
    M[0][0]*x + M[0][1]*y + M[0][2]*z,
    M[1][0]*x + M[1][1]*y + M[1][2]*z,
    M[2][0]*x + M[2][1]*y + M[2][2]*z,
  ];
}

/** IEC 61966-2-1 sRGB OETF (linear → gamma-encoded). */
export function linearToGamma(v: number): number {
  if (v <= 0) return 0;
  if (v <= 0.0031308) return v * 12.92;
  return 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
}
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
bun run test
```

- [ ] **Step 5: `[COMMIT POINT]` Ask user before committing.**

Suggested message: `feat: XYZ to sRGB conversion and OETF`.

---

## Task 8: Pill SDF (TS) + tests

**Files:**
- Create: `src/math/sdfPill.ts`, `tests/sdfPill.test.ts`

- [ ] **Step 1: Write the failing test — `tests/sdfPill.test.ts`**

```ts
import { describe, it, expect } from 'vitest';
import { sdfPill3d } from '../src/math/sdfPill';

const HS  : [number, number, number] = [160, 44, 20];
const EDGE = 14;

describe('sdfPill3d', () => {
  it('returns a negative value inside the pill (at origin)', () => {
    expect(sdfPill3d([0, 0, 0], HS, EDGE)).toBeLessThan(0);
  });

  it('returns a positive value far outside the pill', () => {
    expect(sdfPill3d([500, 0, 0], HS, EDGE)).toBeGreaterThan(0);
    expect(sdfPill3d([0, 500, 0], HS, EDGE)).toBeGreaterThan(0);
    expect(sdfPill3d([0, 0, 500], HS, EDGE)).toBeGreaterThan(0);
  });

  it('returns ~0 on the top face near the center (z = hz)', () => {
    expect(Math.abs(sdfPill3d([0, 0, HS[2]], HS, EDGE))).toBeLessThan(0.5);
  });

  it('along the long axis, leaves the shape near x = hx', () => {
    expect(sdfPill3d([HS[0] - EDGE - 1, 0, 0], HS, EDGE)).toBeLessThan(0);
    expect(sdfPill3d([HS[0] + 2,        0, 0], HS, EDGE)).toBeGreaterThan(0);
  });

  it('is symmetric across all three axes', () => {
    const p: [number, number, number] = [30, 10, 5];
    const base = sdfPill3d(p, HS, EDGE);
    expect(sdfPill3d([-p[0],  p[1],  p[2]], HS, EDGE)).toBeCloseTo(base, 6);
    expect(sdfPill3d([ p[0], -p[1],  p[2]], HS, EDGE)).toBeCloseTo(base, 6);
    expect(sdfPill3d([ p[0],  p[1], -p[2]], HS, EDGE)).toBeCloseTo(base, 6);
  });

  it('rounded edges: stepping diagonally from a corner is smooth (no sharp creases)', () => {
    const a = sdfPill3d([HS[0] - EDGE, HS[1] - EDGE, HS[2] - EDGE], HS, EDGE);
    const b = sdfPill3d([HS[0] - EDGE + 0.1, HS[1] - EDGE + 0.1, HS[2] - EDGE + 0.1], HS, EDGE);
    // Monotonic increase as we leave the interior; difference ≈ sqrt(3) · 0.1 for a rounded edge.
    expect(b - a).toBeGreaterThan(0);
    expect(b - a).toBeLessThan(0.3);
  });
});
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
bun run test
```

- [ ] **Step 3: Write `src/math/sdfPill.ts`**

```ts
type Vec3 = readonly [number, number, number];

/** 3D pill SDF (stadium from top, rounded slab from the side). Matches the WGSL version. */
export function sdfPill3d(p: Vec3, halfSize: Vec3, edgeR: number): number {
  const hsX = halfSize[0] - edgeR;
  const hsY = halfSize[1] - edgeR;
  const rXy = Math.min(hsX, hsY);

  // Step 1: 2D stadium in XY
  const qX = Math.abs(p[0]) - hsX + rXy;
  const qY = Math.abs(p[1]) - hsY + rXy;
  const maxQx = Math.max(qX, 0);
  const maxQy = Math.max(qY, 0);
  const outer2d = Math.hypot(maxQx, maxQy);
  const inner2d = Math.min(Math.max(qX, qY), 0);
  const dXy     = outer2d + inner2d - rXy;

  // Step 2: extrude into Z with edgeR corner rounding
  const wx = dXy;
  const wy = Math.abs(p[2]) - halfSize[2] + edgeR;
  const outer = Math.hypot(Math.max(wx, 0), Math.max(wy, 0));
  const inner = Math.min(Math.max(wx, wy), 0);
  return outer + inner - edgeR;
}
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
bun run test
```

- [ ] **Step 5: `[COMMIT POINT]` Ask user before committing.**

Suggested message: `feat: 3D pill SDF in TypeScript (mirrors WGSL)`.

---

## Task 9: WGSL shared module (SDF + IOR + Wyman) — visual smoke test

Port the TS math to WGSL and render the pill silhouette by sphere-tracing. No refraction yet — just prove the SDF is correct on the GPU.

**Files:**
- Modify: `src/shaders/dispersion.wgsl`, `src/webgpu/uniforms.ts`, `src/webgpu/pipeline.ts`, `src/main.ts`

- [ ] **Step 1: Extend the frame uniform with one hardcoded pill**

Edit `src/webgpu/uniforms.ts`. Replace the file with:

```ts
export type FrameParams = {
  resolution: [number, number];
  photoSize:  [number, number];
  pillCenter: [number, number, number]; // world-space (px, px, px)
  pillHalf:   [number, number, number];
  pillEdgeR:  number;
};

export function createFrameBuffer(device: GPUDevice): GPUBuffer {
  return device.createBuffer({
    label: 'frame',
    size:  48, // 4 × vec4 = 64? actually: vec2+vec2+vec3+pad+vec3+f32+pad = 48
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

export function writeFrame(device: GPUDevice, buf: GPUBuffer, p: FrameParams): void {
  // WGSL std140 alignment: vec2 → 8B, vec3 → 16B (aligned).
  // Layout (offsets):
  //   0:  resolution.xy (vec2<f32>)   [8B, pad to 16 for next vec3 alignment]
  //   16: photoSize.xy  (vec2<f32>)   [but we'll use vec4 pairs for simplicity]
  // Simpler: pack as 3 vec4s.
  //   vec4(resolution.xy, photoSize.xy)
  //   vec4(pillCenter.xyz, pillEdgeR)
  //   vec4(pillHalf.xyz,   0)
  const data = new Float32Array(12);
  data[0] = p.resolution[0];
  data[1] = p.resolution[1];
  data[2] = p.photoSize[0];
  data[3] = p.photoSize[1];
  data[4] = p.pillCenter[0];
  data[5] = p.pillCenter[1];
  data[6] = p.pillCenter[2];
  data[7] = p.pillEdgeR;
  data[8] = p.pillHalf[0];
  data[9] = p.pillHalf[1];
  data[10] = p.pillHalf[2];
  data[11] = 0;
  device.queue.writeBuffer(buf, 0, data);
}
```

And change the buffer size accordingly:

```ts
// size: 48
```

- [ ] **Step 2: Rewrite `src/shaders/dispersion.wgsl` to render the silhouette**

```wgsl
struct Frame {
  resolutionPhoto: vec4<f32>,  // xy = resolution px, zw = photo px
  pillCenterEdge:  vec4<f32>,  // xyz = pill center, w = edgeR
  pillHalfPad:     vec4<f32>,  // xyz = halfSize,   w = 0
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var photoTex: texture_2d<f32>;
@group(0) @binding(2) var photoSmp: sampler;

fn coverUv(uv: vec2<f32>) -> vec2<f32> {
  let res   = frame.resolutionPhoto.xy;
  let ph    = frame.resolutionPhoto.zw;
  let sA    = res.x / res.y;
  let pA    = ph.x  / ph.y;
  var s     = vec2<f32>(1.0, 1.0);
  if (sA > pA) { s = vec2<f32>(1.0, pA / sA); }
  else         { s = vec2<f32>(sA / pA, 1.0); }
  return (uv - vec2<f32>(0.5)) * s + vec2<f32>(0.5);
}

fn sdfPill(p: vec3<f32>, halfSize: vec3<f32>, edgeR: f32) -> f32 {
  let hsXY = halfSize.xy - vec2<f32>(edgeR);
  let rXY  = min(hsXY.x, hsXY.y);
  let qXY  = abs(p.xy) - hsXY + vec2<f32>(rXY);
  let dXy  = length(max(qXY, vec2<f32>(0.0))) + min(max(qXY.x, qXY.y), 0.0) - rXY;
  let w    = vec2<f32>(dXy, abs(p.z) - halfSize.z + edgeR);
  return length(max(w, vec2<f32>(0.0))) + min(max(w.x, w.y), 0.0) - edgeR;
}

fn sceneSdf(p: vec3<f32>) -> f32 {
  let c = frame.pillCenterEdge.xyz;
  let e = frame.pillCenterEdge.w;
  let h = frame.pillHalfPad.xyz;
  return sdfPill(p - c, h, e);
}

struct Hit { ok: bool, p: vec3<f32>, t: f32 };

fn sphereTrace(ro: vec3<f32>, rd: vec3<f32>, maxT: f32) -> Hit {
  var t: f32 = 0.0;
  for (var i: i32 = 0; i < 64; i = i + 1) {
    let p = ro + rd * t;
    let d = sceneSdf(p);
    if (d < 0.5) {
      return Hit(true, p, t);
    }
    t = t + max(d, 0.5);
    if (t > maxT) { break; }
  }
  return Hit(false, vec3<f32>(0.0), 0.0);
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  let res = frame.resolutionPhoto.xy;
  // Screen-space pixel coords with origin at top-left.
  let px = vec2<f32>(uv.x * res.x, (1.0 - uv.y) * res.y);
  // Orthographic camera looking down -Z from large +Z.
  let ro = vec3<f32>(px, 400.0);
  let rd = vec3<f32>(0.0, 0.0, -1.0);
  let h  = sphereTrace(ro, rd, 800.0);
  let bg = textureSample(photoTex, photoSmp, coverUv(uv)).rgb;
  if (h.ok) {
    return vec4<f32>(1.0, 0.0, 1.0, 1.0);  // magenta silhouette
  }
  return vec4<f32>(bg, 1.0);
}
```

- [ ] **Step 3: Update `src/main.ts` to supply pill params**

Change the `writeFrame` call to include the pill:

```ts
    writeFrame(ctx.device, frameBuf, {
      resolution: [width, height],
      photoSize:  [photo.width, photo.height],
      pillCenter: [width / 2, height / 2, 0],
      pillHalf:   [160, 44, 20],
      pillEdgeR:  14,
    });
```

- [ ] **Step 4: Update `src/webgpu/pipeline.ts`**

No code change needed beyond the auto-generated layout picking up the wider uniform. Just rebuild the bind group — already handled.

- [ ] **Step 5: Verify a magenta pill appears centered over the photo**

```bash
bun run dev
```

Expected: photo visible, with a magenta pill-shaped silhouette dead center (320 × 88 px). Resize the window — pill stays centered. Stop server.

- [ ] **Step 6: `[COMMIT POINT]` Ask user before committing.**

Suggested message: `feat: 3D pill SDF silhouette via sphere tracing`.

---

## Task 10: SDF normals visualization

**Files:**
- Modify: `src/shaders/dispersion.wgsl`

- [ ] **Step 1: Add normal computation + render normals**

In `src/shaders/dispersion.wgsl`, add this function above `fs_main`:

```wgsl
fn sceneNormal(p: vec3<f32>) -> vec3<f32> {
  let e = vec2<f32>(0.5, 0.0);
  return normalize(vec3<f32>(
    sceneSdf(p + e.xyy) - sceneSdf(p - e.xyy),
    sceneSdf(p + e.yxy) - sceneSdf(p - e.yxy),
    sceneSdf(p + e.yyx) - sceneSdf(p - e.yyx),
  ));
}
```

Replace `fs_main` with:

```wgsl
@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  let res = frame.resolutionPhoto.xy;
  let px = vec2<f32>(uv.x * res.x, (1.0 - uv.y) * res.y);
  let ro = vec3<f32>(px, 400.0);
  let rd = vec3<f32>(0.0, 0.0, -1.0);
  let h  = sphereTrace(ro, rd, 800.0);
  let bg = textureSample(photoTex, photoSmp, coverUv(uv)).rgb;
  if (h.ok) {
    let n = sceneNormal(h.p);
    return vec4<f32>(n * 0.5 + 0.5, 1.0);  // encode normal to [0,1] RGB
  }
  return vec4<f32>(bg, 1.0);
}
```

- [ ] **Step 2: Verify a smoothly-shaded pill**

```bash
bun run dev
```

Expected: pill shows a smooth gradient of RGB corresponding to normal direction. Top face (normal = +Z) appears ≈ (0.5, 0.5, 1.0) = bluish. Rounded edges fade into red/green tints as the normal tilts outward. No visible sharp creases anywhere. Stop server.

- [ ] **Step 3: `[COMMIT POINT]` Ask user before committing.**

Suggested message: `feat: SDF gradient normals visualized`.

---

## Task 11: Single-IOR refraction (no dispersion yet)

**Files:**
- Modify: `src/shaders/dispersion.wgsl`

- [ ] **Step 1: Add a simple two-surface refraction block**

Replace `fs_main` with:

```wgsl
const IOR_TEST: f32 = 1.5;
const REFRACTION_STRENGTH: f32 = 0.1;

fn screenUvFromWorld(pxWorld: vec2<f32>) -> vec2<f32> {
  let res = frame.resolutionPhoto.xy;
  return vec2<f32>(pxWorld.x / res.x, 1.0 - pxWorld.y / res.y);
}

fn insideTrace(ro: vec3<f32>, rd: vec3<f32>, maxT: f32) -> vec3<f32> {
  var t: f32 = 0.0;
  var p = ro;
  for (var i: i32 = 0; i < 32; i = i + 1) {
    p = ro + rd * t;
    let d = -sceneSdf(p);  // distance to BACK surface from inside
    if (d < 0.5) { return p; }
    t = t + max(d, 0.5);
    if (t > maxT) { break; }
  }
  return p;
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  let res = frame.resolutionPhoto.xy;
  let px  = vec2<f32>(uv.x * res.x, (1.0 - uv.y) * res.y);
  let ro  = vec3<f32>(px, 400.0);
  let rd  = vec3<f32>(0.0, 0.0, -1.0);
  let h   = sphereTrace(ro, rd, 800.0);
  let bg  = textureSample(photoTex, photoSmp, coverUv(uv)).rgb;
  if (!h.ok) { return vec4<f32>(bg, 1.0); }

  let nFront = sceneNormal(h.p);
  let r1     = refract(rd, nFront, 1.0 / IOR_TEST);
  let pExit  = insideTrace(h.p + r1 * 1.0, r1, 300.0);
  let nBack  = -sceneNormal(pExit);
  let r2     = refract(r1, nBack, IOR_TEST);

  let uvOff  = screenUvFromWorld(pExit.xy) + r2.xy * REFRACTION_STRENGTH;
  let col    = textureSample(photoTex, photoSmp, coverUv(uvOff)).rgb;
  return vec4<f32>(col, 1.0);
}
```

- [ ] **Step 2: Verify glass-like distortion**

```bash
bun run dev
```

Expected: pill now acts like a clear glass lens — the photo is warped inside the pill but colorless (no chromatic fringing yet). Drag the mouse around; pill stays centered. The distortion should be strongest near the rounded rim. Stop server.

- [ ] **Step 3: `[COMMIT POINT]` Ask user before committing.**

Suggested message: `feat: two-surface screen-space refraction (single IOR)`.

---

## Task 12: Spectral wavelength loop — the money shot

**Files:**
- Modify: `src/shaders/dispersion.wgsl`, `src/webgpu/uniforms.ts`, `src/main.ts`

- [ ] **Step 1: Extend the frame uniform with spectral params**

Edit `src/webgpu/uniforms.ts`. Replace the whole file:

```ts
export type FrameParams = {
  resolution:         [number, number];
  photoSize:          [number, number];
  pillCenter:         [number, number, number];
  pillHalf:           [number, number, number];
  pillEdgeR:          number;
  n_d:                number;
  V_d:                number;
  sampleCount:        number; // 3, 8, 16, 32
  refractionStrength: number;
  jitter:             number; // 0..1 per frame
  refractionMode:     number; // 0 = exact, 1 = approx
};

export function createFrameBuffer(device: GPUDevice): GPUBuffer {
  return device.createBuffer({
    label: 'frame',
    size:  80, // 5 × vec4 padded
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

export function writeFrame(device: GPUDevice, buf: GPUBuffer, p: FrameParams): void {
  // Layout (5 × vec4):
  // [ res.xy, photo.xy ]
  // [ pillCenter.xyz, pillEdgeR ]
  // [ pillHalf.xyz,   _ ]
  // [ n_d, V_d, sampleCount, refractionStrength ]
  // [ jitter, refractionMode, _, _ ]
  const d = new Float32Array(20);
  d[0]  = p.resolution[0];
  d[1]  = p.resolution[1];
  d[2]  = p.photoSize[0];
  d[3]  = p.photoSize[1];
  d[4]  = p.pillCenter[0];
  d[5]  = p.pillCenter[1];
  d[6]  = p.pillCenter[2];
  d[7]  = p.pillEdgeR;
  d[8]  = p.pillHalf[0];
  d[9]  = p.pillHalf[1];
  d[10] = p.pillHalf[2];
  d[11] = 0;
  d[12] = p.n_d;
  d[13] = p.V_d;
  d[14] = p.sampleCount;
  d[15] = p.refractionStrength;
  d[16] = p.jitter;
  d[17] = p.refractionMode;
  d[18] = 0;
  d[19] = 0;
  device.queue.writeBuffer(buf, 0, d);
}
```

- [ ] **Step 2: Update `src/shaders/dispersion.wgsl`**

Replace the whole file with:

```wgsl
struct Frame {
  resolutionPhoto: vec4<f32>, // xy = resolution px, zw = photo px
  pillCenterEdge:  vec4<f32>, // xyz = pill center, w = edgeR
  pillHalfPad:     vec4<f32>, // xyz = halfSize,    w = 0
  spectralA:       vec4<f32>, // x = n_d, y = V_d, z = N, w = refractionStrength
  spectralB:       vec4<f32>, // x = jitter, y = refractionMode (0 exact, 1 approx)
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var photoTex: texture_2d<f32>;
@group(0) @binding(2) var photoSmp: sampler;

// ---------- coords ----------

fn coverUv(uv: vec2<f32>) -> vec2<f32> {
  let res   = frame.resolutionPhoto.xy;
  let ph    = frame.resolutionPhoto.zw;
  let sA    = res.x / res.y;
  let pA    = ph.x  / ph.y;
  var s     = vec2<f32>(1.0, 1.0);
  if (sA > pA) { s = vec2<f32>(1.0, pA / sA); } else { s = vec2<f32>(sA / pA, 1.0); }
  return (uv - vec2<f32>(0.5)) * s + vec2<f32>(0.5);
}

fn screenUvFromWorld(px: vec2<f32>) -> vec2<f32> {
  let res = frame.resolutionPhoto.xy;
  return vec2<f32>(px.x / res.x, 1.0 - px.y / res.y);
}

// ---------- SDF ----------

fn sdfPill(p: vec3<f32>, halfSize: vec3<f32>, edgeR: f32) -> f32 {
  let hsXY = halfSize.xy - vec2<f32>(edgeR);
  let rXY  = min(hsXY.x, hsXY.y);
  let qXY  = abs(p.xy) - hsXY + vec2<f32>(rXY);
  let dXy  = length(max(qXY, vec2<f32>(0.0))) + min(max(qXY.x, qXY.y), 0.0) - rXY;
  let w    = vec2<f32>(dXy, abs(p.z) - halfSize.z + edgeR);
  return length(max(w, vec2<f32>(0.0))) + min(max(w.x, w.y), 0.0) - edgeR;
}

fn sceneSdf(p: vec3<f32>) -> f32 {
  let c = frame.pillCenterEdge.xyz;
  let e = frame.pillCenterEdge.w;
  let h = frame.pillHalfPad.xyz;
  return sdfPill(p - c, h, e);
}

fn sceneNormal(p: vec3<f32>) -> vec3<f32> {
  let e = vec2<f32>(0.5, 0.0);
  return normalize(vec3<f32>(
    sceneSdf(p + e.xyy) - sceneSdf(p - e.xyy),
    sceneSdf(p + e.yxy) - sceneSdf(p - e.yxy),
    sceneSdf(p + e.yyx) - sceneSdf(p - e.yyx),
  ));
}

// ---------- tracing ----------

struct Hit { ok: bool, p: vec3<f32>, t: f32 };

fn sphereTrace(ro: vec3<f32>, rd: vec3<f32>, maxT: f32) -> Hit {
  var t: f32 = 0.0;
  for (var i: i32 = 0; i < 64; i = i + 1) {
    let p = ro + rd * t;
    let d = sceneSdf(p);
    if (d < 0.5) { return Hit(true, p, t); }
    t = t + max(d, 0.5);
    if (t > maxT) { break; }
  }
  return Hit(false, vec3<f32>(0.0), 0.0);
}

fn insideTrace(ro: vec3<f32>, rd: vec3<f32>, maxT: f32) -> vec3<f32> {
  var t: f32 = 0.0;
  var p = ro;
  for (var i: i32 = 0; i < 32; i = i + 1) {
    p = ro + rd * t;
    let d = -sceneSdf(p);
    if (d < 0.5) { return p; }
    t = t + max(d, 0.5);
    if (t > maxT) { break; }
  }
  return p;
}

// ---------- spectral math ----------

fn cauchyIor(lambda: f32, n_d: f32, V_d: f32) -> f32 {
  return max(n_d + (n_d - 1.0) / V_d * (523655.0 / (lambda * lambda) - 1.5168), 1.0);
}

fn gLobe(lambda: f32, mu: f32, s1: f32, s2: f32) -> f32 {
  let sigma = select(s2, s1, lambda < mu);
  let t = (lambda - mu) / sigma;
  return exp(-0.5 * t * t);
}

fn cieXyz(lambda: f32) -> vec3<f32> {
  let x =  0.362 * gLobe(lambda, 442.0, 16.0, 26.7)
        +  1.056 * gLobe(lambda, 599.8, 37.9, 31.0)
        + -0.065 * gLobe(lambda, 501.1, 20.4, 26.2);
  let y =  0.821 * gLobe(lambda, 568.8, 46.9, 40.5)
        +  0.286 * gLobe(lambda, 530.9, 16.3, 31.1);
  let z =  1.217 * gLobe(lambda, 437.0, 11.8, 36.0)
        +  0.681 * gLobe(lambda, 459.0, 26.0, 13.8);
  return vec3<f32>(x, y, z);
}

fn xyzToSrgb(c: vec3<f32>) -> vec3<f32> {
  let m = mat3x3<f32>(
     3.2404542, -0.9692660,  0.0556434,
    -1.5371385,  1.8760108, -0.2040259,
    -0.4985314,  0.0415560,  1.0572252,
  );
  return m * c;
}

fn luminance(rgb: vec3<f32>) -> f32 {
  return dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// ---------- fragment ----------

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  let res = frame.resolutionPhoto.xy;
  let px  = vec2<f32>(uv.x * res.x, (1.0 - uv.y) * res.y);
  let ro  = vec3<f32>(px, 400.0);
  let rd  = vec3<f32>(0.0, 0.0, -1.0);
  let h   = sphereTrace(ro, rd, 800.0);
  let bgUv = coverUv(uv);
  let bg  = textureSample(photoTex, photoSmp, bgUv).rgb;
  if (!h.ok) { return vec4<f32>(bg, 1.0); }

  let nFront   = sceneNormal(h.p);
  let n_d      = frame.spectralA.x;
  let V_d      = frame.spectralA.y;
  let N        = i32(frame.spectralA.z);
  let strength = frame.spectralA.w;
  let jitter   = frame.spectralB.x;
  let approx   = frame.spectralB.y > 0.5;

  // Approx mode: do one shared exit trace at the central wavelength.
  var sharedExit: vec3<f32>;
  var sharedNBack: vec3<f32>;
  if (approx) {
    let iorMid   = cauchyIor(540.0, n_d, V_d);
    let r1mid    = refract(rd, nFront, 1.0 / iorMid);
    sharedExit   = insideTrace(h.p + r1mid * 1.0, r1mid, 300.0);
    sharedNBack  = -sceneNormal(sharedExit);
  } else {
    sharedExit  = h.p;
    sharedNBack = -nFront;
  }

  var xyz = vec3<f32>(0.0);
  var cmfSum = vec3<f32>(0.0);

  for (var i: i32 = 0; i < N; i = i + 1) {
    let t      = (f32(i) + 0.5 + jitter) / f32(N);
    let lambda = mix(380.0, 700.0, t);
    let ior    = cauchyIor(lambda, n_d, V_d);

    let r1     = refract(rd, nFront, 1.0 / ior);
    var pExit: vec3<f32>;
    var nBack: vec3<f32>;
    if (approx) {
      pExit = sharedExit;
      nBack = sharedNBack;
    } else {
      pExit = insideTrace(h.p + r1 * 1.0, r1, 300.0);
      nBack = -sceneNormal(pExit);
    }
    let r2     = refract(r1, nBack, ior);
    let uvOff  = screenUvFromWorld(pExit.xy) + r2.xy * strength;
    let L      = textureSample(photoTex, photoSmp, coverUv(uvOff)).rgb;

    let cmf    = cieXyz(lambda);
    xyz      = xyz + cmf * luminance(L);
    cmfSum   = cmfSum + cmf;
  }

  // Normalize by summed CMF so white photo → white output at arbitrary N.
  let normXyz = xyz / max(cmfSum, vec3<f32>(1e-4));
  let rgb     = max(xyzToSrgb(normXyz), vec3<f32>(0.0));
  return vec4<f32>(rgb, 1.0);
}
```

- [ ] **Step 3: Update `src/main.ts` to supply spectral defaults**

Replace the `writeFrame` call:

```ts
    writeFrame(ctx.device, frameBuf, {
      resolution:         [width, height],
      photoSize:          [photo.width, photo.height],
      pillCenter:         [width / 2, height / 2, 0],
      pillHalf:           [160, 44, 20],
      pillEdgeR:          14,
      n_d:                1.5168,
      V_d:                40.0,
      sampleCount:        8,
      refractionStrength: 0.1,
      jitter:             Math.random(),
      refractionMode:     0, // exact
    });
```

- [ ] **Step 4: Verify rainbow fringe**

```bash
bun run dev
```

Expected: at the pill's rounded rim, a visible rainbow fringe. The refraction in the center is nearly colorless (normal ≈ up, refraction direction barely differs per wavelength). Rim clearly shows red on one side, blue on the other — the hallmark of chromatic dispersion. Stop server.

- [ ] **Step 5: `[COMMIT POINT]` Ask user before committing.**

Suggested message: `feat: spectral dispersion via per-wavelength refraction loop`.

---

## Task 13: Fresnel + reflection

**Files:**
- Modify: `src/shaders/dispersion.wgsl`

- [ ] **Step 1: Add Fresnel + cheap mirror reflection**

In `src/shaders/dispersion.wgsl`, add before `fs_main`:

```wgsl
fn schlickFresnel(cosT: f32, n_d: f32) -> f32 {
  let f0 = pow((n_d - 1.0) / (n_d + 1.0), 2.0);
  let k  = 1.0 - clamp(cosT, 0.0, 1.0);
  return f0 + (1.0 - f0) * k * k * k * k * k;
}
```

Replace the final `return` in `fs_main` with:

```wgsl
  let cosT     = max(dot(-rd, nFront), 0.0);
  let F        = schlickFresnel(cosT, n_d);

  // Cheap environment mirror: offset the bg sample by the reflected ray, cooler tint.
  let refl     = reflect(rd, nFront);
  let reflUv   = screenUvFromWorld(h.p.xy) + refl.xy * 0.2;
  let reflRgb  = textureSample(photoTex, photoSmp, coverUv(reflUv)).rgb * vec3<f32>(0.85, 0.9, 1.0);

  let refrRgb  = rgb;
  let outRgb   = mix(refrRgb, reflRgb, F);
  return vec4<f32>(outRgb, 1.0);
```

- [ ] **Step 2: Verify Fresnel highlights**

```bash
bun run dev
```

Expected: pill rim is now noticeably brighter (more reflective where the surface tilts away from the camera), interior is unchanged. Still shows spectral dispersion at the rim. Stop server.

- [ ] **Step 3: `[COMMIT POINT]` Ask user before committing.**

Suggested message: `feat: Schlick Fresnel + mirror-reflection approximation`.

---

## Task 14: Multiple pills + mouse dragging

**Files:**
- Create: `src/pills.ts`
- Modify: `src/webgpu/uniforms.ts`, `src/shaders/dispersion.wgsl`, `src/main.ts`

- [ ] **Step 1: Write `src/pills.ts`**

```ts
export type Pill = {
  cx: number; cy: number; cz: number;
  hx: number; hy: number; hz: number;
  edgeR: number;
};

export function defaultPills(width: number, height: number, count = 4): Pill[] {
  const pills: Pill[] = [];
  const step = width / (count + 1);
  for (let i = 0; i < count; i++) {
    pills.push({
      cx: step * (i + 1),
      cy: height * 0.5 + (i % 2 === 0 ? -60 : 60),
      cz: 0,
      hx: 160, hy: 44, hz: 20,
      edgeR: 14,
    });
  }
  return pills;
}

type DragState = { pillIndex: number | null; offsetX: number; offsetY: number };

export function attachDrag(canvas: HTMLCanvasElement, pills: Pill[], dpr: number) {
  const state: DragState = { pillIndex: null, offsetX: 0, offsetY: 0 };

  const toWorld = (e: PointerEvent) => {
    const r = canvas.getBoundingClientRect();
    return { x: (e.clientX - r.left) * dpr, y: (e.clientY - r.top) * dpr };
  };

  const findHit = (x: number, y: number): number => {
    // Pick the topmost pill whose (expanded) AABB contains the pointer.
    for (let i = pills.length - 1; i >= 0; i--) {
      const p = pills[i]!;
      if (Math.abs(x - p.cx) <= p.hx && Math.abs(y - p.cy) <= p.hy) return i;
    }
    return -1;
  };

  canvas.addEventListener('pointerdown', (e) => {
    const { x, y } = toWorld(e);
    const i = findHit(x, y);
    if (i >= 0) {
      state.pillIndex = i;
      state.offsetX   = x - pills[i]!.cx;
      state.offsetY   = y - pills[i]!.cy;
      canvas.setPointerCapture(e.pointerId);
    }
  });

  canvas.addEventListener('pointermove', (e) => {
    if (state.pillIndex === null) return;
    const { x, y } = toWorld(e);
    const p = pills[state.pillIndex]!;
    p.cx = x - state.offsetX;
    p.cy = y - state.offsetY;
  });

  const release = (e: PointerEvent) => {
    if (state.pillIndex !== null) {
      canvas.releasePointerCapture(e.pointerId);
      state.pillIndex = null;
    }
  };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);
}
```

- [ ] **Step 2: Extend uniforms to carry an array of pills**

Replace `src/webgpu/uniforms.ts`:

```ts
import type { Pill } from '../pills';

export const MAX_PILLS = 8;

export type FrameParams = {
  resolution:         [number, number];
  photoSize:          [number, number];
  n_d:                number;
  V_d:                number;
  sampleCount:        number;
  refractionStrength: number;
  jitter:             number;
  refractionMode:     number;
  pillCount:          number;
  pills:              Pill[];
};

// 2 × vec4 head + 2 × vec4 spectral + vec4 meta + MAX_PILLS × 2 × vec4 per pill
//   head:     resolution.xy, photo.xy                           (16B)
//   spectral: n_d, V_d, N, refractionStrength                   (16B)
//             jitter, refractionMode, pillCount, _              (16B)
//   per pill: center.xyz, edgeR                                 (16B)
//             half.xyz, _                                       (16B)
const HEAD_SIZE = 16 + 16 + 16;               // 48
const PILL_SIZE = 32;                         // 32
const TOTAL_SIZE = HEAD_SIZE + PILL_SIZE * MAX_PILLS; // 48 + 256 = 304

export function createFrameBuffer(device: GPUDevice): GPUBuffer {
  return device.createBuffer({
    label: 'frame',
    size:  TOTAL_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

export function writeFrame(device: GPUDevice, buf: GPUBuffer, p: FrameParams): void {
  const d = new Float32Array(TOTAL_SIZE / 4);
  d[0]  = p.resolution[0];
  d[1]  = p.resolution[1];
  d[2]  = p.photoSize[0];
  d[3]  = p.photoSize[1];

  d[4]  = p.n_d;
  d[5]  = p.V_d;
  d[6]  = p.sampleCount;
  d[7]  = p.refractionStrength;

  d[8]  = p.jitter;
  d[9]  = p.refractionMode;
  d[10] = p.pillCount;
  d[11] = 0;

  for (let i = 0; i < MAX_PILLS; i++) {
    const base = 12 + i * 8;
    const pill = i < p.pills.length ? p.pills[i] : null;
    if (pill) {
      d[base + 0] = pill.cx;
      d[base + 1] = pill.cy;
      d[base + 2] = pill.cz;
      d[base + 3] = pill.edgeR;
      d[base + 4] = pill.hx;
      d[base + 5] = pill.hy;
      d[base + 6] = pill.hz;
      d[base + 7] = 0;
    } // else leave zero; pillCount gates iteration in the shader
  }
  device.queue.writeBuffer(buf, 0, d);
}
```

- [ ] **Step 3: Update `sceneSdf` in the shader to iterate pills**

In `src/shaders/dispersion.wgsl`, replace the `Frame` struct and `sceneSdf`:

```wgsl
const MAX_PILLS: u32 = 8u;

struct PillGpu {
  centerEdge: vec4<f32>,  // xyz = center, w = edgeR
  halfPad:    vec4<f32>,  // xyz = halfSize, w = 0
};

struct Frame {
  resolutionPhoto: vec4<f32>,  // xy = resolution px, zw = photo px
  spectralA:       vec4<f32>,  // x = n_d, y = V_d, z = N, w = refractionStrength
  spectralB:       vec4<f32>,  // x = jitter, y = refractionMode, z = pillCount
  pills:           array<PillGpu, MAX_PILLS>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var photoTex: texture_2d<f32>;
@group(0) @binding(2) var photoSmp: sampler;
```

Replace the earlier single-pill `sceneSdf`:

```wgsl
fn sceneSdf(p: vec3<f32>) -> f32 {
  let count = u32(frame.spectralB.z);
  var d: f32 = 1e9;
  for (var i: u32 = 0u; i < count; i = i + 1u) {
    let pill = frame.pills[i];
    let local = p - pill.centerEdge.xyz;
    let hs    = pill.halfPad.xyz;
    let eR    = pill.centerEdge.w;
    d = min(d, sdfPill(local, hs, eR));
  }
  return d;
}
```

- [ ] **Step 4: Wire drag + multi-pill in `src/main.ts`**

Replace `src/main.ts`:

```ts
import { initGpu, resizeCanvas } from './webgpu/device';
import { createPipeline, draw } from './webgpu/pipeline';
import { createFrameBuffer, writeFrame } from './webgpu/uniforms';
import { loadPhoto } from './photo';
import { attachDrag, defaultPills, type Pill } from './pills';

async function main() {
  const ctx = await initGpu('gpu');
  if (!ctx) {
    document.getElementById('fallback')?.classList.add('visible');
    document.getElementById('gpu')?.setAttribute('style', 'display:none');
    return;
  }
  const photo    = await loadPhoto(ctx.device);
  const frameBuf = createFrameBuffer(ctx.device);
  const pl       = createPipeline(ctx, frameBuf, photo);

  // Initial size
  const initSize = resizeCanvas(ctx.canvas, ctx.dpr);
  let pills: Pill[] = defaultPills(initSize.width, initSize.height);
  attachDrag(ctx.canvas, pills, ctx.dpr);

  const loop = () => {
    const { width, height } = resizeCanvas(ctx.canvas, ctx.dpr);
    writeFrame(ctx.device, frameBuf, {
      resolution:         [width, height],
      photoSize:          [photo.width, photo.height],
      n_d:                1.5168,
      V_d:                40.0,
      sampleCount:        8,
      refractionStrength: 0.1,
      jitter:             Math.random(),
      refractionMode:     0,
      pillCount:          pills.length,
      pills,
    });
    draw(ctx, pl);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

main().catch((err) => {
  console.error(err);
  document.getElementById('fallback')?.classList.add('visible');
});
```

- [ ] **Step 5: Verify 4 draggable pills**

```bash
bun run dev
```

Expected: 4 pills visible, staggered horizontally. Drag any pill — it follows the cursor. Rainbow rim persists while dragging. Stop server.

- [ ] **Step 6: `[COMMIT POINT]` Ask user before committing.**

Suggested message: `feat: multi-pill rendering with pointer drag`.

---

## Task 15: Tweakpane UI

**Files:**
- Create: `src/ui.ts`
- Modify: `src/main.ts`

- [ ] **Step 1: Write `src/ui.ts`**

```ts
import { Pane } from 'tweakpane';

export type Params = {
  sampleCount: 3 | 8 | 16 | 32;
  n_d: number;
  V_d: number;
  pillLen: number;     // 2*hx
  pillShort: number;   // 2*hy
  pillThick: number;   // 2*hz
  edgeR: number;
  refractionStrength: number;
  refractionMode: 'exact' | 'approx';
  temporalJitter: boolean;
};

export function initUi(params: Params, reloadPhoto: () => void): Pane {
  const pane = new Pane({ title: 'Spectral Dispersion', expanded: true });

  const spectral = pane.addFolder({ title: 'Spectral' });
  spectral.addBinding(params, 'sampleCount', {
    options: { '3 (fake RGB)': 3, '8 (default)': 8, '16': 16, '32': 32 },
  });
  spectral.addBinding(params, 'n_d', { min: 1.0, max: 2.4, step: 0.001, label: 'IOR n_d' });
  spectral.addBinding(params, 'V_d', { min: 15,  max: 90,  step: 0.5,   label: 'Abbe V_d' });
  spectral.addBinding(params, 'refractionMode', {
    options: { Exact: 'exact', Approx: 'approx' },
  });
  spectral.addBinding(params, 'temporalJitter', { label: 'Temporal jitter' });

  const shape = pane.addFolder({ title: 'Pill shape' });
  shape.addBinding(params, 'pillLen',   { min: 80,  max: 800, step: 1, label: 'Length' });
  shape.addBinding(params, 'pillShort', { min: 20,  max: 200, step: 1, label: 'Short axis' });
  shape.addBinding(params, 'pillThick', { min: 10,  max: 200, step: 1, label: 'Thickness' });
  shape.addBinding(params, 'edgeR',     { min: 1,   max: 100, step: 0.5, label: 'Edge radius' });

  const misc = pane.addFolder({ title: 'Misc' });
  misc.addBinding(params, 'refractionStrength', { min: 0, max: 0.5, step: 0.001, label: 'Refraction' });
  const reload = misc.addButton({ title: 'Reload photo' });
  reload.on('click', reloadPhoto);

  return pane;
}

export function defaultParams(): Params {
  return {
    sampleCount: 8,
    n_d: 1.5168,
    V_d: 40,
    pillLen: 320,
    pillShort: 88,
    pillThick: 40,
    edgeR: 14,
    refractionStrength: 0.1,
    refractionMode: 'exact',
    temporalJitter: true,
  };
}
```

- [ ] **Step 2: Wire the UI into `src/main.ts`**

Replace the body after `attachDrag(...)` with:

```ts
  const params = defaultParams();
  let photoNow = photo;
  initUi(params, async () => { photoNow = await loadPhoto(ctx.device, Date.now()); });
  // Note: reloading the photo also needs the bind group rebuilt — see Step 3.
```

Add imports:

```ts
import { defaultParams, initUi } from './ui';
```

Replace the `writeFrame` call body:

```ts
    for (const pill of pills) {
      pill.hx    = params.pillLen   / 2;
      pill.hy    = params.pillShort / 2;
      pill.hz    = params.pillThick / 2;
      pill.edgeR = Math.min(params.edgeR, pill.hz);  // clamp
    }
    writeFrame(ctx.device, frameBuf, {
      resolution:         [width, height],
      photoSize:          [photoNow.width, photoNow.height],
      n_d:                params.n_d,
      V_d:                params.V_d,
      sampleCount:        params.sampleCount,
      refractionStrength: params.refractionStrength,
      jitter:             params.temporalJitter ? Math.random() : 0.5,
      refractionMode:     params.refractionMode === 'exact' ? 0 : 1,
      pillCount:          pills.length,
      pills,
    });
```

- [ ] **Step 3: Rebuild bind group when photo reloads**

The current pipeline holds a stale `bindGroup` referencing the old photo texture. Extend `src/webgpu/pipeline.ts` to expose a rebuild:

Add this export in `src/webgpu/pipeline.ts`:

```ts
export function rebuildBindGroup(
  ctx: GpuContext,
  pl: Pipeline,
  frameBuf: GPUBuffer,
  photo: PhotoTex,
): void {
  pl.bindGroup = ctx.device.createBindGroup({
    layout: pl.pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: frameBuf } },
      { binding: 1, resource: photo.texture.createView() },
      { binding: 2, resource: photo.sampler },
    ],
  });
}
```

And wire it in `main.ts`:

```ts
  initUi(params, async () => {
    photoNow = await loadPhoto(ctx.device, Date.now());
    rebuildBindGroup(ctx, pl, frameBuf, photoNow);
  });
```

Import `rebuildBindGroup`.

- [ ] **Step 4: Verify the panel**

```bash
bun run dev
```

Expected: Tweakpane panel top-right. Adjust `V_d` → rainbow widens as it drops. Adjust `edgeR` toward `thickness` → dome forms, dispersion fills the pill. Reload photo → new image, pills still visible. Stop server.

- [ ] **Step 5: `[COMMIT POINT]` Ask user before committing.**

Suggested message: `feat: Tweakpane UI for live parameter tuning`.

---

## Task 16: A/B compare hotkey (Z) + reset/randomize keys

**Files:**
- Modify: `src/main.ts`

- [ ] **Step 1: Add keyboard handlers in `main.ts`**

Above the RAF loop, after `initUi(...)`, add:

```ts
  let forceN3 = false;
  window.addEventListener('keydown', (e) => {
    if (e.key.toLowerCase() === 'z') forceN3 = true;
    if (e.key === ' ') {
      const cur = resizeCanvas(ctx.canvas, ctx.dpr);
      pills = defaultPills(cur.width, cur.height).map((p) => ({
        ...p,
        cx: Math.random() * cur.width,
        cy: Math.random() * cur.height,
      }));
      attachDrag(ctx.canvas, pills, ctx.dpr);  // fresh handlers for the new array
    }
    if (e.key.toLowerCase() === 'r') {
      (async () => {
        photoNow = await loadPhoto(ctx.device, Date.now());
        rebuildBindGroup(ctx, pl, frameBuf, photoNow);
      })();
    }
  });
  window.addEventListener('keyup', (e) => {
    if (e.key.toLowerCase() === 'z') forceN3 = false;
  });
```

Change `let pills: Pill[] = defaultPills(...)` to `let`-reassignable (it already is). But `attachDrag` holds a closure over the original array — on Space we need to re-attach. Doing so double-attaches listeners; fix by removing old listeners.

Refactor `attachDrag` in `src/pills.ts` to return a detach function. Full rewritten `src/pills.ts`:

```ts
export type Pill = {
  cx: number; cy: number; cz: number;
  hx: number; hy: number; hz: number;
  edgeR: number;
};

export function defaultPills(width: number, height: number, count = 4): Pill[] {
  const pills: Pill[] = [];
  const step = width / (count + 1);
  for (let i = 0; i < count; i++) {
    pills.push({
      cx: step * (i + 1),
      cy: height * 0.5 + (i % 2 === 0 ? -60 : 60),
      cz: 0,
      hx: 160, hy: 44, hz: 20,
      edgeR: 14,
    });
  }
  return pills;
}

type DragState = { pillIndex: number | null; offsetX: number; offsetY: number };

export function attachDrag(canvas: HTMLCanvasElement, pills: Pill[], dpr: number): () => void {
  const state: DragState = { pillIndex: null, offsetX: 0, offsetY: 0 };

  const toWorld = (e: PointerEvent) => {
    const r = canvas.getBoundingClientRect();
    return { x: (e.clientX - r.left) * dpr, y: (e.clientY - r.top) * dpr };
  };

  const findHit = (x: number, y: number): number => {
    for (let i = pills.length - 1; i >= 0; i--) {
      const p = pills[i]!;
      if (Math.abs(x - p.cx) <= p.hx && Math.abs(y - p.cy) <= p.hy) return i;
    }
    return -1;
  };

  const down = (e: PointerEvent) => {
    const { x, y } = toWorld(e);
    const i = findHit(x, y);
    if (i >= 0) {
      state.pillIndex = i;
      state.offsetX   = x - pills[i]!.cx;
      state.offsetY   = y - pills[i]!.cy;
      canvas.setPointerCapture(e.pointerId);
    }
  };
  const move = (e: PointerEvent) => {
    if (state.pillIndex === null) return;
    const { x, y } = toWorld(e);
    const p = pills[state.pillIndex]!;
    p.cx = x - state.offsetX;
    p.cy = y - state.offsetY;
  };
  const release = (e: PointerEvent) => {
    if (state.pillIndex !== null) {
      canvas.releasePointerCapture(e.pointerId);
      state.pillIndex = null;
    }
  };

  canvas.addEventListener('pointerdown', down);
  canvas.addEventListener('pointermove', move);
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);

  return () => {
    canvas.removeEventListener('pointerdown', down);
    canvas.removeEventListener('pointermove', move);
    canvas.removeEventListener('pointerup', release);
    canvas.removeEventListener('pointercancel', release);
  };
}
```

Update `main.ts` to track the detach function and re-attach on Space:

```ts
  let detach = attachDrag(ctx.canvas, pills, ctx.dpr);
  // ...
    if (e.key === ' ') {
      detach();
      const cur = resizeCanvas(ctx.canvas, ctx.dpr);
      pills = defaultPills(cur.width, cur.height).map((p) => ({
        ...p,
        cx: Math.random() * cur.width,
        cy: Math.random() * cur.height,
      }));
      detach = attachDrag(ctx.canvas, pills, ctx.dpr);
    }
```

And apply `forceN3` to `writeFrame`:

```ts
      sampleCount: forceN3 ? 3 : params.sampleCount,
```

- [ ] **Step 2: Verify hotkeys**

```bash
bun run dev
```

Expected:
- Hold `Z`: rainbow fringe becomes a blocky three-band ghosting. Release: smooth rainbow returns.
- Press Space: pills scatter randomly. Still draggable.
- Press `R`: photo reloads with a new image.

Stop server.

- [ ] **Step 3: `[COMMIT POINT]` Ask user before committing.**

Suggested message: `feat: Z toggles N=3 A/B, Space shuffles pills, R reloads photo`.

---

## Task 17: Temporal jitter via history texture

**Files:**
- Create: `src/webgpu/history.ts`
- Modify: `src/shaders/dispersion.wgsl`, `src/webgpu/pipeline.ts`, `src/main.ts`

The shader already takes a `jitter` uniform. Now we add a history texture that the shader writes-read-blends each frame, amortizing dispersion samples across frames.

- [ ] **Step 1: Write `src/webgpu/history.ts`**

```ts
export type History = {
  textures: [GPUTexture, GPUTexture];
  views:    [GPUTextureView, GPUTextureView];
  sampler:  GPUSampler;
  current:  0 | 1;
  width:    number;
  height:   number;
};

export function createHistory(device: GPUDevice, width: number, height: number): History {
  const make = () => device.createTexture({
    label: 'history',
    size:  [width, height, 1],
    format: 'rgba16float',
    usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  const textures: [GPUTexture, GPUTexture] = [make(), make()];
  const views: [GPUTextureView, GPUTextureView] = [
    textures[0].createView(),
    textures[1].createView(),
  ];
  const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  return { textures, views, sampler, current: 0, width, height };
}

export function resizeHistory(device: GPUDevice, h: History, width: number, height: number): History {
  if (h.width === width && h.height === height) return h;
  h.textures.forEach((t) => t.destroy());
  return createHistory(device, width, height);
}
```

- [ ] **Step 2: Update the render pipeline to render into history, then composite to swapchain**

This doubles the fragment work per frame unless we're careful. Simpler approach, cheap and correct: **render to history, then a lightweight copy-shader blits the current history into the swapchain.** Skip the copy-shader overhead by reading the latest history view in an identity fragment.

Actually the single-pass approach: bind the *previous* history as a read texture, output new color to the *new* history, and also write to the swapchain in the same fragment via a second color target. WebGPU supports multiple color attachments — we'll write both.

Change the render pipeline to have two targets (swapchain, history):

```ts
// pipeline.ts
// Add the history view as a second color target when we create the pipeline.
```

Update `src/webgpu/pipeline.ts`:

```ts
import type { GpuContext } from './device';
import type { PhotoTex } from '../photo';
import type { History } from './history';
import vsSrc from '../shaders/fullscreen.wgsl?raw';
import fsSrc from '../shaders/dispersion.wgsl?raw';

export type Pipeline = {
  pipeline:  GPURenderPipeline;
  bindGroup: GPUBindGroup;
};

export function createPipeline(
  ctx: GpuContext,
  frameBuf: GPUBuffer,
  photo: PhotoTex,
  historyRead: GPUTextureView,
  historySampler: GPUSampler,
): Pipeline {
  const { device, format } = ctx;
  const module = device.createShaderModule({ label: 'dispersion', code: vsSrc + '\n' + fsSrc });

  const pipeline = device.createRenderPipeline({
    label: 'dispersion-pipeline',
    layout: 'auto',
    vertex:   { module, entryPoint: 'vs_main' },
    fragment: {
      module,
      entryPoint: 'fs_main',
      targets: [
        { format },                 // swapchain
        { format: 'rgba16float' },  // history write
      ],
    },
    primitive: { topology: 'triangle-list' },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: frameBuf } },
      { binding: 1, resource: photo.texture.createView() },
      { binding: 2, resource: photo.sampler },
      { binding: 3, resource: historyRead },
      { binding: 4, resource: historySampler },
    ],
  });

  return { pipeline, bindGroup };
}

export function rebuildBindGroup(
  ctx: GpuContext,
  pl: Pipeline,
  frameBuf: GPUBuffer,
  photo: PhotoTex,
  historyRead: GPUTextureView,
  historySampler: GPUSampler,
): void {
  pl.bindGroup = ctx.device.createBindGroup({
    layout: pl.pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: frameBuf } },
      { binding: 1, resource: photo.texture.createView() },
      { binding: 2, resource: photo.sampler },
      { binding: 3, resource: historyRead },
      { binding: 4, resource: historySampler },
    ],
  });
}

export function draw(ctx: GpuContext, pl: Pipeline, historyWrite: GPUTextureView): void {
  const encoder = ctx.device.createCommandEncoder({ label: 'draw' });
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view:      ctx.context.getCurrentTexture().createView(),
        loadOp:    'clear',
        storeOp:   'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      },
      {
        view:      historyWrite,
        loadOp:    'clear',
        storeOp:   'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      },
    ],
  });
  pass.setPipeline(pl.pipeline);
  pass.setBindGroup(0, pl.bindGroup);
  pass.draw(3, 1, 0, 0);
  pass.end();
  ctx.device.queue.submit([encoder.finish()]);
}
```

- [ ] **Step 3: Update the shader to read history + output two targets**

In `src/shaders/dispersion.wgsl`, add bindings:

```wgsl
@group(0) @binding(3) var historyTex: texture_2d<f32>;
@group(0) @binding(4) var historySmp: sampler;
```

Change `fs_main` return type:

```wgsl
struct FsOut {
  @location(0) color:   vec4<f32>,
  @location(1) history: vec4<f32>,
};

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> FsOut {
  // ... existing body up to `let outRgb = mix(refrRgb, reflRgb, F);`

  let prev   = textureSample(historyTex, historySmp, uv).rgb;
  let alpha  = 0.2;          // EMA weight for the current frame
  let blend  = mix(prev, outRgb, alpha);

  var o: FsOut;
  o.color   = vec4<f32>(blend, 1.0);
  o.history = vec4<f32>(blend, 1.0);
  return o;
}
```

- [ ] **Step 4: Wire history in `src/main.ts`**

Replace the bootstrap section:

```ts
  const photo    = await loadPhoto(ctx.device);
  let photoNow   = photo;
  const initSize = resizeCanvas(ctx.canvas, ctx.dpr);
  let history    = createHistory(ctx.device, initSize.width, initSize.height);
  const frameBuf = createFrameBuffer(ctx.device);
  const pl       = createPipeline(ctx, frameBuf, photoNow,
                                  history.views[1 - history.current], history.sampler);

  let pills: Pill[] = defaultPills(initSize.width, initSize.height);
  let detach = attachDrag(ctx.canvas, pills, ctx.dpr);

  const params = defaultParams();
  initUi(params, async () => {
    photoNow = await loadPhoto(ctx.device, Date.now());
    rebuildBindGroup(ctx, pl, frameBuf, photoNow,
                     history.views[1 - history.current], history.sampler);
  });

  let forceN3 = false;
  window.addEventListener('keydown', (e) => {
    if (e.key.toLowerCase() === 'z') forceN3 = true;
    if (e.key === ' ') {
      detach();
      const cur = resizeCanvas(ctx.canvas, ctx.dpr);
      pills = defaultPills(cur.width, cur.height).map((p) => ({
        ...p,
        cx: Math.random() * cur.width,
        cy: Math.random() * cur.height,
      }));
      detach = attachDrag(ctx.canvas, pills, ctx.dpr);
    }
    if (e.key.toLowerCase() === 'r') {
      (async () => {
        photoNow = await loadPhoto(ctx.device, Date.now());
        rebuildBindGroup(ctx, pl, frameBuf, photoNow,
                         history.views[1 - history.current], history.sampler);
      })();
    }
  });
  window.addEventListener('keyup', (e) => {
    if (e.key.toLowerCase() === 'z') forceN3 = false;
  });

  const loop = () => {
    const { width, height } = resizeCanvas(ctx.canvas, ctx.dpr);
    const resized = resizeHistory(ctx.device, history, width, height);
    if (resized !== history) {
      history = resized;
      rebuildBindGroup(ctx, pl, frameBuf, photoNow,
                       history.views[1 - history.current], history.sampler);
    }

    for (const pill of pills) {
      pill.hx    = params.pillLen   / 2;
      pill.hy    = params.pillShort / 2;
      pill.hz    = params.pillThick / 2;
      pill.edgeR = Math.min(params.edgeR, pill.hz);
    }
    writeFrame(ctx.device, frameBuf, {
      resolution:         [width, height],
      photoSize:          [photoNow.width, photoNow.height],
      n_d:                params.n_d,
      V_d:                params.V_d,
      sampleCount:        forceN3 ? 3 : params.sampleCount,
      refractionStrength: params.refractionStrength,
      jitter:             params.temporalJitter ? Math.random() : 0.5,
      refractionMode:     params.refractionMode === 'exact' ? 0 : 1,
      pillCount:          pills.length,
      pills,
    });

    draw(ctx, pl, history.views[history.current]);
    // swap history parity: next frame reads what we just wrote
    history.current = (history.current === 0 ? 1 : 0);
    rebuildBindGroup(ctx, pl, frameBuf, photoNow,
                     history.views[1 - history.current], history.sampler);
    requestAnimationFrame(loop);
  };
```

Import `createHistory, resizeHistory`.

- [ ] **Step 5: Verify temporal jitter**

```bash
bun run dev
```

Expected: with temporal jitter enabled, rainbow fringe is noticeably smoother. Disable via panel: subtle banding appears. Enable again: smooths out. Stop server.

- [ ] **Step 6: `[COMMIT POINT]` Ask user before committing.**

Suggested message: `feat: temporal jitter via ping-pong history texture`.

---

## Task 18: Final visual verification pass

**Files:**
- No code changes.

- [ ] **Step 1: Run the full checklist from the spec's Testing section**

Start the dev server:

```bash
bun run dev
```

Walk through each item manually:
- [ ] Hold `Z` ↔ release: N=3 blocky ↔ N=8 smooth rainbow transition.
- [ ] Drop `V_d` to 20: rainbow widens dramatically.
- [ ] Raise `V_d` to 80: rainbow collapses to near-colorless.
- [ ] Drag a pill across a high-contrast edge: per-frame update, no stutter.
- [ ] Disable temporal jitter at N=8: faint banding visible. Re-enable: smooth.
- [ ] Resize the window: pills/photo reflow cleanly, no black flash > 1 frame.
- [ ] Push `edgeR` to `thickness/2`: dome emerges, dispersion fills the pill. Drop `edgeR`→1: rim halo only.
- [ ] Toggle refraction mode at N=16: Approx mode FPS noticeably higher (check browser dev tools FPS meter), visual difference subtle.

**Deferred polish (not required for MVP, call out to user after the checklist passes):** soft drop shadow under each pill, hover-raise effect, Beer-Lambert absorption color control, edge-highlight toggle. None of these are in the success criteria; skip unless the user asks.

- [ ] **Step 2: Test in target browsers**

Manually load http://localhost:5173 in:
- Chrome 120+
- Edge 120+
- Safari 18+

Confirm: no console errors, pill dispersion visible, UI responsive.

- [ ] **Step 3: Fix anything the checklist found**

Document any deviation from spec as a follow-up note; fix trivial issues inline.

- [ ] **Step 4: `[COMMIT POINT]` Ask user before final commit.**

Suggested message: `chore: visual verification pass — all spec checks green`.

---

## Execution Options

**Plan complete and saved to [docs/superpowers/plans/2026-04-21-real-spectral-dispersion-demo.md](docs/superpowers/plans/2026-04-21-real-spectral-dispersion-demo.md).** Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration with two-stage review.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review.

**Which approach?**
