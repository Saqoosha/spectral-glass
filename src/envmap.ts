import { decodeHdr } from './hdr';

/**
 * HDR environment map loader + GPU uploader.
 *
 * Fetches a Radiance .hdr panorama (equirectangular projection,
 * linear-RGB float), converts to half-precision floats on the host,
 * and uploads to an rgba16float texture for linear-filtered IBL
 * sampling in the shader. Mirrors src/photo.ts's shape — lifecycle
 * owned by main.ts, destroy() on replacement.
 *
 * Why rgba16f rather than rgba32f:
 *   - rgba16f supports linear filtering without the `float32-filterable`
 *     feature that not every adapter exposes (Mac discrete GPUs in
 *     particular).
 *   - Half of the memory footprint (4 MB vs 8 MB for a 1024×512 HDRI),
 *     which matters because users may hop between HDRIs via the random
 *     picker and the browser hangs on to a few cached textures.
 *   - HDR highlights clip at fp16's ~65504 max, which is FAR past the
 *     brightest sun values in typical HDRIs (peak ~1000-2000) — no
 *     visible range loss in practice.
 *
 * Layout: equirectangular; row 0 is the top of the sky (+Y), row
 * (height-1) is the bottom (-Y). Column 0 is one edge of the ±π
 * longitude wrap — shader samples with `addressModeU = 'repeat'` so the
 * seam is invisible.
 */
/** Branded so TypeScript treats `EnvmapTex` and `PhotoTex` as distinct
 *  types even though they share the same runtime shape. Catches the
 *  silent-swap bug where `rebuildBindGroups(ctx, pl, frameBuf, photo,
 *  envmap, history)` or `createPipeline(…, photo, envmap, history)`
 *  gets its two `GPUTexture`-bearing arguments transposed — which
 *  would bind the photo (rgba8unorm-srgb, mirror-repeat, mipmapped)
 *  into the envmap slot and vice versa, producing plausible-but-wrong
 *  output with no compile-time warning under structural typing. */
export type EnvmapTex = {
  readonly __brand: 'envmap';
  readonly texture: GPUTexture;
  readonly sampler: GPUSampler;
  readonly width:   number;
  readonly height:  number;
};

const ENVMAP_FORMAT: GPUTextureFormat = 'rgba16float';

/** Fetch a `.hdr` from `url`, decode, upload to GPU. Throws on fetch or
 *  decode failure — callers should catch and fall back to a default
 *  envmap (e.g. re-load the bundled startup HDRI or disable envmap
 *  sampling until the user picks a different one). */
export async function loadEnvmap(device: GPUDevice, url: string): Promise<EnvmapTex> {
  const res = await fetch(url, { mode: 'cors' });
  if (!res.ok) {
    throw new Error(`Envmap fetch failed: ${res.status} ${res.statusText} (${url})`);
  }
  const bytes = new Uint8Array(await res.arrayBuffer());
  const img   = decodeHdr(bytes);
  return uploadEnvmap(device, img.width, img.height, img.rgb);
}

/** Build an rgba16f texture from interleaved-RGB float data. Exported
 *  for tests that construct synthetic images (gradient fallbacks, etc.)
 *  and for main.ts to ship a tiny built-in default without a network
 *  round-trip on first paint. */
export function uploadEnvmap(
  device: GPUDevice,
  width:  number,
  height: number,
  rgb:    Float32Array,
): EnvmapTex {
  if (rgb.length !== width * height * 3) {
    throw new Error(
      `Envmap upload: rgb length=${rgb.length} does not match ${width}×${height}×3.`,
    );
  }
  // RGB floats → RGBA half-floats. Alpha = 1 so alpha-blended debug
  // draws don't render envmap pixels as transparent.
  //
  // HDR source pixels can exceed fp16's max finite value (65504) —
  // unclipped sun discs in outdoor HDRIs commonly carry 10⁴-10⁵ values,
  // and some studio strip lights hit that range too. Letting those
  // overflow to +Inf looks fine on a single texel but POISONS the
  // linear filter's weighted average: `0 * Inf = NaN` per IEEE 754,
  // so any sample near (but not on) an Inf texel can come out as NaN
  // and render as a black dot against the bright area. Clamp to
  // `F16_MAX_FINITE` first so the filter stays well-defined
  // everywhere; the tiny loss at the very brightest highlights is
  // imperceptible next to downstream exposure scaling + sRGB OETF
  // clamping.
  const pixelCount = width * height;
  const half = new Uint16Array(pixelCount * 4);
  for (let p = 0; p < pixelCount; p++) {
    half[p * 4 + 0] = f32ToF16(clampFiniteF16(rgb[p * 3 + 0]!));
    half[p * 4 + 1] = f32ToF16(clampFiniteF16(rgb[p * 3 + 1]!));
    half[p * 4 + 2] = f32ToF16(clampFiniteF16(rgb[p * 3 + 2]!));
    half[p * 4 + 3] = F16_ONE;
  }

  const texture = device.createTexture({
    label:  'envmap',
    size:   [width, height, 1],
    format: ENVMAP_FORMAT,
    // No mip chain here: the envmap is sampled at LOD 0 for sharp
    // reflections. A future Phase-D PMREM-style pre-convolution would
    // need mipmaps + explicit LOD per roughness, but that's overkill
    // for mirror-smooth diamond/cube/plate reflection — the sample
    // cost is what dominates, not filter smoothness.
    usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  // bytesPerRow: 4 channels × 2 bytes (half-float) × width. Must be a
  // multiple of 256 per the WebGPU spec; widths we care about (1K=1024,
  // 2K=2048) produce 8192 / 16384 which are already aligned. Still
  // worth adding the guard so an odd future size crashes loudly
  // instead of uploading garbage.
  const bytesPerRow = width * 4 * 2;
  if (bytesPerRow % 256 !== 0) {
    throw new Error(
      `Envmap upload: width=${width} gives bytesPerRow=${bytesPerRow}, not a multiple of 256`,
    );
  }
  device.queue.writeTexture(
    { texture },
    half,
    { bytesPerRow, rowsPerImage: height },
    [width, height, 1],
  );
  return {
    __brand: 'envmap',
    texture,
    sampler: sharedEnvmapSampler(device),
    width,
    height,
  };
}

/** Build a tiny synthetic HDR gradient — used as the initial / fallback
 *  envmap so the shader always has SOMETHING to sample (rgba16f with
 *  view-dependent gradient). Far from physically correct but keeps the
 *  rendering coherent while the first real HDRI is fetching. */
export function createDefaultEnvmap(device: GPUDevice): EnvmapTex {
  // Small 32×16 — negligible memory, fast to upload. The gradient runs
  // bright blue at the top (zenith), warm amber at the horizon, dark
  // green at the bottom (ground) so the "reflected environment" on
  // diamonds reads as plausibly outdoorsy until the real HDRI arrives.
  const width = 32;
  const height = 16;
  const rgb = new Float32Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    const t = y / (height - 1);
    // t = 0 → top (sky), t = 1 → bottom (ground).
    const r = (1 - t) * 0.3 + t * 0.15;
    const g = (1 - t) * 0.5 + t * 0.35;
    const b = (1 - t) * 1.2 + t * 0.10;
    const horizonBoost = 1.0 + 0.8 * Math.exp(-Math.pow((t - 0.55) * 8, 2));
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 3;
      rgb[i + 0] = r * horizonBoost;
      rgb[i + 1] = g * horizonBoost;
      rgb[i + 2] = b * horizonBoost;
    }
  }
  return uploadEnvmap(device, width, height, rgb);
}

export function destroyEnvmap(e: EnvmapTex): void {
  e.texture.destroy();
}

// ---- Internals ----

/**
 * IEEE 754 binary32 → binary16 conversion.
 *
 * Subnormals and small-exponent values collapse to zero — we lose some
 * near-zero precision but save the 10-line denormal shift path. For
 * HDR envmap use (values concentrated around 0.01–1000), this is
 * invisible; the true "dark" pixels are already zero or quantisation-
 * floor-clamped.
 *
 * NaN / ±Inf are carried through as fp16 NaN / ±Inf respectively so
 * they don't silently turn into finite poison values downstream.
 */
const F32_SCRATCH_BUF = new ArrayBuffer(4);
const F32_SCRATCH_F32 = new Float32Array(F32_SCRATCH_BUF);
const F32_SCRATCH_U32 = new Uint32Array(F32_SCRATCH_BUF);

function f32ToF16(v: number): number {
  F32_SCRATCH_F32[0] = v;
  const x    = F32_SCRATCH_U32[0]!;
  const sign = (x >>> 16) & 0x8000;
  let   exp  = (x >>> 23) & 0xff;
  let   mant = x & 0x7fffff;

  if (exp === 0xff) {
    // NaN / Inf: preserve with fp16's max exponent.
    return sign | 0x7c00 | (mant !== 0 ? 1 : 0);
  }
  // Re-bias from 127 to 15.
  exp = exp - 127 + 15;
  if (exp >= 31) return sign | 0x7c00;   // Overflow → ±Inf
  if (exp <= 0)  return sign;            // Underflow → ±0 (see doc above)

  // Normalised: 10-bit mantissa (drop the low 13 bits). Plain truncation
  // — not round-to-even — is acceptable here because HDR envmap data
  // sits far above the mantissa-LSB quantum for any visible pixel.
  return sign | (exp << 10) | (mant >>> 13);
}

/** fp16 representation of 1.0 — precomputed to shave one function call
 *  per alpha write in `uploadEnvmap`. */
const F16_ONE = 0x3c00;

/** Largest finite fp16 value. Sources:
 *    sign=0, exp=30 (max finite biased), mant=1023 (max)
 *    → (1 + 1023/1024) * 2^(30-15) = 65504
 *  Clamping inputs to ±this before the f32→f16 cast keeps the upload
 *  output fully finite, which the downstream linear sampler needs to
 *  avoid NaN leakage across bright-pixel edges. */
const F16_MAX_FINITE = 65504;

export function clampFiniteF16(v: number): number {
  // Also guard against NaN input — some HDR sources emit NaN for
  // oversaturated pixels. self-compare is the canonical NaN test.
  if (v !== v) return 0;
  if (v > F16_MAX_FINITE)  return F16_MAX_FINITE;
  if (v < -F16_MAX_FINITE) return -F16_MAX_FINITE;
  return v;
}

/** Exported for regression tests. `F16_MAX_FINITE` is the canonical
 *  "largest finite fp16" reference used by `clampFiniteF16`. */
export const F16_MAX_FINITE_VALUE = F16_MAX_FINITE;

// Shared sampler keyed on device, mirroring photo.ts's pattern. A
// WeakMap keeps the code consistent across hypothetical device-lost
// reinit even though the current app treats device-lost as fatal.
const samplerCache = new WeakMap<GPUDevice, GPUSampler>();
function sharedEnvmapSampler(device: GPUDevice): GPUSampler {
  let sampler = samplerCache.get(device);
  if (sampler) return sampler;
  sampler = device.createSampler({
    label:        'envmap',
    magFilter:    'linear',
    minFilter:    'linear',
    mipmapFilter: 'nearest',   // no mip chain — see texture creation
    // Longitude wraps ±π, so U repeats; V clamps so sky/ground don't
    // bleed across the poles when a ray grazes y≈±1.
    addressModeU: 'repeat',
    addressModeV: 'clamp-to-edge',
  });
  samplerCache.set(device, sampler);
  return sampler;
}
