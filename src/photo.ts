import { generateMipmaps, mipLevelsFor } from './webgpu/mipmap';

export type PhotoTex = {
  readonly texture: GPUTexture;
  readonly sampler: GPUSampler;
  readonly width:   number;
  readonly height:  number;
};

const PHOTO_FORMAT: GPUTextureFormat = 'rgba8unorm-srgb';

export async function loadPhoto(device: GPUDevice, seed = Date.now()): Promise<PhotoTex> {
  const url = `https://picsum.photos/seed/${seed}/1920/1080`;
  // Keep the catch tight to the network + decode path so GPU upload
  // errors (createTexture / copyExternalImageToTexture / mipmap gen)
  // bubble up to the uncapturederror handler instead of being disguised
  // as "photo fetch failed" and silently falling back to the gradient.
  let bitmap: ImageBitmap;
  try {
    const res = await fetch(url, { mode: 'cors' });
    if (!res.ok) throw new Error(`Photo fetch failed: ${res.status} ${res.statusText} (${url})`);
    const blob = await res.blob();
    bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none' });
  } catch (err) {
    console.error('[photo] fetch/decode failed, using gradient fallback:', err);
    return createGradientTexture(device);
  }
  return uploadBitmap(device, bitmap);
}

function uploadBitmap(device: GPUDevice, bitmap: ImageBitmap): PhotoTex {
  const width  = bitmap.width;
  const height = bitmap.height;
  const mipLevelCount = mipLevelsFor(width, height);
  const texture = device.createTexture({
    label:  'photo',
    size:   [width, height, 1],
    mipLevelCount,
    format: PHOTO_FORMAT,
    // RENDER_ATTACHMENT is required both by copyExternalImageToTexture and
    // by the mipmap-blit render passes that fill levels 1..N-1.
    usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture({ source: bitmap }, { texture }, [width, height, 1]);
  bitmap.close();
  generateMipmaps(device, texture, PHOTO_FORMAT, mipLevelCount);
  return { texture, sampler: sharedSampler(device), width, height };
}

// Fallback: a 256×256 vertical gradient that still exercises refraction/dispersion
// when the photo fetch fails.
function createGradientTexture(device: GPUDevice): PhotoTex {
  const W = 256;
  const H = 256;
  const bytes = new Uint8Array(W * H * 4);
  for (let y = 0; y < H; y++) {
    const t = y / (H - 1);
    const r = Math.round(60 + 180 * (1 - t));
    const g = Math.round(80 + 140 * Math.abs(0.5 - t) * 2);
    const b = Math.round(180 - 120 * (1 - t));
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      bytes[i + 0] = r;
      bytes[i + 1] = g;
      bytes[i + 2] = b;
      bytes[i + 3] = 255;
    }
  }
  const mipLevelCount = mipLevelsFor(W, H);
  const texture = device.createTexture({
    label:  'photo-fallback',
    size:   [W, H, 1],
    mipLevelCount,
    format: PHOTO_FORMAT,
    usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.writeTexture({ texture }, bytes, { bytesPerRow: W * 4 }, [W, H, 1]);
  generateMipmaps(device, texture, PHOTO_FORMAT, mipLevelCount);
  return { texture, sampler: sharedSampler(device), width: W, height: H };
}

let cachedSampler: GPUSampler | null = null;
function sharedSampler(device: GPUDevice): GPUSampler {
  if (cachedSampler) return cachedSampler;
  cachedSampler = device.createSampler({
    magFilter:    'linear',
    minFilter:    'linear',
    // Trilinear filtering across the mip chain softens refracted UV
    // aliasing at grazing angles — the shader hands us an explicit LOD
    // per wavelength tap (see dispersion.wgsl: photoLod).
    mipmapFilter: 'linear',
    // Strong refraction pushes sampled UVs well past [0,1]. `mirror-repeat`
    // folds the content back seamlessly — no visible tile seam (like `repeat`)
    // and no edge smear (like `clamp-to-edge`).
    addressModeU: 'mirror-repeat',
    addressModeV: 'mirror-repeat',
  });
  return cachedSampler;
}

export function destroyPhoto(p: PhotoTex): void {
  p.texture.destroy();
}
