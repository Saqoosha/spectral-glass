import type { PhotoTex } from './photo';
import { getPhotoDisplaySampler } from './photo';

const PHOTO_FORMAT: GPUTextureFormat = 'rgba8unorm-srgb';

/** `device.queue.copyElementImageToTexture` from the HTML-in-Canvas / CanvasDrawElement trial. */
export function supportsHtmlInCanvas(device: GPUDevice): boolean {
  const q = device.queue as GPUQueue;
  return typeof q.copyElementImageToTexture === 'function';
}

/** The snapshot source must be a direct child of the WebGPU `canvas` (WICG constraint). */
export function isValidHtmlBgLayer(
  canvas: HTMLCanvasElement,
  layer: HTMLElement,
): boolean {
  return layer.parentElement === canvas;
}

export function createHtmlBackgroundTexture(
  device: GPUDevice,
  width:  number,
  height: number,
): PhotoTex {
  const w = Math.max(1, Math.floor(width));
  const h = Math.max(1, Math.floor(height));
  const texture = device.createTexture({
    label:  'html-bg',
    size:   [w, h, 1],
    // Single mip: dynamic uploads every paint — no mipmap pass.
    mipLevelCount: 1,
    format: PHOTO_FORMAT,
    // COPY_DST + RENDER_ATTACHMENT: Dawn requires both for copyElementImageToTexture
    // (same pattern as copyExternalImageToTexture / photo.ts mipmap path).
    usage:  GPUTextureUsage.TEXTURE_BINDING
      | GPUTextureUsage.COPY_DST
      | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  return {
    texture,
    sampler: getPhotoDisplaySampler(device),
    width:  w,
    height: h,
  };
}

export function destroyHtmlBackgroundTexture(p: PhotoTex): void {
  p.texture.destroy();
}

/**
 * Call only from a canvas `paint` event handler (or the first copy may throw).
 * Copys the element subtree raster into the destination texture.
 */
export function copyHtmlLayerToTexture(
  queue: GPUQueue,
  layer: HTMLElement,
  dest:  GPUTexture,
): void {
  const copy = (queue as GPUQueue).copyElementImageToTexture;
  if (typeof copy !== 'function') return;
  const destTagged: GPUImageCopyTextureTagged = {
    texture:            dest,
    mipLevel:           0,
    origin:             { x: 0, y: 0, z: 0 },
    // Match copyExternalImageToTexture defaults for 8-bit sRGB text/UI.
    colorSpace:         'srgb',
    premultipliedAlpha: false,
  };
  try {
    copy.call(queue, layer, destTagged);
  } catch (err) {
    console.warn('[html-bg] copyElementImageToTexture failed:', err);
  }
}
