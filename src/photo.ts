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
  const width  = bitmap.width;
  const height = bitmap.height;

  const texture = device.createTexture({
    label:  'photo',
    size:   [width, height, 1],
    format: 'rgba8unorm-srgb',
    usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture(
    { source: bitmap },
    { texture },
    [width, height, 1],
  );
  bitmap.close();

  const sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });
  return { texture, sampler, width, height };
}
