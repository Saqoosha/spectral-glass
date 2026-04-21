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
