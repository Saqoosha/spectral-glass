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
