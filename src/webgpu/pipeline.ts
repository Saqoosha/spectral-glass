import type { GpuContext } from './device';
import type { PhotoTex } from '../photo';
import type { History } from './history';
import type { Intermediate } from './postprocess';
import vsSrc from '../shaders/fullscreen.wgsl?raw';
import fsSrc from '../shaders/dispersion.wgsl?raw';
import { diamondWgslConstants } from '../math/diamond';

export type Pipeline = {
  /** Fullscreen bg pass: cheap photo+history blend, covers every pixel. */
  readonly bg:    GPURenderPipeline;
  /** Per-pill proxy pass: instanced 3D cube mesh (12 tris, 36 verts),
   *  optionally rotated for shape==cube. The heavy refraction shader runs
   *  only on fragments inside the proxy silhouette — for the default 4-pill
   *  layout that's ~25 % of screen; scales with on-screen shape area. */
  readonly proxy: GPURenderPipeline;
  bindGroups:     [GPUBindGroup, GPUBindGroup];  // index = history read slot (1 - current)
};

export async function createPipeline(
  ctx: GpuContext,
  frameBuf: GPUBuffer,
  photo: PhotoTex,
  history: History,
): Promise<Pipeline> {
  const { device } = ctx;
  // Prepend the diamond const block (Tolkowsky-derived plane normals +
  // offsets, generated in src/math/diamond.ts) so `sdfDiamond` reads the same
  // numbers the TS side computed. Injecting here rather than hand-copying
  // them into the WGSL source keeps TS as the single source of truth.
  const module = device.createShaderModule({
    label: 'dispersion',
    code:  diamondWgslConstants() + vsSrc + '\n' + fsSrc,
  });

  // Surface WGSL diagnostics immediately — the default WebGPU path swallows compile
  // errors and only reports them at pipeline-creation time as opaque validation errors.
  const info = await module.getCompilationInfo();
  for (const m of info.messages) {
    const line = `[WGSL ${m.type}] line ${m.lineNum}:${m.linePos}: ${m.message}`;
    if (m.type === 'error') console.error(line);
    else if (m.type === 'warning') console.warn(line);
    else console.info(line);
  }
  if (info.messages.some((m) => m.type === 'error')) {
    throw new Error('WGSL shader compile failed — see console for diagnostics');
  }

  // Both color targets are rgba16float now:
  //   @location(0) → post-process intermediate (linear; FXAA/passthrough
  //                  emits the display-encoded swapchain output)
  //   @location(1) → history ping-pong (linear)
  // This keeps every color path linear and defers sRGB encoding to the
  // post pass, so FXAA can do perceptual edge detection on the same data.
  const targets: GPUColorTargetState[] = [
    { format: 'rgba16float' },
    { format: 'rgba16float' },
  ];

  // Explicit bind group layout so both pipelines share it AND `frame` is
  // visible to the vertex stage too (vs_proxy reads `frame.pills` and
  // `frame.resolution`). The `auto` layout derived from the bg pass would
  // only mark `frame` as fragment-visible, mismatching the proxy pipeline.
  const bindGroupLayout = device.createBindGroupLayout({
    label: 'frame-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

  // `createRenderPipelineAsync` surfaces validation errors as a rejection,
  // unlike the sync variant which returns an "invalid" pipeline that only
  // explodes at setPipeline time.
  const bg = await device.createRenderPipelineAsync({
    label: 'bg-pipeline',
    layout: pipelineLayout,
    vertex:   { module, entryPoint: 'vs_main' },
    fragment: { module, entryPoint: 'fs_bg', targets },
    primitive: { topology: 'triangle-list' },
  });

  const proxy = await device.createRenderPipelineAsync({
    label: 'proxy-pipeline',
    layout: pipelineLayout,
    vertex:   { module, entryPoint: 'vs_proxy' },
    fragment: { module, entryPoint: 'fs_main', targets },
    // `back` culling plus CCW-outward winding in CUBE_VERTS gives one fragment
    // shader invocation per covered pixel. Without this, every pixel inside
    // the rotated proxy would be shaded twice (front + back face).
    // CCW-outward 3D winding projects to CW in NDC because our camera looks
    // down -Z with NDC-up mapped to world-down (DOM top-origin). So front
    // faces present CW in screen space; `frontFace: 'cw'` + `cullMode: 'back'`
    // keeps exactly one fragment per covered pixel.
    primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'cw' },
  });

  return {
    bg,
    proxy,
    bindGroups: buildBindGroups(ctx, bg, frameBuf, photo, history),
  };
}

function buildBindGroups(
  ctx: GpuContext,
  pipeline: GPURenderPipeline,
  frameBuf: GPUBuffer,
  photo: PhotoTex,
  history: History,
): [GPUBindGroup, GPUBindGroup] {
  const layout = pipeline.getBindGroupLayout(0);
  const makeFor = (readIndex: 0 | 1): GPUBindGroup => ctx.device.createBindGroup({
    label: `frame-bind-${readIndex}`,
    layout,
    entries: [
      { binding: 0, resource: { buffer: frameBuf } },
      { binding: 1, resource: photo.texture.createView() },
      { binding: 2, resource: photo.sampler },
      { binding: 3, resource: history.views[readIndex] },
      { binding: 4, resource: history.sampler },
    ],
  });
  return [makeFor(0), makeFor(1)];
}

export function rebuildBindGroups(
  ctx: GpuContext,
  pl: Pipeline,
  frameBuf: GPUBuffer,
  photo: PhotoTex,
  history: History,
): void {
  pl.bindGroups = buildBindGroups(ctx, pl.bg, frameBuf, photo, history);
}

/** Encode the scene pass (bg + proxy) into the given command encoder.
 *  Writes to `intermediate` at @location(0) and history at @location(1).
 *  Caller submits the encoder after optionally encoding the post pass. */
export function encodeScene(
  pl:           Pipeline,
  history:      History,
  intermediate: Intermediate,
  pillCount:    number,
  encoder:      GPUCommandEncoder,
  timestampWrites?: GPURenderPassTimestampWrites,
): void {
  const readIndex: 0 | 1 = history.current === 0 ? 1 : 0;
  const writeView = history.views[history.current];
  const pass = encoder.beginRenderPass({
    label: 'scene-pass',
    colorAttachments: [
      {
        view:       intermediate.view,
        loadOp:     'clear',
        storeOp:    'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      },
      {
        view:    writeView,
        loadOp:  'load',  // fragment writes every pixel; 'load' skips the clear cost
        storeOp: 'store',
      },
    ],
    ...(timestampWrites ? { timestampWrites } : {}),
  });
  // Pass 1: fullscreen bg (cheap). Writes every pixel.
  pass.setPipeline(pl.bg);
  pass.setBindGroup(0, pl.bindGroups[readIndex]);
  pass.draw(3, 1, 0, 0);
  // Pass 2: per-pill 3D cube proxy meshes (heavy). Covers only the actual
  // on-screen shape silhouette — for the default 4-pill layout that's ~25 %,
  // scales with shape size. Reuses the bind group set before the bg pass
  // because both pipelines share `pipelineLayout`.
  if (pillCount > 0) {
    pass.setPipeline(pl.proxy);
    pass.draw(36, pillCount, 0, 0);
  }
  pass.end();
}
