import type { GpuContext } from './device';
import type { AaMode } from '../ui';
import postSrc from '../shaders/postprocess.wgsl?raw';

/** Intermediate render target that the scene (bg + proxy) writes linear
 *  rgba16float into. The post-process pass samples this and emits the
 *  display-encoded color to the swapchain. Same size as the canvas.
 *  All fields are readonly — on resize, `resizeIntermediate` replaces
 *  the whole `post.intermediate` wholesale rather than mutating dims. */
export type Intermediate = {
  readonly texture: GPUTexture;
  readonly view:    GPUTextureView;
  readonly width:   number;
  readonly height:  number;
};

/** Post-process pipelines + their shared resources.
 *  - `passthrough` is used for aaMode === 'none' and 'taa' (TAA already
 *    ran in the scene pass, so post just copies + encodes).
 *  - `fxaa` is used for aaMode === 'fxaa'.
 *  Both read from `intermediate` via `bindGroup`, which is rebuilt on
 *  canvas resize.
 *
 *  Only `intermediate` and `bindGroup` change after construction (resize
 *  rebuilds them in lockstep). The remaining wiring — pipelines, UBO,
 *  bgLayout, sampler — is fixed for the lifetime of the GPUDevice. */
export type PostProcess = {
  readonly passthrough:  GPURenderPipeline;
  readonly fxaa:         GPURenderPipeline;
  bindGroup:             GPUBindGroup;
  intermediate:          Intermediate;
  readonly postBuf:      GPUBuffer;
  readonly bgLayout:     GPUBindGroupLayout;
  readonly sampler:      GPUSampler;
};

const POST_FLOATS = 4;               // 16 B UBO (applySrgbOetf + 3 pads)
const POST_BYTES  = POST_FLOATS * 4;
const postScratch = new Float32Array(POST_FLOATS);

function createIntermediate(device: GPUDevice, width: number, height: number): Intermediate {
  const texture = device.createTexture({
    label:  'post-intermediate',
    size:   [width, height, 1],
    format: 'rgba16float',
    // TEXTURE_BINDING for the post pass to sample; RENDER_ATTACHMENT so
    // the scene pipelines can draw into it at @location(0).
    usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  return { texture, view: texture.createView(), width, height };
}

function buildBindGroup(
  device:    GPUDevice,
  layout:    GPUBindGroupLayout,
  postBuf:   GPUBuffer,
  sampler:   GPUSampler,
  interView: GPUTextureView,
): GPUBindGroup {
  return device.createBindGroup({
    label: 'post-bind',
    layout,
    entries: [
      { binding: 0, resource: { buffer: postBuf } },
      { binding: 1, resource: sampler },
      { binding: 2, resource: interView },
    ],
  });
}

export async function createPostProcess(
  ctx:    GpuContext,
  width:  number,
  height: number,
): Promise<PostProcess> {
  const { device, format } = ctx;
  const module = device.createShaderModule({ label: 'post', code: postSrc });

  const info = await module.getCompilationInfo();
  for (const m of info.messages) {
    const line = `[WGSL ${m.type}] line ${m.lineNum}:${m.linePos}: ${m.message}`;
    if (m.type === 'error') console.error(line);
    else if (m.type === 'warning') console.warn(line);
    else console.info(line);
  }
  if (info.messages.some((m) => m.type === 'error')) {
    throw new Error('WGSL post-process shader compile failed — see console');
  }

  const bgLayout = device.createBindGroupLayout({
    label: 'post-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer:  { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgLayout] });

  const targets: GPUColorTargetState[] = [{ format }];
  const makePipeline = (label: string, entryPoint: string) => device.createRenderPipelineAsync({
    label,
    layout:    pipelineLayout,
    vertex:    { module, entryPoint: 'vs' },
    fragment:  { module, entryPoint, targets },
    primitive: { topology: 'triangle-list' },
  });
  const [passthrough, fxaa] = await Promise.all([
    makePipeline('post-passthrough', 'fs_passthrough'),
    makePipeline('post-fxaa',        'fs_fxaa'),
  ]);

  const postBuf = device.createBuffer({
    label: 'post-frame',
    size:  POST_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const sampler = device.createSampler({
    label:     'post-linear',
    magFilter: 'linear',
    minFilter: 'linear',
    // clamp so FXAA's neighbour taps at the screen edge don't wrap.
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });

  const intermediate = createIntermediate(device, width, height);
  const bindGroup    = buildBindGroup(device, bgLayout, postBuf, sampler, intermediate.view);

  return { passthrough, fxaa, bindGroup, intermediate, postBuf, bgLayout, sampler };
}

/** Reallocate the intermediate texture on canvas resize and rebuild the
 *  bind group that points at it. Mutates `post` in place. Returns true
 *  when a realloc happened (so the caller can log or notify).
 *
 *  Destroy timing: the old texture is freed synchronously. WebGPU holds
 *  a strong reference from any command buffer that already named it as
 *  a color attachment, so pending work is unaffected — the new frame's
 *  encoder will see the replacement `post.intermediate` and `post.bindGroup`
 *  (rebuilt below in lockstep). Photo reload waits for
 *  `queue.onSubmittedWorkDone` before destroying because the photo texture
 *  is also read by the next frame's bg sampler; the intermediate only
 *  ever feeds the post pass in the same frame, so no drain is needed. */
export function resizeIntermediate(
  device: GPUDevice,
  post:   PostProcess,
  width:  number,
  height: number,
): boolean {
  if (post.intermediate.width === width && post.intermediate.height === height) return false;
  post.intermediate.texture.destroy();
  post.intermediate = createIntermediate(device, width, height);
  post.bindGroup    = buildBindGroup(device, post.bgLayout, post.postBuf, post.sampler, post.intermediate.view);
  return true;
}

export function writePostFrame(device: GPUDevice, post: PostProcess, applySrgbOetf: boolean): void {
  postScratch[0] = applySrgbOetf ? 1 : 0;
  // Slots 1..3 are padding kept at 0. `postScratch` is zero-initialised
  // once at module scope and only slot 0 is ever rewritten, so no
  // per-call `fill(0)` is needed. If a future field starts varying,
  // mirror `uniforms.ts writeFrame` and add an explicit reset.
  device.queue.writeBuffer(post.postBuf, 0, postScratch);
}

/** Encode the post-process pass into an existing command encoder.
 *  Called after the scene pass (which has written to `intermediate`).
 *  Writes to the swapchain. */
export function encodePost(
  ctx:     GpuContext,
  post:    PostProcess,
  encoder: GPUCommandEncoder,
  aaMode:  AaMode,
): void {
  const pipeline = aaMode === 'fxaa' ? post.fxaa : post.passthrough;
  const pass = encoder.beginRenderPass({
    label: 'post-pass',
    colorAttachments: [{
      view:       ctx.context.getCurrentTexture().createView(),
      loadOp:     'clear',
      storeOp:    'store',
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
    }],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, post.bindGroup);
  pass.draw(3, 1, 0, 0);
  pass.end();
}
