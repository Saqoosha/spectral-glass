import { initGpu, resizeCanvas } from './webgpu/device';
import { createPipeline, draw, rebuildBindGroup } from './webgpu/pipeline';
import { createFrameBuffer, writeFrame } from './webgpu/uniforms';
import { loadPhoto } from './photo';
import { attachDrag, defaultPills, type Pill } from './pills';
import { defaultParams, initUi } from './ui';
import { createHistory, resizeHistory, type History } from './webgpu/history';

function readView(h: History): GPUTextureView {
  return h.views[h.current === 0 ? 1 : 0];
}

async function main() {
  const ctx = await initGpu('gpu');
  if (!ctx) {
    document.getElementById('fallback')?.classList.add('visible');
    document.getElementById('gpu')?.setAttribute('style', 'display:none');
    return;
  }
  let photoNow  = await loadPhoto(ctx.device);
  const frameBuf = createFrameBuffer(ctx.device);

  const initSize = resizeCanvas(ctx.canvas, ctx.dpr);
  let history = createHistory(ctx.device, initSize.width, initSize.height);
  const pl    = createPipeline(ctx, frameBuf, photoNow,
                               readView(history), history.sampler);

  let pills: Pill[] = defaultPills(initSize.width, initSize.height);
  let detach = attachDrag(ctx.canvas, pills, ctx.dpr);

  const params = defaultParams();
  initUi(params, async () => {
    photoNow = await loadPhoto(ctx.device, Date.now());
    rebuildBindGroup(ctx, pl, frameBuf, photoNow,
                     readView(history), history.sampler);
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
                         readView(history), history.sampler);
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
                       readView(history), history.sampler);
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
    history.current = (history.current === 0 ? 1 : 0);
    rebuildBindGroup(ctx, pl, frameBuf, photoNow,
                     readView(history), history.sampler);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

main().catch((err) => {
  console.error(err);
  document.getElementById('fallback')?.classList.add('visible');
});
