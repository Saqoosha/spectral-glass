import { initGpu, resizeCanvas, needsSrgbOetf } from './webgpu/device';
import { createPipeline, draw, rebuildBindGroups } from './webgpu/pipeline';
import { createFrameBuffer, writeFrame } from './webgpu/uniforms';
import { createPerf } from './webgpu/perf';
import { loadPhoto, destroyPhoto } from './photo';
import { attachDrag, defaultPills, type Pill } from './pills';
import { defaultParams, initUi, mergeParams, type Params } from './ui';
import { cameraZForFov } from './math/camera';
import { createHistory, resizeHistory } from './webgpu/history';
import { loadStored, debouncedSaver } from './persistence';
import { createPerfStats, makeFrameTimer } from './perfStats';

function showFatal(message: string): void {
  const fb = document.getElementById('fallback');
  if (fb) {
    fb.textContent = message;
    fb.classList.add('visible');
  }
  document.getElementById('gpu')?.setAttribute('style', 'display:none');
}

function isTypingTarget(t: EventTarget | null): boolean {
  if (!(t instanceof HTMLElement)) return false;
  const tag = t.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || t.isContentEditable;
}

// Must match the WGSL `shape` uniform branches in dispersion.wgsl.
const SHAPE_ID: Record<Params['shape'], number> = {
  pill:  0,
  prism: 1,
  cube:  2,
};

// Must match the WGSL `projection` uniform branches (0 = ortho, 1 = perspective).
const PROJECTION_ID: Record<Params['projection'], number> = {
  ortho:       0,
  perspective: 1,
};

/** Non-fatal user-visible notice — e.g. photo reload failed. Auto-hides. */
function showNotice(message: string, durationMs = 3000): void {
  const fb = document.getElementById('fallback');
  if (!fb) return;
  fb.textContent = message;
  fb.classList.add('visible');
  window.setTimeout(() => fb.classList.remove('visible'), durationMs);
}

async function main(): Promise<void> {
  const init = await initGpu('gpu', showFatal);
  if ('kind' in init) {
    const messages: Record<typeof init.kind, string> = {
      'no-webgpu':  "This demo needs a WebGPU-capable browser (Chrome/Edge 120+ or Safari 18+).",
      'no-adapter': 'No GPU adapter available. Try reloading or updating your GPU driver.',
      'no-context': 'Failed to create a WebGPU canvas context.',
    };
    showFatal(messages[init.kind]);
    return;
  }
  const ctx = init;

  const frameBuf = createFrameBuffer(ctx.device);
  let photoNow   = await loadPhoto(ctx.device);

  const initSize = resizeCanvas(ctx.canvas, ctx.dpr);
  let history    = createHistory(ctx.device, initSize.width, initSize.height);
  const pl       = await createPipeline(ctx, frameBuf, photoNow, history);

  const stored = loadStored();
  const params = mergeParams(defaultParams(), stored?.params ?? {});
  let pills: Pill[] = stored?.pills && stored.pills.length > 0
    ? stored.pills.map((p) => ({ ...p }))
    : defaultPills(initSize.width, initSize.height);
  let detach = attachDrag(ctx.canvas, pills, ctx.dpr, () => SHAPE_ID[params.shape]);

  const saveDebounced = debouncedSaver(250);
  const persist = () => saveDebounced.schedule(params, pills);

  // Scene-change flag — consumed next frame by the render loop to force a
  // full historyBlend (1.0) so the previous scene doesn't ghost in. 2 frames
  // covers the ping-pong double buffering: the "prev" we read from after a
  // change is the one written before the change happened.
  let resetHistoryFrames = 2;
  const markSceneChanged = () => { resetHistoryFrames = 2; };

  // Race guard: a slow photo fetch shouldn't overwrite a newer one if the user
  // clicks Reload twice quickly.
  let photoRevision = 0;
  const reloadPhoto = async () => {
    const rev = ++photoRevision;
    try {
      const next = await loadPhoto(ctx.device, Date.now());
      if (rev !== photoRevision) { destroyPhoto(next); return; }
      const old = photoNow;
      photoNow = next;
      rebuildBindGroups(ctx, pl, frameBuf, photoNow, history);
      markSceneChanged();
      // Hold off the destroy until pending GPU work referencing `old` has drained.
      ctx.device.queue.onSubmittedWorkDone()
        .then(() => destroyPhoto(old))
        .catch((err) => console.error('[photo] queue drain failed, skipping destroy:', err));
    } catch (err) {
      console.error('[photo] reload failed:', err);
      showNotice(`Photo reload failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  };
  // Live perf stats — written by the render loop, read by the UI panel via
  // tweakpane monitor bindings (no manual refresh needed).
  const perfStats = createPerfStats();
  const tickFrameTimer = makeFrameTimer(perfStats);

  initUi(params, () => { void reloadPhoto(); }, persist, markSceneChanged, {
    stats:        perfStats,
    hasGpuTiming: ctx.hasTimestamp,
  });

  // Flush any pending debounced save on page hide so a drag-then-close doesn't
  // lose the last drag position.
  const onPageHide = () => saveDebounced.flush();
  window.addEventListener('pagehide',        onPageHide);
  window.addEventListener('beforeunload',    onPageHide);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) saveDebounced.flush();
  });

  // Save on every canvas pointer release — not drag-aware, but the saver is
  // debounced (250 ms) so spurious releases (non-drag clicks, misses) are cheap,
  // and every real drag ends with one of these events.
  const onPointerRelease = () => persist();
  ctx.canvas.addEventListener('pointerup',     onPointerRelease);
  ctx.canvas.addEventListener('pointercancel', onPointerRelease);

  let forceN3 = false;
  const onKeyDown = (e: KeyboardEvent) => {
    if (isTypingTarget(e.target)) return;
    const k = e.key.toLowerCase();
    if (k === 'z') forceN3 = true;
    if (e.key === ' ') {
      e.preventDefault();
      detach();
      const cur = resizeCanvas(ctx.canvas, ctx.dpr);
      pills = defaultPills(cur.width, cur.height).map((p) => ({
        ...p,
        cx: Math.random() * cur.width,
        cy: Math.random() * cur.height,
      }));
      detach = attachDrag(ctx.canvas, pills, ctx.dpr, () => SHAPE_ID[params.shape]);
      markSceneChanged();
      persist();
    }
    if (k === 'r') { void reloadPhoto(); /* already markSceneChanged'd on success */ }
  };
  const onKeyUp = (e: KeyboardEvent) => {
    if (e.key.toLowerCase() === 'z') forceN3 = false;
  };
  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('keyup',   onKeyUp);

  const applySrgbOetf = needsSrgbOetf(ctx.format);
  const startTime     = performance.now();

  // GPU timestamp queries — always on when the adapter supports them so the
  // perf monitor in the UI has live data without a URL flag. `?perf=1` still
  // exposes `window._perf` for external benchmark scripts that want raw
  // sample arrays.
  const perf        = ctx.hasTimestamp ? createPerf(ctx.device) : null;
  const perfHud     = ctx.hasTimestamp && new URLSearchParams(location.search).has('perf');
  const perfWindow: { samples: number[]; lastMs: number } = { samples: [], lastMs: 0 };
  (window as unknown as { _perf?: typeof perfWindow })._perf = perfHud ? perfWindow : undefined;

  const loop = () => {
    try {
      tickFrameTimer();
      const { width, height } = resizeCanvas(ctx.canvas, ctx.dpr);
      const resized = resizeHistory(ctx.device, history, width, height);
      if (resized !== history) {
        history = resized;
        rebuildBindGroups(ctx, pl, frameBuf, photoNow, history);
      }

      for (const pill of pills) {
        pill.hx    = params.pillLen   / 2;
        pill.hy    = params.pillShort / 2;
        pill.hz    = params.pillThick / 2;
        pill.edgeR = Math.min(params.edgeR, pill.hx, pill.hy, pill.hz);
      }
      const historyBlend = resetHistoryFrames > 0 ? 1.0 : 0.2;
      if (resetHistoryFrames > 0) resetHistoryFrames -= 1;

      // Modulo the time to stay within float32 precision — sin/cos of huge
      // arguments visibly stutter after hours of uptime. Any value above the
      // slowest rotation period (~31.4 s for 0.2 rad/s) is safe.
      const elapsed  = (performance.now() - startTime) * 0.001;
      const timeSafe = elapsed % 1e4;

      const N = forceN3 ? 3 : params.sampleCount;
      // Hero wavelength: one visible-range wavelength per frame, all pixels
      // share it. Temporal history accumulates across hero choices, so the
      // single-trace geometry error averages out over ~5 frames.
      const heroLambda = 380 + Math.random() * 320;
      // cameraZ sets the ortho depth AND implicitly the perspective FOV — for
      // a full FOV of `fov` degrees to fit the canvas height, the camera must
      // sit at `cameraZForFov(fov, height)` pixels above the z=0 plane.
      const cameraZ = cameraZForFov(params.fov, height);
      writeFrame(ctx.device, frameBuf, {
        resolution:         [width, height],
        photoSize:          [photoNow.width, photoNow.height],
        n_d:                params.n_d,
        V_d:                params.V_d,
        sampleCount:        N,
        refractionStrength: params.refractionStrength,
        jitter:             params.temporalJitter ? Math.random() / N : 0,
        refractionMode:     params.refractionMode === 'exact' ? 0 : 1,
        applySrgbOetf,
        shape:              SHAPE_ID[params.shape],
        time:               timeSafe,
        historyBlend,
        heroLambda,
        cameraZ,
        projection:         PROJECTION_ID[params.projection],
        debugProxy:         params.debugProxy,
        pills,
      });

      draw(ctx, pl, history, pills.length, perf?.writes, perf ? (enc) => perf.resolve(enc) : undefined);
      history.current = history.current === 0 ? 1 : 0;

      if (perf) {
        void perf.readMs().then((ms) => {
          if (ms === null || !Number.isFinite(ms)) return;
          perfStats.gpuMs = ms;
          if (perfHud) {
            perfWindow.lastMs = ms;
            perfWindow.samples.push(ms);
            if (perfWindow.samples.length > 240) perfWindow.samples.shift();
          }
        });
      }
    } catch (err) {
      console.error('[frame] render loop aborted:', err);
      showFatal(`Render loop stopped: ${err instanceof Error ? err.message : String(err)}. Please reload the page.`);
      return;  // do NOT reschedule — freezing is better than a flood of identical errors
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

main().catch((err) => {
  console.error(err);
  showFatal(`Couldn't start the demo: ${err instanceof Error ? err.message : String(err)}. See the browser console for details.`);
});
