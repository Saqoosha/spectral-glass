import { initGpu, resizeCanvas, needsSrgbOetf } from './webgpu/device';
import { createPipeline, encodeScene, rebuildBindGroups } from './webgpu/pipeline';
import { createFrameBuffer, writeFrame } from './webgpu/uniforms';
import { createPerf } from './webgpu/perf';
import { loadPhoto, destroyPhoto } from './photo';
import { attachDrag, defaultPills, type Pill } from './pills';
import { defaultParams, initUi, mergeParams, type Params } from './ui';
import { cameraZForFov } from './math/camera';
import { createHistory, resizeHistory } from './webgpu/history';
import { createPostProcess, encodePost, resizeIntermediate, writePostFrame } from './webgpu/postprocess';
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
  pill:    0,
  prism:   1,
  cube:    2,
  plate:   3,
  diamond: 4,
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
  const post     = await createPostProcess(ctx, initSize.width, initSize.height);

  const stored = loadStored();
  const params = mergeParams(defaultParams(), stored?.params ?? {});
  let pills: Pill[] = stored?.pills && stored.pills.length > 0
    ? stored.pills.map((p) => ({ ...p }))
    : defaultPills(initSize.width, initSize.height);
  // Re-attach the pointer-event drag layer with the same shape/wave callbacks.
  // Hoisted into a helper because the loop body re-creates `pills` on Space
  // (random-shuffle), so we need the same arg list at two call sites.
  // Plate's visible silhouette extends past halfSize by `waveAmp` in the
  // tumbling Z direction; report that to the drag layer so the hit circle
  // matches the rendered shape (cube/pill/prism don't bulge → 0 margin).
  const makeDrag = (): (() => void) => attachDrag(
    ctx.canvas, pills, ctx.dpr,
    () => SHAPE_ID[params.shape],
    () => params.shape === 'plate' ? params.waveAmp : 0,
  );
  let detach = makeDrag();

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

  const pane = initUi(params, () => { void reloadPhoto(); }, persist, markSceneChanged, {
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
      detach = makeDrag();
      markSceneChanged();
      persist();
    }
    if (k === 'r') { void reloadPhoto(); /* already markSceneChanged'd on success */ }
    // Diamond view presets — snap to canonical poses for facet geometry
    // checking. T/S/B/F mirror the Tweakpane dropdown so either path works.
    // Only active when the diamond shape is selected (the param exists
    // for other shapes too, but changing it would be a no-op so no point
    // clearing the history).
    if (params.shape === 'diamond' && (k === 't' || k === 's' || k === 'b' || k === 'f')) {
      const nextView = k === 't' ? 'top' : k === 's' ? 'side' : k === 'b' ? 'bottom' : 'free';
      if (params.diamondView !== nextView) {
        params.diamondView = nextView;
        pane.refresh();          // sync the dropdown so the UI reflects the hotkey
        markSceneChanged();      // rotation jumps discontinuously — clear ghost trail
        persist();
      }
    }
  };
  const onKeyUp = (e: KeyboardEvent) => {
    if (e.key.toLowerCase() === 'z') forceN3 = false;
  };
  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('keyup',   onKeyUp);

  const applySrgbOetf = needsSrgbOetf(ctx.format);
  // applySrgbOetf is derived from the swapchain format and never changes;
  // write it into the post UBO once at startup instead of every frame.
  writePostFrame(ctx.device, post, applySrgbOetf);
  const startTime     = performance.now();
  // Wall-clock timestamp of the previous frame's update — used to compute
  // dt for `sceneTime` accumulation. Seeded with `startTime` so the very
  // first frame's dt is essentially zero and sceneTime starts at zero.
  let prevWallTime = startTime;
  // Scene time drives rotation + wave phase. Freezes when `params.paused` is
  // true ("Stop the world") while the noise-time (`timeSafe` computed below)
  // keeps advancing, so TAA sub-pixel jitter keeps accumulating samples and
  // paused scenes continue to converge toward full AA quality.
  //
  // `prevSceneTime` feeds the GPU's motion-vector reprojection for TAA
  // (history reads follow rotating shapes instead of smearing). Seeded with
  // 0 so the first frame's reprojection is a no-op; historyBlend=1.0 on the
  // opening two frames overwrites whatever transient error slips through.
  let sceneTime     = 0;
  let prevSceneTime = 0;

  // Paused-frame counter — drives the progressive-averaging history blend.
  // While "Stop the world" is on and the scene is static, we flip from
  // α = 0.2 EMA (which never fully converges — per-pixel TAA sub-pixel
  // jitter keeps flipping silhouette pixels between hit/miss, leaving ~45 %
  // residual shimmer) to α = max(1/n, 1/256), which computes the true
  // cumulative mean for the first ~256 paused frames and then plateaus at
  // a 256-sample sliding window. Noise drops as 1/√n in the ramp phase and
  // bottoms out at ~6 % residual at the cap.
  //
  // The 1/256 floor is required by the rgba16float history texture: a
  // smaller α would push the new-sample contribution below the fp16
  // quantum (≈ 0.0005 around mid-grey) for high-contrast edge pixels, the
  // contribution would round to 0, and `(1 − α) · prev` decay would slowly
  // fade silhouettes to black over several minutes. See the historyBlend
  // computation below for the full derivation.
  //
  // Resets to 1 whenever motion resumes or `markSceneChanged()` fires the
  // 2-frame history-overwrite reset (photo reload, shape switch, preset,
  // pause toggle). On the first paused frame after a reset, the increment
  // step below bumps `pausedFrames` to 2 first, so α = 1/2 — the correct
  // weight for "average the fresh sample with the single-sample history the
  // reset left behind" (post-reset history holds exactly one sample because
  // `historyBlend = 1.0` overwrote it).
  let pausedFrames = 1;

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
      resizeIntermediate(ctx.device, post, width, height);

      // Plate forces a square XY face (hy ≡ hx) so pillShort is effectively
      // unused. Plate's wave amplitude is driven by `params.waveAmp` →
      // `frame.waveAmp` (separate uniform). `pill.edgeR` is now the rounded-
      // corner radius for the rim that smooths the wavy front Z face into
      // the flat side X / Y faces — same role as in cube/pill/prism, so the
      // same `min(edgeR, halfSize)` clamp applies (edgeR ≥ smallest halfSize
      // would invert the rounded-box SDF into degenerate geometry).
      //
      // Diamond ignores per-pill halfSize entirely on the SDF side (shader
      // reads `frame.diamondSize`), but we still write halfSize to the
      // girdle radius so the drag layer's circular hit test lands on the
      // actual silhouette. edgeR stays at 0 for diamond — the shape uses
      // raw intersection-of-half-spaces without rounding.
      const isPlate   = params.shape === 'plate';
      const isDiamond = params.shape === 'diamond';
      for (const pill of pills) {
        if (isDiamond) {
          // Diamond ignores halfSize on the SDF side (shader reads
          // `frame.diamondSize`). We still write hx/hy/hz to the GIRDLE
          // RADIUS so the drag hit-test — which circumscribes the shape
          // with a circle of radius `max(hx, hy, hz)` (see pills.ts) —
          // lands exactly on the silhouette. hz is a deliberate
          // overestimate of the true half-height (~0.305·d); harmless for
          // the hit circle because the max() picks the girdle radius from
          // hx/hy regardless. edgeR is unused for diamond (sharp facets).
          const half = params.diamondSize / 2;
          pill.hx    = half;
          pill.hy    = half;
          pill.hz    = half;
          pill.edgeR = 0;
        } else {
          pill.hx    = params.pillLen   / 2;
          pill.hy    = isPlate ? pill.hx : params.pillShort / 2;
          pill.hz    = params.pillThick / 2;
          pill.edgeR = Math.min(params.edgeR, pill.hx, pill.hy, pill.hz);
        }
      }
      // Paused-frame accounting for progressive averaging (see declaration
      // above). During a reset-override window we hold pausedFrames at 1 so
      // the first post-reset paused frame gets α = 1/2 — the correct weight
      // for folding the fresh sample into the single-sample history the
      // reset overwrote.
      //
      // The progressive α is floored at 1/256 (≈ 0.004) because the history
      // texture is rgba16float — its mantissa step around 0.5 is ~0.0005, so
      // for an edge pixel where adjacent jitter samples differ by up to ~0.5
      // (hit vs miss), `α · (new − prev)` falls below the fp16 quantum once
      // α drops under ~0.001. The new sample contribution then rounds to 0
      // and the blend collapses to `(1 − α) · prev`, which is pure decay —
      // visibly the silhouette darkens to a black line over a few minutes
      // as (1 − 1/n)^N → 0. Capping α at 1/256 caps the post-pause
      // convergence window at ~256 samples (effective noise floor ≈ 6 %)
      // but keeps every contribution representable in fp16, so paused
      // scenes stay stable instead of slowly fading.
      if (!params.paused || resetHistoryFrames > 0) {
        pausedFrames = 1;
      } else {
        pausedFrames += 1;
      }
      const steadyBlend  = params.paused
        ? Math.max(1.0 / pausedFrames, 1.0 / 256)
        : params.historyAlpha;
      const historyBlend = resetHistoryFrames > 0 ? 1.0 : steadyBlend;
      if (resetHistoryFrames > 0) resetHistoryFrames -= 1;

      // Modulo the time to stay within float32 precision — sin/cos of huge
      // arguments visibly stutter after hours of uptime. Any value above the
      // slowest rotation period (~31.4 s for 0.2 rad/s) is safe. `timeSafe`
      // is the noise stream — driven by wall-clock so jitter keeps decorrel-
      // ating across frames even when the scene is paused. sceneTime is the
      // motion stream — accumulated from per-frame dt and skipped while
      // paused, so unpause resumes rotation/wave from exactly the frozen
      // pose instead of jumping forward by the pause duration.
      const wallNow  = performance.now();
      const elapsed  = (wallNow - startTime) * 0.001;
      const timeSafe = elapsed % 1e4;
      const dt       = (wallNow - prevWallTime) * 0.001;
      prevWallTime   = wallNow;
      if (!params.paused) { sceneTime = (sceneTime + dt) % 1e4; }

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
        // TAA sub-pixel jitter + motion-vector reprojection only enabled when
        // aaMode === 'taa'. FXAA handles AA in the post pass instead, and
        // 'none' renders with neither.
        taaEnabled:         params.aaMode === 'taa',
        sceneTime,
        prevSceneTime,
        // Plate wave params: amp is straight pixels, but the GPU wants an
        // angular frequency (rad/px) so it can feed sin() directly. UI thinks
        // in "wavelength in px" (more intuitive) → convert 2π/λ here.
        waveAmp:            params.waveAmp,
        waveFreq:           (2 * Math.PI) / Math.max(params.waveWavelength, 1),
        // Diamond: single global size (girdle diameter in px). Ignored when
        // the current shape isn't diamond, but the uniform slot is always
        // written so shape switching doesn't leave stale values.
        diamondSize:        params.diamondSize,
        diamondWireframe:   params.diamondWireframe,
        diamondFacetColor:  params.diamondFacetColor,
        diamondView:        params.diamondView,
        pills,
      });

      const encoder = ctx.device.createCommandEncoder({ label: 'frame' });
      encodeScene(pl, history, post.intermediate, pills.length, encoder, perf?.writes);
      encodePost(ctx, post, encoder, params.aaMode);
      if (perf) perf.resolve(encoder);
      ctx.device.queue.submit([encoder.finish()]);
      history.current = history.current === 0 ? 1 : 0;
      // Remember this frame's scene time so the next writeFrame can feed the
      // GPU the rotation state one frame ago for TAA reprojection. This is
      // the *scene* time, not the noise time — when paused, it holds steady
      // so the prev rotation matches the current one (no reprojection delta).
      prevSceneTime = sceneTime;

      if (perf) {
        void perf.readMs()
          .then((ms) => {
            if (ms === null || !Number.isFinite(ms)) return;
            perfStats.gpuMs = ms;
            if (perfHud) {
              perfWindow.lastMs = ms;
              perfWindow.samples.push(ms);
              if (perfWindow.samples.length > 240) perfWindow.samples.shift();
            }
          })
          // Without an explicit catch, a rejection here (typically GPU
          // device-lost mid-frame, buffer destroyed during a resize race,
          // or an exotic browser WebGPU implementation error) would land
          // in the global `unhandledrejection` channel — the perf graph
          // silently freezes at the last successful sample with no
          // explanation. NaN flatlines the readout instead, making the
          // failure debuggable.
          .catch((err) => {
            console.warn('[perf] readMs failed (likely GPU device lost):', err);
            perfStats.gpuMs = NaN;
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
