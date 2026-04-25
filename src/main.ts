import { initGpu, resizeCanvas, needsSrgbOetf } from './webgpu/device';
import { createPipeline, encodeScene, rebuildBindGroups } from './webgpu/pipeline';
import {
  copyHtmlLayerToTexture,
  createHtmlBackgroundTexture,
  destroyHtmlBackgroundTexture,
  isValidHtmlBgLayer,
  supportsHtmlInCanvas,
} from './htmlBgTexture';
import type { PhotoTex } from './photo';
import { createFrameBuffer, writeFrame } from './webgpu/uniforms';
import { createPerf } from './webgpu/perf';
import { loadPhoto, destroyPhoto, picsumPhotoUrl } from './photo';
import { loadEnvmap, createDefaultEnvmap, destroyEnvmap, type EnvmapTex } from './envmap';
import { DEFAULT_ENVMAP_SLUG, envmapUrl, isKnownSlug, pickRandomSlug } from './envmapList';
import {
  attachDrag,
  defaultPills,
  ensurePillInstanceCount,
  setPillInstanceCount,
  DEFAULT_PILL_COUNT,
  type Pill,
} from './pills';
import { defaultParams, initUi, mergeParams, type Params } from './ui';
import { cameraZForFov } from './math/camera';
import { createHistory, resizeHistory } from './webgpu/history';
import { createPostProcess, encodePost, resizeIntermediate, writePostFrame } from './webgpu/postprocess';
import { loadStored, debouncedSaver } from './persistence';
import { createPerfStats, makeFrameTimer } from './perfStats';
import { frameFieldsFromParams } from './shapeParams';
import { spectralSamplingFields } from './spectralSampling';

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

let noticeHideTimer: number | null = null;
let fatalFallbackActive = false;

function clearNoticeHideTimer(): void {
  if (noticeHideTimer === null) return;
  window.clearTimeout(noticeHideTimer);
  noticeHideTimer = null;
}

function showFatal(message: string): void {
  fatalFallbackActive = true;
  clearNoticeHideTimer();
  const fb = document.getElementById('fallback');
  if (fb) {
    fb.textContent = message;
    fb.classList.add('visible');
  }
  document.getElementById('gpu')?.setAttribute('style', 'display:none');
}

/** Non-fatal user-visible notice — e.g. photo reload failed. Auto-hides. */
function showNotice(message: string, durationMs = 3000): void {
  if (fatalFallbackActive) return;
  const fb = document.getElementById('fallback');
  if (!fb) return;
  clearNoticeHideTimer();
  fb.textContent = message;
  fb.classList.add('visible');
  noticeHideTimer = window.setTimeout(() => {
    noticeHideTimer = null;
    if (!fatalFallbackActive) fb.classList.remove('visible');
  }, durationMs);
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
  let photoSeed  = Date.now();
  const bootPhoto = await loadPhoto(ctx.device, photoSeed);
  let photoNow   = bootPhoto.photo;
  // Default envmap is a synthetic sky/horizon gradient — ships without
  // network cost so pipeline creation is offline. The real HDRI is
  // fetched asynchronously after createPipeline returns; bind groups
  // rebuild on arrival.
  let envmapNow: EnvmapTex = createDefaultEnvmap(ctx.device);

  const initSize = resizeCanvas(ctx.canvas, ctx.dpr);
  let history    = createHistory(ctx.device, initSize.width, initSize.height);
  const pl       = await createPipeline(ctx, frameBuf, photoNow, envmapNow, history);
  const post     = await createPostProcess(ctx, initSize.width, initSize.height);

  const stored   = loadStored();
  const params   = mergeParams(defaultParams(), stored?.params ?? {});
  const htmlBgEl         = document.getElementById('html-bg-root');
  const htmlBgForeground = document.getElementById('html-bg-foreground');
  const htmlBgPhoto      = document.getElementById('html-bg-photo');
  const htmlInCanvasReady =
    supportsHtmlInCanvas(ctx.device) &&
    htmlBgEl instanceof HTMLElement &&
    isValidHtmlBgLayer(ctx.canvas, htmlBgEl);
  if (!htmlInCanvasReady && params.bgSource === 'html') {
    params.bgSource = 'photo';
  }
  /** HTML-in-Canvas layer uploaded to GPU when `params.bgSource === 'html'`. */
  let htmlPhoto: PhotoTex | null = null;
  const getActivePhoto = (): PhotoTex => {
    if (htmlInCanvasReady && params.bgSource === 'html' && htmlPhoto) return htmlPhoto;
    return photoNow;
  };
  let pills: Pill[] = stored?.pills && stored.pills.length > 0
    ? stored.pills.map((p) => ({ ...p }))
    : defaultPills(initSize.width, initSize.height);
  const pillCountBeforeEnsure = pills.length;
  const targetPillCountForShape = (shape: Params['shape']): number =>
    shape === 'diamond' ? 1 : DEFAULT_PILL_COUNT;
  // Two-step boot reconciliation: `ensurePillInstanceCount` enforces the
  // FLOOR (so we never start with a lone pill from old localStorage), then
  // `setPillInstanceCount` enforces the EXACT shape-driven count (1 for
  // diamond, 4 for the rest). Composing the two keeps the persistence
  // path's "don't lose user state" semantics while still letting the
  // diamond preset open with a single instance.
  pills = setPillInstanceCount(
    ensurePillInstanceCount(pills, initSize.width, initSize.height, targetPillCountForShape(params.shape)),
    initSize.width,
    initSize.height,
    targetPillCountForShape(params.shape),
  );

  // Scene-change flag — consumed next frame by the render loop to force a
  // full historyBlend (1.0) so the previous scene doesn't ghost in. 2 frames
  // covers the ping-pong double buffering: the "prev" we read from after a
  // change is the one written before the change happened.
  let resetHistoryFrames = 2;
  const markSceneChanged = () => { resetHistoryFrames = 2; };

  // Re-attach the pointer-event drag layer with the same shape/wave callbacks.
  // Hoisted into a helper because the loop body re-creates `pills` on Space
  // (random-shuffle), so we need the same arg list at two call sites.
  // Plate's visible silhouette extends past halfSize by `waveAmp` in the
  // tumbling Z direction; report that to the drag layer so the hit circle
  // matches the rendered shape (cube/pill/prism don't bulge → 0 margin).
  const makeDrag = (): (() => void) => attachDrag(
    ctx.canvas, pills, ctx.dpr,
    () => SHAPE_ID[params.shape],
    () => params.shape === 'plate' ? params.shapes.plate.waveAmp : 0,
    markSceneChanged,
  );
  let detach = makeDrag();

  const setScenePillCount = (count: number): void => {
    detach();
    const cur = resizeCanvas(ctx.canvas, ctx.dpr);
    pills = setPillInstanceCount(pills, cur.width, cur.height, count);
    detach = makeDrag();
    markSceneChanged();
  };

  const saveDebounced = debouncedSaver(250);
  const persist = () => saveDebounced.schedule(params, pills);
  if (pills.length !== pillCountBeforeEnsure) {
    saveDebounced.schedule(params, pills);
  }

  /** Same Picsum image as the GPU photo texture, for the HTML-in-Canvas snapshot.
   *  Race-guarded like `reloadPhoto` — rapid Random clicks shouldn't let an older
   *  fetch's onload/onerror clobber the newer one's class state. */
  let underlayRevision = 0;
  const syncPicsumUnderlay = (): void => {
    if (!htmlInCanvasReady) return;
    if (!(htmlBgPhoto instanceof HTMLImageElement) || !(htmlBgEl instanceof HTMLElement)) return;
    const rev = ++underlayRevision;
    const url = picsumPhotoUrl(photoSeed);
    const rep = (): void => {
      ctx.canvas.requestPaint?.();
      queueMicrotask(() => { ctx.canvas.requestPaint?.(); });
    };
    htmlBgPhoto.onload = () => {
      if (rev !== underlayRevision) return;
      htmlBgEl.classList.remove('html-bg--gradient-fallback');
      htmlBgPhoto.removeAttribute('hidden');
      rep();
    };
    htmlBgPhoto.onerror = (ev) => {
      if (rev !== underlayRevision) return;
      console.error('[html-bg] underlay image failed to load:', url, ev);
      htmlBgPhoto.setAttribute('hidden', '');
      htmlBgEl.classList.add('html-bg--gradient-fallback');
      // GPU photo succeeded but HTML underlay didn't: refraction and
      // pass-through background now show different images. Tell the user
      // so the visual mismatch isn't mistaken for a rendering bug.
      showNotice('HTML background image failed — underlay uses gradient (GPU photo unchanged).');
      rep();
    };
    htmlBgPhoto.removeAttribute('hidden');
    htmlBgPhoto.src = url;
  };

  // Race guard: a slow photo fetch shouldn't overwrite a newer one if the user
  // clicks Reload twice quickly.
  let photoRevision = 0;
  const reloadPhoto = async () => {
    const rev = ++photoRevision;
    try {
      const nextSeed = Date.now();
      const { photo: next, usedGradientFallback } = await loadPhoto(ctx.device, nextSeed);
      if (rev !== photoRevision) {
        // User clicked Reload again before this fetch resolved — discard
        // silently, but log so the developer can see why nothing updated.
        console.info('[photo] reload superseded by newer request, discarding rev', rev);
        destroyPhoto(next);
        return;
      }
      if (usedGradientFallback) {
        destroyPhoto(next);
        showNotice('Picsum photo fetch failed — previous image kept.');
        return;
      }
      const old = photoNow;
      photoNow   = next;
      photoSeed  = nextSeed;
      syncPicsumUnderlay();
      rebuildBindGroups(ctx, pl, frameBuf, getActivePhoto(), envmapNow, history);
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
  // Envmap reload follows the same race-guarded pattern as photo.
  // Failed fetch leaves the current envmap in place; notifies the UI so
  // the user knows their click didn't land. `isBoot` extends the
  // notice duration when the very first panorama fetch fails: the
  // synthetic gradient fallback looks plausibly-outdoorsy and would
  // otherwise hide the failure (user concludes "rendering looks
  // bland" rather than "my network blocked the CDN"), so a longer
  // on-screen message makes the failure mode discoverable.
  let envmapRevision = 0;
  // Tracks whether the REAL HDRI has ever loaded this session — lets
  // the `onEnvmapEnabled` UI hook know whether a lazy fetch is needed
  // when the user flips envmap on after it was off at boot. Flips on
  // the first SUCCESSFUL reload (not on failed fetch), so a failed
  // initial attempt still lets the lazy path retry later.
  let envmapRealLoaded = false;
  const reloadEnvmap = async (slug: string, isBoot = false): Promise<void> => {
    const rev = ++envmapRevision;
    try {
      const next = await loadEnvmap(ctx.device, envmapUrl(slug, params.envmapSize));
      if (rev !== envmapRevision) { destroyEnvmap(next); return; }
      const old = envmapNow;
      envmapNow = next;
      envmapRealLoaded = true;
      params.envmapSlug = slug;
      rebuildBindGroups(ctx, pl, frameBuf, getActivePhoto(), envmapNow, history);
      markSceneChanged();
      persist();
      ctx.device.queue.onSubmittedWorkDone()
        .then(() => destroyEnvmap(old))
        // Narrow catch: ONLY covers the cleanup path. Device-lost
        // scenarios are surfaced separately via the `device.lost`
        // handler in `src/webgpu/device.ts`, which calls `showFatal`
        // and prompts a reload. If that fatal wasn't wired, the
        // old-texture leak here would still be preferable to tearing
        // down app state on every transient hiccup.
        .catch((err) => console.error('[envmap] queue drain failed, skipping destroy:', err));
    } catch (err) {
      console.error('[envmap] reload failed:', err);
      const msg = err instanceof Error ? err.message : String(err);
      // Boot-time failure: extend the notice to 12 s and include
      // "using fallback gradient" so the user understands the on-
      // screen rendering is the default, not the requested HDRI.
      if (isBoot) {
        showNotice(`Envmap fetch failed — using fallback gradient. ${msg}`, 12_000);
      } else {
        showNotice(`Envmap fetch failed: ${msg}`);
      }
    }
  };
  // Live perf stats — written by the render loop, read by the UI panel via
  // tweakpane monitor bindings (no manual refresh needed).
  const perfStats = createPerfStats();
  const tickFrameTimer = makeFrameTimer(perfStats);

  // Forward-declared so the `randomEnvmap` closure can refresh the pane
  // AFTER it's created. Tweakpane's one-way data flow (params → UI)
  // needs a manual refresh() when we mutate params from the outside.
  let paneRef: ReturnType<typeof initUi> | null = null;
  const randomEnvmap = () => {
    const next = pickRandomSlug(params.envmapSlug);
    // Set the slug synchronously BEFORE refresh. `reloadEnvmap` only
    // writes `params.envmapSlug = slug` after its async fetch resolves,
    // so a pre-fetch refresh would read the OLD slug and leave the
    // dropdown stuck on the previous panorama. Updating here keeps the
    // UI and the pending fetch in sync even if the fetch later fails
    // (the user sees what they asked for; the notice explains the
    // failure separately).
    params.envmapSlug = next;
    void reloadEnvmap(next);
    paneRef?.refresh();
  };
  const pane = initUi(
    params,
    () => { void reloadPhoto(); },
    persist,
    markSceneChanged,
    { stats: perfStats, hasGpuTiming: ctx.hasTimestamp },
    (slug) => { void reloadEnvmap(slug); },
    randomEnvmap,
    () => {
      // Enabled toggled to true. Fetch the stored slug now if we
      // haven't already loaded a real HDRI this session. The
      // `envmapRealLoaded` flag lives inside `reloadEnvmap`'s
      // closure and flips on the first successful fetch.
      if (!envmapRealLoaded) void reloadEnvmap(params.envmapSlug);
    },
    htmlInCanvasReady ? { supported: true } : null,
    setScenePillCount,
  );
  paneRef = pane;

  if (htmlInCanvasReady) {
    ctx.canvas.layoutSubtree = true;
    // Persistent copy-failure → fall back to Picsum so the user isn't
    // stuck with a frozen HTML snapshot and no indication why. Threshold
    // is low (3 consecutive frames) because a real failure tends to
    // repeat every paint.
    let htmlBgCopyFailCount = 0;
    const HTML_BG_COPY_FAIL_MAX = 3;
    const onPaint = (): void => {
      if (params.bgSource !== 'html' || !htmlPhoto) return;
      if (!(htmlBgEl instanceof HTMLElement)) return;
      const ok = copyHtmlLayerToTexture(ctx.device.queue, htmlBgEl, htmlPhoto.texture);
      if (ok) {
        htmlBgCopyFailCount = 0;
        return;
      }
      htmlBgCopyFailCount += 1;
      if (htmlBgCopyFailCount >= HTML_BG_COPY_FAIL_MAX && params.bgSource === 'html') {
        params.bgSource = 'photo';
        paneRef?.refresh();
        markSceneChanged();
        persist();
        showNotice('HTML background sync failed — falling back to Picsum.', 6_000);
      }
    };
    ctx.canvas.addEventListener('paint', onPaint);
    if (htmlBgForeground instanceof HTMLElement) {
      htmlBgForeground.addEventListener('input', () => { ctx.canvas.requestPaint?.(); });
    }
    new ResizeObserver(() => {
      if (params.bgSource === 'html') ctx.canvas.requestPaint?.();
    }).observe(ctx.canvas, { box: 'device-pixel-content-box' });
    // If the GPU Picsum path fell back to the gradient, do not point the
    // underlay <img> at a URL that may still load when fetch failed
    // (CORS / timing skew), which would desync HTML vs GPU.
    if (bootPhoto.usedGradientFallback) {
      if (htmlBgEl instanceof HTMLElement && htmlBgPhoto instanceof HTMLImageElement) {
        htmlBgEl.classList.add('html-bg--gradient-fallback');
        htmlBgPhoto.setAttribute('hidden', '');
      }
      showNotice('Picsum photo fetch failed — background uses gradient until reload.', 6_000);
    } else {
      syncPicsumUnderlay();
    }
  } else if (bootPhoto.usedGradientFallback) {
    showNotice('Picsum photo fetch failed — using gradient background.', 6_000);
  }

  // Kick off the initial HDR panorama fetch. Pipeline already has the
  // default gradient envmap bound, so the first few frames render fine
  // (just the gradient) until the real HDRI arrives. If the slug is
  // unknown (e.g. we added the allow-list after a user's localStorage
  // stored something stale), fall back to the bundled default.
  const bootSlug = isKnownSlug(params.envmapSlug) ? params.envmapSlug : DEFAULT_ENVMAP_SLUG;
  // Skip the initial HDRI download when envmap is disabled — it's a
  // 2K file (~5-10 MB) on every page load, and the user-visible
  // rendering uses the Phase A photo-based fallback in that mode.
  // When the user later toggles Enabled from false → true, the
  // `envmapEnabled` binding fires a lazy fetch (see ui.ts) so the
  // HDRI only downloads when it's actually going to be sampled.
  if (params.envmapEnabled) {
    void reloadEnvmap(bootSlug, /* isBoot */ true);
  } else {
    // Still pin the slug to a known value so the lazy fetch on enable
    // uses a working URL even if localStorage had a stale / unknown
    // slug when the page opened.
    params.envmapSlug = bootSlug;
  }

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
      // Shuffle just the visible count — diamond stays single-instance even
      // after Space, otherwise the preset's 1-instance rule would drift on
      // the first random shuffle.
      pills = defaultPills(cur.width, cur.height).slice(0, targetPillCountForShape(params.shape)).map((p) => ({
        ...p,
        cx: Math.random() * cur.width,
        cy: Math.random() * cur.height,
      }));
      detach = makeDrag();
      markSceneChanged();
      persist();
    }
    if (k === 'r' && !params.envmapEnabled) {
      void reloadPhoto(); /* markSceneChanged on success — same as the Random photo button (hidden when HDR env on) */
    }
    if (k === 'h') {
      const panes = document.querySelectorAll<HTMLElement>('.tp-dfwv');
      const hidden = panes[0]?.style.display === 'none';
      panes.forEach((el) => { el.style.display = hidden ? '' : 'none'; });
    }
    // Diamond view presets — snap to canonical poses for facet geometry
    // checking. T/S/B/F mirror the Tweakpane dropdown so either path works.
    // Only active when the diamond shape is selected (the param exists
    // for other shapes too, but changing it would be a no-op so no point
    // clearing the history).
    if (params.shape === 'diamond' && (k === 't' || k === 's' || k === 'b' || k === 'f')) {
      const nextView = k === 't' ? 'top' : k === 's' ? 'side' : k === 'b' ? 'bottom' : 'free';
      if (params.shapes.diamond.diamondView !== nextView) {
        params.shapes.diamond.diamondView = nextView;
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

  // GPU timestamps when the device exposes `timestamp-query` (most desktop GPUs).
  // `?perf` on the URL still turns on `window._perf` sample logging for benchmarks.
  const perf    = ctx.hasTimestamp ? createPerf(ctx.device) : null;
  const perfHud = perf !== null && new URLSearchParams(location.search).has('perf');
  const perfWindow: { samples: number[]; lastMs: number } = { samples: [], lastMs: 0 };
  (window as unknown as { _perf?: typeof perfWindow })._perf = perfHud ? perfWindow : undefined;
  const loop = () => {
    try {
      tickFrameTimer();
      const { width, height } = resizeCanvas(ctx.canvas, ctx.dpr);
      const resized = resizeHistory(ctx.device, history, width, height);
      if (resized !== history) {
        history = resized;
        rebuildBindGroups(ctx, pl, frameBuf, getActivePhoto(), envmapNow, history);
        markSceneChanged();
      }
      resizeIntermediate(ctx.device, post, width, height);

      if (htmlInCanvasReady) {
        if (params.bgSource === 'html') {
          if (!htmlPhoto || htmlPhoto.width !== width || htmlPhoto.height !== height) {
            // Defer destroy until pending GPU work drains — the previous
            // frame's command buffer may still reference the old texture.
            // Same pattern as photo.ts / envmap.ts `reloadPhoto` swap.
            if (htmlPhoto) {
              const old = htmlPhoto;
              ctx.device.queue.onSubmittedWorkDone()
                .then(() => destroyHtmlBackgroundTexture(old))
                .catch((err) => console.error('[html-bg] queue drain failed, skipping destroy:', err));
              htmlPhoto = null;
            }
            htmlPhoto = createHtmlBackgroundTexture(ctx.device, width, height);
            rebuildBindGroups(ctx, pl, frameBuf, getActivePhoto(), envmapNow, history);
            markSceneChanged();
            ctx.canvas.requestPaint?.();
            // One extra paint after layout commits (first snapshot can be empty otherwise).
            queueMicrotask(() => { ctx.canvas.requestPaint?.(); });
          }
        } else if (htmlPhoto) {
          const old = htmlPhoto;
          htmlPhoto = null;
          ctx.device.queue.onSubmittedWorkDone()
            .then(() => destroyHtmlBackgroundTexture(old))
            .catch((err) => console.error('[html-bg] queue drain failed, skipping destroy:', err));
          rebuildBindGroups(ctx, pl, frameBuf, photoNow, envmapNow, history);
        }
      }

      const uf = frameFieldsFromParams(params);

      // Plate forces a square XY face (hy ≡ hx) so pillShort is effectively
      // unused. Plate's wave lives under `shapes.plate` →
      // `frame.waveAmp` (separate uniform). For plate/cube, `pill.edgeR` is
      // the rim fillet clamped to all half-axes. Pill clamps inside the SDF
      // as two radii: XY can grow to the short axis, while Z still clamps to
      // thickness so high edgeR produces a real pill without flattening the
      // top/bottom rounding. Prism has sharp SDF (no fillet).
      //
      // Diamond ignores per-pill halfSize entirely on the SDF side (shader
      // reads `frame.diamondSize`), but we still write halfSize to the
      // girdle radius so the drag layer's circular hit test lands on the
      // actual silhouette. edgeR stays at 0 for diamond — the shape uses
      // raw intersection-of-half-spaces without rounding.
      const isPlate   = params.shape === 'plate';
      const isPrism   = params.shape === 'prism';
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
          const half = uf.diamondSize / 2;
          pill.hx    = half;
          pill.hy    = half;
          pill.hz    = half;
          pill.edgeR = 0;
        } else {
          pill.hx    = uf.pillLen   / 2;
          pill.hy    = isPlate ? pill.hx : uf.pillShort / 2;
          pill.hz    = uf.pillThick / 2;
          // Prism: sharp isosceles solid — GPU uses `sdfPrism` with no fillet; keep 0.
          // Pill receives raw edgeR so sdfPill can split it into XY and Z radii.
          pill.edgeR = isPrism ? 0
            : params.shape === 'pill' ? uf.edgeR
            : Math.min(uf.edgeR, pill.hx, pill.hy, pill.hz);
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
      const spectralSampling = spectralSamplingFields(params.temporalJitter, N);
      // cameraZ sets the ortho depth AND implicitly the perspective FOV — for
      // a full FOV of `fov` degrees to fit the canvas height, the camera must
      // sit at `cameraZForFov(fov, height)` pixels above the z=0 plane.
      const cameraZ = cameraZForFov(params.fov, height);
      writeFrame(ctx.device, frameBuf, {
        resolution:         [width, height],
        photoSize:          [getActivePhoto().width, getActivePhoto().height],
        n_d:                uf.n_d,
        V_d:                uf.V_d,
        sampleCount:        N,
        refractionStrength: uf.refractionStrength,
        jitter:             spectralSampling.wavelengthJitter,
        refractionMode:     params.refractionMode === 'exact' ? 0 : 1,
        applySrgbOetf,
        shape:              SHAPE_ID[params.shape],
        time:               timeSafe,
        historyBlend,
        heroLambda:         spectralSampling.heroLambda,
        cameraZ,
        projection:         PROJECTION_ID[params.projection],
        debugProxy:         params.debugProxy,
        smoothCurvature:    params.smoothCurvature,
        // TAA sub-pixel jitter + motion-vector reprojection only enabled when
        // aaMode === 'taa'. FXAA handles AA in the post pass instead, and
        // 'none' renders with neither.
        taaEnabled:         params.aaMode === 'taa',
        sceneTime,
        prevSceneTime,
        // Plate wave params: amp is straight pixels, but the GPU wants an
        // angular frequency (rad/px) so it can feed sin() directly. UI thinks
        // in "wavelength in px" (more intuitive) → convert 2π/λ here.
        waveAmp:            uf.waveAmp,
        waveFreq:           (2 * Math.PI) / Math.max(uf.waveWavelength, 1),
        // Diamond: single global size (girdle diameter in px). Ignored when
        // the current shape isn't diamond, but the uniform slot is always
        // written so shape switching doesn't leave stale values.
        diamondSize:        uf.diamondSize,
        diamondWireframe:   uf.diamondWireframe,
        diamondFacetColor:  uf.diamondFacetColor,
        diamondTirDebug:    uf.diamondTirDebug,
        diamondTirMaxBounces: uf.diamondTirMaxBounces,
        diamondView:        uf.diamondView,
        // Envmap controls forward to the shader as-is; slug itself is
        // not a uniform (the texture IS the envmap — slug lives in UI
        // state only).
        envmapEnabled:      params.envmapEnabled,
        envmapExposure:     params.envmapExposure,
        envmapRotation:     params.envmapRotation,
        pills,
      });

      const encoder = ctx.device.createCommandEncoder({ label: 'frame' });
      encodeScene(pl, history, post.intermediate, pills.length, SHAPE_ID[params.shape], encoder, perf?.writes);
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
