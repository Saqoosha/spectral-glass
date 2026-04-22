/**
 * Live performance stats shared between the render loop (writer) and the UI
 * panel (reader via tweakpane monitor bindings).
 *
 * The render loop updates `cpuMs` every frame and `gpuMs` whenever a GPU
 * timestamp readback completes. `fps` is derived from a 30-frame rolling
 * average of CPU frame intervals so a single hiccup doesn't make the number
 * jump around.
 *
 * Tweakpane's monitor bindings poll these fields directly, so writing to the
 * object is enough — no manual refresh from the loop.
 */
export type PerfStats = {
  /** Last CPU frame time in ms (raf interval). 0 until the first frame. */
  cpuMs: number;
  /** Smoothed FPS over the last ~30 frames. 0 until the rolling buffer fills. */
  fps:   number;
  /** Last GPU draw-pass time in ms (timestamp-query). 0 if unsupported. */
  gpuMs: number;
};

const FPS_WINDOW = 30;

export function createPerfStats(): PerfStats {
  return { cpuMs: 0, fps: 0, gpuMs: 0 };
}

/**
 * Wrap a `PerfStats` with a frame-time recorder. Each call advances the
 * rolling FPS window using `performance.now()` deltas; the first call only
 * seeds the timestamp and returns without updating cpuMs/fps.
 */
export function makeFrameTimer(stats: PerfStats): () => void {
  let prev: number | null = null;
  const window: number[] = [];
  return () => {
    const now = performance.now();
    if (prev !== null) {
      const dt = now - prev;
      stats.cpuMs = dt;
      window.push(dt);
      if (window.length > FPS_WINDOW) window.shift();
      const sum = window.reduce((a, b) => a + b, 0);
      stats.fps = window.length > 0 ? 1000 * window.length / sum : 0;
    }
    prev = now;
  };
}
