import { describe, it, expect, vi } from 'vitest';
import { createPerfStats, makeFrameTimer } from '../src/perfStats';

function withMockedNow<T>(samples: number[], fn: () => T): T {
  let i = 0;
  const spy = vi.spyOn(performance, 'now').mockImplementation(() => {
    const v = samples[Math.min(i, samples.length - 1)] ?? 0;
    i += 1;
    return v;
  });
  try {
    return fn();
  } finally {
    spy.mockRestore();
  }
}

describe('createPerfStats', () => {
  it('starts zeroed', () => {
    const s = createPerfStats();
    expect(s).toEqual({ cpuMs: 0, fps: 0, gpuMs: 0 });
  });
});

describe('makeFrameTimer', () => {
  it('skips the first tick (no prior timestamp)', () => {
    const stats = createPerfStats();
    withMockedNow([100], () => {
      const tick = makeFrameTimer(stats);
      tick();
    });
    expect(stats.cpuMs).toBe(0);
    expect(stats.fps).toBe(0);
  });

  it('records the raf interval as cpuMs on the second tick', () => {
    const stats = createPerfStats();
    withMockedNow([100, 116.67], () => {
      const tick = makeFrameTimer(stats);
      tick();  // seed
      tick();  // record
    });
    expect(stats.cpuMs).toBeCloseTo(16.67, 5);
    expect(stats.fps).toBeCloseTo(1000 / 16.67, 2);
  });

  it('averages FPS across multiple samples (steady 60 fps → ~60)', () => {
    const stats = createPerfStats();
    const ts: number[] = [];
    for (let i = 0; i <= 30; i++) ts.push(i * (1000 / 60));
    withMockedNow(ts, () => {
      const tick = makeFrameTimer(stats);
      for (const _ of ts) tick();
    });
    expect(stats.fps).toBeCloseTo(60, 0);
  });

  it('keeps cpuMs as the most recent interval, not a smoothed value', () => {
    const stats = createPerfStats();
    // Simulate a hiccup: two 16ms frames then one 100ms frame.
    withMockedNow([0, 16, 32, 132], () => {
      const tick = makeFrameTimer(stats);
      tick(); tick(); tick(); tick();
    });
    expect(stats.cpuMs).toBe(100);
    // FPS averages the three deltas (16, 16, 100) → 1000 * 3 / 132 ≈ 22.7
    expect(stats.fps).toBeCloseTo(22.73, 1);
  });
});
