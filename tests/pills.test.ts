import { describe, it, expect } from 'vitest';
import { defaultPills, ensurePillInstanceCount, DEFAULT_PILL_COUNT } from '../src/pills';

describe('ensurePillInstanceCount', () => {
  it('pads a single pill up to DEFAULT_PILL_COUNT using default layout slots', () => {
    const w = 800;
    const h = 600;
    const one = [{ cx: 100, cy: 200, cz: 0, hx: 1, hy: 1, hz: 1, edgeR: 5 }];
    const out = ensurePillInstanceCount(one, w, h);
    expect(out).toHaveLength(DEFAULT_PILL_COUNT);
    expect(out[0]).toEqual(one[0]);
    const d = defaultPills(w, h);
    for (let i = 1; i < DEFAULT_PILL_COUNT; i++) {
      expect(out[i]?.cx).toBe(d[i]!.cx);
      expect(out[i]?.cy).toBe(d[i]!.cy);
    }
  });

  it('leaves four or more pills unchanged (aside from cloning)', () => {
    const w = 400;
    const h = 300;
    const d = defaultPills(w, h);
    const out = ensurePillInstanceCount(d, w, h);
    expect(out).toHaveLength(4);
    expect(out[0]?.cx).toBe(d[0]!.cx);
  });
});
