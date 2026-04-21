import { describe, it, expect } from 'vitest';
import { xyzToLinearSrgb, linearToGamma } from '../src/math/srgb';

describe('xyzToLinearSrgb', () => {
  it('maps D65 white (X=0.95047, Y=1.0, Z=1.08883) to approximately (1,1,1)', () => {
    const [r, g, b] = xyzToLinearSrgb([0.95047, 1.0, 1.08883]);
    expect(r).toBeCloseTo(1.0, 2);
    expect(g).toBeCloseTo(1.0, 2);
    expect(b).toBeCloseTo(1.0, 2);
  });

  it('maps pure luminance (Y only) to a neutral-ish gray', () => {
    const [r, g, b] = xyzToLinearSrgb([0, 0.5, 0]);
    // Y-only has non-zero R/G/B but G dominates.
    expect(g).toBeGreaterThan(0);
    expect(g).toBeGreaterThan(r);
    expect(g).toBeGreaterThan(b);
  });
});

describe('linearToGamma', () => {
  it('identity at 0 and 1', () => {
    expect(linearToGamma(0)).toBeCloseTo(0);
    expect(linearToGamma(1)).toBeCloseTo(1);
  });

  it('linear segment for dark values (< 0.0031308)', () => {
    expect(linearToGamma(0.001)).toBeCloseTo(0.001 * 12.92, 6);
  });

  it('gamma-power segment for brighter values', () => {
    const v = 0.5;
    const expected = 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
    expect(linearToGamma(v)).toBeCloseTo(expected, 6);
  });
});
