import { describe, it, expect } from 'vitest';
import { cieXyz } from '../src/math/wyman';

describe('cieXyz (Wyman-Sloan-Shirley approximation)', () => {
  it('peaks of ȳ are near 555 nm (photopic luminosity)', () => {
    let peakLambda = 555;
    let peakY = 0;
    for (let l = 400; l <= 700; l += 1) {
      const y = cieXyz(l)[1];
      if (y > peakY) { peakY = y; peakLambda = l; }
    }
    expect(peakLambda).toBeGreaterThan(545);
    expect(peakLambda).toBeLessThan(570);
  });

  it('x̄ is higher than ȳ and z̄ in the long-wavelength red (650 nm)', () => {
    const [x, y, z] = cieXyz(650);
    expect(x).toBeGreaterThan(y);
    expect(x).toBeGreaterThan(z);
  });

  it('z̄ dominates in the short-wavelength blue (450 nm)', () => {
    const [x, y, z] = cieXyz(450);
    expect(z).toBeGreaterThan(x);
    expect(z).toBeGreaterThan(y);
  });

  it('returns near-zero for far-UV and far-IR', () => {
    const uv = cieXyz(350);
    const ir = cieXyz(780);
    for (const v of [...uv, ...ir]) expect(Math.abs(v)).toBeLessThan(0.05);
  });
});
