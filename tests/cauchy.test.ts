import { describe, it, expect } from 'vitest';
import { cauchyIor } from '../src/math/cauchy';

describe('cauchyIor', () => {
  it('returns n_d at the d-line wavelength (587.56 nm)', () => {
    // At λ = 587.56 nm, the formula should return ~n_d regardless of V_d.
    // glTF dispersion formula: n(λ) = n_d + (n_d - 1)/V_d * (523655/λ² − 1.5168)
    // 523655 / 587.56² ≈ 1.5168, so the offset term ≈ 0.
    expect(cauchyIor(587.56, 1.5168, 64.2)).toBeCloseTo(1.5168, 3);
  });

  it('returns a larger IOR for shorter (blue) wavelengths', () => {
    const blue = cauchyIor(440, 1.5168, 40);
    const d    = cauchyIor(587.56, 1.5168, 40);
    const red  = cauchyIor(660, 1.5168, 40);
    expect(blue).toBeGreaterThan(d);
    expect(red).toBeLessThan(d);
  });

  it('increases dispersion (blue − red gap) as V_d decreases', () => {
    const lowVd  = cauchyIor(440, 1.5168, 20) - cauchyIor(660, 1.5168, 20);
    const highVd = cauchyIor(440, 1.5168, 80) - cauchyIor(660, 1.5168, 80);
    expect(lowVd).toBeGreaterThan(highVd);
  });

  it('never returns an IOR below 1.0', () => {
    expect(cauchyIor(1000, 1.01, 10)).toBeGreaterThanOrEqual(1.0);
  });
});
