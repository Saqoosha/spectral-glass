import { describe, it, expect } from 'vitest';
import {
  FIXED_HERO_LAMBDA,
  SPECTRAL_JITTER_DISABLED,
  spectralSamplingFields,
} from '../src/spectralSampling';

describe('spectralSamplingFields', () => {
  it('uses sentinel jitter and fixed hero wavelength when temporal jitter is disabled', () => {
    const fields = spectralSamplingFields(false, 16, () => {
      throw new Error('disabled temporal jitter must not consume random numbers');
    });

    expect(fields.wavelengthJitter).toBe(SPECTRAL_JITTER_DISABLED);
    expect(fields.heroLambda).toBe(FIXED_HERO_LAMBDA);
  });

  it('uses per-frame random wavelength jitter and hero wavelength when enabled', () => {
    const draws = [0.25, 0.75];
    let i = 0;
    const fields = spectralSamplingFields(true, 16, () => draws[i++] ?? 0);

    expect(fields.wavelengthJitter).toBeCloseTo(0.25 / 16);
    expect(fields.heroLambda).toBeCloseTo(380 + 0.75 * 320);
    expect(i).toBe(2);
  });

  it('pins hero wavelength to the visible-spectrum endpoints when rand returns 0 or 1', () => {
    const lo = spectralSamplingFields(true, 8, () => 0);
    expect(lo.heroLambda).toBe(380);
    const draws = [0, 1];
    let i = 0;
    const hi = spectralSamplingFields(true, 8, () => draws[i++] ?? 0);
    expect(hi.heroLambda).toBe(700);
  });

  it('falls back to n=1 when sampleCount is non-finite or below 1 so no NaN reaches the UBO', () => {
    const cases: number[] = [0, -3, 0.5, NaN, Number.POSITIVE_INFINITY];
    for (const sampleCount of cases) {
      const fields = spectralSamplingFields(true, sampleCount, () => 0.4);
      expect(Number.isFinite(fields.wavelengthJitter)).toBe(true);
      expect(fields.wavelengthJitter).toBeCloseTo(0.4);
    }
  });
});
