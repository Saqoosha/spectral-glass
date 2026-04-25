/**
 * Builds the per-frame `jitter` and `heroLambda` UBO fields. The two values
 * are coupled — when temporal jitter is off, `wavelengthJitter` becomes the
 * negative sentinel that `dispersion/fragment.wgsl::spectralStratumJitter`
 * checks for (`if (jitter < 0.0)`), and `heroLambda` is pinned to 540 nm so
 * the Approx-mode shared back-face trace doesn't wander each frame.
 *
 * Changing the sentinel value or `FIXED_HERO_LAMBDA` requires touching
 * `dispersion/fragment.wgsl` and the WGSL `Frame` field comments.
 */

/** Sentinel written to `frame.jitter` when temporal jitter is off; the WGSL
 *  `spectralStratumJitter` helper short-circuits on `jitter < 0`. Any negative
 *  value would do — the host emits exactly `-1` so the round-trip is obvious
 *  in a UBO dump. */
export const SPECTRAL_JITTER_DISABLED = -1;

/** Fixed hero wavelength (nm) used when temporal jitter is off. 540 nm sits
 *  near the centre of the [380, 700] visible range and the photopic peak,
 *  so the Approx single-trace back-face geometry stays photographically
 *  reasonable instead of drifting toward the spectrum edges. */
export const FIXED_HERO_LAMBDA = 540;

const VISIBLE_LAMBDA_MIN = 380;
const VISIBLE_LAMBDA_RANGE = 320; // 700 nm − 380 nm — matches the per-λ loop in dispersion/fragment.wgsl

export type SpectralSamplingFields = {
  readonly wavelengthJitter: number;
  readonly heroLambda: number;
};

export function spectralSamplingFields(
  temporalJitter: boolean,
  sampleCount: number,
  rand: () => number = Math.random,
): SpectralSamplingFields {
  if (!temporalJitter) {
    return {
      wavelengthJitter: SPECTRAL_JITTER_DISABLED,
      heroLambda:       FIXED_HERO_LAMBDA,
    };
  }

  // Defensive: NaN / Infinity / sub-1 sampleCount would otherwise leak NaN
  // into the UBO, which the WGSL `jitter < 0.0` test treats as false →
  // sentinel never fires. Clamp to >= 1 with an explicit finite check.
  const n = Number.isFinite(sampleCount) && sampleCount >= 1 ? Math.floor(sampleCount) : 1;
  return {
    wavelengthJitter: rand() / n,
    heroLambda:       VISIBLE_LAMBDA_MIN + rand() * VISIBLE_LAMBDA_RANGE,
  };
}
