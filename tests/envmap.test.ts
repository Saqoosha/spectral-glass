import { describe, it, expect } from 'vitest';
import {
  clampFiniteF16,
  F16_MAX_FINITE_VALUE,
} from '../src/envmap';
import {
  ENVMAPS,
  ENVMAP_SIZES,
  DEFAULT_ENVMAP_SIZE,
  DEFAULT_ENVMAP_SLUG,
  envmapUrl,
  isKnownSlug,
  pickRandomSlug,
} from '../src/envmapList';

describe('clampFiniteF16', () => {
  // Phase C review flagged this as Critical load-bearing: if the clamp
  // breaks (boundary off-by-one, NaN guard removed, `v !== v` turned
  // into `isNaN(v)` by a transpiler, …), bright HDR pixels overflow to
  // +Inf on upload, the linear sampler computes `0 * Inf = NaN` during
  // filtering, and the render sprouts black dots at the brightest
  // highlights of every HDRI. Keep these tight.

  it('passes values within the fp16 finite range untouched', () => {
    expect(clampFiniteF16(0)).toBe(0);
    expect(clampFiniteF16(1)).toBe(1);
    expect(clampFiniteF16(-1)).toBe(-1);
    expect(clampFiniteF16(100)).toBe(100);
    expect(clampFiniteF16(-10000)).toBe(-10000);
    // Boundary value itself: MUST survive, otherwise the clamp is
    // off-by-one in the `> F16_MAX_FINITE` direction (strict vs. >=).
    expect(clampFiniteF16(F16_MAX_FINITE_VALUE)).toBe(F16_MAX_FINITE_VALUE);
    expect(clampFiniteF16(-F16_MAX_FINITE_VALUE)).toBe(-F16_MAX_FINITE_VALUE);
  });

  it('clamps values above fp16 max to F16_MAX_FINITE (fixes "Inf-seeds-NaN" dots)', () => {
    // Typical HDR sun / strip-light pixels.
    expect(clampFiniteF16(1e5)).toBe(F16_MAX_FINITE_VALUE);
    expect(clampFiniteF16(1e10)).toBe(F16_MAX_FINITE_VALUE);
    expect(clampFiniteF16(Number.MAX_VALUE)).toBe(F16_MAX_FINITE_VALUE);
    expect(clampFiniteF16(Infinity)).toBe(F16_MAX_FINITE_VALUE);
  });

  it('clamps values below -fp16-max to -F16_MAX_FINITE', () => {
    // HDR values are non-negative in practice, but defensive coverage
    // catches a future uploader that passes diff-space / signed data.
    expect(clampFiniteF16(-1e5)).toBe(-F16_MAX_FINITE_VALUE);
    expect(clampFiniteF16(-Infinity)).toBe(-F16_MAX_FINITE_VALUE);
  });

  it('collapses NaN to 0 rather than silently passing it through', () => {
    // The self-compare (`v !== v`) is the canonical NaN test. If a
    // refactor swaps it for `isNaN(...)` and a transpiler mangles
    // that path, a NaN pixel would sneak through to the GPU texture.
    expect(clampFiniteF16(NaN)).toBe(0);
    expect(clampFiniteF16(Number.NaN)).toBe(0);
  });

  it('F16_MAX_FINITE_VALUE matches the IEEE fp16 max (65504)', () => {
    // Sign=0, exp=30 (max normal biased), mant=1023 (max):
    //   (1 + 1023/1024) * 2^(30 - 15) = 65504.
    expect(F16_MAX_FINITE_VALUE).toBe(65504);
  });
});

describe('envmapList — slug allow-list + URL builder', () => {
  it('every entry in ENVMAPS is a known slug', () => {
    // Sanity: the runtime allow-list has to cover the full curated set.
    // A typo'd slug in ENVMAPS would silently 404 on fetch today.
    for (const e of ENVMAPS) {
      expect(isKnownSlug(e.slug)).toBe(true);
    }
  });

  it('rejects unknown slugs (guards stale localStorage)', () => {
    // This is the gate `persistence.ts` uses to block stale
    // localStorage entries from triggering a 404 at boot. A refactor
    // of `.some` → `.every` flips every check — catch it here.
    expect(isKnownSlug('nonexistent_hdri')).toBe(false);
    expect(isKnownSlug('')).toBe(false);
    expect(isKnownSlug(' studio_small_03 ')).toBe(false);  // whitespace-sensitive
    expect(isKnownSlug('STUDIO_SMALL_03')).toBe(false);     // case-sensitive
  });

  it('envmapUrl builds the canonical Poly Haven CDN path', () => {
    // Pin the URL format. A silent refactor dropping `_{size}` from
    // the filename would produce 404s on every fetch; the main-loop
    // catch swallows with a notice, making regression silent.
    expect(envmapUrl('studio_small_03', '1k'))
      .toBe('https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/studio_small_03_1k.hdr');
    expect(envmapUrl('venice_sunset', '2k'))
      .toBe('https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/venice_sunset_2k.hdr');
    expect(envmapUrl('dikhololo_night', '4k'))
      .toBe('https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/4k/dikhololo_night_4k.hdr');
  });

  it('envmapUrl defaults to the project default size when size omitted', () => {
    // Guards against a drive-by change to the default size param that
    // would silently flip every call-site that relies on the default
    // (e.g. a future boot path).
    expect(envmapUrl('studio_small_03'))
      .toBe(`https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/${DEFAULT_ENVMAP_SIZE}/studio_small_03_${DEFAULT_ENVMAP_SIZE}.hdr`);
  });

  it('ENVMAP_SIZES exposes exactly the tiers the URL builder accepts', () => {
    // Keeps the union + the `envmapUrl(size)` param type in sync.
    // Adding a new size in one place and forgetting the other would
    // trip this test.
    expect([...ENVMAP_SIZES]).toEqual(['1k', '2k', '4k']);
  });

  it('DEFAULT_ENVMAP_SLUG points to a known entry', () => {
    expect(isKnownSlug(DEFAULT_ENVMAP_SLUG)).toBe(true);
  });
});

describe('pickRandomSlug', () => {
  it('always returns a slug different from the current one', () => {
    // The "Random" button's load-bearing invariant: double-clicking
    // MUST switch panoramas. A refactor of `while (pick === current)`
    // into `if (...)` would silently leave ~1/N clicks stuck on the
    // same slug.
    const all = ENVMAPS.map(e => e.slug);
    for (const current of all) {
      // Deterministic: for any `rand()`, the loop must find a
      // different entry. Try a handful of seeds.
      for (let seed = 0; seed < 5; seed++) {
        const rand = mulberry32(seed * 7919 + 42);
        const next = pickRandomSlug(current, rand);
        expect(next).not.toBe(current);
        expect(isKnownSlug(next)).toBe(true);
      }
    }
  });

  it('makes progress when the first draw matches the current slug (varied rand)', () => {
    // The `while (pick === current)` loop depends on `rand` producing
    // at least one value that maps to a non-current slug. With a
    // cycling rand (0, 1/N, 2/N, …) we're guaranteed to step past the
    // current entry on some draw. This pins "rand that eventually
    // varies will find a different entry"; an `if → while` refactor
    // on a stuck rand is a separate class of bug (the stuck case
    // WOULD infinite-loop the current implementation, which is why
    // the JSDoc at `pickRandomSlug` documents the `rand`-varies
    // assumption). A hard adversarial `rand = () => 0` test would
    // need an iteration cap in the implementation first — Phase D
    // follow-up.
    const entry0 = ENVMAPS[0]!.slug;
    let call = 0;
    const rand = () => (call++ % ENVMAPS.length) / ENVMAPS.length;
    const next = pickRandomSlug(entry0, rand);
    expect(next).not.toBe(entry0);
  });
});

/** Minimal mulberry32 PRNG for deterministic test draws. Not exported;
 *  kept inline so tests stay self-contained and don't import a dep. */
function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
