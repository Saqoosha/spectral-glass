/**
 * Curated list of Poly Haven HDRI panoramas for the random-picker UI.
 *
 * Entries were picked for visual variety with a jewellery/diamond
 * rendering bias: every slot needs SOMETHING bright (sun disc, a
 * strip light, a neon tube) to drive the Fresnel highlight that
 * makes the diamond's facets sparkle against its refraction layer.
 * Uniformly-lit cloudy skies look "dead" on the crown — no glint
 * sources — so they're deliberately absent here.
 *
 * All entries are Poly Haven CC0 assets: `https://polyhaven.com`.
 * No attribution required for personal or commercial use. Each asset
 * ships at multiple resolutions via the `ENVMAP_SIZES` tier below
 * (1K / 2K / 4K); `DEFAULT_ENVMAP_SIZE = '2k'` balances diamond-facet
 * highlight sharpness on retina screens against the random-panorama
 * fetch latency. Users can drop to 1K on slow networks or bump to 4K
 * for hero-shot quality via the UI.
 */

export type EnvmapEntry = {
  /** Poly Haven asset slug — maps 1:1 to the URL on their CDN.
   *
   *  Design debate (review-fix-loop iter 1 + iter 2 — STUCK): the slug
   *  could be a string-literal union derived from `typeof
   *  ENVMAPS[number]['slug']` for compile-time validation at every
   *  call site. Deferred because `Tweakpane.addBinding(params,
   *  'envmapSlug')` widens the value back to `string` when the user
   *  picks from the dropdown, and narrowing every consumer with a
   *  `isKnownSlug` type guard adds friction at the UI boundary
   *  without catching real bugs (persistence + UI dropdown already
   *  funnel all legitimate values through the runtime allow-list). */
  readonly slug:  string;
  /** Human-readable name for the UI dropdown. */
  readonly label: string;
  /** Category tag, shown next to the label so the user can tell at a
   *  glance what kind of lighting they're jumping to. 'sunset' and
   *  'night' are split out from the broader 'outdoor' bucket because
   *  their colour temperature and highlight distribution change the
   *  diamond's appearance enough to warrant a distinct label. */
  readonly kind:  'studio' | 'indoor' | 'outdoor' | 'sunset' | 'night';
};

export const ENVMAPS: readonly EnvmapEntry[] = [
  // Studios — bright, controlled point lights. Best for showing off
  // individual Fresnel highlights on crown facets.
  { slug: 'studio_small_03',            label: 'Small Studio',        kind: 'studio'  },
  { slug: 'studio_small_08',            label: 'Small Studio 2',      kind: 'studio'  },
  { slug: 'studio_small_09',            label: 'Small Studio 3',      kind: 'studio'  },
  { slug: 'photo_studio_01',            label: 'Photo Studio',        kind: 'studio'  },
  { slug: 'brown_photostudio_02',       label: 'Brown Studio',        kind: 'studio'  },
  { slug: 'brown_photostudio_06',       label: 'Brown Studio 2',      kind: 'studio'  },
  { slug: 'neon_photostudio',           label: 'Neon Studio',         kind: 'studio'  },
  { slug: 'blue_photo_studio',          label: 'Blue Studio',         kind: 'studio'  },
  { slug: 'christmas_photo_studio_04',  label: 'Christmas Studio',    kind: 'studio'  },

  // Indoor — enclosed spaces with mixed natural + artificial light.
  // Reflections read as "sitting on a warm desk" vs studio's "lit
  // for a product shoot".
  { slug: 'pine_attic',                 label: 'Pine Attic',          kind: 'indoor'  },

  // Outdoor daylight — natural sun + sky. Large bright disc (sun)
  // drives hot sparkle on any facet facing that direction.
  { slug: 'royal_esplanade',                         label: 'Royal Esplanade',   kind: 'outdoor' },
  { slug: 'kloofendal_43d_clear_puresky',            label: 'Clear Sky',         kind: 'outdoor' },
  { slug: 'kloofendal_48d_partly_cloudy_puresky',    label: 'Partly Cloudy',     kind: 'outdoor' },
  { slug: 'autumn_forest_04',                        label: 'Autumn Forest',     kind: 'outdoor' },
  { slug: 'spaichingen_hill',                        label: 'Spaichingen Hill',  kind: 'outdoor' },

  // Golden hour — warm low sun, strong directional shadows.
  // Produces orange/red reflections on the crown, cooler blue sky
  // bounce on the pavilion side.
  { slug: 'belfast_sunset_puresky',     label: 'Belfast Sunset',      kind: 'sunset'  },
  { slug: 'golden_gate_hills',          label: 'Golden Gate Hills',   kind: 'sunset'  },
  { slug: 'venice_sunset',              label: 'Venice Sunset',       kind: 'sunset'  },
  { slug: 'kiara_1_dawn',               label: 'Dawn',                kind: 'sunset'  },

  // Night — minimal light sources, diamonds look "closed" except
  // where a moon/point light reflects. Good for stress-testing the
  // TIR fallback path.
  { slug: 'dikhololo_night',            label: 'Starry Night',        kind: 'night'   },
  { slug: 'satara_night',               label: 'Savanna Night',       kind: 'night'   },

  // Urban daylight — city scenes with lots of surface detail (shop
  // signs, windows, buildings). Tagged `outdoor` since the sky still
  // dominates the upper hemisphere; keeps the kind union compact
  // rather than splitting off an `urban` label with only 2 entries.
  { slug: 'little_paris_eiffel_tower',  label: 'Paris',               kind: 'outdoor' },
  { slug: 'shanghai_bund',              label: 'Shanghai',            kind: 'outdoor' },
] as const;

/** Default slug used before the user picks one — the first entry
 *  above. Picked rather than selected randomly so the initial frame
 *  is deterministic for regression screenshots. */
export const DEFAULT_ENVMAP_SLUG: string = ENVMAPS[0]!.slug;

/** Size tier for Poly Haven HDRI downloads. Exposed as a user toggle
 *  (UI) because the angular resolution / file-size trade-off depends
 *  on how much the user cares about crisp bright-highlight reflections
 *  vs. first-click latency. Approximate file sizes per panorama:
 *    1k → ~1.5–2  MB (~0.35° per texel; fine for small viewports)
 *    2k → ~5–8    MB (~0.18° per texel; sharp strip-light reflections)
 *    4k → ~20–30  MB (~0.09° per texel; overkill except on large
 *                    screens at very oblique angles)
 *  8K is available on Poly Haven but files push ~100 MB per download
 *  which makes the random-panorama workflow unusable; deliberately
 *  capped at 4K here. */
export const ENVMAP_SIZES = ['1k', '2k', '4k'] as const;
export type EnvmapSize = typeof ENVMAP_SIZES[number];

/** 2K balances visual quality vs download latency — sharp enough for
 *  diamond reflection highlights on typical screens without making
 *  the random-panorama button feel sluggish. Users can bump to 4K
 *  for hero-shot quality or drop to 1K on slow connections. */
export const DEFAULT_ENVMAP_SIZE: EnvmapSize = '2k';

/** Build the canonical Poly Haven CDN URL for a slug at a given size.
 *  Factored out so persistence can store just the slug + size (smaller
 *  + robust if Poly Haven's hosting domain changes) and the URL is
 *  rebuilt on load. */
export function envmapUrl(slug: string, size: EnvmapSize = DEFAULT_ENVMAP_SIZE): string {
  return `https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/${size}/${slug}_${size}.hdr`;
}

/** Pick a random slug, different from the current one so double-clicks
 *  on "Random" always produce a visible change. Returns the current
 *  slug only when the list has a single entry (defensive corner). */
export function pickRandomSlug(current: string, rand: () => number = Math.random): string {
  if (ENVMAPS.length <= 1) return current;
  let pick = current;
  while (pick === current) {
    const idx = Math.floor(rand() * ENVMAPS.length);
    pick = ENVMAPS[idx]!.slug;
  }
  return pick;
}

/** Validate that a string corresponds to a known slug. Used by
 *  persistence.ts so a hand-edited / stale localStorage entry doesn't
 *  cause a 404 on the next startup. */
export function isKnownSlug(slug: string): boolean {
  return ENVMAPS.some(e => e.slug === slug);
}
