import { Pane } from 'tweakpane';
import type { PerfStats } from './perfStats';
import { DIAMOND_SIZE_MIN, DIAMOND_SIZE_MAX, type DiamondView } from './math/diamond';
import { ENVMAPS, DEFAULT_ENVMAP_SLUG, DEFAULT_ENVMAP_SIZE, ENVMAP_SIZES, type EnvmapSize } from './envmapList';
import { defaultShapesParams, mergePrismDims, type ShapesParams } from './shapeParams';
export type { ShapesParams, CommonBodyParams, PlateShapeParams, DiamondShapeParams, PrismBodyParams } from './shapeParams';

/** Bounds for the envmap exposure slider. Exposed so `persistence.ts`
 *  clamps hand-edited / stale localStorage to the exact same range —
 *  mirrors the `PILL_LEN_MIN/MAX` / `DIAMOND_SIZE_MIN/MAX` pattern used
 *  for every other user-tunable range. */
export const ENVMAP_EXPOSURE_MIN = 0.01;
export const ENVMAP_EXPOSURE_MAX = 2.0;
/** Yaw rotation around world-Y in radians. Full turn wraps naturally
 *  via the sampler's `repeat` addressing mode, but the slider and
 *  persistence clamp to [-π, π] so the UI always shows a canonical
 *  representative. */
export const ENVMAP_ROTATION_MIN = -Math.PI;
export const ENVMAP_ROTATION_MAX =  Math.PI;

export type { DiamondView } from './math/diamond';

// Re-exported so persistence.ts can clamp hand-edited storage to the same
// slider range the UI uses, without importing diamond.ts in two places.
export { DIAMOND_SIZE_MIN, DIAMOND_SIZE_MAX };

/** Diamond TIR-bounce loop cap (inspector + GPU `clamp(…,1,32)`). */
export const DIAMOND_TIR_BOUNCES_MIN = 1;
export const DIAMOND_TIR_BOUNCES_MAX = 32;

/** Perspective FOV slider / persistence bounds, in degrees.
 *  Anything outside the range drives `cameraZ = (height/2) / tan(fov/2)` into
 *  ±Infinity / 0, which blows up the projection math. */
export const FOV_MIN = 20;
export const FOV_MAX = 120;

/** Plate wave-amplitude bounds (px). Mirrors the slider range below.
 *  Persistence clamps to these so a hand-edited storage with `waveAmp =
 *  100000` can't push `waveLipFactor` so close to zero that sphereTrace
 *  stalls on every plate fragment (visible as a black plate). The slider
 *  reads the same constants below so the two stay in sync. */
export const WAVE_AMP_MIN = 0;
export const WAVE_AMP_MAX = 60;

/** Plate wavelength bounds (px). The lower bound (60 px) is high enough that
 *  even at WAVE_AMP_MAX (60 px) the worst-case Lipschitz factor stays
 *  ≈ 0.16 — small but non-zero, so `MIN_STEP = 0.5` in sphere-trace keeps
 *  marching forward instead of stalling. The upper bound matches the UI
 *  slider so persistence and slider agree. */
export const WAVE_WAVELENGTH_MIN = 60;
export const WAVE_WAVELENGTH_MAX = 800;

/** Rounded-edge radius bounds (px). Lower bound is 1 (slider min) — `edgeR=0`
 *  is technically valid for the SDF but the slider can't produce it, so
 *  persistence rejects it too to keep the "slider is the source of truth"
 *  invariant intact. Upper bound is 100; the render-loop already clamps
 *  `min(edgeR, hx, hy, hz)` so a stale storage value that exceeds halfSize
 *  is harmless, but clamping at the persistence boundary too prevents a
 *  one-frame visual glitch when oversized values get loaded. */
export const EDGE_R_MIN = 1;
export const EDGE_R_MAX = 100;

/** Pill / cube / plate dimension bounds (px). Slider ranges below — these
 *  exports let persistence reject negative or absurd hand-edited values
 *  that would either invert the SDF (negative half-size makes `abs(p) - h`
 *  always positive → shape vanishes) or fill the entire screen. */
export const PILL_LEN_MIN   = 80;
export const PILL_LEN_MAX   = 800;
export const PILL_SHORT_MIN = 20;
export const PILL_SHORT_MAX = 200;
export const PILL_THICK_MIN = 10;
export const PILL_THICK_MAX = 200;

/** History EMA blend weight bounds. Lower bound 0.05 keeps some smoothing —
 *  going below it makes the wavelength stratification noise visibly flicker
 *  per pixel because each frame's stratum lottery isn't averaged with prior
 *  frames at all. Upper bound 1.0 means "throw away history every frame"
 *  (instant updates, no temporal anything). */
export const HISTORY_ALPHA_MIN = 0.05;
export const HISTORY_ALPHA_MAX = 1.0;

/** Shared Cauchy / refraction — same range as the Spectral sliders + persistence. */
export const N_D_MIN = 1.0;
export const N_D_MAX = 3.5;
export const V_D_MIN = 1;
export const V_D_MAX = 90;
export const REFRACTION_STRENGTH_MIN = 0;
export const REFRACTION_STRENGTH_MAX = 1.0;

/** Antialiasing strategy. `taa` drives in-scene sub-pixel jitter +
 *  motion-vector history reprojection (converges under motion but can
 *  add noise around rotating shapes). `fxaa` is a single-frame post
 *  filter — no jitter, softer edges, zero temporal instability. `none`
 *  disables both strategies for A/B comparison; the history EMA
 *  (controlled by `historyAlpha`) and the passthrough post pass still
 *  run, so moving scenes will still ghost unless historyAlpha = 1. */
export type AaMode = 'none' | 'fxaa' | 'taa';

export type Params = {
  sampleCount: 3 | 8 | 16 | 32 | 64;
  shape: 'pill' | 'prism' | 'cube' | 'plate' | 'diamond';
  /** Cauchy IOR, Abbe number, and overall refraction strength — shared by every shape. */
  n_d:                 number;
  V_d:                 number;
  refractionStrength:  number;
  /** Per-type geometry, plate wave, and diamond controls. `shape` picks which block feeds
   *  the render path; changing the dropdown does not overwrite other types' size sliders. */
  shapes: ShapesParams;
  refractionMode: 'exact' | 'approx';
  temporalJitter: boolean;
  projection: 'ortho' | 'perspective';
  fov: number;  // full vertical field-of-view in degrees
  debugProxy: boolean;  // tint proxy fragments pink
  aaMode: AaMode;
  paused: boolean;  // "Stop the world" — freeze rotation/wave while keeping AA converging
  historyAlpha: number;  // steady-state EMA blend weight (0..1). Lower = more motion blur, less noise; higher = sharper but noisier.
  // HDR environment map controls (Phase C). Replace the Phase A
  // reflSrc hack (background-photo-at-UV-offset) with a proper
  // environment panorama so reflections sample a REAL sky/studio
  // rather than a shifted view of the thing behind the glass.
  envmapEnabled: boolean;
  // Linear-light multiplier on envmap samples. HDR peaks can be
  // 100×+, so exposure acts like a camera aperture: higher =
  // brighter reflections, lower = dimmer. Slider range covers
  // typical HDR dynamic range without the user's eyes blowing out.
  envmapExposure: number;
  // Radians, rotates the sky around world-Y. Users drag the sun
  // without re-downloading a different panorama.
  envmapRotation: number;
  // Poly Haven asset slug — see src/envmapList.ts. Stored verbatim
  // so persistence survives CDN URL-path changes. Validated on load
  // against the known-slug allow-list.
  envmapSlug: string;
  // Resolution tier for the downloaded HDRI. Higher = sharper
  // reflections on crisp bright highlights (strip lights, sun disc)
  // but multiplies the fetch size by 4× per step (1K ~ 2MB, 2K ~ 6MB,
  // 4K ~ 25MB). See ENVMAP_SIZES in src/envmapList.ts.
  envmapSize: EnvmapSize;
  /** Background image for refraction: Picsum fetch, or HTML textarea (HTML-in-Canvas). */
  bgSource: 'photo' | 'html';
};

// Object-style `options` are reordered by Tweakpane (often by `String(value)` or
// label sort). Array-style `{ text, value }[]` keeps declaration order
// (see @tweakpane/core `ListParamsOptions` / `ArrayStyleListOptions`).
const SAMPLE_COUNT_PANE_OPTIONS: { text: string; value: Params['sampleCount'] }[] = [
  { text: '3 (fake RGB)', value: 3 },
  { text: '8 (default)', value: 8 },
  { text: '16', value: 16 },
  { text: '32', value: 32 },
  { text: '64 (max)', value: 64 },
];
const REFRACTION_MODE_PANE_OPTIONS: { text: string; value: Params['refractionMode'] }[] = [
  { text: 'Exact', value: 'exact' },
  { text: 'Approx', value: 'approx' },
];
const SHAPE_PANE_OPTIONS: { text: string; value: Params['shape'] }[] = [
  { text: 'Pill', value: 'pill' },
  { text: 'Prism (rainbow)', value: 'prism' },
  { text: 'Cube (rotating)', value: 'cube' },
  { text: 'Plate (wavy)', value: 'plate' },
  { text: 'Diamond (brilliant)', value: 'diamond' },
];
const DIAMOND_VIEW_PANE_OPTIONS: { text: string; value: DiamondView }[] = [
  { text: 'Free (tumble)', value: 'free' },
  { text: 'Top (table)', value: 'top' },
  { text: 'Side (girdle)', value: 'side' },
  { text: 'Bottom (culet)', value: 'bottom' },
];
const PROJECTION_PANE_OPTIONS: { text: string; value: Params['projection'] }[] = [
  { text: 'Orthographic', value: 'ortho' },
  { text: 'Perspective', value: 'perspective' },
];
const AA_MODE_PANE_OPTIONS: { text: string; value: AaMode }[] = [
  { text: 'None', value: 'none' },
  { text: 'FXAA', value: 'fxaa' },
  { text: 'TAA', value: 'taa' },
];
const ENVMAP_SLUG_PANE_OPTIONS: { text: string; value: string }[] = ENVMAPS.map((e) => ({
  text: `${e.label} (${e.kind})`,
  value: e.slug,
}));
const ENVMAP_SIZE_PANE_OPTIONS: { text: string; value: EnvmapSize }[] = ENVMAP_SIZES.map((s) => ({
  text: s.toUpperCase(),
  value: s,
}));
const BG_SOURCE_PANE_OPTIONS: { text: string; value: Params['bgSource'] }[] = [
  { text: 'Picsum only', value: 'photo' },
  { text: 'Picsum + text (HTML)', value: 'html' },
];

type Preset = {
  label: string;
  apply: (p: Params) => void;
};

type Material = {
  label: string;
  n_d:   number;
  V_d:   number;
};

// Real-world reference values. Sources: Schott glass catalog, Wikipedia
// optical property tables. Diamond's "fire" comes from the high n_d multiplying
// (n_d - 1)/V_d in the Cauchy formula — moderate V_d, strong visual dispersion.
const MATERIALS: readonly Material[] = [
  { label: 'Water',           n_d: 1.333, V_d: 55.7 },
  { label: 'Fused silica',    n_d: 1.458, V_d: 67.7 },
  { label: 'PMMA (acrylic)',  n_d: 1.491, V_d: 58.0 },
  { label: 'Crown (BK7)',     n_d: 1.517, V_d: 64.2 },
  { label: 'Polycarbonate',   n_d: 1.586, V_d: 30.0 },
  { label: 'Lead crystal',    n_d: 1.600, V_d: 32.0 },
  { label: 'Flint (SF10)',    n_d: 1.728, V_d: 28.4 },
  { label: 'Dense flint (SF11)', n_d: 1.785, V_d: 25.4 },
  { label: 'Cubic zirconia',  n_d: 2.150, V_d: 30.0 },
  { label: 'Diamond',         n_d: 2.418, V_d: 55.0 },
  { label: 'Moissanite',      n_d: 2.648, V_d: 31.0 },
  // Fantasy — not real, but fun.
  { label: '✨ Rainbow glass', n_d: 1.55,  V_d: 8.0  },
  { label: '✨ Fire crystal',  n_d: 2.2,   V_d: 12.0 },
  { label: '✨ Unobtanium',    n_d: 3.2,   V_d: 4.0  },
  // Low IOR × extreme dispersion — weak refraction but huge chromatic split
  // per pixel. Rainbow everywhere.
  { label: '✨ Rainbow soap',  n_d: 1.272, V_d: 2.0  },
];

const PRESETS: readonly Preset[] = [
  {
    label: 'Subtle pill',
    apply: (p) => {
      p.shape       = 'pill';
      p.sampleCount = 8;
      p.n_d   = 1.5168;
      p.V_d   = 40;
      p.refractionStrength = 0.1;
      // Mutate in place — Tweakpane bindings hold references from initUi; replacing
      // the object would leave sliders writing stale copies that the render loop
      // never reads.
      Object.assign(p.shapes.pill, { pillLen: 320, pillShort: 88, pillThick: 40, edgeR: 14 });
      p.refractionMode = 'exact';
    },
  },
  {
    label: 'Strong dispersion',
    apply: (p) => {
      p.shape       = 'pill';
      p.sampleCount = 16;
      p.n_d   = 1.6;
      p.V_d   = 18;
      p.refractionStrength = 0.35;
      Object.assign(p.shapes.pill, { pillLen: 320, pillShort: 88, pillThick: 40, edgeR: 14 });
      p.refractionMode = 'exact';
    },
  },
  {
    label: 'Prism rainbow',
    apply: (p) => {
      p.shape        = 'prism';
      p.sampleCount  = 16;
      p.n_d   = 1.6;
      p.V_d   = 12;
      p.refractionStrength = 0.18;
      Object.assign(p.shapes.prism, { pillLen: 400, pillShort: 80, pillThick: 80 });
      p.refractionMode = 'exact';
    },
  },
  {
    label: 'Rotating cube',
    apply: (p) => {
      p.shape       = 'cube';
      p.sampleCount = 16;
      p.n_d   = 1.55;
      p.V_d   = 18;
      p.refractionStrength = 0.2;
      Object.assign(p.shapes.cube, { pillLen: 160, pillShort: 160, pillThick: 160, edgeR: 10 });
      p.refractionMode = 'exact';
    },
  },
  {
    label: 'Wavy plate',
    apply: (p) => {
      p.shape        = 'plate';
      p.sampleCount  = 16;
      p.n_d   = 1.272;
      p.V_d   = 2.0;
      p.refractionStrength = 0.2;
      Object.assign(p.shapes.plate, {
        pillLen: 400, pillThick: 100, edgeR: 4, pillShort: 400,
        waveAmp: 20, waveWavelength: 300,
      });
      p.refractionMode = 'exact';
    },
  },
];

export type PerfBinding = {
  /** Live-updating object the loop writes into; the panel polls it. */
  stats: PerfStats;
  /** Adapter exposes timestamp queries — gates the GPU ms graph in the UI. */
  hasGpuTiming: boolean;
};

export function initUi(
  params: Params,
  reloadPhoto:      () => void,
  onChange:         () => void,
  markSceneChanged: () => void = () => {},
  perf:             PerfBinding | null = null,
  reloadEnvmap:     (slug: string) => void = () => {},
  randomEnvmap:     () => void = () => {},
  /** Called when the user toggles `envmapEnabled` from false → true.
   *  Lets main.ts do a LAZY fetch of the HDRI when envmap was off at
   *  boot (so we didn't download anything), avoiding the fallback
   *  gradient rendering after the user opted back in. */
  onEnvmapEnabled:  () => void = () => {},
  /** When set, shows Background controls for HTML-in-Canvas. */
  htmlBackground:   { supported: true; focusEditor: () => void } | null = null,
): Pane {
  const pane = new Pane({ title: 'Spectral Dispersion', expanded: true });

  const spectral = pane.addFolder({ title: 'Spectral' });
  spectral.addBinding(params, 'sampleCount', { options: SAMPLE_COUNT_PANE_OPTIONS });
  spectral.addBinding(params, 'n_d', { min: 1.0, max: 3.5, step: 0.001, label: 'IOR n_d' });
  spectral.addBinding(params, 'V_d', { min: 1, max: 90, step: 0.5, label: 'Abbe V_d' });
  spectral.addBinding(params, 'refractionStrength', { min: 0, max: 1.0, step: 0.001, label: 'Refraction' });
  spectral.addBinding(params, 'refractionMode', {
    // Approx: one back-face trace shared across all wavelengths (jittered each
    // frame). On this engine it's texture-bandwidth bound, so the speedup vs
    // Exact is modest (~15% at N=32) and dynamic scenes show more variance.
    options: REFRACTION_MODE_PANE_OPTIONS,
  });
  spectral.addBinding(params, 'temporalJitter', { label: 'Temporal jitter' });

  const shape = pane.addFolder({ title: 'Shape' });
  const shapeBinding = shape.addBinding(params, 'shape', { options: SHAPE_PANE_OPTIONS });

  // One inspector subfolder per object type — only the active shape’s folder
  // is visible so each mode gets a dedicated control surface.
  const inspPill = shape.addFolder({ title: 'Pill', expanded: true });
  inspPill.addBinding(params.shapes.pill, 'pillLen',   { min: PILL_LEN_MIN,   max: PILL_LEN_MAX,   step: 1, label: 'Length (X)' });
  inspPill.addBinding(params.shapes.pill, 'pillShort', { min: PILL_SHORT_MIN, max: PILL_SHORT_MAX, step: 1, label: 'Short (Y)'  });
  inspPill.addBinding(params.shapes.pill, 'pillThick', { min: PILL_THICK_MIN, max: PILL_THICK_MAX, step: 1, label: 'Thick (Z)'  });
  inspPill.addBinding(params.shapes.pill, 'edgeR',     { min: EDGE_R_MIN, max: EDGE_R_MAX, step: 0.5, label: 'Edge radius' });

  const inspPrism = shape.addFolder({ title: 'Prism (rainbow)', expanded: true });
  inspPrism.addBinding(params.shapes.prism, 'pillLen',   { min: PILL_LEN_MIN,   max: PILL_LEN_MAX,   step: 1, label: 'Length (X)' });
  inspPrism.addBinding(params.shapes.prism, 'pillShort', { min: PILL_SHORT_MIN, max: PILL_SHORT_MAX, step: 1, label: 'Short (Y)'  });
  inspPrism.addBinding(params.shapes.prism, 'pillThick', { min: PILL_THICK_MIN, max: PILL_THICK_MAX, step: 1, label: 'Thick (Z)'  });

  const inspCube = shape.addFolder({ title: 'Cube (rotating)', expanded: true });
  const cubeSize = { value: params.shapes.cube.pillLen };
  const sizeBinding = inspCube.addBinding(cubeSize, 'value', { min: PILL_LEN_MIN, max: 600, step: 1, label: 'Size' });
  sizeBinding.on('change', () => {
    if (params.shape !== 'cube') return;
    params.shapes.cube.pillLen   = cubeSize.value;
    params.shapes.cube.pillShort = cubeSize.value;
    params.shapes.cube.pillThick = cubeSize.value;
  });
  inspCube.addBinding(params.shapes.cube, 'edgeR', { min: EDGE_R_MIN, max: EDGE_R_MAX, step: 0.5, label: 'Edge radius' });

  const inspPlate = shape.addFolder({ title: 'Plate (wavy)', expanded: true });
  inspPlate.addBinding(params.shapes.plate, 'pillLen', { min: PILL_LEN_MIN, max: PILL_LEN_MAX, step: 1, label: 'Face (square)' });
  inspPlate.addBinding(params.shapes.plate, 'pillThick', { min: PILL_THICK_MIN, max: PILL_THICK_MAX, step: 1, label: 'Thick (Z)' });
  inspPlate.addBinding(params.shapes.plate, 'waveAmp', {
    min: WAVE_AMP_MIN, max: WAVE_AMP_MAX, step: 0.5, label: 'Wave amp',
  });
  inspPlate.addBinding(params.shapes.plate, 'waveWavelength', {
    min: WAVE_WAVELENGTH_MIN, max: WAVE_WAVELENGTH_MAX, step: 1, label: 'Wavelength',
  });
  inspPlate.addBinding(params.shapes.plate, 'edgeR', { min: EDGE_R_MIN, max: EDGE_R_MAX, step: 0.5, label: 'Rim radius' });

  const inspDiamond = shape.addFolder({ title: 'Diamond (brilliant)', expanded: true });
  inspDiamond.addBinding(params.shapes.diamond, 'diamondSize', {
    min: DIAMOND_SIZE_MIN, max: DIAMOND_SIZE_MAX, step: 1, label: 'Girdle size',
  });
  inspDiamond.addBinding(params.shapes.diamond, 'diamondView', {
    label: 'View',
    options: DIAMOND_VIEW_PANE_OPTIONS,
  });
  inspDiamond.addBinding(params.shapes.diamond, 'diamondWireframe',  { label: 'Wireframe' });
  inspDiamond.addBinding(params.shapes.diamond, 'diamondFacetColor', { label: 'Facet color' });
  inspDiamond.addBinding(params.shapes.diamond, 'diamondTirDebug',   { label: 'TIR debug' });
  inspDiamond.addBinding(params.shapes.diamond, 'diamondTirMaxBounces', {
    min: DIAMOND_TIR_BOUNCES_MIN,
    max: DIAMOND_TIR_BOUNCES_MAX,
    step: 1,
    label: 'TIR max bounces (costly ↑)',
  });

  function syncShapeSliders(): void {
    const s = params.shape;
    inspPill.hidden    = s !== 'pill';
    inspPrism.hidden = s !== 'prism';
    inspCube.hidden  = s !== 'cube';
    inspPlate.hidden = s !== 'plate';
    inspDiamond.hidden = s !== 'diamond';

    const isCube    = s === 'cube';
    const isPlate   = s === 'plate';
    if (isCube) {
      const c  = params.shapes.cube;
      const avg = Math.round((c.pillLen + c.pillShort + c.pillThick) / 3);
      cubeSize.value   = avg;
      c.pillLen   = avg;
      c.pillShort = avg;
      c.pillThick = avg;
    } else {
      cubeSize.value = params.shapes.cube.pillLen;
      if (isPlate) {
        params.shapes.plate.pillShort = params.shapes.plate.pillLen;
      }
    }
  }
  syncShapeSliders();
  shapeBinding.on('change', () => { syncShapeSliders(); pane.refresh(); });

  // HDR environment map controls (Phase C). Placed in its own folder so
  // the reflection-source swap + HDRI picker aren't buried in the
  // existing Misc wagon wheel.
  const env = pane.addFolder({ title: 'Environment' });
  const envmapEnabledBinding = env.addBinding(params, 'envmapEnabled', { label: 'HDR env' });
  // Lazy-fetch the HDRI when envmap flips from false → true. Boots
  // with envmapEnabled=false skip the initial download (main.ts
  // optimisation) — this handler kicks off the fetch on opt-in so
  // the user doesn't have to click a separate button to get the
  // panorama they configured.
  const envmapSlugBinding = env.addBinding(params, 'envmapSlug', {
    label: 'Panorama',
    options: ENVMAP_SLUG_PANE_OPTIONS,
  });
  envmapSlugBinding.on('change', (ev) => { reloadEnvmap(ev.value); });
  // Size selector: trade download latency for highlight sharpness.
  // 2K default — sharp enough for diamond facets on retina screens
  // without making the random-panorama button feel sluggish.
  const envmapSizeBinding = env.addBinding(params, 'envmapSize', {
    label: 'Size',
    options: ENVMAP_SIZE_PANE_OPTIONS,
  });
  // Changing size forces a re-fetch of the current slug at the new
  // resolution — envmap textures are immutable, so swap the whole
  // thing rather than try to up/down-sample in-place.
  envmapSizeBinding.on('change', () => { reloadEnvmap(params.envmapSlug); });
  const randomPanoramaBtn = env.addButton({ title: 'Random panorama' });
  randomPanoramaBtn.on('click', () => { randomEnvmap(); });
  const envmapExposureBinding = env.addBinding(params, 'envmapExposure', {
    min: ENVMAP_EXPOSURE_MIN, max: ENVMAP_EXPOSURE_MAX, step: 0.01, label: 'Exposure',
  });
  const envmapRotationBinding = env.addBinding(params, 'envmapRotation', {
    // Radians directly (-π..+π); label says "(rad)" so the user sees
    // the unit. Step = 1° in radians for smooth drag resolution without
    // Tweakpane rounding 0.1° to zero. Value flows unchanged through
    // writeFrame into `frame.envmapRotation`.
    min: ENVMAP_ROTATION_MIN, max: ENVMAP_ROTATION_MAX, step: Math.PI / 180, label: 'Rotation (rad)',
  });
  const reloadPhotoBtn  = env.addButton({ title: 'Reload photo' });
  const randomPhotoBtn  = env.addButton({ title: 'Random photo' });
  reloadPhotoBtn.on('click', reloadPhoto);
  randomPhotoBtn.on('click', reloadPhoto);

  let bgSourceBinding: ReturnType<typeof env.addBinding> | null = null;
  let editHtmlBgBtn: ReturnType<typeof env.addButton> | null = null;
  if (htmlBackground) {
    bgSourceBinding = env.addBinding(params, 'bgSource', {
      label: 'Background',
      options: BG_SOURCE_PANE_OPTIONS,
    });
    bgSourceBinding.on('change', () => {
      syncEnvHdrControls();
      pane.refresh();
      onChange();
      markSceneChanged();
    });
    editHtmlBgBtn = env.addButton({ title: 'Edit text…' });
    editHtmlBgBtn.on('click', () => {
      if (params.bgSource !== 'html') return;
      htmlBackground.focusEditor();
    });
  }

  function syncEnvHdrControls(): void {
    const on = params.envmapEnabled;
    envmapSlugBinding.hidden   = !on;
    envmapSizeBinding.hidden  = !on;
    randomPanoramaBtn.hidden  = !on;
    envmapExposureBinding.hidden = !on;
    envmapRotationBinding.hidden  = !on;
    const htmlOn = htmlBackground !== null && params.bgSource === 'html';
    // Picsum: "Random photo" fetches a new seed (HTML underlay + GPU texture) whenever
    // HDR is off, or the background is the HTML+photo composite (even with HDR on).
    randomPhotoBtn.hidden = on && !htmlOn;
    // Legacy "Reload" only when a lone Picsum pass would be visible (not superseded by Random).
    reloadPhotoBtn.hidden = on || htmlOn || (!on && params.bgSource === 'photo');
    if (bgSourceBinding) {
      bgSourceBinding.hidden = false;
    }
    if (editHtmlBgBtn) {
      editHtmlBgBtn.hidden = !htmlOn;
    }
  }
  syncEnvHdrControls();

  const misc = pane.addFolder({ title: 'Misc' });
  misc.addBinding(params, 'projection', { options: PROJECTION_PANE_OPTIONS });
  misc.addBinding(params, 'fov', { min: FOV_MIN, max: FOV_MAX, step: 1, label: 'FOV°' });
  misc.addBinding(params, 'paused', { label: 'Stop the world' });
  misc.addBinding(params, 'aaMode', { label: 'AA', options: AA_MODE_PANE_OPTIONS });
  // History blend weight controls the steady-state EMA. Higher = sharper
  // motion (less ghost) but noisier. The shader auto-adapts toward 1.0
  // when it detects color jumps (variance clamp), so for fast motion the
  // effective alpha is bumped automatically — this slider just sets the
  // baseline for static / slow-motion frames.
  misc.addBinding(params, 'historyAlpha', {
    min: HISTORY_ALPHA_MIN, max: HISTORY_ALPHA_MAX, step: 0.01,
    label: 'History α',
  });
  misc.addBinding(params, 'debugProxy', { label: 'Show proxy' });

  const presets = pane.addFolder({ title: 'Presets' });
  for (const preset of PRESETS) {
    const btn = presets.addButton({ title: preset.label });
    btn.on('click', () => {
      preset.apply(params);
      // Presets can switch shape AND change pill dims, so re-evaluate which
      // sliders are visible and reseed cubeSize before the refresh paints.
      syncShapeSliders();
      pane.refresh();
      markSceneChanged();
      onChange();
    });
  }

  // Materials only change n_d + V_d — refraction strength and per-shape
  // sizes stay as-is.
  const materials = pane.addFolder({ title: 'Materials', expanded: false });
  for (const m of MATERIALS) {
    const btn = materials.addButton({ title: `${m.label}  (n=${m.n_d}, V=${m.V_d})` });
    btn.on('click', () => {
      params.n_d = m.n_d;
      params.V_d = m.V_d;
      pane.refresh();
      markSceneChanged();
      onChange();
    });
  }

  // Live perf monitor — fed by the render loop via the `perf.stats` object.
  // Tweakpane's graph view hides the numeric value until hover; pair each
  // metric with a text monitor (same underlying numbers via getters) so the
  // readout is always visible. The graph row label is kept minimal (trend only).
  if (perf) {
    const stats  = perf.stats;
    const live = {
      get fps() { return stats.fps; },
      get cpuMs() { return stats.cpuMs; },
      get gpuMs() { return stats.gpuMs; },
    };
    const monitor  = pane.addFolder({ title: 'Perf', expanded: true });
    const pollMs   = 250;
    // Values first — always show numbers; `format` is supported by the number monitor plugin.
    const fmtMs   = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : '—');
    const fmtGpu = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : '—');
    monitor.addBinding(live, 'fps', {
      readonly:   true,
      label:      'FPS',
      interval:   pollMs,
      format:     (v: number) => v.toFixed(1),
    });
    monitor.addBinding(perf.stats, 'fps', {
      readonly:   true,
      label:      'FPS trend',
      view:       'graph',
      min:        0,
      max:        120,
      interval:   pollMs,
    });
    monitor.addBinding(live, 'cpuMs', {
      readonly:   true,
      label:      'CPU ms',
      interval:   pollMs,
      format:     fmtMs,
    });
    monitor.addBinding(perf.stats, 'cpuMs', {
      readonly:   true,
      label:      'CPU trend',
      view:       'graph',
      min:        0,
      max:        33,
      interval:   pollMs,
    });
    if (perf.hasGpuTiming) {
      monitor.addBinding(live, 'gpuMs', {
        readonly:   true,
        label:      'GPU ms (scene)',
        interval:   pollMs,
        format:     fmtGpu,
      });
      monitor.addBinding(perf.stats, 'gpuMs', {
        readonly:   true,
        label:      'GPU (scene) trend',
        view:       'graph',
        min:        0,
        max:        16,
        interval:   pollMs,
      });
    }
  }

  // `ev.last === true` marks a committed value (slider release, dropdown pick,
  // checkbox toggle) — use those as scene-change triggers so the history clears
  // when geometry jumps. Mid-drag slider ticks (last: false) keep their
  // temporal smoothing.
  //
  // We subscribe per-folder (skipping Perf) instead of the whole pane because
  // tweakpane's readonly graph bindings emit a `change` event on every poll —
  // a global `pane.on('change')` handler would call markSceneChanged 4 times
  // a second and visibly clobber the temporal accumulation on the cube.
  // Tweakpane exposes `last: boolean` on both folder-level and
  // per-binding change events, but the concrete type shapes differ
  // (`{last}` for folders, `TpChangeEvent<T>` for bindings). Use a
  // permissive structural shape so both subscription surfaces accept
  // this callback — JS ignores the extra fields on bindings.
  const onUserChange = (ev: { readonly last?: boolean }) => {
    if (ev.last) markSceneChanged();
    onChange();
  };
  spectral.on('change',  onUserChange);
  shape.on('change',     onUserChange);
  misc.on('change',      onUserChange);
  // Environment folder is subscribed PER-BINDING rather than at folder
  // level because slug/size changes must NOT persist until the async
  // reloadEnvmap fetch succeeds — a folder-level subscription would
  // commit a failed-fetch slug/size to localStorage before we know
  // whether the new panorama loaded (Codex P2 regression flagged iter 2).
  // Slug/size bindings have their own `.on('change')` handlers that
  // delegate to reloadEnvmap → persist + markSceneChanged on success
  // only. Exposure/rotation/enabled mutate params directly and DO need
  // the change hook so a) config survives reload, b) TAA history
  // doesn't ghost an old exposure into the next frame on a paused
  // scene.
  //
  // Per-binding `.on('change')` fires for each committed user change
  // (no intermediate/last distinction exists at the binding level —
  // that's a folder-level aggregation), so we always want both
  // markSceneChanged + onChange, no `last` gating needed.
  const onBindingChange = () => { markSceneChanged(); onChange(); };
  envmapEnabledBinding.on('change', (ev) => {
    if (ev.value) onEnvmapEnabled();
    syncEnvHdrControls();
    pane.refresh();
    onBindingChange();
  });
  envmapExposureBinding.on('change', onBindingChange);
  envmapRotationBinding.on('change', onBindingChange);

  return pane;
}

export function defaultParams(): Params {
  // Opening scene: four Rainbow-soap cubes in perspective. Extreme dispersion
  // (V_d = 2) over a random Picsum photo gives a dramatic per-wavelength
  // rainbow on every rotating face — the kind of first impression the spectral
  // pipeline exists to produce. Users can switch materials or shape from the
  // Tweakpane Presets / Materials folders.
  return {
    sampleCount: 16,
    shape: 'cube',
    n_d: 1.272,
    V_d: 2.0,
    refractionStrength: 0.2,
    shapes:      defaultShapesParams(),
    refractionMode: 'exact',
    temporalJitter: true,
    projection: 'perspective',
    fov: 60,
    debugProxy: false,
    aaMode: 'taa',
    paused: false,
    historyAlpha: 0.2,
    envmapEnabled: true,
    // Exposure = 0.25 keeps typical HDRI peaks (studio strip-lights at
    // 50-200, sunny sky at 5-20) inside the [0, 2] display range without
    // flattening the mid-tones. Users with brighter/dimmer HDRIs tune
    // via the slider.
    envmapExposure: 0.25,
    envmapRotation: 0,
    envmapSlug: DEFAULT_ENVMAP_SLUG,
    envmapSize: DEFAULT_ENVMAP_SIZE,
    // Prefer the HTML layer when the browser exposes HTML-in-Canvas; main
    // maps back to Picsum if `copyElementImageToTexture` is missing.
    bgSource: 'html',
  };
}

export function mergeParams(base: Params, patch: Partial<Params>): Params {
  const s = patch.shapes;
  const shapes = s
    ? {
        pill:    { ...base.shapes.pill,    ...s.pill },
        prism:   mergePrismDims(base.shapes.prism, s.prism),
        cube:    { ...base.shapes.cube,    ...s.cube },
        plate:   { ...base.shapes.plate,   ...s.plate },
        diamond: { ...base.shapes.diamond, ...s.diamond },
      }
    : base.shapes;
  return { ...base, ...patch, shapes };
}
