import { Pane } from 'tweakpane';
import type { PerfStats } from './perfStats';

/** Perspective FOV slider / persistence bounds, in degrees.
 *  Anything outside the range drives `cameraZ = (height/2) / tan(fov/2)` into
 *  ±Infinity / 0, which blows up the projection math. */
export const FOV_MIN = 20;
export const FOV_MAX = 120;

export type Params = {
  sampleCount: 3 | 8 | 16 | 32 | 64;
  shape: 'pill' | 'prism' | 'cube' | 'plate';
  n_d: number;
  V_d: number;
  pillLen: number;
  pillShort: number;
  pillThick: number;
  edgeR: number;
  refractionStrength: number;
  refractionMode: 'exact' | 'approx';
  temporalJitter: boolean;
  projection: 'ortho' | 'perspective';
  fov: number;  // full vertical field-of-view in degrees
  debugProxy: boolean;  // tint proxy fragments pink
  // Plate-only wave controls. Amp in pixels (midsurface z-displacement);
  // wavelength in pixels (converted to angular frequency 2π/wavelength on
  // the GPU side via the waveFreq uniform). Exposed as length so the UI
  // label reads naturally; the conversion happens in main.ts.
  waveAmp: number;
  waveWavelength: number;
};

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
      p.shape              = 'pill';
      p.sampleCount        = 8;
      p.n_d                = 1.5168;
      p.V_d                = 40;
      p.pillLen            = 320;
      p.pillShort          = 88;
      p.pillThick          = 40;
      p.edgeR              = 14;
      p.refractionStrength = 0.1;
      p.refractionMode     = 'exact';
    },
  },
  {
    label: 'Strong dispersion',
    apply: (p) => {
      p.shape              = 'pill';
      p.sampleCount        = 16;
      p.n_d                = 1.6;
      p.V_d                = 18;
      p.pillLen            = 320;
      p.pillShort          = 88;
      p.pillThick          = 40;
      p.edgeR              = 14;
      p.refractionStrength = 0.35;
      p.refractionMode     = 'exact';
    },
  },
  {
    label: 'Prism rainbow',
    apply: (p) => {
      p.shape              = 'prism';
      p.sampleCount        = 16;
      p.n_d                = 1.6;
      p.V_d                = 12;
      p.pillLen            = 400;
      p.pillShort          = 80;
      p.pillThick          = 80;
      p.edgeR              = 4;
      p.refractionStrength = 0.18;
      p.refractionMode     = 'exact';
    },
  },
  {
    label: 'Rotating cube',
    apply: (p) => {
      p.shape              = 'cube';
      p.sampleCount        = 16;
      p.n_d                = 1.55;
      p.V_d                = 18;
      p.pillLen            = 160;
      p.pillShort          = 160;
      p.pillThick          = 160;
      p.edgeR              = 10;
      p.refractionStrength = 0.2;
      p.refractionMode     = 'exact';
    },
  },
  {
    label: 'Wavy plate',
    apply: (p) => {
      // Thick square plate, tumbling. Constant-thickness bent sheet means the
      // chromatic effect tracks the bend in the midsurface, so pairing with
      // extreme-dispersion "Rainbow soap" paints the ripple pattern with
      // rainbow everywhere on the photo behind.
      p.shape              = 'plate';
      p.sampleCount        = 16;
      p.n_d                = 1.272;
      p.V_d                = 2.0;
      p.pillLen            = 400;  // square face
      p.pillShort          = 400;  // unused (forced = pillLen in main.ts)
      p.pillThick          = 100;  // thick slab — thin plates lose chroma
      p.edgeR              = 15;   // unused for plate (slider hidden)
      p.waveAmp            = 20;
      p.waveWavelength     = 300;
      p.refractionStrength = 0.2;
      p.refractionMode     = 'exact';
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
): Pane {
  const pane = new Pane({ title: 'Spectral Dispersion', expanded: true });

  const spectral = pane.addFolder({ title: 'Spectral' });
  spectral.addBinding(params, 'sampleCount', {
    options: { '3 (fake RGB)': 3, '8 (default)': 8, '16': 16, '32': 32, '64 (max)': 64 },
  });
  spectral.addBinding(params, 'n_d', { min: 1.0, max: 3.5, step: 0.001, label: 'IOR n_d' });
  spectral.addBinding(params, 'V_d', { min: 1,   max: 90,  step: 0.5,   label: 'Abbe V_d' });
  spectral.addBinding(params, 'refractionMode', {
    // Approx: one back-face trace shared across all wavelengths (jittered each
    // frame). On this engine it's texture-bandwidth bound, so the speedup vs
    // Exact is modest (~15% at N=32) and dynamic scenes show more variance.
    options: { Exact: 'exact', Approx: 'approx' },
  });
  spectral.addBinding(params, 'temporalJitter', { label: 'Temporal jitter' });

  const shape = pane.addFolder({ title: 'Shape' });
  const shapeBinding = shape.addBinding(params, 'shape', {
    options: {
      Pill:              'pill',
      'Prism (rainbow)': 'prism',
      'Cube (rotating)': 'cube',
      'Plate (wavy)':    'plate',
    },
  });
  // Pill / prism keep the three independent X/Y/Z sliders (their geometries
  // care about the asymmetry — a tall thin pill vs a wide slab look very
  // different). Cube collapses them into a single "Size" slider since all
  // three half-extents must stay equal for the rotation to be a true cube.
  const lenBinding   = shape.addBinding(params, 'pillLen',   { min: 80, max: 800, step: 1, label: 'Length (X)' });
  const shortBinding = shape.addBinding(params, 'pillShort', { min: 20, max: 200, step: 1, label: 'Short (Y)'  });
  const thickBinding = shape.addBinding(params, 'pillThick', { min: 10, max: 200, step: 1, label: 'Thick (Z)'  });
  // Cube size proxy — writes to all three pill dims so the existing per-pill
  // halfSize.xyz pipeline doesn't need a separate code path.
  const cubeSize = { value: params.pillLen };
  const sizeBinding = shape.addBinding(cubeSize, 'value', { min: 80, max: 600, step: 1, label: 'Size' });
  sizeBinding.on('change', () => {
    if (params.shape !== 'cube') return;
    params.pillLen   = cubeSize.value;
    params.pillShort = cubeSize.value;
    params.pillThick = cubeSize.value;
  });
  const edgeBinding = shape.addBinding(params, 'edgeR', { min: 1, max: 100, step: 0.5, label: 'Edge radius' });
  // Plate-only wave controls. Both stay hidden for pill/prism/cube and
  // animate in only when shape === 'plate' (syncShapeSliders below).
  const waveAmpBinding = shape.addBinding(params, 'waveAmp', {
    min: 0, max: 60, step: 0.5, label: 'Wave amp',
  });
  const waveLenBinding = shape.addBinding(params, 'waveWavelength', {
    min: 60, max: 800, step: 1, label: 'Wavelength',
  });

  // Show the right subset of sliders for each shape. Pill/prism use all three
  // X/Y/Z; cube collapses into a single Size slider (halfSize must stay equal
  // for the rotation to be a true cube); plate uses Length (square face,
  // hy ≡ hx forced in main.ts) + Thick, so we hide the Short slider. Also
  // keeps cubeSize in sync with pillLen so switching shape mid-session
  // doesn't surprise the user.
  function syncShapeSliders(): void {
    const isCube  = params.shape === 'cube';
    const isPlate = params.shape === 'plate';
    lenBinding.hidden      = isCube;
    shortBinding.hidden    = isCube || isPlate;
    thickBinding.hidden    = isCube;
    sizeBinding.hidden     = !isCube;
    edgeBinding.hidden     = isPlate;   // plate reuses the field? No — wave controls below. Hide edgeR.
    waveAmpBinding.hidden  = !isPlate;
    waveLenBinding.hidden  = !isPlate;
    if (isCube) {
      // Average the three dims to seed Size — covers the case where shape
      // was just switched from pill/prism with non-equal extents.
      const avg = Math.round((params.pillLen + params.pillShort + params.pillThick) / 3);
      cubeSize.value   = avg;
      params.pillLen   = avg;
      params.pillShort = avg;
      params.pillThick = avg;
    } else {
      cubeSize.value = params.pillLen;
      if (isPlate) {
        // Mirror pillLen into pillShort so persistence / non-plate shapes
        // inherit a sensible value if the user later switches back.
        params.pillShort = params.pillLen;
      }
    }
  }
  syncShapeSliders();
  shapeBinding.on('change', () => { syncShapeSliders(); pane.refresh(); });

  const misc = pane.addFolder({ title: 'Misc' });
  misc.addBinding(params, 'refractionStrength', { min: 0, max: 1.0, step: 0.001, label: 'Refraction' });
  misc.addBinding(params, 'projection', {
    options: { Orthographic: 'ortho', Perspective: 'perspective' },
  });
  misc.addBinding(params, 'fov', { min: FOV_MIN, max: FOV_MAX, step: 1, label: 'FOV°' });
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
  const reload = presets.addButton({ title: 'Reload photo' });
  reload.on('click', reloadPhoto);

  // Materials only change n_d + V_d — leaves shape/size/refraction strength
  // alone so you can compare glass types on the same geometry.
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
  // Tweakpane's monitor bindings poll the fields directly (interval ms below),
  // so neither the loop nor the UI has to call `pane.refresh()`.
  if (perf) {
    const monitor = pane.addFolder({ title: 'Perf', expanded: true });
    monitor.addBinding(perf.stats, 'fps', {
      readonly: true,
      label:    'FPS',
      view:     'graph',
      min:      0,
      max:      120,
      interval: 250,
    });
    monitor.addBinding(perf.stats, 'cpuMs', {
      readonly: true,
      label:    'CPU ms',
      view:     'graph',
      min:      0,
      max:      33,
      interval: 250,
    });
    if (perf.hasGpuTiming) {
      monitor.addBinding(perf.stats, 'gpuMs', {
        readonly: true,
        label:    'GPU ms',
        view:     'graph',
        min:      0,
        max:      16,
        interval: 250,
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
  const onUserChange = (ev: { last: boolean }) => {
    if (ev.last) markSceneChanged();
    onChange();
  };
  spectral.on('change',  onUserChange);
  shape.on('change',     onUserChange);
  misc.on('change',      onUserChange);

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
    pillLen: 300,
    pillShort: 300,
    pillThick: 300,
    edgeR: 30,
    refractionStrength: 0.2,
    refractionMode: 'exact',
    temporalJitter: true,
    projection: 'perspective',
    fov: 60,
    debugProxy: false,
    waveAmp: 20,
    waveWavelength: 300,
  };
}

export function mergeParams(base: Params, patch: Partial<Params>): Params {
  return { ...base, ...patch };
}
