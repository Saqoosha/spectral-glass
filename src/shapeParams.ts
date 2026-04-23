import {
  DIAMOND_SIZE_MIN, DIAMOND_SIZE_MAX,
  DIAMOND_VIEW_VALUES, type DiamondView,
} from './math/diamond';
// Mirror `ui.ts` slider bounds — do not import `ui` (would circular with `shapeParams`).
const DIAMOND_TIR_BOUNCES_MIN = 1;
const DIAMOND_TIR_BOUNCES_MAX = 32;
const PILL_LEN_MIN = 80;
const PILL_LEN_MAX   = 800;
const PILL_SHORT_MIN = 20;
const PILL_SHORT_MAX = 200;
const PILL_THICK_MIN = 10;
const PILL_THICK_MAX = 200;
const EDGE_R_MIN   = 1;
const EDGE_R_MAX   = 100;
const WAVE_AMP_MIN = 0;
const WAVE_AMP_MAX   = 60;
const WAVE_WAVELENGTH_MIN = 60;
const WAVE_WAVELENGTH_MAX = 800;

/**
 * Per-object-type geometry. IOR, Abbe, and refraction strength live on
 * `Params` (shared) — see `frameFieldsFromParams`.
 */
export type CommonBodyParams = {
  pillLen:             number;
  pillShort:           number;
  pillThick:           number;
  edgeR:               number;
};

/** Triangular prism: sharp edges only (no rim rounding) — `edgeR` is not used. */
export type PrismBodyParams = {
  pillLen:   number;
  pillShort: number;
  pillThick: number;
};

export type PlateShapeParams = CommonBodyParams & {
  waveAmp:         number;
  waveWavelength:  number;
};

export type DiamondShapeParams = {
  diamondSize:          number;
  diamondWireframe:     boolean;
  diamondFacetColor:    boolean;
  diamondTirDebug:      boolean;
  diamondTirMaxBounces: number;
  diamondView:          DiamondView;
};

export type ShapesParams = {
  pill:    CommonBodyParams;
  prism:   PrismBodyParams;
  cube:    CommonBodyParams;
  plate:   PlateShapeParams;
  diamond: DiamondShapeParams;
};

const DEF_BODY: CommonBodyParams = {
  pillLen: 300, pillShort: 300, pillThick: 300, edgeR: 30,
};

const DEF_PLATE: PlateShapeParams = {
  ...DEF_BODY,
  waveAmp: 20, waveWavelength: 300,
};

const DEF_DIAMOND: DiamondShapeParams = {
  diamondSize:        200,
  diamondWireframe:   false,
  diamondFacetColor:  false,
  diamondTirDebug:    false,
  diamondTirMaxBounces: 6,
  diamondView:        'free',
};

export function defaultShapesParams(): ShapesParams {
  return {
    pill:    { ...DEF_BODY },
    prism:   { pillLen: 400, pillShort: 80, pillThick: 80 },
    cube:    { ...DEF_BODY, pillLen: 300, pillShort: 300, pillThick: 300, edgeR: 30 },
    plate:   { ...DEF_PLATE, pillLen: 400, pillThick: 100, edgeR: 4 },
    diamond: { ...DEF_DIAMOND },
  };
}

export type ParamsShape = 'pill' | 'prism' | 'cube' | 'plate' | 'diamond';

/** Read-only view of the active shape for GPU + pill sizing (one place for main). */
export type ActiveFrameFields = {
  n_d:                  number;
  V_d:                  number;
  refractionStrength:   number;
  pillLen:              number;
  pillShort:            number;
  pillThick:            number;
  edgeR:                number;
  waveAmp:              number;
  waveWavelength:       number;
  diamondSize:          number;
  diamondWireframe:     boolean;
  diamondFacetColor:    boolean;
  diamondTirDebug:      boolean;
  diamondTirMaxBounces: number;
  diamondView:          DiamondView;
};

/** Flatten `params` for `writeFrame` and the per-frame pill write in main. */
export function frameFieldsFromParams(params: {
  shape:  ParamsShape;
  shapes: ShapesParams;
  n_d:                 number;
  V_d:                 number;
  refractionStrength:  number;
}): ActiveFrameFields {
  const { shape, shapes, n_d, V_d, refractionStrength } = params;
  if (shape === 'diamond') {
    const d = shapes.diamond;
    return {
      n_d, V_d, refractionStrength,
      pillLen: d.diamondSize, pillShort: d.diamondSize, pillThick: d.diamondSize, edgeR: 0,
      waveAmp:        shapes.plate.waveAmp,
      waveWavelength: shapes.plate.waveWavelength,
      diamondSize:    d.diamondSize,
      diamondWireframe: d.diamondWireframe, diamondFacetColor: d.diamondFacetColor, diamondTirDebug: d.diamondTirDebug,
      diamondTirMaxBounces: d.diamondTirMaxBounces, diamondView: d.diamondView,
    };
  }
  if (shape === 'plate') {
    const p = shapes.plate;
    return {
      n_d, V_d, refractionStrength,
      pillLen: p.pillLen, pillShort: p.pillLen, pillThick: p.pillThick, edgeR: p.edgeR,
      waveAmp: p.waveAmp, waveWavelength: p.waveWavelength,
      diamondSize:          shapes.diamond.diamondSize,
      diamondWireframe:     shapes.diamond.diamondWireframe, diamondFacetColor: shapes.diamond.diamondFacetColor,
      diamondTirDebug:      shapes.diamond.diamondTirDebug, diamondTirMaxBounces: shapes.diamond.diamondTirMaxBounces,
      diamondView:          shapes.diamond.diamondView,
    };
  }
  if (shape === 'prism') {
    const p = shapes.prism;
    return {
      n_d, V_d, refractionStrength,
      pillLen: p.pillLen, pillShort: p.pillShort, pillThick: p.pillThick, edgeR: 0,
      waveAmp:        shapes.plate.waveAmp,
      waveWavelength: shapes.plate.waveWavelength,
      diamondSize:          shapes.diamond.diamondSize,
      diamondWireframe:     shapes.diamond.diamondWireframe, diamondFacetColor: shapes.diamond.diamondFacetColor,
      diamondTirDebug:      shapes.diamond.diamondTirDebug, diamondTirMaxBounces: shapes.diamond.diamondTirMaxBounces,
      diamondView:          shapes.diamond.diamondView,
    };
  }
  const b  = shape === 'pill' ? shapes.pill : shapes.cube;
  return {
    n_d, V_d, refractionStrength,
    pillLen: b.pillLen, pillShort: b.pillShort, pillThick: b.pillThick, edgeR: b.edgeR,
    waveAmp:        shapes.plate.waveAmp,
    waveWavelength: shapes.plate.waveWavelength,
    diamondSize:          shapes.diamond.diamondSize,
    diamondWireframe:     shapes.diamond.diamondWireframe, diamondFacetColor: shapes.diamond.diamondFacetColor,
    diamondTirDebug:      shapes.diamond.diamondTirDebug, diamondTirMaxBounces: shapes.diamond.diamondTirMaxBounces,
    diamondView:          shapes.diamond.diamondView,
  };
}

/**
 * If old flat localStorage (no `shapes`) is loaded, build per-type defaults and
 * overlay the legacy single pillLen / … onto every body-like shape.
 * Root-level `n_d` / `V_d` / `refractionStrength` are handled in `persistence` +
 * `Params` — not here.
 */
export function shapesFromLegacyFlat(p: Record<string, unknown>, defaults: ShapesParams): ShapesParams {
  const num = (k: string, d: number) => (isFiniteNumber(p[k]) ? (p[k] as number) : d);
  const bool = (k: string, d: boolean) => (typeof p[k] === 'boolean' ? (p[k] as boolean) : d);
  const pl = Math.min(PILL_LEN_MAX, Math.max(PILL_LEN_MIN, num('pillLen', DEF_BODY.pillLen)));
  const ps = Math.min(PILL_SHORT_MAX, Math.max(PILL_SHORT_MIN, num('pillShort', DEF_BODY.pillShort)));
  const pt = Math.min(PILL_THICK_MAX, Math.max(PILL_THICK_MIN, num('pillThick', DEF_BODY.pillThick)));
  const er = Math.min(EDGE_R_MAX, Math.max(EDGE_R_MIN, num('edgeR', DEF_BODY.edgeR)));
  const body: CommonBodyParams = { pillLen: pl, pillShort: ps, pillThick: pt, edgeR: er };
  const wa = Math.min(WAVE_AMP_MAX, Math.max(WAVE_AMP_MIN, num('waveAmp', DEF_PLATE.waveAmp)));
  const ww = Math.min(WAVE_WAVELENGTH_MAX, Math.max(WAVE_WAVELENGTH_MIN, num('waveWavelength', DEF_PLATE.waveWavelength)));
  const dsz  = Math.min(DIAMOND_SIZE_MAX, Math.max(DIAMOND_SIZE_MIN, num('diamondSize', DEF_DIAMOND.diamondSize)));
  const dMax = Math.min(DIAMOND_TIR_BOUNCES_MAX, Math.max(DIAMOND_TIR_BOUNCES_MIN, Math.round(
    isFiniteNumber(p.diamondTirMaxBounces) ? (p.diamondTirMaxBounces as number) : DEF_DIAMOND.diamondTirMaxBounces,
  )));
  return {
    pill:  { ...defaults.pill, ...body },
    prism: { ...defaults.prism, pillLen: pl, pillShort: ps, pillThick: pt },
    cube:  { ...defaults.cube, ...body },
    plate: { ...defaults.plate, ...body, waveAmp: wa, waveWavelength: ww },
    diamond: {
      diamondSize: dsz,
      diamondWireframe:  bool('diamondWireframe', DEF_DIAMOND.diamondWireframe),
      diamondFacetColor: bool('diamondFacetColor', DEF_DIAMOND.diamondFacetColor),
      diamondTirDebug:   bool('diamondTirDebug', DEF_DIAMOND.diamondTirDebug),
      diamondTirMaxBounces: dMax,
      diamondView: (typeof p.diamondView === 'string' && (DIAMOND_VIEW_VALUES as readonly string[]).includes(p.diamondView as string)
        ? p.diamondView
        : DEF_DIAMOND.diamondView) as DiamondView,
    },
  };
}

function clampF(v: number, lo: number, hi: number): number {
  return Math.min(Math.max(v, lo), hi);
}

function mergeBody(o: unknown, d: CommonBodyParams): CommonBodyParams {
  if (o === null || typeof o !== 'object') return { ...d };
  const r = o as Record<string, unknown>;
  return {
    pillLen:   isFiniteNumber(r.pillLen)   ? clampF(r.pillLen as number,   PILL_LEN_MIN,   PILL_LEN_MAX)   : d.pillLen,
    pillShort: isFiniteNumber(r.pillShort) ? clampF(r.pillShort as number, PILL_SHORT_MIN, PILL_SHORT_MAX) : d.pillShort,
    pillThick: isFiniteNumber(r.pillThick) ? clampF(r.pillThick as number, PILL_THICK_MIN, PILL_THICK_MAX) : d.pillThick,
    edgeR:     isFiniteNumber(r.edgeR)     ? clampF(r.edgeR as number,     EDGE_R_MIN,     EDGE_R_MAX)     : d.edgeR,
  };
}

export function mergePrismDims(d: PrismBodyParams, o: unknown): PrismBodyParams {
  if (o === null || typeof o !== 'object' || Array.isArray(o)) return { ...d };
  const r = o as Record<string, unknown>;
  return {
    pillLen:   isFiniteNumber(r.pillLen)   ? clampF(r.pillLen as number,   PILL_LEN_MIN,   PILL_LEN_MAX)   : d.pillLen,
    pillShort: isFiniteNumber(r.pillShort) ? clampF(r.pillShort as number, PILL_SHORT_MIN, PILL_SHORT_MAX) : d.pillShort,
    pillThick: isFiniteNumber(r.pillThick) ? clampF(r.pillThick as number, PILL_THICK_MIN, PILL_THICK_MAX) : d.pillThick,
  };
}

function mergePlate(o: unknown, d: PlateShapeParams): PlateShapeParams {
  if (o === null || typeof o !== 'object') return { ...d };
  const b = mergeBody(o, d);
  const r = o as Record<string, unknown>;
  return {
    ...b,
    waveAmp:         isFiniteNumber(r.waveAmp)         ? clampF(r.waveAmp as number,         WAVE_AMP_MIN,         WAVE_AMP_MAX)         : d.waveAmp,
    waveWavelength:  isFiniteNumber(r.waveWavelength)  ? clampF(r.waveWavelength as number,  WAVE_WAVELENGTH_MIN,  WAVE_WAVELENGTH_MAX)  : d.waveWavelength,
  };
}

function mergeDiamond(o: unknown, d: DiamondShapeParams): DiamondShapeParams {
  if (o === null || typeof o !== 'object') return { ...d };
  const r = o as Record<string, unknown>;
  const dv = r.diamondView;
  return {
    diamondSize:        isFiniteNumber(r.diamondSize) ? clampF(r.diamondSize as number, DIAMOND_SIZE_MIN, DIAMOND_SIZE_MAX) : d.diamondSize,
    diamondWireframe:   typeof r.diamondWireframe  === 'boolean' ? r.diamondWireframe  : d.diamondWireframe,
    diamondFacetColor:  typeof r.diamondFacetColor === 'boolean' ? r.diamondFacetColor : d.diamondFacetColor,
    diamondTirDebug:    typeof r.diamondTirDebug  === 'boolean' ? r.diamondTirDebug  : d.diamondTirDebug,
    diamondTirMaxBounces: isFiniteNumber(r.diamondTirMaxBounces)
      ? Math.round(clampF(r.diamondTirMaxBounces as number, DIAMOND_TIR_BOUNCES_MIN, DIAMOND_TIR_BOUNCES_MAX))
      : d.diamondTirMaxBounces,
    diamondView: (typeof dv === 'string' && (DIAMOND_VIEW_VALUES as readonly string[]).includes(dv) ? dv : d.diamondView) as DiamondView,
  };
}

/**
 * `shapes` from JSON, or null → build from legacy flat `p` (pre–per-type storage).
 */
export function loadShapesFromStorage(shapes: unknown, legacyRoot: Record<string, unknown>): ShapesParams {
  if (shapes === null || typeof shapes !== 'object' || Array.isArray(shapes)) {
    return shapesFromLegacyFlat(legacyRoot, defaultShapesParams());
  }
  const s  = shapes as Record<string, unknown>;
  const d0 = defaultShapesParams();
  return {
    pill:    mergeBody(s.pill,    d0.pill),
    prism:   mergePrismDims(d0.prism, s.prism),
    cube:    mergeBody(s.cube,    d0.cube),
    plate:   mergePlate(s.plate,  d0.plate),
    diamond: mergeDiamond(s.diamond, d0.diamond),
  };
}

function isFiniteNumber(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}
