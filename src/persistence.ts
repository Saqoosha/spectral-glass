import {
  DIAMOND_SIZE_MAX, DIAMOND_SIZE_MIN,
  EDGE_R_MAX, EDGE_R_MIN, FOV_MAX, FOV_MIN,
  HISTORY_ALPHA_MAX, HISTORY_ALPHA_MIN,
  PILL_LEN_MAX, PILL_LEN_MIN, PILL_SHORT_MAX, PILL_SHORT_MIN,
  PILL_THICK_MAX, PILL_THICK_MIN,
  WAVE_AMP_MAX, WAVE_AMP_MIN, WAVE_WAVELENGTH_MAX, WAVE_WAVELENGTH_MIN,
  type AaMode, type Params,
} from './ui';
import type { Pill } from './pills';

const KEY     = 'realrefraction:config';
// `taa: boolean` was renamed to `aaMode: 'none' | 'fxaa' | 'taa'` without
// bumping the version. validateParams migrates legacy `taa: boolean`
// payloads (`true → 'taa'`, `false → 'none'`) when no `aaMode` field is
// present, preserving the user's prior intent instead of snapping them
// to the default. Explicit `aaMode` in the payload always wins. Bumping
// the version would have cost every user their layout for one low-value
// field.
const VERSION = 1;

type Stored = {
  version: number;
  params:  Partial<Params>;
  pills:   Pill[];
};

type Loaded = {
  params: Partial<Params>;
  pills:  Pill[] | null;
};

// Allow-lists for the enum-like fields so a corrupted / hand-edited localStorage
// can't push invalid strings into the render pipeline.
const SHAPES        = new Set<Params['shape']>(['pill', 'prism', 'cube', 'plate', 'diamond']);
const MODES         = new Set<Params['refractionMode']>(['exact', 'approx']);
const PROJECTIONS   = new Set<Params['projection']>(['ortho', 'perspective']);
const SAMPLE_COUNTS = new Set<Params['sampleCount']>([3, 8, 16, 32, 64]);
const AA_MODES      = new Set<AaMode>(['none', 'fxaa', 'taa']);

function isFiniteNumber(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

function validateParams(u: unknown): Partial<Params> {
  if (u === null || typeof u !== 'object') return {};
  const p   = u as Record<string, unknown>;
  const out: Partial<Params> = {};
  if (typeof p.shape          === 'string' && SHAPES.has(p.shape as Params['shape']))                 out.shape          = p.shape as Params['shape'];
  if (typeof p.refractionMode === 'string' && MODES.has(p.refractionMode as Params['refractionMode'])) out.refractionMode = p.refractionMode as Params['refractionMode'];
  if (typeof p.projection     === 'string' && PROJECTIONS.has(p.projection as Params['projection']))   out.projection     = p.projection as Params['projection'];
  if (isFiniteNumber(p.fov))                out.fov                = Math.min(Math.max(p.fov, FOV_MIN), FOV_MAX);
  if (isFiniteNumber(p.sampleCount) && SAMPLE_COUNTS.has(p.sampleCount as Params['sampleCount']))      out.sampleCount    = p.sampleCount as Params['sampleCount'];
  if (isFiniteNumber(p.n_d))                out.n_d                = p.n_d;
  if (isFiniteNumber(p.V_d))                out.V_d                = p.V_d;
  // Pill / cube / plate dimension clamps. Negative or absurd values would
  // either invert the SDF (`abs(p) - h` always positive → shape vanishes)
  // or push proxy bounds to fill the screen. Slider ranges in ui.ts are the
  // source of truth.
  if (isFiniteNumber(p.pillLen))            out.pillLen            = clamp(p.pillLen,   PILL_LEN_MIN,   PILL_LEN_MAX);
  if (isFiniteNumber(p.pillShort))          out.pillShort          = clamp(p.pillShort, PILL_SHORT_MIN, PILL_SHORT_MAX);
  if (isFiniteNumber(p.pillThick))          out.pillThick          = clamp(p.pillThick, PILL_THICK_MIN, PILL_THICK_MAX);
  if (isFiniteNumber(p.edgeR))              out.edgeR              = clamp(p.edgeR, EDGE_R_MIN, EDGE_R_MAX);
  if (isFiniteNumber(p.refractionStrength)) out.refractionStrength = p.refractionStrength;
  if (typeof p.temporalJitter === 'boolean') out.temporalJitter    = p.temporalJitter;
  if (typeof p.debugProxy === 'boolean')     out.debugProxy        = p.debugProxy;
  if (typeof p.aaMode === 'string' && AA_MODES.has(p.aaMode as AaMode)) out.aaMode = p.aaMode as AaMode;
  // Legacy migration: older payloads carry `taa: boolean`. Preserve the
  // user's intent (taa:false -> 'none', taa:true -> 'taa') instead of
  // silently snapping them to the default. Only apply when aaMode is
  // absent, so an explicit aaMode in the payload always wins.
  else if (typeof p.taa === 'boolean')                                 out.aaMode = p.taa ? 'taa' : 'none';
  if (typeof p.paused === 'boolean')         out.paused            = p.paused;
  if (isFiniteNumber(p.historyAlpha))        out.historyAlpha      = clamp(p.historyAlpha, HISTORY_ALPHA_MIN, HISTORY_ALPHA_MAX);
  // Plate wave controls. Clamp to UI slider bounds so hand-edited storage
  // can't push `(amp·freq)²` arbitrarily large — at the slider extremes
  // (amp=60, wavelength=60) waveLipFactor still bottoms out at ≈ 0.16
  // which `MIN_STEP = 0.5` in sphereTrace handles fine; without these
  // clamps a value like `waveAmp = 1e6` would cut waveLipFactor to ~1e-7
  // and stall the trace per fragment (visible black plate). The slider
  // ranges in ui.ts are the source of truth; persistence mirrors them.
  if (isFiniteNumber(p.waveAmp))         out.waveAmp        = clamp(p.waveAmp, WAVE_AMP_MIN, WAVE_AMP_MAX);
  if (isFiniteNumber(p.waveWavelength))  out.waveWavelength = clamp(p.waveWavelength, WAVE_WAVELENGTH_MIN, WAVE_WAVELENGTH_MAX);
  // Diamond size (girdle diameter in px). Same reason for clamping as the
  // pill/plate dimensions above: a hand-edited negative/zero value would
  // invert the polytope SDF into a degenerate shape, and a huge value would
  // blow the proxy AABB past the viewport.
  if (isFiniteNumber(p.diamondSize))     out.diamondSize    = clamp(p.diamondSize, DIAMOND_SIZE_MIN, DIAMOND_SIZE_MAX);
  if (typeof p.diamondWireframe === 'boolean') out.diamondWireframe = p.diamondWireframe;
  return out;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(Math.max(v, lo), hi);
}

function validatePills(u: unknown): Pill[] | null {
  if (!Array.isArray(u)) return null;
  const pills: Pill[] = [];
  for (const raw of u) {
    if (raw === null || typeof raw !== 'object') continue;
    const p = raw as Record<string, unknown>;
    if (isFiniteNumber(p.cx) && isFiniteNumber(p.cy) && isFiniteNumber(p.cz) &&
        isFiniteNumber(p.hx) && isFiniteNumber(p.hy) && isFiniteNumber(p.hz) &&
        isFiniteNumber(p.edgeR)) {
      pills.push({ cx: p.cx, cy: p.cy, cz: p.cz, hx: p.hx, hy: p.hy, hz: p.hz, edgeR: p.edgeR });
    }
  }
  return pills.length > 0 ? pills : null;
}

/**
 * Read persisted config from localStorage. Returns null for: empty storage,
 * unavailable storage (quota / disabled / private mode / SSR), corrupt JSON,
 * or a schema-version mismatch. The returned `params` passes validateParams so
 * callers can merge it into defaults without worrying about NaN / bogus enums.
 */
export function loadStored(): Loaded | null {
  let raw: string | null;
  try {
    raw = localStorage.getItem(KEY);
  } catch (err) {
    console.warn('[persistence] localStorage unavailable:', err);
    return null;
  }
  if (!raw) return null;

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    console.warn('[persistence] stored config corrupt, resetting:', err);
    try { localStorage.removeItem(KEY); } catch { /* best effort */ }
    return null;
  }

  if (parsed === null || typeof parsed !== 'object') return null;
  const rec = parsed as Record<string, unknown>;
  if (rec.version !== VERSION) {
    console.info(`[persistence] schema mismatch (have ${String(rec.version)}, want ${VERSION}); using defaults`);
    return null;
  }

  return {
    params: validateParams(rec.params),
    pills:  validatePills(rec.pills),
  };
}

export function save(params: Params, pills: readonly Pill[]): void {
  try {
    const payload: Stored = { version: VERSION, params, pills: [...pills] };
    localStorage.setItem(KEY, JSON.stringify(payload));
  } catch (err) {
    // Common causes: QuotaExceededError, SecurityError (private mode Safari),
    // disabled storage. Not fatal — surface the error so it's debuggable.
    console.warn('[persistence] save failed:', err);
  }
}

export type Saver = {
  /** Schedule a trailing-edge write; coalesces bursts within `delayMs`. */
  schedule(params: Params, pills: readonly Pill[]): void;
  /** Write any pending payload immediately. Call on pagehide / unload. */
  flush(): void;
};

/**
 * Trailing-edge debounced saver. Each call resets the timer so the write
 * happens `delayMs` after activity stops, not `delayMs` after the first event.
 * Combined with a `flush()` on pagehide to catch close-during-drag.
 */
export function debouncedSaver(delayMs = 250): Saver {
  let handle: ReturnType<typeof setTimeout> | null = null;
  let latest: { params: Params; pills: readonly Pill[] } | null = null;

  const write = () => {
    handle = null;
    if (latest) save(latest.params, latest.pills);
  };

  return {
    schedule(params, pills) {
      latest = { params, pills };
      if (handle !== null) clearTimeout(handle);
      handle = setTimeout(write, delayMs);
    },
    flush() {
      if (handle !== null) {
        clearTimeout(handle);
        write();
      }
    },
  };
}
