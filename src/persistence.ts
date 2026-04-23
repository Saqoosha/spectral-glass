import {
  FOV_MAX, FOV_MIN,
  ENVMAP_EXPOSURE_MAX, ENVMAP_EXPOSURE_MIN,
  ENVMAP_ROTATION_MAX, ENVMAP_ROTATION_MIN,
  HISTORY_ALPHA_MAX, HISTORY_ALPHA_MIN,
  N_D_MAX, N_D_MIN,
  V_D_MAX, V_D_MIN,
  REFRACTION_STRENGTH_MAX, REFRACTION_STRENGTH_MIN,
  type AaMode, type DiamondView, type Params,
} from './ui';
import { loadShapesFromStorage } from './shapeParams';
import { DIAMOND_VIEW_VALUES } from './math/diamond';
import { isKnownSlug, ENVMAP_SIZES, type EnvmapSize } from './envmapList';
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
// DIAMOND_VIEWS derives from the canonical list in math/diamond.ts, so
// adding a new preset is a one-site change there — the runtime allow-list
// and the compile-time union can't drift. The older pattern (hand-written
// Set literal mirroring a hand-written union) could silently drop
// newly-added presets on load if someone forgot to update both.
const DIAMOND_VIEWS = new Set<DiamondView>(DIAMOND_VIEW_VALUES);

function isFiniteNumber(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

const SHAPE_MATERIAL_KEYS = ['pill', 'prism', 'cube', 'plate', 'diamond'] as const;

/**
 * Pre–shared-material `shapes.*` stored n_d / V_d / refraction on each
 * sub-object. If the root payload has no new keys, read the first
 * per-shape value we find so old saves keep their numbers after upgrade.
 */
function materialFallbackFromNestedShapes(
  shapes: unknown,
): Partial<Pick<Params, 'n_d' | 'V_d' | 'refractionStrength'>> {
  if (shapes === null || typeof shapes !== 'object' || Array.isArray(shapes)) return {};
  const s = shapes as Record<string, unknown>;
  const out: Partial<Pick<Params, 'n_d' | 'V_d' | 'refractionStrength'>> = {};
  for (const k of SHAPE_MATERIAL_KEYS) {
    const o = s[k];
    if (o === null || typeof o !== 'object') continue;
    const r = o as Record<string, unknown>;
    if (out.n_d === undefined && isFiniteNumber(r.n_d)) out.n_d = r.n_d;
    if (out.V_d === undefined && isFiniteNumber(r.V_d)) out.V_d = r.V_d;
    if (out.refractionStrength === undefined && isFiniteNumber(r.refractionStrength)) {
      out.refractionStrength = r.refractionStrength;
    }
  }
  return out;
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
  out.shapes = loadShapesFromStorage(p.shapes, p);
  const matFb = materialFallbackFromNestedShapes(p.shapes);
  if (isFiniteNumber(p.n_d)) {
    out.n_d = clamp(p.n_d, N_D_MIN, N_D_MAX);
  } else if (isFiniteNumber(matFb.n_d)) {
    out.n_d = clamp(matFb.n_d, N_D_MIN, N_D_MAX);
  }
  if (isFiniteNumber(p.V_d)) {
    out.V_d = clamp(p.V_d, V_D_MIN, V_D_MAX);
  } else if (isFiniteNumber(matFb.V_d)) {
    out.V_d = clamp(matFb.V_d, V_D_MIN, V_D_MAX);
  }
  if (isFiniteNumber(p.refractionStrength)) {
    out.refractionStrength = clamp(p.refractionStrength, REFRACTION_STRENGTH_MIN, REFRACTION_STRENGTH_MAX);
  } else if (isFiniteNumber(matFb.refractionStrength)) {
    out.refractionStrength = clamp(matFb.refractionStrength, REFRACTION_STRENGTH_MIN, REFRACTION_STRENGTH_MAX);
  }
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
  // Envmap (Phase C). Slug is validated against the known Poly Haven
  // allow-list so a hand-edited / stale entry doesn't trigger a 404
  // on next startup.
  if (typeof p.envmapEnabled === 'boolean')     out.envmapEnabled  = p.envmapEnabled;
  if (isFiniteNumber(p.envmapExposure))         out.envmapExposure = clamp(p.envmapExposure, ENVMAP_EXPOSURE_MIN, ENVMAP_EXPOSURE_MAX);
  if (isFiniteNumber(p.envmapRotation))         out.envmapRotation = clamp(p.envmapRotation, ENVMAP_ROTATION_MIN, ENVMAP_ROTATION_MAX);
  if (typeof p.envmapSlug === 'string' && isKnownSlug(p.envmapSlug)) out.envmapSlug = p.envmapSlug;
  if (typeof p.envmapSize === 'string'
    && (ENVMAP_SIZES as readonly string[]).includes(p.envmapSize)) {
    out.envmapSize = p.envmapSize as EnvmapSize;
  }
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
