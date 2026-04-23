import { describe, it, expect, beforeEach, vi } from 'vitest';
import { loadStored, save } from '../src/persistence';
import type { Params } from '../src/ui';

// localStorage doesn't exist in Node's vitest env by default. We install a
// minimal in-memory mock on globalThis before each test so
// persistence.ts' `localStorage.getItem`/`.setItem`/`.removeItem` calls
// work. Each test gets a fresh mock so one test's payload can't leak into
// the next.
class MemoryStorage {
  private store = new Map<string, string>();
  getItem(key: string): string | null { return this.store.get(key) ?? null; }
  setItem(key: string, value: string): void { this.store.set(key, value); }
  removeItem(key: string): void { this.store.delete(key); }
  clear(): void { this.store.clear(); }
  get length(): number { return this.store.size; }
  key(i: number): string | null { return Array.from(this.store.keys())[i] ?? null; }
}

function installStorage(): MemoryStorage {
  const mem = new MemoryStorage();
  (globalThis as unknown as { localStorage: Storage }).localStorage = mem as unknown as Storage;
  return mem;
}

const KEY = 'realrefraction:config';
const VERSION = 1;

function writeRaw(mem: MemoryStorage, params: Record<string, unknown>): void {
  mem.setItem(KEY, JSON.stringify({ version: VERSION, params, pills: [] }));
}

// Minimal defaults — loadStored only validates fields that are present in
// the payload, so tests that poke at a single field don't need a complete
// Params object.
function defaultParamsForSave(): Params {
  return {
    sampleCount: 8,
    shape: 'diamond',
    n_d: 2.418,
    V_d: 55,
    pillLen: 200,
    pillShort: 200,
    pillThick: 200,
    edgeR: 10,
    refractionStrength: 0.2,
    refractionMode: 'exact',
    temporalJitter: true,
    projection: 'perspective',
    fov: 60,
    debugProxy: false,
    aaMode: 'taa',
    paused: false,
    historyAlpha: 0.2,
    waveAmp: 20,
    waveWavelength: 300,
    diamondSize: 200,
    diamondWireframe: false,
    diamondFacetColor: false,
    diamondTirDebug: false,
    diamondView: 'free',
    envmapEnabled: true,
    envmapExposure: 0.25,
    envmapRotation: 0,
    envmapSlug: 'studio_small_03',
    envmapSize: '2k',
  };
}

describe('persistence — diamondView allow-list validation', () => {
  beforeEach(() => {
    installStorage();
  });

  for (const view of ['free', 'top', 'side', 'bottom'] as const) {
    it(`accepts the canonical view "${view}"`, () => {
      const mem = installStorage();
      writeRaw(mem, { diamondView: view });
      const loaded = loadStored();
      expect(loaded?.params.diamondView).toBe(view);
    });
  }

  it('rejects a non-canonical view string — silently drops to default path', () => {
    const mem = installStorage();
    writeRaw(mem, { diamondView: 'isometric' });
    const loaded = loadStored();
    // Field should be absent from returned params so the caller's merge
    // with defaultParams() re-seeds it to 'free'. Dropping-to-default is
    // the documented failure mode for unknown enum strings — same as how
    // `shape` / `aaMode` / `projection` behave.
    expect(loaded?.params.diamondView).toBeUndefined();
  });

  it('rejects non-string types for diamondView', () => {
    const mem = installStorage();
    // Test a representative set of type-confused inputs. A boolean would
    // slip through a naive truthy check; null would slip through a
    // `typeof === 'object'` check; a number would slip through an
    // in-keyword check.
    for (const bogus of [null, 123, true, { view: 'top' }, ['top']]) {
      mem.clear();
      writeRaw(mem, { diamondView: bogus as unknown });
      const loaded = loadStored();
      expect(loaded?.params.diamondView).toBeUndefined();
    }
  });

  it('round-trips diamondView through save() + loadStored()', () => {
    // End-to-end: save() writes JSON, loadStored() parses + validates. A
    // regression in EITHER the allow-list set OR the JSON shape is caught.
    installStorage();
    const params = defaultParamsForSave();
    for (const view of ['free', 'top', 'side', 'bottom'] as const) {
      save({ ...params, diamondView: view }, []);
      const loaded = loadStored();
      expect(loaded?.params.diamondView).toBe(view);
    }
  });
});

describe('persistence — diamondFacetColor boolean guard', () => {
  beforeEach(() => {
    installStorage();
  });

  it('accepts true and false', () => {
    for (const val of [true, false]) {
      const mem = installStorage();
      writeRaw(mem, { diamondFacetColor: val });
      const loaded = loadStored();
      expect(loaded?.params.diamondFacetColor).toBe(val);
    }
  });

  it('rejects truthy non-boolean values (strict boolean check)', () => {
    // A regression where someone swapped `typeof === 'boolean'` for a
    // truthy check would let these through and corrupt the uniform write
    // (scratch[base+2] = p.diamondFacetColor ? 1 : 0 already exists but
    // downstream readers treat the value as strictly boolean). The
    // current implementation uses `typeof === 'boolean'` — this test pins
    // it against the "truthy" refactor.
    for (const bogus of ['true', 1, 0, 'yes', null, {}, []]) {
      const mem = installStorage();
      writeRaw(mem, { diamondFacetColor: bogus as unknown });
      const loaded = loadStored();
      expect(loaded?.params.diamondFacetColor).toBeUndefined();
    }
  });

  it('round-trips diamondFacetColor through save() + loadStored()', () => {
    installStorage();
    const params = defaultParamsForSave();
    for (const val of [true, false]) {
      save({ ...params, diamondFacetColor: val }, []);
      const loaded = loadStored();
      expect(loaded?.params.diamondFacetColor).toBe(val);
    }
  });
});

describe('persistence — envmap field validation (Phase C)', () => {
  beforeEach(() => {
    installStorage();
  });

  it('rejects unknown envmap slugs (allow-list gate vs stale localStorage)', () => {
    // Matches the same "drop to default" pattern diamondView uses —
    // a hand-edited or migration-stale slug should be dropped so the
    // default kicks in, not trigger a 404 at boot.
    const mem = installStorage();
    writeRaw(mem, { envmapSlug: 'nonexistent_hdri' });
    const loaded = loadStored();
    expect(loaded?.params.envmapSlug).toBeUndefined();
  });

  it('accepts known envmap slugs', () => {
    const mem = installStorage();
    writeRaw(mem, { envmapSlug: 'studio_small_03' });
    const loaded = loadStored();
    expect(loaded?.params.envmapSlug).toBe('studio_small_03');
  });

  it('rejects unknown envmap sizes', () => {
    const mem = installStorage();
    writeRaw(mem, { envmapSize: '8k' });
    const loaded = loadStored();
    expect(loaded?.params.envmapSize).toBeUndefined();
  });

  it('clamps envmapExposure to [MIN, MAX]', () => {
    // UI slider range is 0.01 .. 2.0 — hand-edited values outside
    // that range should be pulled back in.
    const mem = installStorage();
    writeRaw(mem, { envmapExposure: 99 });
    expect(loadStored()?.params.envmapExposure).toBe(2.0);
    mem.clear();
    writeRaw(mem, { envmapExposure: -5 });
    expect(loadStored()?.params.envmapExposure).toBe(0.01);
    mem.clear();
    writeRaw(mem, { envmapExposure: 0.5 });
    expect(loadStored()?.params.envmapExposure).toBe(0.5);
  });

  it('clamps envmapRotation to [-π, π]', () => {
    const mem = installStorage();
    writeRaw(mem, { envmapRotation: 100 });
    expect(loadStored()?.params.envmapRotation).toBeCloseTo(Math.PI, 6);
    mem.clear();
    writeRaw(mem, { envmapRotation: -100 });
    expect(loadStored()?.params.envmapRotation).toBeCloseTo(-Math.PI, 6);
  });

  it('rejects non-boolean envmapEnabled', () => {
    const mem = installStorage();
    for (const bogus of ['true', 1, null, {}, []]) {
      mem.clear();
      writeRaw(mem, { envmapEnabled: bogus as unknown });
      expect(loadStored()?.params.envmapEnabled).toBeUndefined();
    }
  });

  it('rejects non-string envmapSlug / envmapSize payloads', () => {
    // Diamond view / facet color get their "non-string reject" test
    // via the allow-list code path. Mirror it for envmap so the
    // `typeof === 'string'` guard can't silently relax into a truthy
    // check without this firing. Covers null (typeof 'object' trap),
    // numbers, booleans, arrays, objects.
    const mem = installStorage();
    for (const bogus of [null, 123, true, ['2k'], { size: '2k' }]) {
      mem.clear();
      writeRaw(mem, { envmapSize: bogus as unknown });
      expect(loadStored()?.params.envmapSize).toBeUndefined();
      mem.clear();
      writeRaw(mem, { envmapSlug: bogus as unknown });
      expect(loadStored()?.params.envmapSlug).toBeUndefined();
    }
  });

  it('round-trips every envmap field through save() + loadStored()', () => {
    installStorage();
    const params = defaultParamsForSave();
    const patched = {
      ...params,
      envmapEnabled: false,
      envmapExposure: 0.75,
      envmapRotation: 1.23,
      envmapSlug: 'neon_photostudio',
      envmapSize: '4k' as const,
    };
    save(patched, []);
    const loaded = loadStored();
    expect(loaded?.params.envmapEnabled).toBe(false);
    expect(loaded?.params.envmapExposure).toBeCloseTo(0.75, 6);
    expect(loaded?.params.envmapRotation).toBeCloseTo(1.23, 6);
    expect(loaded?.params.envmapSlug).toBe('neon_photostudio');
    expect(loaded?.params.envmapSize).toBe('4k');
  });
});

describe('persistence — unavailable / corrupt storage', () => {
  it('returns null when localStorage.getItem throws', () => {
    const fail: Storage = {
      getItem: vi.fn(() => { throw new Error('SecurityError'); }),
      setItem: vi.fn(),
      removeItem: vi.fn(),
      clear: vi.fn(),
      key: vi.fn(() => null),
      length: 0,
    };
    (globalThis as unknown as { localStorage: Storage }).localStorage = fail;
    expect(loadStored()).toBeNull();
  });

  it('returns null AND clears storage when JSON is corrupt', () => {
    const mem = installStorage();
    mem.setItem(KEY, '{{ not valid json');
    expect(loadStored()).toBeNull();
    // Corrupt payload should have been removed so the next load doesn't
    // hit the same trap.
    expect(mem.getItem(KEY)).toBeNull();
  });

  it('returns null when schema version mismatches', () => {
    const mem = installStorage();
    mem.setItem(KEY, JSON.stringify({ version: 999, params: {}, pills: [] }));
    expect(loadStored()).toBeNull();
  });
});
