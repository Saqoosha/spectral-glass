import { describe, it, expect } from 'vitest';
import {
  diamondRotationColumns,
  diamondWgslConstants,
  DIAMOND_HEIGHT_RATIO,
  DIAMOND_INTERNALS,
  DIAMOND_PROXY_TRI_COUNT,
  DIAMOND_PROXY_VERT_COUNT,
  DIAMOND_SIZE_MIN,
  DIAMOND_SIZE_MAX,
} from '../src/math/diamond';

// Reconstruct a row-major 3x3 from the WGSL mat3x3<f32> layout that
// diamondRotationColumns emits. Buffer is 12 floats: column k at
// indices [k*4, k*4+3); k*4+3 is per-column pad.
function rowMajor(cols: Float32Array): number[][] {
  const at = (i: number): number => cols[i] ?? NaN;
  return [
    [at(0), at(4), at(8)],
    [at(1), at(5), at(9)],
    [at(2), at(6), at(10)],
  ];
}

function row(m: number[][], r: number): number[] {
  return m[r] ?? [];
}

function mulMatVec(m: number[][], v: [number, number, number]): [number, number, number] {
  const r0 = row(m, 0);
  const r1 = row(m, 1);
  const r2 = row(m, 2);
  const ax = (arr: number[], i: number): number => arr[i] ?? 0;
  return [
    ax(r0, 0) * v[0] + ax(r0, 1) * v[1] + ax(r0, 2) * v[2],
    ax(r1, 0) * v[0] + ax(r1, 1) * v[1] + ax(r1, 2) * v[2],
    ax(r2, 0) * v[0] + ax(r2, 1) * v[1] + ax(r2, 2) * v[2],
  ];
}

describe('diamondRotationColumns', () => {
  it('returns a fresh 12-element Float32Array in the padded WGSL layout', () => {
    const a = diamondRotationColumns(1);
    const b = diamondRotationColumns(1);
    expect(a).not.toBe(b);
    expect(a.length).toBe(12);
    expect(b.length).toBe(12);
  });

  it('leaves the per-column stride pad at zero', () => {
    const m = diamondRotationColumns(2.3);
    expect(m[3]).toBe(0);
    expect(m[7]).toBe(0);
    expect(m[11]).toBe(0);
  });

  it('is a pure tilt (no spin) at t=0 — Y basis maps to (0, cos 20°, -sin 20°)', () => {
    // At t=0 only the fixed -20° X-axis tilt applies; the Y-basis image is
    // the 2nd column of Rx(-20°) = (0, cos(-20°), sin(-20°)) = (0, cos 20°,
    // -sin 20°). Hard-coded numeric values (rather than `Math.cos(-20°)`
    // wrappers around the same DEG constant the implementation uses) so the
    // tilt sign is verified independently — a typo flipping DIAMOND_TILT to
    // +20° would otherwise still pass because `cos(±20°)` is symmetric.
    const m = rowMajor(diamondRotationColumns(0));
    const yImage = mulMatVec(m, [0, 1, 0]);
    expect(yImage[0]).toBeCloseTo(0, 6);
    expect(yImage[1]).toBeCloseTo(0.9396926208, 6);   // +cos(20°)
    expect(yImage[2]).toBeCloseTo(-0.3420201433, 6);  // -sin(20°), CATCHES tilt sign flips
  });

  it('matches the documented Rx(-20°)·Ry(t·0.30) composition at t=10', () => {
    // Closed-form cross-check: re-derive Rx · Ry independently and compare
    // every cell. Catches three regression families in one test:
    //   - tilt sign flip (DIAMOND_TILT from -20° to +20°)
    //   - spin rate change (0.30 → 0.20 or 3.0)
    //   - composition order swap (Ry · Rx instead of Rx · Ry)
    // Any of these would leave the "Y basis at t=0" test above passing but
    // produce visibly wrong motion; this test pins all 9 matrix cells.
    const t  = 10;
    const ay = t * 0.30;
    const ax = -20 * Math.PI / 180;
    const cy = Math.cos(ay), sy = Math.sin(ay);
    const cx = Math.cos(ax), sx = Math.sin(ax);
    // R = Rx(ax) · Ry(ay), row-major:
    //   [  cy,      0,    sy    ]
    //   [  sx·sy,   cx,  -sx·cy ]
    //   [ -cx·sy,   sx,   cx·cy ]
    const expected = [
      [ cy,        0,    sy       ],
      [ sx * sy,   cx,  -sx * cy  ],
      [-cx * sy,   sx,   cx * cy  ],
    ];
    const got = rowMajor(diamondRotationColumns(t));
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        expect(row(got, r)[c] ?? NaN).toBeCloseTo(row(expected, r)[c] ?? NaN, 5);
      }
    }
  });

  it('preserves vector length (orthonormal rotation)', () => {
    const m = rowMajor(diamondRotationColumns(1.234));
    const v: [number, number, number] = [3, 4, 5];
    const r = mulMatVec(m, v);
    const inLen = Math.hypot(...v);
    const outLen = Math.hypot(...r);
    expect(outLen).toBeCloseTo(inLen, 5);
  });

  it('rejects non-finite time instead of producing NaN rotations', () => {
    expect(() => diamondRotationColumns(NaN)).toThrow(/finite/);
    expect(() => diamondRotationColumns(Infinity)).toThrow(/finite/);
    expect(() => diamondRotationColumns(-Infinity)).toThrow(/finite/);
  });
});

describe('diamond geometry (Tolkowsky ideal)', () => {
  it('girdle radius is 0.5 (half the unit diameter)', () => {
    expect(DIAMOND_INTERNALS.R_GIRDLE).toBeCloseTo(0.5, 10);
  });

  it('crown height matches tan(crown_angle) × (girdle − table radial gap)', () => {
    // Regression pin on the derived H_CROWN. Drift here probably means
    // TABLE_RATIO or CROWN_ANGLE_DEG moved; test double-checks both.
    const expected = (DIAMOND_INTERNALS.R_GIRDLE - DIAMOND_INTERNALS.R_TABLE_APOTHEM)
      * Math.tan(DIAMOND_INTERNALS.CROWN_ANGLE_DEG * Math.PI / 180);
    expect(DIAMOND_INTERNALS.H_CROWN).toBeCloseTo(expected, 8);
    // And land near the gemology book value of ~0.162.
    expect(DIAMOND_INTERNALS.H_CROWN).toBeGreaterThan(0.15);
    expect(DIAMOND_INTERNALS.H_CROWN).toBeLessThan(0.18);
  });

  it('pavilion depth matches tan(pavilion_angle) × girdle radius', () => {
    const expected = DIAMOND_INTERNALS.R_GIRDLE
      * Math.tan(DIAMOND_INTERNALS.PAVILION_ANGLE_DEG * Math.PI / 180);
    expect(DIAMOND_INTERNALS.H_PAVILION).toBeCloseTo(expected, 8);
    expect(DIAMOND_INTERNALS.H_PAVILION).toBeGreaterThan(0.40);
    expect(DIAMOND_INTERNALS.H_PAVILION).toBeLessThan(0.45);
  });

  it('total height ratio sums crown + girdle + pavilion', () => {
    const expected = DIAMOND_INTERNALS.H_TOP - DIAMOND_INTERNALS.H_BOT;
    expect(DIAMOND_HEIGHT_RATIO).toBeCloseTo(expected, 8);
    // Ideal brilliant is ~60% of diameter — pin within a realistic range.
    expect(DIAMOND_HEIGHT_RATIO).toBeGreaterThan(0.55);
    expect(DIAMOND_HEIGHT_RATIO).toBeLessThan(0.65);
  });

  it('all facet plane normals are unit length', () => {
    const planes = Object.values(DIAMOND_INTERNALS.planes);
    for (const p of planes) {
      const len = Math.hypot(p.nx, p.ny, p.nz);
      expect(len).toBeCloseTo(1, 6);
    }
  });

  it('bezel + pavilion normals have no Y component (centred at φ=0)', () => {
    expect(DIAMOND_INTERNALS.planes.bezel.ny).toBeCloseTo(0, 10);
    expect(DIAMOND_INTERNALS.planes.pavilion.ny).toBeCloseTo(0, 10);
  });

  it('crown normals have +Z, pavilion normals have -Z', () => {
    expect(DIAMOND_INTERNALS.planes.bezel.nz).toBeGreaterThan(0);
    expect(DIAMOND_INTERNALS.planes.star.nz).toBeGreaterThan(0);
    expect(DIAMOND_INTERNALS.planes.upperHalf.nz).toBeGreaterThan(0);
    expect(DIAMOND_INTERNALS.planes.lowerHalf.nz).toBeLessThan(0);
    expect(DIAMOND_INTERNALS.planes.pavilion.nz).toBeLessThan(0);
  });

  it('bezel plane passes through the girdle-top rim at φ=0', () => {
    // Anchor is (R_GIRDLE, 0, +H_GIRDLE_HALF); plane offset should be the
    // dot product of that anchor with the normal (sin 34.5°, 0, cos 34.5°).
    // Catches accidental anchor typos (e.g. switching sign of H_GIRDLE_HALF,
    // which would shift the bezel vertically and break the crown silhouette
    // with no visible test failure today — unit-length + sign tests pass
    // regardless).
    const { R_GIRDLE, planes } = DIAMOND_INTERNALS;
    const alpha  = 34.5 * Math.PI / 180;
    const expected = R_GIRDLE * Math.sin(alpha) + 0.01 * Math.cos(alpha);
    expect(planes.bezel.offset).toBeCloseTo(expected, 8);
  });

  it('pavilion plane passes through the girdle-bottom rim at φ=0', () => {
    // Symmetric check for the pavilion: anchor is (R_GIRDLE, 0, -H_GIRDLE_HALF)
    // and the outward normal has -Z component (pavilion tilts downward).
    // `planeFromAngles` computes offset = dot(anchor, normal) — for the
    // pavilion that's R_GIRDLE · sin(40.75°) + (-0.01) · (-cos(40.75°)).
    const { R_GIRDLE, planes } = DIAMOND_INTERNALS;
    const alpha  = 40.75 * Math.PI / 180;
    const expected = R_GIRDLE * Math.sin(alpha) + 0.01 * Math.cos(alpha);
    expect(planes.pavilion.offset).toBeCloseTo(expected, 8);
  });

  it('star plane passes through a table vertex on the mirror boundary (φ=π/8)', () => {
    // Star anchor = (R_TABLE_VERTEX·cos(π/8), R_TABLE_VERTEX·sin(π/8), H_TOP).
    // Normal = (cos(π/8)·sin(22°), sin(π/8)·sin(22°), cos(22°)).
    // Offset = dot(anchor, normal), which simplifies to
    //   R_TABLE_VERTEX · sin(22°) + H_TOP · cos(22°)
    // (the cos²+sin² collapses the in-plane dot to R_TABLE_VERTEX · sin 22°).
    //
    // This pins the star-plane anchor on R_TABLE_VERTEX specifically — catches
    // a silent revert from flat-to-flat TABLE_RATIO convention back to vertex-
    // to-vertex, which would otherwise pass every other plane test because
    // only the star plane uses R_TABLE_VERTEX in its anchor.
    const { R_TABLE_VERTEX, H_TOP, planes } = DIAMOND_INTERNALS;
    const alpha    = 22.0 * Math.PI / 180;
    const expected = R_TABLE_VERTEX * Math.sin(alpha) + H_TOP * Math.cos(alpha);
    expect(planes.star.offset).toBeCloseTo(expected, 8);
  });

  it('slider bounds span a reasonable visible range', () => {
    expect(DIAMOND_SIZE_MIN).toBeGreaterThan(0);
    expect(DIAMOND_SIZE_MAX).toBeGreaterThan(DIAMOND_SIZE_MIN);
  });
});

describe('diamondWgslConstants', () => {
  it('emits WGSL const declarations for every plane and the table/girdle scalars', () => {
    const wgsl = diamondWgslConstants();
    // Pins the names the shader reads — a rename here breaks the shader build.
    // Scalars (geometry + proxy-mesh helpers):
    expect(wgsl).toContain('const DIAMOND_H_TOP:');
    expect(wgsl).toContain('const DIAMOND_H_BOT:');
    expect(wgsl).toContain('const DIAMOND_H_GIRDLE_HALF:');
    expect(wgsl).toContain('const DIAMOND_R_GIRDLE:');
    expect(wgsl).toContain('const DIAMOND_R_TABLE_VERTEX:');
    expect(wgsl).toContain('const DIAMOND_GIRDLE_R_CIRC:');
    expect(wgsl).toContain('const DIAMOND_PROXY_VERT_COUNT:');
    // Facet planes (5 classes × normal + offset = 10 consts):
    expect(wgsl).toContain('const DIAMOND_BEZEL_N:');
    expect(wgsl).toContain('const DIAMOND_BEZEL_O:');
    expect(wgsl).toContain('const DIAMOND_STAR_N:');
    expect(wgsl).toContain('const DIAMOND_STAR_O:');
    expect(wgsl).toContain('const DIAMOND_UPPER_HALF_N:');
    expect(wgsl).toContain('const DIAMOND_UPPER_HALF_O:');
    expect(wgsl).toContain('const DIAMOND_LOWER_HALF_N:');
    expect(wgsl).toContain('const DIAMOND_LOWER_HALF_O:');
    expect(wgsl).toContain('const DIAMOND_PAVILION_N:');
    expect(wgsl).toContain('const DIAMOND_PAVILION_O:');
  });

  it('does not emit obsolete constants from earlier iterations', () => {
    // Guard against accidental re-introduction of the AABB / bipyramid
    // constants that were replaced with the exact-mesh geometry. Keeping
    // them around would silently diverge from whatever the shader ends up
    // using and eventually confuse debuggers.
    const wgsl = diamondWgslConstants();
    expect(wgsl).not.toContain('DIAMOND_AABB_HALF_Z');
    expect(wgsl).not.toContain('DIAMOND_AABB_OFFSET_Z');
    expect(wgsl).not.toContain('DIAMOND_BASE_R_RATIO');
  });

  it('radii are ordered table_apothem < table_vertex < girdle < girdle_r_circ', () => {
    // The octagon geometry demands r_apothem = r_vertex · cos(π/8), so the
    // strict ordering is a direct cross-check on the flat-to-flat table
    // convention (TABLE_RATIO applies to apothem, not vertex radius) and on
    // the circumscribing-octagon formula for the proxy's girdle ring.
    const i = DIAMOND_INTERNALS;
    expect(i.R_TABLE_APOTHEM).toBeLessThan(i.R_TABLE_VERTEX);
    expect(i.R_TABLE_VERTEX).toBeLessThan(i.R_GIRDLE);
    expect(i.R_GIRDLE).toBeLessThan(i.GIRDLE_R_CIRC);
    // Pin the circumscribing-octagon formula — the proxy mesh's over/under
    // coverage analysis depends on it exactly.
    expect(i.GIRDLE_R_CIRC / i.R_GIRDLE).toBeCloseTo(1 / Math.cos(Math.PI / 8), 8);
    // Flat-to-flat convention: R_TABLE_APOTHEM = R_GIRDLE · TABLE_RATIO
    // (not R_TABLE_VERTEX = R_GIRDLE · TABLE_RATIO). Catches a future
    // accidental revert to vertex-to-vertex, which would silently make the
    // rendered table narrower than a real brilliant cut.
    expect(i.R_TABLE_APOTHEM / i.R_GIRDLE).toBeCloseTo(0.53, 8);
  });

  it('formats vectors as vec3<f32>(x, y, z)', () => {
    const wgsl = diamondWgslConstants();
    // Structural check — any syntactically-invalid WGSL here would fail
    // pipeline compilation at startup, but surfacing it in a unit test
    // catches regressions without needing a GPU.
    expect(wgsl).toMatch(/vec3<f32>\(-?\d+\.\d+(?:e-?\d+)?, -?\d+\.\d+(?:e-?\d+)?, -?\d+\.\d+(?:e-?\d+)?\)/);
  });

  it('emits DIAMOND_PROXY_VERT_COUNT matching the exported TS constant', () => {
    // Keeps the shader's maxVerts guard literal in sync with the host draw
    // call (both read the same constant, see src/webgpu/pipeline.ts). A
    // Phase B mesh change only has to update DIAMOND_PROXY_TRI_COUNT.
    const wgsl = diamondWgslConstants();
    expect(wgsl).toContain(`${DIAMOND_PROXY_VERT_COUNT}u;`);
    expect(DIAMOND_PROXY_VERT_COUNT).toBe(DIAMOND_PROXY_TRI_COUNT * 3);
  });
});
