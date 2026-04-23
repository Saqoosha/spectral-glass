import { describe, it, expect } from 'vitest';
import {
  diamondRotationColumns,
  diamondViewRotationColumns,
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

describe('diamondViewRotationColumns (fixed view presets)', () => {
  it('top preset leaves local axes unrotated (identity — table directly toward camera)', () => {
    const m = rowMajor(diamondViewRotationColumns('top'));
    const z = mulMatVec(m, [0, 0, 1]);
    expect(z[0]).toBeCloseTo(0, 6);
    expect(z[1]).toBeCloseTo(0, 6);
    expect(z[2]).toBeCloseTo(1, 6);  // local +Z stays on world +Z
  });

  it('side preset maps local +Z to world +Y (girdle axis vertical, profile visible)', () => {
    const m = rowMajor(diamondViewRotationColumns('side'));
    const z = mulMatVec(m, [0, 0, 1]);
    expect(z[0]).toBeCloseTo(0, 6);
    expect(z[1]).toBeCloseTo(1, 6);
    expect(z[2]).toBeCloseTo(0, 6);
  });

  it('bottom preset maps local +Z to world -Z (culet toward camera)', () => {
    const m = rowMajor(diamondViewRotationColumns('bottom'));
    const z = mulMatVec(m, [0, 0, 1]);
    expect(z[0]).toBeCloseTo(0, 6);
    expect(z[1]).toBeCloseTo(0, 6);
    expect(z[2]).toBeCloseTo(-1, 6);
  });

  it('preserves vector length for every preset (orthonormal rotations)', () => {
    // Pins the matrices as orthonormal — a transcription bug that breaks
    // unit-length preservation would silently scale the rendered diamond
    // by the (1 ± ε) factor the bad matrix introduces. Catching it here
    // also guards against future view additions dropping in a non-rotation.
    for (const view of ['top', 'side', 'bottom'] as const) {
      const m = rowMajor(diamondViewRotationColumns(view));
      const v: [number, number, number] = [3, 4, 12];
      const r = mulMatVec(m, v);
      expect(Math.hypot(...r)).toBeCloseTo(Math.hypot(...v), 5);
    }
  });
});

describe('diamond geometry (Tolkowsky ideal)', () => {
  it('girdle radius is 0.5 (half the unit diameter)', () => {
    expect(DIAMOND_INTERNALS.R_GIRDLE).toBeCloseTo(0.5, 10);
  });

  it('crown height matches tan(crown_angle) × (girdle − table-vertex radial gap)', () => {
    // Regression pin on the derived H_CROWN. The bezel axis runs from a
    // table VERTEX (not the edge midpoint) down to a girdle point at the
    // same azimuth, so the horizontal run is (R_GIRDLE − R_TABLE_VERTEX).
    // With TABLE_RATIO = 0.53 vertex-to-vertex and CROWN_ANGLE = 34.5°,
    // this lands at Tolkowsky's 16.2 % crown height.
    const expected = (DIAMOND_INTERNALS.R_GIRDLE - DIAMOND_INTERNALS.R_TABLE_VERTEX)
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
    // Bezel and pavilion axes run through a table VERTEX at φ = 0 (on
    // the +X axis). The Y component of their normals is
    // sin(φ)·sin(α) = 0 because sin(0) = 0. Star sits at φ = π/8 (the
    // table edge direction between two adjacent vertices), so its
    // normal DOES have a non-zero Y.
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
    // Bezel axis runs along φ = 0 (through a table VERTEX at (R_TABLE_VERTEX,
    // 0, H_TOP) down to a girdle POINT at (R_GIRDLE, 0, +H_GIRDLE_HALF)).
    // Anchor here is the girdle point; offset = dot with normal
    // (sin(34.5°), 0, cos(34.5°)).
    const { R_GIRDLE, planes } = DIAMOND_INTERNALS;
    const alpha    = 34.5 * Math.PI / 180;
    const expected = R_GIRDLE * Math.sin(alpha) + 0.01 * Math.cos(alpha);
    expect(planes.bezel.offset).toBeCloseTo(expected, 8);
  });

  it('pavilion plane passes through the girdle-bottom rim at φ=0', () => {
    // Pavilion mains share the bezel's azimuth (φ = 0, through a table
    // vertex). The -Z sign on the normal component cancels with the
    // -H_GIRDLE_HALF in the anchor, giving the same clean formula as
    // the bezel test.
    const { R_GIRDLE, planes } = DIAMOND_INTERNALS;
    const alpha    = 40.75 * Math.PI / 180;
    const expected = R_GIRDLE * Math.sin(alpha) + 0.01 * Math.cos(alpha);
    expect(planes.pavilion.offset).toBeCloseTo(expected, 8);
  });

  it('upper half plane passes through BOTH girdle-top rim corners at φ=0 and φ=π/8', () => {
    // The upper half facet's bottom edge must touch the shared girdle-rim
    // corners at φ=0 (shared with the neighbour UH across the adjacent
    // bezel) and φ=π/8 (shared with the neighbour UH across the star — the
    // UH-UH meeting on the girdle between adjacent bezels). Both corners
    // must lie exactly on the plane, independent of the tilt angle choice
    // — this is the invariant that fixes the bezel kite's corner and makes
    // the star/UH/bezel three-way junction well-defined.
    //
    // Catches a regression to the earlier anchor at (R_GIRDLE·cos(π/16),
    // R_GIRDLE·sin(π/16), H_GIRDLE_HALF), which pushes the plane outside
    // the girdle circle onto the circumscribing octagon rim — neither
    // girdle corner then lies on the plane, and the facets stop meeting
    // at the shared corners.
    const { R_GIRDLE, H_GIRDLE_HALF, planes } = DIAMOND_INTERNALS;
    const p = planes.upperHalf;

    const c0 = [R_GIRDLE, 0, H_GIRDLE_HALF] as const;
    const dot0 = c0[0] * p.nx + c0[1] * p.ny + c0[2] * p.nz;
    expect(dot0).toBeCloseTo(p.offset, 8);

    const cPi8 = [
      R_GIRDLE * Math.cos(Math.PI / 8),
      R_GIRDLE * Math.sin(Math.PI / 8),
      H_GIRDLE_HALF,
    ] as const;
    const dotPi8 = cPi8[0] * p.nx + cPi8[1] * p.ny + cPi8[2] * p.nz;
    expect(dotPi8).toBeCloseTo(p.offset, 8);
  });

  it('lower half plane passes through BOTH girdle-bottom rim corners at φ=0 and φ=π/8', () => {
    // Mirror of the upper half test across z=0 — same invariant: both
    // shared girdle-rim corners must lie on the plane, independent of
    // tilt angle. The LH plane's -cos(α) normal component and the
    // -H_GIRDLE_HALF anchor z cancel into the same offset formula as the
    // upper half, so the "pass through the φ=0 corner" constraint extends
    // to φ=π/8 automatically.
    const { R_GIRDLE, H_GIRDLE_HALF, planes } = DIAMOND_INTERNALS;
    const p = planes.lowerHalf;

    const c0 = [R_GIRDLE, 0, -H_GIRDLE_HALF] as const;
    const dot0 = c0[0] * p.nx + c0[1] * p.ny + c0[2] * p.nz;
    expect(dot0).toBeCloseTo(p.offset, 8);

    const cPi8 = [
      R_GIRDLE * Math.cos(Math.PI / 8),
      R_GIRDLE * Math.sin(Math.PI / 8),
      -H_GIRDLE_HALF,
    ] as const;
    const dotPi8 = cPi8[0] * p.nx + cPi8[1] * p.ny + cPi8[2] * p.nz;
    expect(dotPi8).toBeCloseTo(p.offset, 8);
  });

  it('upper half tilt stays below 40° so the bezel-star-UH junction fits in the D_8 wedge', () => {
    // Geometric invariant: at α = 40° the three-way junction of bezel,
    // star, and upper-half planes lands exactly on the φ=π/8 wedge-mirror
    // line. At α > 40° the junction escapes into the neighbouring wedge,
    // the bezel kite stops closing at its corner, and a green bezel sliver
    // appears between the star and UH facets. The cap is a property of
    // the current anchor/normal setup (UH anchored on the girdle rim at
    // φ=0, normal at φ=π/16, with the table-apothem-anchored star); change
    // any of those and this bound shifts.
    //
    // Upper bound only — any α below 35° collapses the apex back to (or
    // above) the table vertex and the star facet degenerates.
    const p = DIAMOND_INTERNALS.planes.upperHalf;
    const alphaRad = Math.acos(p.nz);   // nz = cos(α), α ∈ (0, π/2)
    const alphaDeg = alphaRad * 180 / Math.PI;
    expect(alphaDeg).toBeLessThan(40);
    expect(alphaDeg).toBeGreaterThan(35);
  });

  it('lower half tilt stays above the pavilion-main angle so LH surfaces instead of hiding inside', () => {
    // Invariant: if α_LH < α_pavilion, the LH plane lies INSIDE the
    // pavilion main everywhere below the shared girdle corner and never
    // becomes the outermost (largest signed distance) plane — the facet
    // vanishes. The constraint is a one-sided floor: any α between the
    // pavilion angle and ~50° produces a visible LH; above 50° the facet
    // just shrinks.
    const p = DIAMOND_INTERNALS.planes.lowerHalf;
    const alphaRad = Math.acos(-p.nz);  // nz = -cos(α) for pavilion-side planes
    const alphaDeg = alphaRad * 180 / Math.PI;
    expect(alphaDeg).toBeGreaterThan(DIAMOND_INTERNALS.PAVILION_ANGLE_DEG);
    expect(alphaDeg).toBeLessThan(50);
  });

  it('star plane passes through the table-edge midpoint at φ=π/8', () => {
    // Star facets sit DIRECTLY OUTSIDE a table edge: base on the edge
    // between two adjacent table vertices at φ = 0 and φ = π/4, apex
    // pointing outward onto the bezel surface. Centreline along φ = π/8
    // (edge midpoint direction). Anchor is the table-edge midpoint
    // (R_TABLE_APOTHEM·cos(π/8), R_TABLE_APOTHEM·sin(π/8), H_TOP); the
    // cos²+sin² collapse in the offset reduces to the clean formula
    // R_TABLE_APOTHEM · sin(22°) + H_TOP · cos(22°).
    //
    // Pins the star-plane anchor on R_TABLE_APOTHEM — catches a silent
    // regression back to either R_TABLE_VERTEX (vertex anchor) or to
    // the φ = 0 placement, both of which produce a visibly off cut.
    const { R_TABLE_APOTHEM, H_TOP, planes } = DIAMOND_INTERNALS;
    const alpha    = 22.0 * Math.PI / 180;
    const expected = R_TABLE_APOTHEM * Math.sin(alpha) + H_TOP * Math.cos(alpha);
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
    // strict ordering is a direct cross-check on the vertex-to-vertex table
    // convention (TABLE_RATIO applies to the vertex radius, per GIA's
    // "bezel point to bezel point" definition) and on the circumscribing-
    // octagon formula for the proxy's girdle ring.
    const i = DIAMOND_INTERNALS;
    expect(i.R_TABLE_APOTHEM).toBeLessThan(i.R_TABLE_VERTEX);
    expect(i.R_TABLE_VERTEX).toBeLessThan(i.R_GIRDLE);
    expect(i.R_GIRDLE).toBeLessThan(i.GIRDLE_R_CIRC);
    // Pin the circumscribing-octagon formula — the proxy mesh's over/under
    // coverage analysis depends on it exactly.
    expect(i.GIRDLE_R_CIRC / i.R_GIRDLE).toBeCloseTo(1 / Math.cos(Math.PI / 8), 8);
    // Vertex-to-vertex convention: R_TABLE_VERTEX = R_GIRDLE · TABLE_RATIO.
    // Catches a future accidental revert to flat-to-flat (which would
    // silently make the rendered table ~8 % larger and break the Tolkowsky
    // crown-height derivation).
    expect(i.R_TABLE_VERTEX / i.R_GIRDLE).toBeCloseTo(0.53, 8);
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
