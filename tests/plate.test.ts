import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { plateRotationColumns } from '../src/math/plate';

const here = dirname(fileURLToPath(import.meta.url));

// Mirrors tests/cube.test.ts. Plate uses the same WGSL mat3x3<f32> layout as
// cube — column k at indices [k*4, k*4+3) with `k*4+3` as the stride pad —
// but composes rx·ry instead of rz·rx, so the closed-form derivation differs
// while the layout / orthonormality / NaN guard tests are identical.
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

describe('plateRotationColumns', () => {
  it('is identity at t=0', () => {
    const m = rowMajor(plateRotationColumns(0));
    const expected = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        expect(row(m, r)[c] ?? NaN).toBeCloseTo(row(expected, r)[c] ?? NaN, 6);
      }
    }
  });

  it('preserves vector length (orthonormal rotation)', () => {
    const m = rowMajor(plateRotationColumns(1.234));
    const v: [number, number, number] = [3, 4, 5];
    const r = mulMatVec(m, v);
    const inLen = Math.hypot(...v);
    const outLen = Math.hypot(...r);
    expect(outLen).toBeCloseTo(inLen, 5);
  });

  it('matches the original rx*ry composition at t=10', () => {
    // Re-derive R = Rx · Ry directly to cross-check the column-major output.
    const t  = 10;
    const ay = t * 0.20;
    const ax = t * 0.30;
    const cy = Math.cos(ay), sy = Math.sin(ay);
    const cx = Math.cos(ax), sx = Math.sin(ax);
    const expected = [
      [ cy,          0,        sy         ],
      [ sx * sy,     cx,      -sx * cy    ],
      [-cx * sy,     sx,       cx * cy    ],
    ];
    const got = rowMajor(plateRotationColumns(t));
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        expect(row(got, r)[c] ?? NaN).toBeCloseTo(row(expected, r)[c] ?? NaN, 5);
      }
    }
  });

  it('returns a fresh 12-element Float32Array in the WGSL padded layout', () => {
    const a = plateRotationColumns(1);
    const b = plateRotationColumns(1);
    expect(a).not.toBe(b);
    expect(a.length).toBe(12);
    expect(b.length).toBe(12);
  });

  it('leaves the per-column stride pad at zero', () => {
    const m = plateRotationColumns(1.7);
    expect(m[3]).toBe(0);
    expect(m[7]).toBe(0);
    expect(m[11]).toBe(0);
  });

  it('rejects non-finite time instead of producing NaN rotations', () => {
    expect(() => plateRotationColumns(NaN)).toThrow(/finite/);
    expect(() => plateRotationColumns(Infinity)).toThrow(/finite/);
    expect(() => plateRotationColumns(-Infinity)).toThrow(/finite/);
  });
});

describe('sdfWavyPlate shader rim', () => {
  it('uses the shared smooth-curvature rounded box rim', () => {
    const wgsl = readFileSync(resolve(here, '../src/shaders/dispersion/sdf_primitives.wgsl'), 'utf8');
    const match = /fn sdfWavyPlate[\s\S]*?return box \* frame\.waveLipFactor;\n\}/.exec(wgsl);
    expect(match).not.toBeNull();
    const body = match?.[0] ?? '';
    expect(body).toContain('visualRoundRadius(edgeR');
    expect(body).toContain('roundedLength3(max(q');
    expect(body).not.toContain('let box    = length(max(q');
  });
});
