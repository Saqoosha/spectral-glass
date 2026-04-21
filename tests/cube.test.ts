import { describe, it, expect } from 'vitest';
import { cubeRotationColumns } from '../src/math/cube';

// Reconstruct a row-major 3x3 from the WGSL mat3x3<f32> layout that
// `cubeRotationColumns` emits. The buffer is 12 floats: column k lives at
// indices [k*4, k*4+3), with index `k*4+3` being the stride pad.
function rowMajor(cols: Float32Array): number[][] {
  const at = (i: number): number => cols[i] ?? NaN;  // noUncheckedIndexedAccess
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

describe('cubeRotationColumns', () => {
  it('is identity at t=0', () => {
    const m = rowMajor(cubeRotationColumns(0));
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
    const m = rowMajor(cubeRotationColumns(1.234));
    const v: [number, number, number] = [3, 4, 5];
    const r = mulMatVec(m, v);
    const inLen = Math.hypot(...v);
    const outLen = Math.hypot(...r);
    expect(outLen).toBeCloseTo(inLen, 5);
  });

  it('matches the original rz*rx composition at t=10', () => {
    // Re-derive rz * rx directly to cross-check the column-major output.
    const t  = 10;
    const ax = t * 0.31;
    const az = t * 0.20;
    const cx = Math.cos(ax), sx = Math.sin(ax);
    const cz = Math.cos(az), sz = Math.sin(az);
    const expected = [
      [ cz,          -sz * cx,      sz * sx     ],
      [ sz,           cz * cx,     -cz * sx     ],
      [ 0,            sx,           cx          ],
    ];
    const got = rowMajor(cubeRotationColumns(t));
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        expect(row(got, r)[c] ?? NaN).toBeCloseTo(row(expected, r)[c] ?? NaN, 5);
      }
    }
  });

  it('returns a fresh 12-element Float32Array in the WGSL padded layout', () => {
    const a = cubeRotationColumns(1);
    const b = cubeRotationColumns(1);
    expect(a).not.toBe(b);
    expect(a.length).toBe(12);
    expect(b.length).toBe(12);
  });

  it('leaves the per-column stride pad at zero', () => {
    const m = cubeRotationColumns(1.7);
    expect(m[3]).toBe(0);
    expect(m[7]).toBe(0);
    expect(m[11]).toBe(0);
  });

  it('rejects non-finite time instead of producing NaN rotations', () => {
    expect(() => cubeRotationColumns(NaN)).toThrow(/finite/);
    expect(() => cubeRotationColumns(Infinity)).toThrow(/finite/);
    expect(() => cubeRotationColumns(-Infinity)).toThrow(/finite/);
  });
});
