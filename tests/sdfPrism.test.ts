import { describe, it, expect } from 'vitest';
import { sdfPrism } from '../src/math/sdfPrism';

// Prism: extrusion length 200 in X, base half-width 30 in Y, apex height 80 in Z.
const HS: [number, number, number] = [200, 30, 80];

describe('sdfPrism', () => {
  it('returns a negative value at the interior centroid', () => {
    expect(sdfPrism([0, 0, 0], HS)).toBeLessThan(0);
  });

  it('returns a positive value far outside along each axis', () => {
    expect(sdfPrism([500,  0,  0], HS)).toBeGreaterThan(0);
    expect(sdfPrism([  0, 500, 0], HS)).toBeGreaterThan(0);
    expect(sdfPrism([  0,  0, 500], HS)).toBeGreaterThan(0);
  });

  it('returns ~0 on the apex edge (y=0, z=hZ)', () => {
    expect(Math.abs(sdfPrism([0, 0, HS[2]], HS))).toBeLessThan(0.5);
  });

  it('returns ~0 on the base at the centerline (z=-hZ, |y|<hY)', () => {
    expect(Math.abs(sdfPrism([0, 0, -HS[2]], HS))).toBeLessThan(0.5);
  });

  it('is symmetric across the YZ mirror (x→-x)', () => {
    const p: [number, number, number] = [50, 10, 20];
    const base = sdfPrism(p, HS);
    expect(sdfPrism([-p[0], p[1], p[2]], HS)).toBeCloseTo(base, 6);
  });

  it('is symmetric across the XZ mirror (y→-y)', () => {
    const p: [number, number, number] = [50, 10, 20];
    const base = sdfPrism(p, HS);
    expect(sdfPrism([p[0], -p[1], p[2]], HS)).toBeCloseTo(base, 6);
  });

  it('apex is narrower than base (small |y| allowed at high z)', () => {
    const nearApexInside  = sdfPrism([0,  2, HS[2] - 10], HS);
    const nearApexOutside = sdfPrism([0, 20, HS[2] - 10], HS);
    expect(nearApexInside).toBeLessThan(nearApexOutside);
    expect(nearApexOutside).toBeGreaterThan(0);
  });
});
