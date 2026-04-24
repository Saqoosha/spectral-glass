import { describe, it, expect } from 'vitest';
import { sdfPill3d } from '../src/math/sdfPill';

const HS  : [number, number, number] = [160, 44, 20];
const EDGE = 14;
const L4_VISUAL_RADIUS_SCALE = (1 - Math.SQRT1_2) / (1 - Math.SQRT1_2 ** 0.5);

describe('sdfPill3d', () => {
  it('returns a negative value inside the pill (at origin)', () => {
    expect(sdfPill3d([0, 0, 0], HS, EDGE)).toBeLessThan(0);
  });

  it('returns a positive value far outside the pill', () => {
    expect(sdfPill3d([500, 0, 0], HS, EDGE)).toBeGreaterThan(0);
    expect(sdfPill3d([0, 500, 0], HS, EDGE)).toBeGreaterThan(0);
    expect(sdfPill3d([0, 0, 500], HS, EDGE)).toBeGreaterThan(0);
  });

  it('returns ~0 on the top face near the center (z = hz)', () => {
    expect(Math.abs(sdfPill3d([0, 0, HS[2]], HS, EDGE))).toBeLessThan(0.5);
  });

  it('along the long axis, leaves the shape near x = hx', () => {
    expect(sdfPill3d([HS[0] - EDGE - 1, 0, 0], HS, EDGE)).toBeLessThan(0);
    expect(sdfPill3d([HS[0] + 2,        0, 0], HS, EDGE)).toBeGreaterThan(0);
  });

  it('is symmetric across all three axes', () => {
    const p: [number, number, number] = [30, 10, 5];
    const base = sdfPill3d(p, HS, EDGE);
    expect(sdfPill3d([-p[0],  p[1],  p[2]], HS, EDGE)).toBeCloseTo(base, 6);
    expect(sdfPill3d([ p[0], -p[1],  p[2]], HS, EDGE)).toBeCloseTo(base, 6);
    expect(sdfPill3d([ p[0],  p[1], -p[2]], HS, EDGE)).toBeCloseTo(base, 6);
  });

  it('rounded edges: stepping diagonally from a corner is smooth (no sharp creases)', () => {
    const a = sdfPill3d([HS[0] - EDGE, HS[1] - EDGE, HS[2] - EDGE], HS, EDGE);
    const b = sdfPill3d([HS[0] - EDGE + 0.1, HS[1] - EDGE + 0.1, HS[2] - EDGE + 0.1], HS, EDGE);
    // Monotonic increase as we leave the interior; L4 rounding stays flatter than an L2 rim.
    expect(b - a).toBeGreaterThan(0);
    expect(b - a).toBeLessThan(0.3);
  });

  it('eases the top face into the side rim with squircle curvature', () => {
    const zR = Math.min(EDGE * L4_VISUAL_RADIUS_SCALE, HS[2]);
    const q = 1;
    const zq = Math.pow(zR ** 4 - q ** 4, 0.25);
    const p: [number, number, number] = [0, HS[1] - zR + q, HS[2] - zR + zq];
    expect(sdfPill3d(p, HS, EDGE)).toBeCloseTo(0, 5);
    expect(HS[2] - p[2]).toBeLessThan(0.001);
  });

  it('compensates smooth L4 so its 45-degree Z roundover matches the legacy L2 rim', () => {
    const hs: [number, number, number] = [160, 44, 80];
    const q = EDGE / Math.SQRT2;
    const p: [number, number, number] = [0, hs[1] - EDGE + q, hs[2] - EDGE + q];
    expect(sdfPill3d(p, hs, EDGE, false)).toBeCloseTo(0, 5);
    expect(sdfPill3d(p, hs, EDGE, true)).toBeCloseTo(0, 5);
  });

  it('becomes a true pill in XY when edgeR exceeds the short half-axis', () => {
    const hugeEdge = 100;
    const cap45: [number, number, number] = [
      HS[0] - HS[1] + HS[1] / Math.SQRT2,
      HS[1] / Math.SQRT2,
      0,
    ];
    for (const smooth of [false, true]) {
      expect(sdfPill3d([0, HS[1], 0], HS, hugeEdge, smooth)).toBeCloseTo(0, 5);
      expect(sdfPill3d([HS[0], 0, 0], HS, hugeEdge, smooth)).toBeCloseTo(0, 5);
      expect(sdfPill3d(cap45, HS, hugeEdge, smooth)).toBeCloseTo(0, 5);
      expect(sdfPill3d([HS[0], 0, HS[2]], HS, hugeEdge, smooth)).toBeGreaterThan(0);
    }
  });

  it('keeps the XY rounded-rect silhouette unchanged when smooth curvature is toggled', () => {
    const q = EDGE / Math.SQRT2;
    const capPoint: [number, number, number] = [HS[0] - EDGE + q, HS[1] - EDGE + q, 0];
    expect(sdfPill3d(capPoint, HS, EDGE, false)).toBeCloseTo(0, 5);
    expect(sdfPill3d(capPoint, HS, EDGE, true)).toBeCloseTo(0, 5);
  });
});
