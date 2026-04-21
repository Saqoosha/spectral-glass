import { describe, it, expect } from 'vitest';
import { sdfPill3d } from '../src/math/sdfPill';

const HS  : [number, number, number] = [160, 44, 20];
const EDGE = 14;

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
    // Monotonic increase as we leave the interior; difference ≈ sqrt(3) · 0.1 for a rounded edge.
    expect(b - a).toBeGreaterThan(0);
    expect(b - a).toBeLessThan(0.3);
  });
});
