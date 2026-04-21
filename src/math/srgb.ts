/** D65 XYZ (normalized, Y=1) → linear sRGB matrix. */
const M = [
  [ 3.2404542, -1.5371385, -0.4985314],
  [-0.9692660,  1.8760108,  0.0415560],
  [ 0.0556434, -0.2040259,  1.0572252],
] as const;

export function xyzToLinearSrgb(xyz: readonly [number, number, number]): [number, number, number] {
  const [x, y, z] = xyz;
  return [
    M[0][0]*x + M[0][1]*y + M[0][2]*z,
    M[1][0]*x + M[1][1]*y + M[1][2]*z,
    M[2][0]*x + M[2][1]*y + M[2][2]*z,
  ];
}

/** IEC 61966-2-1 sRGB OETF (linear → gamma-encoded). */
export function linearToGamma(v: number): number {
  if (v <= 0) return 0;
  if (v <= 0.0031308) return v * 12.92;
  return 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
}
