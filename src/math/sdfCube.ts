type Vec3 = readonly [number, number, number];

const L4_VISUAL_RADIUS_SCALE = (1 - Math.SQRT1_2) / (1 - Math.SQRT1_2 ** 0.5);

/**
 * Rounded box (cuboid) distance estimator. The rim uses an L4
 * superellipsoid/squircle norm instead of a circular L2 fillet, so the
 * face-to-rim curvature eases in from zero instead of jumping suddenly.
 * With equal halfSize components it's a cube; with unequal it's a rectangular
 * slab.
 *
 * Mirrored in `dispersion.wgsl`. Rotation is applied by rotating `p` BEFORE
 * calling this function (we don't store rotation per-shape).
 */
function superellipsoidLength3(x: number, y: number, z: number): number {
  const x2 = x * x;
  const y2 = y * y;
  const z2 = z * z;
  return Math.sqrt(Math.sqrt(x2 * x2 + y2 * y2 + z2 * z2));
}

export function sdfCube(p: Vec3, halfSize: Vec3, edgeR: number, smoothCurvature = true): number {
  // L4 corners cut much less than circular L2 corners at the same radius.
  // Scale the smooth radius so the 45-degree cross-section matches L2's
  // visible inset while keeping axis-aligned faces in the same place.
  const r = smoothCurvature
    ? Math.min(edgeR * L4_VISUAL_RADIUS_SCALE, halfSize[0], halfSize[1], halfSize[2])
    : edgeR;
  const qx = Math.abs(p[0]) - halfSize[0] + r;
  const qy = Math.abs(p[1]) - halfSize[1] + r;
  const qz = Math.abs(p[2]) - halfSize[2] + r;
  const ax = Math.max(qx, 0);
  const ay = Math.max(qy, 0);
  const az = Math.max(qz, 0);
  const rim = smoothCurvature ? superellipsoidLength3(ax, ay, az) : Math.hypot(ax, ay, az);
  return rim
       + Math.min(Math.max(qx, qy, qz), 0) - r;
}
