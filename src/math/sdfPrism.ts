type Vec3 = readonly [number, number, number];

/**
 * Triangular prism SDF. Isosceles triangle cross-section in the YZ plane
 * (apex at +Z, base at -Z), extruded along X. Sharp edges (no fillet) —
 * matches `sdfPrism` in `dispersion/sdf_primitives.wgsl`.
 *
 * halfSize semantics match `sdfPill3d`:
 *   halfSize.x = extrusion length
 *   halfSize.y = triangle base half-width
 *   halfSize.z = triangle apex height
 */
export function sdfPrism(p: Vec3, halfSize: Vec3): number {
  const hY = halfSize[1];
  const hZ = halfSize[2];

  const qy = Math.abs(p[1]);
  const qz = p[2];
  const lenInv = 1 / Math.hypot(hY, 2 * hZ);
  const dSlant = (qy * 2 * hZ + (qz - hZ) * hY) * lenInv;
  const dBase  = -hZ - qz;
  const d2     = Math.max(dSlant, dBase);

  const dX = Math.abs(p[0]) - halfSize[0];
  const wx = d2;
  const wy = dX;
  return Math.hypot(Math.max(wx, 0), Math.max(wy, 0))
       + Math.min(Math.max(wx, wy), 0);
}
