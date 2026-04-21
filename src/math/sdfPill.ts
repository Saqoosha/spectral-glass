type Vec3 = readonly [number, number, number];

/** 3D pill SDF (stadium from top, rounded slab from the side). Matches the WGSL version. */
export function sdfPill3d(p: Vec3, halfSize: Vec3, edgeR: number): number {
  const hsX = halfSize[0] - edgeR;
  const hsY = halfSize[1] - edgeR;
  const rXy = Math.min(hsX, hsY);

  // Step 1: 2D stadium in XY
  const qX = Math.abs(p[0]) - hsX + rXy;
  const qY = Math.abs(p[1]) - hsY + rXy;
  const maxQx = Math.max(qX, 0);
  const maxQy = Math.max(qY, 0);
  const outer2d = Math.hypot(maxQx, maxQy);
  const inner2d = Math.min(Math.max(qX, qY), 0);
  const dXy     = outer2d + inner2d - rXy;

  // Step 2: extrude into Z with edgeR corner rounding
  const wx = dXy;
  const wy = Math.abs(p[2]) - halfSize[2] + edgeR;
  const outer = Math.hypot(Math.max(wx, 0), Math.max(wy, 0));
  const inner = Math.min(Math.max(wx, wy), 0);
  return outer + inner - edgeR;
}
