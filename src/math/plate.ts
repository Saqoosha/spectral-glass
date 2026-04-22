/**
 * Plate tumble rotation shared between host (uniform upload) and WGSL.
 *
 * Composition: Rx(t·0.30) * Ry(t·0.20). We rotate around Y first (gentle yaw)
 * and then around X (pitch over), so a flat plate facing the camera tilts
 * upward while slowly turning — a "floating paper" motion that reads more
 * organic than a single-axis spin. Rates are chosen as an irrational ratio
 * so the combined orientation doesn't loop visibly within a ~1 min window.
 *
 * Output matches cubeRotationColumns: three vec3 columns each aligned to
 * 16 B (12 floats total, with indices 3/7/11 kept zero).
 */
export function plateRotationColumns(time: number): Float32Array {
  if (!Number.isFinite(time)) {
    throw new Error(`plateRotationColumns: time must be finite, got ${time}`);
  }
  const ay = time * 0.20;
  const ax = time * 0.30;
  const cy = Math.cos(ay);
  const sy = Math.sin(ay);
  const cx = Math.cos(ax);
  const sx = Math.sin(ax);

  // R = Rx · Ry. See src/shaders/dispersion.wgsl `sdfWavyPlate` — the closed
  // form was derived there first, hoisted to the host to avoid 2 cos + 2 sin
  // on every SDF evaluation.
  const out = new Float32Array(12);
  // Column 0 — image of the X basis vector
  out[0]  = cy;        out[1]  = sx * sy;  out[2]  = -cx * sy;
  // Column 1 — image of the Y basis vector
  out[4]  = 0;         out[5]  = cx;       out[6]  = sx;
  // Column 2 — image of the Z basis vector
  out[8]  = sy;        out[9]  = -sx * cy; out[10] = cx * cy;
  // [3], [7], [11] are the per-column padding — left zero.
  return out;
}
