/**
 * Cube rotation matrix shared between host (uniform upload) and WGSL.
 *
 * Mirrors `cubeRotation(t)` that used to live in dispersion.wgsl: rz(t·0.20) *
 * rx(t·0.31), so the cube tumbles slowly enough to tease out per-frame
 * dispersion changes without making it nausea-inducing.
 *
 * Output is the exact byte layout WGSL uses for `mat3x3<f32>` uniforms: three
 * vec3 columns each aligned to 16 B. Data lives at indices 0..2, 4..6, 8..10;
 * indices 3, 7, 11 are zero padding. Caller just `Float32Array.set`s this at
 * the matrix's base offset.
 */
export function cubeRotationColumns(time: number): Float32Array {
  // NaN/±Infinity would poison the rotation matrix with NaN, which then slips
  // into the GPU uniform and gets "healed" at the far end by the shader's
  // degenerate-normal fallback — a silent visual corruption instead of a hard
  // failure. Catch it here while the call site is still on the stack.
  if (!Number.isFinite(time)) {
    throw new Error(`cubeRotationColumns: time must be finite, got ${time}`);
  }
  const ax = time * 0.31;
  const az = time * 0.20;
  const cx = Math.cos(ax);
  const sx = Math.sin(ax);
  const cz = Math.cos(az);
  const sz = Math.sin(az);

  const out = new Float32Array(12);  // 3 columns × (vec3 data + 1 pad) = 12
  // Column 0
  out[0]  = cz;       out[1]  = sz;       out[2]  = 0;
  // Column 1
  out[4]  = -sz * cx; out[5]  = cz * cx;  out[6]  = sx;
  // Column 2
  out[8]  = sz * sx;  out[9]  = -cz * sx; out[10] = cx;
  // out[3], [7], [11] are the 4-byte pads at the end of each column — left
  // at 0 by Float32Array's default init.
  return out;
}
