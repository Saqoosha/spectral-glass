fn sceneSdf(p: vec3<f32>) -> f32 {
  let count   = min(u32(frame.pillCount), MAX_PILLS);
  let shapeId = i32(frame.shape + 0.5);
  var d: f32 = 1e9;
  for (var i: u32 = 0u; i < count; i = i + 1u) {
    let pill  = frame.pills[i];
    let local = p - pill.center;
    var pd: f32;
    if (shapeId == 2) {
      // Cube is rotated in local space before SDF evaluation. Rotation is a
      // uniform (computed once on the host per frame), so every SDF call here
      // is just one mat-vec — the heavy cos/sin used to run per call.
      pd = sdfCube(frame.cubeRot * local, pill.halfSize, pill.edgeR);
    } else if (shapeId == 1) {
      pd = sdfPrism(local, pill.halfSize, pill.edgeR);
    } else if (shapeId == 3) {
      // Plate: wave amp + freq + time come from global uniforms
      // (frame.waveAmp / frame.waveFreq / frame.sceneTime). pill.edgeR is
      // the rim radius for the rounded corner that smooths the wavy front
      // Z face into the flat side X / Y faces (eliminates the rim crease).
      pd = sdfWavyPlate(local, pill.halfSize, pill.edgeR);
    } else if (shapeId == 4) {
      // Diamond: size comes from `frame.diamondSize` (single global slider),
      // so per-pill `halfSize` / `edgeR` slots are ignored. Multi-instance
      // still works because `local` is relative to each pill's `center`.
      pd = sdfDiamond(local, frame.diamondSize);
    } else {
      pd = sdfPill(local, pill.halfSize, pill.edgeR);
    }
    d = min(d, pd);
  }
  return d;
}

// Upper bound on the internal path length a ray can take inside any pill in
// the scene. Used to cap insideTrace. For a rotated cube the diagonal (√3
// times the max half-side, doubled) is the longest possible chord; for pill
// and prism the 3D diagonal of the AABB is an upper bound too.
//
// Diamond takes a shape-specific path: its halfSize is coincidental (main.ts
// writes `pill.hx/hy/hz = diamondSize/2` so the drag hit-test works, but the
// SDF reads `frame.diamondSize` directly). Using the diamondSize-derived
// upper bound here decouples the internal-trace cap from the pills array,
// so a future refactor that stops initialising halfSize for diamond can't
// silently clip the pavilion's deepest rays.
fn maxInternalPath() -> f32 {
  let shapeId = i32(frame.shape + 0.5);
  if (shapeId == 4) {
    // Diamond: true longest interior chord is 2·R_GIRDLE = diameter
    // (between opposite girdle points). Doubled again for safety covers
    // multi-bounce paths Phase B will eventually add; `* 2.0` is what
    // cube/pill/prism/plate also use (2× the half-size = full extent).
    return max(frame.diamondSize * 2.0, 32.0);
  }
  let count = min(u32(frame.pillCount), MAX_PILLS);
  var m: f32 = 0.0;
  for (var i: u32 = 0u; i < count; i = i + 1u) {
    let hs = frame.pills[i].halfSize;
    m = max(m, length(hs) * 2.0);
  }
  return max(m, 32.0);  // floor so degenerate zero-size pills don't stop march
}

// Returns a unit-length surface normal at `p`, OR the zero vector as a
// sentinel meaning "the local SDF gradient is degenerate, don't trust this
// hit". Callers must check `dot(n, n) > 0.5` and fall back to the bg path
// for sentinel hits — see fs_main below.
//
// Degenerate-gradient guard exists because at a silhouette / wave crest
// the six finite-diff probes can land on near-equal SDF values (e.g. a
// wavy plate hit exactly at the apex of a bump where ∇sdf is tangent to
// the view ray), making `g` near-zero. `normalize(0)` returns NaN, which
// would cascade:
//   refract(rd, NaN, eta) → all-NaN r1
//   dot(r1, r1) < 1e-4 compares NaN < 1e-4 → false → continue NOT taken
//   the NaN flows through inside-trace, photo sample, CMF normalise →
//   final outRgb NaN → written to history → permanent white/black pixel.
// Visible specifically when TAA is off: with the ray frozen at the pixel
// centre every frame, the same degenerate hit re-fires forever; with TAA
// jitter the ±0.5 px sub-pixel offset usually misses the singular point.
//
// An arbitrary fallback normal (e.g. `(0, 0, 1)`) is worse than no normal:
// it produces a valid-but-wrong refraction that visibly differs from the
// neighbours' correct refraction, leaving a dot speckle along edges. The
// honest sentinel + bg fallback at the call site renders these pixels as
// background, which blends seamlessly with the surrounding silhouette.
fn sceneNormal(p: vec3<f32>) -> vec3<f32> {
  let e = vec2<f32>(HIT_EPS, 0.0);
  let g = vec3<f32>(
    sceneSdf(p + e.xyy) - sceneSdf(p - e.xyy),
    sceneSdf(p + e.yxy) - sceneSdf(p - e.yxy),
    sceneSdf(p + e.yyx) - sceneSdf(p - e.yyx),
  );
  let len2 = dot(g, g);
  if (len2 < 1e-8) {
    return vec3<f32>(0.0);
  }
  return g * inverseSqrt(len2);
}
