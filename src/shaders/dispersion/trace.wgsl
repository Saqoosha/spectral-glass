// ---------- tracing ----------

struct Hit { ok: bool, p: vec3<f32>, t: f32 };

fn sphereTrace(ro: vec3<f32>, rd: vec3<f32>, maxT: f32) -> Hit {
  var t: f32 = 0.0;
  for (var i: i32 = 0; i < 64; i = i + 1) {
    let p = ro + rd * t;
    let d = sceneSdf(p);
    if (d < HIT_EPS) { return Hit(true, p, t); }
    t = t + max(d, MIN_STEP);
    if (t > maxT) { break; }
  }
  return Hit(false, vec3<f32>(0.0), 0.0);
}

// March from just inside the front surface until we reach the back surface.
// `ro` is the front-hit point; we skip a small entry band so the on-surface
// start pixel doesn't short-circuit at t=0.
fn insideTrace(ro: vec3<f32>, rd: vec3<f32>, maxT: f32) -> vec3<f32> {
  var t: f32 = 2.0;
  var p = ro + rd * t;
  for (var i: i32 = 0; i < 48; i = i + 1) {
    p = ro + rd * t;
    let d = -sceneSdf(p);
    if (abs(d) < HIT_EPS) { return p; }
    t = t + max(abs(d), MIN_STEP);
    if (t > maxT) { break; }
  }
  return p;
}

// Which cube pill does a world-space point belong to? Used once per fragment
// after the front hit to pick the pill whose analytical back-face intersection
// we'll evaluate in the wavelength loop. Only meaningful when the front hit is
// itself on a cube; the call site gates this with `isCube` so the result is
// untouched in pill/prism mode.
//
// Uses per-pill rotated-cube SDF (same expression as sceneSdf's cube branch),
// then ranks by ABSOLUTE distance to the surface so that:
//   - On the surface (typical sphere-trace exit, |d| ≈ HIT_EPS): the pill we
//     actually hit wins.
//   - When MIN_STEP=0.5 > HIT_EPS=0.25 lets the trace overshoot and report a
//     point slightly INSIDE a pill (sceneSdf < 0): the pill containing the
//     point still wins, because |d| is smallest for the surface closest to it.
// Picking by center distance is wrong when pills overlap or have asymmetric
// sizes — a hit on pill A's big face can be closer to pill B's center, which
// would then drive `cubeAnalyticExit` with B's halfSize/edgeR/center and
// produce a bogus exit. Linear scan, capped at MAX_PILLS=8 so it's cheap.
//
// pillCount==0 returns the unmodified default (best=0u). The caller never
// reaches this with pillCount==0 because the front sphere-trace would miss,
// but documenting the invariant keeps the function safe to repurpose.
fn hitCubePillIdx(p: vec3<f32>) -> u32 {
  let count = min(u32(frame.pillCount), MAX_PILLS);
  var best:  u32 = 0u;
  var bestD: f32 = 1e9;
  for (var i: u32 = 0u; i < count; i = i + 1u) {
    let pill  = frame.pills[i];
    let local = frame.cubeRot * (p - pill.center);
    let d     = abs(sdfCube(local, pill.halfSize, pill.edgeR));
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

// Analytical back-face intersection for a rotated rounded cube. Given a ray
// starting inside the cube (typically the refracted direction at the front
// hit), returns (pExit_world, nBack_world) where nBack faces INTO the glass —
// matching the `-sceneNormal(pExit)` convention the caller passes to
// `refract()`. Replaces the per-wavelength `insideTrace` (up to 48 SDF evals)
// plus `sceneNormal` (6 SDF evals for finite-diff gradient) with one slab
// intersection, so the savings multiply by N in Exact mode.
//
// Uses ray-box slab intersection in the cube's local (rotated) frame:
//   `pExit_local = roL + rdL * tExit`, where tExit is the smallest t at which
//   the ray leaves one of the three axis-aligned slabs [-halfSize, +halfSize].
// The outward normal is the rounded-box gradient at `pExit_local`
//   `normalize(pExit_local - clamp(pExit_local, -(h-edgeR), +(h-edgeR)))`
// which smoothly blends face→rim→corner exactly like the finite-diff normal
// did, so the rounded rim keeps its soft refraction.
//
// The slab intersection alone overshoots the rounded surface by up to edgeR
// at rim and corner exits — invisible on a still cube but visibly flickering
// on a rotating one because the overshoot direction shifts axis frame to
// frame. After the slab we run 1–2 Newton-style refinement steps using the
// rounded-box SDF gradient (`grad = pL - clamp(pL, -inner, inner)`,
// `|grad| - edgeR` is the SDF distance). Each step moves pL by `d *
// grad/|grad|` along the surface normal, converging to the true rim in
// closed form for the flat-face case and within HIT_EPS for the rim case.
// Cost: at most 2 extra clamp + length per back-trace, vs the 48-eval
// inside-trace + 6-eval finite-diff normal we replaced — still ≈ 16× fewer
// SDF evals than the original path.
struct CubeExit {
  pWorld: vec3<f32>,
  nBack:  vec3<f32>,  // inward-facing (into the glass), same sign convention as -sceneNormal
};

fn cubeAnalyticExit(roWorld: vec3<f32>, rdWorld: vec3<f32>, pillIdx: u32) -> CubeExit {
  let pill = frame.pills[pillIdx];
  let h    = pill.halfSize;

  // World → local (axis-aligned cube at origin). The rotation is orthonormal,
  // so rdL is already unit-length if rdWorld was.
  let roL = frame.cubeRot * (roWorld - pill.center);
  let rdL = frame.cubeRot * rdWorld;

  // Slab intersection per axis. A zero component in rdL gives ±inf which
  // `max` naturally rejects: ro is inside the cube so `h - roL > 0` and
  // `-h - roL < 0`, and inf carries the rejection across `max`. The tricky
  // case `(h - roL) == 0` with `rdL == 0` would give 0/0 = NaN, but that
  // requires ro exactly on the slab boundary AND a ray parallel to it — not
  // reachable from a refracted ray starting strictly inside.
  let rdInv = vec3<f32>(1.0) / rdL;
  let tHi   = (h - roL) * rdInv;
  let tLo   = (-h - roL) * rdInv;
  let tExitAxis = max(tHi, tLo);

  // First slab crossing on the way out wins. `faceDir` is +1 if we exit the
  // positive slab and -1 if we exit the negative one — derived from tHi vs tLo
  // (not `sign(rdL)`, which is 0 for rdL == 0 and would zero the normal).
  var tExit: f32;
  var faceAxisN: vec3<f32>;
  if (tExitAxis.x <= tExitAxis.y && tExitAxis.x <= tExitAxis.z) {
    tExit = tExitAxis.x;
    let faceDir = select(-1.0, 1.0, tHi.x >= tLo.x);
    faceAxisN = vec3<f32>(faceDir, 0.0, 0.0);
  } else if (tExitAxis.y <= tExitAxis.z) {
    tExit = tExitAxis.y;
    let faceDir = select(-1.0, 1.0, tHi.y >= tLo.y);
    faceAxisN = vec3<f32>(0.0, faceDir, 0.0);
  } else {
    tExit = tExitAxis.z;
    let faceDir = select(-1.0, 1.0, tHi.z >= tLo.z);
    faceAxisN = vec3<f32>(0.0, 0.0, faceDir);
  }

  var pL    = roL + rdL * tExit;
  let inner = h - vec3<f32>(pill.edgeR);

  // Refine pL onto the rounded surface. For flat-face exits this loop exits
  // immediately because the slab plane IS the rounded surface; for rim/corner
  // exits each iteration moves pL by the SDF distance along the gradient
  // direction (Newton's method on the rounded-box SDF). 2 iterations gets us
  // within float precision of the surface even at acute corners — without
  // refinement the exit position oscillates by up to edgeR per frame as the
  // cube rotates and the dominant slab axis flips, producing visible
  // blur-sharp shimmer on the rim.
  for (var i: i32 = 0; i < 2; i = i + 1) {
    let q       = clamp(pL, -inner, inner);
    let grad    = pL - q;
    let gradLen = length(grad);
    let d       = gradLen - pill.edgeR;
    if (gradLen < 1e-4) { break; }  // pL is at the inner core (degenerate)
    if (abs(d)  < HIT_EPS * 0.1) { break; }  // already on the surface
    pL = pL - (grad / gradLen) * d;
  }

  // Rounded-box gradient at the refined pL: normal blends from axis-aligned
  // in the flat face region to radial in the rim/corner region, matching the
  // finite-diff normal the sphere-trace path would compute.
  let q         = clamp(pL, -inner, inner);
  let roundedN  = pL - q;
  let roundedL2 = dot(roundedN, roundedN);
  var nOutL: vec3<f32>;
  if (roundedL2 > 1e-8) {
    nOutL = roundedN * inverseSqrt(roundedL2);
  } else {
    // Degenerate: pL is exactly on the inner-core AABB (only reachable when
    // edgeR ≥ smallest halfSize, which is filtered host-side). Fall back to
    // the slab face normal so refraction still has a defined direction.
    nOutL = faceAxisN;
  }

  // Local → world. rotation is orthonormal, so transpose == inverse.
  let rotT    = transpose(frame.cubeRot);
  let pWorld  = rotT * pL     + pill.center;
  let nOut    = rotT * nOutL;
  return CubeExit(pWorld, -nOut);
}

// (Earlier `plateCreaseAt` detector was retired in favour of the rounded-rim
// `sdfWavyPlate` — the geometric crease no longer exists, and the sentinel
// `sceneNormal` degenerate guard catches any residual cases at no extra
// per-pixel cost.)

// Same idea as hitCubePillIdx but for plates. Picks the plate whose surface
// is closest (by absolute SDF) to the front-hit point, so `plateAnalyticExit`
// operates on the plate we actually hit even when multiple plates overlap.
fn hitPlatePillIdx(p: vec3<f32>) -> u32 {
  let count = min(u32(frame.pillCount), MAX_PILLS);
  var best:  u32 = 0u;
  var bestD: f32 = 1e9;
  for (var i: u32 = 0u; i < count; i = i + 1u) {
    let pill  = frame.pills[i];
    let local = p - pill.center;
    let d     = abs(sdfWavyPlate(local, pill.halfSize, pill.edgeR));
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

// (Diamond's `hitDiamondPillIdx` lives in src/shaders/diamond.wgsl alongside
// sdfDiamond. The dispatch in fs_main below references it directly.)

// Analytical back-face intersection for a rotated wavy plate. Same payoff as
// `cubeAnalyticExit`: replaces the per-wavelength `insideTrace` (up to 48 SDF
// evals) + finite-diff `sceneNormal` (6 SDF evals) with one slab intersection
// plus 3 Newton iterations for the Z-face refinement, so in N-sample Exact
// mode we save ~60× N SDF evals per fragment.
//
// Pipeline:
//   1. World → plate-local via `frame.plateRot`, same trick as cube.
//   2. Axis-aligned slab intersection over (halfXY, halfXY, halfZ) — identical
//      to cube except the box is asymmetric (thinner in Z).
//   3. Pick the earliest-exit axis. Most refracted rays exit through the
//      opposite Z face because the plate is thin relative to its face; side
//      exits only happen at steep grazing angles near the plate rim.
//   4. For a Z-face exit: Newton-refine t against the true wavy surface
//      z = faceSign·halfZ + waveZ(x, y). Converges in 3 steps from the flat
//      slab guess because the waveZ gradient magnitude is small (≤ amp·freq
//      ≈ 0.4 at defaults) — the fixed point is very close to the slab plane.
//   5. Normal on a wavy Z face is the gradient of z − faceSign·halfZ − waveZ:
//      `(−faceSign·∂waveZ/∂x, −faceSign·∂waveZ/∂y, faceSign)` normalized. For
//      X/Y face exits (flat) it's just the unit axis.
//   6. Local → world via transpose(plateRot). Return inward-facing normal.
fn plateAnalyticExit(roWorld: vec3<f32>, rdWorld: vec3<f32>, pillIdx: u32) -> CubeExit {
  let pill = frame.pills[pillIdx];
  // Plate forces a square face (y ≡ x), so override halfSize.y from the slot
  // that might still be carrying a pill/cube value.
  let h = vec3<f32>(pill.halfSize.x, pill.halfSize.x, pill.halfSize.z);

  let roL = frame.plateRot * (roWorld - pill.center);
  let rdL = frame.plateRot * rdWorld;

  let rdInv = vec3<f32>(1.0) / rdL;
  let tHi   = (h - roL) * rdInv;
  let tLo   = (-h - roL) * rdInv;
  let tExitAxis = max(tHi, tLo);

  var tExit: f32;
  var faceSign: f32;
  var axis: i32;
  if (tExitAxis.x <= tExitAxis.y && tExitAxis.x <= tExitAxis.z) {
    axis = 0;
    tExit = tExitAxis.x;
    faceSign = select(-1.0, 1.0, tHi.x >= tLo.x);
  } else if (tExitAxis.y <= tExitAxis.z) {
    axis = 1;
    tExit = tExitAxis.y;
    faceSign = select(-1.0, 1.0, tHi.y >= tLo.y);
  } else {
    axis = 2;
    tExit = tExitAxis.z;
    faceSign = select(-1.0, 1.0, tHi.z >= tLo.z);
  }

  var pL    = roL + rdL * tExit;
  var nOutL = vec3<f32>(0.0);

  if (axis == 0) {
    nOutL = vec3<f32>(faceSign, 0.0, 0.0);
  } else if (axis == 1) {
    nOutL = vec3<f32>(0.0, faceSign, 0.0);
  } else {
    // Z face: Newton-refine against the wavy surface. f(t) = pL.z − z*(pL.xy),
    // where z*(x, y) = faceSign·halfZ + amp·sin(kx+φ)·sin(ky+φ). Phase is
    // driven by scene time so it freezes with "Stop the world" instead of
    // drifting past the frozen rotation.
    let k     = frame.waveFreq;
    let amp   = frame.waveAmp;
    let phase = frame.sceneTime * 2.0;
    for (var i: i32 = 0; i < 3; i = i + 1) {
      let sx = sin(k * pL.x + phase);
      let sy = sin(k * pL.y + phase);
      let cx = cos(k * pL.x + phase);
      let cy = cos(k * pL.y + phase);
      let waveZ = amp * sx * sy;
      let dWdx  = amp * k * cx * sy;
      let dWdy  = amp * k * sx * cy;
      let f     = pL.z - faceSign * h.z - waveZ;
      let dfdt  = rdL.z - dWdx * rdL.x - dWdy * rdL.y;
      if (abs(f) < HIT_EPS * 0.1) { break; }         // already on the surface
      if (abs(dfdt) < 1e-4)       { break; }         // tangential — bail out
      tExit = tExit - f / dfdt;
      pL    = roL + rdL * tExit;
    }
    // Validate the refined pL is actually inside the plate's XY extent.
    // The slab pre-pick used the FLAT z slab (`±h.z`), but the real plate
    // surface is `±h.z + waveZ(x, y)`; at corner cases (grazing rays at
    // the plate's XY edges) the flat-Z `tExit` can win the axis race when
    // the true exit is through an X or Y face. Newton then refines pL
    // against a Z surface that doesn't apply at this XY, and pL ends up
    // outside the plate.
    //
    // Returning a zero-normal sentinel here makes `refract(r1, vec3(0),
    // ior)` collapse to (0,0,0) — the wavelength loop's `r2TIR` branch
    // catches that and falls back to the external reflection (`reflSrc`).
    // That's the same path real TIR takes, which is acceptable for a
    // back-exit corner case (the front-side `plateCreaseAt` gate already
    // catches most of these as bg fallbacks at the front-hit stage; this
    // post-Newton bail is the secondary defense for back-exit corners
    // that the front gate can miss). Most plate creases hit the front
    // gate first, so this path is rarely executed in practice.
    if (any(abs(pL.xy) > h.xy + vec2<f32>(HIT_EPS))) {
      let rotT   = transpose(frame.plateRot);
      let pWorld = rotT * pL + pill.center;
      return CubeExit(pWorld, vec3<f32>(0.0));
    }
    // Rebuild waveZ gradient at the refined pL for the normal.
    let sx = sin(k * pL.x + phase);
    let sy = sin(k * pL.y + phase);
    let cx = cos(k * pL.x + phase);
    let cy = cos(k * pL.y + phase);
    let dWdx = amp * k * cx * sy;
    let dWdy = amp * k * sx * cy;
    // Outward normal on the wavy Z face. Derivation: for the back face
    // (faceSign = -1), outward is (+∂waveZ/∂x, +∂waveZ/∂y, -1); for the front
    // face (faceSign = +1), outward is (-∂waveZ/∂x, -∂waveZ/∂y, +1). Combined:
    //   (−faceSign·∂waveZ/∂x, −faceSign·∂waveZ/∂y, faceSign).
    nOutL = normalize(vec3<f32>(-faceSign * dWdx, -faceSign * dWdy, faceSign));
  }

  let rotT   = transpose(frame.plateRot);
  let pWorld = rotT * pL + pill.center;
  let nOut   = rotT * nOutL;
  return CubeExit(pWorld, -nOut);
}
