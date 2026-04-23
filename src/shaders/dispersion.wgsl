const MAX_PILLS: u32 = 8u;
const MAX_N:      i32 = 64;
const HIT_EPS:    f32 = 0.25;  // hit tolerance (small — thin pills survive)
const MIN_STEP:   f32 = 0.5;   // min march step (larger — loop doesn't stall on near-zero SDF)

struct PillGpu {
  center:   vec3<f32>,
  edgeR:    f32,
  halfSize: vec3<f32>,
  _pad:     f32,
};

struct Frame {
  resolution:         vec2<f32>,
  photoSize:          vec2<f32>,
  n_d:                f32,
  V_d:                f32,
  sampleCount:        f32,
  refractionStrength: f32,
  jitter:             f32,
  refractionMode:     f32,
  pillCount:          f32,
  applySrgbOetf:      f32,  // unused by this shader (sRGB encoding moved to postprocess.wgsl). Slot is kept so the Frame UBO layout stays stable — host still writes 0/1 (uniforms.ts) and tests/uniformsLayout.test.ts pins the name. Reclaiming it means touching all three together.
  shape:              f32,  // 0 = pill (stadium), 1 = prism, 2 = cube (rotates), 3 = plate (wavy, tumbles), 4 = diamond (round brilliant, rotates)
  time:               f32,  // wall-clock seconds since start (always advancing, even while paused). Drives the noise streams: TAA sub-pixel jitter and per-pixel wavelength stratification. Rotation matrices are derived from `sceneTime` (below), NOT from this field — see `cubeRot` / `plateRot` / `diamondRot` and the time-stream split in src/main.ts.
  historyBlend:       f32,  // 0.2 steady state, 1.0 when the scene changed this frame
  heroLambda:         f32,  // jittered each frame in [380,700]; Hero mode uses this
  cameraZ:            f32,  // distance from screen plane (z=0) to camera, in pixels
  projection:         f32,  // 0 = orthographic, 1 = perspective
  debugProxy:         f32,  // 1 = tint every proxy fragment pink (debug view)
  taaEnabled:         f32,  // 1 = jitter ray origin within the pixel so history EMA antialiases shader-decided silhouettes
  // Cube rotation (rz·rx composed on the host from `sceneTime` — see
  // src/math/cube.ts). Driven by the scene/motion stream so "Stop the world"
  // freezes the tumble while the noise stream (`time`) keeps advancing AA.
  // Uploaded as a uniform so every cube SDF evaluation is just one mat-vec
  // instead of four cos/sin. With cubeAnalyticExit replacing
  // the per-wavelength inside-trace, the cube path's remaining sdfCube traffic
  // is sphereTrace (up to 64 iters) + the front sceneNormal (6 evals), each
  // going through sceneSdf which scans all live pills, plus a per-pill scan
  // inside hitCubePillIdx — sdfCube evaluations still total roughly 70 ×
  // pillCount per fragment, so that many cos/sin pairs are saved per frame.
  // cubeAnalyticExit also uses cubeRot a few more times directly (two forward
  // multiplies, one transpose, two inverse multiplies — all from the same
  // uniform, no extra cos/sin).
  cubeRot:            mat3x3<f32>,
  // Previous frame's cube rotation. Together with the current `cubeRot` this
  // lets the TAA reprojection path recover the screen position of the hit
  // point one frame ago, so the history read follows the rotating cube's
  // surface instead of smearing stale neighboring pixels into it.
  cubeRotPrev:        mat3x3<f32>,
  // Plate rotation (rx·ry composed on the host from `sceneTime` — see
  // src/math/plate.ts). Same scene/motion-stream gating as `cubeRot` above,
  // so paused plates freeze too. Used by sdfWavyPlate for both SDF evaluation
  // and by vs_proxy to rotate the proxy bounding box so it stays tight
  // (cube-style).
  plateRot:           mat3x3<f32>,
  // Previous frame's plate rotation. Analogous to cubeRotPrev above — feeds
  // the TAA reprojection path so tumbling plates also keep sharp refracted
  // texture instead of ghosting.
  plateRotPrev:       mat3x3<f32>,
  // Diamond rotation (rx·ry composed on the host from `sceneTime` — see
  // src/math/diamond.ts). Fixed -20° X-axis tilt + Y-axis spin, same host-side
  // pattern as cube/plate. Used by sdfDiamond (fold to fundamental wedge) and
  // by vs_proxy (rotate the AABB so the proxy stays tight as the diamond
  // spins).
  diamondRot:         mat3x3<f32>,
  // Previous frame's diamond rotation for the TAA reprojection path — mirrors
  // cubeRotPrev / plateRotPrev so spinning diamonds keep sharp refracted
  // texture instead of ghosting.
  diamondRotPrev:     mat3x3<f32>,
  // Plate wave parameters. `waveLipFactor` is the precomputed safety factor
  // 1/sqrt(1 + (amp·freq)²) applied inside sdfWavyPlate so the raw
  // (box - wave) value is a valid Lipschitz bound — host-side computation
  // avoids an `inverseSqrt` on every SDF eval. (See the derivation in
  // src/webgpu/uniforms.ts for why max|∇waveZ|² = (amp·k)², not 2·(amp·k)².)
  //
  // `sceneTime` is the time value for scene motion (rotation + wave phase).
  // It freezes when the user hits "Stop the world" while the noise-seeding
  // `time` field above keeps advancing — so TAA sub-pixel jitter keeps
  // accumulating and AA quality continues to improve on paused scenes.
  waveAmp:            f32,
  waveFreq:           f32,
  waveLipFactor:      f32,
  sceneTime:          f32,
  // Diamond parameters. `diamondSize` is the girdle diameter in pixels — the
  // sdfDiamond SDF scales all facet offsets by this at eval time so one uniform
  // covers every diamond in the scene (all instances share the size slider).
  // `diamondWireframe` toggles the facet-edge overlay (1.0 = on).
  // `diamondFacetColor` toggles a flat-shaded debug fill where each facet
  // class gets a distinct colour — useful for checking adjacency + coverage
  // without refraction / dispersion confusing the signal. The remaining
  // slot is reserved for future diamond controls (e.g., angle overrides).
  diamondSize:        f32,
  diamondWireframe:   f32,
  diamondFacetColor:  f32,
  _diamondPad2:       f32,
  pills:              array<PillGpu, MAX_PILLS>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var photoTex: texture_2d<f32>;
@group(0) @binding(2) var photoSmp: sampler;
@group(0) @binding(3) var historyTex: texture_2d<f32>;
@group(0) @binding(4) var historySmp: sampler;

// ---------- coords ----------

fn coverUv(uv: vec2<f32>) -> vec2<f32> {
  let sA = frame.resolution.x / frame.resolution.y;
  let pA = frame.photoSize.x  / frame.photoSize.y;
  var s  = vec2<f32>(1.0);
  if (sA > pA) { s = vec2<f32>(1.0, pA / sA); } else { s = vec2<f32>(sA / pA, 1.0); }
  return (uv - vec2<f32>(0.5)) * s + vec2<f32>(0.5);
}

// ---------- SDF ----------

fn sdfPill(p: vec3<f32>, halfSize: vec3<f32>, edgeR: f32) -> f32 {
  let hsXY = halfSize.xy - vec2<f32>(edgeR);
  let rXY  = min(hsXY.x, hsXY.y);
  let qXY  = abs(p.xy) - hsXY + vec2<f32>(rXY);
  let dXy  = length(max(qXY, vec2<f32>(0.0))) + min(max(qXY.x, qXY.y), 0.0) - rXY;
  let w    = vec2<f32>(dXy, abs(p.z) - halfSize.z + edgeR);
  return length(max(w, vec2<f32>(0.0))) + min(max(w.x, w.y), 0.0) - edgeR;
}

// Rounded box / cuboid. Equal halfSize = cube.
fn sdfCube(p: vec3<f32>, halfSize: vec3<f32>, edgeR: f32) -> f32 {
  let q = abs(p) - halfSize + vec3<f32>(edgeR);
  return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - edgeR;
}

// Isosceles triangle in YZ (apex +Z, base -Z), extruded along X. Half-sizes
// match sdfPill: halfSize.x is extrusion length, halfSize.y the triangle base
// half-width, halfSize.z the apex height.
fn sdfPrism(p: vec3<f32>, halfSize: vec3<f32>, edgeR: f32) -> f32 {
  let hY = halfSize.y;
  let hZ = halfSize.z;
  let qy = abs(p.y);
  let qz = p.z;
  let lenInv = 1.0 / sqrt(hY * hY + 4.0 * hZ * hZ);
  let dSlant = (qy * 2.0 * hZ + (qz - hZ) * hY) * lenInv;
  let dBase  = -hZ - qz;
  let d2     = max(dSlant, dBase);

  let dX = abs(p.x) - halfSize.x;
  let w  = vec2<f32>(d2, dX);
  return length(max(w, vec2<f32>(0.0))) + min(max(w.x, w.y), 0.0) - edgeR;
}

// Diamond (round brilliant cut) SDF + proxy-mesh helpers live in a separate
// shader unit: src/shaders/diamond.wgsl. It's concatenated at pipeline build
// time so sdfDiamond / hitDiamondPillIdx / diamondProxyVertex are visible to
// the dispatch sites below (sceneSdf, fs_main, reprojectHit, vs_proxy). The
// split keeps dispersion.wgsl focused on trace + SDF framework and leaves
// Phase B's multi-bounce TIR trace for diamond a clear home.

// Thick square plate as a CONSTANT-THICKNESS bent sheet. The midsurface
// follows a sin·sin cross wave; both faces of the plate ride that midsurface
// together so the thickness stays uniform at every (x, y) — the plate reads
// as a rippling sheet rather than a pulsating blob.
//
// `halfSize.x` drives the square face extent (y is forced to match so the
// plate stays square regardless of the pillShort slider), `halfSize.z` is
// the half-thickness. Wave amplitude + frequency come from uniforms so they
// can be wired to UI sliders without per-shape slot juggling.
//
// Tumble rotation is precomputed on the host as `frame.plateRot` (see
// src/math/plate.ts) — same trick as cubeRot, saves four cos/sin per SDF eval.
//
// The returned distance is scaled by `frame.waveLipFactor` (computed on the
// host as 1/sqrt(1 + (amp·freq)²)) because the z-shift breaks the Lipschitz
// constant along x/y: the shifted SDF has gradient magnitude
// sqrt(1 + (∂waveZ/∂x)² + (∂waveZ/∂y)²) which grows with amp·freq. Dividing
// by that tight bound — see the derivation in src/webgpu/uniforms.ts on why
// max(|∇waveZ|²) = (amp·k)², NOT 2·(amp·k)² — keeps raymarch steps safely
// inside the true distance so the trace never tunnels through the thin slab,
// while staying as aggressive as the current wave parameters allow (~0.92 at
// defaults, previously a conservative hardcoded 0.6 — ≈35% fewer sphereTrace
// steps at the same safety margin, or equivalently ≈53% more progress per
// step).
fn sdfWavyPlate(pIn: vec3<f32>, halfSize: vec3<f32>, edgeR: f32) -> f32 {
  let p = frame.plateRot * pIn;

  // Force square XY face (halfSize.y ignored — UI hides pillShort for plate).
  let h = vec3<f32>(halfSize.x, halfSize.x, halfSize.z);

  // Midsurface z-displacement. Animated via phase-shift on scene time so the
  // plate "pulses" independently of its tumble; 2·sceneTime reads faster
  // than the 0.2–0.3 rad/s tumble so both motions are visible at once.
  // Scene time is used (not the caller's `t`) so pause/unpause freezes the
  // wave in sync with the rotation.
  let k     = frame.waveFreq;
  let st    = frame.sceneTime;
  let waveZ = frame.waveAmp * sin(k * p.x + st * 2.0) * sin(k * p.y + st * 2.0);

  // Slab centered on the wavy midsurface. Both faces ride `waveZ` together,
  // giving a constant-thickness bent sheet instead of a bulge/pinch volume.
  // Rounded box (same shrink-then-rim pattern as sdfCube, edgeR for the
  // fillet between the wavy front Z face and the flat side X / Y faces) so
  // the rim crease that would otherwise live at `q.x = q.z = 0` doesn't
  // exist as a math singularity — the rounded transition gives a continuous
  // gradient across the corner, which removes the dominant source of the
  // "speckle along plate edge" artifact at its origin instead of relying
  // on `plateCreaseAt` to detect-and-divert it.
  let pShift = vec3<f32>(p.x, p.y, p.z - waveZ);
  let q      = abs(pShift) - h + vec3<f32>(edgeR);
  let box    = length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - edgeR;

  return box * frame.waveLipFactor;
}

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

// ---------- spectral math ----------

// Cauchy + Abbe number (glTF KHR_materials_dispersion formulation).
// Clamped to 1.0 because ior<1 breaks Snell direction via refract().
fn cauchyIor(lambda: f32, n_d: f32, V_d: f32) -> f32 {
  return max(n_d + (n_d - 1.0) / V_d * (523655.0 / (lambda * lambda) - 1.5168), 1.0);
}

fn gLobe(lambda: f32, mu: f32, s1: f32, s2: f32) -> f32 {
  let sigma = select(s2, s1, lambda < mu);
  let t = (lambda - mu) / sigma;
  return exp(-0.5 * t * t);
}

// Wyman-Sloan-Shirley (JCGT 2013) analytic CIE 1931 2° XYZ matching functions.
fn cieXyz(lambda: f32) -> vec3<f32> {
  let x =  0.362 * gLobe(lambda, 442.0, 16.0, 26.7)
        +  1.056 * gLobe(lambda, 599.8, 37.9, 31.0)
        -  0.065 * gLobe(lambda, 501.1, 20.4, 26.2);
  let y =  0.821 * gLobe(lambda, 568.8, 46.9, 40.5)
        +  0.286 * gLobe(lambda, 530.9, 16.3, 31.1);
  let z =  1.217 * gLobe(lambda, 437.0, 11.8, 36.0)
        +  0.681 * gLobe(lambda, 459.0, 26.0, 13.8);
  return vec3<f32>(x, y, z);
}

// D65 XYZ → linear sRGB. WGSL mat3x3 is column-major, so each row of literals
// below is the column of the textbook matrix.
fn xyzToSrgb(c: vec3<f32>) -> vec3<f32> {
  let m = mat3x3<f32>(
     3.2404542, -0.9692660,  0.0556434,
    -1.5371385,  1.8760108, -0.2040259,
    -0.4985314,  0.0415560,  1.0572252,
  );
  return m * c;
}

fn schlickFresnel(cosT: f32, n_d: f32) -> f32 {
  let f0 = pow((n_d - 1.0) / (n_d + 1.0), 2.0);
  let k  = 1.0 - clamp(cosT, 0.0, 1.0);
  return f0 + (1.0 - f0) * k * k * k * k * k;
}

// Hash a 2D input to [0,1). Dave Hoskins' hash12, small-footprint + low bias
// enough for variance reduction (we're not doing security here).
fn hash21(p: vec2<f32>) -> f32 {
  var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
  p3 = p3 + vec3<f32>(dot(p3, p3.yzx + vec3<f32>(33.33)));
  return fract((p3.x + p3.y) * p3.z);
}

// sRGB OETF lives in src/shaders/postprocess.wgsl now — this shader writes
// linear RGB to the rgba16f intermediate and the post-process pass handles
// the display encoding in one place (same place FXAA runs).

// ---------- proxy vertex shader ----------
//
// Draws a per-pill 3D bounding box (a unit cube scaled to halfSize+edgeR,
// optionally rotated for shape==cube) instead of a fullscreen triangle. The
// rasterizer produces fragments for the exact projected silhouette of the
// proxy mesh, so tight coverage on rotated cubes is automatic.

// Unit cube, 36 verts (= CUBE_PROXY_VERT_COUNT from src/math/diamond.ts),
// 12 tris, CCW outward winding (so `cullMode: 'back'` leaves one invocation
// per covered pixel). The array size must match CUBE_PROXY_VERT_COUNT, which
// the pipeline.ts draw call and the maxVerts guard below also read from.
const CUBE_VERTS: array<vec3<f32>, CUBE_PROXY_VERT_COUNT> = array<vec3<f32>, CUBE_PROXY_VERT_COUNT>(
  // +X face (CCW outward: swap V1,V2 vs the other faces' pattern because the
  // outward normal flips sign of cross(E1, E2) when the face is on the +X side)
  vec3<f32>( 1.0,-1.0,-1.0), vec3<f32>( 1.0, 1.0, 1.0), vec3<f32>( 1.0,-1.0, 1.0),
  vec3<f32>( 1.0,-1.0,-1.0), vec3<f32>( 1.0, 1.0,-1.0), vec3<f32>( 1.0, 1.0, 1.0),
  // -X face
  vec3<f32>(-1.0,-1.0,-1.0), vec3<f32>(-1.0,-1.0, 1.0), vec3<f32>(-1.0, 1.0, 1.0),
  vec3<f32>(-1.0,-1.0,-1.0), vec3<f32>(-1.0, 1.0, 1.0), vec3<f32>(-1.0, 1.0,-1.0),
  // +Y face
  vec3<f32>(-1.0, 1.0,-1.0), vec3<f32>(-1.0, 1.0, 1.0), vec3<f32>( 1.0, 1.0, 1.0),
  vec3<f32>(-1.0, 1.0,-1.0), vec3<f32>( 1.0, 1.0, 1.0), vec3<f32>( 1.0, 1.0,-1.0),
  // -Y face
  vec3<f32>(-1.0,-1.0,-1.0), vec3<f32>( 1.0,-1.0,-1.0), vec3<f32>( 1.0,-1.0, 1.0),
  vec3<f32>(-1.0,-1.0,-1.0), vec3<f32>( 1.0,-1.0, 1.0), vec3<f32>(-1.0,-1.0, 1.0),
  // +Z face
  vec3<f32>(-1.0,-1.0, 1.0), vec3<f32>( 1.0,-1.0, 1.0), vec3<f32>( 1.0, 1.0, 1.0),
  vec3<f32>(-1.0,-1.0, 1.0), vec3<f32>( 1.0, 1.0, 1.0), vec3<f32>(-1.0, 1.0, 1.0),
  // -Z face
  vec3<f32>(-1.0,-1.0,-1.0), vec3<f32>(-1.0, 1.0,-1.0), vec3<f32>( 1.0, 1.0,-1.0),
  vec3<f32>(-1.0,-1.0,-1.0), vec3<f32>( 1.0, 1.0,-1.0), vec3<f32>( 1.0,-1.0,-1.0),
);

// Perspective projection: a world point `p` seen through a pinhole camera at
// `(cx, cy, cameraZ)` looking down -Z, with the z=0 plane mapping to the
// screen exactly 1:1 in world pixels. A point at the screen plane projects to
// its own (x, y). A point closer to the camera than z=0 projects outward; a
// point beyond z=0 projects inward. The perspective divide is applied here in
// xy (not by the rasterizer), so the return always has w=1 for in-front
// vertices; w=-1 is used as a near-plane sentinel that the rasterizer clips.
fn projectWorld(p: vec3<f32>) -> vec4<f32> {
  let persp = frame.projection > 0.5;
  let camXY = frame.resolution * 0.5;

  var uv: vec2<f32>;
  if (persp) {
    let dz = frame.cameraZ - p.z;
    // `dz <= 0` means the vertex is at or behind the camera. Return w=-1 so
    // WebGPU's near-plane clipper drops the vertex; legitimate front-facing
    // vertices with tiny-but-positive dz still rasterize (threshold used to
    // be 1.0 which incorrectly clipped near-camera proxy corners at wide FOV).
    if (dz <= 0.0) {
      return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }
    uv = (p.xy - camXY) * (frame.cameraZ / dz) + camXY;
  } else {
    uv = p.xy;
  }
  let ndcX = 2.0 * uv.x / frame.resolution.x - 1.0;
  let ndcY = 1.0 - 2.0 * uv.y / frame.resolution.y;
  return vec4<f32>(ndcX, ndcY, 0.5, 1.0);
}

// Project a world point to screen-space pixel coordinates using the same
// pinhole model as projectWorld() but returning pixel units directly (no NDC
// flip). The z component carries a validity flag: > 0 = in front of camera,
// <= 0 = behind / at camera. Used by reprojectHit() below — the NDC return
// shape of projectWorld is awkward for "did this reprojection land inside
// the history texture?" checks.
fn worldToScreenPx(p: vec3<f32>) -> vec3<f32> {
  let camXY = frame.resolution * 0.5;
  if (frame.projection > 0.5) {
    let dz = frame.cameraZ - p.z;
    if (dz <= 0.0) {
      return vec3<f32>(0.0, 0.0, -1.0);
    }
    let px = (p.xy - camXY) * (frame.cameraZ / dz) + camXY;
    return vec3<f32>(px.x, px.y, 1.0);
  }
  return vec3<f32>(p.x, p.y, 1.0);
}

// TAA motion-vector reprojection. Given a world-space hit point on a rotating
// shape AND the unjittered fragment coordinate it came from, returns the
// screen UV where that SAME point on the shape was one frame ago — so the
// history read tracks the rotating surface instead of smearing stale
// neighbouring pixels into the output.
//
// Cube / plate both cache their previous rotation as a uniform (`cubeRotPrev`
// / `plateRotPrev`), so the math is: world → local (current rotation) → world
// (previous rotation) → screen. pill / prism don't rotate, so their hit
// point's prev screen position is just the current one — the caller passes
// in `fallbackUv` and we return it untouched (modulo the out-of-bounds check
// below, which catches disocclusion at the screen edge).
//
// Critical detail: `hitWorld` came from a JITTERED ray, so its raw screen
// projection isn't pixel-aligned. Using `projection(prevWorld) / resolution`
// directly would shift the history read by a fresh sub-pixel offset every
// frame, and bilinear filtering would compound a fractional-pixel blur into
// the history with each accumulation step (visible as "super blurry plates"
// when long-pause progressive averaging drives α toward 0).
//
// The fix below is the standard TAA technique: compute the motion DELTA in
// screen pixels (jitter cancels because it's present in both the current and
// previous projections) and add it to the unjittered FRAGCOORD. Static scenes
// then read history at exactly the pixel centre — no bilinear blur — and
// moving scenes still get the correct motion-corrected sample.
fn reprojectHit(hitWorld: vec3<f32>, fragCoord: vec2<f32>, pillIdx: u32, shapeId: i32, fallbackUv: vec2<f32>) -> vec2<f32> {
  var prevWorld = hitWorld;
  if (shapeId == 2) {
    let pill  = frame.pills[pillIdx];
    let local = frame.cubeRot * (hitWorld - pill.center);
    prevWorld = transpose(frame.cubeRotPrev) * local + pill.center;
  } else if (shapeId == 3) {
    let pill  = frame.pills[pillIdx];
    let local = frame.plateRot * (hitWorld - pill.center);
    prevWorld = transpose(frame.plateRotPrev) * local + pill.center;
  } else if (shapeId == 4) {
    // Diamond: same trick as cube/plate. The fold inside sdfDiamond is a
    // non-linear symmetry operation, but reprojection only cares about the
    // surface's rigid-body motion (rotation + translation), which `diamondRot`
    // captures fully. Reading history via `transpose(diamondRotPrev)` pulls
    // each spinning facet's own pixel from the previous frame.
    let pill  = frame.pills[pillIdx];
    let local = frame.diamondRot * (hitWorld - pill.center);
    prevWorld = transpose(frame.diamondRotPrev) * local + pill.center;
  }

  let prevPx = worldToScreenPx(prevWorld);
  let currPx = worldToScreenPx(hitWorld);
  // Behind-camera fallback: if the rotation moved the hit point through the
  // near plane between frames (rare — needs a fast tumble + perspective
  // mode), there's no meaningful history pixel to reproject from. Falling
  // back to `fallbackUv` reads stale history at the current pixel; under
  // the steady-state α = 0.2 EMA that washes out in ~5 frames. The pause
  // path can't hit this because plateRot == plateRotPrev when paused, so
  // the reprojection collapses to identity and prevPx == currPx — safe.
  if (prevPx.z <= 0.0 || currPx.z <= 0.0) { return fallbackUv; }

  // `currPx` carries the sub-pixel jitter of the original ray, and
  // `prevPx` is the SAME world point reprojected to the previous frame's
  // camera — both share that jitter, so it cancels in the delta to first
  // order. Strictly the perspective divide is non-linear in p.z, so there's
  // a residual of order (jitter × per-frame rotation × perspective factor),
  // ≤ 0.5 px × ≈ 0.005 rad ≈ < 0.01 px at typical FOVs — well below the
  // pixel grid, invisible after EMA. (Adding to the unjittered `fragCoord`
  // — not to `currPx` — is the part that actually pins the read to the
  // pixel grid for static scenes.)
  let motionPx = prevPx.xy - currPx.xy;
  let prevUv   = (fragCoord + motionPx) / frame.resolution;
  // Disocclusion: the point was outside the screen a frame ago. Same
  // fallback strategy — read stale and let EMA fade in fresh data.
  if (any(prevUv < vec2<f32>(0.0)) || any(prevUv > vec2<f32>(1.0))) {
    return fallbackUv;
  }
  return prevUv;
}

@vertex
fn vs_proxy(
  @builtin(vertex_index)   vi: u32,
  @builtin(instance_index) ii: u32,
) -> @builtin(position) vec4<f32> {
  if (ii >= u32(frame.pillCount)) {
    // Over pillCount → degenerate position so the triangle is clipped.
    return vec4<f32>(2.0, 2.0, 0.5, 1.0);
  }
  let pill    = frame.pills[ii];
  let shapeId = i32(frame.shape + 0.5);

  // Per-shape vertex budget: CUBE_PROXY_VERT_COUNT for cube/pill/prism/plate
  // (the CUBE_VERTS array size above), DIAMOND_PROXY_VERT_COUNT for diamond.
  // Both constants are injected from src/math/diamond.ts so the draw call
  // (pipeline.ts), this guard, and the CUBE_VERTS array literal all track
  // the same TS numbers. A Phase B mesh change updates the TS constants
  // (or, for diamond, the mesh body in diamond.wgsl) — the guard and draw
  // count follow automatically. The guard prevents CUBE_VERTS out-of-bounds
  // access for non-diamond shapes when vi ≥ CUBE_PROXY_VERT_COUNT; past
  // DIAMOND_PROXY_VERT_COUNT the diamond branch has no valid vertex either.
  let maxVerts = select(CUBE_PROXY_VERT_COUNT, DIAMOND_PROXY_VERT_COUNT, shapeId == 4);
  if (vi >= maxVerts) {
    return vec4<f32>(2.0, 2.0, 0.5, 1.0);
  }

  // Unit cube corner in [-1, 1]^3 → local box sized to the pill's halfSize
  // (plus edgeR for the rounded rim so the proxy always fully covers the
  // actual shape).
  var corner: vec3<f32>;
  if (shapeId == 3) {
    // Plate: box sized to the square face + wave-amplitude margin on the Z
    // (thickness) axis so the rippling midsurface never pokes through. Then
    // apply the plate's current rotation so the proxy tracks the tumble —
    // same tight-bounding trick as the cube path below.
    let extent = vec3<f32>(pill.halfSize.x,
                           pill.halfSize.x,
                           pill.halfSize.z + frame.waveAmp);
    corner = transpose(frame.plateRot) * (CUBE_VERTS[vi] * extent);
  } else if (shapeId == 4) {
    // Diamond: exact convex-hull proxy mesh (DIAMOND_PROXY_VERT_COUNT
    // vertices = 46 triangles — see diamondProxyVertex in diamond.wgsl
    // for the topology breakdown). The split keeps the geometry details
    // next to sdfDiamond where future diamond-only trace work can land.
    //
    // vi < DIAMOND_PROXY_VERT_COUNT is guaranteed by the maxVerts guard
    // at the top of vs_proxy.
    let local = diamondProxyVertex(vi, frame.diamondSize);
    corner    = transpose(frame.diamondRot) * local;
  } else {
    let extent = pill.halfSize + vec3<f32>(pill.edgeR);
    corner     = CUBE_VERTS[vi] * extent;
    if (shapeId == 2) {
      // The shader defines the cube via `local = rot * (p - center)`, so a
      // world-space proxy corner that maps to the unit-cube local-space
      // corner `c` is `center + transpose(rot) * (c * extent)`.
      corner = transpose(frame.cubeRot) * corner;
    }
  }
  return projectWorld(pill.center + corner);
}

// ---------- fragment ----------

struct FsOut {
  @location(0) color:   vec4<f32>,
  @location(1) history: vec4<f32>,
};

// Cheap background pass: sample photo, blend history, done. No sphere-trace,
// no refraction, no per-wavelength loop. Runs for the whole screen; the proxy
// pass then overrides covered pixels with the heavy shader's output.
@fragment
fn fs_bg(@builtin(position) fragCoord: vec4<f32>) -> FsOut {
  let uv    = fragCoord.xy / frame.resolution;
  let bg    = textureSampleLevel(photoTex, photoSmp, coverUv(uv), 0.0).rgb;
  let prev  = textureSampleLevel(historyTex, historySmp, uv, 0.0).rgb;
  let blend = mix(prev, bg, frame.historyBlend);
  var o: FsOut;
  o.color   = vec4<f32>(blend, 1.0);
  o.history = vec4<f32>(blend, 1.0);
  return o;
}

// Karis-2014-style single-tap variance clamp + EMA blend. When the new
// sample disagrees with history (large `diff`) the effective alpha is
// pushed toward 1.0 so the new sample dominates — kills motion ghost on
// rotating refractive shapes. Gated by historyBlend so paused-scene
// progressive averaging (`α = max(1/n, 1/256)` → 0) stays unfaded.
//
// Used by both the bg-fallback path (for hit↔miss transitions at
// silhouettes) and the hit-path EMA at the end of fs_main.
fn adaptiveBlend(prev: vec3<f32>, next: vec3<f32>) -> vec3<f32> {
  let diff  = length(next - prev);
  let gate  = smoothstep(0.05, 0.15, frame.historyBlend);
  let mixT  = gate * smoothstep(0.05, 0.30, diff);
  let alpha = mix(frame.historyBlend, 1.0, mixT);
  return mix(prev, next, alpha);
}

// Back-face exit dispatcher. Centralises the entry-bias trick (push the
// front-hit point one MIN_STEP inward along the refracted ray so the
// analytic exits' slab math doesn't divide 0/0 → silent wrong-axis pick →
// edge speckle) plus the cube/plate/other dispatch — both used by the
// hero-mode shared trace and the per-wavelength loop.
fn backExit(hitP: vec3<f32>, r1: vec3<f32>, shapeId: i32, pillIdx: u32, internalMax: f32) -> CubeExit {
  let roEntry = hitP + r1 * MIN_STEP;
  if (shapeId == 2) { return cubeAnalyticExit(roEntry, r1, pillIdx); }
  if (shapeId == 3) { return plateAnalyticExit(roEntry, r1, pillIdx); }
  let pExit = insideTrace(hitP, r1, internalMax);
  return CubeExit(pExit, -sceneNormal(pExit));
}

@fragment
fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> FsOut {
  // DOM-top-origin pixel coords so they match pointer events and defaultPills.
  let px = fragCoord.xy;
  let uv = px / frame.resolution;

  // Temporal antialiasing: jitter the ray's sub-pixel position by a
  // per-frame-decorrelated vector in [-0.5, 0.5)², and let the history EMA
  // average the shifted samples. The fragment still writes to the
  // unjittered fragCoord, so history reads/writes use `uv` as-is — only
  // ray generation and sampling-along-the-ray see the jittered position.
  // Handles the shape silhouette (decided inside this shader by
  // sphere-trace hit/miss, which MSAA can't touch) AND smooths any
  // wave/refraction texture aliasing for free. Steady-state α = 0.2 =>
  // full convergence in ~5 frames; imperceptible ghost for slow tumble,
  // only noticeable on fast drags.
  var rayPx = px;
  if (frame.taaEnabled > 0.5) {
    let taaJit = vec2<f32>(
      hash21(px + vec2<f32>(frame.time * 7.19, 3.141)) - 0.5,
      hash21(px + vec2<f32>(frame.time * 11.23, 6.283)) - 0.5,
    );
    rayPx = px + taaJit;
  }

  // Ortho: parallel rays going -Z, origin just above the scene volume.
  // Perspective: rays diverge from the camera through each screen pixel,
  // hitting the z=0 screen plane exactly at (rayPx.x, rayPx.y, 0). FOV is
  // encoded in cameraZ (smaller = wider FOV).
  var ro: vec3<f32>;
  var rd: vec3<f32>;
  if (frame.projection > 0.5) {
    ro = vec3<f32>(frame.resolution * 0.5, frame.cameraZ);
    rd = normalize(vec3<f32>(rayPx, 0.0) - ro);
  } else {
    ro = vec3<f32>(rayPx, 400.0);
    rd = vec3<f32>(0.0, 0.0, -1.0);
  }
  // One upper bound that works for both projections: a ray can't travel
  // further than the camera-to-far-plane distance in the scene.
  let maxT = 2.0 * max(frame.cameraZ, 400.0) + 400.0;
  let h    = sphereTrace(ro, rd, maxT);
  let bg   = textureSampleLevel(photoTex, photoSmp, coverUv(uv), 0.0).rgb;

  // Helper for the "render as background" path — used for miss, for hits
  // whose front normal came out degenerate, and for the no-data fallback at
  // the end of the hit path. Wraps the bg sample in `adaptiveBlend` so a
  // pixel transitioning from hit-path refraction to bg also gets the same
  // variance clamp the hit path uses (without it, 20% of stale refraction
  // colour bleeds into each frame's blend → visible silhouette flicker).
  // `frame.debugProxy` tints these pixels pink so the proxy over-coverage
  // halo is visible in the UI's "Show proxy" mode.
  // (kept inline rather than a helper because it returns from fs_main; WGSL
  // can't early-return from a callee.)

  let shapeId   = i32(frame.shape + 0.5);
  let isCube    = shapeId == 2;
  let isPlate   = shapeId == 3;
  let isDiamond = shapeId == 4;
  // `hasAnalyticExit` gates back-exit dispatch (cube/plate only — diamond
  // reuses the generic `insideTrace` per Phase A scope). `hasMotionPivot`
  // gates the TAA reprojection call below: a shape needs a per-frame
  // rotation uniform (cubeRot / plateRot / diamondRot) for reprojection to
  // make sense, which includes diamond even though diamond has no analytic
  // back-exit. Keeping the two names distinct stops future refactors from
  // accidentally coupling "has analytic exit" and "has motion pivot".
  let hasAnalyticExit = isCube || isPlate;
  let hasMotionPivot  = isCube || isPlate || isDiamond;

  // Bg-fallback short-circuit. Run BEFORE `sceneNormal` because WGSL `select`
  // evaluates both arms — wrapping `sceneNormal(h.p)` in select(_, _, h.ok)
  // would still cost 6 SDF evals per miss pixel, and the proxy mesh
  // intentionally over-covers the silhouette so misses are common.
  if (!h.ok) {
    var bgFinal = bg;
    if (frame.debugProxy > 0.5) {
      bgFinal = mix(bg, vec3<f32>(1.0, 0.3, 0.7), 0.5);
    }
    let prevMiss = textureSampleLevel(historyTex, historySmp, uv, 0.0).rgb;
    let blend    = adaptiveBlend(prevMiss, bgFinal);
    var bgOut: FsOut;
    bgOut.color   = vec4<f32>(blend, 1.0);
    bgOut.history = vec4<f32>(blend, 1.0);
    return bgOut;
  }

  // Hit path. Pill-index scan is per-shape; gated on h.ok above so misses
  // skip it entirely. Diamond still participates — not for analytic back-exit
  // (that's Phase B) but because `reprojectHit` needs the right pill center
  // to compute motion vectors for multi-instance scenes.
  var analyticIdx: u32 = 0u;
  if      (isCube)    { analyticIdx = hitCubePillIdx(h.p); }
  else if (isPlate)   { analyticIdx = hitPlatePillIdx(h.p); }
  else if (isDiamond) { analyticIdx = hitDiamondPillIdx(h.p); }

  // `sceneNormal` returns the zero vector when the local gradient is too
  // small to normalise (silhouette / wave-crest singularity). Falling back
  // to an arbitrary normal would produce visible-but-wrong refraction
  // colours; routing those pixels through the bg path instead lets them
  // blend cleanly with the silhouette neighbourhood.
  let nFront = sceneNormal(h.p);
  if (dot(nFront, nFront) <= 0.5) {
    var bgFinal = bg;
    if (frame.debugProxy > 0.5) {
      bgFinal = mix(bg, vec3<f32>(1.0, 0.3, 0.7), 0.5);
    }
    let prevMiss = textureSampleLevel(historyTex, historySmp, uv, 0.0).rgb;
    let blend    = adaptiveBlend(prevMiss, bgFinal);
    var bgOut: FsOut;
    bgOut.color   = vec4<f32>(blend, 1.0);
    bgOut.history = vec4<f32>(blend, 1.0);
    return bgOut;
  }

  let n_d      = frame.n_d;
  let V_d      = frame.V_d;
  let N        = clamp(i32(frame.sampleCount), 1, MAX_N);
  let strength = frame.refractionStrength;
  let jitter   = frame.jitter;
  let useHero  = frame.refractionMode > 0.5;

  // Cap the inside-trace at the longest possible chord through the largest
  // pill in the scene. Thick configurations would otherwise bail out mid-body.
  let internalMax = maxInternalPath();

  // Hero mode: one back-face trace at a PER-FRAME-RANDOMIZED wavelength
  // (Wilkie 2014). Unlike plain "approx" fixed at 540 nm, the randomization +
  // temporal history accumulation averages out the single-trace error across
  // ~5 frames — so 4–8 samples in Hero mode look like 16–32 samples in Exact.
  var sharedExit  = h.p;
  var sharedNBack = -nFront;
  if (useHero) {
    let iorHero = cauchyIor(frame.heroLambda, n_d, V_d);
    let r1hero  = refract(rd, nFront, 1.0 / iorHero);
    // Entry TIR — leave sharedExit / sharedNBack at their defaults
    // (h.p, -nFront) so the wavelength loop falls through to the reflection
    // path via Fresnel. Practically this can't fire today (cauchyIor clamps
    // ior >= 1.0 and the entry is vacuum→glass), but both back-trace branches
    // would misbehave with r1hero ≈ 0: cube/plateAnalyticExit divide by rdL
    // and would emit NaN, insideTrace would stall with no forward progress.
    if (dot(r1hero, r1hero) >= 1e-4) {
      let ex = backExit(h.p, r1hero, shapeId, analyticIdx, internalMax);
      sharedExit  = ex.pWorld;
      sharedNBack = ex.nBack;
    }
  }

  // External front-face reflection — used BOTH as TIR fallback and as the
  // per-wavelength reflection color (mixed via per-λ Fresnel below).
  // Falls back to the local bg when the reflected UV lands outside the
  // photo (the sampler is mirror-repeat — see photo.ts — and an OOB
  // reflUv would mirror back to a visually-unrelated photo region,
  // matching the UV-OOB symptom we already guard against on the
  // refraction sample below). Bg is the same colour the miss-path
  // neighbours render, so silhouette TIR pixels blend cleanly.
  // Refraction / reflection UV base: the pixel's own screen UV. Adding
  // the DEVIATION `(out_ray - rd).xy` (rather than `out_ray.xy`) makes
  // non-refracting configurations collapse to the miss-path sample —
  // critical for perspective, where `rd.xy != 0` away from the optical
  // axis so `out_ray.xy * strength` gave a spurious offset even for a
  // plate facing straight at the camera (visible as a slightly shrunk
  // bg inside the glass). `(refl - rd).xy` / `(r2 - rd).xy` isolate the
  // optical bend itself; refraction strength now scales only the bend.
  let refl       = reflect(rd, nFront);
  let reflUv     = uv + (refl - rd).xy * 0.2;
  let reflCover  = coverUv(reflUv);
  let reflInBnds = all(reflCover >= vec2<f32>(0.0)) && all(reflCover <= vec2<f32>(1.0));
  let reflRaw    = textureSampleLevel(photoTex, photoSmp, reflCover, 0.0).rgb;
  let reflSrc    = select(bg, reflRaw, reflInBnds) * vec3<f32>(0.85, 0.9, 1.0);

  // Front-face cosine (angle of incidence). Identical for every wavelength at
  // the front face, so compute once.
  let cosT = max(dot(-rd, nFront), 0.0);

  // Texture-footprint LOD for the per-wavelength photo sample below. Two
  // regimes cause the refracted-UV Jacobian w.r.t. screen pixels to spike,
  // each driving a separate term:
  //
  //   (A) Grazing incidence — d(uv)/d(px) scales as ~1/cosT. Captured by
  //       `-log2(cosT)`. The -1.0 bias keeps level 0 active for cosT >= 0.5
  //       so normal viewing stays sharp.
  //
  //   (B) Rounded-edge normal turn — on the cube/plate's rounded rim the
  //       surface normal rotates ~90° across the rim's ~edgeR-wide footprint
  //       in screen space. Per-pixel δn there is large, and refract's
  //       Jacobian amplifies that into a big δuv even at moderate cosT.
  //       Detect by transforming nFront into the shape's local frame (where
  //       flat faces align with one axis) and measuring tilt via the
  //       complement of the largest |axis component|: flat face → 0, rim
  //       edge → ~0.29, corner → ~0.42. Multiply by 8 for a ~2.3 LOD boost
  //       at mid-rim — enough to suppress the sparkle while leaving flat
  //       faces untouched. Plate's wavy ±Z faces also pick up a small tilt
  //       that softens their own mild refraction aliasing.
  //
  // Pill / prism don't participate in (B) — their silhouettes are either
  // flat faces or continuous quadric curves that (A) already handles.
  //
  // Clamp 6.0 caps the LOD at roughly mip-6 — ~30 px wide on a 1920x1080
  // photo, ~4 px on the 256² gradient fallback. Further blurring stops
  // adding AA benefit and just washes the refraction color to an average.
  var curvatureTilt: f32 = 0.0;
  if (isCube) {
    let nLocal    = frame.cubeRot * nFront;
    curvatureTilt = 1.0 - max(max(abs(nLocal.x), abs(nLocal.y)), abs(nLocal.z));
  } else if (isPlate) {
    let nLocal    = frame.plateRot * nFront;
    curvatureTilt = 1.0 - max(max(abs(nLocal.x), abs(nLocal.y)), abs(nLocal.z));
  }
  let photoLod = clamp(-log2(max(cosT, 0.02)) - 1.0 + curvatureTilt * 8.0, 0.0, 6.0);

  // Per-pixel stratified jitter: each pixel gets its own wavelength phase so
  // adjacent pixels sample DIFFERENT λ. The eye (and post-process history
  // accumulation) averages the spatial noise, so the rainbow looks smooth at
  // a given N — effectively 2-4× more samples worth of quality for free.
  // `jitter` is a per-frame random offset (host-side Math.random()/N); using
  // it as part of the hash seed decorrelates the stratum choice across
  // frames so history accumulates new samples instead of locking on.
  //
  // Shift hash output from [0,1) to [-0.5, 0.5) so the jitter is SIGNED around
  // the stratum center. With t = (i + 0.5 + pxJit)/N that keeps every sample
  // inside its own stratum [i/N, (i+1)/N) — critically, the i=N-1 sample
  // cannot spill past t=1 into invisible λ > 700nm. Without this shift, at
  // N=3 about a third of pixels lose their red-end sample (CMF≈0 at 720-753
  // nm), rgbWeight.b drops to zero there, and the renormaliser flips a white
  // background pixel to yellow.
  let pxJit = hash21(px + vec2<f32>(jitter * 1000.0, frame.time * 37.0)) - 0.5;

  // For each wavelength λ:
  //   1. compute per-λ IOR via Cauchy
  //   2. refract into the glass
  //   3. find the back-face exit point + inward normal (skipped in Approx
  //      mode — reuses sharedExit/sharedNBack):
  //        cube  → cubeAnalyticExit  (O(1) slab + rounded-box gradient)
  //        plate → plateAnalyticExit (O(1) slab + Newton-refined wavy Z face)
  //        other → insideTrace + sceneNormal finite differences
  //   4. refract out
  //   5. sample photo at the exit UV; TIR → reflection color instead
  //   6. PER-WAVELENGTH Fresnel mix between refract and reflect (blue λ has
  //      higher IOR → higher Fresnel → more reflective at rim, produces the
  //      visible blue-tinged rim we see in real prisms and diamonds)
  //   7. weight by xyzToSrgb(cmf(λ)) and accumulate
  var rgbAccum  = vec3<f32>(0.0);
  var rgbWeight = vec3<f32>(0.0);

  for (var i: i32 = 0; i < N; i = i + 1) {
    let t      = (f32(i) + 0.5 + pxJit) / f32(N);
    let lambda = mix(380.0, 700.0, t);
    let ior    = cauchyIor(lambda, n_d, V_d);
    let r1     = refract(rd, nFront, 1.0 / ior);
    if (dot(r1, r1) < 1e-4) { continue; }  // TIR on entry (shouldn't fire for vacuum→denser)

    var pExit = sharedExit;
    var nBack = sharedNBack;
    if (!useHero) {
      let ex = backExit(h.p, r1, shapeId, analyticIdx, internalMax);
      pExit = ex.pWorld;
      nBack = ex.nBack;
    }
    let r2 = refract(r1, nBack, ior);

    var refractL: vec3<f32>;
    // Bad-r2 gate covers three failure modes that all produce single-pixel
    // speckles along plate edges if left to fall through:
    //   1. Real TIR — refract returned (0,0,0) because k < 0. Standard.
    //   2. NaN r2 — happens when nBack came out NaN-ish from a corner-case
    //      back exit. WGSL's `<` against NaN is always false, so we add an
    //      explicit `r2dot != r2dot` self-comparison to catch it.
    //   3. The refracted UV would land outside the photo's [0, 1] range.
    //      The sampler is mirror-repeat (see photo.ts), so a wildly-off
    //      UV folds back to an unrelated photo region: bright photo
    //      content there → white speckle, dark there → black.
    //
    // For #2 and #3 we substitute the LOCAL bg sample (the photo at this
    // fragment's own pixel position) — that's the same colour the miss
    // path / silhouette neighbours render, so the bad pixel blends into
    // the surrounding silhouette instead of sticking out as either a
    // bright reflection sample or a wrong-photo-region speckle. For real
    // TIR (#1) we still use the external reflection because it's the
    // physically-correct response, even if it can be visually noisy.
    let uvOff       = uv + (r2 - rd).xy * strength;
    let uvCover     = coverUv(uvOff);
    let uvInBounds  = all(uvCover >= vec2<f32>(0.0)) && all(uvCover <= vec2<f32>(1.0));
    let r2dot       = dot(r2, r2);
    let r2NaN       = r2dot != r2dot;
    let r2TIR       = r2dot < 1e-4 && !r2NaN;
    let r2OOB       = !uvInBounds;
    if (r2TIR) {
      refractL = reflSrc;
    } else if (r2NaN || r2OOB) {
      refractL = bg;
    } else {
      refractL = textureSampleLevel(photoTex, photoSmp, uvCover, photoLod).rgb;
    }

    // Per-wavelength Schlick Fresnel: short λ (blue) has higher IOR → higher F.
    let F_lambda = schlickFresnel(cosT, ior);
    let L        = mix(refractL, reflSrc, F_lambda);

    let lambdaRgb = max(xyzToSrgb(cieXyz(lambda)), vec3<f32>(0.0));
    rgbAccum  = rgbAccum  + L * lambdaRgb;
    rgbWeight = rgbWeight + lambdaRgb;
  }

  // Normalize against the per-wavelength primary sum — keeps a flat white
  // spectrum neutral for any N. Then sanitize: if every wavelength entered
  // the loop's `continue` branch (TIR-on-entry — shouldn't happen with the
  // 1.0-clamped IOR but defensively guarded), `rgbWeight` is exactly 0 and
  // `max(_, 1e-4)` saves the divide but the result can spike to absurdly
  // large values from tiny `rgbAccum` jitter; downstream that becomes a
  // saturated white pixel. NaN can also slip in from a borderline-
  // degenerate normal that passed sceneNormal's gradient threshold, or
  // from a Newton bail in plateAnalyticExit that left `pL` outside the
  // plate's XY extent and made nBack point in a near-tangent direction
  // that refracts to a NaN r2.
  //
  // We replace bad outputs with the BG photo sample, not zero — these
  // failure modes happen at single pixels along silhouettes, and
  // surrounding pixels render as either bg (miss path) or normal
  // refraction. Substituting the local bg lets the dot vanish into the
  // silhouette neighbourhood; substituting black would leave a visible
  // dark speckle. (Without the substitution, NaN burns into history and
  // becomes permanent garbage at that pixel until a scene-change reset.)
  let raw     = rgbAccum / max(rgbWeight, vec3<f32>(1e-4));
  // NaN-only whole-pixel guard. The `any(...)` reduces vec3<bool> to a
  // scalar so a single bad channel sends the entire pixel to bg (otherwise
  // component-wise select would replace e.g. just R, leaving G/B as the
  // raw NaN-y values → off-coloured speckle worse than a clean fallback).
  // Magnitude is bounded by the clamp on the next line.
  let safe    = select(raw, bg, vec3<bool>(any(raw != raw)));
  let clamped = clamp(safe, vec3<f32>(0.0), vec3<f32>(8.0));

  // Silhouette anti-alias by blending toward bg at near-grazing angles.
  // Without this, the rasterised proxy edge produces a 1-pixel binary
  // hit/miss boundary; the hit side has wildly view-dependent refraction
  // (the per-wavelength loop's UV samples are extreme at cosT → 0 and
  // even legitimate TIR fallbacks to reflSrc can be uncorrelated with
  // the miss side's plain bg). With TAA off we have no temporal averaging
  // to dilute the discontinuity, so the silhouette flickers as a hard
  // pixel-wide colour jump.
  //
  // Threshold 0.05 is deliberately tight: cosT < 0.05 means incidence
  // angle > ~87°, the regime where refraction is genuinely meaningless
  // (Fresnel ≈ 1, the proxy mesh is in pixel-thin coverage of the
  // actual silhouette). A wider band (we tried 0.15 ≈ 81°) dimmed
  // legitimate interior pixels of plates tumbling steeply edge-on,
  // where the entire face is at a grazing-but-fully-hit angle and
  // the rim Fresnel reflection is the visually-correct answer.
  //
  // Gated by historyBlend so paused-scene progressive averaging
  // (`α = max(1/n, 1/256)` → 0) sees `silhouetteMix = 1` and stays on
  // the unfaded refraction. Otherwise the bg-tinted output would
  // accumulate into the converged paused mean and pull silhouettes
  // toward bg over time — the same failure mode the variance clamp
  // is also gated against.
  let silhouetteGate = smoothstep(0.05, 0.15, frame.historyBlend);
  let silhouetteMix  = mix(1.0, smoothstep(0.0, 0.05, cosT), silhouetteGate);
  let outRgb = mix(bg, clamped, silhouetteMix);

  // History is stored in rgba16float (linear). Blend in linear space; encode
  // for display only on the swapchain write. `historyBlend` is normally 0.2 —
  // host bumps it to 1.0 for one frame on any scene change (photo reload,
  // preset click, shape switch) so the previous scene doesn't ghost in.
  //
  // TAA motion-vector reprojection: for rotating shapes (cube / plate /
  // diamond), read history at the screen location where the hit point was
  // one frame ago. This is what turns naive TAA (which blurs tumbling
  // refraction texture into mush) into real TAA that keeps refracted detail
  // sharp under motion. Pill / prism pass through unchanged (reprojectHit
  // returns fallbackUv for non-rotating shapes, and we short-circuit when
  // TAA is off).
  var historyUv = uv;
  if (frame.taaEnabled > 0.5 && hasMotionPivot) {
    // Pass the unjittered `px` so the reprojection cancels jitter in the
    // motion delta, keeping static-scene history reads pixel-aligned.
    historyUv = reprojectHit(h.p, px, analyticIdx, shapeId, uv);
  }
  let prev  = textureSampleLevel(historyTex, historySmp, historyUv, 0.0).rgb;

  // EMA blend with adaptive variance clamp — see `adaptiveBlend` above.
  var blend = adaptiveBlend(prev, outRgb);

  // Debug: tint every proxy fragment pink so the proxy silhouette is visible.
  // This path (ray hit the cube) gets a light tint; the miss path above gets
  // a heavier tint so the over-coverage halo stands out.
  if (frame.debugProxy > 0.5) {
    blend = mix(blend, vec3<f32>(1.0, 0.3, 0.7), 0.2);
  }

  // Diamond debug overlays — helpful for cross-checking the cut geometry
  // against a real brilliant-cut reference. Both write to the DISPLAY
  // output only, not the history texture, so they don't accumulate into
  // TAA / history EMA and muddy themselves on subsequent frames.
  //
  //   diamondFacetColor: flat-shade each facet class with a distinct
  //     colour so adjacency + coverage are visible without refraction
  //     confusing the signal (disable refraction alongside for the
  //     cleanest view).
  //   diamondWireframe:  overlay the facet edges on top. Uses the
  //     plane-gap trick from `sdfDiamondEdgeWeight` — two plane SDFs
  //     almost equal → facet boundary.
  var display = blend;
  if (isDiamond && frame.diamondFacetColor > 0.5) {
    let pillCenter = frame.pills[analyticIdx].center;
    display = sdfDiamondFacetColor(h.p - pillCenter, frame.diamondSize);
  }
  if (isDiamond && frame.diamondWireframe > 0.5) {
    let pillCenter = frame.pills[analyticIdx].center;
    let edgeW      = sdfDiamondEdgeWeight(h.p - pillCenter, frame.diamondSize);
    display = mix(display, vec3<f32>(1.0, 0.25, 0.25), edgeW * 0.85);
  }

  var o: FsOut;
  o.color   = vec4<f32>(display, 1.0);
  o.history = vec4<f32>(blend, 1.0);
  return o;
}
