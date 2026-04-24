// ---------- SDF ----------

const L4_VISUAL_RADIUS_SCALE: f32 = 1.84089642;

fn superellipseLength2(v: vec2<f32>) -> f32 {
  let v2 = v * v;
  return sqrt(sqrt(dot(v2, v2)));
}

fn superellipsoidLength3(v: vec3<f32>) -> f32 {
  let v2 = v * v;
  return sqrt(sqrt(dot(v2, v2)));
}

fn visualRoundRadius(edgeR: f32, limit: f32) -> f32 {
  if (frame.smoothCurvature > 0.5) {
    return min(edgeR * L4_VISUAL_RADIUS_SCALE, limit);
  }
  return min(edgeR, limit);
}

fn roundedLength2(v: vec2<f32>) -> f32 {
  if (frame.smoothCurvature > 0.5) {
    return superellipseLength2(v);
  }
  return length(v);
}

fn roundedLength3(v: vec3<f32>) -> f32 {
  if (frame.smoothCurvature > 0.5) {
    return superellipsoidLength3(v);
  }
  return length(v);
}

fn sdfPill(p: vec3<f32>, halfSize: vec3<f32>, edgeR: f32) -> f32 {
  let xyR  = min(edgeR, min(halfSize.x, halfSize.y));
  let zR   = visualRoundRadius(edgeR, halfSize.z);
  let qXY  = abs(p.xy) - (halfSize.xy - vec2<f32>(xyR));
  // Keep the front silhouette a true circular capsule. Smooth curvature only
  // affects the Z roundover, so toggling it changes refraction joins without
  // turning the pill outline into a squircle.
  let dXy  = length(max(qXY, vec2<f32>(0.0))) + min(max(qXY.x, qXY.y), 0.0) - xyR;
  let w    = vec2<f32>(dXy + zR, abs(p.z) - halfSize.z + zR);
  return roundedLength2(max(w, vec2<f32>(0.0))) + min(max(w.x, w.y), 0.0) - zR;
}

// Unnormalised ∇(sdfPill) via central differences — for Newton refinement and
// single-pill surface normals (avoids scanning all instances in `sceneSdf`).
fn sdfPillGrad(p: vec3<f32>, halfSize: vec3<f32>, edgeR: f32) -> vec3<f32> {
  let e = HIT_EPS;
  let s = 0.5 / e;
  return vec3<f32>(
    (sdfPill(p + vec3<f32>(e, 0.0, 0.0), halfSize, edgeR) - sdfPill(p - vec3<f32>(e, 0.0, 0.0), halfSize, edgeR)) * s,
    (sdfPill(p + vec3<f32>(0.0, e, 0.0), halfSize, edgeR) - sdfPill(p - vec3<f32>(0.0, e, 0.0), halfSize, edgeR)) * s,
    (sdfPill(p + vec3<f32>(0.0, 0.0, e), halfSize, edgeR) - sdfPill(p - vec3<f32>(0.0, 0.0, e), halfSize, edgeR)) * s,
  );
}

// Rounded box / cuboid. The rim uses an L4 superellipsoid/squircle norm so
// face-to-rim curvature eases in from zero instead of stepping from flat to
// circular. Equal halfSize = cube.
fn sdfCube(p: vec3<f32>, halfSize: vec3<f32>, edgeR: f32) -> f32 {
  let r = visualRoundRadius(edgeR, min(halfSize.x, min(halfSize.y, halfSize.z)));
  let q = abs(p) - halfSize + vec3<f32>(r);
  return roundedLength3(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

fn sdfCubeGrad(p: vec3<f32>, halfSize: vec3<f32>, edgeR: f32) -> vec3<f32> {
  let r = visualRoundRadius(edgeR, min(halfSize.x, min(halfSize.y, halfSize.z)));
  let q = abs(p) - halfSize + vec3<f32>(r);
  let a = max(q, vec3<f32>(0.0));
  let s = select(vec3<f32>(-1.0), vec3<f32>(1.0), p >= vec3<f32>(0.0));
  if (frame.smoothCurvature > 0.5) {
    let l = superellipsoidLength3(a);
    if (l > 1e-6) {
      return s * (a * a * a) / (l * l * l);
    }
  } else {
    let l = length(a);
    if (l > 1e-6) {
      return s * a / l;
    }
  }
  if (q.x >= q.y && q.x >= q.z) {
    return vec3<f32>(s.x, 0.0, 0.0);
  }
  if (q.y >= q.z) {
    return vec3<f32>(0.0, s.y, 0.0);
  }
  return vec3<f32>(0.0, 0.0, s.z);
}

// Isosceles triangle in YZ (apex +Z, base -Z), extruded along X. Half-sizes
// match sdfPill: halfSize.x is extrusion length, halfSize.y the triangle base
// half-width, halfSize.z the apex height. Sharp edges (no rim radius) so the
// proxy AABB `±halfSize` tightly bounds the solid.
fn sdfPrism(p: vec3<f32>, halfSize: vec3<f32>) -> f32 {
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
  return length(max(w, vec2<f32>(0.0))) + min(max(w.x, w.y), 0.0);
}

fn sdfPrismGrad(p: vec3<f32>, halfSize: vec3<f32>) -> vec3<f32> {
  let e = HIT_EPS;
  let s = 0.5 / e;
  return vec3<f32>(
    (sdfPrism(p + vec3<f32>(e, 0.0, 0.0), halfSize) - sdfPrism(p - vec3<f32>(e, 0.0, 0.0), halfSize)) * s,
    (sdfPrism(p + vec3<f32>(0.0, e, 0.0), halfSize) - sdfPrism(p - vec3<f32>(0.0, e, 0.0), halfSize)) * s,
    (sdfPrism(p + vec3<f32>(0.0, 0.0, e), halfSize) - sdfPrism(p - vec3<f32>(0.0, 0.0, e), halfSize)) * s,
  );
}

// Diamond (round brilliant cut) SDF + proxy-mesh helpers live in a separate
// shader unit: src/shaders/diamond.wgsl. It's concatenated at pipeline build
// time so sdfDiamond / diamondAnalyticExit / diamondAnalyticHitScene /
// diamondProxyVertex are visible to the dispatch sites below (sceneSdf,
// fs_main, fs_main_diamond, vs_proxy). The split keeps the dispersion/
// shader bundle focused on trace + SDF framework and leaves Phase B's
// multi-bounce TIR trace for diamond a clear home.

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
