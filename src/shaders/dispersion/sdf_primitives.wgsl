// ---------- SDF ----------

fn sdfPill(p: vec3<f32>, halfSize: vec3<f32>, edgeR: f32) -> f32 {
  let hsXY = halfSize.xy - vec2<f32>(edgeR);
  let rXY  = min(hsXY.x, hsXY.y);
  let qXY  = abs(p.xy) - hsXY + vec2<f32>(rXY);
  let dXy  = length(max(qXY, vec2<f32>(0.0))) + min(max(qXY.x, qXY.y), 0.0) - rXY;
  let w    = vec2<f32>(dXy, abs(p.z) - halfSize.z + edgeR);
  return length(max(w, vec2<f32>(0.0))) + min(max(w.x, w.y), 0.0) - edgeR;
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

// Rounded box / cuboid. Equal halfSize = cube.
fn sdfCube(p: vec3<f32>, halfSize: vec3<f32>, edgeR: f32) -> f32 {
  let q = abs(p) - halfSize + vec3<f32>(edgeR);
  return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - edgeR;
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
// time so sdfDiamond / hitDiamondPillIdx / diamondProxyVertex are visible to
// the dispatch sites below (sceneSdf, fs_main, reprojectHit, vs_proxy). The
// split keeps the dispersion/ shader bundle focused on trace + SDF framework and leaves
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
