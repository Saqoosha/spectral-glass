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
  applySrgbOetf:      f32,  // 1.0 if canvas is non-sRGB and we must encode; 0.0 if -srgb
  shape:              f32,  // 0 = pill (stadium), 1 = prism, 2 = cube (rotates)
  time:               f32,  // seconds since start (used for cube rotation)
  historyBlend:       f32,  // 0.2 steady state, 1.0 when the scene changed this frame
  heroLambda:         f32,  // jittered each frame in [380,700]; Hero mode uses this
  cameraZ:            f32,  // distance from screen plane (z=0) to camera, in pixels
  projection:         f32,  // 0 = orthographic, 1 = perspective
  debugProxy:         f32,  // 1 = tint every proxy fragment pink (debug view)
  _pad0:              f32,
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

// World-space is top-origin pixels (matches DOM pointer coords and defaultPills).
fn screenUvFromWorld(px: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(px.x / frame.resolution.x, px.y / frame.resolution.y);
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

// Compose rotations around X and Z. Slow tumble — cube faces tip in and out
// of the viewing direction, which modulates refraction angles per frame.
fn cubeRotation(t: f32) -> mat3x3<f32> {
  let ax = t * 0.31;
  let az = t * 0.20;
  let cx = cos(ax); let sx = sin(ax);
  let cz = cos(az); let sz = sin(az);
  // WGSL mat3x3 literals fill columns. Each row of literals below is one column.
  let rx = mat3x3<f32>(
    1.0, 0.0, 0.0,
    0.0,  cx,  sx,
    0.0, -sx,  cx,
  );
  let rz = mat3x3<f32>(
     cz,  sz, 0.0,
    -sz,  cz, 0.0,
    0.0, 0.0, 1.0,
  );
  return rz * rx;
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

fn sceneSdf(p: vec3<f32>) -> f32 {
  let count   = min(u32(frame.pillCount), MAX_PILLS);
  let shapeId = i32(frame.shape + 0.5);
  var d: f32 = 1e9;
  for (var i: u32 = 0u; i < count; i = i + 1u) {
    let pill  = frame.pills[i];
    let local = p - pill.center;
    var pd: f32;
    if (shapeId == 2) {
      // Cube is rotated in local space before SDF evaluation. `frame.time` is
      // constant per fragment so the rotation matrix is consistent across all
      // sceneSdf calls within one pixel (sphere trace + normal finite diffs).
      pd = sdfCube(cubeRotation(frame.time) * local, pill.halfSize, pill.edgeR);
    } else if (shapeId == 1) {
      pd = sdfPrism(local, pill.halfSize, pill.edgeR);
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
fn maxInternalPath() -> f32 {
  let count = min(u32(frame.pillCount), MAX_PILLS);
  var m: f32 = 0.0;
  for (var i: u32 = 0u; i < count; i = i + 1u) {
    let hs = frame.pills[i].halfSize;
    m = max(m, length(hs) * 2.0);
  }
  return max(m, 32.0);  // floor so degenerate zero-size pills don't stop march
}

fn sceneNormal(p: vec3<f32>) -> vec3<f32> {
  let e = vec2<f32>(HIT_EPS, 0.0);
  return normalize(vec3<f32>(
    sceneSdf(p + e.xyy) - sceneSdf(p - e.xyy),
    sceneSdf(p + e.yxy) - sceneSdf(p - e.yxy),
    sceneSdf(p + e.yyx) - sceneSdf(p - e.yyx),
  ));
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

// sRGB OETF (linear → gamma-encoded). Applied iff `frame.applySrgbOetf == 1`,
// i.e. when the canvas format is non-sRGB (getPreferredCanvasFormat typically
// returns bgra8unorm) and the hardware won't auto-encode.
fn linearToSrgb(c: vec3<f32>) -> vec3<f32> {
  let cutoff = vec3<f32>(0.0031308);
  let low    = c * 12.92;
  let high   = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
  return select(high, low, c <= cutoff);
}

fn encodeDisplay(c: vec3<f32>) -> vec3<f32> {
  if (frame.applySrgbOetf > 0.5) { return linearToSrgb(c); }
  return c;
}

// ---------- proxy vertex shader ----------
//
// Draws a per-pill 3D bounding box (a unit cube scaled to halfSize+edgeR,
// optionally rotated for shape==cube) instead of a fullscreen triangle. The
// rasterizer produces fragments for the exact projected silhouette of the
// proxy mesh, so tight coverage on rotated cubes is automatic.

// Unit cube, 36 verts, 12 tris, CCW outward winding (so `cullMode: 'back'`
// leaves one invocation per covered pixel).
const CUBE_VERTS: array<vec3<f32>, 36> = array<vec3<f32>, 36>(
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
// `(cx, cy, cz)` looking down -Z, with the z=0 plane mapping to the screen
// exactly 1:1 in world pixels. A point at the screen plane projects to its own
// (x, y). A point closer to the camera than z=0 projects outward; a point
// beyond z=0 projects inward. Returns NDC (x in [-1,1], y in [-1,1] with DOM
// top = +1) and a clip-W equal to depth-from-camera (positive for in-front).
fn projectWorld(p: vec3<f32>) -> vec4<f32> {
  let persp = frame.projection > 0.5;
  let camXY = frame.resolution * 0.5;

  var uv: vec2<f32>;
  if (persp) {
    let dz = frame.cameraZ - p.z;
    // Bail out when the vertex is behind the camera — degenerate clip vector
    // that fails the rasterizer's front-of-camera cull.
    if (dz <= 1.0) {
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

  // Unit cube corner in [-1, 1]^3 → local box sized to the pill's halfSize
  // (plus edgeR for the rounded rim so the proxy always fully covers the
  // actual shape).
  let extent  = pill.halfSize + vec3<f32>(pill.edgeR);
  var corner  = CUBE_VERTS[vi] * extent;

  if (shapeId == 2) {
    // The shader defines the cube via `local = rot * (p - center)`, so a
    // world-space proxy corner that maps to the unit-cube local-space corner
    // `c` is `center + transpose(rot) * (c * extent)`.
    let rot = cubeRotation(frame.time);
    corner  = transpose(rot) * corner;
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
  o.color   = vec4<f32>(encodeDisplay(blend), 1.0);
  o.history = vec4<f32>(blend, 1.0);
  return o;
}

@fragment
fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> FsOut {
  // DOM-top-origin pixel coords so they match pointer events and defaultPills.
  let px = fragCoord.xy;
  let uv = px / frame.resolution;

  // Ortho: parallel rays going -Z, origin just above the scene volume.
  // Perspective: rays diverge from the camera through each screen pixel,
  // hitting the z=0 screen plane exactly at (px.x, px.y, 0). FOV is encoded
  // in cameraZ (smaller = wider FOV).
  var ro: vec3<f32>;
  var rd: vec3<f32>;
  if (frame.projection > 0.5) {
    ro = vec3<f32>(frame.resolution * 0.5, frame.cameraZ);
    rd = normalize(vec3<f32>(px, 0.0) - ro);
  } else {
    ro = vec3<f32>(px, 400.0);
    rd = vec3<f32>(0.0, 0.0, -1.0);
  }
  // One upper bound that works for both projections: a ray can't travel
  // further than the camera-to-far-plane distance in the scene.
  let maxT = 2.0 * max(frame.cameraZ, 400.0) + 400.0;
  let h    = sphereTrace(ro, rd, maxT);
  let bg   = textureSampleLevel(photoTex, photoSmp, coverUv(uv), 0.0).rgb;

  if (!h.ok) {
    // Proxy covered this pixel but no cube surface was reached — pure waste
    // (the proxy over-covers the shape). In debug mode, tint these MORE
    // pinkly so the over-coverage "halo" around the cube silhouette is
    // visible. In normal mode, fall through to the bg color.
    var bgFinal = bg;
    if (frame.debugProxy > 0.5) {
      bgFinal = mix(bg, vec3<f32>(1.0, 0.3, 0.7), 0.5);
    }
    var bgOut: FsOut;
    bgOut.color   = vec4<f32>(encodeDisplay(bgFinal), 1.0);
    bgOut.history = vec4<f32>(bgFinal, 1.0);
    return bgOut;
  }

  let nFront   = sceneNormal(h.p);
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
    sharedExit  = insideTrace(h.p, r1hero, internalMax);
    sharedNBack = -sceneNormal(sharedExit);
  }

  // External front-face reflection — used BOTH as TIR fallback and as the
  // per-wavelength reflection color (mixed via per-λ Fresnel below).
  let refl     = reflect(rd, nFront);
  let reflUv   = screenUvFromWorld(h.p.xy) + refl.xy * 0.2;
  let reflSrc  = textureSampleLevel(photoTex, photoSmp, coverUv(reflUv), 0.0).rgb
              * vec3<f32>(0.85, 0.9, 1.0);

  // Front-face cosine (angle of incidence). Identical for every wavelength at
  // the front face, so compute once.
  let cosT = max(dot(-rd, nFront), 0.0);

  // Per-pixel stratified jitter: each pixel gets its own wavelength phase so
  // adjacent pixels sample DIFFERENT λ. The eye (and post-process history
  // accumulation) averages the spatial noise, so the rainbow looks smooth at
  // a given N — effectively 2-4× more samples worth of quality for free.
  // `jitter` (frame counter) breaks temporal coherence so history accumulates
  // new choices each frame instead of locking onto one stratum.
  let pxJit = hash21(px + vec2<f32>(jitter * 1000.0, frame.time * 37.0));

  // For each wavelength λ:
  //   1. compute per-λ IOR via Cauchy
  //   2. refract into the glass
  //   3. insideTrace to the back face (skipped in Approx mode — reuses sharedExit)
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
      pExit = insideTrace(h.p, r1, internalMax);
      nBack = -sceneNormal(pExit);
    }
    let r2 = refract(r1, nBack, ior);

    var refractL: vec3<f32>;
    if (dot(r2, r2) < 1e-4) {
      refractL = reflSrc;  // exit TIR → take the external reflection
    } else {
      let uvOff = screenUvFromWorld(pExit.xy) + r2.xy * strength;
      refractL  = textureSampleLevel(photoTex, photoSmp, coverUv(uvOff), 0.0).rgb;
    }

    // Per-wavelength Schlick Fresnel: short λ (blue) has higher IOR → higher F.
    let F_lambda = schlickFresnel(cosT, ior);
    let L        = mix(refractL, reflSrc, F_lambda);

    let lambdaRgb = max(xyzToSrgb(cieXyz(lambda)), vec3<f32>(0.0));
    rgbAccum  = rgbAccum  + L * lambdaRgb;
    rgbWeight = rgbWeight + lambdaRgb;
  }

  // Normalize against the per-wavelength primary sum — keeps a flat white
  // spectrum neutral for any N.
  let outRgb = max(rgbAccum / max(rgbWeight, vec3<f32>(1e-4)), vec3<f32>(0.0));

  // History is stored in rgba16float (linear). Blend in linear space; encode
  // for display only on the swapchain write. `historyBlend` is normally 0.2 —
  // host bumps it to 1.0 for one frame on any scene change (photo reload,
  // preset click, shape switch) so the previous scene doesn't ghost in.
  let prev  = textureSampleLevel(historyTex, historySmp, uv, 0.0).rgb;
  var blend = mix(prev, outRgb, frame.historyBlend);

  // Debug: tint every proxy fragment pink so the proxy silhouette is visible.
  // This path (ray hit the cube) gets a light tint; the miss path above gets
  // a heavier tint so the over-coverage halo stands out.
  if (frame.debugProxy > 0.5) {
    blend = mix(blend, vec3<f32>(1.0, 0.3, 0.7), 0.2);
  }

  var o: FsOut;
  o.color   = vec4<f32>(encodeDisplay(blend), 1.0);
  o.history = vec4<f32>(blend, 1.0);
  return o;
}
