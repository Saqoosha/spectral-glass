const MAX_PILLS: u32 = 8u;

struct PillGpu {
  centerEdge: vec4<f32>,  // xyz = center, w = edgeR
  halfPad:    vec4<f32>,  // xyz = halfSize, w = 0
};

struct Frame {
  resolutionPhoto: vec4<f32>,  // xy = resolution px, zw = photo px
  spectralA:       vec4<f32>,  // x = n_d, y = V_d, z = N, w = refractionStrength
  spectralB:       vec4<f32>,  // x = jitter, y = refractionMode, z = pillCount
  pills:           array<PillGpu, MAX_PILLS>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var photoTex: texture_2d<f32>;
@group(0) @binding(2) var photoSmp: sampler;
@group(0) @binding(3) var historyTex: texture_2d<f32>;
@group(0) @binding(4) var historySmp: sampler;

// ---------- coords ----------

fn coverUv(uv: vec2<f32>) -> vec2<f32> {
  let res   = frame.resolutionPhoto.xy;
  let ph    = frame.resolutionPhoto.zw;
  let sA    = res.x / res.y;
  let pA    = ph.x  / ph.y;
  var s     = vec2<f32>(1.0, 1.0);
  if (sA > pA) { s = vec2<f32>(1.0, pA / sA); } else { s = vec2<f32>(sA / pA, 1.0); }
  return (uv - vec2<f32>(0.5)) * s + vec2<f32>(0.5);
}

fn screenUvFromWorld(px: vec2<f32>) -> vec2<f32> {
  let res = frame.resolutionPhoto.xy;
  return vec2<f32>(px.x / res.x, 1.0 - px.y / res.y);
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

fn sceneSdf(p: vec3<f32>) -> f32 {
  let count = u32(frame.spectralB.z);
  var d: f32 = 1e9;
  for (var i: u32 = 0u; i < count; i = i + 1u) {
    let pill = frame.pills[i];
    let local = p - pill.centerEdge.xyz;
    let hs    = pill.halfPad.xyz;
    let eR    = pill.centerEdge.w;
    d = min(d, sdfPill(local, hs, eR));
  }
  return d;
}

fn sceneNormal(p: vec3<f32>) -> vec3<f32> {
  let e = vec2<f32>(0.5, 0.0);
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
    if (d < 0.5) { return Hit(true, p, t); }
    t = t + max(d, 0.5);
    if (t > maxT) { break; }
  }
  return Hit(false, vec3<f32>(0.0), 0.0);
}

fn insideTrace(ro: vec3<f32>, rd: vec3<f32>, maxT: f32) -> vec3<f32> {
  var t: f32 = 0.0;
  var p = ro;
  for (var i: i32 = 0; i < 32; i = i + 1) {
    p = ro + rd * t;
    let d = -sceneSdf(p);
    if (d < 0.5) { return p; }
    t = t + max(d, 0.5);
    if (t > maxT) { break; }
  }
  return p;
}

// ---------- spectral math ----------

fn cauchyIor(lambda: f32, n_d: f32, V_d: f32) -> f32 {
  return max(n_d + (n_d - 1.0) / V_d * (523655.0 / (lambda * lambda) - 1.5168), 1.0);
}

fn gLobe(lambda: f32, mu: f32, s1: f32, s2: f32) -> f32 {
  let sigma = select(s2, s1, lambda < mu);
  let t = (lambda - mu) / sigma;
  return exp(-0.5 * t * t);
}

fn cieXyz(lambda: f32) -> vec3<f32> {
  let x =  0.362 * gLobe(lambda, 442.0, 16.0, 26.7)
        +  1.056 * gLobe(lambda, 599.8, 37.9, 31.0)
        + -0.065 * gLobe(lambda, 501.1, 20.4, 26.2);
  let y =  0.821 * gLobe(lambda, 568.8, 46.9, 40.5)
        +  0.286 * gLobe(lambda, 530.9, 16.3, 31.1);
  let z =  1.217 * gLobe(lambda, 437.0, 11.8, 36.0)
        +  0.681 * gLobe(lambda, 459.0, 26.0, 13.8);
  return vec3<f32>(x, y, z);
}

fn xyzToSrgb(c: vec3<f32>) -> vec3<f32> {
  let m = mat3x3<f32>(
     3.2404542, -0.9692660,  0.0556434,
    -1.5371385,  1.8760108, -0.2040259,
    -0.4985314,  0.0415560,  1.0572252,
  );
  return m * c;
}

fn luminance(rgb: vec3<f32>) -> f32 {
  return dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn schlickFresnel(cosT: f32, n_d: f32) -> f32 {
  let f0 = pow((n_d - 1.0) / (n_d + 1.0), 2.0);
  let k  = 1.0 - clamp(cosT, 0.0, 1.0);
  return f0 + (1.0 - f0) * k * k * k * k * k;
}

// sRGB OETF (linear → gamma-encoded). Needed because the canvas format is
// bgra8unorm (non-sRGB) — it treats our output bytes as already gamma-encoded,
// so we must apply the encoding ourselves.
fn linearToSrgb(c: vec3<f32>) -> vec3<f32> {
  let cutoff = vec3<f32>(0.0031308);
  let low    = c * 12.92;
  let high   = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
  return select(high, low, c <= cutoff);
}

// ---------- fragment ----------

struct FsOut {
  @location(0) color:   vec4<f32>,
  @location(1) history: vec4<f32>,
};

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> FsOut {
  let res = frame.resolutionPhoto.xy;
  let px  = vec2<f32>(uv.x * res.x, (1.0 - uv.y) * res.y);
  let ro  = vec3<f32>(px, 400.0);
  let rd  = vec3<f32>(0.0, 0.0, -1.0);
  let h   = sphereTrace(ro, rd, 800.0);
  let bgUv = coverUv(uv);
  let bg  = textureSampleLevel(photoTex, photoSmp, bgUv, 0.0).rgb;
  if (!h.ok) {
    let bgSrgb = linearToSrgb(bg);
    var bgOut: FsOut;
    bgOut.color   = vec4<f32>(bgSrgb, 1.0);
    bgOut.history = vec4<f32>(bgSrgb, 1.0);
    return bgOut;
  }

  let nFront   = sceneNormal(h.p);
  let n_d      = frame.spectralA.x;
  let V_d      = frame.spectralA.y;
  let N        = i32(frame.spectralA.z);
  let strength = frame.spectralA.w;
  let jitter   = frame.spectralB.x;
  let approx   = frame.spectralB.y > 0.5;

  var sharedExit: vec3<f32>;
  var sharedNBack: vec3<f32>;
  if (approx) {
    let iorMid   = cauchyIor(540.0, n_d, V_d);
    let r1mid    = refract(rd, nFront, 1.0 / iorMid);
    sharedExit   = insideTrace(h.p + r1mid * 1.0, r1mid, 300.0);
    sharedNBack  = -sceneNormal(sharedExit);
  } else {
    sharedExit  = h.p;
    sharedNBack = -nFront;
  }

  var xyz      = vec3<f32>(0.0);
  var xyzWhite = vec3<f32>(0.0);  // XYZ of a flat L=1 spectrum — reference white

  for (var i: i32 = 0; i < N; i = i + 1) {
    let t      = (f32(i) + 0.5 + jitter) / f32(N);
    let lambda = mix(380.0, 700.0, t);
    let ior    = cauchyIor(lambda, n_d, V_d);

    let r1     = refract(rd, nFront, 1.0 / ior);
    var pExit: vec3<f32>;
    var nBack: vec3<f32>;
    if (approx) {
      pExit = sharedExit;
      nBack = sharedNBack;
    } else {
      pExit = insideTrace(h.p + r1 * 1.0, r1, 300.0);
      nBack = -sceneNormal(pExit);
    }
    let r2     = refract(r1, nBack, ior);
    let uvOff  = screenUvFromWorld(pExit.xy) + r2.xy * strength;
    let L      = textureSampleLevel(photoTex, photoSmp, coverUv(uvOff), 0.0).rgb;

    let cmf    = cieXyz(lambda);
    xyz        = xyz      + cmf * luminance(L);
    xyzWhite   = xyzWhite + cmf;
  }

  // Normalize in sRGB space against the same-pipeline sRGB of a flat white spectrum
  // so a grayscale photo maps to neutral gray output (independent of N).
  let rgbRaw   = xyzToSrgb(xyz);
  let rgbWhite = xyzToSrgb(xyzWhite);
  let rgb      = max(rgbRaw / max(rgbWhite, vec3<f32>(1e-4)), vec3<f32>(0.0));

  let cosT     = max(dot(-rd, nFront), 0.0);
  let F        = schlickFresnel(cosT, n_d);

  let refl     = reflect(rd, nFront);
  let reflUv   = screenUvFromWorld(h.p.xy) + refl.xy * 0.2;
  let reflRgb  = textureSampleLevel(photoTex, photoSmp, coverUv(reflUv), 0.0).rgb * vec3<f32>(0.85, 0.9, 1.0);

  let refrRgb  = rgb;
  let outRgb   = mix(refrRgb, reflRgb, F);

  // History is stored in rgba16float (linear), so blend in linear space then encode once for display.
  let prev   = textureSampleLevel(historyTex, historySmp, uv, 0.0).rgb;
  let alpha  = 0.2;
  let blend  = mix(prev, outRgb, alpha);

  var o: FsOut;
  o.color   = vec4<f32>(linearToSrgb(blend), 1.0);
  o.history = vec4<f32>(blend, 1.0);
  return o;
}
