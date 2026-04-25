const MAX_PILLS: u32 = 8u;
const MAX_N:      i32 = 64;
const HIT_EPS:    f32 = 0.25;  // hit tolerance (small — thin pills survive)
const MIN_STEP:   f32 = 0.5;   // min march step (larger — loop doesn't stall on near-zero SDF)
// Diamond TIR chain: nudge the bounce origin along the internal ray (pixel
// space) so the start is unambiguously inside. Scales with `diamondSize`
// but clamps: floor ~ DIAMOND_BOUNCE_EPS, ceil below neighbour-facet gaps
// so we never replay the 0.5 px MIN_STEP overshoot failure mode.
const TIR_BOUNCE_RO_NUDGE_SCALE: f32 = 0.0001;
const TIR_BOUNCE_RO_NUDGE_FLOOR: f32 = 0.01;
const TIR_BOUNCE_RO_NUDGE_CEIL:  f32 = 0.08;

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
  jitter:             f32,  // < 0 disables wavelength jitter; >= 0 seeds per-pixel/per-frame spectral strata
  refractionMode:     f32,
  pillCount:          f32,
  applySrgbOetf:      f32,  // unused by this shader (sRGB encoding moved to postprocess.wgsl). Slot is kept so the Frame UBO layout stays stable — host still writes 0/1 (uniforms.ts) and tests/uniformsLayout.test.ts pins the name. Reclaiming it means touching all three together.
  shape:              f32,  // 0 = pill (stadium), 1 = prism, 2 = cube (rotates), 3 = plate (wavy, tumbles), 4 = diamond (round brilliant, rotates)
  time:               f32,  // wall-clock seconds since start (always advancing, even while paused). Drives the noise streams: TAA sub-pixel jitter and per-pixel wavelength stratification (the latter only when `jitter >= 0`; the sentinel branch in `spectralStratumJitter` skips the time-seeded hash). Rotation matrices are derived from `sceneTime` (below), NOT from this field — see `cubeRot` / `plateRot` / `diamondRot` and the time-stream split in src/main.ts.
  historyBlend:       f32,  // 0.2 steady state, 1.0 when the scene changed this frame
  heroLambda:         f32,  // jittered each frame in [380,700], or fixed 540nm when temporal jitter is off
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
  // without refraction / dispersion confusing the signal.
  // `diamondTirDebug` (1.0 = on): where the TIR-bounce path doesn't resolve,
  // tint pixels — hot pink = chain used the full `diamondTirMaxBounces` budget
  // but refract(s→air) still TIRs; orange = `diamondAnalyticExit` miss (zero
  // or invalid nBack) so the failure isn't necessarily "need more bounces".
  // When off, the exhausted path blends with silhouette `bg` (no envmap) or
  // envmap-at-front-reflection; see the wavelength loop "exhausted" branch.
  // `diamondTirMaxBounces`: cap on internal reflections in the
  // diamond TIR-bounce loop (clamped 1..32 in fs_main; host default 6).
  // Three trailing pads keep this sub-block 32 B for uniform alignment.
  diamondSize:         f32,
  diamondWireframe:    f32,
  diamondFacetColor:   f32,
  diamondTirDebug:     f32,
  diamondTirMaxBounces: f32,
  _diamondParamsPad0:  f32,
  _diamondParamsPad1:  f32,
  _diamondParamsPad2:  f32,
  // HDR environment map parameters (Phase C).
  // `envmapExposure`: linear-light multiplier on the sampled panorama.
  // Most HDRIs have peaks in the 100-1000 range; 0.25 keeps them
  // visible without washing out the Fresnel mix.
  // `envmapRotation`: radians, rotate the sky around world-Y. Lets the
  // user "move the sun" without re-downloading.
  // `envmapEnabled`: 1.0 = use envmap for reflection/TIR paths, 0.0 =
  // keep the Phase A `reflSrc` fallback for A/B comparison and as a
  // graceful degrade while a new envmap is still fetching.
  envmapExposure:     f32,
  envmapRotation:     f32,
  envmapEnabled:      f32,
  smoothCurvature:    f32,  // 1 = L4 squircle rim on pill/cube/plate, 0 = legacy circular rim
  pills:              array<PillGpu, MAX_PILLS>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var photoTex: texture_2d<f32>;
@group(0) @binding(2) var photoSmp: sampler;
@group(0) @binding(3) var historyTex: texture_2d<f32>;
@group(0) @binding(4) var historySmp: sampler;
// HDR environment panorama (Phase C). rgba16float, equirectangular
// projection: U spans longitude ±π (repeat), V spans latitude from +π/2
// at the top to -π/2 at the bottom (clamp). Sampled by `sampleEnvmap()`
// below for reflection/TIR paths so diamond facets reflect a real sky
// /room instead of a UV-shifted photo hack.
@group(0) @binding(5) var envmapTex: texture_2d<f32>;
@group(0) @binding(6) var envmapSmp: sampler;

// Sample the equirectangular envmap in a given WORLD-space direction.
// Convention:
//   - WORLD +Y points VISUALLY DOWN on screen (DOM-top-origin world;
//     see pipeline.ts's `frontFace:'cw'` comment — NDC-up maps to
//     world-down). So a ray going world -Y is going visually UP toward
//     the HDRI's zenith, and we sample the panorama's top row (V=0)
//     there. The sign convention is baked into the V formula below.
//   - Longitude 0 at world +Z (toward camera), wraps ±π as you sweep
//     around Y — matches Poly Haven's canonical orientation so users
//     who've seen an HDRI in another tool see the same sun position.
//   - `envmapRotation` (radians) rotates the sky around Y, giving the
//     user a "turn the sun" knob without re-downloading a different
//     HDRI.
//   - `envmapExposure` multiplies linear-light intensity. HDRIs can
//     be 100×+ brighter than 1.0; the default 0.25 keeps typical peaks
//     visible without blowing out the rest of the image.
fn sampleEnvmap(dir: vec3<f32>) -> vec3<f32> {
  // Defend against zero / near-zero direction vectors. The refraction
  // call sites hand in `mix(rd, r2, strength)` which collapses to ~0
  // when r2 ≈ -rd at strength ≈ 0.5 (can happen on glancing-angle
  // refractions at diamond pavilion/crown rims). WGSL's normalize is
  // undefined for zero-length input — typically NaN, which then
  // poisons atan2 / asin / textureSampleLevel and re-introduces the
  // exact "black dots along bright edges" symptom the fp16 clamp in
  // envmap.ts was added to eliminate. Return the panorama's zenith as
  // a finite, "roughly upward" fallback — the ray was trying to look
  // somewhere ambiguous and any coherent direction beats NaN.
  let lenSq = dot(dir, dir);
  if (!(lenSq > 1.0e-8)) {
    return textureSampleLevel(envmapTex, envmapSmp, vec2<f32>(0.5, 0.0), 0.0).rgb
         * frame.envmapExposure;
  }
  let d = dir * inverseSqrt(lenSq);
  let twoPi = 6.28318530717958647693;
  let lon = atan2(d.x, d.z) + frame.envmapRotation;
  let u   = lon / twoPi + 0.5;
  // Map world-Y to HDRI-V with the sign that puts the zenith at V=0.
  // Since world +Y is visually down:
  //   d.y = -1 (ray pointing up)   → V = 0 (top of panorama = sky)
  //   d.y = +1 (ray pointing down) → V = 1 (bottom = ground)
  // The `+ asin(...)` (rather than `- asin(...)`) gets this mapping in
  // one go. A sign flip here inverts the HDRI upside-down on the
  // reflecting surface — the "gray region fed by upward rays" symptom
  // that the Phase C review caught.
  let v   = 0.5 + asin(clamp(d.y, -1.0, 1.0)) / 3.14159265358979323846;
  let raw = textureSampleLevel(envmapTex, envmapSmp, vec2<f32>(u, v), 0.0).rgb;
  return raw * frame.envmapExposure;
}

// ---------- coords ----------

fn coverUv(uv: vec2<f32>) -> vec2<f32> {
  let sA = frame.resolution.x / frame.resolution.y;
  let pA = frame.photoSize.x  / frame.photoSize.y;
  var s  = vec2<f32>(1.0);
  if (sA > pA) { s = vec2<f32>(1.0, pA / sA); } else { s = vec2<f32>(sA / pA, 1.0); }
  return (uv - vec2<f32>(0.5)) * s + vec2<f32>(0.5);
}
