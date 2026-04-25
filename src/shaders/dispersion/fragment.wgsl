// ---------- fragment ----------

struct FsOut {
  @location(0) color:   vec4<f32>,
  @location(1) history: vec4<f32>,
};

struct ProxyFsIn {
  @builtin(position) fragCoord: vec4<f32>,
  @location(0) @interpolate(flat) instanceIdx: u32,
};

// Cheap background pass: sample photo, blend history, done. No sphere-trace,
// no refraction, no per-wavelength loop. Runs for the whole screen; the proxy
// pass then overrides covered pixels with the heavy shader's output.
@fragment
fn fs_bg(@builtin(position) fragCoord: vec4<f32>) -> FsOut {
  let uv    = fragCoord.xy / frame.resolution;
  // Match fs_main's bg source: envmap when enabled (unified scene),
  // Picsum photo otherwise. Rebuild the per-pixel view ray matching
  // fs_main's construction — MINUS the sub-pixel TAA jitter. fs_bg is
  // a single fullscreen pass with no per-sample accumulation, so
  // jitter would just add temporal noise to the sky without aiding
  // convergence; the proxy fragments that DO need jitter sit inside
  // the silhouette and are shaded by fs_main instead. The small
  // sub-pixel offset at the silhouette edge is invisible because the
  // proxy pass overwrites those pixels anyway.
  let bgPhoto = textureSampleLevel(photoTex, photoSmp, coverUv(uv), 0.0).rgb;
  var rdBg: vec3<f32>;
  if (frame.projection > 0.5) {
    let roP = vec3<f32>(frame.resolution * 0.5, frame.cameraZ);
    rdBg = normalize(vec3<f32>(fragCoord.xy, 0.0) - roP);
  } else {
    rdBg = vec3<f32>(0.0, 0.0, -1.0);
  }
  let bgEnv = sampleEnvmap(rdBg);
  let bg    = select(bgPhoto, bgEnv, frame.envmapEnabled > 0.5);
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

fn proxyBgOut(uv: vec2<f32>, bg: vec3<f32>) -> FsOut {
  var bgFinal = bg;
  if (frame.debugProxy > 0.5) {
    bgFinal = mix(bg, vec3<f32>(1.0, 0.3, 0.7), 0.5);
  }

  let prevMiss = textureSampleLevel(historyTex, historySmp, uv, 0.0).rgb;
  let blend    = adaptiveBlend(prevMiss, bgFinal);
  var out: FsOut;
  out.color   = vec4<f32>(blend, 1.0);
  out.history = vec4<f32>(blend, 1.0);
  return out;
}

// Per-stratum spatial/temporal noise for the wavelength loop. The host writes
// `frame.jitter` to the `SPECTRAL_JITTER_DISABLED` sentinel (-1; defined in
// `src/spectralSampling.ts`) when **Temporal jitter** is off — we treat any
// negative value as the off signal and return 0 so every pixel samples the
// stratum centre, making the toggle visibly stop fizzling at high `sampleCount`.
fn spectralStratumJitter(px: vec2<f32>, jitter: f32) -> f32 {
  if (jitter < 0.0) {
    return 0.0;
  }
  return hash21(px + vec2<f32>(jitter * 1000.0, frame.time * 37.0)) - 0.5;
}

// Back-face exit dispatcher. Centralises the entry-bias trick (push the
// front-hit point one MIN_STEP inward along the refracted ray so the
// analytic exits' slab math doesn't divide 0/0 → silent wrong-axis pick →
// edge speckle) plus the cube/plate/other dispatch — both used by the
// hero-mode shared trace and the per-wavelength loop.
fn backExit(hitP: vec3<f32>, r1: vec3<f32>, shapeId: i32, pillIdx: u32, internalMax: f32) -> CubeExit {
  let roEntry = hitP + r1 * MIN_STEP;
  if (shapeId == 0) { return pillAnalyticExit(roEntry, r1, pillIdx); }
  if (shapeId == 1) { return prismAnalyticExit(roEntry, r1, pillIdx); }
  if (shapeId == 2) { return cubeAnalyticExit(roEntry, r1, pillIdx); }
  if (shapeId == 3) { return plateAnalyticExit(roEntry, r1, pillIdx); }
  // Phase B: diamond uses an analytical polytope exit + short TIR chain (≤3
  // bounces, wired in the wavelength loop below). The analytical normal eliminates
  // the finite-diff gradient degeneracy at facet edges that previously
  // sent TIR fallback to `reflSrc` and produced the "sudden face
  // appearing" tumble artifact.
  if (shapeId == 4) { return diamondAnalyticExit(roEntry, r1, pillIdx); }
  let pExit = insideTrace(hitP, r1, internalMax);
  return CubeExit(pExit, -sceneNormal(pExit));
}

@fragment
fn fs_main(in: ProxyFsIn) -> FsOut {
  // DOM-top-origin pixel coords so they match pointer events and defaultPills.
  let px = in.fragCoord.xy;
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
  // Background source: when envmap is enabled, sample the HDR panorama
  // in the VIEW direction (rd) so bg, reflection, AND refraction all
  // read from the same unified environment — the scene reads as a
  // diamond floating in a real studio/outdoor space instead of a
  // diamond pasted on top of an unrelated Picsum photo. Legacy photo
  // sampling path preserved for A/B comparison and for when the user
  // disables envmap entirely.
  let bgPhoto  = textureSampleLevel(photoTex, photoSmp, coverUv(uv), 0.0).rgb;
  let bgEnv    = sampleEnvmap(rd);
  let bg       = select(bgPhoto, bgEnv, frame.envmapEnabled > 0.5);

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
  let isPill    = shapeId == 0;
  let isPrism   = shapeId == 1;
  let isCube    = shapeId == 2;
  let isPlate   = shapeId == 3;
  // Diamond (shapeId == 4) is dispatched to `fs_main_diamond` via the
  // `diamondProxy` pipeline (see encodeScene in src/webgpu/pipeline.ts),
  // so it never reaches `fs_main`. No `isDiamond` branches here.
  //
  // `hasMotionPivot` gates the TAA reprojection call below: a shape needs
  // a per-frame rotation uniform (cubeRot / plateRot) for reprojection to
  // make sense. (Diamond has its own motion-pivot path inside
  // `fs_main_diamond`.)
  let hasMotionPivot  = isCube || isPlate;

  // Bg-fallback short-circuit. Run BEFORE `sceneNormal` because WGSL `select`
  // evaluates both arms — wrapping `sceneNormal(h.p)` in select(_, _, h.ok)
  // would still cost 6 SDF evals per miss pixel, and the proxy mesh
  // intentionally over-covers the silhouette so misses are common.
  if (!h.ok) {
    return proxyBgOut(uv, bg);
  }

  // Hit path. Pill-index scan is per-shape; gated on h.ok above so misses
  // skip it entirely. (Diamond's instanceIdx dispatch lives in
  // `fs_main_diamond` — it never reaches this `fs_main` path.)
  var analyticIdx: u32 = 0u;
  if      (isPill)    { analyticIdx = hitPillPillIdx(h.p); }
  else if (isPrism)   { analyticIdx = hitPrismPillIdx(h.p); }
  else if (isCube)    { analyticIdx = hitCubePillIdx(h.p); }
  else if (isPlate)   { analyticIdx = hitPlatePillIdx(h.p); }

  // `sceneNormal` returns the zero vector when the local gradient is too
  // small to normalise (silhouette / wave-crest singularity). Falling back
  // to an arbitrary normal would produce visible-but-wrong refraction
  // colours; routing those pixels through the bg path instead lets them
  // blend cleanly with the silhouette neighbourhood.
  var nFront: vec3<f32>;
  if (isPill) {
    nFront = sceneNormalPill(h.p, analyticIdx);
  } else if (isPrism) {
    nFront = sceneNormalPrism(h.p, analyticIdx);
  } else {
    nFront = sceneNormal(h.p);
  }
  if (dot(nFront, nFront) <= 0.5) {
    return proxyBgOut(uv, bg);
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
  let refl = reflect(rd, nFront);
  // Two source strategies for the reflection colour:
  //   Phase A legacy: sample the BACKGROUND photo at a UV offset along
  //     the reflection vector. Cheap, looks "reflective" at first glance,
  //     but physically wrong — the reflection should depend on the
  //     ENVIRONMENT direction, not on a shifted view of the thing BEHIND
  //     the glass. Produces the "same photo slightly shifted" artefact
  //     that looks like mis-refraction at glancing angles.
  //   Phase C envmap: sample a linear HDR panorama at the reflection
  //     direction. Real reflections of a real environment; drives the
  //     Fresnel highlight with bright sky/studio-light HDR peaks which
  //     is what makes diamonds actually sparkle.
  // Gate by `frame.envmapEnabled` so a user can toggle between the two
  // for A/B comparison and so the startup frame (before the real HDRI
  // arrives) falls back gracefully to the legacy path.
  let reflUv     = uv + (refl - rd).xy * 0.2;
  let reflCover  = coverUv(reflUv);
  let reflInBnds = all(reflCover >= vec2<f32>(0.0)) && all(reflCover <= vec2<f32>(1.0));
  let reflRaw    = textureSampleLevel(photoTex, photoSmp, reflCover, 0.0).rgb;
  let reflLegacy = select(bg, reflRaw, reflInBnds) * vec3<f32>(0.85, 0.9, 1.0);
  let reflSrc    = select(reflLegacy, sampleEnvmap(refl), frame.envmapEnabled > 0.5);

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
  //
  // When **Temporal jitter** is off the host writes `SPECTRAL_JITTER_DISABLED`
  // (-1) and `spectralStratumJitter` returns 0, so `t` lands at every stratum's
  // exact centre — the on/off difference becomes visible instead of being
  // washed out by per-pixel hash.
  let pxJit = spectralStratumJitter(px, jitter);

  // For each wavelength λ:
  //   1. compute per-λ IOR via Cauchy
  //   2. refract into the glass
  //   3. find the back-face exit point + inward normal (skipped in Approx
  //      mode — reuses sharedExit/sharedNBack):
  //        pill  → pillAnalyticExit  (AABB slab + sdfPill / sdfPillGrad)
  //        prism → prismAnalyticExit (AABB slab + sdfPrism / sdfPrismGrad)
  //        cube  → cubeAnalyticExit  (O(1) slab + rounded-box gradient)
  //        plate → plateAnalyticExit (O(1) slab + Newton-refined wavy Z face)
  //        diam. → diamondAnalyticExit
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
      // Non-diamond shapes (pill/prism/cube/plate) fall back to `reflSrc` on
      // TIR. Their TIR still benefits from the envmap indirectly: `reflSrc`
      // resolves to `sampleEnvmap(refl)` when envmap is enabled, so cube/plate
      // TIR samples the real environment instead of the legacy Phase A
      // UV-offset photo hack. Upgrading them to their own bounce chain would
      // need analytical cube/plate TIR facet picking — Phase D territory.
      // (Diamond has its own multi-bounce analytic TIR chain in
      // `fs_main_diamond`.)
      refractL = reflSrc;
    } else if (r2NaN || r2OOB) {
      refractL = bg;
    } else {
      // Successful refract. Envmap: sample the panorama at the
      // refracted direction; strength interpolates between rd (bg
      // direction = no refraction visible) and r2 (full refraction).
      // Photo: classic UV-offset sample.
      if (frame.envmapEnabled > 0.5) {
        refractL = sampleEnvmap(mix(rd, r2, strength));
      } else {
        refractL = textureSampleLevel(photoTex, photoSmp, uvCover, photoLod).rgb;
      }
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

  // Diamond debug overlays live in `fs_main_diamond`; `fs_main` only handles
  // non-diamond shapes and has no facet-color / wireframe pass to draw.

  var o: FsOut;
  o.color   = vec4<f32>(blend, 1.0);
  o.history = vec4<f32>(blend, 1.0);
  return o;
}

@fragment
fn fs_main_diamond(in: ProxyFsIn) -> FsOut {
  let px = in.fragCoord.xy;
  let uv = px / frame.resolution;

  var rayPx = px;
  if (frame.taaEnabled > 0.5) {
    let taaJit = vec2<f32>(
      hash21(px + vec2<f32>(frame.time * 7.19, 3.141)) - 0.5,
      hash21(px + vec2<f32>(frame.time * 11.23, 6.283)) - 0.5,
    );
    rayPx = px + taaJit;
  }

  var ro: vec3<f32>;
  var rd: vec3<f32>;
  if (frame.projection > 0.5) {
    ro = vec3<f32>(frame.resolution * 0.5, frame.cameraZ);
    rd = normalize(vec3<f32>(rayPx, 0.0) - ro);
  } else {
    ro = vec3<f32>(rayPx, 400.0);
    rd = vec3<f32>(0.0, 0.0, -1.0);
  }

  let bgPhoto = textureSampleLevel(photoTex, photoSmp, coverUv(uv), 0.0).rgb;
  let bgEnv   = sampleEnvmap(rd);
  let bg      = select(bgPhoto, bgEnv, frame.envmapEnabled > 0.5);

  // Unlike the generic path, this dedicated diamond path bypasses the
  // scene-wide sphereTrace. Keep the front hit scene-wide anyway so overlapping
  // diamond proxies still resolve to the nearest instance even though the proxy
  // pass has no depth buffer.
  let front       = diamondAnalyticHitScene(ro, rd);
  if (!front.ok) {
    return proxyBgOut(uv, bg);
  }

  let analyticIdx = front.pillIdx;
  let hP     = front.pWorld;
  // Keep the analytic front-hit POSITION (cheap, exact, no 64-step march),
  // but use the SDF gradient for the entry normal. The exact per-facet normal
  // makes every crown facet transition razor-sharp, which visually doubles the
  // back-facet lines seen through the front surface under FXAA. `sceneNormal`
  // preserves the old soft blend across facet boundaries while retaining the
  // performance win from the analytic hit point.
  let nFront = sceneNormal(front.pWorld);
  if (dot(nFront, nFront) <= 0.5) {
    return proxyBgOut(uv, bg);
  }

  let n_d      = frame.n_d;
  let V_d      = frame.V_d;
  let N        = clamp(i32(frame.sampleCount), 1, MAX_N);
  let strength = frame.refractionStrength;
  let jitter   = frame.jitter;
  let useHero  = frame.refractionMode > 0.5;

  var sharedExit  = hP;
  var sharedNBack = -nFront;
  if (useHero) {
    let iorHero = cauchyIor(frame.heroLambda, n_d, V_d);
    let r1hero  = refract(rd, nFront, 1.0 / iorHero);
    if (dot(r1hero, r1hero) >= 1e-4) {
      let ex = diamondAnalyticExit(hP + r1hero * MIN_STEP, r1hero, analyticIdx);
      sharedExit  = ex.pWorld;
      sharedNBack = ex.nBack;
    }
  }

  let refl = reflect(rd, nFront);
  let reflUv     = uv + (refl - rd).xy * 0.2;
  let reflCover  = coverUv(reflUv);
  let reflInBnds = all(reflCover >= vec2<f32>(0.0)) && all(reflCover <= vec2<f32>(1.0));
  let reflRaw    = textureSampleLevel(photoTex, photoSmp, reflCover, 0.0).rgb;
  let reflLegacy = select(bg, reflRaw, reflInBnds) * vec3<f32>(0.85, 0.9, 1.0);
  let reflSrc    = select(reflLegacy, sampleEnvmap(refl), frame.envmapEnabled > 0.5);

  let cosT     = max(dot(-rd, nFront), 0.0);
  let photoLod = clamp(-log2(max(cosT, 0.02)) - 1.0, 0.0, 6.0);
  let pxJit    = spectralStratumJitter(px, jitter);

  var rgbAccum  = vec3<f32>(0.0);
  var rgbWeight = vec3<f32>(0.0);
  for (var i: i32 = 0; i < N; i = i + 1) {
    let t      = (f32(i) + 0.5 + pxJit) / f32(N);
    let lambda = mix(380.0, 700.0, t);
    let ior    = cauchyIor(lambda, n_d, V_d);
    let r1     = refract(rd, nFront, 1.0 / ior);
    if (dot(r1, r1) < 1e-4) { continue; }

    var pExit = sharedExit;
    var nBack = sharedNBack;
    if (!useHero) {
      let ex = diamondAnalyticExit(hP + r1 * MIN_STEP, r1, analyticIdx);
      pExit = ex.pWorld;
      nBack = ex.nBack;
    }
    let r2 = refract(r1, nBack, ior);

    var refractL: vec3<f32>;
    let uvOff      = uv + (r2 - rd).xy * strength;
    let uvCover    = coverUv(uvOff);
    let uvInBounds = all(uvCover >= vec2<f32>(0.0)) && all(uvCover <= vec2<f32>(1.0));
    let r2dot      = dot(r2, r2);
    let r2NaN      = r2dot != r2dot;
    let r2TIR      = r2dot < 1e-4 && !r2NaN;
    let r2OOB      = !uvInBounds;
    if (r2TIR) {
      if (!useHero) {
        var curR1    = r1;
        var curNBack = nBack;
        var curP     = pExit;
        var outDir: vec3<f32> = vec3<f32>(0.0);
        var resolved: bool     = false;
        var tirDbgAnalyticMiss: bool = false;
        let tirMaxB = u32(clamp(round(frame.diamondTirMaxBounces), 1.0, 32.0));
        for (var bounce: u32 = 0u; bounce < tirMaxB; bounce = bounce + 1u) {
          let bouncedR1 = reflect(curR1, curNBack);
          let roNudge = clamp(
            frame.diamondSize * TIR_BOUNCE_RO_NUDGE_SCALE,
            TIR_BOUNCE_RO_NUDGE_FLOOR,
            TIR_BOUNCE_RO_NUDGE_CEIL,
          );
          let roChain = curP + bouncedR1 * roNudge;
          let exN     = diamondAnalyticExit(roChain, bouncedR1, analyticIdx);
          if (dot(exN.nBack, exN.nBack) < 0.25) {
            tirDbgAnalyticMiss = true;
            break;
          }
          let trial    = refract(bouncedR1, exN.nBack, ior);
          let trialDot = dot(trial, trial);
          let trialNaN = trialDot != trialDot;
          if (trialDot >= 1e-4 && !trialNaN) {
            outDir   = trial;
            resolved = true;
            break;
          }
          curR1    = bouncedR1;
          curNBack = exN.nBack;
          curP     = exN.pWorld;
        }
        if (resolved) {
          let uvOffB    = uv + (outDir - rd).xy * strength;
          let uvCoverB  = coverUv(uvOffB);
          let inBoundsB = all(uvCoverB >= vec2<f32>(0.0)) && all(uvCoverB <= vec2<f32>(1.0));
          if (frame.envmapEnabled > 0.5) {
            refractL = sampleEnvmap(mix(rd, outDir, strength));
          } else if (inBoundsB) {
            refractL = textureSampleLevel(photoTex, photoSmp, uvCoverB, photoLod).rgb;
          } else {
            refractL = bg;
          }
        } else {
          let exhaustedFallback = select(bg, reflSrc, frame.envmapEnabled > 0.5);
          let dbgTir    = vec3<f32>(1.0, 0.2, 0.75);
          let dbgAnMiss = vec3<f32>(1.0, 0.45, 0.05);
          let dbgTint   = select(dbgTir, dbgAnMiss, tirDbgAnalyticMiss);
          refractL = select(exhaustedFallback, dbgTint, frame.diamondTirDebug > 0.5);
        }
      } else {
        refractL = reflSrc;
      }
    } else if (r2NaN || r2OOB) {
      refractL = bg;
    } else {
      if (frame.envmapEnabled > 0.5) {
        refractL = sampleEnvmap(mix(rd, r2, strength));
      } else {
        refractL = textureSampleLevel(photoTex, photoSmp, uvCover, photoLod).rgb;
      }
    }

    let F_lambda = schlickFresnel(cosT, ior);
    let L        = mix(refractL, reflSrc, F_lambda);
    let lambdaRgb = max(xyzToSrgb(cieXyz(lambda)), vec3<f32>(0.0));
    rgbAccum  = rgbAccum  + L * lambdaRgb;
    rgbWeight = rgbWeight + lambdaRgb;
  }

  let raw     = rgbAccum / max(rgbWeight, vec3<f32>(1e-4));
  let safe    = select(raw, bg, vec3<bool>(any(raw != raw)));
  let clamped = clamp(safe, vec3<f32>(0.0), vec3<f32>(8.0));

  let silhouetteGate = smoothstep(0.05, 0.15, frame.historyBlend);
  let silhouetteMix  = mix(1.0, smoothstep(0.0, 0.05, cosT), silhouetteGate);
  let outRgb         = mix(bg, clamped, silhouetteMix);

  var historyUv = uv;
  if (frame.taaEnabled > 0.5) {
    historyUv = reprojectHit(hP, px, analyticIdx, 4, uv);
  }
  let prev  = textureSampleLevel(historyTex, historySmp, historyUv, 0.0).rgb;
  var blend = adaptiveBlend(prev, outRgb);
  if (frame.debugProxy > 0.5) {
    blend = mix(blend, vec3<f32>(1.0, 0.3, 0.7), 0.2);
  }

  var display = blend;
  if (frame.diamondFacetColor > 0.5) {
    let pillCenter = frame.pills[analyticIdx].center;
    display = sdfDiamondFacetColor(hP - pillCenter, frame.diamondSize);
  }
  if (frame.diamondWireframe > 0.5) {
    let pillCenter = frame.pills[analyticIdx].center;
    let edgeW      = sdfDiamondEdgeWeight(hP - pillCenter, frame.diamondSize);
    display = mix(display, vec3<f32>(1.0, 0.25, 0.25), edgeW * 0.85);
  }

  var o: FsOut;
  o.color   = vec4<f32>(display, 1.0);
  o.history = vec4<f32>(blend, 1.0);
  return o;
}
