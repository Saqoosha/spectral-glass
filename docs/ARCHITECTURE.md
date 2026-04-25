# Architecture

Two-pass WebGPU renderer. Per-pixel SDF sphere-tracing inside a per-pill 3D
proxy mesh, with a cheap fullscreen background pass underneath — so the heavy
refraction shader only runs on fragments the proxy actually covers.

## Frame path

```
┌──────────────────────────────────────────────────────────────────────┐
│  every RequestAnimationFrame:                                        │
│                                                                      │
│  1. resize canvas + history + post-intermediate if needed            │
│  2. push params → pills (hx/hy/hz/edgeR)                             │
│  3. writeFrame → uniform buffer (688 B: scalars + 6×mat3 + plate    │
│     + diamond 32B + envmap 16B blocks + pills)                     │
│  4. scene pass (writes → intermediate(rgba16f) + history[write]):    │
│     a. bg sub-pass: fullscreen triangle → fs_bg (active bg + history)│
│     b. proxy sub-pass: instanced 3D proxy mesh → fs_main             │
│          per-fragment camera ray (ortho OR perspective)              │
│          sphere-trace scene SDF                                      │
│          if miss: return bg (over-covered proxy fragment)            │
│          if hit:  per-pixel stratified λ jitter, for each λ:         │
│                    refract → inside-trace → refract out              │
│                    sample photo at uv_λ (TIR → reflect)              │
│                    per-wavelength Fresnel mix                        │
│                    accumulate weighted by xyzToSrgb(cmf(λ))          │
│                  EMA-blend with history[read] (historyBlend)         │
│                  write linear → intermediate @location(0)            │
│                  write linear → history[write] @location(1)          │
│  5. post pass (reads intermediate, writes → swapchain):              │
│     aaMode === 'fxaa' ? fs_fxaa : fs_passthrough                     │
│     sRGB OETF applied here (once) if swapchain is non-sRGB           │
│  6. flip history.current                                             │
└──────────────────────────────────────────────────────────────────────┘
```

Scene and post are encoded into a single command buffer and submitted
together (`main.ts` loop). The intermediate stays canvas-sized and is
reallocated in `resizeIntermediate` whenever the canvas size changes.

Two pipelines share one explicit bind group layout (so `frame` is visible to
both vertex and fragment stages). Two bind groups are pre-built at pipeline
creation (one per history-read slot) and swapped based on `history.current` —
no per-frame bind group allocation.

### Proxy mesh

`vs_proxy` emits a per-shape proxy mesh — pill/prism/cube/plate use a
`CUBE_PROXY_VERT_COUNT`-vertex unit cube (36 verts, 12 tris, CCW-outward
winding) scaled to `halfSize` (SDF already accounts for `edgeR`); diamond uses a
`DIAMOND_PROXY_VERT_COUNT`-vertex exact convex hull (138 verts, 46 tris)
synthesized from Tolkowsky constants in `diamondProxyVertex` (see
`src/shaders/diamond.wgsl`). Non-diamond scenes use the default four instances;
diamond shape/preset switches trim the live instance list to one so the
brilliant cut reads as a single object. The draw call issues
`max(CUBE_PROXY_VERT_COUNT, DIAMOND_PROXY_VERT_COUNT)` vertices per
instance; the `maxVerts` guard at the top of `vs_proxy` clips the upper
range for non-diamond shapes to an off-screen degenerate position.
For `shape == cube` / `plate` / `diamond` the corners are transformed by
`transpose(cubeRot)` / `transpose(plateRot)` / `transpose(diamondRot)`
so the rasterised silhouette matches the shader's world-space shape
exactly. Plate extends the Z half-extent by `waveAmp` so the rippling
surface never pokes through. Pill and prism are axis-aligned in world
space, so their AABB cube proxy already covers the shape tightly.
Back-face culling (`frontFace: 'cw'` because our Y-flipped projection
inverts winding from 3D to NDC) leaves one fragment invocation per
covered pixel.

### Camera

Projection is a uniform flag (ortho / perspective). In ortho, per-pixel rays
are parallel down `-Z` from `ro = (px, py, 400)`. In perspective, rays
diverge from a camera at `(w/2, h/2, cameraZ)` and pass through the pixel's
world point at `z = 0`. `cameraZ` is derived on the CPU from the user-facing
FOV: `cameraZ = (height/2) / tan(fov/2)`. `projectWorld` in vs_proxy mirrors
the same transform so rasterised triangles land on the same pixels the
fragment shader's rays will trace.

## Module responsibilities

| Module | Owns |
|---|---|
| `src/webgpu/device.ts` | Adapter / device acquisition, canvas context config, resize, `device.lost` + `uncapturederror` handlers. |
| `src/webgpu/pipeline.ts` | Bg + proxy pipelines (async creation) with shared explicit bind group layout; two pre-built bind groups; `encodeScene` records the scene pass into a caller-supplied encoder. |
| `src/webgpu/postprocess.ts` | Post-process pipelines (passthrough + FXAA), canvas-sized intermediate `rgba16float` render target, post UBO (sRGB flag), bind group + resize logic, `encodePost` records the post pass into the caller's encoder. |
| `src/webgpu/mipmap.ts` | Fullscreen-triangle blit used by `photo.ts` to generate a mipmap chain after upload. Per-device pipeline + sampler cache (`WeakMap<GPUDevice,…>`). `pushErrorScope` catches invalid pipelines so a bad format isn't silently cached. |
| `src/webgpu/uniforms.ts` | `FrameParams` type + buffer writer. Module-scope `Float32Array` scratch. |
| `src/webgpu/history.ts` | Ping-pong `rgba16float` texture pair. Recreated on resize. |
| `src/webgpu/perf.ts` | GPU timestamp-query harness (ping-ponged readback) when the adapter supports it — Tweakpane **GPU ms** by default. `?perf` on the URL adds `window._perf` sample logging. Scene pass only; FXAA post pass isn't in the HUD. |
| `src/photo.ts` | Picsum fetch → `ImageBitmap` → GPU texture with mipmaps. Gradient fallback on fetch/decode failure (GPU-upload errors are let through to `uncapturederror` instead). `destroyPhoto` for the queue-drained cleanup path in `main.ts`. |
| `src/htmlBgTexture.ts` | Chrome HTML-in-canvas support checks and `GPUQueue.copyElementImageToTexture` upload into a GPU texture. `main.ts` falls back to Picsum if the API is missing or repeated paint copies fail. |
| `src/pills.ts` | Pill state (mutated by drag) + pointer-event lifecycle with a discriminated-union drag state. |
| `src/ui.ts` | Tweakpane bindings for `Params`, presets, material buttons, support-gated Background controls, and shape/preset-driven instance-count sync. |
| `src/main.ts` | Wires everything, runs the RAF loop inside a `try/catch`, owns reload-race protection via `photoRevision`, HTML-background fallback, and shape-aware instance count (`diamond` = 1, others = 4). |
| `src/math/{cauchy,wyman,srgb,sdfPill,sdfPrism,sdfCube,camera,cube,plate,diamond,diamondExit}.ts` | Pure functions mirrored by the WGSL of the same name. The vitest suite is the reference. `cube.ts` / `plate.ts` / `diamond.ts` precompute the tumble rotations (rz·rx for cube, rx·ry for plate, Rx·Ry for diamond) on the host so the shader avoids per-SDF-eval cos/sin. `diamond.ts` also generates a WGSL `const` block containing its Tolkowsky-derived facet plane coefficients AND the unfolded normal arrays the analytical exit iterates over — single source of truth. `diamondExit.ts` mirrors the ray-polytope analytical exit so its behaviour can be pinned by a vitest regression without GPU access. |
| `src/hdr.ts` | Radiance .hdr (RGBE) decoder + round-trip encoder. Pure JS, no GPU dependency — tested via synthetic encode→decode round-trips. Supports the adaptive-RLE format Poly Haven ships (width 8-32767); legacy per-pixel RLE throws with a clear message. |
| `src/envmap.ts`, `src/envmapList.ts` | HDR environment panorama loader. `envmap.ts` fetches from a URL, decodes via `hdr.ts`, converts RGB float → RGBA half-float (with an `F16_MAX_FINITE = 65504` clamp to stop bright HDR pixels from overflowing into +Inf and seeding NaN through the linear sampler), uploads to an rgba16f texture for linear-filtered IBL sampling. `envmapList.ts` curates Poly Haven CC0 HDRIs across studio / indoor / outdoor / sunset / night categories at 1K / 2K / 4K resolution and exposes a `pickRandomSlug` helper for the UI Random button. |
| `src/shaders/dispersion/*.wgsl` | Split shader bundle (see `pipeline.ts` concat order): `frame` (uniforms + envmap sample), `sdf_primitives`, `scene` (sceneSdf aggregate), `trace` (sphere trace + analytic exits), `spectral`, `proxy` (`vs_proxy`), `fragment` (`fs_bg` / `fs_main`). Still one GPU module and one `fs_main` entry — physical split is for maintainability. |
| `src/shaders/diamond.wgsl` | Diamond-specific geometry: sdfDiamond, `diamondAnalyticExit`, `diamondAnalyticHit` / `diamondAnalyticHitScene` (front-hit picker across instances), diamondProxyVertex, facet debug helpers. Concatenated between `sdf_primitives` and `scene` so `sceneSdf` can call `sdfDiamond`. |
| `src/shaders/postprocess.wgsl` | Passthrough + FXAA fragment shaders. FXAA runs in perceptual (sRGB) luma for edge detection and blends color in linear space. Applies the sRGB OETF when the swapchain is non-sRGB. |
| `src/persistence.ts` | localStorage read/write with schema versioning, field validation, legacy `taa: boolean` → `aaMode` migration, and a trailing-edge debounced saver (+ `flush()` for pagehide). |

## Uniform layout

Mirrors the WGSL `Frame` struct exactly (std140-ish rules):

```
offset   0 │ resolution.xy,  photoSize.xy                        (16 B)
offset  16 │ n_d, V_d, sampleCount, refractionStrength           (16 B)
offset  32 │ jitter, refractionMode, pillCount, applySrgbOetf    (16 B)
offset  48 │ shape, time, historyBlend, heroLambda               (16 B)
offset  64 │ cameraZ, projection, debugProxy, taaEnabled         (16 B)
offset  80 │ cubeRot:        mat3x3<f32>                         (48 B)
offset 128 │ cubeRotPrev:    mat3x3<f32>                         (48 B)
offset 176 │ plateRot:       mat3x3<f32>                         (48 B)
offset 224 │ plateRotPrev:   mat3x3<f32>                         (48 B)
offset 272 │ diamondRot:     mat3x3<f32>                         (48 B)
offset 320 │ diamondRotPrev: mat3x3<f32>                         (48 B)
offset 368 │ waveAmp, waveFreq, waveLipFactor, sceneTime         (16 B)
offset 384 │ diamondSize, diamondWireframe, diamondFacetColor,   (32 B)
           │   diamondTirDebug, diamondTirMaxBounces, _pad×3
offset 416 │ envmapExposure, envmapRotation, envmapEnabled,      (16 B)
           │   _envmapPad
offset 432 │ pills[0..8]     each pill is:                       (32 B each)
           │   center.xyz, edgeR,   halfSize.xyz, _pad
```

Total 688 bytes (80 B head + 6 × 48 B rotation matrices + 16 B plate
wave/scene-time block + **32 B** diamond params block + 16 B envmap params
block + 8 × 32 B pills).
Uniform size is fixed — pills beyond `pillCount` are zeros.

- `shape` selects the SDF (0=pill, 1=prism, 2=cube, 3=plate, 4=diamond).
- `pillCount` gates the instance loop. UI shape changes and preset clicks keep
  it exact: `diamond` uses one instance, all other scene presets use four.
- `time` is the noise stream — wall-clock seconds, always advancing so TAA
  jitter and wavelength stratification keep decorrelating across frames
  even while the scene is paused.
- `sceneTime` is the motion stream — accumulated from per-frame `dt` and
  skipped whenever `params.paused` is true. Drives the rotation matrices
  and the plate's wave phase. Decoupled from `time` so "Stop the world"
  freezes motion without freezing AA convergence.
- `cubeRot` / `plateRot` / `diamondRot` are the current frame's rotations
  (rz·rx for cube, rx·ry for plate, Rx·Ry for diamond with a fixed 20°
  forward tilt), precomputed on the host from `sceneTime` so the shader
  does one mat-vec instead of multiple cos/sin in every SDF evaluation.
  `cubeRotPrev` / `plateRotPrev` / `diamondRotPrev` are the same matrices
  computed from the previous frame's `sceneTime` — feed `reprojectHit`
  for TAA motion-vector history reads. When paused they equal the
  current matrices, so the reprojection collapses to identity and history
  reads land on the pixel centre (no iterated bilinear blur). For the
  diamond's fixed view presets, `uniforms.ts` writes the same canonical
  pose matrix into BOTH `diamondRot` and `diamondRotPrev`, producing a
  zero motion vector so the frozen shape doesn't smear under TAA.
- `diamondSize` is the girdle diameter in pixels (slider in `ui.ts`).
  `diamondWireframe` / `diamondFacetColor` / `diamondTirDebug` are debug
  flags; `diamondTirMaxBounces` (1…32, default 6) caps the exact-mode TIR
  bounce loop. TIR debug tints unresolved pixels hot pink (bounce budget
  exhausted, refract out still TIR) or orange (`diamondAnalyticExit` miss).
  All are ignored when `shape != diamond`.
- `taaEnabled` toggles temporal antialiasing. When on, `fs_main` jitters
  the primary ray by a per-pixel hash within ±0.5 px, and reads history at
  `fragCoord + (projected_prev_world − projected_curr_world)` — jitter
  cancels in the delta so static scenes read pixel-aligned history while
  moving shapes keep refracted texture sharp. Driven by `aaMode === 'taa'`
  in `Params` (see `src/ui.ts`); `aaMode === 'fxaa'` runs FXAA in the post
  pass instead, and `aaMode === 'none'` does neither.
- `waveAmp` / `waveFreq` drive the plate's midsurface displacement
  (`waveAmp · sin(waveFreq · x + 2·sceneTime) · sin(waveFreq · y +
  2·sceneTime)`). Ignored for other shapes.
- `waveLipFactor = 1/√(1 + (waveAmp·waveFreq)²)` is the precomputed
  Lipschitz safety factor `sdfWavyPlate` multiplies into its output, so
  sphere-trace steps stay inside the true distance without running
  `inverseSqrt` on every SDF eval. The bound is tight because the wave's
  two partial derivatives never reach `amp·k` simultaneously — see the
  derivation in `src/webgpu/uniforms.ts`. Defaults give ≈ 0.92 vs the
  older hardcoded 0.6 — ~53 % more progress per step at the same safety
  margin.
- `historyBlend` defaults to the **History α** slider (currently `0.5` in
  `defaultParams()` and every preset; user-tunable in the Misc folder) for
  steady state, and 1.0 for one frame after a scene change (preset click,
  photo reload, shape switch, pill shuffle, pause toggle) so stale temporal
  history doesn't ghost in. Switches to
  progressive averaging `α = max(1/n, 1/256)` while "Stop the world" is
  on — noise drops as 1/√n in the convergence ramp and bottoms out at a
  256-sample sliding window (~6 % residual). The 1/256 floor is required
  by the `rgba16float` history texture: a smaller α would push the new-
  sample contribution below the fp16 quantum (≈ 0.0005 around mid-grey)
  for high-contrast edge pixels, the contribution would round to 0, and
  `(1 − α) · prev` decay would fade silhouettes to a black line over
  several minutes of pause. See `main.ts pausedFrames` for the full
  derivation.
- `heroLambda` is a frame-jittered wavelength in [380, 700] — used by Approx
  mode for the one shared back-face trace. `spectralSamplingFields()` pins it
  to 540 nm and writes `jitter = -1` when **Temporal jitter** is off; the WGSL
  treats negative jitter as "use the centre of each spectral stratum" so the
  toggle produces a stable, visible A/B state.
- `cameraZ` / `projection` drive ortho vs perspective (CPU derives `cameraZ`
  from the UI's FOV and canvas height).
- `debugProxy` tints every proxy fragment pink for visual inspection.
- `applySrgbOetf` is kept in the UBO slot for layout parity only — the
  scene shader no longer reads it, because the post pass owns all sRGB
  encoding. Host still writes 0/1 into the slot to keep
  `tests/uniformsLayout.test.ts` happy; reclaiming the slot means
  touching that test + `uniforms.ts` + the WGSL struct together.

## Post-process pass (AA / sRGB OETF)

The scene pass writes linear RGB into a canvas-sized `rgba16float`
intermediate. A second render pass then reads that intermediate and
writes the swapchain:

- `aaMode === 'none'` → `fs_passthrough` copies and applies sRGB OETF
  (identity if the swapchain is already `*-srgb`).
- `aaMode === 'fxaa'` → `fs_fxaa` runs a 9-tap FXAA 3.x-style spatial
  filter (sRGB-space luma for edge detection, linear blend for color),
  then the same sRGB encode.
- `aaMode === 'taa'` → `fs_passthrough` again; TAA already ran in the
  scene pass (sub-pixel jitter + motion-vector history reprojection),
  so post has nothing to do except encode.

Keeping both scene color targets (`@location(0)` intermediate,
`@location(1)` history) as `rgba16float` lets FXAA work on the same
linear pixels the history EMA already sees, and consolidates the sRGB
OETF in one place. The post UBO is 16 B (one `applySrgbOetf` flag + 3
padding floats); only the flag is written, at startup — it's constant
for the lifetime of the session.

`src/webgpu/postprocess.ts` owns the intermediate's lifecycle. The
texture is freed synchronously on resize because WebGPU holds a strong
reference from any command buffer already naming it as a color
attachment (photo reload drains `queue.onSubmittedWorkDone` because the
photo is sampled across frames; the intermediate is single-frame).

## Photo mipmaps

`src/photo.ts` uploads the photo with a full mip chain generated by a
fullscreen-triangle blit pipeline in `src/webgpu/mipmap.ts`. The
per-wavelength refraction sample in the fragment shader picks a LOD
based on two terms:

- **Grazing incidence** — `-log2(cosT) - 1` — sample footprint grows as
  `~1/cosT` at grazing angles. cosT is clamped to ≥ 0.02 inside the log
  so the term stays bounded at extreme angles.
- **Rounded-rim curvature** — `(1 - max(|nLocal|)) · 8` — on the rounded
  edges of cube / plate the front normal rotates ~90° across an
  `edgeR`-wide screen region, blowing up the refracted-UV Jacobian
  independently of `cosT`. Pill / prism skip this term.

The sum of the two terms is clamped to `[0, 6]`. The individual terms
are not floored at 0; a near-head-on pixel gives a negative grazing
contribution that cancels part of the curvature boost, and the final
clamp catches the rest. Trilinear filtering on the sampler
(`mipmapFilter: 'linear'`) gives the sub-level blending. Background
samples (bg + reflection fallback) stay at LOD 0 because they're
1:1 with screen pixels.

## Why per-wavelength sRGB weighting?

The textbook path — accumulate `cmf(λ) * L(λ)` into XYZ, then one
`XYZ → sRGB` — collapses if `L(λ)` is a scalar derived from a photo's
luminance: you lose all chroma, and the only color left is whatever
`xyzToSrgb(sum(cmf))` produces (a slight salmon tint for a flat-white
spectrum, because the CMF sum isn't D65).

Per-wavelength weighting — `xyzToSrgb(cmf(λ)) * L_rgb(uv_λ)` — gives each
wavelength its own sRGB primary color, and preserves photo RGB in the
flat-UV case:

- Uniform input (same UV for all λ): `L * sum(lambdaRgb) / sum(lambdaRgb) = L`.
  Preserves chroma exactly.
- Varying input (different UV per λ): red-wavelength samples contribute to
  the R channel, blue to B, classic chromatic aberration.

The normalization denominator is the same per-wavelength primary-sum, which
keeps the output neutral for any `N`.

## SDFs and sphere tracing

Five shapes. `sceneSdf` dispatches on the `shape` uniform:

- **Pill** — 2D stadium silhouette in XY (`roundedBox` shrunk by `edgeR`, then
  rounded by the shortest shrunk half-axis), extruded into Z with the same
  rounded-corner trick on `|z|`.
- **Prism** — isosceles triangle cross-section in **YZ** (apex at +Z, base at
  −Z), extruded along X. `halfSize.x` is the extrusion length, `halfSize.y`
  the base half-width, `halfSize.z` the apex height. From top-down the
  silhouette is a rectangle; the triangle's slanted YZ faces bend rays
  laterally, producing the classic prism rainbow at contrast edges in the
  photo.
- **Cube** — standard rounded box. `local = frame.cubeRot * (p - center)`.
  `cubeRot` is the rz·rx rotation tumbling around X+Z at 0.31 + 0.20 rad/s,
  computed on the host from `time` and uploaded as a uniform so every SDF
  evaluation is just one mat-vec.
- **Plate** — thick square slab as a **constant-thickness bent sheet**. In
  the plate's local (rotated) frame, define a midsurface `z* = waveAmp ·
  sin(waveFreq · x + 2t) · sin(waveFreq · y + 2t)`; both Z faces ride that
  midsurface together so the slab thickness stays uniform everywhere (vs
  the naïve `sdBox - wave` which pulses the thickness). The SDF is
  `sdBox(p with z ← p.z − z*) × waveLipFactor` — the Lipschitz factor
  corrects for the extra gradient the z-shift adds along x/y. Rotation is
  precomputed on the host as `plateRot` (rx·ry at 0.30 + 0.20 rad/s, rate
  ratio chosen as coprime small integers (2:3) so the combined orientation
  takes ~63 s to repeat — long enough to read as non-looping in practice;
  see `src/math/plate.ts` for the explicit period derivation).
- **Diamond** — round brilliant cut as a convex polytope pinned to
  **Tolkowsky's 1919 "ideal" proportions**: 53 % table (vertex-to-vertex,
  GIA "bezel point" convention), 34.5° crown, 40.75° pavilion, 22° star,
  39.9° upper half, 42° lower half, 2 % girdle thickness. 58 facets
  collapse to 7 distance terms via D_8 (octagonal) symmetry folding, then
  `max(...)` of the half-space distances gives the SDF. The fold reduces
  the 8-fold azimuthal repeat to a π/8 wedge with 3 mirror reflections
  (abs on x, abs on y, then reflect across the y = x · tan(π/8) line).
  Plane coefficients are derived in `src/math/diamond.ts` and injected
  into the shader as `const` declarations so host math and GPU constants
  can't drift. The upper-half angle is pinned at 39.9° (not the 42° that
  physical cut specs usually quote) because with the plane anchored on
  the actual girdle rim at φ = 0, the bezel-star-UH three-way junction
  must sit inside the wedge — above 40° it escapes the π/8 mirror and
  the bezel kite stops closing at its corner; `diamond.test.ts` pins
  both the corner-passage invariant and the 40° ceiling. Rotation is
  `Rx·Ry` (fixed 20° forward tilt + vertical Y-axis spin) precomputed
  as `diamondRot` on the host — same uniform pattern as cube/plate.
  Four view presets (`free` / `top` / `side` / `bottom`, bound to the
  `T` / `S` / `B` / `F` hotkeys) swap `diamondRot` with a canonical pose
  for cross-checking facet geometry against a reference illustration.
  Two debug overlays — `diamondWireframe` (facet-edge draw via a
  top-two-plane-gap smoothstep) and `diamondFacetColor` (flat-shade per
  facet class: table=red, bezel=green, star=blue, upper-half=yellow,
  girdle=cyan, lower-half=magenta, pavilion=orange) — surface coverage
  + adjacency without refraction muddying the signal. SDF code lives in
  `src/shaders/diamond.wgsl` alongside `diamondAnalyticHit` /
  `diamondAnalyticHitScene` (analytical front-hit picker across diamond
  instances) and the `diamondProxyVertex` exact convex-hull proxy mesh
  (46 triangles: 6-tri table fan + 16-tri crown trapezoids + 16-tri
  girdle band + 8-tri pavilion cone).
  Back-face exit is **analytical** (`diamondAnalyticExit`, also in
  `src/shaders/diamond.wgsl`): ray tested against all 57 unfolded facet
  planes (8 bezel + 8 star + 16 upper half + 16 lower half + 8 pavilion
  + 1 table cap) plus the girdle cylinder, minimum positive t wins. The
  exact facet normal that comes out sidesteps the finite-diff gradient
  degeneracy at facet edges that previously sent TIR pixels to the
  external-reflection fallback and produced "other faces suddenly
  appearing" artifacts during tumble.   On TIR the diamond path runs a **bounded internal bounce loop** (count from
  `diamondTirMaxBounces`, with a small origin nudge and miss guards) so
  brilliant-cut sparkle uses meaningful paths. The chain is exact-mode
  only — approx mode’s shared hero-wavelength exit would flicker with
  `heroLambda` jitter, so approx keeps Phase A’s `reflSrc` TIR fallback.
  Runtime shape/preset sync intentionally renders a single diamond instance;
  pill/prism/cube/plate keep the four-instance layout.

Sphere trace starts from a per-pixel ray origin and direction (see Camera
above), marches with `HIT_EPS = 0.25` and `MIN_STEP = 0.5`. For pill and
prism, the back-surface exit comes from marching `-sceneSdf` (inside-trace),
capped at `maxInternalPath()` — the longest possible chord through any live
pill. **Cube**, **plate**, and **diamond** take a shortcut:

- `cubeAnalyticExit` — ray-box slab intersection in the cube's rotated
  local frame, followed by 2 Newton steps on the rounded-box SDF to snap
  onto the rounded rim. Replaces the per-wavelength 48-iter inside-trace
  and the 6-eval finite-diff normal with an O(1) closed form. Roughly 7–8×
  faster for the cube case.
- `plateAnalyticExit` — ray-box slab intersection in the plate's rotated
  local frame (XYZ, picking the earliest-exit axis), then 3 Newton steps
  refining t against the true wavy surface `z = faceSign · halfZ + z*(x,y)`
  if the exit is through a Z face. X/Y face exits are flat (no refinement).
  Normal on a wavy Z face is `(−faceSign · ∂z*/∂x, −faceSign · ∂z*/∂y,
  faceSign)` normalized, using cos(kx+2t) / cos(ky+2t) recovered from the
  final Newton step. Same ~10× per-wavelength SDF-eval reduction as cube.
- `diamondAnalyticExit` — ray tested against all 57 unfolded facet planes
  (8 bezel + 8 star + 16 UH + 16 LH + 8 pavilion + 1 table cap) plus the
  girdle cylinder (quadratic in t), minimum positive t wins. Unlike cube
  and plate there's no Newton refinement — the polytope is piecewise
  planar, so the first-hit plane equation IS the exact exit. Returns the
  plane's exact outward normal, which resolves the facet-edge finite-diff
  degeneracy that previously sent TIR pixels through the front-reflection
  hack. JS mirror in `src/math/diamondExit.ts` pins the behaviour.

Normals come from central differences on the scene SDF for pill/prism — six
extra SDF evaluations per shaded pixel (one pair per axis), cheap. Cube and
plate use analytical gradients at the exit point (matching what the
finite-diff would converge to), so the rounded rim / wavy surface keeps its
soft refraction.

### Per-wavelength loop

```
for i in 0..N:
  pxJit  = hash21(pixel ⊕ time) - 0.5      // signed [-0.5, 0.5) — see note below
  λ      = mix(380, 700, (i + 0.5 + pxJit) / N)
  ior    = cauchyIor(λ, n_d, V_d)
  r1     = refract(-z, nFront, 1/ior)
  // Entry biased one MIN_STEP inward along r1 — `h.p` lives ON the
  // surface (within HIT_EPS), and the analytic exits' slab math
  // (h - roL) / rdL would compute 0/0 = NaN at exact-boundary entries.
  roEntry = h.p + r1 * MIN_STEP
  if approx:       (pExit, nBack) ← shared back-face trace at heroLambda
  else if cube:    (pExit, nBack) ← cubeAnalyticExit(roEntry, r1, analyticIdx)
  else if plate:   (pExit, nBack) ← plateAnalyticExit(roEntry, r1, analyticIdx)
  else if diamond: (pExit, nBack) ← diamondAnalyticExit(roEntry, r1, analyticIdx)
  else:            pExit = insideTrace(h.p, r1, internalMax), nBack = -sceneNormal(pExit)
  r2     = refract(r1, nBack, ior)
  // Failure-mode routing — see "Failure-mode fallbacks" below.
  L      = TIR && diamond && !approx ? bounce_chain(r1, nBack, pExit; cap=diamondTirMaxBounces) :
           TIR                        ? reflSrc :
           NaN || OOB                 ? bg      :
                                        photo[uv_with_offset(r2)]
  F_λ    = schlickFresnel(cosT, ior)  // per-wavelength Fresnel
  accum += mix(L, reflSrc, F_λ) * xyzToSrgb(cieXyz(λ))
```

Per-pixel stratification (`hash21`) means neighbouring pixels pick different
wavelengths — the eye and temporal accumulation average the spatial noise,
so N=8 stratified looks like N=16 uniform.

The `- 0.5` on `pxJit` centres the jitter on each stratum so `t = (i + 0.5 +
pxJit)/N` stays inside `[i/N, (i+1)/N)`. Without that shift the last stratum
(i = N-1) can overflow past `t = 1` into λ > 700 nm where the CIE matching
functions are effectively zero, so ~30 % of pixels at N=3 lose their red
sample and the renormaliser flips a flat-white background to yellow. The
same off-by-half-stratum also existed at larger N (20 nm overflow at N=8,
10 nm at N=16), but the artefact was masked by history accumulation.

### Failure-mode fallbacks

The wavelength loop has three distinct failure modes for the back-face
sample (TIR, NaN, out-of-bounds UV). TIR splits into two rows (diamond
exact-mode bounce chain vs everything else falling back to reflSrc), so
the table below lists four TIR/NaN/OOB rows in total:

| Failure | Detection | Fallback | Why |
|---|---|---|---|
| Real TIR (non-diamond) | `dot(r2, r2) < 1e-4` AND not NaN | `reflSrc` | Physically correct — the wavelength is fully reflected by the front face |
| Real TIR (diamond, exact mode) | `shape == diamond` AND `!useHero` | Bounce loop up to `diamondTirMaxBounces` (default 6, max 32): each step reflects, calls `diamondAnalyticExit` with a nudged origin, tries `refract` out. On success, sample photo/envmap. If the chain exhausts, blend to `bg` or envmap `reflSrc` (not the old facet-unrelated `reflSrc` stand-in for that path). `diamondTirDebug` tints exhausted pixels pink (still TIR) or orange (analytic exit miss). |
| Real TIR (diamond, approx mode) | `shape == diamond` AND `useHero` | `reflSrc` (Phase A fallback) | Approx mode shares hero's exit across λ. Running the bounce chain from that shared origin while heroLambda jitters frame-to-frame produces TIR-boundary flicker; we stick with reflSrc here until a per-λ solution (Phase C) lands. |
| NaN r2 | `r2dot != r2dot` (self-comparison catches NaN) | `bg` (local pixel's photo sample) | Renders the same colour as miss-path neighbours so the silhouette stays clean instead of showing a single bright reflection sample |
| UV out of bounds | `coverUv(uvOff)` not in [0, 1]² | `bg` | The mirror-repeat sampler would otherwise fold a wildly-off UV back to an unrelated photo region (bright photo → white speckle, dark → black) |

The front-face / pre-loop bg fallback gate is similar but covers
sphere-trace miss, degenerate-gradient `sceneNormal`, and plate
face-junction creases (`plateCreaseAt`) — all routed to bg for the
same "matches surrounding bg / silhouette neighbours" reason.

reflSrc itself also gets the OOB check (the reflected UV `h.p.xy +
refl.xy * 0.2` can land outside the photo) and falls back to bg when
the reflected sample would otherwise mirror-repeat to garbage.

### Approx (hero wavelength) mode

`refractionMode == approx` does one insideTrace per frame at
`frame.heroLambda` and shares that exit point/normal across all N
wavelengths. Temporal accumulation averages out the per-frame error. Minor
speedup on TBDR (~10-15 % at high N) since texture bandwidth dominates on
Apple Silicon.

## Error handling

- `device.lost` + `uncapturederror`: logged + shown via `#fallback`.
- Shader compile errors: `getCompilationInfo()` logs all messages and throws
  on error (default WebGPU swallows these into opaque validation errors).
- Photo fetch failure: graceful fallback to a bundled gradient texture (not
  an exception).
- Render-loop exception: `try/catch` surfaces the error and stops the loop
  instead of freezing silently.
- Reload race: a monotonic `photoRevision` counter discards stale async
  results; old textures are destroyed only after `queue.onSubmittedWorkDone`.
- Typing-target hotkey filter: pointer events in Tweakpane number inputs
  don't fire `Space`/`Z`/`R`.

## Testing

Math modules and uniform wiring are unit-tested (currently 200 tests, all pass — exact
count drifts with each new case, see `bun run test`):

- `cauchyIor` at d-line, monotonicity, `V_d` sensitivity, 1.0 clamp.
- `cieXyz` Y-peak near 555 nm, red dominance at 650 nm, blue at 450 nm, near-zero at UV/IR.
- `xyzToLinearSrgb` D65 white, Y-only luminance-biased gray.
- `linearToGamma` identity endpoints, linear segment, power-curve segment.
- `sdfPill3d` sign, symmetry, top-face zero-crossing, rounded-edge smoothness.
- `sdfPrism` interior sign, far-field positivity, apex/base edge values, both mirror symmetries, apex narrowing.
- `sdfCube` interior, far-field, face zero-crossings, symmetry, rounded-corner smoothness.
- `cameraZForFov` at 60°/90°, monotonicity with FOV, linearity with height, slider bounds.
- `cubeRotationColumns` identity at t=0, orthonormality, matches the original rz·rx derivation, WGSL padded layout, pad slots zero, rejects non-finite time. `plateRotationColumns` / `diamondRotationColumns` follow the same invariants for their rx·ry and Rx·Ry compositions respectively.
- Diamond geometry — Tolkowsky-constant sanity (crown / pavilion height ratios, total height 0.55–0.65), facet-plane normals unit-length, bezel/pavilion Y component zero (φ=0 axis), upper+lower half pass through BOTH shared girdle-rim corners at φ=0 and φ=π/8 (catches anchor regression to the circumscribing-octagon rim), UH tilt stays below the 40° wedge-validity ceiling, LH tilt stays above the pavilion angle so it surfaces.
- Diamond unfolded plane arrays (Phase B) — counts per class (8 / 8 / 16 / 16 / 8), first entry matches the fundamental-wedge plane so SDF and analytical-exit stay consistent, normals unit-length + shared tilt within a class, consecutive normals differ by the expected rotation step (π/4 or π/8), every plane passes through its designated shared girdle corners (pins the "facets meet at shared corners" invariant through the analytical path as well).
- Diamond analytical exit (`diamondAnalyticExit` JS mirror) — axis-aligned rays exit through the culet / table with the expected class and normal, horizontal rays at z=0 exit through the girdle cylinder, rays slightly above/below the girdle band exit through a crown/pavilion facet (band-rejection test), exit normal always satisfies `dot(n, rd) > 0` (ray leaves the half-space), exit point satisfies the reported facet's plane equation (consistency between which-class and what-point), vertical rays still find the table cap when the cylinder's `a > 1e-6` guard fires.
- Diamond view presets — `top` preserves local +Z as world +Z, `side` rotates local +Z to world +Y, `bottom` rotates it to world -Z; all three preserve vector length (orthonormal).
- `uniform layout drift detector` parses the WGSL `struct Frame` declaration and pins the field set + order (including `diamondRot` / `diamondRotPrev` / the diamond 32B params block) so anyone editing it gets nudged to update `src/webgpu/uniforms.ts` too.
- Spectral sampling fields — temporal jitter on writes random jitter + hero wavelength; off writes the negative-jitter sentinel and fixed 540 nm hero wavelength used by WGSL.

WGSL versions are hand-mirrored by the corresponding TS module; the TS tests
act as the reference. Shader correctness beyond that is verified visually —
no automated GPU tests.

## Performance

Measured on Apple Silicon (Metal 3) via WebGPU `timestamp-query` (`?perf=1`
URL flag exposes `window._perf.samples`). Numbers below are p50 over ≥ 30
samples at 1292×1073 with 4 instances on screen (diamond preset intentionally uses 1):

| Config | GPU time |
|---|---:|
| pill N=8  | 1.70 ms |
| pill N=32 | 6.42 ms |
| cube N=8  | 1.05 ms |
| cube N=16 | 1.38 ms |
| cube N=32 | 1.97 ms |
| cube N=64 | 3.21 ms |

Cube is noticeably cheaper than pill at matching `N` because its back-face
exit is `cubeAnalyticExit` — a single analytical slab intersection plus 2
Newton refinement steps on the rounded-box SDF — instead of the per-λ
sphere-traced `insideTrace` + finite-diff normal the pill/prism path still
runs for every wavelength. On the earlier sphere-traced cube, N=8 took
~4 ms and N=32 took ~9 ms, so the analytic path is ~4-5× faster for the
same sample count. Plate follows the same trick (`plateAnalyticExit`,
~3 Newton iterations for the wavy Z face), ending up in the cube/plate
band rather than the pill/prism band even though its surface is curved.

All configurations hold 60 fps with zero dropped frames. On TBDR hardware
(Apple M-series) background pixels are already efficiently culled; the proxy
pass mostly helps by emitting zero heavy fragments outside the shape
silhouette. Discrete (non-TBDR) GPUs see a larger relative win.

### Cost breakdown

Pill / prism pixel (N=8):
- Up to 64 sphereTrace SDF evals (front trace) + 8 × up to 48 inside-trace
  evals + 6 normal evals + 8 × 6 back-normal evals — dominated by the per-λ
  back-trace.
- 8 photo texture taps + 1 reflection tap + 1 history tap.
- 8 Cauchy + Wyman CIE evaluations + 8 per-λ Fresnel mixes.
- Usually texture-bandwidth bound on Apple Silicon.

Cube pixel (N=8):
- Same 64 sphereTrace + 6 front-normal evals as above, but the per-λ back
  trace is `cubeAnalyticExit` — O(1) slab intersection + 2 Newton refinement
  steps on the rounded-box SDF to snap the exit onto the rounded rim. The
  8 × (48 + 6) inside-trace + back-normal evals from the pill/prism path
  collapse to ~40 ALU ops per wavelength.
- Texture taps + spectral math identical to pill/prism.

Plate pixel (N=8):
- Front sphere-trace runs ~35 % fewer steps than pill/prism at matching
  size because `waveLipFactor` ≈ 0.92 at default wave params (vs the
  conservative 0.6 we had before) lets each step consume more true distance
  while still staying inside it.
- Per-λ back trace is `plateAnalyticExit`: slab pick + 3 Newton iterations
  against the wavy Z face (2 cos + 2 sin per iteration) + analytical
  gradient normal. Roughly the same cost envelope as cube's analytic path.
- Texture taps + spectral math identical to pill/prism.
