# Architecture

Two-pass WebGPU renderer. Per-pixel SDF sphere-tracing inside a per-pill 3D
proxy mesh, with a cheap fullscreen background pass underneath — so the heavy
refraction shader only runs on fragments the proxy actually covers.

## Frame path

```
┌──────────────────────────────────────────────────────────────────────┐
│  every RequestAnimationFrame:                                        │
│                                                                      │
│  1. resize canvas + history if needed                                │
│  2. push params → pills (hx/hy/hz/edgeR)                             │
│  3. writeFrame → uniform buffer (384 B: scalars + mat3 + pills) │
│  4. draw pass:                                                       │
│     a. bg sub-pass: fullscreen triangle → fs_bg (photo + history)    │
│     b. proxy sub-pass: instanced 3D cube mesh → fs_main              │
│          per-fragment camera ray (ortho OR perspective)              │
│          sphere-trace scene SDF                                      │
│          if miss: return bg (over-covered proxy fragment)            │
│          if hit:  per-pixel stratified λ jitter, for each λ:         │
│                    refract → inside-trace → refract out              │
│                    sample photo at uv_λ (TIR → reflect)              │
│                    per-wavelength Fresnel mix                        │
│                    accumulate weighted by xyzToSrgb(cmf(λ))          │
│                  EMA-blend with history[read] (historyBlend)         │
│                  write blended → swapchain (+ OETF if needed)        │
│                  write blended → history[write] (linear)             │
│  5. flip history.current                                             │
└──────────────────────────────────────────────────────────────────────┘
```

Two pipelines share one explicit bind group layout (so `frame` is visible to
both vertex and fragment stages). Two bind groups are pre-built at pipeline
creation (one per history-read slot) and swapped based on `history.current` —
no per-frame bind group allocation.

### Proxy mesh

`vs_proxy` emits a 36-vertex unit cube (12 tris, CCW-outward winding) per
pill, scaled to `halfSize + edgeR`. For `shape == cube` the corners are
transformed by `transpose(rot)` so the rasterised silhouette matches the
shader's world-space cube exactly — no √3 bounding-box slack. Pill and prism
are axis-aligned in world space, so their AABB cube proxy already covers the
shape tightly. Back-face culling (`frontFace: 'cw'` because our Y-flipped
projection inverts winding from 3D to NDC) leaves one fragment invocation per
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
| `src/webgpu/pipeline.ts` | Bg + proxy pipelines (async creation) with shared explicit bind group layout; two pre-built bind groups; draw submission. |
| `src/webgpu/uniforms.ts` | `FrameParams` type + buffer writer. Module-scope `Float32Array` scratch. |
| `src/webgpu/history.ts` | Ping-pong `rgba16float` texture pair. Recreated on resize. |
| `src/webgpu/perf.ts` | Optional GPU timestamp-query harness (ping-ponged readback). Enabled via `?perf=1`, surfaces timings on `window._perf`. |
| `src/photo.ts` | Picsum fetch → `ImageBitmap` → GPU texture. Gradient fallback on failure. `destroyPhoto` for the queue-drained cleanup path in `main.ts`. |
| `src/pills.ts` | Pill state (mutated by drag) + pointer-event lifecycle with a discriminated-union drag state. |
| `src/ui.ts` | Tweakpane bindings for `Params`. |
| `src/main.ts` | Wires everything, runs the RAF loop inside a `try/catch`, owns reload-race protection via `photoRevision`. |
| `src/math/{cauchy,wyman,srgb,sdfPill,sdfPrism,sdfCube}.ts` | Pure functions mirrored by the WGSL of the same name. The 31 vitest tests are the reference. |
| `src/shaders/dispersion.wgsl` | Everything visible: SDFs (pill/prism/cube + rotation), sphere-trace, Cauchy, CIE, sRGB, Fresnel, OETF, spectral accumulation, TIR fallback. |
| `src/persistence.ts` | localStorage read/write with schema versioning, field validation, and a trailing-edge debounced saver (+ `flush()` for pagehide). |

## Uniform layout

Mirrors the WGSL `Frame` struct exactly (std140-ish rules):

```
offset   0 │ resolution.xy,  photoSize.xy                        (16 B)
offset  16 │ n_d, V_d, sampleCount, refractionStrength           (16 B)
offset  32 │ jitter, refractionMode, pillCount, applySrgbOetf    (16 B)
offset  48 │ shape, time, historyBlend, heroLambda               (16 B)
offset  64 │ cameraZ, projection, debugProxy, _pad0              (16 B)
offset  80 │ cubeRot: mat3x3<f32>  (3 × 16 B padded columns)     (48 B)
offset 128 │ pills[0..8]   each pill is:                         (32 B each)
           │   center.xyz, edgeR,   halfSize.xyz, _pad
```

Total 384 bytes (80 B head + 48 B cubeRot + 8 × 32 B pills). Uniform size is
fixed — pills beyond `pillCount` are zeros.

- `shape` selects the SDF (0=pill, 1=prism, 2=cube).
- `time` is used by the GPU as the per-frame jitter hash seed; the host
  separately derives `cubeRot` from the same value (see `src/math/cube.ts`).
- `cubeRot` is the cube's rz·rx rotation, precomputed on the host once per
  frame so the shader does one mat-vec instead of four cos/sin in every SDF
  evaluation.
- `historyBlend` is 0.2 in steady state and 1.0 for one frame after a scene
  change (preset click, photo reload, shape switch, pill shuffle) so stale
  temporal history doesn't ghost in.
- `heroLambda` is a frame-jittered wavelength in [380, 700] — used by Approx
  mode for the one shared back-face trace.
- `cameraZ` / `projection` drive ortho vs perspective (CPU derives `cameraZ`
  from the UI's FOV and canvas height).
- `debugProxy` tints every proxy fragment pink for visual inspection.

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

Three shapes, all rounded (edgeR) extrusions. `sceneSdf` dispatches on the
`shape` uniform:

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

Sphere trace starts from a per-pixel ray origin and direction (see Camera
above), marches with `HIT_EPS = 0.25` and `MIN_STEP = 0.5`. For pill and
prism, the back-surface exit comes from marching `-sceneSdf` (inside-trace),
capped at `maxInternalPath()` — the longest possible chord through any live
pill. **Cube** takes a shortcut: `cubeAnalyticExit` solves ray-box slab
intersection in the cube's rotated local frame, which replaces the per-
wavelength 48-iter inside-trace and 6-iter finite-diff normal with an O(1)
closed-form exit point + rounded-box gradient normal. Roughly 7–8× faster
for the cube case; pill/prism are unchanged.

Normals come from central differences on the scene SDF for pill/prism — four
extra SDF evaluations per shaded pixel, cheap. Cube uses the analytical
rounded-box gradient at the exit point (same formula as the finite-diff
would converge to), so the rounded rim keeps its soft refraction.

### Per-wavelength loop

```
for i in 0..N:
  pxJit  = hash21(pixel ⊕ time)       // spatial stratification
  λ      = mix(380, 700, (i + 0.5 + pxJit) / N)
  ior    = cauchyIor(λ, n_d, V_d)
  r1     = refract(-z, nFront, 1/ior)
  if approx: (pExit, nBack) ← shared back-face trace at heroLambda
  else if cube:  (pExit, nBack) ← cubeAnalyticExit(h.p, r1, cubeIdx)  // O(1) slab + rounded-box gradient
  else:          pExit = insideTrace(h.p, r1, internalMax), nBack = -sceneNormal(pExit)
  r2     = refract(r1, nBack, ior)
  L      = TIR ? reflSrc : photo[uv_with_offset(r2)]
  F_λ    = schlickFresnel(cosT, ior)  // per-wavelength Fresnel
  accum += mix(L, reflSrc, F_λ) * xyzToSrgb(cieXyz(λ))
```

Per-pixel stratification (`hash21`) means neighbouring pixels pick different
wavelengths — the eye and temporal accumulation average the spatial noise,
so N=8 stratified looks like N=16 uniform.

### TIR fallback

When `refract()` returns a zero vector at the back face (total internal
reflection), the wavelength would otherwise drop out and leave a black hole
where every λ TIR'd. Instead the loop substitutes the external front-face
reflection sample for that wavelength, which matches the physics ("total
reflection" → sample what would reflect off the front) and keeps the spectral
weighting balanced.

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

Math modules are unit-tested (41 tests, all pass):

- `cauchyIor` at d-line, monotonicity, `V_d` sensitivity, 1.0 clamp.
- `cieXyz` Y-peak near 555 nm, red dominance at 650 nm, blue at 450 nm, near-zero at UV/IR.
- `xyzToLinearSrgb` D65 white, Y-only luminance-biased gray.
- `linearToGamma` identity endpoints, linear segment, power-curve segment.
- `sdfPill3d` sign, symmetry, top-face zero-crossing, rounded-edge smoothness.
- `sdfPrism` interior sign, far-field positivity, apex/base edge values, both mirror symmetries, apex narrowing.
- `sdfCube` interior, far-field, face zero-crossings, symmetry, rounded-corner smoothness.
- `cameraZForFov` at 60°/90°, monotonicity with FOV, linearity with height, slider bounds.
- `cubeRotationColumns` identity at t=0, orthonormality, matches the original rz·rx derivation, WGSL padded layout, pad slots zero, rejects non-finite time.

WGSL versions are hand-mirrored by the corresponding TS module; the TS tests
act as the reference. Shader correctness beyond that is verified visually —
no automated GPU tests.

## Performance

Measured on Apple Silicon (Metal 3) via WebGPU `timestamp-query` (`?perf=1`
URL flag exposes `window._perf.samples`).

After the analytical cube exit + host-side `cubeRot` uniform landed (≈ 800×760,
4 rotating cubes):

| Config | GPU time |
|---|---:|
| cube N=8  | 0.58 ms |
| cube N=32 | 1.05 ms |

That's ~7× faster at N=8 and ~8.5× faster at N=32 versus the earlier
sphere-traced cube (4.01 ms and 8.97 ms respectively). Pill and prism use
the same sphere-trace path as before — their timings are unchanged.

All configurations hold 60 fps with zero dropped frames. On TBDR hardware
(Apple M-series) background pixels are already efficiently culled; the proxy
pass mostly helps by emitting zero heavy fragments outside the cube
silhouette. Discrete (non-TBDR) GPUs see a larger relative win.

### Cost breakdown (pill pixel, N=8)

- ~48 SDF evals (front trace + per-λ inside-trace × 8).
- 8 photo texture taps + 1 reflection tap + 1 history tap.
- 8 Cauchy + Wyman CIE evaluations + 8 per-λ Fresnel mixes.
- Usually texture-bandwidth bound on Apple Silicon.
