# Real Spectral Dispersion — Liquid Glass Background (Design Spec)

**Date:** 2026-04-21
**Status:** Draft — awaiting review
**Author:** Saqoosha + Claude (Opus 4.7)

## Purpose

Build a realtime WebGPU demo that renders Apple "Liquid Glass"-style floating pill shapes with **physically accurate wavelength-dependent refraction** (spectral dispersion) over a photographic background. The demo is the proof-of-technique for a later website background; this spec covers the technique in isolation.

The deliverable must make one thing obvious on screen: the difference between the common fake "shift R/G/B IORs" hack and a real spectral integration. That A/B difference is the whole point.

## Non-Goals

- Production website integration (separate project)
- Mobile/touch UX polish
- WebGL2 fallback (this is a "latest web tech" test; browsers without WebGPU get a message)
- Loading arbitrary user photos (a Unsplash random fetch is enough)
- Animated background content (a still photo is enough to evaluate dispersion)
- Accessibility pass (reduced-motion, contrast) — out of scope for a tech demo

## Success Criteria

1. **Visually correct spectral dispersion** at pill edges: a smooth rainbow fringe with no banding at the default sample count, matching the character of real photographic chromatic aberration through glass.
2. **Runtime A/B toggle** between `N=3` (fake RGB) and `N=8` (real spectral). The difference must be unambiguous to the eye within 3 seconds.
3. **≥60fps at 1080p** on a mid-range discrete GPU (RTX 3060 class) with default settings. Stretch: `N=16` stays at 60fps.
4. **Draggable pills.** Pressing and dragging a pill moves it; refraction updates in realtime.
5. **Live parameter panel** exposes the technique's moving parts — anyone looking at the demo can poke it.

## Visual Design

Liquid Glass "Floating Pills" (concept variant A from brainstorm):

- 3–5 pill-shaped glass elements drifting gently on the canvas
- Each pill is a **true 3D shape**, not a 2D silhouette with a fake height pop:
  - Viewed from the top (camera axis): a **stadium silhouette** (rectangle + two half-circle ends)
  - Viewed from the side: a **rounded slab** (flat top face + rounded edges + flat bottom face)
  - All transitions between faces are rounded in 3D (no hard creases anywhere)
- Solid photo background, filling the viewport, covered by the pills
- Pills are transparent glass with:
  - Fresnel reflection highlights at grazing angles (brighter where the 3D edge curves away from the viewer)
  - Spectral refraction of the background behind them (dispersion is strongest on the curved rim where the surface normal tilts most)
  - Soft ambient shadow under each pill for anchoring
- Hover cursor → pill slightly raises (z-scale + shadow spread, no color change)
- Drag → pill follows cursor; release keeps new position

## Technology Stack

| Layer | Choice | Why |
|---|---|---|
| Rendering | **WebGPU + WGSL** | "Latest web tech" user ask; compute + render in one API |
| Language | **TypeScript** | Safety around GPU buffer layouts |
| Build | **Vite + bun** | Fast HMR, zero-config for WGSL imports |
| UI panel | **Tweakpane v4** | Lightweight, polished, good for parameter demos |
| Photo source | `https://picsum.photos/1920/1080` | Random landscape photos, CORS-friendly |

No framework (React/Vue/etc.). Raw WebGPU + minimal TS.

## Rendering Approach

### Pipeline (single fullscreen pass)

One render pass, one fullscreen quad, one fragment shader. No separate pill geometry — pills are rendered via SDF inside the fragment shader.

```
[Background photo texture]  ─┐
[Pill uniform buffer]        ─┤→  Fullscreen fragment shader  →  Swapchain
[Frame params (time, N, …)]  ─┘
```

Per fragment:

1. Build a camera ray from the pixel (orthographic; camera looks straight down `-Z` into the XY plane the pills live in).
2. **Sphere-trace the scene SDF** (min over all pill SDFs) up to 32 iterations or until the ray exits the scene AABB.
3. Miss (no hit): output background photo directly.
4. Hit: run the spectral refraction block (below), which itself does a second short sphere-trace to find the back-face exit.

This keeps the tech focused on the dispersion math while rendering genuinely 3D shapes. No mesh data, no vertex pipeline — everything is driven by per-pill uniforms.

### Pill SDF (true 3D)

Each pill is defined by center `(cx, cy, cz=0)`, half-size `(hx, hy, hz)` and edge rounding radius `edgeR`. The SDF is built as a 2-step rounded extrusion so the same shape is a stadium from the top and a rounded slab from the side:

```wgsl
fn sdf_pill_3d(p: vec3f, halfSize: vec3f, edgeR: f32) -> f32 {
  // Step 1: 2D stadium silhouette in XY (corner radius = shortest half-axis)
  let hsXY = halfSize.xy - vec2(edgeR);       // shrink so edge rounding stays inside halfSize
  let r_xy = min(hsXY.x, hsXY.y);             // full pill rounding along the short axis
  let qXY  = abs(p.xy) - hsXY + vec2(r_xy);
  let d_xy = length(max(qXY, vec2(0.0)))
           + min(max(qXY.x, qXY.y), 0.0) - r_xy;

  // Step 2: extrude into Z with edgeR rounding where top/bottom faces meet the side
  let w = vec2(d_xy, abs(p.z) - halfSize.z + edgeR);
  return length(max(w, vec2(0.0)))
       + min(max(w.x, w.y), 0.0) - edgeR;
}
```

Scene SDF = `min(sdf_pill_3d(p - pill_i.center, pill_i.halfSize, pill_i.edgeR))` across pills.

**Defaults:** `halfSize = (160, 44, 20) px`, `edgeR = 14 px`. Pill is thus 320×88×40 px. With `edgeR/hz ≈ 0.7` the top face has a visible gentle dome — closer to Apple Liquid Glass than a flat slab. Tweakpane exposes the short axis, length, thickness and edge radius per-pill (or globally).

**Normals** are computed by central differences on the SDF:

```wgsl
let e = vec2(0.5, 0.0);
let n = normalize(vec3(
  scene_sdf(p + e.xyy) - scene_sdf(p - e.xyy),
  scene_sdf(p + e.yxy) - scene_sdf(p - e.yxy),
  scene_sdf(p + e.yyx) - scene_sdf(p - e.yyx),
));
```

Four extra SDF evals per shaded pixel — cheap.

**Cost budget:** ~32 SDF evals for the front sphere-trace, ~16 for the exit sphere-trace, 4 for the normal. Pill SDF is ~20 cheap ALU ops. Only pixels inside the scene's screen-space AABB do any work (early-out for background pixels). At 1080p with pills covering ~20% of the screen: ~0.4M shaded pixels × (52 SDF × 20 ops + N·CMF work) ≈ 0.5–1.2ms on a mid-range discrete GPU. Fits inside the 16.6ms frame budget with room to spare.

### Spectral Refraction (the core of the demo)

For each fragment inside a pill, sample `N` wavelengths uniformly across 380–700nm. For each wavelength:

**IOR via Cauchy + Abbe number** (glTF KHR_materials_dispersion formulation):

```wgsl
fn ior_at(lambda_nm: f32, n_d: f32, V_d: f32) -> f32 {
  return max(n_d + (n_d - 1.0) / V_d * (523655.0 / (lambda_nm * lambda_nm) - 1.5168), 1.0);
}
```

Constants `523655` and `1.5168` are precomputed from the Fraunhofer F/d/C lines. Default glass: `n_d = 1.5168, V_d = 40.0` (slightly more dispersive than BK7 so the effect is clearly visible).

**Two-surface refraction through the 3D SDF:**

The front hit point `p_front` and its normal `n_front` are already available from the primary sphere-trace (wavelength-independent — geometric ray entry). For each wavelength:

1. Refract the camera ray through the front face: `r1 = refract(view_dir, n_front, 1.0/ior(λ))`
2. Starting from `p_front + r1 * ε`, sphere-trace *inside* the SDF using `-sdf(p)` as the distance (we're inside the solid, so distances flip). ~16 iterations finds the back-face exit point `p_exit`.
3. Back-face normal `n_back = -gradient(sdf)` at `p_exit` (proper 3D normal, not the `-n_front` flat-slab approximation).
4. Refract again (dense → air): `r2 = refract(r1, n_back, ior(λ))`.
5. Map the exit ray to a background texture UV: `bg_uv = screenUV(p_exit) + r2.xy * refractionStrength`.

`refractionStrength` is a user-exposed scalar (default `~0.1` in normalized UV space) that stands in for "how far behind the pill the photo sits" — pure screen-space refraction has no true depth, so this is the one physically-unprincipled knob.

**Per-wavelength cost optimization:** step 2's 16-iteration sphere-trace per wavelength is the expensive part of the loop. Two options, both exposed in the UI:

- **Exact (default at `N ≤ 8`):** sphere-trace per wavelength. Physically correct dispersion on the rim where exit normals differ most per wavelength.
- **Approx (default at `N ≥ 16`):** trace once (at the central wavelength), reuse `p_exit` and `n_back` for all wavelengths, only refract vector `r2` varies. Visually near-identical for thin-ish pills; ~N× cheaper.

Both surfaces bend light; dispersion accumulates over two refractions, matching real glass.

**Spectral → RGB via Wyman-Sloan-Shirley (JCGT 2013):**

Analytic Gaussian-sum approximation of the CIE 1931 2° color matching functions. ~7 Gaussians total, no LUT, no texture:

```wgsl
fn cie_xyz(lambda: f32) -> vec3<f32> {
  // Wyman et al. 2013, Table 1 — multi-lobe Gaussian fit
  let x = 1.056 * g(lambda, 599.8, 37.9, 31.0)
        + 0.362 * g(lambda, 442.0, 16.0, 26.7)
        - 0.065 * g(lambda, 501.1, 20.4, 26.2);
  let y = 0.821 * g(lambda, 568.8, 46.9, 40.5)
        + 0.286 * g(lambda, 530.9, 16.3, 31.1);
  let z = 1.217 * g(lambda, 437.0, 11.8, 36.0)
        + 0.681 * g(lambda, 459.0, 26.0, 13.8);
  return vec3(x, y, z);
}
// g(lambda, mu, sigma1, sigma2) — piecewise Gaussian (narrower below mu, wider above)
```

Accumulation loop:

```wgsl
var xyz = vec3(0.0);
for (var i = 0u; i < N; i = i + 1u) {
  let lambda = mix(380.0, 700.0, (f32(i) + 0.5 + jitter) / f32(N));
  let ior    = ior_at(lambda, n_d, V_d);
  let uv     = compute_refracted_uv(...);          // two-surface step
  let L      = textureSample(bg, samp, uv).rgb;    // background radiance
  let cmf    = cie_xyz(lambda);
  xyz = xyz + cmf * luminance(L);                  // scalar radiance per wavelength
}
xyz = xyz / f32(N);
let rgb = xyz_to_srgb * xyz;
```

Note on the background term: we treat the photo as a **reflectance × illuminant** product. For a tech demo, sampling `luminance(L)` per wavelength is a reasonable simplification — the dispersion direction math is correct, and the color shift at edges comes from the CMF weighting, which is the whole point. A full spectral-upsampling (Jakob/Mallett) of the background RGB into per-wavelength reflectance is deferred as a stretch goal.

**Temporal jitter (quality free-lunch):**

Each frame offsets the wavelength sampling by `jitter ∈ [0, 1/N)` and blends with the previous frame (simple exponential moving average in a history texture). Effective sample count ≈ 2–3× N without raising per-frame cost. Toggleable.

### Glass shading on top of dispersion

- **Fresnel (Schlick):** `F = F0 + (1 − F0)(1 − n·v)^5` with `F0` computed from `n_d`. Modulates reflection vs. transmission.
- **Reflection:** cheap environment approximation — sample the background with a mirror UV offset, tinted slightly cooler. Cheap but reads as "reflective".
- **Beer-Lambert absorption:** disabled by default (clear glass); parameter exposed for tinting demos.

Final composite inside a pill:
```
color = lerp(refraction_rgb, reflection_rgb, F) * (1 - absorption)
```

## Controls

Tweakpane panel, top-right:

| Control | Default | Range | Notes |
|---|---|---|---|
| Sample count `N` | 8 | 3 / 8 / 16 / 32 | Dropdown; 3 is the "fake RGB" reference |
| `n_d` (base IOR) | 1.5168 | 1.0–2.4 | |
| `V_d` (Abbe number) | 40 | 15–90 | Lower = more dispersion |
| Pill thickness (2·hz) | 40 px | 0–200 | Short Z half-axis |
| Pill short axis (2·hy) | 88 px | 20–200 | Drives XY stadium rounding |
| Pill length (2·hx) | 320 px | 80–800 | Long axis |
| Edge radius `edgeR` | 14 px | 0–hz | Ratio to `hz` tunes flat-top ↔ domed-top |
| Refraction strength | 0.1 | 0–0.5 | UV offset scale for exit-ray step |
| Refraction mode | Exact | Exact / Approx | Per-wavelength exit sphere-trace vs reuse |
| Temporal jitter | on | toggle | |
| Absorption | (0,0,0) | color | Beer-Lambert tint |
| Show edge highlight | on | toggle | Fresnel modulation |
| **A/B compare key** | `Z` | — | Hold: force `N=3`; release: restore |

Drag to move any pill. `Space` to randomize pill positions. `R` to reload the photo.

## File Layout

```
RealRefraction/
├── index.html                          # single canvas + fallback message
├── package.json                        # bun / vite / tweakpane / typescript
├── vite.config.ts                      # + wgsl ?raw import plugin
├── tsconfig.json
├── src/
│   ├── main.ts                         # entry: init WebGPU, photo load, RAF loop
│   ├── webgpu/
│   │   ├── device.ts                   # adapter + device + no-WebGPU message
│   │   ├── pipeline.ts                 # render pipeline, bind group layouts
│   │   └── uniforms.ts                 # typed uniform buffer writers
│   ├── pills.ts                        # pill state, drag handling, layout
│   ├── photo.ts                        # picsum fetch + texture upload
│   ├── ui.ts                           # Tweakpane binding
│   └── shaders/
│       └── dispersion.wgsl             # the whole fragment shader
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-04-21-spectral-dispersion-design.md
```

Each file has one clear purpose. `dispersion.wgsl` holds all the shader math; `main.ts` wires state and frame loop; everything else is thin glue.

## Data Flow

1. On load: `photo.ts` fetches Picsum → `ImageBitmap` → GPU texture (sRGB view).
2. `pills.ts` creates initial pill array (positions jittered across viewport).
3. Each frame (`main.ts` RAF):
   - `pills.ts` updates drag state and writes pill data to a uniform buffer
   - `ui.ts` has already pushed any knob changes to a params uniform buffer
   - `pipeline.ts` issues one draw (6 verts, fullscreen triangle pair)
   - If temporal jitter is on, a history texture is swapped and blended in the shader
4. On resize: swapchain + history texture re-created.

## Error Handling

Minimal on purpose — this is a local tech demo:

- No WebGPU adapter → DOM message "This demo needs a WebGPU-capable browser (Chrome/Edge 2023+, Safari 18+)." No fallback path.
- Photo fetch fails → fall back to a bundled solid gradient texture so the pills still show *something*.
- Shader compile error → log to console, keep previous pipeline if any; otherwise black screen + error text.

No runtime user-facing error UI beyond the above.

## Testing

A tech demo, not a library — no unit tests. Verification is visual:

- [ ] Press `Z` to toggle `N=3` vs `N=8`; rainbow fringe should go from blocky three-band to smooth gradient.
- [ ] Drop `V_d` to 20; rainbow should widen dramatically.
- [ ] Raise `V_d` to 80; rainbow should shrink to near-colorless.
- [ ] Drag a pill across a high-contrast edge in the photo; dispersion band should update per-frame with no stutter.
- [ ] Disable temporal jitter at `N=8`; a faint banding should become visible at pill edges. Re-enable; banding smooths out.
- [ ] Resize the window; pills and photo reflow, no crash, no black flash > 1 frame.
- [ ] Push `edgeR` to its max (equal to `hz`); pill top should become a visible dome and dispersion should spread across the whole pill, not just the rim. Drop `edgeR` near zero; the rim becomes a thin sharp-cornered halo and dispersion concentrates there. This verifies the 3D SDF is truly 3D (not a 2D silhouette trick).
- [ ] Toggle refraction mode Exact ↔ Approx at `N=16`; verify visual difference is subtle and Approx is noticeably faster.

Browser targets: Chrome 120+, Edge 120+, Safari 18+ (manual smoke test each before declaring done).

## Open Decisions Deferred to Implementation

- Exact Gaussian coefficients for Wyman CMF: copy verbatim from JCGT 2013 Table 1 during implementation.
- Pill animation idle behavior (slow drift? static? Lissajous?) — pick during implementation, not worth a design round.
- History texture format for temporal jitter (`rgba16float` likely).
- Photo orientation handling (if Picsum returns a portrait on a landscape canvas): cover-fit UV in shader.

## Stretch Goals (explicitly out of scope for first cut)

- Hero Wavelength Sampling with proper stratification across frames
- Spectral upsampling of the background photo (Jakob 2019 or Mallett-Yuksel 2019) to treat it as real reflectance
- Iridescence / thin-film interference on the pill top surface
- Mouse-parallax "depth" on the photo for a richer parallax look
- Light source simulation instead of just photo background (caustics)

## References

1. Khronos. **KHR_materials_dispersion** — [github.com/KhronosGroup/glTF/.../KHR_materials_dispersion](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_dispersion/README.md)
2. Wyman, Sloan, Shirley (2013). **Simple Analytic Approximations to the CIE XYZ Color Matching Functions.** JCGT 2(2). [jcgt.org/published/0002/02/01](https://jcgt.org/published/0002/02/01/)
3. Wilkie et al. (2014). **Hero Wavelength Spectral Sampling.** EGSR. [jo.dreggn.org/home/2014_herowavelength.pdf](https://jo.dreggn.org/home/2014_herowavelength.pdf)
4. Peters (2025). **Spectral Rendering, Part 2.** [momentsingraphics.de/SpectralRendering2Rendering.html](https://momentsingraphics.de/SpectralRendering2Rendering.html)
5. Heckel. **Refraction, dispersion, and other shader light effects.** [blog.maximeheckel.com/posts/refraction-dispersion-and-other-shader-light-effects](https://blog.maximeheckel.com/posts/refraction-dispersion-and-other-shader-light-effects/)
6. Bruffen. **spectral2d-webgpu.** [github.com/Bruffen/spectral2d-webgpu](https://github.com/Bruffen/spectral2d-webgpu)
