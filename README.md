# Spectral Glass

> **Live demo**: <https://saqoosha.github.io/spectral-glass/> (Chrome/Edge 120+ or Safari 18+)

A realtime WebGPU demo of **physically accurate spectral dispersion** through
Apple "Liquid Glass"-style floating pills, triangular prisms, and rotating
cubes. Unlike the common "shift R/G/B IORs" hack that most web implementations
use (including Three.js's `MeshPhysicalMaterial.dispersion`), this samples the
full visible spectrum per-wavelength and reconstructs the final color via CIE
1931 color matching functions.

![Cubes over rooftops](docs/images/demo-default.png)

Above: four rotating glass cubes (Rainbow soap material, `n_d = 1.272`,
`V_d = 1.5`, perspective FOV 46°, N = 32) over a grayscale Picsum photo.
A monochrome background lets you see the per-wavelength dispersion as pure
spectral colors instead of mixing with the photo's own chroma — every band
on the cube is the shader splitting the photo by wavelength in real time at
60 fps.

Below: same cubes against a Paris streetscape — top-left cube shows a full
red→blue spectrum where the back-face exit lands on the bright sky.

![Cubes over Paris](docs/images/demo-strong-dispersion.png)

## Why not just shift R/G/B?

A 3-sample RGB IOR is visibly a *three-band* rainbow. Real glass is a continuous
spectrum. When dispersion is strong you can see the difference — the 3-sample
version looks like bad chromatic aberration, while the 8-sample spectral version
looks like a prism.

Press and hold **`Z`** in the demo to force `N = 3`. Release to go back to
`N = 8`. The quality difference is unmistakable.

## Quick start

```bash
bun install
bun run dev        # http://localhost:5173
bun run test       # Vitest on the math modules
bun run build      # tsc --noEmit + vite build
```

Requires a WebGPU-capable browser (Chrome / Edge 120+, Safari 18+).

## Controls

| Input | Action |
|---|---|
| Drag a shape | Move it around the canvas (cube uses a circular hit radius) |
| **`Z`** (hold) | Force `N = 3` (fake RGB dispersion) for A/B comparison |
| **Space** | Shuffle pills to random positions |
| **`R`** | Reload a new random Picsum photo |
| Tweakpane | IOR, Abbe, sample count, shape (pill / prism / cube), dimensions, refraction strength, projection (ortho / perspective), FOV, temporal jitter, refraction mode |
| Presets | Subtle pill · Strong dispersion · Prism rainbow · Rotating cube |
| Materials | 10 real-world glasses (water → BK7 → SF flints → diamond → moissanite) + 4 fantasy (n_d up to 3.5, V_d down to 2) |

Add `?perf=1` to the URL to enable the GPU timestamp HUD — timings are
published on `window._perf.samples`. Check **Show proxy** in the UI to tint
every proxy fragment pink and see the rasterised silhouette.

## Technical approach

- **WebGPU + WGSL, two-pass.** Cheap fullscreen bg pass (photo + history)
  followed by an instanced 3D-cube mesh proxy per pill. The heavy per-pixel
  refraction shader only runs on fragments inside the proxy silhouette.
  Back-face culling (CCW-outward 3D → CW NDC after Y-flip) gives exactly one
  invocation per covered pixel.
- **3D SDFs, three shapes.** Pill (stadium XY + rounded Z), prism
  (isosceles triangle in YZ extruded in X), and rotating cube (rounded box +
  per-frame `rot * (p - center)` via `cubeRotation(time)`). Cube's proxy
  corners are transformed by `transpose(rot)` so the rasterised silhouette
  tracks the shader's rotation exactly — no √3 bounding-box slack.
- **Ortho or perspective projection.** UI toggle. Ortho keeps the flat Liquid
  Glass aesthetic; perspective uses a pinhole camera at `(w/2, h/2, cameraZ)`
  with `cameraZ = (height/2) / tan(fov/2)` derived from the user-facing FOV.
- **Cauchy + Abbe IOR.** Wavelength-dependent index via the glTF
  `KHR_materials_dispersion` formula.
- **Wyman-Sloan-Shirley CIE XYZ** (JCGT 2013) analytic approximation — no
  lookup tables.
- **Two-surface refraction.** Front hit via primary sphere-trace, back exit via
  per-wavelength inside-trace (Exact mode) or shared hero-wavelength trace
  (Approx mode, Wilkie 2014).
- **Per-wavelength Fresnel.** Blue λ has higher IOR → higher Schlick Fresnel
  → visible blue-tinged rim on diamonds and prisms (the classic "fire" of
  high-index crystals).
- **Per-wavelength sRGB weighting.** Each sampled photo pixel is weighted by
  `xyzToSrgb(cmf(λ))` — short-wavelength samples contribute to blue, long to
  red. This preserves photo color when refraction UVs coincide and produces
  real chromatic fringing where they diverge.
- **Spatial + temporal jitter.** Per-pixel wavelength phase via `hash21` so
  neighbouring pixels sample different λ — the eye and history accumulation
  average the noise, so N=8 stratified looks like N=16 uniform.
- **TIR fallback.** When `refract()` returns zero at the back face, the
  wavelength contributes the external reflection instead of dropping — no
  black holes inside the cube.
- **Temporal accumulation.** `rgba16float` ping-pong history with EMA blend
  (α = 0.2 steady-state, 1.0 for one frame after a scene change so cube
  tail doesn't ghost in).
- **sRGB OETF** applied manually when the swapchain format is non-sRGB.
- **localStorage persistence.** Validated load (rejects NaN / bogus enums),
  trailing-edge debounced save, pagehide flush.

## Project structure

```
src/
├── main.ts                     Frame loop + glue
├── math/                       Pure math modules (unit-tested)
│   ├── cauchy.ts               Wavelength → IOR (glTF formulation)
│   ├── wyman.ts                Wyman CIE XYZ approximation
│   ├── srgb.ts                 XYZ → linear sRGB matrix + OETF
│   ├── sdfPill.ts              3D pill SDF (mirrors WGSL version)
│   ├── sdfPrism.ts             Triangular prism SDF (mirrors WGSL version)
│   └── sdfCube.ts              Rounded box / cube SDF (mirrors WGSL version)
├── persistence.ts              localStorage: validated load, debounced save, pagehide flush
├── photo.ts                    Picsum fetch → GPU texture (w/ gradient fallback)
├── pills.ts                    Pill state + shape-aware pointer drag
├── ui.ts                       Tweakpane bindings (shape selector, presets, materials)
├── webgpu/
│   ├── device.ts               Adapter + device + error handlers
│   ├── history.ts              Ping-pong history textures
│   ├── pipeline.ts             Bg + proxy pipelines + shared bind groups
│   ├── perf.ts                 GPU timestamp harness (?perf=1)
│   └── uniforms.ts             Typed uniform buffer writer
└── shaders/
    ├── fullscreen.wgsl         Fullscreen triangle vertex shader
    └── dispersion.wgsl         SDFs (pill/prism/cube) + spectral dispersion fragment

tests/                          Vitest unit tests for each math module
docs/
└── ARCHITECTURE.md             Frame path, uniform layout, SDF & tracing details
```

Math modules in `src/math/` are mirrored 1:1 by functions in
`src/shaders/dispersion.wgsl` — the 31 vitest tests act as the reference
implementation for the shader.

## Design

- [Architecture notes](docs/ARCHITECTURE.md) — module map, frame path, uniform layout, proxy mesh + camera, per-wavelength loop (spatial stratification, per-λ Fresnel, TIR fallback), measured performance

## Performance

Apple Silicon (1132×1046, WebGPU `timestamp-query`):

| Config | GPU time |
|---|---:|
| pill N=8  | 1.40 ms |
| pill N=32 | 6.70 ms |
| cube N=8  | 2.08 ms |
| cube N=32 | 9.64 ms |
| cube N=64 | 11.49 ms |

All within the 16.67 ms vsync budget. Background pixels cost ~nothing; the
per-λ loop dominates on pill / prism / cube pixels. Apple's TBDR already
culls background efficiently, but discrete GPUs gain more from the proxy
pass.

## References

1. Khronos. [**KHR_materials_dispersion**](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_dispersion/README.md) — the Cauchy + Abbe formulation used here.
2. Wyman, Sloan, Shirley (2013). [**Simple Analytic Approximations to the CIE XYZ Color Matching Functions.**](https://jcgt.org/published/0002/02/01/) JCGT 2(2).
3. Wilkie et al. (2014). [**Hero Wavelength Spectral Sampling.**](https://jo.dreggn.org/home/2014_herowavelength.pdf) EGSR.
4. Peters (2025). [**Spectral Rendering, Part 2.**](https://momentsingraphics.de/SpectralRendering2Rendering.html)
5. Heckel. [**Refraction, dispersion, and other shader light effects.**](https://blog.maximeheckel.com/posts/refraction-dispersion-and-other-shader-light-effects/)

## Status

Tech demo / proof of technique. Not a library. No production website
integration. If you want to pull the spectral-refraction technique into your
own project, the interesting files are `src/shaders/dispersion.wgsl` and the
six math modules in `src/math/`.
