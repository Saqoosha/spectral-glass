# Diamond Shape — Phase A Design

**Goal:** Add round brilliant cut diamond as a new shape alongside pill/prism/cube/plate, reusing the existing entry-refraction + exit-refraction trace. No fire/sparkle yet — that's Phase B.

**Status:** approved by user (both design sections).

---

## Scope

**Phase A (this spec)**
- New SDF `sdfDiamond` with Full 58-facet round brilliant geometry (table + 8 bezel + 8 star + 16 upper half + 16 lower half + 8 pavilion main + pointed culet)
- Fixed Tolkowsky proportions; one user slider `diamondSize` (girdle diameter)
- Y-axis auto-rotation with 20° forward tilt (jewelry display stand pose)
- Draggable XY position (reuses existing pills drag UX), multi-instance up to `MAX_PILLS=8`
- Trace: existing `sphereTrace` (entry) + `insideTrace` (exit). No TIR.

**Phase B (separate spec later)**
- Multi-bounce internal TIR trace path → enables fire/sparkle
- Per-instance proportions / angle sliders

---

## Architecture

### 1. SDF: D_8 fundamental-domain fold + half-space intersection

The brilliant cut is a convex polytope (except the cylindrical girdle, special-cased). D_8 symmetry (8 rotations + 8 mirror planes = 16 elements) has a fundamental domain of a 22.5° wedge. We fold input XY to that wedge in three reflections, then evaluate 7 distance terms:

```wgsl
fn sdfDiamond(pIn: vec3<f32>, diameter: f32) -> f32 {
  let p0 = frame.diamondRot * pIn;        // tilt + Y-axis spin

  // 1/16 fold: abs(y), swap if y>x, reflect across θ=π/8 line
  var q = abs(p0.xy);
  if (q.y > q.x) { q = q.yx; }
  let fn_ = vec2<f32>(-sin(pi/8.0), cos(pi/8.0));
  let dAb = dot(q, fn_);
  if (dAb > 0.0) { q = q - 2.0 * dAb * fn_; }

  let p = vec3<f32>(q, p0.z);
  let r = length(q);   // radial distance, for the girdle cylinder term

  // All offsets are in units of "diameter", scaled to world at evaluation time.
  let d_table   = p.z - H_TABLE * diameter;
  let d_bezel   = dot(p, N_BEZEL)    - O_BEZEL   * diameter;
  let d_star    = dot(p, N_STAR)     - O_STAR    * diameter;
  let d_uhalf   = dot(p, N_UHALF)    - O_UHALF   * diameter;
  let d_girdle  = r                  - R_GIRDLE  * diameter;
  let d_lhalf   = dot(p, N_LHALF)    - O_LHALF   * diameter;
  let d_pavmain = dot(p, N_PAVMAIN)  - O_PAVMAIN * diameter;

  return max(max(max(d_table, d_bezel), max(d_star, d_uhalf)),
             max(d_girdle, max(d_lhalf, d_pavmain)));
}
```

**7 terms** for 58 facets: table, bezel (crown main), star, upper half, girdle cylinder, lower half, pavilion main. The pointed culet is naturally handled by the pavilion main planes converging at the apex — no separate plane needed. Inside the fundamental wedge every facet class has exactly one representative, so no `min()` over multiple representatives.

**Girdle is a cylinder, not a plane.** `length(q.xy) - R_GIRDLE * d` is the signed distance to the cylindrical side.

### 2. Plane constants: single source via WGSL injection

`src/math/diamond.ts` is the single source of truth for Tolkowsky proportions AND the derived plane normals/offsets. It exports `diamondWgslConstants(): string` which returns a block of WGSL `const` declarations (`N_BEZEL`, `O_BEZEL`, `H_TABLE`, `R_GIRDLE`, etc.). `src/webgpu/pipeline.ts` concatenates this block onto the shader source before compilation. No drift risk between TS and WGSL.

Why WGSL `const` (baked into the shader) rather than `frame.*` uniforms: the plane geometry is FIXED Tolkowsky — it never changes at runtime. Only `diameter` (the runtime parameter) scales the whole shape. Zero uniform bandwidth, zero per-frame work.

### 3. Rotation: Y-axis spin + 20° forward tilt

Same pattern as `cube.ts` and `plate.ts`:
- New uniforms: `diamondRot`, `diamondRotPrev` (both `mat3x3<f32>`, 48 B each)
- Host builds from `sceneTime`: `Rx(-20° · π/180) · Ry(sceneTime · 0.30)`
- Negative tilt so the crown tilts toward the camera (table partially visible)
- Frozen when paused (`sceneTime` driver)

`src/math/diamond.ts` exports:
- `DIAMOND_SIZE_MIN = 50`, `DIAMOND_SIZE_MAX = 400` — slider bounds
- `DIAMOND_HEIGHT_RATIO ≈ 0.61` — for proxy AABB (crown + girdle + pavilion)
- `diamondRotationColumns(time: number): Float32Array` — WGSL-padded 12-float layout, same contract as `cubeRotationColumns`
- `diamondWgslConstants(): string` — WGSL `const` declarations for SDF plane table

### 4. Trace: reuse existing generic path

- `sceneSdf` gets a `shapeId == 4 → sdfDiamond(local, frame.diamondSize)` branch
- `sphereTrace` (front hit): unchanged, runs the generic `sceneSdf`
- Back hit: falls through `backExit`'s default path → `insideTrace` + `sceneNormal` finite-diff. Same path pill/prism use. No analytical exit for Phase A.
- `reprojectHit` gets a diamond branch using `frame.diamondRot` / `frame.diamondRotPrev`, so TAA history follows the spin

### 5. Proxy bounds

AABB: `halfSize = (d/2, d/2, d · DIAMOND_HEIGHT_RATIO / 2)`. Rotated by `diamondRot` in `vs_proxy` so the rasterized silhouette stays tight under rotation (same trick cube and plate use).

### 6. No `edgeR` rounding

Sharp facets are the look. The SDF uses raw `max()` without the rounded-box `length(max(q, 0)) - edgeR` trick. `edgeR` is irrelevant for diamond — the slider stays hidden via the same conditional UI logic that hides `pillShort` on plate.

Facet creases are sharp in screen space; the existing `sceneNormal` zero-gradient sentinel routes crease pixels to the bg path, which blends cleanly. If sparkle along creases turns out visually objectionable in practice, a smooth-max mitigation goes into Phase B polish.

---

## Parameters

### User-facing

| Field | Type | Range | Default | Notes |
|---|---|---|---|---|
| `diamondSize` | number | 50–400 px | 200 | Girdle diameter. Shown in Shape folder only when `shape === 'diamond'`. |

### Fixed (Tolkowsky ideal, derived in `diamond.ts`)

| Constant | Value | Role |
|---|---|---|
| `TABLE_RATIO` | 0.53 | table width / diameter |
| `CROWN_ANGLE` | 34.5° | bezel facet angle from horizontal |
| `PAVILION_ANGLE` | 40.75° | pavilion main angle from horizontal |
| `GIRDLE_THICKNESS_RATIO` | 0.02 | girdle band height / diameter |
| `STAR_LENGTH_RATIO` | 0.5 | star facet tip as fraction of table→girdle radial distance |
| `UPPER_HALF_SPLIT_RATIO` | 0.5 | upper-half/bezel edge split along the bezel side |
| `LOWER_HALF_SPLIT_RATIO` | 0.5 | lower-half/pavilion split along the pavilion side |

Derived (computed once at module load):
- `R_GIRDLE = 0.5`
- `R_TABLE = 0.5 · TABLE_RATIO ≈ 0.265`
- `H_CROWN = (R_GIRDLE − R_TABLE) · tan(CROWN_ANGLE) ≈ 0.161`
- `H_PAVILION = R_GIRDLE · tan(PAVILION_ANGLE) ≈ 0.431`
- `DIAMOND_HEIGHT_RATIO = H_CROWN + GIRDLE_THICKNESS_RATIO + H_PAVILION ≈ 0.612`
- 7 plane (nx, ny, nz) + offset values — derivation lives in `diamond.ts` alongside its test.

All offsets are in "units of diameter" — the runtime diameter multiplies each offset inside `sdfDiamond`.

---

## Uniform layout changes

Current layout (544 B):
```
HEAD (80) | cubeRot (48) | cubeRotPrev (48) | plateRot (48) | plateRotPrev (48)
         | plateParams (16) | pills[8] (256)
```

New layout (+112 B = 656 B):
```
HEAD (80) | cubeRot (48) | cubeRotPrev (48) | plateRot (48) | plateRotPrev (48)
         | diamondRot (48) | diamondRotPrev (48)
         | plateParams (16) | diamondParams (16, diamondSize + 3 pad) | pills[8] (256)
```

- `diamondRot` / `diamondRotPrev`: mat3x3, same padded layout as cube/plate
- `diamondParams` block: `diamondSize` + 3 pad floats (16-B alignment preserved)
- `pills` array stays at the end (layout test still pins this)

---

## Files

### New
- `src/math/diamond.ts` — Tolkowsky constants, rotation builder, WGSL-const string generator
- `tests/diamond.test.ts` — mirrors `cube.test.ts`: identity at t=0, orthonormality, non-finite-time rejection; plus a numeric sanity check on the derived plane offsets (e.g., `H_CROWN ≈ 0.161 · diameter`)

### Modified
- `src/shaders/dispersion.wgsl` — `sdfDiamond` function; Frame struct adds `diamondRot`, `diamondRotPrev`, `diamondSize`; `sceneSdf` dispatch; `reprojectHit` branch
- `src/webgpu/uniforms.ts` — byte layout update (+112 B), `FrameParams` gets `diamondSize`; ordering constants updated
- `src/webgpu/pipeline.ts` — prepend `diamondWgslConstants()` block to the shader source before `createShaderModule`
- `src/ui.ts` — `shape` type extended, `DIAMOND_SIZE_MIN`/`DIAMOND_SIZE_MAX` exports, dropdown option, `diamondSize` slider + `syncShapeSliders` update so it's visible only for diamond (and `pillLen/Short/Thick/edgeR` are hidden for diamond)
- `src/persistence.ts` — `SHAPES` allowlist extended, `diamondSize` field with `clamp`
- `src/main.ts` — `SHAPE_ID.diamond = 4`, `writeFrame` call carries `diamondSize`, `sceneTime` continues driving rotations
- `src/pills.ts` — drag `findHit` treats shape 4 as circular (like cube)
- `tests/uniformsLayout.test.ts` — update the expected field list for the Frame struct

### Unchanged
- Bind group layout (`src/webgpu/pipeline.ts`'s `bindGroupLayout`) — same texture / sampler slots
- Trace internals (`sphereTrace`, `insideTrace`, `sceneNormal`) — generic path reused
- `src/webgpu/history.ts`, `src/webgpu/postprocess.ts` — not touched

---

## Out of scope (confirmed)

- Fire / sparkle — requires multi-bounce TIR, deferred to Phase B
- Per-instance `diamondSize` — all diamond instances share the global slider
- Adjustable angles (crown, pavilion, table%) — fixed Tolkowsky for Phase A
- Faceted girdle (32 vertical sub-facets) — polished girdle only
- Environment map for photorealistic reflections — Phase C territory

---

## Verification

1. **Silhouette**: at `diamondSize` 100 / 200 / 400 the shape reads as a brilliant cut — octagonal table visible from above, crown + pavilion split visible from tilted angle (which the 20° fixed tilt guarantees).
2. **Rotation smoothness + TAA**: Y-axis spin with no history smearing — `reprojectHit` for shape 4 reads the right prev-rotation pixel.
3. **Drag**: click-and-drag moves the diamond; release persists.
4. **Multi-instance**: two diamonds render independently, both rotate in sync (shared `diamondRot`), both draggable.
5. **Persistence**: reload preserves `diamondSize` and instance positions.
6. **Non-regression**: pill/prism/cube/plate visuals and drag behavior unchanged.
7. **Unit tests**: `tests/diamond.test.ts` + updated `tests/uniformsLayout.test.ts` pass.
8. **Build**: `bun run build` clean (tsc + vite).

---

## Implementation handoff

1. `superpowers:writing-plans` produces the step-by-step plan from this spec.
2. Implement inline (scope is small enough that subagent overhead isn't worth it).
3. `/review-fix-loop` standard mode to catch critical + important issues.
4. Commit (no push until user returns and says so).
