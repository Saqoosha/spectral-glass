# Diamond Phase B — Analytical back-exit + short-chain TIR

**Decision date**: 2026-04-23 (approved by user)
**Context branch**: commit `db86855` (Phase A complete, review-fix-loop CLEAN, 106/106 tests)

> **Subsequent code (not retro-edited below):** the live shader uses a
> **configurable** max internal bounce count (`diamondTirMaxBounces`, 1…32,
> default 6), a **scaled/clamped** bounce-origin nudge, `diamondAnalyticExit`
> miss handling, and **TIR debug** tints (pink = chain exhaust / still TIR;
> orange = analytic miss). See `README.md` and `docs/ARCHITECTURE.md`.

**Spec revisions**:
- 2026-04-23: extended bounce budget from 1 to 2 after observing that the
  Tolkowsky cut's top-view light-return path requires 2 internal
  reflections before escaping through the crown (empirical: with 1 bounce
  the top view was ~100 % unresolved, with 2 bounces it drops to ~5 %).
  Added a `diamondTirDebug` uniform + UI toggle to paint the chain-
  exhaustion fallback hot pink as a diagnostic aid.
- 2026-04-23: gated the bounce chain on exact mode (`!useHero`). In approx
  mode the shared hero-wavelength exit combined with per-frame heroLambda
  jitter produced TIR-boundary flicker; approx mode keeps Phase A's
  `reflSrc` fallback.
- 2026-04-23: first attempt added MIN_STEP bias on the bounce re-entry
  origin; then reverted after review found it overshoots genuinely
  nearby adjacent facets (e.g. upper-half→girdle second bounce in
  ~0.08 px of travel). Final resolution: `diamondAnalyticExit` uses a
  dedicated `DIAMOND_BOUNCE_EPS = 0.01 px` as its self-hit threshold
  (~25× tighter than the sphere-tracer `HIT_EPS = 0.25 px`), and the
  bounce loop hands in `curP` directly without any bias. The first
  `backExit` call keeps its legacy `MIN_STEP` bias because cube/plate
  analytic exits still need it for their slab-math 0/0 guard.

## Why

Phase A renders a Tolkowsky-ideal round brilliant cut as a D_8-folded convex
polytope SDF. Back-face exit uses the generic `insideTrace` + finite-diff
`sceneNormal(pExit)` path. At sharp facet edges the finite-diff normal is
degenerate and `refract()` at steep internal angles routes to the TIR
fallback, which substitutes `reflSrc` (the front-face external reflection
photo sample) — the same hack cube/pill/prism/plate share.

User symptom (confirmed via facet-color debug isolating the refraction
path): during slow tumble, pixels switch abruptly from refraction color to
`reflSrc` as facets rotate across TIR boundaries. Looks like "other faces
suddenly appearing". Partially physical — real diamonds DO spark at facet
alignments — but the `reflSrc` substitute is directionally unrelated to the
actual bounced-light path, so the flashes read as an artifact rather than
fire.

User accepts that a proper Phase B will still show sudden flashes ("that's
what real diamonds do"), but wants them to be physically meaningful sparkle
instead of front-reflection stand-ins.

## Goals

1. **B1**: Replace `insideTrace` + `sceneNormal` for diamond with an
   analytical ray-polytope back-exit that returns the exact facet plane's
   outward normal. Eliminates finite-diff degeneracy at facet edges.
2. **B2**: On back-face TIR, reflect the ray inside the diamond, call the
   analytical exit a second time, refract out at the second facet. One
   bounce is enough for most of the visual payoff; further bounces add
   cost for marginal returns.
3. Keep cube/plate analytic exits (already done) and pill/prism generic
   path untouched. Only diamond changes.

## Scope & non-goals

- **In**: analytical exit for diamond, short-chain TIR for diamond (up to
  2 internal bounces per wavelength, exact mode only), a
  `diamondTirDebug` diagnostic toggle, tests, documentation update.
- **Out**: >2 bounce depth (3rd bounce buys only a few % more coverage at
  1.5× the per-pixel TIR cost — revisit as Phase C). Bounce chain in
  approx mode (hero-exit sharing causes frame-to-frame flicker when
  heroLambda jitters — approx mode keeps Phase A's `reflSrc` fallback).
  Analytical exit for pill/prism. Refactoring the generic `insideTrace`.

## Design

### B1. `diamondAnalyticExit(roWorld, rdWorld, pillIdx) -> CubeExit`

Lives in `src/shaders/diamond.wgsl` next to `sdfDiamond` /
`hitDiamondPillIdx` / `diamondProxyVertex`.

Returns the same `CubeExit { pWorld, nBack }` struct the existing cube/plate
analytic exits return so `backExit()` in `dispersion.wgsl` can dispatch
uniformly.

**Algorithm**:

```
1. Transform ray to diamond-local space:
     roLocal = frame.diamondRot * (roWorld - pill.center)
     rdLocal = frame.diamondRot * rdWorld
   (diamondRot is world->local; rotation preserves direction, no translation for rd)

2. For each facet plane in the unfolded polytope (in local space):
     t_candidate = (offset - dot(n, roLocal)) / dot(n, rdLocal)
   Keep the MIN positive t and remember the corresponding plane normal.

3. For the girdle cylinder (r = R_GIRDLE, z in [-H_GIRDLE_HALF, +H_GIRDLE_HALF]):
     solve (roLocal.xy + t*rdLocal.xy).length() = R_GIRDLE
     quadratic in t; take smallest positive root whose z is in band.
     Normal at that point = (x, y, 0) / R_GIRDLE.

4. tMin across all facets + cylinder gives the exit.

5. Transform result back to world:
     pExitWorld = transpose(frame.diamondRot) * pExitLocal + pill.center
     nWorldBack = transpose(frame.diamondRot) * nLocal
   (normal needs the inverse rotation, which equals transpose for orthonormal R)
```

**Plane enumeration**:

The diamond has 57 planes total:
- 1 table (+Z cap)
- 8 bezels at φ = k·π/4
- 8 stars at φ = π/8 + k·π/4
- 16 upper halves at φ = ±π/16 + k·π/4
- 16 lower halves at φ = ±π/16 + k·π/4
- 8 pavilion mains at φ = k·π/4

Plus 1 girdle cylinder.

D_8 fold in `sdfDiamond` reduces these to 7 plane CLASSES (bezel, star, UH,
LH, pavilion, table, girdle). The `DIAMOND_*_N` / `DIAMOND_*_O` WGSL
constants hold the representative normal/offset for each class in the
fundamental wedge [φ ∈ [0, π/8]].

For analytical exit we need ALL unfolded planes (can't use the fold
because the ray crosses wedge boundaries). Two options:

**Option A (preferred)**: emit rotated plane arrays from `src/math/diamond.ts`
as WGSL consts. E.g. `DIAMOND_BEZEL_N_ARR: array<vec3<f32>, 8>` +
`DIAMOND_BEZEL_O` (shared — offset is rotation-invariant for azimuthal
symmetry). Small memory (57 vec3s + scalars), zero per-frame cost.

**Option B**: rotate in the shader via `rotateZ(n, k*π/4)`. Saves const
space but does 57 mat-vec per pixel per bounce — probably slower.

Go with Option A.

### B2. Short-chain TIR (up to 2 bounces) for diamond

In `src/shaders/dispersion.wgsl` wavelength loop, replace the current:

```wgsl
if (r2TIR) { refractL = reflSrc; }
```

with a diamond-specific bounce chain capped at 2 iterations:

```wgsl
if (r2TIR) {
  if (isDiamond && !useHero) {
    var curR1    = r1;
    var curNBack = nBack;
    var curP     = pExit;
    var outDir: vec3<f32> = vec3<f32>(0.0);
    var resolved: bool    = false;
    for (var bounce: u32 = 0u; bounce < 2u; bounce = bounce + 1u) {
      let bouncedR1 = reflect(curR1, curNBack);
      // Use curP directly; diamondAnalyticExit uses DIAMOND_BOUNCE_EPS
      // (0.01 px) for self-hit rejection — tight enough to admit real
      // sub-pixel adjacent facets while still filtering ro-on-facet t≈0.
      let exN = diamondAnalyticExit(curP, bouncedR1, analyticIdx);
      let trial    = refract(bouncedR1, exN.nBack, ior);
      let trialDot = dot(trial, trial);
      let trialNaN = trialDot != trialDot;   // self-compare catches NaN
      if (trialDot >= 1e-4 && !trialNaN) { outDir = trial; resolved = true; break; }
      curR1 = bouncedR1; curNBack = exN.nBack; curP = exN.pWorld;
    }
    if (resolved) {
      // Sample the photo at the shifted UV, same parallax approximation
      // as the non-bounced path; curP drift is negligible at the photo's
      // assumed distance behind the plane.
      let uvOff2 = uv + (outDir - rd).xy * strength;
      // ... coverUv + in-bounds + textureSampleLevel; see
      // dispersion.wgsl:1313-1321 for the full form.
    } else {
      // Chain exhausted. Production: bg (silhouette blend). Debug:
      // hot pink when `frame.diamondTirDebug > 0.5`.
      refractL = select(bg, vec3<f32>(1.0, 0.2, 0.75), frame.diamondTirDebug > 0.5);
    }
  } else {
    refractL = reflSrc;   // existing behaviour for non-diamond + diamond-approx
  }
}
```

Subtleties:
- **Self-hit tolerance: `DIAMOND_BOUNCE_EPS = 0.01 px`** (25× tighter
  than the global sphere-tracer `HIT_EPS = 0.25 px`). Adjacent diamond
  facets can be separated by under 0.1 px at typical sizes (e.g.
  upper-half to girdle at the crown-base edge), so HIT_EPS was too
  coarse and caused valid nearby second bounces to be silently
  rejected. The tighter DIAMOND_BOUNCE_EPS admits them while still
  filtering the t≈0 self-hit on the previous facet. No MIN_STEP bias
  on the bounce origin — it would overshoot the same sub-pixel
  neighbours the tighter epsilon is designed to catch.
- **Gate on `!useHero`**. In approx (hero-wavelength) mode, `pExit` /
  `nBack` are the hero's shared exit values — running the bounce chain
  from that shared origin while `heroLambda` jitters per frame produces
  TIR-boundary flicker. Approx mode falls back to the Phase A `reflSrc`
  path.
- **2 iterations, not 1**. Empirically: with 1 bounce the Tolkowsky
  top-view light-return path stays unresolved (~100 % pink in debug
  mode); 2 bounces picks up pavilion→pavilion→crown paths and drops the
  unresolved rate to ~5 %. A 3rd iteration recovers only a few extra %
  at 1.5× the per-pixel cost — Phase C territory.
- **`analyticIdx` stays the same across iterations** — the ray is still
  inside the same diamond pill.
- **`trialNaN` self-compare** matches the first-bounce NaN gate above
  the `r2TIR` branch.
- **`bestN` default is zero** in the inner analytic exit so a chain
  iteration that somehow misses every facet returns a zero normal;
  `refract(*, 0, ior)` then yields zero and the outer `trialDot >= 1e-4`
  test naturally absorbs it into the `resolved = false` fallback.

## Files to touch

| File | Change |
|------|--------|
| `src/math/diamond.ts` | Emit `DIAMOND_*_N_ARR` constants from `diamondWgslConstants()` (rotated copies of each facet class's plane). Offsets stay as existing scalar consts — rotation-invariant under orthogonal Z-axis rotation. |
| `src/math/diamondExit.ts` (new) | JS mirror of the analytical exit for regression pinning. Algorithm matches WGSL line-for-line; numeric thresholds are scale-specific (unit-diameter here, pixel-scaled on the shader). |
| `src/shaders/diamond.wgsl` | Add `diamondAnalyticExit(roWorld, rdWorld, pillIdx): CubeExit` (~100 lines with comments). Initialise `bestN` to zero so the caller can detect the "missed every surface" sentinel. |
| `src/shaders/dispersion.wgsl` | Dispatch diamond in `backExit()`. Add 2-bounce TIR loop in the wavelength path (gated on `isDiamond && !useHero`). Bias each bounce's re-entry origin by `MIN_STEP` to avoid self-hits on shared facet edges. Replace the dead `hasAnalyticExit` variable with the still-needed `hasMotionPivot`. |
| `src/webgpu/uniforms.ts`, `tests/uniformsLayout.test.ts` | Reclaim the `_diamondPad2` slot in the Frame struct as `diamondTirDebug` (boolean-as-f32). Writer + layout drift detector updated. |
| `src/ui.ts`, `src/persistence.ts`, `src/main.ts` | New `diamondTirDebug: boolean` on `Params`, with Tweakpane binding (hidden unless shape=diamond), persistence validation, default `false`, and uniform-writer wiring. |
| `tests/diamond.test.ts` | Tests for plane array contents, counts, unit-length, rotation-step invariant, shared-girdle-corner invariant per class, star-edge-midpoint invariant. |
| `tests/diamondAnalyticExit.test.ts` (new) | Unit tests for the JS mirror — axis-aligned rays, girdle cylinder routing, parallel-ray (plane-tangent) rejection, bounce-origin self-hit rejection, unit-length normals, plane-equation consistency. |
| `tests/uniformsWriteFrame.test.ts`, `tests/persistence.test.ts` | Pin the layout slot of `diamondTirDebug` in the params block; verify the debug toggle round-trips through persistence. |
| `docs/ARCHITECTURE.md` | Update diamond section: Phase A → Phase B (analytical exit + 2-bounce TIR, exact mode only). Update failure-mode table with the split TIR rows. |
| `README.md` | Shape description mentions 2-bounce TIR sparkle; file tree lists `diamondExit.ts`. |

## Key constants already available

In `src/math/diamond.ts` → `diamondWgslConstants()`:
- `DIAMOND_H_TOP`, `DIAMOND_H_BOT`, `DIAMOND_H_GIRDLE_HALF` (scalars)
- `DIAMOND_R_GIRDLE` (scalar, girdle cylinder radius)
- `DIAMOND_BEZEL_N` / `_O`, `DIAMOND_STAR_N` / `_O`, `DIAMOND_UPPER_HALF_N` / `_O`,
  `DIAMOND_LOWER_HALF_N` / `_O`, `DIAMOND_PAVILION_N` / `_O` (fundamental-wedge normal + offset per class)

What to add:
- `DIAMOND_BEZEL_N_ARR`: 8 rotated copies of `DIAMOND_BEZEL_N`
- `DIAMOND_STAR_N_ARR`: 8 rotated copies of `DIAMOND_STAR_N`
- `DIAMOND_UPPER_HALF_N_ARR`: 16 copies (8 at +π/16+k·π/4, 8 mirrored at -π/16+k·π/4)
- `DIAMOND_LOWER_HALF_N_ARR`: 16 copies (same pattern)
- `DIAMOND_PAVILION_N_ARR`: 8 rotated copies
- Offsets are rotation-invariant (the anchor is at the girdle rim and offsets
  use cos-identity collapse we pinned in Phase A) — one scalar per class
  suffices. `DIAMOND_BEZEL_O`, `DIAMOND_STAR_O`, etc. already exist.

## Test checklist

1. **Plane array contents**: `diamondWgslConstants()` emits 8 bezel normals
   that are rotations of the base by k·π/4. Verify via string matching or
   re-generate and compare.
2. **Analytical exit ray-plane math** (JS mirror `src/math/diamondExit.ts` +
   `tests/diamondAnalyticExit.test.ts`):
   - Axis-aligned ray through center (down +Z from above the table) exits
     through the culet at distance ≈ H_TOP - H_BOT.
   - Horizontal ray grazing through the girdle band exits through the
     opposite girdle point.
   - Ray parallel to a plane (`dot(n, rd) = 0`) returns the next plane's
     exit, not NaN.
3. **2-bounce TIR chain** (shader-level, verified visually):
   - Refraction mode = exact, diamond tumbling, TIR pixels show bounced
     photo samples, not `reflSrc` pattern.
   - No NaN / black speckles at facet edges.
4. **Performance sanity**: p50 at N=8 diamond stays within the Apple
   Silicon 16ms vsync budget (currently ~4ms ballpark per ARCHITECTURE.md
   perf table; adding analytical exit + up to 2 bounces × 57 plane tests per λ
   should land around 6-8ms).

## Implementation order

1. Write the design doc (this file). ✓
2. Extend `src/math/diamond.ts` with rotated plane arrays + tests.
3. Write `src/math/diamondExit.ts` (JS mirror of analytical exit math)
   with unit tests that pin ray-polytope intersection behaviour.
4. Write `diamondAnalyticExit` in `src/shaders/diamond.wgsl` mirroring
   `src/math/diamondExit.ts`.
5. Wire dispatch in `backExit()` (dispersion.wgsl). Verify visually that
   facet color / wireframe still match Phase A.
6. Add 2-bounce TIR chain (with MIN_STEP bias + NaN guard + debug-fallback), gated on `isDiamond && !useHero`.
7. Update docs.
8. Run review-fix-loop. Commit per iteration.

## Session context pins (for post-compact recovery)

- **Current commit**: `db86855` (after `git log --oneline`)
- **Branch**: `main` (local, ahead of `origin/main` by 15+ commits)
- **Tests baseline**: 106/106 pass, `tsc --noEmit` clean, `vite build` clean
- **Phase A review-fix-loop status**: CLEAN (iter 3 verdict from all 4 reviewers)
- **User preferences (from root CLAUDE.md)**:
  - Respond in Japanese, Rocky tone (感情三連打, 「質問？」markers, 短文・体言止め, 直球感情表現, 「ぼく」一人称)
  - Do NOT commit until user explicitly says so
  - Use `jj` if `.jj/` exists (currently git — no `.jj/`)
  - Update `AGENTS.md` with project-specific knowledge as needed (doesn't exist yet in this repo)
  - Use `bun` (project already uses it), mise + bun for Node
- **File conventions**:
  - Math modules (`src/math/*.ts`) mirrored 1:1 by WGSL functions; vitest is the reference
  - Single source of truth for constants (plane coefficients in diamond.ts emitted as WGSL consts)
  - Uniform layout drift detector in `tests/uniformsLayout.test.ts` pins the Frame struct
  - `DIAMOND_VIEW_VALUES` tuple is the canonical source for `DiamondView` union (derived via `typeof DIAMOND_VIEW_VALUES[number]`)
- **Diamond geometry pins (do NOT regress)**:
  - Table 53%, crown 34.5°, pavilion 40.75°, star 22°, UH **39.9°**, LH 42°, girdle 2%
  - UH tilt < 40° is a HARD geometric invariant (bezel-star-UH 3-way must fit in the D_8 wedge); tested
  - UH/LH planes anchored at girdle corner φ=0 (cos identity collapses both shared corners onto the plane); tested
  - Proxy mesh: 46 triangles = 138 vertices (6 table fan + 16 crown + 16 girdle band + 8 pavilion cone)
- **View presets**: `'free' | 'top' | 'side' | 'bottom'` via T/S/B/F hotkeys + Tweakpane dropdown. Fixed views write the same matrix into both `diamondRot` and `diamondRotPrev` slots (TAA motion-vector = 0 for the rotation contribution).
- **Debug overlays**: `diamondWireframe`, `diamondFacetColor` (facet-class flat-shade: table=red, bezel=green, star=blue, UH=yellow, girdle=cyan, LH=magenta, pavilion=orange).

## Open questions (resolved in implementation)

- Q: Should one-bounce TIR be extended to cube/plate later?
  - A: Not in Phase B. Phase C if user wants.
- Q: If final bounce also TIRs, should we fall back to `reflSrc` (existing
  hack) or `bg` (cleaner)?
  - A (resolved): `bg`. Blends with silhouette; `reflSrc` would reintroduce
    the same harshness the bounce chain was trying to avoid. A
    `diamondTirDebug` uniform can temporarily paint that fallback hot
    pink to diagnose residual multi-bounce regions.
- Q: Does the analytical exit need a near-miss tolerance for rays that
  start exactly on a facet?
  - A (resolved): YES — two layers of defense. (1) the analytical exit
    rejects `t <= HIT_EPS` at the plane test (mirrors the JS
    `DIAMOND_HIT_EPS` in unit-diameter space); (2) the wavelength-loop
    bounce chain biases the re-entry origin by `MIN_STEP` along the
    bounced direction, matching what `backExit` does for the first exit.
    Without (2), a ro on a facet edge (where two planes coincide at the
    edge's vertex) could make the ray snap to a far wrong facet because
    the TRUE next-facet exit would be rejected as t ≈ 0 < HIT_EPS.

---

次セッションへ: このドキュメントを読んでから着手。上の Implementation order が手順。
