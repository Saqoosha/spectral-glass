/**
 * Round brilliant cut diamond — Tolkowsky "ideal" proportions.
 *
 * The SDF lives in WGSL (`sdfDiamond` in src/shaders/dispersion.wgsl) and uses
 * 8-fold (D_8) fundamental-domain folding to reduce 58 facets to 7 distance
 * terms evaluated per pixel. This module is the single source of truth for:
 *
 *   1. The Tolkowsky constants (table ratio, crown angle, pavilion angle, etc.)
 *   2. The derived facet plane normals/offsets (7 of them, unit-diameter scale)
 *   3. The `diamondRot` rotation uniform builder (Y-axis spin + 20° forward
 *      tilt, driven by `sceneTime`, mirrors cube.ts / plate.ts)
 *   4. The WGSL `const` block injected into the shader at pipeline build time
 *      (see src/webgpu/pipeline.ts) so host-derived plane coefficients land on
 *      the GPU without a drift-prone hand-copy.
 *
 * Coordinate system (same as sceneSdf): +Z up, girdle band centred on z=0.
 * Crown (table side) sits at +Z, pavilion (culet side) at -Z.
 */

/** Bounds for the `diamondSize` slider (pixels = girdle diameter). Exported so
 *  persistence validation clamps hand-edited storage to the same range the UI
 *  slider accepts. 50 px keeps the total ~30 px tall on screen — small but
 *  still readable as a brilliant cut. 400 px is the upper bound before the
 *  shape starts crowding a typical 1080p viewport. */
export const DIAMOND_SIZE_MIN = 50;
export const DIAMOND_SIZE_MAX = 400;

// ---------- Tolkowsky ideal cut (1919) ----------

/** Table size as a fraction of the girdle diameter, measured vertex-to-vertex
 *  across the octagonal table — the DIAGONAL across opposite table corners.
 *  This is GIA's convention: "table percentage" averages four measurements
 *  from bezel point to bezel point, where the bezel points ARE the table's
 *  octagonal vertices. So TABLE_RATIO = 2·R_TABLE_VERTEX / diameter.
 *
 *  The apothem (flat-to-flat, perpendicular edge distance) derives as
 *  R_TABLE_APOTHEM = R_TABLE_VERTEX · cos(π/8) ≈ 0.924 · R_TABLE_VERTEX.
 *
 *  Tolkowsky's 1919 "ideal cut" prescribes 53 % here; GIA's "Excellent"
 *  grade spans 52–62 %. */
const TABLE_RATIO = 0.53;
/** Crown main (bezel) facet angle measured from the horizontal girdle plane. */
const CROWN_ANGLE_DEG = 34.5;
/** Pavilion main facet angle measured from the horizontal girdle plane. */
const PAVILION_ANGLE_DEG = 40.75;
/** Star facet inclination from horizontal. Flatter than the bezel — classic
 *  "ideal" star length of ~50% implies ≈22°. */
const STAR_ANGLE_DEG = 22.0;
/** Upper half (upper girdle) facet inclination from horizontal. Hard upper
 *  limit is 40° for THIS anchor/normal setup: at α = 40° the bezel-star-UH
 *  three-way junction lands exactly on the φ = π/8 mirror (the D_8 wedge
 *  boundary); at α > 40° the junction escapes the wedge, the bezel kite
 *  can't close, and a green bezel sliver appears between star and upper-half.
 *
 *  39.9° pushes the tilt as close to that ceiling as the numerics tolerate.
 *  The effect on the rendered shape: the UH-star touch point (the star's
 *  lower tip on the crown) drops to z ≈ 0.125 — about 71 % of the way from
 *  the girdle up to the table plane, matching classic ideal-cut references.
 *  Lower α would raise this tip toward the table and make the star look
 *  short / stubby; higher α can't fit inside the wedge.
 *
 *  Real brilliant cuts often cite UH ≈ 42° but under different anchor
 *  conventions (e.g. UH anchored partway along the bezel-UH kite edge
 *  rather than on the actual girdle rim), so those numbers don't transfer
 *  directly to this geometry. */
const UPPER_HALF_ANGLE_DEG = 39.9;
/** Lower half (lower girdle) facet inclination from horizontal. Must stay
 *  above PAVILION_ANGLE_DEG (40.75°) — below that the LH plane lies INSIDE
 *  the pavilion main and never surfaces, so the facet vanishes. 42° sits
 *  just above the limit, matching GIA "Excellent" lower-half proportions
 *  (LH apex at ~75 % pavilion depth, leaving room for a visible pavilion
 *  main near the culet). Unlike the upper half, there's no "star" equivalent
 *  on the pavilion side — the LH plane only has to stay valid against the
 *  pavilion and the wedge mirror, so the 40° cap that constrains UH doesn't
 *  apply here. */
const LOWER_HALF_ANGLE_DEG = 42.0;
/** Girdle band height as a fraction of girdle diameter. 2% is "medium" —
 *  within GIA's acceptable range for commercial cuts. */
const GIRDLE_THICKNESS_RATIO = 0.02;

// Angular positions (azimuthal) of each facet class's centreline. Matches
// the standard 96-index-wheel layout for a round brilliant cut (pavilion
// mains on indices 0-12-24-…-84, stars on 6-18-30-…-90 — the stars staggered
// between the mains):
//
//   - Table orientation: its 8 VERTICES sit at φ = k·π/4 (= the bezel /
//     pavilion-main azimuths), its 8 EDGES sit between vertices at
//     φ = π/8 + k·π/4 (= the star azimuths).
//   - BEZEL kite: top corner at a TABLE VERTEX, bottom corner on the
//     girdle at the same azimuth. Centreline φ_bezel = 0.
//   - PAVILION main: aligned with the bezel above it — same azimuth,
//     same "bezel point to bezel point" GIA measurement. φ_pavilion = 0.
//   - STAR triangle: base spans a TABLE EDGE (between two adjacent
//     vertices at φ = 0 and φ = π/4). Centreline φ_star = π/8, midway
//     between those vertices — the edge midpoint direction.
//   - UPPER / LOWER GIRDLE HALVES flank the mains at ±π/16; a single
//     representative in the fundamental wedge sits at φ = π/16.
//
// After the 1/16 D_8 fold the fundamental wedge is [0, π/8]. Bezel and
// pavilion land ON the lower wedge boundary (φ = 0, the mirror line);
// star lands on the upper boundary (φ = π/8). Upper / lower halves sit
// mid-wedge. Placing every facet on a mirror line (or midway between)
// means each plane's outward normal is symmetric under the fold, so
// the fold never evaluates a plane at a point it doesn't cover.
const PHI_BEZEL      = 0;
const PHI_PAVILION   = 0;
const PHI_STAR       = Math.PI / 8;
const PHI_UPPER_HALF = Math.PI / 16;
const PHI_LOWER_HALF = Math.PI / 16;

// ---------- derived unit-diameter dimensions ----------

const DEG = Math.PI / 180;
const CROWN_ANGLE      = CROWN_ANGLE_DEG      * DEG;
const PAVILION_ANGLE   = PAVILION_ANGLE_DEG   * DEG;
const STAR_ANGLE       = STAR_ANGLE_DEG       * DEG;
const UPPER_HALF_ANGLE = UPPER_HALF_ANGLE_DEG * DEG;
const LOWER_HALF_ANGLE = LOWER_HALF_ANGLE_DEG * DEG;

const R_GIRDLE        = 0.5;                                         // girdle outer radius at unit diameter
// TABLE_RATIO is vertex-to-vertex (GIA "bezel point to bezel point", see
// constant doc above), so it's the radius to the table's octagonal CORNERS.
// The flat-to-flat apothem derives as r_apothem = r_vertex · cos(π/8).
const R_TABLE_VERTEX  = R_GIRDLE * TABLE_RATIO;                      // radius to a table octagon corner (Tolkowsky "table %")
const R_TABLE_APOTHEM = R_TABLE_VERTEX * Math.cos(Math.PI / 8);      // half the flat-to-flat table width

/** Crown height from girdle-top to table plane. The bezel axis runs from
 *  the table VERTEX at (R_TABLE_VERTEX·cos(π/8), …, H_TOP) down to the
 *  girdle-top POINT at (R_GIRDLE·cos(π/8), …, H_GIRDLE_HALF) — both at the
 *  same φ = π/8. Horizontal run along that axis is R_GIRDLE −
 *  R_TABLE_VERTEX, and the vertical drop is H_CROWN, so
 *  tan(CROWN_ANGLE) = H_CROWN / (R_GIRDLE − R_TABLE_VERTEX).
 *  With TABLE_RATIO = 0.53 and CROWN_ANGLE = 34.5°, H_CROWN ≈ 0.162 — the
 *  classic Tolkowsky value. */
const H_CROWN    = (R_GIRDLE - R_TABLE_VERTEX) * Math.tan(CROWN_ANGLE);
/** Pavilion depth from girdle-bottom to culet apex. At ~0.431 · diameter the
 *  pavilion sits well past the diameter's 40% mark where TIR behaviour is
 *  optimal — Phase B will exercise this. */
const H_PAVILION = R_GIRDLE * Math.tan(PAVILION_ANGLE);

/** Half-thickness of the thin girdle band (the straight cylindrical side). */
const H_GIRDLE_HALF = GIRDLE_THICKNESS_RATIO / 2;

/** Z-coordinate of the table plane (top of diamond). */
const H_TOP = H_GIRDLE_HALF + H_CROWN;
/** Z-coordinate of the culet apex (pointed bottom of diamond, negative). */
const H_BOT = -(H_GIRDLE_HALF + H_PAVILION);

/** Total diamond height / diameter (H_TOP − H_BOT ≈ 0.61).
 *
 *  Not consumed directly by the proxy mesh or the SDF — the proxy synthesises
 *  its vertices from H_TOP and H_BOT individually (see diamondProxyVertex in
 *  src/shaders/diamond.wgsl), and the SDF folds into a fundamental wedge that
 *  doesn't need a total-height scalar. This constant lives on as a
 *  regression pin: diamond.test.ts asserts it stays in the realistic 0.55–
 *  0.65 range, so drift in TABLE_RATIO / CROWN_ANGLE / PAVILION_ANGLE is
 *  caught before the rendered shape stops reading as a brilliant cut. */
export const DIAMOND_HEIGHT_RATIO = H_TOP - H_BOT;   // H_TOP + |H_BOT|

// ---------- facet plane derivation (unit-diameter scale) ----------

/** A facet plane in the form `dot(p, n) = offset`. `n` is unit length;
 *  `offset` is in units of diameter (scales at evaluation time by the
 *  runtime `diamondSize` uniform). Stored as a plain record instead of a
 *  Float32Array so individual fields are grep-friendly in the WGSL-const
 *  generator below. */
type FacetPlane = {
  readonly nx: number; readonly ny: number; readonly nz: number;
  readonly offset: number;
};

/** Build a facet plane from a centreline azimuth (φ in radians), inclination
 *  (α from horizontal, radians), and a point the plane must pass through. For
 *  crown facets `vSign = +1` (upward-tilted); for pavilion facets `vSign = -1`.
 *
 *  Normal derivation: rotate the face-perpendicular vector into 3D. A facet
 *  tilted `α` above horizontal, centred at azimuth φ on the outward side,
 *  has outward normal (cos φ · sin α, sin φ · sin α, vSign · cos α). At α=0
 *  (flat face) the normal points straight up (+Z) regardless of φ; at α=π/2
 *  (vertical face) the normal points radially outward with no Z component. */
function planeFromAngles(
  phi:    number,
  alpha:  number,
  vSign:  1 | -1,
  anchor: readonly [number, number, number],
): FacetPlane {
  const nx = Math.cos(phi) * Math.sin(alpha);
  const ny = Math.sin(phi) * Math.sin(alpha);
  const nz = vSign * Math.cos(alpha);
  const offset = anchor[0] * nx + anchor[1] * ny + anchor[2] * nz;
  return { nx, ny, nz, offset };
}

// Anchors chosen for each facet class so the plane passes through a
// physically plausible point on the diamond (girdle rim, table vertex, etc.).
// See the φ_* constants above for the geometric rationale behind each
// facet's centreline orientation.

/** Bezel (crown main) — kite axis runs from a TABLE VERTEX at φ = 0
 *  (R_TABLE_VERTEX, 0, H_TOP) down to a GIRDLE POINT at the same azimuth
 *  (R_GIRDLE, 0, H_GIRDLE_HALF). By H_CROWN's derivation the plane passes
 *  through both anchor choices — we pick the girdle point. */
const BEZEL_PLANE:   FacetPlane = planeFromAngles(PHI_BEZEL,   CROWN_ANGLE,      +1,
  [R_GIRDLE, 0, H_GIRDLE_HALF]);

/** Star — its BASE lies on a TABLE EDGE (the segment between two adjacent
 *  table vertices at φ = 0 and φ = π/4). The star plane is centred on the
 *  edge midpoint direction (φ = π/8). Anchor at the edge midpoint itself:
 *  (R_TABLE_APOTHEM·cos(π/8), R_TABLE_APOTHEM·sin(π/8), H_TOP). The cos²+
 *  sin² collapse in planeFromAngles' offset computation reduces this to
 *  R_TABLE_APOTHEM · sin(α_star) + H_TOP · cos(α_star). */
const STAR_PLANE:    FacetPlane = planeFromAngles(PHI_STAR,    STAR_ANGLE,       +1,
  [R_TABLE_APOTHEM * Math.cos(PHI_STAR), R_TABLE_APOTHEM * Math.sin(PHI_STAR), H_TOP]);

/** Upper half — anchored at the girdle-top CORNER at φ=0 (shared with the
 *  adjacent bezel and the mirrored neighbour UH across it). With the normal
 *  centred at φ=π/16, the plane automatically also passes through the
 *  girdle-top corner at φ=π/8 because the chord between the two corners is
 *  perpendicular to the facet's centreline (cos(π/8 − π/16) = cos(π/16)
 *  identity collapses both anchor candidates to the same offset). So this
 *  single anchor pins BOTH girdle-rim corners the facet's bottom edge has
 *  to touch — anchoring at φ=π/16 on a circle of radius R_GIRDLE (the
 *  earlier convention) would place the plane off the actual girdle chord,
 *  outside the rim at both φ=0 and φ=π/8, and leave visible gaps where
 *  adjacent UHs fail to meet at the shared corners. (Not to be confused
 *  with the proxy mesh's "circumscribing octagon" at R_GIRDLE/cos(π/8) —
 *  that's a different octagon, used only for the coverage-safe proxy
 *  silhouette.) */
const UPPER_HALF_PLANE: FacetPlane = planeFromAngles(PHI_UPPER_HALF, UPPER_HALF_ANGLE, +1,
  [R_GIRDLE, 0, H_GIRDLE_HALF]);

/** Lower half — mirror of the upper half across the girdle plane. Anchored
 *  at the girdle-bottom corner at φ=0 for the same reason: the plane is
 *  pinned through the two shared girdle-rim corners at φ=0 and φ=π/8, and
 *  an anchor at φ=π/16 would push it off the girdle circle. */
const LOWER_HALF_PLANE: FacetPlane = planeFromAngles(PHI_LOWER_HALF, LOWER_HALF_ANGLE, -1,
  [R_GIRDLE, 0, -H_GIRDLE_HALF]);

/** Pavilion main — shares the bezel's azimuth (φ = 0, through the table
 *  vertex above) as is standard for a round brilliant. The plane passes
 *  through the girdle-bottom POINT and converges toward the culet at
 *  (0, 0, H_BOT); by H_PAVILION = R_GIRDLE · tan(PAVILION_ANGLE) the plane
 *  exactly hits the culet too. */
const PAVILION_PLANE: FacetPlane = planeFromAngles(PHI_PAVILION, PAVILION_ANGLE,   -1,
  [R_GIRDLE, 0, -H_GIRDLE_HALF]);

// ---------- rotation uniform ----------

/** Forward tilt of the diamond (radians). 20° leaves the table's silhouette
 *  visible from above while still exposing the crown facets — the standard
 *  "jewelry display" pose. Negative sign so the table tilts toward the
 *  camera (camera looks down -Z so +X rotation tilts front-edge downward). */
const DIAMOND_TILT = -20 * DEG;
/** Y-axis spin rate (rad/s). 0.30 matches the cube/plate rates for visual
 *  consistency without being dizzying. */
const DIAMOND_SPIN_RATE = 0.30;

/** Canonical list of valid DiamondView values, exported as a readonly tuple
 *  so the runtime allow-list (persistence.ts) and the compile-time union
 *  below derive from ONE source of truth. Adding a new preset is a single-
 *  site change: append here and add the matrix branch in
 *  diamondViewRotationColumns. TypeScript's exhaustive-`never` guard in
 *  that function will flag the missing branch at compile time. */
export const DIAMOND_VIEW_VALUES = ['free', 'top', 'side', 'bottom'] as const;

/** Preset view angles for the "click / hotkey to rotate to a canonical
 *  pose" flow. `free` is the default (tumble via `diamondRotationColumns`);
 *  the fixed poses pin the diamond so facet geometry can be cross-checked
 *  against a reference illustration without waiting for the right frame of
 *  the tumble animation.
 *
 *  Convention (world +Z toward camera, +Y up on screen):
 *    - `top`    — diamond's table toward the camera (identity rotation);
 *                 symmetry axis aligned with world +Z.
 *    - `side`   — diamond's symmetry axis rotated onto world +Y (vertical
 *                 on screen), so the girdle band is seen edge-on as a
 *                 horizontal line across the middle (profile view).
 *                 Implementation: R_x(-π/2).
 *    - `bottom` — culet toward the camera (R_x(π) flip). */
export type DiamondView = typeof DIAMOND_VIEW_VALUES[number];

/**
 * Fixed-view rotation matrix columns in the same WGSL-padded 12-float layout
 * as `diamondRotationColumns`. Callers pair this with a `view !== 'free'`
 * guard and pass the same result as both the current AND previous-frame
 * rotation — a fixed pose zeroes the ROTATION contribution to the TAA
 * reprojection motion vector, which is what we want when the shape itself
 * is frozen. Pill translation + camera motion still contribute their own
 * motion-vector components through other paths.
 */
export function diamondViewRotationColumns(view: Exclude<DiamondView, 'free'>): Float32Array {
  const out = new Float32Array(12);
  // Row-major 3x3 expressed inline per preset, then written into the
  // column-major + per-column-pad layout the WGSL mat3x3<f32> expects.
  let m00 = 1, m01 = 0, m02 = 0;
  let m10 = 0, m11 = 1, m12 = 0;
  let m20 = 0, m21 = 0, m22 = 1;
  if (view === 'top') {
    // Identity — baseline values above already encode R_I. Kept as an
    // explicit branch so the exhaustive enumeration is visible here;
    // adding a new preset means editing ONE place.
  } else if (view === 'side') {
    // R_x(-π/2): local +Z → world +Y (table up), local +Y → world -Z (into screen).
    m10 = 0; m11 = 0; m12 = 1;
    m20 = 0; m21 = -1; m22 = 0;
  } else if (view === 'bottom') {
    // R_x(π): local +Z → world -Z (culet toward camera), local +Y → world -Y.
    m11 = -1;
    m22 = -1;
  } else {
    // Unreachable given the `Exclude<DiamondView, 'free'>` parameter type,
    // but WGSL-fallthrough would silently return an identity matrix — wrong
    // for any pose except `'top'`. Fail fast so a future caller that bypasses
    // the type narrowing (e.g. a new preset added to the union without a
    // branch here) sees a clear error instead of a visual regression.
    // Same spirit as the `!Number.isFinite(time)` guard in
    // diamondRotationColumns below.
    const exhaustive: never = view;
    throw new Error(`diamondViewRotationColumns: unknown view ${String(exhaustive)}`);
  }
  // Column 0 — image of the X basis
  out[0]  = m00; out[1]  = m10; out[2]  = m20;
  // Column 1 — image of the Y basis
  out[4]  = m01; out[5]  = m11; out[6]  = m21;
  // Column 2 — image of the Z basis
  out[8]  = m02; out[9]  = m12; out[10] = m22;
  return out;
}

/**
 * Diamond rotation matrix columns in WGSL-padded layout (12 floats, same
 * format as cubeRotationColumns / plateRotationColumns).
 *
 * Composition: Rx(DIAMOND_TILT) · Ry(time · DIAMOND_SPIN_RATE). The spin is
 * applied first (local Y-axis) so the tilt stays fixed — spinning a tilted
 * diamond, not tilting a spinning one. Mirrors the "rx · ry" order plate uses.
 */
export function diamondRotationColumns(time: number): Float32Array {
  // Same NaN/±Infinity guard as cube/plate: a poisoned rotation matrix slips
  // into the GPU uniform and gets silently "healed" by sceneNormal's
  // degenerate-gradient fallback, producing visual corruption without a
  // hard failure. Fail fast while the caller is still on the stack.
  if (!Number.isFinite(time)) {
    throw new Error(`diamondRotationColumns: time must be finite, got ${time}`);
  }
  const ay = time * DIAMOND_SPIN_RATE;
  const ax = DIAMOND_TILT;
  const cy = Math.cos(ay);
  const sy = Math.sin(ay);
  const cx = Math.cos(ax);
  const sx = Math.sin(ax);

  // R = Rx(ax) · Ry(ay). Row-major:
  //   [  cy          0       sy      ]
  //   [  sx·sy       cx     -sx·cy   ]
  //   [ -cx·sy       sx      cx·cy   ]
  //
  // WGSL mat3x3 is column-major with 16-B column stride, so column k lives at
  // indices [k*4, k*4+3); index k*4+3 is the per-column pad (left at 0).
  const out = new Float32Array(12);
  // Column 0 — image of the X basis
  out[0]  = cy;        out[1]  = sx * sy;   out[2]  = -cx * sy;
  // Column 1 — image of the Y basis
  out[4]  = 0;         out[5]  = cx;        out[6]  = sx;
  // Column 2 — image of the Z basis
  out[8]  = sy;        out[9]  = -sx * cy;  out[10] = cx * cy;
  // out[3], out[7], out[11] stay at 0 — required per-column padding.
  return out;
}

// ---------- WGSL const block (injected into shader source) ----------

/** Format a float as a WGSL numeric literal with enough precision (9 digits is
 *  more than float32 resolution and keeps the generated text readable). */
function fwgsl(v: number): string {
  return Number.isInteger(v) ? `${v}.0` : v.toPrecision(9);
}

function emitPlaneConst(prefix: string, p: FacetPlane): string {
  return `const ${prefix}_N: vec3<f32> = vec3<f32>(${fwgsl(p.nx)}, ${fwgsl(p.ny)}, ${fwgsl(p.nz)});\n`
       + `const ${prefix}_O: f32       = ${fwgsl(p.offset)};\n`;
}

/**
 * Proxy mesh triangle/vertex count — the single source of truth that both
 * the host draw call (src/webgpu/pipeline.ts) and the WGSL proxy vertex
 * shader (dispersion.wgsl's `maxVerts` guard + diamond.wgsl's
 * `diamondProxyVertex` bound check) read from. A mesh-topology change
 * touches two places: `DIAMOND_PROXY_TRI_COUNT` below AND the
 * `diamondProxyVertex` body in src/shaders/diamond.wgsl. The draw count
 * (pipeline.ts) and WGSL `maxVerts` guard read from the TS constant, so
 * they auto-follow — no drift risk.
 *
 * Mesh topology: 6 table fan + 16 crown trapezoids + 16 girdle band
 * (2 rings × 8 quads, covers the cylindrical girdle thickness that a
 * single-ring mesh misses at edge midpoints) + 8 pavilion cone = 46 tri.
 */
export const DIAMOND_PROXY_TRI_COUNT  = 46;
export const DIAMOND_PROXY_VERT_COUNT = DIAMOND_PROXY_TRI_COUNT * 3;  // 138

/**
 * Cube-topology proxy vertex count — 12 triangles × 3 = 36. Shared across
 * pill/prism/cube/plate via the `CUBE_VERTS` array in dispersion.wgsl.
 * Exported alongside the diamond counts so every "proxy vertex budget"
 * literal in the project has one canonical definition; a future change to
 * the cube mesh updates three consumers (pipeline.ts draw call, WGSL
 * array size, WGSL maxVerts guard) from one place.
 */
export const CUBE_PROXY_VERT_COUNT = 36;

/**
 * WGSL `const` declarations for the diamond SDF plane table. Prepended to the
 * dispersion shader source in src/webgpu/pipeline.ts so the shader reads
 * from module-scope constants rather than uniforms — zero uniform bandwidth,
 * no drift risk between TS-computed numbers and shader-embedded numbers.
 *
 * All plane offsets are in "units of diameter"; the shader multiplies each by
 * `frame.diamondSize` at SDF-eval time so the runtime diameter slider scales
 * the whole shape uniformly.
 */
export function diamondWgslConstants(): string {
  // Proxy mesh tightly wraps the diamond's convex hull: 8-gon table on top,
  // 8 crown trapezoids, 8 girdle band quads (covers the thin cylindrical
  // girdle that a single-ring mesh would miss at octagon edge midpoints),
  // 8 pavilion-cone triangles down to the culet. Total 46 triangles = 138
  // vertices. Octagonal approximations use a CIRCUMSCRIBING octagon
  // (radius = R_GIRDLE/cos(π/8)) so the inscribed circle through edge
  // midpoints stays tangent to the true girdle cylinder — zero
  // under-coverage at edge midpoints, 8.2 % over-coverage at octagon
  // corners only.
  //
  // DIAMOND_R_TABLE_VERTEX is the radius from the table octagon's centre
  // to its corners. The table's APOTHEM (= R_GIRDLE·TABLE_RATIO per the
  // flat-to-flat Tolkowsky convention) is the perpendicular half-width
  // and relates as r_apothem = r_vertex · cos(π/8).
  //
  // DIAMOND_GIRDLE_R_CIRC = R_GIRDLE / cos(π/8) is the radius to the
  // circumscribing octagon's vertices — the smallest octagon that fully
  // contains the girdle cylinder.
  //
  // DIAMOND_H_GIRDLE_HALF is the half-thickness of the girdle cylindrical
  // band (z ∈ [-H_GIRDLE_HALF, +H_GIRDLE_HALF]), needed by the proxy's
  // top- and bottom- girdle rings that flank the band.
  const girdleRCirc = R_GIRDLE / Math.cos(Math.PI / 8);
  return [
    '// ---- generated by src/math/diamond.ts — do not edit here ----',
    `const DIAMOND_H_TOP:             f32 = ${fwgsl(H_TOP)};`,
    `const DIAMOND_H_BOT:             f32 = ${fwgsl(H_BOT)};`,
    `const DIAMOND_H_GIRDLE_HALF:     f32 = ${fwgsl(H_GIRDLE_HALF)};`,
    `const DIAMOND_R_GIRDLE:          f32 = ${fwgsl(R_GIRDLE)};`,
    `const DIAMOND_R_TABLE_VERTEX:    f32 = ${fwgsl(R_TABLE_VERTEX)};`,
    `const DIAMOND_GIRDLE_R_CIRC:     f32 = ${fwgsl(girdleRCirc)};`,
    `const CUBE_PROXY_VERT_COUNT:     u32 = ${CUBE_PROXY_VERT_COUNT}u;`,
    `const DIAMOND_PROXY_VERT_COUNT:  u32 = ${DIAMOND_PROXY_VERT_COUNT}u;`,
    emitPlaneConst('DIAMOND_BEZEL',      BEZEL_PLANE),
    emitPlaneConst('DIAMOND_STAR',       STAR_PLANE),
    emitPlaneConst('DIAMOND_UPPER_HALF', UPPER_HALF_PLANE),
    emitPlaneConst('DIAMOND_LOWER_HALF', LOWER_HALF_PLANE),
    emitPlaneConst('DIAMOND_PAVILION',   PAVILION_PLANE),
    '// ---- end generated ----',
    '',
  ].join('\n');
}

// ---------- exports for tests ----------

/**
 * Internal values exported strictly for regression tests (diamond.test.ts).
 * Not a stable API — fields may appear or disappear as the geometry evolves.
 * Tests use these so they can cross-check numerics without re-deriving them.
 *
 * `Object.freeze` is applied at both levels so a bad test can't mutate the
 * shared object and poison later tests in the same process. `as const` is
 * the compile-time readonly; the runtime freeze is belt-and-suspenders.
 */
const DIAMOND_INTERNALS_MUT = {
  R_GIRDLE,
  R_TABLE_VERTEX,
  R_TABLE_APOTHEM,
  H_CROWN,
  H_PAVILION,
  H_GIRDLE_HALF,
  H_TOP,
  H_BOT,
  CROWN_ANGLE_DEG,
  PAVILION_ANGLE_DEG,
  // R_CIRC = girdle circumscribing octagon radius — used by the proxy mesh;
  // exported so tests can pin `R_CIRC = R_GIRDLE / cos(π/8)` and the
  // circumscribing-octagon slack math doesn't drift silently.
  GIRDLE_R_CIRC: R_GIRDLE / Math.cos(Math.PI / 8),
  planes: Object.freeze({
    bezel:      BEZEL_PLANE,
    star:       STAR_PLANE,
    upperHalf:  UPPER_HALF_PLANE,
    lowerHalf:  LOWER_HALF_PLANE,
    pavilion:   PAVILION_PLANE,
  }),
} as const;
export const DIAMOND_INTERNALS = Object.freeze(DIAMOND_INTERNALS_MUT);
