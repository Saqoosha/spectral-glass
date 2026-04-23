// ---------- proxy vertex shader ----------
//
// Draws a per-pill 3D AABB (unit cube scaled to `halfSize`, optionally rotated
// for shape==cube). `sdfPill` / `sdfCube` fold rim radius into the field;
// their outer extent is `halfSize` (no extra proxy pad). The prism SDF is
// sharp and fits the same AABB, so the cube proxy is tight. Adding `edgeR`
// here was wrong for prisms and is no longer used in their SDF.

// Unit cube, 36 verts (= CUBE_PROXY_VERT_COUNT from src/math/diamond.ts),
// 12 tris, CCW outward winding (so `cullMode: 'back'` leaves one invocation
// per covered pixel). The array size must match CUBE_PROXY_VERT_COUNT, which
// the pipeline.ts draw call and the maxVerts guard below also read from.
const CUBE_VERTS: array<vec3<f32>, CUBE_PROXY_VERT_COUNT> = array<vec3<f32>, CUBE_PROXY_VERT_COUNT>(
  // +X face (CCW outward: swap V1,V2 vs the other faces' pattern because the
  // outward normal flips sign of cross(E1, E2) when the face is on the +X side)
  vec3<f32>( 1.0,-1.0,-1.0), vec3<f32>( 1.0, 1.0, 1.0), vec3<f32>( 1.0,-1.0, 1.0),
  vec3<f32>( 1.0,-1.0,-1.0), vec3<f32>( 1.0, 1.0,-1.0), vec3<f32>( 1.0, 1.0, 1.0),
  // -X face
  vec3<f32>(-1.0,-1.0,-1.0), vec3<f32>(-1.0,-1.0, 1.0), vec3<f32>(-1.0, 1.0, 1.0),
  vec3<f32>(-1.0,-1.0,-1.0), vec3<f32>(-1.0, 1.0, 1.0), vec3<f32>(-1.0, 1.0,-1.0),
  // +Y face
  vec3<f32>(-1.0, 1.0,-1.0), vec3<f32>(-1.0, 1.0, 1.0), vec3<f32>( 1.0, 1.0, 1.0),
  vec3<f32>(-1.0, 1.0,-1.0), vec3<f32>( 1.0, 1.0, 1.0), vec3<f32>( 1.0, 1.0,-1.0),
  // -Y face
  vec3<f32>(-1.0,-1.0,-1.0), vec3<f32>( 1.0,-1.0,-1.0), vec3<f32>( 1.0,-1.0, 1.0),
  vec3<f32>(-1.0,-1.0,-1.0), vec3<f32>( 1.0,-1.0, 1.0), vec3<f32>(-1.0,-1.0, 1.0),
  // +Z face
  vec3<f32>(-1.0,-1.0, 1.0), vec3<f32>( 1.0,-1.0, 1.0), vec3<f32>( 1.0, 1.0, 1.0),
  vec3<f32>(-1.0,-1.0, 1.0), vec3<f32>( 1.0, 1.0, 1.0), vec3<f32>(-1.0, 1.0, 1.0),
  // -Z face
  vec3<f32>(-1.0,-1.0,-1.0), vec3<f32>(-1.0, 1.0,-1.0), vec3<f32>( 1.0, 1.0,-1.0),
  vec3<f32>(-1.0,-1.0,-1.0), vec3<f32>( 1.0, 1.0,-1.0), vec3<f32>( 1.0,-1.0,-1.0),
);

// Perspective projection: a world point `p` seen through a pinhole camera at
// `(cx, cy, cameraZ)` looking down -Z, with the z=0 plane mapping to the
// screen exactly 1:1 in world pixels. A point at the screen plane projects to
// its own (x, y). A point closer to the camera than z=0 projects outward; a
// point beyond z=0 projects inward. The perspective divide is applied here in
// xy (not by the rasterizer), so the return always has w=1 for in-front
// vertices; w=-1 is used as a near-plane sentinel that the rasterizer clips.
fn projectWorld(p: vec3<f32>) -> vec4<f32> {
  let persp = frame.projection > 0.5;
  let camXY = frame.resolution * 0.5;

  var uv: vec2<f32>;
  if (persp) {
    let dz = frame.cameraZ - p.z;
    // `dz <= 0` means the vertex is at or behind the camera. Return w=-1 so
    // WebGPU's near-plane clipper drops the vertex; legitimate front-facing
    // vertices with tiny-but-positive dz still rasterize (threshold used to
    // be 1.0 which incorrectly clipped near-camera proxy corners at wide FOV).
    if (dz <= 0.0) {
      return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }
    uv = (p.xy - camXY) * (frame.cameraZ / dz) + camXY;
  } else {
    uv = p.xy;
  }
  let ndcX = 2.0 * uv.x / frame.resolution.x - 1.0;
  let ndcY = 1.0 - 2.0 * uv.y / frame.resolution.y;
  return vec4<f32>(ndcX, ndcY, 0.5, 1.0);
}

// Project a world point to screen-space pixel coordinates using the same
// pinhole model as projectWorld() but returning pixel units directly (no NDC
// flip). The z component carries a validity flag: > 0 = in front of camera,
// <= 0 = behind / at camera. Used by reprojectHit() below — the NDC return
// shape of projectWorld is awkward for "did this reprojection land inside
// the history texture?" checks.
fn worldToScreenPx(p: vec3<f32>) -> vec3<f32> {
  let camXY = frame.resolution * 0.5;
  if (frame.projection > 0.5) {
    let dz = frame.cameraZ - p.z;
    if (dz <= 0.0) {
      return vec3<f32>(0.0, 0.0, -1.0);
    }
    let px = (p.xy - camXY) * (frame.cameraZ / dz) + camXY;
    return vec3<f32>(px.x, px.y, 1.0);
  }
  return vec3<f32>(p.x, p.y, 1.0);
}

// TAA motion-vector reprojection. Given a world-space hit point on a rotating
// shape AND the unjittered fragment coordinate it came from, returns the
// screen UV where that SAME point on the shape was one frame ago — so the
// history read tracks the rotating surface instead of smearing stale
// neighbouring pixels into the output.
//
// Cube / plate both cache their previous rotation as a uniform (`cubeRotPrev`
// / `plateRotPrev`), so the math is: world → local (current rotation) → world
// (previous rotation) → screen. pill / prism don't rotate, so their hit
// point's prev screen position is just the current one — the caller passes
// in `fallbackUv` and we return it untouched (modulo the out-of-bounds check
// below, which catches disocclusion at the screen edge).
//
// Critical detail: `hitWorld` came from a JITTERED ray, so its raw screen
// projection isn't pixel-aligned. Using `projection(prevWorld) / resolution`
// directly would shift the history read by a fresh sub-pixel offset every
// frame, and bilinear filtering would compound a fractional-pixel blur into
// the history with each accumulation step (visible as "super blurry plates"
// when long-pause progressive averaging drives α toward 0).
//
// The fix below is the standard TAA technique: compute the motion DELTA in
// screen pixels (jitter cancels because it's present in both the current and
// previous projections) and add it to the unjittered FRAGCOORD. Static scenes
// then read history at exactly the pixel centre — no bilinear blur — and
// moving scenes still get the correct motion-corrected sample.
fn reprojectHit(hitWorld: vec3<f32>, fragCoord: vec2<f32>, pillIdx: u32, shapeId: i32, fallbackUv: vec2<f32>) -> vec2<f32> {
  var prevWorld = hitWorld;
  if (shapeId == 2) {
    let pill  = frame.pills[pillIdx];
    let local = frame.cubeRot * (hitWorld - pill.center);
    prevWorld = transpose(frame.cubeRotPrev) * local + pill.center;
  } else if (shapeId == 3) {
    let pill  = frame.pills[pillIdx];
    let local = frame.plateRot * (hitWorld - pill.center);
    prevWorld = transpose(frame.plateRotPrev) * local + pill.center;
  } else if (shapeId == 4) {
    // Diamond: same trick as cube/plate. The fold inside sdfDiamond is a
    // non-linear symmetry operation, but reprojection only cares about the
    // surface's rigid-body motion (rotation + translation), which `diamondRot`
    // captures fully. Reading history via `transpose(diamondRotPrev)` pulls
    // each spinning facet's own pixel from the previous frame.
    let pill  = frame.pills[pillIdx];
    let local = frame.diamondRot * (hitWorld - pill.center);
    prevWorld = transpose(frame.diamondRotPrev) * local + pill.center;
  }

  let prevPx = worldToScreenPx(prevWorld);
  let currPx = worldToScreenPx(hitWorld);
  // Behind-camera fallback: if the rotation moved the hit point through the
  // near plane between frames (rare — needs a fast tumble + perspective
  // mode), there's no meaningful history pixel to reproject from. Falling
  // back to `fallbackUv` reads stale history at the current pixel; under
  // the steady-state α = 0.2 EMA that washes out in ~5 frames. The pause
  // path can't hit this because plateRot == plateRotPrev when paused, so
  // the reprojection collapses to identity and prevPx == currPx — safe.
  if (prevPx.z <= 0.0 || currPx.z <= 0.0) { return fallbackUv; }

  // `currPx` carries the sub-pixel jitter of the original ray, and
  // `prevPx` is the SAME world point reprojected to the previous frame's
  // camera — both share that jitter, so it cancels in the delta to first
  // order. Strictly the perspective divide is non-linear in p.z, so there's
  // a residual of order (jitter × per-frame rotation × perspective factor),
  // ≤ 0.5 px × ≈ 0.005 rad ≈ < 0.01 px at typical FOVs — well below the
  // pixel grid, invisible after EMA. (Adding to the unjittered `fragCoord`
  // — not to `currPx` — is the part that actually pins the read to the
  // pixel grid for static scenes.)
  let motionPx = prevPx.xy - currPx.xy;
  let prevUv   = (fragCoord + motionPx) / frame.resolution;
  // Disocclusion: the point was outside the screen a frame ago. Same
  // fallback strategy — read stale and let EMA fade in fresh data.
  if (any(prevUv < vec2<f32>(0.0)) || any(prevUv > vec2<f32>(1.0))) {
    return fallbackUv;
  }
  return prevUv;
}

@vertex
fn vs_proxy(
  @builtin(vertex_index)   vi: u32,
  @builtin(instance_index) ii: u32,
) -> @builtin(position) vec4<f32> {
  if (ii >= u32(frame.pillCount)) {
    // Over pillCount → degenerate position so the triangle is clipped.
    return vec4<f32>(2.0, 2.0, 0.5, 1.0);
  }
  let pill    = frame.pills[ii];
  let shapeId = i32(frame.shape + 0.5);

  // Per-shape vertex budget: CUBE_PROXY_VERT_COUNT for cube/pill/prism/plate
  // (the CUBE_VERTS array size above), DIAMOND_PROXY_VERT_COUNT for diamond.
  // Both constants are injected from src/math/diamond.ts so the draw call
  // (pipeline.ts), this guard, and the CUBE_VERTS array literal all track
  // the same TS numbers. A Phase B mesh change updates the TS constants
  // (or, for diamond, the mesh body in diamond.wgsl) — the guard and draw
  // count follow automatically. The guard prevents CUBE_VERTS out-of-bounds
  // access for non-diamond shapes when vi ≥ CUBE_PROXY_VERT_COUNT; past
  // DIAMOND_PROXY_VERT_COUNT the diamond branch has no valid vertex either.
  let maxVerts = select(CUBE_PROXY_VERT_COUNT, DIAMOND_PROXY_VERT_COUNT, shapeId == 4);
  if (vi >= maxVerts) {
    return vec4<f32>(2.0, 2.0, 0.5, 1.0);
  }

  // Unit cube corner in [-1, 1]^3 → local AABB `±halfSize`. Matches the
  // true shape's outer bounds; `edgeR` must not be added (see file header).
  var corner: vec3<f32>;
  if (shapeId == 3) {
    // Plate: box sized to the square face + wave-amplitude margin on the Z
    // (thickness) axis so the rippling midsurface never pokes through. Then
    // apply the plate's current rotation so the proxy tracks the tumble —
    // same tight-bounding trick as the cube path below.
    let extent = vec3<f32>(pill.halfSize.x,
                           pill.halfSize.x,
                           pill.halfSize.z + frame.waveAmp);
    corner = transpose(frame.plateRot) * (CUBE_VERTS[vi] * extent);
  } else if (shapeId == 4) {
    // Diamond: exact convex-hull proxy mesh (DIAMOND_PROXY_VERT_COUNT
    // vertices = 46 triangles — see diamondProxyVertex in diamond.wgsl
    // for the topology breakdown). The split keeps the geometry details
    // next to sdfDiamond where future diamond-only trace work can land.
    //
    // vi < DIAMOND_PROXY_VERT_COUNT is guaranteed by the maxVerts guard
    // at the top of vs_proxy.
    let local = diamondProxyVertex(vi, frame.diamondSize);
    corner    = transpose(frame.diamondRot) * local;
  } else {
    let extent = pill.halfSize;
    corner     = CUBE_VERTS[vi] * extent;
    if (shapeId == 2) {
      // The shader defines the cube via `local = rot * (p - center)`, so a
      // world-space proxy corner that maps to the unit-cube local-space
      // corner `c` is `center + transpose(rot) * (c * extent)`.
      corner = transpose(frame.cubeRot) * corner;
    }
  }
  return projectWorld(pill.center + corner);
}
