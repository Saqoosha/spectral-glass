// Post-process pipelines: passthrough and FXAA. Both sample the scene's
// linear-rgba16f intermediate and emit display-encoded output to the
// swapchain. FXAA runs on perceptual-space luma for accurate edge
// detection, but blends color in linear space (physically correct).
//
// Uses the same vertex-index fullscreen-triangle trick as
// fullscreen.wgsl's vs_main; the vertex code is copied here (not
// shared) so this module stays self-contained — different bind-group
// layout, and `Vout` carries an explicit `uv` that `VsOut` in
// fullscreen.wgsl would need to gain too. UV is V-flipped so (0,0)
// lands at the top-left like the DOM origin.

struct Post {
  // f32 flag: 1 = apply sRGB OETF manually (swapchain is non-sRGB, e.g.
  // bgra8unorm); 0 = pass linear through (sRGB swapchain auto-encodes).
  applySrgbOetf: f32,
  // 3 pad floats to round the UBO up to 16 B.
  _pad0:         f32,
  _pad1:         f32,
  _pad2:         f32,
};

@group(0) @binding(0) var<uniform> post: Post;
@group(0) @binding(1) var samp:  sampler;
@group(0) @binding(2) var tex:   texture_2d<f32>;

struct Vout {
  @builtin(position) pos: vec4<f32>,
  @location(0)       uv:  vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> Vout {
  // Fullscreen triangle covering NDC [-1,1]²; UV [0,1]² via (x+1)/2 etc.
  let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
  let y = f32(vi & 2u) * 2.0 - 1.0;
  var o: Vout;
  o.pos = vec4<f32>(x, y, 0.0, 1.0);
  o.uv  = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
  return o;
}

// sRGB OETF — same impl as dispersion.wgsl's linearToSrgb. Kept local so
// this module is self-contained.
fn linearToSrgb(c: vec3<f32>) -> vec3<f32> {
  let cutoff = vec3<f32>(0.0031308);
  let low    = c * 12.92;
  let high   = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
  return select(high, low, c <= cutoff);
}

fn encodeOut(c: vec3<f32>) -> vec3<f32> {
  if (post.applySrgbOetf > 0.5) { return linearToSrgb(c); }
  return c;
}

// Perceptual luma. FXAA edge detection thresholds are tuned for
// gamma-encoded (sRGB-space) luma, so we transform linear → sRGB first.
// Blending still happens in linear space (below), which is physically
// correct for refracted light.
fn lumaOf(c: vec3<f32>) -> f32 {
  let g = linearToSrgb(c);
  return dot(g, vec3<f32>(0.299, 0.587, 0.114));  // Rec.601
}

// Passthrough: linear intermediate → swapchain with optional sRGB encode.
// Used for aaMode = 'none' and aaMode = 'taa' (TAA already ran in the
// scene pass — post-process just forwards the result).
@fragment
fn fs_passthrough(v: Vout) -> @location(0) vec4<f32> {
  let c = textureSampleLevel(tex, samp, v.uv, 0.0).rgb;
  return vec4<f32>(encodeOut(c), 1.0);
}

// FXAA 3.x-style spatial AA. 9-tap neighbourhood: NW/NE/SW/SE + center +
// 4 samples along the detected edge direction. Parameters follow Lottes'
// canonical tuning — 0.0625 minimum absolute contrast (i.e. skip flat
// regions), 0.125 relative contrast (skip very dark-but-low-contrast),
// 8-pixel direction clamp for long edges, and a luma-scaled `dirReduce`
// (average of the 4 corner lumas × 0.125, floored at 1/128) that damps
// the step on bright edges where the raw direction vector would dominate
// and guards axis-aligned edges from a divide-by-zero.
//
// Runs entirely in a single pass, no temporal history. Cost ≈ 0.3 ms at
// 1080p — dominated by the bilinear taps, which are fast on every GPU.
@fragment
fn fs_fxaa(v: Vout) -> @location(0) vec4<f32> {
  let rcpFrame = 1.0 / vec2<f32>(textureDimensions(tex, 0));

  let rgbNW = textureSampleLevel(tex, samp, v.uv + vec2<f32>(-1.0, -1.0) * rcpFrame, 0.0).rgb;
  let rgbNE = textureSampleLevel(tex, samp, v.uv + vec2<f32>( 1.0, -1.0) * rcpFrame, 0.0).rgb;
  let rgbSW = textureSampleLevel(tex, samp, v.uv + vec2<f32>(-1.0,  1.0) * rcpFrame, 0.0).rgb;
  let rgbSE = textureSampleLevel(tex, samp, v.uv + vec2<f32>( 1.0,  1.0) * rcpFrame, 0.0).rgb;
  let rgbM  = textureSampleLevel(tex, samp, v.uv, 0.0).rgb;

  let lumaNW = lumaOf(rgbNW);
  let lumaNE = lumaOf(rgbNE);
  let lumaSW = lumaOf(rgbSW);
  let lumaSE = lumaOf(rgbSE);
  let lumaM  = lumaOf(rgbM);

  let lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
  let lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
  let range   = lumaMax - lumaMin;

  // Skip flat regions — no perceivable edge here.
  if (range < max(0.0625, lumaMax * 0.125)) {
    return vec4<f32>(encodeOut(rgbM), 1.0);
  }

  // Edge direction from 2×2 luma gradient. Perpendicular to the edge is
  // the axis along which we step for the blend.
  var dir = vec2<f32>(
    -((lumaNW + lumaNE) - (lumaSW + lumaSE)),
     ((lumaNW + lumaSW) - (lumaNE + lumaSE)),
  );

  // Prevent axis-aligned edges from degenerating into a zero vector.
  let dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.25 * 0.125, 1.0 / 128.0);
  let rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
  dir = clamp(dir * rcpDirMin, vec2<f32>(-8.0), vec2<f32>(8.0)) * rcpFrame;

  // Two pairs of samples along the edge: the first pair averages two taps
  // 1/6 of a pixel from center (fine AA), the second pair extends further
  // out for thicker edges. `rgbB` is the "wider" blend; we fall back to
  // `rgbA` (tighter) if the wide blend's luma falls outside the
  // neighbourhood range — guard against over-blur at corners.
  let rgbA = 0.5 * (
    textureSampleLevel(tex, samp, v.uv + dir * (1.0 / 3.0 - 0.5), 0.0).rgb +
    textureSampleLevel(tex, samp, v.uv + dir * (2.0 / 3.0 - 0.5), 0.0).rgb
  );
  let rgbB = rgbA * 0.5 + 0.25 * (
    textureSampleLevel(tex, samp, v.uv + dir * -0.5, 0.0).rgb +
    textureSampleLevel(tex, samp, v.uv + dir *  0.5, 0.0).rgb
  );

  let lumaB = lumaOf(rgbB);
  let result = select(rgbA, rgbB, lumaB >= lumaMin && lumaB <= lumaMax);
  return vec4<f32>(encodeOut(result), 1.0);
}
