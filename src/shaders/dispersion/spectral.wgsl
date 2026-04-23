// ---------- spectral math ----------

// Cauchy + Abbe number (glTF KHR_materials_dispersion formulation).
// Clamped to 1.0 because ior<1 breaks Snell direction via refract().
fn cauchyIor(lambda: f32, n_d: f32, V_d: f32) -> f32 {
  return max(n_d + (n_d - 1.0) / V_d * (523655.0 / (lambda * lambda) - 1.5168), 1.0);
}

fn gLobe(lambda: f32, mu: f32, s1: f32, s2: f32) -> f32 {
  let sigma = select(s2, s1, lambda < mu);
  let t = (lambda - mu) / sigma;
  return exp(-0.5 * t * t);
}

// Wyman-Sloan-Shirley (JCGT 2013) analytic CIE 1931 2° XYZ matching functions.
fn cieXyz(lambda: f32) -> vec3<f32> {
  let x =  0.362 * gLobe(lambda, 442.0, 16.0, 26.7)
        +  1.056 * gLobe(lambda, 599.8, 37.9, 31.0)
        -  0.065 * gLobe(lambda, 501.1, 20.4, 26.2);
  let y =  0.821 * gLobe(lambda, 568.8, 46.9, 40.5)
        +  0.286 * gLobe(lambda, 530.9, 16.3, 31.1);
  let z =  1.217 * gLobe(lambda, 437.0, 11.8, 36.0)
        +  0.681 * gLobe(lambda, 459.0, 26.0, 13.8);
  return vec3<f32>(x, y, z);
}

// D65 XYZ → linear sRGB. WGSL mat3x3 is column-major, so each row of literals
// below is the column of the textbook matrix.
fn xyzToSrgb(c: vec3<f32>) -> vec3<f32> {
  let m = mat3x3<f32>(
     3.2404542, -0.9692660,  0.0556434,
    -1.5371385,  1.8760108, -0.2040259,
    -0.4985314,  0.0415560,  1.0572252,
  );
  return m * c;
}

fn schlickFresnel(cosT: f32, n_d: f32) -> f32 {
  let f0 = pow((n_d - 1.0) / (n_d + 1.0), 2.0);
  let k  = 1.0 - clamp(cosT, 0.0, 1.0);
  return f0 + (1.0 - f0) * k * k * k * k * k;
}

// Hash a 2D input to [0,1). Dave Hoskins' hash12, small-footprint + low bias
// enough for variance reduction (we're not doing security here).
fn hash21(p: vec2<f32>) -> f32 {
  var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
  p3 = p3 + vec3<f32>(dot(p3, p3.yzx + vec3<f32>(33.33)));
  return fract((p3.x + p3.y) * p3.z);
}
