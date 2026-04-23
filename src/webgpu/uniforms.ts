import type { Pill } from '../pills';
import { cubeRotationColumns } from '../math/cube';
import { plateRotationColumns } from '../math/plate';
import { diamondRotationColumns, diamondViewRotationColumns, type DiamondView } from '../math/diamond';

export const MAX_PILLS = 8;

export type FrameParams = {
  readonly resolution:         readonly [number, number];
  readonly photoSize:          readonly [number, number];
  readonly n_d:                number;
  readonly V_d:                number;
  readonly sampleCount:        number;
  readonly refractionStrength: number;
  readonly jitter:             number;
  readonly refractionMode:     number;
  readonly applySrgbOetf:      boolean;
  readonly shape:              number;  // 0 = pill, 1 = prism, 2 = cube, 3 = plate, 4 = diamond
  readonly time:               number;  // seconds since start. Host derives the cube/plate/diamond rotations from this; the GPU also sees it raw for the jitter hash seed.
  readonly historyBlend:       number;  // 0..1 — 0.2 steady-state, 1.0 on scene-change frames to clear stale history
  readonly heroLambda:         number;  // [380, 700] — jittered each frame; drives hero-wavelength back-face trace
  readonly cameraZ:            number;  // distance from screen plane (z=0) to the camera, in pixels
  readonly projection:         number;  // 0 = orthographic, 1 = perspective
  readonly debugProxy:         boolean; // tint proxy fragments pink for debugging
  readonly taaEnabled:         boolean; // jitter ray origin per frame so history EMA antialiases shader-decided silhouettes
  readonly sceneTime:          number;  // time value driving rotation + wave phase. Freezes on "Stop the world" while `time` above keeps advancing so AA continues to converge.
  readonly prevSceneTime:      number;  // sceneTime of the previous frame — drives motion-vector reprojection so history follows rotating shapes
  readonly waveAmp:            number;  // plate: displacement amplitude (px). Ignored for other shapes.
  readonly waveFreq:           number;  // plate: angular frequency (rad/px). Wavelength ≈ 2π/waveFreq.
  readonly diamondSize:        number;  // diamond: girdle diameter (px). Ignored for other shapes.
  readonly diamondWireframe:   boolean; // diamond: overlay facet edges on top of the rendering for debugging.
  readonly diamondFacetColor:  boolean; // diamond: flat-shade each facet class with a distinct debug colour.
  readonly diamondTirDebug:    boolean; // diamond: diagnostic tints for unresolved TIR path — hot pink = bounce budget exhausted with refract still TIR; orange = diamondAnalyticExit miss (see dispersion.wgsl).
  /** Diamond TIR path: max internal bounces (1..32, GPU-clamped). */
  readonly diamondTirMaxBounces: number;
  readonly diamondView:        DiamondView; // diamond: 'free' tumbles, fixed views pin for reference-checking.
  readonly envmapEnabled:      boolean; // Phase C: use HDR panorama for reflection / TIR (false = Phase A reflSrc hack).
  readonly envmapExposure:     number;  // linear-light multiplier on envmap samples. Typical 0.05–1.0 depending on HDRI peak.
  readonly envmapRotation:     number;  // radians around world-Y; rotates the sky panorama.
  readonly pills:              readonly Pill[];
};

// Byte layout mirrors the WGSL `Frame` struct exactly (std140-ish rules):
//   offset   0: resolution.xy,  photoSize.xy                      (16 B)
//   offset  16: n_d, V_d, sampleCount, refractionStrength         (16 B)
//   offset  32: jitter, refractionMode, pillCount, applySrgbOetf  (16 B)
//   offset  48: shape, time, historyBlend, heroLambda             (16 B)
//   offset  64: cameraZ, projection, debugProxy, taaEnabled       (16 B)
//   offset  80: cubeRot        mat3x3<f32>                        (48 B)
//   offset 128: cubeRotPrev    mat3x3<f32>                        (48 B)
//   offset 176: plateRot       mat3x3<f32>                        (48 B)
//   offset 224: plateRotPrev   mat3x3<f32>                        (48 B)
//   offset 272: diamondRot     mat3x3<f32>                        (48 B)
//   offset 320: diamondRotPrev mat3x3<f32>                        (48 B)
//   offset 368: waveAmp, waveFreq, waveLipFactor, sceneTime       (16 B)
//   offset 384: diamondSize + 3 diamond bool-flags + tirMaxBounces + 3 pad  (32 B)
//   offset 416: envmapExposure, envmapRotation, envmapEnabled, pad (16 B)
//   offset 432: pills[0..MAX_PILLS] — each pill is vec3 + f32 + vec3 + f32 (32 B)
const HEAD_FLOATS              = 20;                                // 80 B
const CUBE_ROT_FLOATS          = 12;                                // 48 B (3 padded cols)
const CUBE_ROT_PREV_FLOATS     = 12;                                // 48 B (prev-frame cube for reprojection)
const PLATE_ROT_FLOATS         = 12;                                // 48 B
const PLATE_ROT_PREV_FLOATS    = 12;                                // 48 B (prev-frame plate for reprojection)
const DIAMOND_ROT_FLOATS       = 12;                                // 48 B
const DIAMOND_ROT_PREV_FLOATS  = 12;                                // 48 B (prev-frame diamond for reprojection)
const PLATE_PARAMS_FLOATS      = 4;                                 // 16 B (3 used + 1 pad)
const DIAMOND_PARAMS_FLOATS    = 8;                                 // 32 B (size + 3 bool-flags + tir max + 3× pad)
const ENVMAP_PARAMS_FLOATS     = 4;                                 // 16 B (exposure + rotation + enabled + pad)
const PILL_FLOATS              = 8;                                 // 32 B per pill
const TOTAL_FLOATS    = HEAD_FLOATS
                       + CUBE_ROT_FLOATS    + CUBE_ROT_PREV_FLOATS
                       + PLATE_ROT_FLOATS   + PLATE_ROT_PREV_FLOATS
                       + DIAMOND_ROT_FLOATS + DIAMOND_ROT_PREV_FLOATS
                       + PLATE_PARAMS_FLOATS + DIAMOND_PARAMS_FLOATS + ENVMAP_PARAMS_FLOATS
                       + PILL_FLOATS * MAX_PILLS;
const TOTAL_BYTES     = TOTAL_FLOATS * 4;

// Reused across frames so we don't allocate a fresh Float32Array every tick.
// Total size grows with the field set — see TOTAL_BYTES above (172 floats → 688 B).
const scratch = new Float32Array(TOTAL_FLOATS);

export function createFrameBuffer(device: GPUDevice): GPUBuffer {
  return device.createBuffer({
    label: 'frame',
    size:  TOTAL_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

export function writeFrame(device: GPUDevice, buf: GPUBuffer, p: FrameParams): void {
  scratch.fill(0);
  scratch[0] = p.resolution[0];
  scratch[1] = p.resolution[1];
  scratch[2] = p.photoSize[0];
  scratch[3] = p.photoSize[1];

  scratch[4] = p.n_d;
  scratch[5] = p.V_d;
  scratch[6] = p.sampleCount;
  scratch[7] = p.refractionStrength;

  scratch[8]  = p.jitter;
  scratch[9]  = p.refractionMode;
  scratch[10] = Math.min(p.pills.length, MAX_PILLS);
  scratch[11] = p.applySrgbOetf ? 1 : 0;

  scratch[12] = p.shape;
  scratch[13] = p.time;
  scratch[14] = p.historyBlend;
  scratch[15] = p.heroLambda;

  scratch[16] = p.cameraZ;
  scratch[17] = p.projection;
  scratch[18] = p.debugProxy ? 1 : 0;
  scratch[19] = p.taaEnabled ? 1 : 0;

  // Cube + plate + diamond rotations are driven by `sceneTime` (frozen when
  // paused, see main.ts) so "Stop the world" stops spin even though the noise
  // time (`p.time` above) keeps advancing for TAA convergence. Prev-frame
  // matrices feed the TAA reprojection path — see reprojectHit in
  // src/shaders/dispersion.wgsl.
  let base = HEAD_FLOATS;
  scratch.set(cubeRotationColumns(p.sceneTime),        base); base += CUBE_ROT_FLOATS;
  scratch.set(cubeRotationColumns(p.prevSceneTime),    base); base += CUBE_ROT_PREV_FLOATS;
  scratch.set(plateRotationColumns(p.sceneTime),       base); base += PLATE_ROT_FLOATS;
  scratch.set(plateRotationColumns(p.prevSceneTime),   base); base += PLATE_ROT_PREV_FLOATS;
  // Diamond rotation: 'free' uses the tumble animation driven by sceneTime;
  // the fixed views pin to a canonical pose and write the same matrix into
  // current AND prev so the TAA reprojection sees zero motion vector and
  // doesn't smear the frozen shape.
  if (p.diamondView === 'free') {
    scratch.set(diamondRotationColumns(p.sceneTime),     base); base += DIAMOND_ROT_FLOATS;
    scratch.set(diamondRotationColumns(p.prevSceneTime), base); base += DIAMOND_ROT_PREV_FLOATS;
  } else {
    const fixed = diamondViewRotationColumns(p.diamondView);
    scratch.set(fixed, base); base += DIAMOND_ROT_FLOATS;
    scratch.set(fixed, base); base += DIAMOND_ROT_PREV_FLOATS;
  }

  // Plate wave parameters: amp (px), freq (rad/px), and the precomputed
  // Lipschitz safety factor `1/sqrt(1 + (amp·freq)²)` that sdfWavyPlate
  // multiplies into its output. Hoisting the factor here saves an inverseSqrt
  // on every SDF evaluation (up to ~70 per fragment on the plate path) since
  // both amp and freq are uniform across the frame.
  //
  // Derivation: the shifted SDF reads `box(p with z ← p.z − waveZ)`, whose
  // gradient magnitude is sqrt(1 + (∂waveZ/∂x)² + (∂waveZ/∂y)²). With
  // waveZ = amp·sin(kx+φ)·sin(ky+φ), the partials are amp·k·cos(kx+φ)·sin(ky+φ)
  // and amp·k·sin(kx+φ)·cos(ky+φ). Substituting a = cos²(kx+φ), b = cos²(ky+φ)
  // and maximizing a + b − 2ab over the unit square pins the maximum at the
  // corners (a, b) = (1, 0) or (0, 1) where it evaluates to 1, NOT 2 — the
  // partials never reach amp·k simultaneously. So the tight bound on
  // |∇waveZ|² is (amp·k)², not 2·(amp·k)².
  //
  // Trailing slot is `sceneTime` (same as what drives the rotations — wave
  // phase freezes together with the tumble when paused).
  const plateParamsBase = base;
  const ampFreq         = p.waveAmp * p.waveFreq;
  scratch[plateParamsBase + 0] = p.waveAmp;
  scratch[plateParamsBase + 1] = p.waveFreq;
  scratch[plateParamsBase + 2] = 1 / Math.sqrt(1 + ampFreq * ampFreq);
  scratch[plateParamsBase + 3] = p.sceneTime;

  // Diamond params: girdle diameter + 3 debug flags + TIR max bounces; three
  // padding floats zero-filled so the sub-block is 32 B and the envmap group
  // stays 16 B aligned.
  const diamondParamsBase = plateParamsBase + PLATE_PARAMS_FLOATS;
  scratch[diamondParamsBase + 0] = p.diamondSize;
  scratch[diamondParamsBase + 1] = p.diamondWireframe  ? 1 : 0;
  scratch[diamondParamsBase + 2] = p.diamondFacetColor ? 1 : 0;
  scratch[diamondParamsBase + 3] = p.diamondTirDebug   ? 1 : 0;
  scratch[diamondParamsBase + 4] = Math.min(
    32,
    Math.max(1, Math.round(p.diamondTirMaxBounces)),
  );
  // +5, +6, +7 remain zero (scratch.fill(0))

  // Envmap params. `envmapEnabled` is a boolean-as-f32 so the shader's
  // `frame.envmapEnabled > 0.5` guard matches the other diamond debug
  // flags' convention.
  const envmapParamsBase = diamondParamsBase + DIAMOND_PARAMS_FLOATS;
  scratch[envmapParamsBase + 0] = p.envmapExposure;
  scratch[envmapParamsBase + 1] = p.envmapRotation;
  scratch[envmapParamsBase + 2] = p.envmapEnabled ? 1 : 0;
  // slot 3 is the alignment pad left at scratch.fill(0)'s zero — reserved
  // for a future envmap control (IBL mip bias, tint, etc.) without
  // another layout bump.

  const pillBase  = envmapParamsBase + ENVMAP_PARAMS_FLOATS;
  const pillCount = Math.min(p.pills.length, MAX_PILLS);
  for (let i = 0; i < pillCount; i++) {
    const pill = p.pills[i]!;
    const base = pillBase + i * PILL_FLOATS;
    scratch[base + 0] = pill.cx;
    scratch[base + 1] = pill.cy;
    scratch[base + 2] = pill.cz;
    scratch[base + 3] = pill.edgeR;
    scratch[base + 4] = pill.hx;
    scratch[base + 5] = pill.hy;
    scratch[base + 6] = pill.hz;
  }
  device.queue.writeBuffer(buf, 0, scratch);
}
