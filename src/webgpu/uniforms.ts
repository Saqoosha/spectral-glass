import type { Pill } from '../pills';
import { cubeRotationColumns } from '../math/cube';
import { plateRotationColumns } from '../math/plate';

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
  readonly shape:              number;  // 0 = pill, 1 = prism, 2 = cube, 3 = plate
  readonly time:               number;  // seconds since start. Host derives the cube/plate rotations from this; the GPU also sees it raw for the jitter hash seed.
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
  readonly pills:              readonly Pill[];
};

// Byte layout mirrors the WGSL `Frame` struct exactly (std140-ish rules):
//   offset   0: resolution.xy,  photoSize.xy                      (16 B)
//   offset  16: n_d, V_d, sampleCount, refractionStrength         (16 B)
//   offset  32: jitter, refractionMode, pillCount, applySrgbOetf  (16 B)
//   offset  48: shape, time, historyBlend, heroLambda             (16 B)
//   offset  64: cameraZ, projection, debugProxy, taaEnabled       (16 B)
//   offset  80: cubeRot      mat3x3<f32>                          (48 B)
//   offset 128: cubeRotPrev  mat3x3<f32>                          (48 B)
//   offset 176: plateRot     mat3x3<f32>                          (48 B)
//   offset 224: plateRotPrev mat3x3<f32>                          (48 B)
//   offset 272: waveAmp, waveFreq, waveLipFactor, _padWave1       (16 B)
//   offset 288: pills[0..MAX_PILLS] — each pill is vec3 + f32 + vec3 + f32 (32 B)
const HEAD_FLOATS           = 20;                                   // 80 B
const CUBE_ROT_FLOATS       = 12;                                   // 48 B (3 padded cols)
const CUBE_ROT_PREV_FLOATS  = 12;                                   // 48 B (prev-frame cube for reprojection)
const PLATE_ROT_FLOATS      = 12;                                   // 48 B
const PLATE_ROT_PREV_FLOATS = 12;                                   // 48 B (prev-frame plate for reprojection)
const PLATE_PARAMS_FLOATS   = 4;                                    // 16 B (3 used + 1 pad)
const PILL_FLOATS           = 8;                                    // 32 B per pill
const TOTAL_FLOATS    = HEAD_FLOATS
                       + CUBE_ROT_FLOATS  + CUBE_ROT_PREV_FLOATS
                       + PLATE_ROT_FLOATS + PLATE_ROT_PREV_FLOATS
                       + PLATE_PARAMS_FLOATS + PILL_FLOATS * MAX_PILLS;
const TOTAL_BYTES     = TOTAL_FLOATS * 4;

// Reused across frames so we don't allocate a 384-byte Float32Array every tick.
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

  // Cube + plate rotations are driven by `sceneTime` (frozen when paused,
  // see main.ts) so "Stop the world" stops spin even though the noise time
  // (`p.time` above) keeps advancing for TAA convergence. Prev-frame
  // matrices feed the TAA reprojection path — see reprojectHit in
  // src/shaders/dispersion.wgsl.
  let base = HEAD_FLOATS;
  scratch.set(cubeRotationColumns(p.sceneTime),      base); base += CUBE_ROT_FLOATS;
  scratch.set(cubeRotationColumns(p.prevSceneTime),  base); base += CUBE_ROT_PREV_FLOATS;
  scratch.set(plateRotationColumns(p.sceneTime),     base); base += PLATE_ROT_FLOATS;
  scratch.set(plateRotationColumns(p.prevSceneTime), base); base += PLATE_ROT_PREV_FLOATS;

  // Plate wave parameters: amp (px), freq (rad/px), and the precomputed
  // Lipschitz safety factor `1/sqrt(1 + 2·(amp·freq)²)` that sdfWavyPlate
  // multiplies into its output. Hoisting the factor here saves an inverseSqrt
  // on every SDF evaluation (up to ~70 per fragment on the plate path) since
  // both amp and freq are uniform across the frame. Trailing slot is
  // `sceneTime` (same as what drives the rotations — wave phase freezes
  // together with the tumble when paused).
  const plateParamsBase = base;
  const ampFreq         = p.waveAmp * p.waveFreq;
  scratch[plateParamsBase + 0] = p.waveAmp;
  scratch[plateParamsBase + 1] = p.waveFreq;
  scratch[plateParamsBase + 2] = 1 / Math.sqrt(1 + 2 * ampFreq * ampFreq);
  scratch[plateParamsBase + 3] = p.sceneTime;

  const pillBase  = plateParamsBase + PLATE_PARAMS_FLOATS;
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
