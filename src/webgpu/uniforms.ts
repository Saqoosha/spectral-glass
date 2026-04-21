import type { Pill } from '../pills';

export const MAX_PILLS = 8;

export type FrameParams = {
  resolution:         [number, number];
  photoSize:          [number, number];
  n_d:                number;
  V_d:                number;
  sampleCount:        number;
  refractionStrength: number;
  jitter:             number;
  refractionMode:     number;
  pillCount:          number;
  pills:              Pill[];
};

// 2 × vec4 head + 2 × vec4 spectral + vec4 meta + MAX_PILLS × 2 × vec4 per pill
//   head:     resolution.xy, photo.xy                           (16B)
//   spectral: n_d, V_d, N, refractionStrength                   (16B)
//             jitter, refractionMode, pillCount, _              (16B)
//   per pill: center.xyz, edgeR                                 (16B)
//             half.xyz, _                                       (16B)
const HEAD_SIZE = 16 + 16 + 16;               // 48
const PILL_SIZE = 32;                         // 32
const TOTAL_SIZE = HEAD_SIZE + PILL_SIZE * MAX_PILLS; // 48 + 256 = 304

export function createFrameBuffer(device: GPUDevice): GPUBuffer {
  return device.createBuffer({
    label: 'frame',
    size:  TOTAL_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

export function writeFrame(device: GPUDevice, buf: GPUBuffer, p: FrameParams): void {
  const d = new Float32Array(TOTAL_SIZE / 4);
  d[0]  = p.resolution[0];
  d[1]  = p.resolution[1];
  d[2]  = p.photoSize[0];
  d[3]  = p.photoSize[1];

  d[4]  = p.n_d;
  d[5]  = p.V_d;
  d[6]  = p.sampleCount;
  d[7]  = p.refractionStrength;

  d[8]  = p.jitter;
  d[9]  = p.refractionMode;
  d[10] = p.pillCount;
  d[11] = 0;

  for (let i = 0; i < MAX_PILLS; i++) {
    const base = 12 + i * 8;
    const pill = i < p.pills.length ? p.pills[i] : null;
    if (pill) {
      d[base + 0] = pill.cx;
      d[base + 1] = pill.cy;
      d[base + 2] = pill.cz;
      d[base + 3] = pill.edgeR;
      d[base + 4] = pill.hx;
      d[base + 5] = pill.hy;
      d[base + 6] = pill.hz;
      d[base + 7] = 0;
    }
  }
  device.queue.writeBuffer(buf, 0, d);
}
