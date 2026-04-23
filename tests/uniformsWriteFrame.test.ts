import { describe, it, expect } from 'vitest';
import { writeFrame, type FrameParams } from '../src/webgpu/uniforms';
import {
  DIAMOND_VIEW_VALUES,
  diamondRotationColumns,
  diamondViewRotationColumns,
} from '../src/math/diamond';

// Byte-layout offsets (in floats) mirrored from src/webgpu/uniforms.ts.
// If the uniforms writer changes its internal layout, these need to move
// with it — the uniform-layout drift detector pins the WGSL struct order,
// and this file pins the host's write positions for the diamond rotation
// slots. Keeps both sides honest.
const DIAMOND_ROT_FLOAT_OFFSET      = 20 + 12 + 12 + 12 + 12;   // HEAD + cubeRot + cubeRotPrev + plateRot + plateRotPrev
const DIAMOND_ROT_PREV_FLOAT_OFFSET = DIAMOND_ROT_FLOAT_OFFSET + 12;
const DIAMOND_ROT_FLOATS            = 12;
// Diamond params block: 4 floats (diamondSize + wireframe + facetColor +
// tirDebug), starting right after the plate params block (which itself
// follows both diamond rotation blocks).
const DIAMOND_PARAMS_FLOAT_OFFSET    = DIAMOND_ROT_PREV_FLOAT_OFFSET + 12 + 4; // + plateParams(4)
// Envmap params block follows immediately after diamond params: 4 floats
// (exposure, rotation, enabled, pad). Kept on a pixel of its own in tests
// so the Phase C slot packing doesn't silently shift under a future
// uniform reshuffle.
const ENVMAP_PARAMS_FLOAT_OFFSET     = DIAMOND_PARAMS_FLOAT_OFFSET + 4;

// Minimal device + buffer mocks. writeFrame calls queue.writeBuffer exactly
// once per invocation; we capture the source Float32Array and the offset
// into that mock instead of allocating a real GPUBuffer.
type WriteRecord = { offset: GPUSize64; src: Float32Array };

function mockDevice(): { device: GPUDevice; writes: WriteRecord[] } {
  const writes: WriteRecord[] = [];
  const queue = {
    writeBuffer: (_buf: GPUBuffer, offset: GPUSize64, src: BufferSource | SharedArrayBuffer) => {
      // `src` will be the scratch Float32Array writeFrame owns. Copy so a
      // subsequent writeFrame call (on the same scratch buffer) doesn't
      // mutate earlier captures.
      const copy = new Float32Array((src as Float32Array));
      writes.push({ offset, src: copy });
    },
  } as unknown as GPUQueue;
  // `createBuffer` is only called by createFrameBuffer — we supply a real
  // mock so setup matches production, even if we never dereference it.
  const createBuffer = (_desc: GPUBufferDescriptor) => ({} as GPUBuffer);
  const device = { queue, createBuffer } as unknown as GPUDevice;
  return { device, writes };
}

function baseParams(overrides: Partial<FrameParams>): FrameParams {
  return {
    resolution:         [1280, 720],
    photoSize:          [1024, 768],
    n_d:                1.517,
    V_d:                40,
    sampleCount:        8,
    refractionStrength: 0.2,
    jitter:             0,
    refractionMode:     0,
    applySrgbOetf:      false,
    shape:              4,          // diamond
    time:               1.5,
    historyBlend:       0.2,
    heroLambda:         500,
    cameraZ:            400,
    projection:         1,
    debugProxy:         false,
    taaEnabled:         true,
    sceneTime:          1.5,
    prevSceneTime:      1.4,        // deliberately != sceneTime so a smear would be visible
    waveAmp:            0,
    waveFreq:           0,
    diamondSize:        200,
    diamondWireframe:   false,
    diamondFacetColor:  false,
    diamondTirDebug:    false,
    diamondView:        'free',
    envmapEnabled:      false,
    envmapExposure:     0.25,
    envmapRotation:     0,
    pills:              [],
    ...overrides,
  };
}

function slice12(buf: Float32Array, offset: number): Float32Array {
  return buf.slice(offset, offset + DIAMOND_ROT_FLOATS);
}

function closeToEvery(actual: Float32Array, expected: Float32Array, precision = 6): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i]!, precision);
  }
}

describe('writeFrame — diamond rotation slot selection', () => {
  it("diamondView='free' writes the tumble matrix for current+prev using sceneTime and prevSceneTime", () => {
    const { device, writes } = mockDevice();
    // writeFrame only uses buf as an opaque handle passed to queue.writeBuffer.
    // A stub is sufficient; we don't need the real createFrameBuffer path
    // (which requires GPUBufferUsage from the WebGPU browser globals, absent
    // in Node test env).
    const buf = {} as GPUBuffer;
    const params = baseParams({ diamondView: 'free', sceneTime: 1.5, prevSceneTime: 1.4 });
    writeFrame(device, buf, params);

    expect(writes.length).toBe(1);
    const scratch = writes[0]!.src;

    // current should match diamondRotationColumns(sceneTime)
    closeToEvery(
      slice12(scratch, DIAMOND_ROT_FLOAT_OFFSET),
      diamondRotationColumns(params.sceneTime),
    );
    // prev should match diamondRotationColumns(prevSceneTime) — and differ
    // from current because sceneTime != prevSceneTime.
    closeToEvery(
      slice12(scratch, DIAMOND_ROT_PREV_FLOAT_OFFSET),
      diamondRotationColumns(params.prevSceneTime),
    );
    // Sanity: the two slots are distinguishable floats. If someone
    // "deduplicates" the branch into a single write, this fails.
    const curr = slice12(scratch, DIAMOND_ROT_FLOAT_OFFSET);
    const prev = slice12(scratch, DIAMOND_ROT_PREV_FLOAT_OFFSET);
    let anyDiff = false;
    for (let i = 0; i < curr.length; i++) {
      if (Math.abs(curr[i]! - prev[i]!) > 1e-9) { anyDiff = true; break; }
    }
    expect(anyDiff).toBe(true);
  });

  for (const view of ['top', 'side', 'bottom'] as const) {
    it(`diamondView='${view}' writes the SAME fixed-view matrix into current AND prev (zero rotation motion vector for TAA)`, () => {
      const { device, writes } = mockDevice();
      // writeFrame only uses buf as an opaque handle passed to queue.writeBuffer.
    // A stub is sufficient; we don't need the real createFrameBuffer path
    // (which requires GPUBufferUsage from the WebGPU browser globals, absent
    // in Node test env).
    const buf = {} as GPUBuffer;
      const params = baseParams({ diamondView: view, sceneTime: 1.5, prevSceneTime: 1.4 });
      writeFrame(device, buf, params);

      const scratch = writes[0]!.src;
      const expected = diamondViewRotationColumns(view);

      // current slot matches the preset matrix
      closeToEvery(slice12(scratch, DIAMOND_ROT_FLOAT_OFFSET), expected);
      // prev slot matches the SAME preset matrix — this is the load-bearing
      // invariant. If someone regresses and writes
      // diamondRotationColumns(prevSceneTime) into prev, the motion vector
      // for the first frame after switching to this preset is spurious and
      // TAA smears the shape. This test catches exactly that.
      closeToEvery(slice12(scratch, DIAMOND_ROT_PREV_FLOAT_OFFSET), expected);
    });
  }

  it('writes diamond debug-boolean flags into the correct slots of the params block', () => {
    // The diamond params block is 4 floats:
    //   [diamondSize, diamondWireframe, diamondFacetColor, diamondTirDebug]
    // Toggling each boolean must flip exactly ONE slot (not leak into a
    // neighbour). Catches the "slot 2 vs slot 3 swap" regression that
    // would silently re-route the TIR-debug toggle onto the facet-color
    // path. Table of (param key, expected slot offset relative to
    // diamondParamsBase).
    type BoolKey = 'diamondWireframe' | 'diamondFacetColor' | 'diamondTirDebug';
    const slots: [BoolKey, number][] = [
      ['diamondWireframe',  1],
      ['diamondFacetColor', 2],
      ['diamondTirDebug',   3],
    ];
    for (const [key, slotOffset] of slots) {
      for (const val of [false, true] as const) {
        const { device, writes } = mockDevice();
        const buf = {} as GPUBuffer;
        writeFrame(device, buf, baseParams({ [key]: val } as Partial<FrameParams>));
        const scratch = writes[0]!.src;
        // The flagged slot should hold the boolean-as-float.
        expect(scratch[DIAMOND_PARAMS_FLOAT_OFFSET + slotOffset]).toBe(val ? 1 : 0);
      }
    }
    // Cross-check: with all three booleans false, slots 1-3 are all 0
    // (slot 0 is diamondSize). Pins the layout so future additions don't
    // silently stomp the existing values.
    const { device, writes } = mockDevice();
    const buf = {} as GPUBuffer;
    writeFrame(device, buf, baseParams({
      diamondWireframe:  false,
      diamondFacetColor: false,
      diamondTirDebug:   false,
    }));
    const scratch = writes[0]!.src;
    expect(scratch[DIAMOND_PARAMS_FLOAT_OFFSET + 1]).toBe(0);
    expect(scratch[DIAMOND_PARAMS_FLOAT_OFFSET + 2]).toBe(0);
    expect(scratch[DIAMOND_PARAMS_FLOAT_OFFSET + 3]).toBe(0);
  });

  it('writes envmap params into the correct slots (exposure/rotation/enabled)', () => {
    // Envmap params block is [exposure, rotation, enabled, pad]. Pin
    // each slot so a future 16B-block reshuffle doesn't cross-wire the
    // exposure and rotation values (which would flip panorama brightness
    // and aim in perfect opposite directions).
    const { device, writes } = mockDevice();
    const buf = {} as GPUBuffer;
    writeFrame(device, buf, baseParams({
      envmapEnabled: true,
      envmapExposure: 0.37,
      envmapRotation: 1.23,
    }));
    const scratch = writes[0]!.src;
    expect(scratch[ENVMAP_PARAMS_FLOAT_OFFSET + 0]).toBeCloseTo(0.37, 6);
    expect(scratch[ENVMAP_PARAMS_FLOAT_OFFSET + 1]).toBeCloseTo(1.23, 6);
    expect(scratch[ENVMAP_PARAMS_FLOAT_OFFSET + 2]).toBe(1);
    // Pad slot stays zero (scratch.fill(0)); if a future field lands
    // there this test catches the collision.
    expect(scratch[ENVMAP_PARAMS_FLOAT_OFFSET + 3]).toBe(0);

    // envmapEnabled = false maps to 0 — the shader's `> 0.5` gate
    // needs exact 0/1, so regression-pin the boolean conversion.
    const off = mockDevice();
    writeFrame(off.device, buf, baseParams({ envmapEnabled: false }));
    expect(off.writes[0]!.src[ENVMAP_PARAMS_FLOAT_OFFSET + 2]).toBe(0);
  });

  it('writeFrame accepts every view in DIAMOND_VIEW_VALUES without throwing', () => {
    // The DiamondView union is derived from DIAMOND_VIEW_VALUES (see
    // src/math/diamond.ts), so iterating that tuple here gives automatic
    // sync with the union — adding a new preset makes this loop pick it up
    // without a test edit. Catches the "forgot to add the matrix branch"
    // regression via diamondViewRotationColumns' fail-fast throw.
    for (const view of DIAMOND_VIEW_VALUES) {
      const { device, writes } = mockDevice();
      const buf = {} as GPUBuffer;
      expect(() =>
        writeFrame(device, buf, baseParams({ diamondView: view })),
      ).not.toThrow();
      expect(writes.length).toBe(1);
    }
  });
});
