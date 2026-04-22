import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

// Minimal drift detector for the host ↔ WGSL Frame uniform layout.
//
// uniforms.ts and dispersion.wgsl declare the same struct twice. WebGPU won't
// catch a host/shader mismatch — it just reads garbled bytes. Even a strict
// integration test would need a real GPU device. The compromise: pin the
// expected Frame field list here. The host side (uniforms.ts) is designed to
// mirror this list field-for-field; any WGSL change that adds, removes, or
// reorders a field will trip an assertion below and prompt a review of
// `HEAD_FLOATS` / `CUBE_ROT_FLOATS` / `pillBase` in uniforms.ts. The reverse
// (someone changing those host constants without touching WGSL) is NOT caught
// here — that would still produce silent uniform corruption on the GPU.

const here   = dirname(fileURLToPath(import.meta.url));
const wgsl   = readFileSync(resolve(here, '../src/shaders/dispersion.wgsl'), 'utf8');

// Pull the body of the WGSL `struct Frame { ... }` block as a single string.
const FRAME_BODY = (() => {
  const m = /struct\s+Frame\s*\{([\s\S]*?)\}/m.exec(wgsl);
  if (!m) throw new Error('Could not locate `struct Frame {...}` in dispersion.wgsl');
  return m[1] ?? '';
})();

// Extract every `<name>: <type>` declaration from the struct body and ignore
// anything that's clearly a comment line. Order matters; the host writes by
// offset so a swapped pair would silently corrupt every following field.
//
// The type string can itself contain commas (e.g. `array<PillGpu, MAX_PILLS>`)
// so we strip the trailing line-terminating comma first and only then split
// on the first `:`.
function declaredFields(body: string): { name: string; type: string }[] {
  const out: { name: string; type: string }[] = [];
  for (const raw of body.split('\n')) {
    const stripped = raw.replace(/\/\/.*$/, '').trim().replace(/,$/, '');
    if (!stripped) continue;
    const colon = stripped.indexOf(':');
    if (colon < 0) continue;
    const name = stripped.slice(0, colon).trim();
    const type = stripped.slice(colon + 1).trim();
    // Hard-fail rather than silently skip — a malformed line means the parser
    // would otherwise drop a field and let drift sneak through. If somebody
    // splits a declaration across lines (e.g. `name:\n  type,`), the resulting
    // empty halves trip these throws and force the parser to be revisited
    // alongside the WGSL change.
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) {
      throw new Error(`Frame field has malformed name: ${JSON.stringify(raw)}`);
    }
    if (!type) {
      throw new Error(`Frame field has empty type for "${name}": ${JSON.stringify(raw)}`);
    }
    out.push({ name, type });
  }
  return out;
}

describe('uniform layout drift detector', () => {
  it('Frame struct declares the fields uniforms.ts expects, in order', () => {
    const fields = declaredFields(FRAME_BODY);
    // The WGSL struct must end with cubeRot / plateRot / plate-wave scalars
    // and then `pills: array<PillGpu, MAX_PILLS>`. Anything else means
    // uniforms.ts' HEAD_FLOATS / CUBE_ROT_FLOATS / PLATE_ROT_FLOATS /
    // PLATE_PARAMS_FLOATS / pillBase need to move with it.
    const names = fields.map((f) => f.name);
    expect(names).toEqual([
      'resolution', 'photoSize',
      'n_d', 'V_d', 'sampleCount', 'refractionStrength',
      'jitter', 'refractionMode', 'pillCount', 'applySrgbOetf',
      'shape', 'time', 'historyBlend', 'heroLambda',
      'cameraZ', 'projection', 'debugProxy', 'taaEnabled',
      'cubeRot',
      'cubeRotPrev',
      'plateRot',
      'plateRotPrev',
      'diamondRot',
      'diamondRotPrev',
      'waveAmp', 'waveFreq', 'waveLipFactor', 'sceneTime',
      'diamondSize', '_diamondPad0', '_diamondPad1', '_diamondPad2',
      'pills',
    ]);
  });

  it('all six rotation slots are mat3x3<f32> (12 floats / 48 B each)', () => {
    // If the field order changes, the previous test fires first. This pins
    // the types so a swap to e.g. mat4x4 (which is 64 B, not 48) trips a
    // separate failure — otherwise the order test would pass but pills
    // would be corrupted. All prev-frame slots are pinned too so a
    // refactor that changes their representation can't slip through.
    const fields = declaredFields(FRAME_BODY);
    expect(fields.find((f) => f.name === 'cubeRot')?.type).toBe('mat3x3<f32>');
    expect(fields.find((f) => f.name === 'cubeRotPrev')?.type).toBe('mat3x3<f32>');
    expect(fields.find((f) => f.name === 'plateRot')?.type).toBe('mat3x3<f32>');
    expect(fields.find((f) => f.name === 'plateRotPrev')?.type).toBe('mat3x3<f32>');
    expect(fields.find((f) => f.name === 'diamondRot')?.type).toBe('mat3x3<f32>');
    expect(fields.find((f) => f.name === 'diamondRotPrev')?.type).toBe('mat3x3<f32>');
  });

  it('pills is the last field (so HEAD + rotations + plateParams + diamondParams is the right base)', () => {
    const fields = declaredFields(FRAME_BODY);
    expect(fields[fields.length - 1]?.name).toBe('pills');
    expect(fields[fields.length - 1]?.type).toBe('array<PillGpu, MAX_PILLS>');
  });
});
