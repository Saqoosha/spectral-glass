import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const root = join(__dirname, '..');
const read = (p: string) => readFileSync(join(root, p), 'utf8');

/**
 * Guardrails so the pill fast path (analytic back exit + per-instance front
 * normal) is not regressed to full-scene SDF marches.
 */
describe('pill analytic path (WGSL source)', () => {
  it('exports pill + prism analytic exits and hit idx helpers in trace', () => {
    const t = read('src/shaders/dispersion/trace.wgsl');
    expect(t).toContain('fn pillAnalyticExit(');
    expect(t).toContain('fn prismAnalyticExit(');
    expect(t).toContain('fn hitPillPillIdx(');
    expect(t).toContain('fn hitPrismPillIdx(');
  });
  it('exports sdfPillGrad for Newton + normals', () => {
    const s = read('src/shaders/dispersion/sdf_primitives.wgsl');
    expect(s).toContain('fn sdfPillGrad(');
  });
  it('backExit and fs_main wire shape 0 to pill fast path', () => {
    const f = read('src/shaders/dispersion/fragment.wgsl');
    expect(f).toContain('if (shapeId == 0) { return pillAnalyticExit');
    expect(f).toContain('sceneNormalPill(');
    expect(f).toContain('hitPillPillIdx(h.p)');
  });
  it('backExit and fs_main wire shape 1 to prism fast path', () => {
    const f = read('src/shaders/dispersion/fragment.wgsl');
    expect(f).toContain('if (shapeId == 1) { return prismAnalyticExit');
    expect(f).toContain('sceneNormalPrism(');
    expect(f).toContain('hitPrismPillIdx(h.p)');
  });
});
