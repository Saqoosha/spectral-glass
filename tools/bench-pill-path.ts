/**
 * Static cost model for pill (shapeId 0) vs the old generic path.
 * Run: `bun run tools/bench-pill-path.ts`
 *
 * GPU ms must be measured in-browser with `?perf=1` (see README) — this
 * script only quantifies SDF eval counts from the shader sources.
 */
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, '..');

const trace = readFileSync(join(root, 'src/shaders/dispersion/trace.wgsl'), 'utf8');
const scene = readFileSync(join(root, 'src/shaders/dispersion/scene.wgsl'), 'utf8');
const sdf = readFileSync(join(root, 'src/shaders/dispersion/sdf_primitives.wgsl'), 'utf8');
const frag = readFileSync(join(root, 'src/shaders/dispersion/fragment.wgsl'), 'utf8');

const P = 4; // default pill count (demo)
const N = 8; // typical spectral sample count
const SDF_PER_SCENE = P; // sceneSdf min over instances
const INSIDE_TRACE_ITERS = 48;

// Old back path per wavelength: insideTrace (~48) * sceneSdf + sceneNormal (6) * sceneSdf
const oldBackSdf = INSIDE_TRACE_ITERS * SDF_PER_SCENE + 6 * SDF_PER_SCENE;
// New: pillAnalyticExit — slab (0 SDF) + up to 4 * (1 sdfPill + 1 grad) + 1 grad for normal
//      grad = 6 sdfPill; Newton loop can call sdfPill+grad: ~4 * (1+6) + 6 for final
const newPillAnalyticSdf = 4 * 7 + 6; // upper bound ≈ 34 single-pill evals
const newFrontPill = 6; // sceneNormalPill: one sdfPillGrad = 6 sdfPill
const oldFront = 6 * SDF_PER_SCENE; // sceneNormal: 6 * sceneSdf

console.log('Spectral Glass — pill path static estimate (4 instances, Exact mode per-λ back trace)\n');
console.log('Assumptions: sphere-trace + front normal unchanged; compare back-exit + front normal SDF load.\n');
console.log('┌─ Path ─────────────────────────────┬─ Approx sdfPill-weighted evals (back) ─┐');
console.log(`│ old: insideTrace+sceneNormal at exit │ ${oldBackSdf} (× N wavelengths in Exact)     │`);
console.log(`│ new: pillAnalyticExit                 │ ${newPillAnalyticSdf} (× N)                    │`);
console.log('└──────────────────────────────────────┴────────────────────────────────────────┘\n');
console.log(`Front hit normal: old ≈ ${oldFront} sceneSdf inner evals, new = ${newFrontPill} sdfPill (single instance)\n`);
console.log(`back-exit old / new ≈ ${(oldBackSdf * N) / (newPillAnalyticSdf * N) | 0}:1  (per fragment, per-λ; ignores hero shortcut)\n`);

if (!/pillAnalyticExit/.test(trace)) {
  throw new Error('expected pillAnalyticExit in trace.wgsl');
}
if (!/fn sdfPillGrad/.test(sdf)) {
  throw new Error('expected sdfPillGrad in sdf_primitives.wgsl');
}
if (!/sceneNormalPill/.test(scene)) {
  throw new Error('expected sceneNormalPill in scene.wgsl');
}
if (!frag.includes('if (shapeId == 0) { return pillAnalyticExit')) {
  throw new Error('expected pill branch in backExit (fragment.wgsl)');
}

console.log('Shader source checks: OK (pill fast path present).\n');
console.log('To measure real GPU time: run `bun run dev`, add ?perf=1, pick Shape → Pill, read Perf → GPU ms.\n');
