export type Pill = {
  cx: number; cy: number; cz: number;
  hx: number; hy: number; hz: number;
  edgeR: number;
};

export function defaultPills(width: number, height: number): Pill[] {
  // Hand-tuned asymmetric layout for the four-cube opening scene (see
  // `defaultParams` in ui.ts). Expressed as fractions of the viewport so the
  // composition scales from a phone-width window up to a 4K monitor.
  // `hx/hy/hz`/`edgeR` here are just placeholders — the render loop overwrites
  // them from `params.pillLen/Short/Thick` every frame.
  const layout: ReadonlyArray<readonly [number, number]> = [
    [0.22, 0.31],
    [0.49, 0.43],
    [0.31, 0.59],
    [0.67, 0.77],
  ];
  return layout.map(([fx, fy]) => ({
    cx: width  * fx,
    cy: height * fy,
    cz: 0,
    hx: 150, hy: 150, hz: 150,
    edgeR: 30,
  }));
}

type DragState =
  | { kind: 'idle' }
  | { kind: 'dragging'; pillIndex: number; offsetX: number; offsetY: number; pointerId: number };

/**
 * Shape IDs mirror `SHAPE_ID` in main.ts / WGSL: 0 pill, 1 prism, 2 cube.
 * `getShapeId` lets the drag layer tune hit testing without owning render state.
 */
export type ShapeIdFn = () => number;

export function attachDrag(
  canvas: HTMLCanvasElement,
  pills: Pill[],
  dpr: number,
  getShapeId: ShapeIdFn = () => 0,
): () => void {
  let state: DragState = { kind: 'idle' };

  const toWorld = (e: PointerEvent): { x: number; y: number } => {
    const r = canvas.getBoundingClientRect();
    return { x: (e.clientX - r.left) * dpr, y: (e.clientY - r.top) * dpr };
  };

  // Hit testing is shape-aware:
  //   pill / prism (shape 0 / 1): XY axis-aligned box (Z is not visible top-down)
  //   cube (shape 2):             circular radius using the 3D diagonal, because
  //                               the rotating cube's silhouette changes as it
  //                               tumbles — a circle around the center always
  //                               contains the visible footprint.
  const findHit = (x: number, y: number): number => {
    const shapeId = getShapeId();
    for (let i = pills.length - 1; i >= 0; i--) {
      const p  = pills[i]!;
      const dx = x - p.cx;
      const dy = y - p.cy;
      if (shapeId === 2) {
        const r = Math.max(p.hx, p.hy, p.hz);
        if (dx * dx + dy * dy <= r * r) return i;
      } else {
        if (Math.abs(dx) <= p.hx && Math.abs(dy) <= p.hy) return i;
      }
    }
    return -1;
  };

  const release = (pointerId: number) => {
    if (state.kind === 'dragging') {
      try { canvas.releasePointerCapture(pointerId); } catch (err) {
        console.debug('[pills] releasePointerCapture failed:', err);
      }
      state = { kind: 'idle' };
    }
  };

  const down = (e: PointerEvent) => {
    const { x, y } = toWorld(e);
    const i = findHit(x, y);
    if (i < 0) return;
    state = {
      kind:      'dragging',
      pillIndex: i,
      offsetX:   x - pills[i]!.cx,
      offsetY:   y - pills[i]!.cy,
      pointerId: e.pointerId,
    };
    try { canvas.setPointerCapture(e.pointerId); } catch (err) {
      console.debug('[pills] setPointerCapture failed:', err);
    }
  };

  const move = (e: PointerEvent) => {
    if (state.kind !== 'dragging') return;
    const { x, y } = toWorld(e);
    const p = pills[state.pillIndex];
    if (!p) { release(e.pointerId); return; }
    p.cx = x - state.offsetX;
    p.cy = y - state.offsetY;
  };

  const onRelease = (e: PointerEvent) => release(e.pointerId);
  const onBlur    = () => { if (state.kind === 'dragging') release(state.pointerId); };
  const onVis     = () => { if (document.hidden && state.kind === 'dragging') release(state.pointerId); };

  canvas.addEventListener('pointerdown',   down);
  canvas.addEventListener('pointermove',   move);
  canvas.addEventListener('pointerup',     onRelease);
  canvas.addEventListener('pointercancel', onRelease);
  window.addEventListener('blur',              onBlur);
  document.addEventListener('visibilitychange', onVis);

  return () => {
    if (state.kind === 'dragging') release(state.pointerId);
    canvas.removeEventListener('pointerdown',   down);
    canvas.removeEventListener('pointermove',   move);
    canvas.removeEventListener('pointerup',     onRelease);
    canvas.removeEventListener('pointercancel', onRelease);
    window.removeEventListener('blur',              onBlur);
    document.removeEventListener('visibilitychange', onVis);
  };
}
