export type Pill = {
  cx: number; cy: number; cz: number;
  hx: number; hy: number; hz: number;
  edgeR: number;
};

export function defaultPills(width: number, height: number, count = 4): Pill[] {
  const pills: Pill[] = [];
  const step = width / (count + 1);
  for (let i = 0; i < count; i++) {
    pills.push({
      cx: step * (i + 1),
      cy: height * 0.5 + (i % 2 === 0 ? -60 : 60),
      cz: 0,
      hx: 160, hy: 44, hz: 20,
      edgeR: 14,
    });
  }
  return pills;
}

type DragState = { pillIndex: number | null; offsetX: number; offsetY: number };

export function attachDrag(canvas: HTMLCanvasElement, pills: Pill[], dpr: number): () => void {
  const state: DragState = { pillIndex: null, offsetX: 0, offsetY: 0 };

  const toWorld = (e: PointerEvent) => {
    const r = canvas.getBoundingClientRect();
    return { x: (e.clientX - r.left) * dpr, y: (e.clientY - r.top) * dpr };
  };

  const findHit = (x: number, y: number): number => {
    for (let i = pills.length - 1; i >= 0; i--) {
      const p = pills[i]!;
      if (Math.abs(x - p.cx) <= p.hx && Math.abs(y - p.cy) <= p.hy) return i;
    }
    return -1;
  };

  const down = (e: PointerEvent) => {
    const { x, y } = toWorld(e);
    const i = findHit(x, y);
    if (i >= 0) {
      state.pillIndex = i;
      state.offsetX   = x - pills[i]!.cx;
      state.offsetY   = y - pills[i]!.cy;
      canvas.setPointerCapture(e.pointerId);
    }
  };
  const move = (e: PointerEvent) => {
    if (state.pillIndex === null) return;
    const { x, y } = toWorld(e);
    const p = pills[state.pillIndex]!;
    p.cx = x - state.offsetX;
    p.cy = y - state.offsetY;
  };
  const release = (e: PointerEvent) => {
    if (state.pillIndex !== null) {
      canvas.releasePointerCapture(e.pointerId);
      state.pillIndex = null;
    }
  };

  canvas.addEventListener('pointerdown', down);
  canvas.addEventListener('pointermove', move);
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);

  return () => {
    canvas.removeEventListener('pointerdown', down);
    canvas.removeEventListener('pointermove', move);
    canvas.removeEventListener('pointerup', release);
    canvas.removeEventListener('pointercancel', release);
  };
}
