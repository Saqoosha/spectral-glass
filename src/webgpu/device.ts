export type GpuContext = {
  device: GPUDevice;
  canvas: HTMLCanvasElement;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
  dpr: number;
};

export async function initGpu(canvasId: string): Promise<GpuContext | null> {
  if (!('gpu' in navigator)) return null;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return null;
  const device = await adapter.requestDevice();

  const canvas = document.getElementById(canvasId);
  if (!(canvas instanceof HTMLCanvasElement)) {
    throw new Error(`Canvas #${canvasId} not found`);
  }
  const context = canvas.getContext('webgpu');
  if (!context) return null;

  const format = navigator.gpu.getPreferredCanvasFormat();
  const dpr = Math.min(window.devicePixelRatio ?? 1, 2);
  resizeCanvas(canvas, dpr);
  context.configure({ device, format, alphaMode: 'opaque' });

  return { device, canvas, context, format, dpr };
}

export function resizeCanvas(canvas: HTMLCanvasElement, dpr: number): { width: number; height: number } {
  const width = Math.floor(canvas.clientWidth * dpr);
  const height = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { width, height };
}
