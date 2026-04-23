/** HTML-in-Canvas (Chromium dev trial, `CanvasDrawElement`) — optional queue method. */
interface GPUQueue {
  copyElementImageToTexture?(
    source: Element,
    destination: GPUImageCopyTextureTagged,
  ): void;
}

interface HTMLCanvasElement {
  /** Opts direct children into layout for `drawElementImage` / `copyElementImageToTexture`. */
  layoutSubtree?: boolean;
  requestPaint?: () => void;
  onpaint?: ((this: HTMLCanvasElement, ev: Event) => void) | null;
}
