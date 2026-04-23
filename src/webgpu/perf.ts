// GPU timestamp-query harness. Captures start/end timestamps around the draw
// pass via `timestampWrites`, resolves to a staging buffer, and maps a separate
// readback buffer on demand. `record()` uses a double-buffered ping-pong so the
// CPU can read frame N-1 while frame N is still in flight.
//
// Spec: resolved values are uint64 **nanoseconds** (§20.4). Browsers also apply
// `coarsen time` to queue timestamps for security, so deltas often look stepped
// or nearly constant — that is not a bug. `t1 - t0` is the scene render pass.

export type Perf = {
  readonly querySet: GPUQuerySet;
  readonly writes:   GPURenderPassTimestampWrites;
  /** Copy query results into the readback chain. Call after the render pass ends. */
  resolve(encoder: GPUCommandEncoder): void;
  /** Asynchronously read the most recently resolved pair. Returns null while busy. */
  readMs(): Promise<number | null>;
};

type Slot = {
  resolveBuf:  GPUBuffer;  // GPU-visible, QUERY_RESOLVE | COPY_SRC
  readBuf:     GPUBuffer;  // MAP_READ | COPY_DST
  inFlight:    boolean;
};

export function createPerf(device: GPUDevice): Perf {
  const querySet = device.createQuerySet({
    label: 'perf-timestamps',
    type:  'timestamp',
    count: 2,
  });

  const slots: Slot[] = [0, 1].map(() => ({
    resolveBuf: device.createBuffer({
      label: 'perf-resolve',
      size:  2 * 8,  // 2 u64 timestamps
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    }),
    readBuf: device.createBuffer({
      label: 'perf-read',
      size:  2 * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }),
    inFlight: false,
  }));
  let writeSlot = 0;

  const writes: GPURenderPassTimestampWrites = {
    querySet,
    beginningOfPassWriteIndex: 0,
    endOfPassWriteIndex:       1,
  };

  return {
    querySet,
    writes,
    resolve(encoder) {
      const slot = slots[writeSlot]!;
      if (slot.inFlight) return;  // previous readback hasn't completed; skip this frame
      encoder.resolveQuerySet(querySet, 0, 2, slot.resolveBuf, 0);
      encoder.copyBufferToBuffer(slot.resolveBuf, 0, slot.readBuf, 0, 16);
      slot.inFlight = true;
      writeSlot = 1 - writeSlot;
    },
    async readMs() {
      const slot = slots[1 - writeSlot]!;  // the one we just wrote to
      if (!slot.inFlight) return null;
      await slot.readBuf.mapAsync(GPUMapMode.READ);
      // Read timestamps before unmap. Avoid `getMappedRange().slice(0)` — that
      // copied 16 B every frame and could trigger periodic GC + main-thread
      // hitches (felt like ~1s stutters while dragging) on top of mapAsync.
      const range = slot.readBuf.getMappedRange();
      const view  = new BigUint64Array(range);
      const t0    = view[0]!;
      const t1    = view[1]!;
      slot.readBuf.unmap();
      slot.inFlight = false;
      if (t1 < t0) return null;  // counter reset or invalid pair (rare per spec)
      return Number(t1 - t0) / 1e6;  // ns → ms
    },
  };
}
