/**
 * Minimal Radiance .hdr decoder.
 *
 * Parses the "#?RADIANCE" RGBE format used by Poly Haven and most
 * HDR pipelines. The output is a linear-RGB float image suitable for
 * uploading into an rgba16f WebGPU texture for IBL reflection sampling.
 *
 * Supports:
 *   - RGBE (not XYZE — explicitly rejected).
 *   - "Adaptive" RLE (width in [8, 32767]) — the format Poly Haven ships.
 *   - Both `-Y H +X W` and `+Y H +X W` scanline ordering (the latter
 *     flipped so row 0 ends up at the top of our output buffer).
 *
 * Does NOT support:
 *   - The legacy per-pixel RLE used for ultra-narrow/ultra-wide scanlines
 *     (width < 8 or > 32767). Poly Haven 1K/2K assets never hit these
 *     limits; we throw with a clear message if one slips through.
 *   - XYZE colour space (CIE XYZ values). The jewelry HDRIs curated for
 *     this project are all sRGB-primaries; drop-in support would cost a
 *     matrix multiply per pixel and an entire colour-space story, so it's
 *     out of scope until an XYZE file actually arrives.
 *
 * Spec: Greg Ward's Radiance file format, documented in "Real Pixels"
 * (Graphics Gems II, 1991). The RGBE encoding packs an HDR RGB triplet
 * into 4 bytes:
 *     value_channel = byte_channel * 2^(byte_e - 128) / 256
 * with a shared exponent across the three colour channels. The /256 is
 * Ward's mantissa normalisation; equivalent to a left-shift of -8 in the
 * exponent (`2^(e - 136)`), which we evaluate directly via `Math.pow`.
 */

export type HdrImage = {
  readonly width:  number;
  readonly height: number;
  /** Row-major, top-left origin, 3 floats per pixel (R, G, B) — linear light.
   *  No pre-multiplied alpha, no sRGB encoding. Caller handles tone-mapping.
   *
   *  INVARIANT: `rgb.length === width * height * 3`. Not expressible at the
   *  type level in TypeScript (no dependent-type support for Float32Array
   *  length), so enforced at consumption boundaries: `uploadEnvmap`
   *  validates the invariant at `src/envmap.ts:61` and throws if violated.
   *
   *  Design debate (review-fix-loop iter 1 + iter 2 — STUCK): promoting
   *  `HdrImage` to a branded/newtype wrapper would catch hand-assembled
   *  objects that violate the length invariant at compile time. Deferred
   *  because the upload-side throw already catches violators at the
   *  single real consumption site, and the branded type would ripple
   *  through every test that constructs a synthetic HDR image. */
  readonly rgb:    Float32Array;
};

/** Decode a Radiance .hdr byte stream into a linear-RGB float image.
 *  Throws on malformed / unsupported input; callers should wrap in a
 *  try/catch and fall back to an LDR default if the user-facing mode
 *  should degrade gracefully (see envmap.ts for the standard pattern). */
export function decodeHdr(bytes: Uint8Array): HdrImage {
  // ---- Header ----
  // The header is ASCII up through the first blank line; we walk byte-
  // by-byte to avoid allocating a giant string for the whole file.
  let cursor = 0;
  const decoder = new TextDecoder('ascii');

  const readLine = (): string => {
    let end = cursor;
    while (end < bytes.length && bytes[end] !== 0x0a) end++;   // '\n'
    const line = decoder.decode(bytes.subarray(cursor, end));
    cursor = end + 1;   // skip the newline
    return line;
  };

  const magic = readLine();
  if (magic !== '#?RADIANCE' && !magic.startsWith('#?')) {
    throw new Error(`HDR: missing #?RADIANCE magic line, got: ${magic}`);
  }

  let format: string | null = null;
  for (;;) {
    const line = readLine();
    if (line === '') break;                    // blank line = end of header
    if (line.startsWith('#')) continue;        // comment
    const eq = line.indexOf('=');
    if (eq < 0) continue;                      // ignore stray lines
    const key = line.slice(0, eq);
    const val = line.slice(eq + 1);
    if (key === 'FORMAT') format = val;
    // EXPOSURE / GAMMA / COLORCORR intentionally ignored — tone mapping
    // happens at use-site, not decode-site.
  }
  if (format !== '32-bit_rle_rgbe') {
    throw new Error(`HDR: unsupported FORMAT=${format}; only RGBE handled.`);
  }

  // Resolution line, e.g. "-Y 512 +X 1024". The standard order is -Y
  // first (top-down scanlines), but some writers emit +Y (bottom-up);
  // we flip row index if so. X is always +X in practice; reject -X for
  // now rather than invent a reversed-column writer that likely doesn't
  // exist for our curated HDRI set.
  const resLine = readLine();
  const resMatch = /([-+]Y)\s+(\d+)\s+([-+]X)\s+(\d+)/.exec(resLine);
  if (!resMatch) throw new Error(`HDR: unparseable resolution line: ${resLine}`);
  const yDir  = resMatch[1]!;
  const height = parseInt(resMatch[2]!, 10);
  if (resMatch[3]! !== '+X') {
    throw new Error(`HDR: reversed X direction (${resMatch[3]}) not supported.`);
  }
  const width  = parseInt(resMatch[4]!, 10);
  const yFlip = yDir === '+Y';

  if (width < 8 || width > 0x7fff) {
    throw new Error(`HDR: width=${width} requires legacy RLE (unsupported).`);
  }

  // ---- Scanlines ----
  const rgb = new Float32Array(width * height * 3);
  // Reused per-scanline buffer — avoids `height` allocations.
  const scan = new Uint8Array(width * 4);

  for (let y = 0; y < height; y++) {
    // Adaptive RLE header: 4 bytes = 2, 2, (width hi), (width lo).
    // The combined preamble `magic0==2 && magic1==2 && (wHi & 0x80)==0`
    // confirms adaptive RLE. The `wHi & 0x80` guard defends against a
    // legacy-RLE scanline whose first pixel happens to start with
    // R=G=2 — without it, such a legacy pixel would be mis-read as an
    // adaptive-RLE preamble declaring a garbage scanline width.
    if (cursor + 4 > bytes.length) {
      throw new Error(`HDR: unexpected EOF at scanline ${y}`);
    }
    const magic0 = bytes[cursor]!;
    const magic1 = bytes[cursor + 1]!;
    const wHi    = bytes[cursor + 2]!;
    const wLo    = bytes[cursor + 3]!;
    if (magic0 !== 2 || magic1 !== 2 || (wHi & 0x80) !== 0) {
      throw new Error(
        `HDR: scanline ${y} not in adaptive RLE format `
        + `(got [${magic0}, ${magic1}, ${wHi}, ${wLo}])`,
      );
    }
    const scanWidth = (wHi << 8) | wLo;
    if (scanWidth !== width) {
      throw new Error(`HDR: scanline ${y} width=${scanWidth} != image width=${width}`);
    }
    cursor += 4;

    // 4 channels, each RLE-encoded independently. Each run starts with a
    // byte `code`:
    //   code > 128 → run of (code - 128) same values (next byte).
    //   code ≤ 128 → dump of code distinct values (next `code` bytes).
    // We store the scanline channel-separated (RRR…GGG…BBB…EEE…) here
    // and interleave into rgb[] after the full scanline is decoded.
    for (let ch = 0; ch < 4; ch++) {
      let x = 0;
      while (x < width) {
        if (cursor >= bytes.length) {
          throw new Error(`HDR: EOF mid-RLE at scanline ${y}, channel ${ch}, x=${x}`);
        }
        const code = bytes[cursor++]!;
        if (code > 128) {
          const runLen = code - 128;
          if (x + runLen > width || cursor >= bytes.length) {
            throw new Error(`HDR: RLE run overflows scanline at y=${y} ch=${ch}`);
          }
          const val = bytes[cursor++]!;
          const base = ch * width;
          for (let i = 0; i < runLen; i++) scan[base + x + i] = val;
          x += runLen;
        } else {
          const dumpLen = code;
          if (x + dumpLen > width || cursor + dumpLen > bytes.length) {
            throw new Error(`HDR: RLE dump overflows scanline at y=${y} ch=${ch}`);
          }
          const base = ch * width;
          for (let i = 0; i < dumpLen; i++) scan[base + x + i] = bytes[cursor++]!;
          x += dumpLen;
        }
      }
    }

    // Interleave channel-separated scanline into row-major RGB floats.
    // `2^(e - 136)` = (2^(e - 128)) / 256 — Ward's /256 mantissa
    // normalisation folded into the exponent for a single Math.pow call.
    const dstY = yFlip ? (height - 1 - y) : y;
    const dstBase = dstY * width * 3;
    for (let x = 0; x < width; x++) {
      const r = scan[x]!;
      const g = scan[x + width]!;
      const b = scan[x + width * 2]!;
      const e = scan[x + width * 3]!;
      if (e === 0) {
        // RGBE convention: any channel with e=0 is exactly zero; no
        // negative exponent encoding. The three mantissa bytes are
        // ignored per the spec.
        rgb[dstBase + x * 3 + 0] = 0;
        rgb[dstBase + x * 3 + 1] = 0;
        rgb[dstBase + x * 3 + 2] = 0;
      } else {
        const scale = Math.pow(2, e - 136);
        rgb[dstBase + x * 3 + 0] = r * scale;
        rgb[dstBase + x * 3 + 1] = g * scale;
        rgb[dstBase + x * 3 + 2] = b * scale;
      }
    }
  }

  return { width, height, rgb };
}

/** Round-trip encoder used by tests + any future save path. NOT shipped
 *  to production hot paths — the encoder loops over every pixel in JS
 *  and allocates a fresh output per scanline. */
export function encodeHdr(img: HdrImage): Uint8Array {
  const { width, height, rgb } = img;
  if (width < 8 || width > 0x7fff) {
    throw new Error(`HDR encode: width=${width} outside supported range.`);
  }
  const header
    = '#?RADIANCE\n'
    + 'FORMAT=32-bit_rle_rgbe\n'
    + '\n'
    + `-Y ${height} +X ${width}\n`;
  const headerBytes = new TextEncoder().encode(header);

  // Worst-case RLE expansion is width + ceil(width/128) per channel, plus
  // a 4-byte header per scanline. `2 * width` over-allocates safely.
  const out: number[] = [];
  for (let i = 0; i < headerBytes.length; i++) out.push(headerBytes[i]!);

  const scan = new Uint8Array(width * 4);
  for (let y = 0; y < height; y++) {
    const srcBase = y * width * 3;
    for (let x = 0; x < width; x++) {
      const r = rgb[srcBase + x * 3 + 0]!;
      const g = rgb[srcBase + x * 3 + 1]!;
      const b = rgb[srcBase + x * 3 + 2]!;
      const m = Math.max(r, g, b);
      if (m < 1e-32) {
        scan[x]               = 0;
        scan[x + width]       = 0;
        scan[x + width * 2]   = 0;
        scan[x + width * 3]   = 0;
      } else {
        // Clamp the exponent to what fits in RGBE's (e + 128) uint8
        // byte: the format's theoretical range is e ∈ [-128, +127],
        // giving stored bytes [0, 255]. Without this clamp, a raw HDR
        // value above 2^127 would wrap `e + 128` past 255 into 0 and
        // silently encode as all-black. Poly Haven HDRIs don't reach
        // there (fp32 keeps them well under), but callers supplying
        // synthetic floats might, and `encodeHdr` is the test oracle
        // so it must be trustworthy at the format boundary.
        const e = Math.min(127, Math.max(-128, Math.ceil(Math.log2(m))));
        const s = Math.pow(2, 8 - e);
        scan[x]               = Math.min(255, Math.max(0, Math.round(r * s)));
        scan[x + width]       = Math.min(255, Math.max(0, Math.round(g * s)));
        scan[x + width * 2]   = Math.min(255, Math.max(0, Math.round(b * s)));
        scan[x + width * 3]   = e + 128;
      }
    }
    // Adaptive RLE header.
    out.push(2, 2, (width >> 8) & 0xff, width & 0xff);
    // Emit each channel as a single dump (no compression — tests only
    // need round-trip correctness, not file-size optimality).
    for (let ch = 0; ch < 4; ch++) {
      let x = 0;
      while (x < width) {
        const remaining = width - x;
        const chunk = Math.min(128, remaining);
        out.push(chunk);
        for (let i = 0; i < chunk; i++) out.push(scan[ch * width + x + i]!);
        x += chunk;
      }
    }
  }
  return new Uint8Array(out);
}
