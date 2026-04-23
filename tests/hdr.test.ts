import { describe, it, expect } from 'vitest';
import { decodeHdr, encodeHdr, type HdrImage } from '../src/hdr';

/** Build a small synthetic HdrImage with diverse pixel values for
 *  round-trip testing. Covers: black (e=0 path), mid-range, high-
 *  dynamic-range values above 1.0, and an edge pixel at full saturation
 *  for a channel. */
function makeTestImage(width = 16, height = 8): HdrImage {
  const rgb = new Float32Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 3;
      if (x === 0 && y === 0) {
        // Exact black — tests the e=0 short-circuit.
        rgb[i + 0] = 0; rgb[i + 1] = 0; rgb[i + 2] = 0;
      } else if (x === 1 && y === 0) {
        // Huge highlight (sun-disc bright) — tests large-exponent encoding.
        rgb[i + 0] = 100.0; rgb[i + 1] = 95.0; rgb[i + 2] = 85.0;
      } else {
        // Smooth gradient in linear light, capped below exponent quantum
        // to make round-trip predictable.
        rgb[i + 0] = 0.1 + 0.8 * (x / width);
        rgb[i + 1] = 0.05 + 0.9 * (y / height);
        rgb[i + 2] = 0.3 + 0.6 * ((x + y) / (width + height));
      }
    }
  }
  return { width, height, rgb };
}

describe('decodeHdr + encodeHdr', () => {
  it('rejects byte streams missing the RADIANCE magic', () => {
    const bytes = new TextEncoder().encode('garbage\nnot an hdr\n\n-Y 1 +X 1\n');
    expect(() => decodeHdr(bytes)).toThrow(/magic/i);
  });

  it('rejects non-RGBE formats (XYZE, GL-PFM, …)', () => {
    const bytes = new TextEncoder().encode(
      '#?RADIANCE\nFORMAT=32-bit_rle_xyze\n\n-Y 8 +X 8\n',
    );
    // Will throw before scanline decoding because of the format check.
    expect(() => decodeHdr(bytes)).toThrow(/RGBE|unsupported/i);
  });

  it('rejects scanlines that miss the adaptive-RLE magic bytes', () => {
    // Valid header, invalid scanline start. Build inline.
    const header = '#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 16\n';
    const headerBytes = new TextEncoder().encode(header);
    // Scanline magic should be [2, 2, 0, 16]; put garbage instead.
    const bad = new Uint8Array(headerBytes.length + 4 + 16 * 4);
    bad.set(headerBytes, 0);
    bad[headerBytes.length + 0] = 0xaa;  // not 2
    bad[headerBytes.length + 1] = 0xaa;
    bad[headerBytes.length + 2] = 0x00;
    bad[headerBytes.length + 3] = 0x10;
    expect(() => decodeHdr(bad)).toThrow(/adaptive RLE|magic/i);
  });

  it('rejects widths below the adaptive-RLE lower bound (legacy RLE)', () => {
    const bytes = new TextEncoder().encode(
      '#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 4\n',
    );
    expect(() => decodeHdr(bytes)).toThrow(/legacy|width/i);
  });

  it('round-trips every pixel of a synthetic image within RGBE quantisation', () => {
    const src = makeTestImage(16, 8);
    const encoded = encodeHdr(src);
    const decoded = decodeHdr(encoded);

    expect(decoded.width).toBe(src.width);
    expect(decoded.height).toBe(src.height);
    expect(decoded.rgb.length).toBe(src.rgb.length);

    // RGBE has ~1 part in 256 mantissa resolution. A per-channel
    // tolerance of 1% of the pixel's max channel is safe — smaller
    // than human perception, larger than the quantisation floor.
    // Exact black must round-trip exactly (the e=0 path).
    const blackIdx = 0;
    expect(decoded.rgb[blackIdx + 0]).toBe(0);
    expect(decoded.rgb[blackIdx + 1]).toBe(0);
    expect(decoded.rgb[blackIdx + 2]).toBe(0);

    // RGBE's 8-bit mantissa per channel gives ~0.5% worst-case relative
    // error when the pixel's max channel sets the shared exponent and
    // smaller channels get squashed to a coarser step. A 1% tolerance
    // accepts that quantisation without being so loose we'd miss a real
    // encoding bug.
    for (let p = 0; p < src.width * src.height; p++) {
      const i = p * 3;
      const r = src.rgb[i]!, g = src.rgb[i + 1]!, b = src.rgb[i + 2]!;
      const tol = Math.max(r, g, b, 1e-6) * 0.01;
      expect(Math.abs(decoded.rgb[i + 0]! - r)).toBeLessThan(tol);
      expect(Math.abs(decoded.rgb[i + 1]! - g)).toBeLessThan(tol);
      expect(Math.abs(decoded.rgb[i + 2]! - b)).toBeLessThan(tol);
    }
  });

  it('preserves the high-dynamic-range highlight', () => {
    // Pixel (1,0) is (100, 95, 85) — well above 1.0, the whole point of HDR.
    // LDR 8-bit would clamp to (1, 1, 1); we assert the round-trip keeps
    // the full magnitude.
    const src = makeTestImage();
    const encoded = encodeHdr(src);
    const decoded = decodeHdr(encoded);
    const hi = 1 * 3;   // pixel index 1, 3 floats per pixel
    expect(decoded.rgb[hi + 0]!).toBeGreaterThan(50);
    expect(decoded.rgb[hi + 1]!).toBeGreaterThan(50);
    expect(decoded.rgb[hi + 2]!).toBeGreaterThan(50);
  });

  it('flips +Y scanline order into our top-origin output buffer', () => {
    // Synthesise a 16×2 image where row 0 is red and row 1 is green.
    const width = 16, height = 2;
    const rgb = new Float32Array(width * height * 3);
    for (let x = 0; x < width; x++) {
      rgb[(0 * width + x) * 3 + 0] = 1;  // row 0: red
      rgb[(1 * width + x) * 3 + 1] = 1;  // row 1: green
    }
    const encoded = encodeHdr({ width, height, rgb });

    // Sanity: encodeHdr writes `-Y height +X width`, so standard
    // top-to-bottom decode yields our original layout.
    const decodedStd = decodeHdr(encoded);
    expect(decodedStd.rgb[0 + 0]!).toBeCloseTo(1, 2);           // row 0 red
    expect(decodedStd.rgb[width * 3 + 1]!).toBeCloseTo(1, 2);   // row 1 green

    // Patch the resolution line to `+Y` and verify the decoder flips
    // the scanlines. Without the flip, a +Y file (which stores row 0 at
    // the bottom of the file) would put green on top of our buffer.
    const idx = encoded.indexOf(0x0a, encoded.indexOf(0x0a, encoded.indexOf(0x0a) + 1) + 1);
    // Above indexOf chain skips: magic line + format line → resolution
    // line starts after the blank line's newline.
    const resLineStart = idx + 1;
    const resLineEnd   = encoded.indexOf(0x0a, resLineStart);
    const newResLine   = new TextEncoder().encode(`+Y ${height} +X ${width}`);
    const flipped = new Uint8Array(encoded.length);
    flipped.set(encoded.subarray(0, resLineStart), 0);
    flipped.set(newResLine, resLineStart);
    // Rest of the file (the newline + scanlines) shifts only if the new
    // resolution string length differs from the original; for small
    // dimensions "±Y NN +X NN" both have the same byte count.
    const oldResLen = resLineEnd - resLineStart;
    if (newResLine.length !== oldResLen) {
      throw new Error('test invariant: resolution line length must match for in-place patch');
    }
    flipped.set(encoded.subarray(resLineEnd), resLineEnd);

    const decoded = decodeHdr(flipped);
    // Row 0 of output should correspond to the LAST file-scanline; in
    // our encoded file that's row 1 (green).
    expect(decoded.rgb[0 + 1]!).toBeCloseTo(1, 2);           // top now green
    expect(decoded.rgb[width * 3 + 0]!).toBeCloseTo(1, 2);   // bottom now red
  });
});
