/**
 * Wavelength-dependent IOR using the glTF KHR_materials_dispersion formulation:
 *   n(λ) = n_d + (n_d - 1) / V_d * (523655 / λ² − 1.5168)
 *
 * - λ in nm (visible: 380–700)
 * - n_d: refractive index at the sodium d-line (≈587.56 nm)
 * - V_d: Abbe number (lower = more dispersion)
 */
export function cauchyIor(lambdaNm: number, n_d: number, V_d: number): number {
  const offset = (n_d - 1) / V_d * (523655 / (lambdaNm * lambdaNm) - 1.5168);
  return Math.max(n_d + offset, 1.0);
}
