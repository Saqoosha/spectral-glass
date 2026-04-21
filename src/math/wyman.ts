type Lobe = readonly [amp: number, mu: number, s1: number, s2: number];

const X_LOBES: readonly Lobe[] = [
  [ 0.362, 442.0, 16.0, 26.7],
  [ 1.056, 599.8, 37.9, 31.0],
  [-0.065, 501.1, 20.4, 26.2],
];
const Y_LOBES: readonly Lobe[] = [
  [0.821, 568.8, 46.9, 40.5],
  [0.286, 530.9, 16.3, 31.1],
];
const Z_LOBES: readonly Lobe[] = [
  [1.217, 437.0, 11.8, 36.0],
  [0.681, 459.0, 26.0, 13.8],
];

function sumLobes(lobes: readonly Lobe[], lambda: number): number {
  let acc = 0;
  for (const [amp, mu, s1, s2] of lobes) {
    const sigma = lambda < mu ? s1 : s2;
    const t = (lambda - mu) / sigma;
    acc += amp * Math.exp(-0.5 * t * t);
  }
  return acc;
}

/** Wyman-Sloan-Shirley (JCGT 2013) analytic CIE 1931 2° XYZ matching functions. */
export function cieXyz(lambdaNm: number): [number, number, number] {
  return [
    sumLobes(X_LOBES, lambdaNm),
    sumLobes(Y_LOBES, lambdaNm),
    sumLobes(Z_LOBES, lambdaNm),
  ];
}
