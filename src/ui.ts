import { Pane } from 'tweakpane';

export type Params = {
  sampleCount: 3 | 8 | 16 | 32;
  n_d: number;
  V_d: number;
  pillLen: number;
  pillShort: number;
  pillThick: number;
  edgeR: number;
  refractionStrength: number;
  refractionMode: 'exact' | 'approx';
  temporalJitter: boolean;
};

export function initUi(params: Params, reloadPhoto: () => void): Pane {
  const pane = new Pane({ title: 'Spectral Dispersion', expanded: true });

  const spectral = pane.addFolder({ title: 'Spectral' });
  spectral.addBinding(params, 'sampleCount', {
    options: { '3 (fake RGB)': 3, '8 (default)': 8, '16': 16, '32': 32 },
  });
  spectral.addBinding(params, 'n_d', { min: 1.0, max: 2.4, step: 0.001, label: 'IOR n_d' });
  spectral.addBinding(params, 'V_d', { min: 15,  max: 90,  step: 0.5,   label: 'Abbe V_d' });
  spectral.addBinding(params, 'refractionMode', {
    options: { Exact: 'exact', Approx: 'approx' },
  });
  spectral.addBinding(params, 'temporalJitter', { label: 'Temporal jitter' });

  const shape = pane.addFolder({ title: 'Pill shape' });
  shape.addBinding(params, 'pillLen',   { min: 80,  max: 800, step: 1, label: 'Length' });
  shape.addBinding(params, 'pillShort', { min: 20,  max: 200, step: 1, label: 'Short axis' });
  shape.addBinding(params, 'pillThick', { min: 10,  max: 200, step: 1, label: 'Thickness' });
  shape.addBinding(params, 'edgeR',     { min: 1,   max: 100, step: 0.5, label: 'Edge radius' });

  const misc = pane.addFolder({ title: 'Misc' });
  misc.addBinding(params, 'refractionStrength', { min: 0, max: 0.5, step: 0.001, label: 'Refraction' });
  const reload = misc.addButton({ title: 'Reload photo' });
  reload.on('click', reloadPhoto);

  return pane;
}

export function defaultParams(): Params {
  return {
    sampleCount: 8,
    n_d: 1.5168,
    V_d: 40,
    pillLen: 320,
    pillShort: 88,
    pillThick: 40,
    edgeR: 14,
    refractionStrength: 0.1,
    refractionMode: 'exact',
    temporalJitter: true,
  };
}
