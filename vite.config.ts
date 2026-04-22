import { defineConfig } from 'vite';

// `base` matters only for `vite build` — production needs the GitHub Pages
// subpath so /assets/foo.js resolves under https://saqoosha.github.io/
// spectral-glass/. `vite dev` and `vite preview` keep base='/' so localhost
// continues to work without a rewrite.
export default defineConfig(({ command }) => ({
  base:          command === 'build' ? '/spectral-glass/' : '/',
  server:        { port: 5173 },
  assetsInclude: ['**/*.wgsl'],
}));
