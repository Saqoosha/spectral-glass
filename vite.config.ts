import { defineConfig } from 'vite';

// `base` matters only for `vite build` — production needs the GitHub Pages
// subpath so /assets/foo.js resolves under https://saqoosha.github.io/
// <repo>/. It must match the repository name (case-sensitive in the path).
// `vite dev` and `vite preview` keep base='/' so localhost works without a rewrite.
export default defineConfig(({ command }) => ({
  base:          command === 'build' ? '/Spectral-Glass/' : '/',
  server:        { port: 5173 },
  assetsInclude: ['**/*.wgsl'],
}));
