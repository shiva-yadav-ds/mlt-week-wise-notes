(() => {
  const media = window.matchMedia('(prefers-color-scheme: dark)');
  const valid = value => ['system', 'light', 'dark'].includes(value);
  let mode = 'system';
  try { const saved = localStorage.getItem('mlt-theme'); if (valid(saved)) mode = saved; } catch {}
  const apply = () => {
    document.documentElement.dataset.theme = mode === 'system' ? (media.matches ? 'dark' : 'light') : mode;
  };
  apply();
  media.addEventListener('change', apply);
  window.addEventListener('storage', event => {
    if (event.key === 'mlt-theme' || event.key === null) {
      mode = valid(event.newValue) ? event.newValue : 'system';
      apply();
      const select = document.getElementById('theme');
      if (select) select.value = mode;
    }
  });
  document.addEventListener('DOMContentLoaded', () => {
    const select = document.getElementById('theme');
    select.value = mode;
    select.addEventListener('change', () => {
      mode = select.value;
      try { localStorage.setItem('mlt-theme', mode); } catch {}
      apply();
    });
  });
})();
