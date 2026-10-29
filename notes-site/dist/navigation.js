(() => {
  const prefetched = new Set();
  const prefetch = event => {
    if (navigator.connection?.saveData) return;
    const link = event.target.closest('.week-link, .next-prev a');
    if (!link || link.hasAttribute('aria-current') || prefetched.has(link.href)) return;
    prefetched.add(link.href);
    const hint = document.createElement('link');
    hint.rel = 'prefetch';
    hint.href = link.href;
    document.head.append(hint);
  };
  document.addEventListener('pointerover', prefetch);
  document.addEventListener('focusin', prefetch);
})();
