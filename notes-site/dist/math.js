// Yield between batches so long notes do not block week navigation.
window.renderNotesMath = async () => {
  const blocks = [...document.querySelectorAll('article > *, .contents nav')];
  for (let i = 0; i < blocks.length; i += 8) {
    await new Promise(resolve => setTimeout(resolve, 0));
    await MathJax.typesetPromise(blocks.slice(i, i + 8));
  }
  // Restore deep links after formula layout has settled, unless the reader moved.
  if (location.hash && !window.scrollY) {
    let id;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
    document.getElementById(id)?.scrollIntoView();
  }
};
