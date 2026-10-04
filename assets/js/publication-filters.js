// One bibliography DOM: filters never duplicate paper IDs or abstract controls.
(() => {
  const controls = document.querySelector('.publication-filters');
  if (!controls) return;
  const buttons = Array.from(controls.querySelectorAll('button'));
  const entries = Array.from(document.querySelectorAll('.publications li')).filter(li => li.querySelector('.publication-entry'));
  const status = document.getElementById('publication-filter-status');
  function applyFilter(key, label) {
    let count = 0;
    entries.forEach(li => {
      const group = li.querySelector('.publication-entry').dataset.researchGroup;
      li.hidden = key !== 'all' && group !== key;
      if (!li.hidden) count += 1;
    });
    document.querySelectorAll('.publications ol.bibliography').forEach(list => {
      const hasVisibleEntries = Array.from(list.children).some(li => !li.hidden);
      list.hidden = !hasVisibleEntries;
      if (list.previousElementSibling?.tagName === 'H2') list.previousElementSibling.hidden = !hasVisibleEntries;
    });
    buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.publicationFilter === key)));
    status.textContent = `${count} ${count === 1 ? 'paper' : 'papers'} · ${label}`;
  }
  controls.hidden = false;
  buttons.forEach(button => button.addEventListener('click', () => applyFilter(button.dataset.publicationFilter, button.textContent.trim())));
  applyFilter('all', 'All papers');
})();
