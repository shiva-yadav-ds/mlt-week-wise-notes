const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
(async () => {
  const blocks = Array.from({length: 19}, (_, i) => ({i}));
  const batches = []; let yields = 0;
  const context = {
    window: {scrollY: 1}, location: {hash: '#topic-3'},
    document: {querySelectorAll: () => blocks},
    setTimeout: callback => { yields++; callback(); },
    MathJax: {typesetPromise: async batch => batches.push(batch)},
  };
  vm.runInNewContext(fs.readFileSync(`${__dirname}/dist/math.js`, 'utf8'), context);
  await context.window.renderNotesMath();
  assert.deepEqual(batches.flat(), blocks);
  assert.equal(yields, 3);
  assert.ok(batches.every(batch => batch.length <= 8));
  const listeners = {}; const hints = [];
  const navigator = {connection: {saveData: false}};
  vm.runInNewContext(fs.readFileSync(`${__dirname}/dist/navigation.js`, 'utf8'), {
    navigator, document: {
      addEventListener: (name, fn) => {listeners[name] = fn;},
      createElement: () => ({}), head: {append: hint => hints.push(hint)},
    },
  });
  const event = (href, selected = false) => ({target: {closest: () => ({href, hasAttribute: () => selected})}});
  listeners.pointerover(event('week2.html'));
  listeners.focusin(event('week2.html'));
  listeners.pointerover(event('index.html', true));
  navigator.connection.saveData = true;
  listeners.pointerover(event('week3.html'));
  assert.equal(hints.length, 1);
  assert.equal(hints[0].rel, 'prefetch');
  assert.equal(hints[0].href, 'week2.html');
  console.log('Passed: complete ordered math batches, yielding, navigation prefetch deduplication and data-saver protection.');
})().catch(error => {console.error(error); process.exitCode = 1;});
