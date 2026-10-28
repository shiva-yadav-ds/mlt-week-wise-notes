const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const code = fs.readFileSync(__dirname + '/dist/theme.js', 'utf8');
function run(saved, dark, blocked = false) {
 const handlers = {}, select = { addEventListener: (k,f) => handlers.select = f };
 const media = { matches: dark, addEventListener: (k,f) => handlers.media = f };
 const document = { documentElement: { dataset: {} }, getElementById: () => select, addEventListener: (k,f) => handlers.ready = f };
 const localStorage = { getItem: () => { if(blocked) throw Error(); return saved; }, setItem: (k,v) => { if(blocked) throw Error(); saved = v; } };
 vm.runInNewContext(code, { document, localStorage, window: { matchMedia: () => media, addEventListener: (k,f) => handlers.storage = f } });
 handlers.ready();
 return { document, media, handlers, select };
}
for (const dark of [true,false]) {
 const x=run(null,dark); assert.equal(x.document.documentElement.dataset.theme,dark?'dark':'light');
 x.media.matches=!dark;x.handlers.media();assert.equal(x.document.documentElement.dataset.theme,dark?'light':'dark');
 x.select.value='dark';x.handlers.select();x.media.matches=false;x.handlers.media();assert.equal(x.document.documentElement.dataset.theme,'dark');
 x.handlers.storage({key:'mlt-theme',newValue:'light'});assert.equal(x.select.value,'light');
}
assert.equal(run('dark',false).document.documentElement.dataset.theme,'dark');
assert.equal(run('invalid',true).select.value,'system');
const blocked=run(null,true,true);blocked.select.value='light';blocked.handlers.select();assert.equal(blocked.document.documentElement.dataset.theme,'light');
console.log('Theme checks passed: system changes, manual override, persistence, storage sync and unavailable storage.');
