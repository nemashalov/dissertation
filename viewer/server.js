// Self-contained live markdown viewer for the dissertation draft.
// No external dependencies. Node >= 14.
// Usage: node viewer/server.js   (serves http://localhost:8765)
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 8765;
const ROOT = path.join(__dirname, '..', 'draft');

const CHAPTERS = [
  { id: 'plan',       file: '00-plan.md',                     title: 'План' },
  { id: 'intro',      file: '01-introduction.md',             title: 'Введение' },
  { id: 'ch1',        file: '02-chapter-1-fragmentation/README.md', title: 'Глава 1. Фрагментация и политики' },
  { id: 'ch2',        file: '03-chapter-2-overview/README.md', title: 'Глава 2. Обзор' },
  { id: 'ch3',        file: '04-chapter-3-model/README.md',   title: 'Глава 3. Модель' },
  { id: 'ch4',        file: '05-chapter-4-algorithms/README.md', title: 'Глава 4. Алгоритмы' },
  { id: 'ch5',        file: '06-chapter-5-game-theory/README.md', title: 'Глава 5. Теор.-игр. анализ' },
  { id: 'ch6',        file: '07-chapter-6-experiments/README.md', title: 'Глава 6. Эксперименты' },
  { id: 'conclusion', file: '08-conclusion.md',               title: 'Заключение' },
  { id: 'refs',       file: '09-references.md',               title: 'Список литературы' },
  { id: 'simplan',    file: '10-simulation-plan.md',          title: 'План симуляции' },
  { id: 'annotation', file: '11-annotation.md',               title: 'Аннотация' },
  { id: 'litrev',     file: '12-literature-review.md',        title: 'Обзор литературы' },
  { id: 'glossary',   file: '13-glossary.md',                 title: 'Словарь терминов' },
];

function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function mathTag(latex, display) {
  const cls = display ? 'math-block' : 'math';
  return '<' + (display ? 'div' : 'span') + ' class="' + cls + '" data-raw="' +
    encodeURIComponent(latex) + '">' + escapeHtml(latex) + '</' + (display ? 'div' : 'span') + '>';
}

function inline(s) {
  s = s.replace(/`([^`]+)`/g, (m, c) => '<code>' + escapeHtml(c) + '</code>');
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/(^|[^*])\*([^*]+)\*(?!\*)/g, '$1<em>$2</em>');
  s = s.replace(/\$([^$\n]+)\$/g, (m, c) => mathTag(c, false));
  s = s.replace(/\[([0-9,\-\s]+)\]/g, (m, refs) => {
    const nums = refs.split(/[,\s]+/).filter(Boolean);
    return nums.map(n => '<a class="cit" href="#refs" data-cit="' + n + '">[' + n + ']</a>').join(',');
  });
  return s;
}

function render(md) {
  const lines = md.split('\n');
  const out = [];
  let para = [];
  const flushPara = () => {
    if (para.length) {
      out.push('<p>' + para.map(inline).join(' ') + '</p>');
      para = [];
    }
  };
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const t = line.trim();
    // fenced code block
    if (/^```/.test(t)) {
      flushPara();
      const lang = (/^```([A-Za-z0-9_+-]*)/.exec(t) || [])[1] || '';
      i++;
      let buf = [];
      while (i < lines.length && !/^```\s*$/.test(lines[i])) { buf.push(lines[i]); i++; }
      i++; // closing fence
      if (lang === 'mermaid') {
        out.push('<pre class="mermaid">' + escapeHtml(buf.join('\n')) + '</pre>');
      } else {
        out.push('<pre><code>' + escapeHtml(buf.join('\n')) + '</code></pre>');
      }
      continue;
    }
    // math block: $$ ... $$
    if (/^\$\$/.test(t) && /\$\$$/.test(t.slice(2))) {
      flushPara();
      out.push(mathTag(t.slice(2, t.length - 2), true));
      i++;
      continue;
    }
    if (/^\$\$\s*$/.test(t)) {
      flushPara();
      i++;
      let buf = [];
      while (i < lines.length && !/^\$\$\s*$/.test(lines[i].trim())) { buf.push(lines[i]); i++; }
      i++;
      out.push(mathTag(buf.join(' '), true));
      continue;
    }
    // headings
    const h = /^(#{1,6})\s+(.*)$/.exec(line);
    if (h) {
      flushPara();
      out.push('<h' + h[1].length + '>' + inline(h[2]) + '</h' + h[1].length + '>');
      i++;
      continue;
    }
    // blockquote
    if (/^>\s?/.test(line)) {
      flushPara();
      let buf = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) { buf.push(lines[i].replace(/^>\s?/, '')); i++; }
      out.push('<blockquote>' + buf.map(inline).join('<br>') + '</blockquote>');
      continue;
    }
    // hr
    if (/^\s*-{3,}\s*$/.test(line)) { flushPara(); out.push('<hr>'); i++; continue; }
    // image: ![alt](path)
    const img = /^!\[(.*?)\]\((.*?)\)$/.exec(line);
    if (img) {
      flushPara();
      out.push('<div class="figure"><img src="' + escapeHtml(img[2]) + '" alt="' + escapeHtml(img[1]) + '"></div>');
      i++;
      continue;
    }
    // unordered list
    const bullet = /^[-*]\s+(.*)$/.exec(line);
    if (bullet) {
      flushPara();
      let items = [];
      while (i < lines.length && (m = /^[-*]\s+(.*)$/.exec(lines[i]))) { items.push(inline(m[1])); i++; }
      out.push('<ul>' + items.map(x => '<li>' + x + '</li>').join('') + '</ul>');
      continue;
    }
    // ordered list
    const ordered = /^\d+[.)]\s+(.*)$/.exec(line);
    if (ordered) {
      flushPara();
      let items = [];
      while (i < lines.length && (m = /^\d+[.)]\s+(.*)$/.exec(lines[i]))) { items.push(inline(m[1])); i++; }
      out.push('<ol>' + items.map(x => '<li>' + x + '</li>').join('') + '</ol>');
      continue;
    }
    // paragraph
    if (t === '') { flushPara(); i++; continue; }
    para.push(t);
    i++;
  }
  flushPara();
  return out.join('\n');
}

function loadChapters() {
  return CHAPTERS.map(c => {
    const p = path.join(ROOT, c.file);
    let html = '';
    let raw = '';
    try {
      raw = fs.readFileSync(p, 'utf8');
      html = render(raw);
    } catch (e) {
      html = '<p class="err">Файл не найден: ' + escapeHtml(c.file) + '</p>';
    }
    return { id: c.id, title: c.title, html, raw };
  });
}

const PAGE = `<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Черновик диссертации — веб-просмотр</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"></script>
<style>
  :root{--bg:#f5f6f8;--card:#ffffff;--ink:#1c1e21;--muted:#6b7280;--accent:#2563eb;--border:#e5e7eb;}
  *{box-sizing:border-box;}
  body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg);}
  .app{display:flex;min-height:100vh;}
  nav{width:280px;flex:0 0 280px;background:var(--card);border-right:1px solid var(--border);padding:16px;position:sticky;top:0;height:100vh;overflow:auto;}
  nav h1{font-size:15px;line-height:1.4;margin:0 0 14px;}
  nav a{display:block;padding:8px 10px;border-radius:8px;color:var(--ink);text-decoration:none;font-size:14px;margin-bottom:2px;}
  nav a:hover{background:var(--bg);}
  nav a.active{background:var(--accent);color:#fff;}
  main{flex:1;padding:32px 40px;max-width:860px;}
  article{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:32px 40px;}
  article h1{font-size:24px;line-height:1.3;margin:0 0 8px;}
  article h2{font-size:20px;margin:26px 0 10px;border-bottom:1px solid var(--border);padding-bottom:6px;}
  article h3{font-size:17px;margin:20px 0 8px;}
  article h4{font-size:15px;margin:16px 0 6px;}
  article p{line-height:1.65;margin:0 0 12px;}
  article li{line-height:1.6;margin-bottom:4px;}
  code{background:#f0f1f3;border-radius:4px;padding:1px 5px;font-size:13px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}
  pre{background:#f6f8fa;border:1px solid var(--border);border-radius:8px;padding:14px;overflow:auto;line-height:1.5;font-size:13px;}
  pre code{background:none;padding:0;}
  blockquote{border-left:3px solid var(--accent);margin:14px 0;padding:4px 16px;color:var(--muted);}
  hr{border:none;border-top:1px solid var(--border);margin:22px 0;}
  .math{background:#eef2ff;border-radius:4px;padding:1px 4px;font-size:1.02em;}
  .math-block{background:#eef2ff;border:1px solid #c7d2fe;border-radius:8px;padding:12px 16px;margin:12px 0;overflow-x:auto;}
  .err{color:#b91c1c;}
  .figure{margin:14px 0;}
  .figure img{max-width:100%;height:auto;border:1px solid var(--border);border-radius:8px;background:#fff;}
  pre.mermaid{background:none;border:none;padding:0;overflow:visible;}
  .mmd{overflow-x:auto;margin:14px 0;}
  .mmd svg{max-width:100%;height:auto;}
  /* strict black-and-white engineering style (Barroso-inspired) */
  .mmd svg{background:#fff;}
  .mmd .node rect,.mmd .node circle,.mmd .node polygon,.mmd .node path,
  .mmd .node ellipse,.mmd .node diamond{fill:#fff;stroke:#000;stroke-width:1.4px;}
  .mmd .node text,.mmd .edgeLabel text,.mmd .cluster text{fill:#000;}
  .mmd .edgePath .path{stroke:#000;stroke-width:1.4px;}
  .mmd .edgeLabel{background:#fff;}
  .mmd .cluster rect{fill:#fff;stroke:#000;stroke-width:1.4px;}
  .mmd .marker,.mmd .arrowheadPath{fill:#000;stroke:#000;}
  .mmd .node rect,.mmd .node polygon{filter:none;}
  a.cit{color:var(--accent);text-decoration:none;white-space:nowrap;}
  a.cit:hover{text-decoration:underline;}
  .cit-hl{background:#fef3c7;border-radius:3px;padding:1px 3px;transition:background 1s;}
  .status{position:fixed;bottom:14px;right:18px;background:var(--card);border:1px solid var(--border);border-radius:20px;padding:6px 14px;font-size:12px;color:var(--muted);box-shadow:0 2px 8px rgba(0,0,0,.06);}
  .status.live{color:#047857;border-color:#a7f3d0;background:#ecfdf5;}
</style>
</head>
<body>
<div class="app">
  <nav>
    <h1>О справедливом разделении эластичной ёмкости вычислительного кластера</h1>
    <div id="nav"></div>
  </nav>
  <main><article id="content"></article></main>
</div>
<div id="status" class="status">подключение…</div>
<script>
const state = { chapters: [], active: 'ch1' };
function renderNav() {
  document.getElementById('nav').innerHTML = state.chapters.map(c =>
    '<a href="#" data-id="' + c.id + '" class="' + (c.id === state.active ? 'active' : '') + '">' + c.title + '</a>'
  ).join('');
  document.querySelectorAll('nav a').forEach(a => a.addEventListener('click', e => {
    e.preventDefault();
    state.active = a.getAttribute('data-id');
    renderNav(); renderContent();
    history.replaceState(null, '', '#' + state.active);
  }));
}
function renderMath(root) {
  if (!window.katex) { /* KaTeX ещё не загрузился */ return; }
  root.querySelectorAll('.math, .math-block').forEach(el => {
    const latex = decodeURIComponent(el.getAttribute('data-raw') || '');
    try {
      katex.render(latex, el, { throwOnError: false, displayMode: el.classList.contains('math-block') });
    } catch (e) { /* оставить исходный текст */ }
  });
}
function renderMermaid(root) {
  if (!window.mermaid) { /* Mermaid ещё не загрузился */ return; }
  if (!window.__mermaidBW) {
    window.__mermaidBW = true;
    mermaid.initialize({
      startOnLoad: false,
      theme: 'base',
      securityLevel: 'loose',
      themeVariables: {
        lineColor: '#000000', edgeColor: '#000000',
        primaryColor: '#ffffff', primaryBorderColor: '#000000',
        primaryTextColor: '#000000',
        secondaryColor: '#ffffff', secondaryBorderColor: '#000000',
        secondaryTextColor: '#000000',
        tertiaryColor: '#ffffff', tertiaryBorderColor: '#000000',
        tertiaryTextColor: '#000000',
        clusterBkg: '#ffffff', clusterBorder: '#000000',
        clusterTextColor: '#000000',
        edgeLabelBackground: '#ffffff', nodeBorder: '#000000',
        nodeTextColor: '#000000', fontFamily: 'Georgia, serif',
      },
    });
  }
  root.querySelectorAll('pre.mermaid').forEach(el => {
    try {
      mermaid.render('mmd-' + (el._id = (el._id || 0) + 1), el.textContent).then(res => {
        const wrap = document.createElement('div');
        wrap.className = 'mmd';
        wrap.innerHTML = res.svg;
        el.replaceWith(wrap);
      });
    } catch (e) { /* оставить исходный текст диаграммы */ }
  });
}
function renderContent() {
  const c = state.chapters.find(x => x.id === state.active);
  const content = document.getElementById('content');
  content.innerHTML = c ? c.html : '';
  renderMath(content);
  renderMermaid(content);
  document.title = (c ? c.title + ' — ' : '') + 'Черновик диссертации';
  window.scrollTo(0, 0);
}
function onHash() {
  const id = location.hash.slice(1);
  if (id && state.chapters.some(c => c.id === id)) { state.active = id; renderNav(); renderContent(); }
}
async function refresh() {
  try {
    const res = await fetch('/chapters');
    const data = await res.json();
    const sig = JSON.stringify(data.map(c => c.html));
    if (sig !== state._sig) {
      state.chapters = data; state._sig = sig;
      renderNav(); renderContent();
      const st = document.getElementById('status');
      st.textContent = '● live · ' + new Date().toLocaleTimeString();
      st.classList.add('live');
    }
  } catch (e) { /* server briefly down */ }
}
window.addEventListener('hashchange', onHash);
function onCiteClick(e) {
  const a = e.target.closest('a.cit');
  if (!a) return;
  const n = a.getAttribute('data-cit');
  // Перейти к списку литературы (hash #refs) и подсветить строку N.
  if (location.hash !== '#refs') { location.hash = '#refs'; }
  const tryHl = () => {
    const refs = document.getElementById('content');
    const lines = refs.querySelectorAll('p');
    let target = null;
    lines.forEach(p => {
      const t = p.textContent.trim();
      if (/^N\s*\./.test(t) ? t.startsWith(n + '.') : t.indexOf(n + '.') === 0) target = p;
    });
    if (!target) {
      lines.forEach(p => { if (p.textContent.trim().indexOf(n + '.') === 0) target = p; });
    }
    if (target) { target.classList.add('cit-hl'); target.scrollIntoView({ behavior: 'smooth', block: 'center' }); }
  };
  setTimeout(tryHl, 50);
  setTimeout(tryHl, 300);
  e.preventDefault();
}
document.addEventListener('click', onCiteClick);
(function init() {
  refresh().then(() => onHash());
  // KaTeX грузится через defer; дождаться его и перерисовать математику.
  const waiter = setInterval(() => {
    if (window.katex) {
      clearInterval(waiter);
      renderContent();
    }
  }, 100);
  setInterval(refresh, 2000);
})();
</script>
</body>
</html>`;

const server = http.createServer((req, res) => {
  const url = req.url.split('?')[0];
  if (url === '/' || url === '/index.html') {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(PAGE);
    return;
  }
  if (url === '/chapters') {
    res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify(loadChapters()));
    return;
  }
  // static: SVG figures in draft/figures/
  if (url.startsWith('/figures/') && url.endsWith('.svg')) {
    const p = path.join(ROOT, 'figures', path.basename(url));
    fs.readFile(p, (err, data) => {
      if (err) {
        res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
        res.end('Not found');
        return;
      }
      res.writeHead(200, { 'Content-Type': 'image/svg+xml; charset=utf-8' });
      res.end(data);
    });
    return;
  }
  res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
  res.end('Not found');
});

server.listen(PORT, () => {
  console.log('Dissertation viewer: http://localhost:' + PORT);
  console.log('Watching ' + ROOT);
});