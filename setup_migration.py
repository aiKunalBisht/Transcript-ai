#!/usr/bin/env python3
"""
TranscriptAI v3.0 - Migration File Creator
Run from your project root: python setup_migration.py
Creates: static/style.css, templates/base.html,
         templates/index.html, templates/export.html
Updates: requirements.txt
NOTE: main.py is a separate download - copy it manually.
"""
import os, re

os.makedirs("static",    exist_ok=True)
os.makedirs("templates", exist_ok=True)
print("ok  directories: static/ and templates/")


# ── static/style.css ─────────────────────────────────────────────────────────
CSS = """:root {
  --accent:#E8829A;--green:#2D9E6B;--ink:#2C1810;--ink-mid:#5A4030;
  --ink-s:#8B7060;--glass:rgba(60,36,22,0.03);--glass-b:rgba(60,36,22,0.10);
  --sidebar-w:272px;--r:14px;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter','Noto Sans JP',sans-serif;background:#FDF8F5;color:var(--ink);min-height:100vh}
.app-shell{display:flex;min-height:100vh}
.sidebar{width:var(--sidebar-w);background:#FEFAF8;border-right:1px solid var(--glass-b);
  padding:24px 18px;display:flex;flex-direction:column;gap:22px;
  position:sticky;top:0;height:100vh;overflow-y:auto;flex-shrink:0}
.logo{display:flex;align-items:center;gap:10px;padding-bottom:20px;border-bottom:1px solid var(--glass-b)}
.logo-mark{width:36px;height:36px;background:linear-gradient(135deg,var(--accent) 0%,#F4A07A 100%);
  border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.1rem;flex-shrink:0}
.logo-name{font-weight:800;font-size:1rem;letter-spacing:-0.02em;color:var(--ink)}
.logo-ver{font-size:0.6rem;color:var(--ink-s);letter-spacing:0.08em;text-transform:uppercase;margin-top:1px}
.sidebar-nav{display:flex;flex-direction:column;gap:2px}
.nav-link{display:flex;align-items:center;gap:9px;padding:9px 12px;border-radius:10px;
  font-size:0.83rem;font-weight:500;color:var(--ink-mid);text-decoration:none;
  transition:background 0.18s,color 0.18s}
.nav-link:hover{background:rgba(232,130,154,0.08);color:var(--accent)}
.nav-link.active{background:rgba(232,130,154,0.13);color:var(--accent);font-weight:600}
.section-label{font-size:0.58rem;font-weight:700;color:var(--ink-s);
  letter-spacing:0.14em;text-transform:uppercase;margin-bottom:8px}
.field-row{display:flex;flex-direction:column;gap:5px;margin-bottom:12px}
.field-label{font-size:0.72rem;font-weight:600;color:var(--ink-mid)}
.field-select{width:100%;padding:8px 11px;background:var(--glass);border:1px solid var(--glass-b);
  border-radius:8px;font-size:0.82rem;color:var(--ink);font-family:inherit;outline:none;
  cursor:pointer;transition:border-color 0.2s}
.field-select:focus{border-color:var(--accent)}
.toggle-row{display:flex;justify-content:space-between;align-items:center;padding:7px 0}
.toggle-label{font-size:0.79rem;color:var(--ink-mid);font-weight:500}
.toggle{position:relative;width:38px;height:22px;background:var(--glass-b);
  border-radius:999px;cursor:pointer;transition:background 0.2s;flex-shrink:0}
.toggle input{display:none}
.toggle:has(input:checked){background:var(--green)}
.toggle input:checked + .toggle-thumb{transform:translateX(16px)}
.toggle-thumb{position:absolute;top:3px;left:3px;width:16px;height:16px;background:#fff;
  border-radius:50%;transition:transform 0.2s;box-shadow:0 1px 4px rgba(0,0,0,0.15)}
.main-content{flex:1;padding:36px 44px;max-width:860px;overflow-y:auto}
.page-title{font-size:1.45rem;font-weight:800;letter-spacing:-0.02em;color:var(--ink);margin-bottom:6px}
.page-sub{font-size:0.84rem;color:var(--ink-s);line-height:1.65;margin-bottom:28px}
.upload-zone{border:2px dashed var(--glass-b);border-radius:var(--r);padding:38px 28px;
  text-align:center;cursor:pointer;transition:border-color 0.25s,background 0.25s;
  background:var(--glass);margin-bottom:16px;position:relative}
.upload-zone:hover,.upload-zone.drag-over{border-color:var(--accent);background:rgba(232,130,154,0.04)}
.upload-zone input[type="file"]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.upload-icon{font-size:2rem;margin-bottom:8px}
.upload-label{font-size:0.88rem;font-weight:600;color:var(--ink-mid)}
.upload-hint{font-size:0.71rem;color:var(--ink-s);margin-top:4px}
.or-divider{display:flex;align-items:center;gap:12px;margin:18px 0;color:var(--ink-s);font-size:0.74rem}
.or-divider::before,.or-divider::after{content:'';flex:1;height:1px;background:var(--glass-b)}
.transcript-input{width:100%;min-height:152px;padding:15px 17px;background:var(--glass);
  border:1px solid var(--glass-b);border-radius:var(--r);font-size:0.86rem;color:var(--ink);
  font-family:inherit;resize:vertical;outline:none;transition:border-color 0.2s;line-height:1.7}
.transcript-input:focus{border-color:var(--accent)}
.btn-primary{display:inline-flex;align-items:center;justify-content:center;gap:8px;
  padding:11px 22px;background:var(--accent);color:#fff;border:none;border-radius:10px;
  font-size:0.87rem;font-weight:600;cursor:pointer;font-family:inherit;
  transition:opacity 0.2s,transform 0.15s;text-decoration:none}
.btn-primary:hover{opacity:0.9;transform:translateY(-1px)}
.btn-primary:active{transform:none}
.btn-primary:disabled{opacity:0.45;cursor:not-allowed;transform:none}
.btn-secondary{display:inline-flex;align-items:center;justify-content:center;gap:8px;
  padding:9px 18px;background:var(--glass);color:var(--ink-mid);border:1px solid var(--glass-b);
  border-radius:10px;font-size:0.84rem;font-weight:500;cursor:pointer;font-family:inherit;
  transition:border-color 0.2s,background 0.2s;text-decoration:none}
.btn-secondary:hover{border-color:rgba(232,130,154,0.3);background:rgba(232,130,154,0.06);color:var(--accent)}
.btn-secondary:disabled{opacity:0.45;cursor:not-allowed}
.htmx-indicator{display:none}
.htmx-request .htmx-indicator{display:flex}
.htmx-request .hide-on-load{display:none !important}
.loading-bar{display:flex;align-items:center;gap:12px;padding:18px 20px;margin-top:16px;
  background:var(--glass);border:1px solid var(--glass-b);border-radius:12px;
  font-size:0.84rem;color:var(--ink-s)}
.spinner{width:18px;height:18px;border:2px solid var(--glass-b);border-top-color:var(--accent);
  border-radius:50%;animation:spin 0.7s linear infinite;flex-shrink:0}
@keyframes spin{to{transform:rotate(360deg)}}
#results{min-height:20px;margin-top:8px}
.notice{background:rgba(232,130,154,0.06);border:1px solid rgba(232,130,154,0.18);
  border-radius:12px;padding:18px 22px;font-size:0.83rem;color:var(--ink-mid);
  line-height:1.6;margin-bottom:22px}
.export-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin-top:8px}
@media(max-width:600px){.export-grid{grid-template-columns:1fr}}
.export-card{background:var(--glass);border:1px solid var(--glass-b);border-radius:var(--r);
  padding:22px 20px;transition:border-color 0.2s,transform 0.2s}
.export-card:hover{border-color:rgba(232,130,154,0.3);transform:translateY(-2px)}
.export-card-icon{font-size:1.5rem;margin-bottom:8px}
.export-card-title{font-weight:700;font-size:0.9rem;color:var(--ink);margin-bottom:4px}
.export-card-desc{font-size:0.74rem;color:var(--ink-s);margin-bottom:14px;line-height:1.55}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:0.6rem;
  font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:10px}
.badge-primary{background:rgba(232,130,154,0.12);color:var(--accent);border:1px solid rgba(232,130,154,0.25)}
.badge-standard{background:rgba(45,158,107,0.1);color:var(--green);border:1px solid rgba(45,158,107,0.2)}
.badge-dev{background:rgba(90,125,107,0.1);color:#5A7D6B;border:1px solid rgba(90,125,107,0.2)}
.dev-footer{margin-top:auto;padding-top:16px;border-top:1px solid var(--glass-b);
  font-size:0.69rem;color:var(--ink-s);text-align:center;line-height:1.8}
.dev-footer a{color:var(--accent);text-decoration:none}
@media(max-width:768px){
  .app-shell{flex-direction:column}
  .sidebar{width:100%;height:auto;position:relative;padding:14px 16px;
    flex-direction:row;flex-wrap:wrap;align-items:center;gap:12px}
  .logo-ver,.section-label,.field-row,.toggle-row{display:none}
  .sidebar-nav{flex-direction:row}
  .main-content{padding:20px 16px}
}
"""
with open("static/style.css", "w", encoding="utf-8") as f:
    f.write(CSS)
print("ok  static/style.css")


# ── templates/base.html ───────────────────────────────────────────────────────
BASE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{% block title %}TranscriptAI{% endblock %}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+JP:wght@400;500;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="{{ url_for('static', path='style.css') }}">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/htmx/1.9.12/htmx.min.js" defer></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/alpinejs/3.14.1/cdn.min.js" defer></script>
  <style>[x-cloak]{display:none!important}</style>
</head>
<body>
<div class="app-shell" x-data="tai()">
  <aside class="sidebar">
    <div class="logo">
      <div class="logo-mark">\U0001f399</div>
      <div>
        <div class="logo-name">TranscriptAI</div>
        <div class="logo-ver">v3.0 \u00b7 APPI Compliant</div>
      </div>
    </div>
    <nav class="sidebar-nav">
      <a href="/"       class="nav-link {% if active_page == 'index'  %}active{% endif %}">\U0001f4dd Analyze</a>
      <a href="/export" class="nav-link {% if active_page == 'export' %}active{% endif %}">\U0001f4e4 Export</a>
      <a href="/docs"   class="nav-link" target="_blank" rel="noopener">\U0001f50c API</a>
    </nav>
    <div>
      <div class="section-label">Settings</div>
      <div class="field-row">
        <label class="field-label" for="lang-select">Language</label>
        <select id="lang-select" class="field-select" x-model="language">
          <option value="">Auto-detect</option>
          <option value="en">English</option>
          <option value="ja">Japanese (\u65e5\u672c\u8a9e)</option>
          <option value="hi">Hindi (\u0939\u093f\u0928\u094d\u0926\u0940)</option>
          <option value="mixed">Mixed / Multilingual</option>
        </select>
      </div>
      <div class="toggle-row">
        <span class="toggle-label">\U0001f512 PII masking (APPI)</span>
        <label class="toggle">
          <input type="checkbox" x-model="maskPii" checked>
          <span class="toggle-thumb"></span>
        </label>
      </div>
    </div>
    <div class="dev-footer">
      Built by <strong>Kunal Bisht</strong><br>
      <a href="https://github.com/aiKunalBisht/Transcript-ai" target="_blank" rel="noopener">GitHub</a>
      \u00b7 <a href="/health" target="_blank">Health</a>
      \u00b7 <a href="/docs" target="_blank">API</a>
    </div>
  </aside>
  <main class="main-content">
    {% block content %}{% endblock %}
  </main>
</div>
<script>
function tai() {
  return {
    language: '', maskPii: true, lastResult: null,
    captureResult(s) { try { this.lastResult = JSON.parse(s); } catch(e) {} },
    hasResult() { return this.lastResult !== null; },
    async exportAs(format) {
      if (!this.lastResult) { alert('Run an analysis first.'); return; }
      const ext = {pptx:'pptx',markdown:'md',json:'json'}[format] ?? format;
      const btn = document.getElementById('btn-'+format);
      if (btn) btn.disabled = true;
      try {
        const r = await fetch('/export/'+format, {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({result: this.lastResult})
        });
        if (!r.ok) { alert('Export failed: '+(await r.text())); return; }
        const blob = await r.blob();
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href = url; a.download = 'transcript_'+Date.now()+'.'+ext;
        document.body.appendChild(a); a.click(); a.remove();
        URL.revokeObjectURL(url);
      } finally { if (btn) btn.disabled = false; }
    },
    async exportGijiroku() {
      if (!this.lastResult) { alert('Run an analysis first.'); return; }
      const r = await fetch('/export/gijiroku', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({result: this.lastResult})
      });
      if (!r.ok) { alert('Gijiroku failed.'); return; }
      const d    = await r.json();
      const blob = new Blob([d.gijiroku], {type:'text/plain;charset=utf-8'});
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href = url; a.download = 'gijiroku_'+Date.now()+'.txt';
      document.body.appendChild(a); a.click(); a.remove();
      URL.revokeObjectURL(url);
    }
  };
}
</script>
{% block extra_scripts %}{% endblock %}
</body>
</html>
"""
with open("templates/base.html", "w", encoding="utf-8") as f:
    f.write(BASE)
print("ok  templates/base.html")


# ── templates/index.html ─────────────────────────────────────────────────────
INDEX = """{% extends "base.html" %}
{% set active_page = "index" %}
{% block title %}TranscriptAI \u2014 Analyze{% endblock %}
{% block content %}
<div class="page-title">Meeting Intelligence</div>
<div class="page-sub">
  Upload an audio file or paste a transcript.
  Supports English, Japanese (\u8b70\u4e8b\u9332), Hindi, and mixed-language conversations.
</div>

<form hx-post="/upload" hx-target="#results" hx-swap="innerHTML"
      hx-encoding="multipart/form-data" hx-indicator="#loading"
      hx-on::after-settle="onResultReady()">
  <input type="hidden" name="language" :value="language">
  <input type="hidden" name="mask_pii" :value="maskPii">
  <div class="upload-zone" x-data="{fname:''}"
       @dragover.prevent
       @drop.prevent="fname=$event.dataTransfer.files[0]?.name;$el.querySelector('input[type=file]').files=$event.dataTransfer.files">
    <input type="file" name="file" accept=".mp3,.wav,.m4a,.mp4,.ogg,.flac,.webm"
           @change="fname=$event.target.files[0]?.name">
    <div class="upload-icon">\U0001f3a4</div>
    <div class="upload-label" x-text="fname||'Drop audio file or click to browse'"></div>
    <div class="upload-hint">MP3 \u00b7 WAV \u00b7 M4A \u00b7 FLAC \u00b7 OGG \u00b7 WebM \u00b7 max 50 MB</div>
  </div>
  <button type="submit" class="btn-primary" style="width:100%">
    <span class="hide-on-load">\u26a1 Transcribe &amp; Analyze</span>
    <span class="htmx-indicator" style="display:none">Transcribing\u2026</span>
  </button>
</form>

<div class="or-divider">or paste transcript text</div>

<form hx-post="/analyze-text" hx-target="#results" hx-swap="innerHTML"
      hx-indicator="#loading" hx-on::after-settle="onResultReady()">
  <input type="hidden" name="language" :value="language">
  <input type="hidden" name="mask_pii" :value="maskPii">
  <textarea class="transcript-input" name="transcript"
    placeholder="Paste meeting transcript here\u2026&#10;&#10;Tanaka: \u304a\u306f\u3088\u3046\u3054\u3056\u3044\u307e\u3059\u3002&#10;Sato: Good morning. Let's begin the Q3 review."></textarea>
  <button type="submit" class="btn-primary" style="margin-top:12px">
    \u26a1 Analyze Transcript
  </button>
</form>

<div id="loading" class="loading-bar htmx-indicator">
  <div class="spinner"></div>
  Analyzing\u2026 15\u201330 seconds depending on length
</div>

<div id="results"></div>

<div style="margin-top:18px" x-show="hasResult()" x-cloak>
  <a href="/export" class="btn-secondary">\U0001f4e4 Export results \u2192</a>
</div>
{% endblock %}

{% block extra_scripts %}
<script>
function onResultReady() {
  var el = document.getElementById('tai-result-data');
  if (!el) return;
  var root = document.querySelector('[x-data]');
  if (root && root._x_dataStack && root._x_dataStack[0])
    root._x_dataStack[0].captureResult(el.textContent.trim());
  document.getElementById('results').scrollIntoView({behavior:'smooth',block:'start'});
}
</script>
{% endblock %}
"""
with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write(INDEX)
print("ok  templates/index.html")


# ── templates/export.html ─────────────────────────────────────────────────────
EXPORT = """{% extends "base.html" %}
{% set active_page = "export" %}
{% block title %}TranscriptAI \u2014 Export{% endblock %}
{% block content %}
<div class="page-title">Export Documents</div>
<div class="page-sub">Download your meeting intelligence. Run an analysis first.</div>

<div class="notice" x-show="!hasResult()" x-cloak>
  \u2139\ufe0f No analysis loaded.
  <a href="/" style="color:var(--accent);font-weight:600;text-decoration:none">Run an analysis \u2192</a>
</div>

<div class="export-grid">
  {% if pptx_available %}
  <div class="export-card">
    <div class="export-card-icon">\U0001f4ca</div>
    <div class="badge badge-primary">Primary export</div>
    <div class="export-card-title">Slide Deck (.pptx)</div>
    <div class="export-card-desc">Midnight Executive layout \u2014 summary, actions, speakers, risk flags.</div>
    <button id="btn-pptx" class="btn-primary" style="width:100%"
            @click="exportAs('pptx')" :disabled="!hasResult()">\u2193 Download PPTX</button>
  </div>
  {% endif %}

  {% if gijiroku_available %}
  <div class="export-card">
    <div class="export-card-icon">\U0001f3ef</div>
    <div class="badge badge-primary">Primary export</div>
    <div class="export-card-title">\u8b70\u4e8b\u9332 (Japanese Minutes)</div>
    <div class="export-card-desc">Structured Japanese business minutes \u2014 Fujitsu / NTT DATA enterprise format.</div>
    <button class="btn-primary" style="width:100%"
            @click="exportGijiroku()" :disabled="!hasResult()">\u2193 Download \u8b70\u4e8b\u9332</button>
  </div>
  {% endif %}

  <div class="export-card">
    <div class="export-card-icon">\U0001f4dd</div>
    <div class="badge badge-standard">Standard</div>
    <div class="export-card-title">Markdown Notes (.md)</div>
    <div class="export-card-desc">Summary, actions, sentiment, speakers. Works in Notion, Obsidian, GitHub.</div>
    <button id="btn-markdown" class="btn-secondary" style="width:100%"
            @click="exportAs('markdown')" :disabled="!hasResult()">\u2193 Download Markdown</button>
  </div>

  <div class="export-card">
    <div class="export-card-icon">\U0001f527</div>
    <div class="badge badge-dev">Developer</div>
    <div class="export-card-title">Raw Analysis (.json)</div>
    <div class="export-card-desc">Full output \u2014 confidence scores, soft rejections, PII report, flags.</div>
    <button id="btn-json" class="btn-secondary" style="width:100%"
            @click="exportAs('json')" :disabled="!hasResult()">\u2193 Download JSON</button>
  </div>
</div>
{% endblock %}
"""
with open("templates/export.html", "w", encoding="utf-8") as f:
    f.write(EXPORT)
print("ok  templates/export.html")


# ── requirements.txt ─────────────────────────────────────────────────────────
import re
with open("requirements.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

filtered = [l for l in lines if not re.match(r"^\s*streamlit", l, re.IGNORECASE)]
existing = "".join(filtered)

new_deps = [
    "fastapi>=0.111.0\n",
    "uvicorn[standard]>=0.29.0\n",
    "python-multipart>=0.0.9\n",
    "jinja2>=3.1.4\n",
    "aiofiles>=23.2.1\n",
]
for dep in new_deps:
    pkg = dep.split(">=")[0].split("[")[0].strip().lower()
    if pkg not in existing.lower():
        filtered.append(dep)

with open("requirements.txt", "w", encoding="utf-8") as f:
    f.writelines(filtered)
print("ok  requirements.txt  (streamlit removed, fastapi deps added)")

print()
print("=" * 52)
print("All done. Now:")
print("  1. Copy main.py into this folder (separate download)")
print("  2. pip install -r requirements.txt")
print("  3. uvicorn main:app --reload --port 7860")
print("=" * 52)
