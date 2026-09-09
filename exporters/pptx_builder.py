# exporters/pptx_builder.py
# Converts a SlidePlan dict into a .pptx byte stream via pptxgenjs (Node.js).
#
# Called by : main.py /export/pptx route
#             SlideArchitectAgent.plan() feeds the input
#
# ── Data flow ─────────────────────────────────────────────────────────────────
# build_pptx(plan: dict) → bytes
#   1. Serialize plan → tmp/plan.json
#   2. Write Node.js generator script → tmp/gen.js
#      (JS uses __PLAN_PATH__ / __OUT_PATH__ placeholders, replaced in Python)
#   3. subprocess: node tmp/gen.js
#   4. Read tmp/output.pptx → return bytes
#   5. Cleanup temp dir
#
# ── Algorithm ─────────────────────────────────────────────────────────────────
# O(S × C)  S = slides, C = content items per slide (table rows, bar entries)
# Subprocess adds ~1–3 s fixed overhead for Node.js startup + pptxgenjs init.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import Optional

# ── pptxgenjs Node.js generator script ───────────────────────────────────────
# Raw string — no Python f-string escaping needed.
# __PLAN_PATH__ and __OUT_PATH__ are replaced at call time.
# Design: white content slides, dark cover/closing, clean typography.
# Follows skill rules: no # in hex, isTextBox:true everywhere,
#   bullet:true (never literal •), no shared option objects,
#   shadow offset ≥ 0, no decorative stripes.
_JS = r"""
'use strict';
const pptxgen = require('pptxgenjs');
const fs      = require('fs');

const plan = JSON.parse(fs.readFileSync('__PLAN_PATH__', 'utf8'));
const prs  = new pptxgen();
prs.layout  = 'LAYOUT_WIDE';   // 13.3" × 7.5"
prs.title   = plan.meeting_title || 'Meeting Analysis';
prs.company = 'TranscriptAI';
prs.subject = 'Meeting Intelligence Report';

// ── Palette ───────────────────────────────────────────────────────────────────
var COV_BG   = '1E1428';
var CLO_BG   = '15102A';
var ACCENT   = 'C2566A';
var WHITE    = 'FFFFFF';
var TXT_DARK = '1A1228';
var TXT_MID  = '5C4B5A';
var TXT_SOFT = '9080A0';
var SLD_BG   = 'FFFFFF';
var BORDER   = 'E4DCE8';

// Urgency row colours — background and text
var URG_BG = {
  immediate: 'FFF0F2', next_day:  'FFF4EB',
  this_week: 'FEFCE8', standard:  'EFFAF4', unknown: 'F8F6F8'
};
var URG_TXT = {
  immediate: 'B91C1C', next_day:  'C05700',
  this_week: 'A16207', standard:  '166534', unknown: '6B7280'
};

// Risk badge colour
var RISK_CLR = {
  CRITICAL: '7C3AED', HIGH: 'C0392B', MEDIUM: 'D97706',
  LOW: 'BE4060', MINIMAL: 'A87868', NONE: '2D7A55'
};

// Sentiment score badge colour
var SCORE_CLR = {
  POSITIVE: '166534', NEUTRAL: '374151', NEGATIVE: 'B91C1C',
  CONCERNED: 'C05700', DEFENSIVE: 'C05700', PROFESSIONAL: '1E40AF',
  FORMAL: '1E40AF', ANXIOUS: 'D97706', TENSE: 'C0392B'
};

// Bar colours for speaker chart
var BAR_CLRS = ['C2566A','7D4E8A','1E6B9A','2D7A55','D97706','8B5E3C','4A5568','9B2C2C'];

// ── Helpers ───────────────────────────────────────────────────────────────────

function hdr(slide, title, sub) {
  slide.addText(title, {
    x:0.55, y:0.22, w:12.2, h:0.65,
    fontSize:26, bold:true, color:TXT_DARK,
    fontFace:'Calibri', align:'left',
    isTextBox:true, margin:0
  });
  if (sub) {
    slide.addText(sub, {
      x:0.55, y:0.85, w:12.2, h:0.28,
      fontSize:9, color:TXT_SOFT,
      fontFace:'Calibri', align:'left',
      isTextBox:true, margin:0
    });
  }
}

function ftr(slide) {
  slide.addText('TranscriptAI  \u00B7  Meeting Intelligence', {
    x:0.55, y:7.22, w:12.2, h:0.22,
    fontSize:7.5, color:TXT_SOFT, align:'right',
    fontFace:'Calibri', isTextBox:true, margin:0
  });
}

// ── Cover ─────────────────────────────────────────────────────────────────────
function buildCover(slide, s) {
  slide.background = { color: COV_BG };

  slide.addText('TRANSCRIPTAI  \u00B7  MEETING INTELLIGENCE', {
    x:0.8, y:0.7, w:11.7, h:0.42,
    fontSize:9, bold:true, color:ACCENT,
    fontFace:'Calibri', charSpacing:2,
    isTextBox:true, margin:0
  });

  var tlen = (s.title || '').length;
  var tfs  = tlen > 65 ? 27 : tlen > 45 ? 32 : 38;
  slide.addText(s.title || 'Meeting Analysis', {
    x:0.8, y:1.5, w:11.7, h:2.3,
    fontSize:tfs, bold:true, color:WHITE,
    fontFace:'Calibri', align:'left', valign:'top',
    isTextBox:true, margin:0
  });

  if (s.subtitle) {
    slide.addText(s.subtitle, {
      x:0.8, y:3.9, w:11.7, h:0.55,
      fontSize:14, color:'A090B8',
      fontFace:'Calibri', isTextBox:true, margin:0
    });
  }

  slide.addText(s.date || '', {
    x:0.8, y:6.6, w:5.5, h:0.38,
    fontSize:10, color:'70607A',
    fontFace:'Calibri', isTextBox:true, margin:0
  });

  // Language is already in the subtitle line — no separate badge needed.
}

// ── Overview ──────────────────────────────────────────────────────────────────
function buildOverview(slide, s) {
  slide.background = { color: SLD_BG };
  hdr(slide, s.title || 'Meeting Overview', 'Key points from this meeting');
  ftr(slide);

  var bullets = s.bullets || [];
  if (!bullets.length) return;

  var runs = bullets.map(function(b, i) {
    return {
      text: b,
      options: {
        bullet: true, fontSize: 13.5, color: TXT_DARK,
        fontFace: 'Calibri', paraSpaceAfter: 10,
        breakLine: i < bullets.length - 1
      }
    };
  });

  slide.addText(runs, {
    x:0.65, y:1.25, w:12.0, h:5.8,
    valign:'top', isTextBox:true, margin:0
  });
}

// ── Action Items ──────────────────────────────────────────────────────────────
function buildActionItems(slide, s) {
  slide.background = { color: SLD_BG };
  var items = (s.items || []).slice(0, 7);
  var sub   = items.length + ' commitment' + (items.length !== 1 ? 's' : '') + ' tracked';
  hdr(slide, s.title || 'Action Items', sub);
  ftr(slide);

  if (!items.length) {
    slide.addText('No action items extracted from this meeting.', {
      x:0.6, y:3.0, w:12.0, h:0.5,
      fontSize:13, color:TXT_SOFT, align:'center',
      fontFace:'Calibri', isTextBox:true, margin:0
    });
    return;
  }

  var hFill = { color: 'EEE4F0' };
  var headerRow = [
    { text:'TASK / ACTION',   options:{ bold:true, fontSize:8.5, color:TXT_MID, fill:hFill, align:'left',   fontFace:'Calibri' } },
    { text:'OWNER',           options:{ bold:true, fontSize:8.5, color:TXT_MID, fill:hFill, align:'center', fontFace:'Calibri' } },
    { text:'DEADLINE',        options:{ bold:true, fontSize:8.5, color:TXT_MID, fill:hFill, align:'center', fontFace:'Calibri' } },
    { text:'URGENCY',         options:{ bold:true, fontSize:8.5, color:TXT_MID, fill:hFill, align:'center', fontFace:'Calibri' } }
  ];

  var rows = [headerRow];
  items.forEach(function(item) {
    var tier   = item.urgency_tier || 'unknown';
    var bgClr  = URG_BG[tier]  || URG_BG.unknown;
    var fgClr  = URG_TXT[tier] || URG_TXT.unknown;
    var urgLbl = tier.replace('_', ' ').toUpperCase();
    var taskTxt = (item.flagged ? '\u26A0 ' : '') + (item.task || '\u2014');
    var taskClr = item.flagged ? 'B91C1C' : TXT_DARK;
    var bg      = { color: bgClr };

    rows.push([
      { text:taskTxt,               options:{ fontSize:10, color:taskClr, fill:bg, align:'left',   fontFace:'Calibri' } },
      { text:item.owner   || 'TBD', options:{ fontSize:10, color:TXT_DARK, fill:bg, align:'center', fontFace:'Calibri' } },
      { text:item.deadline|| 'N/A', options:{ fontSize:10, color:TXT_MID,  fill:bg, align:'center', fontFace:'Calibri' } },
      { text:urgLbl,                options:{ fontSize:9,  color:fgClr,    fill:bg, align:'center', fontFace:'Calibri', bold:true } }
    ]);
  });

  slide.addTable(rows, {
    x:0.4, y:1.2, w:12.4,
    colW:[5.9, 2.1, 2.4, 2.0],
    rowH:0.65,
    border:{ pt:0.5, color:BORDER }
  });
}

// ── Speakers ──────────────────────────────────────────────────────────────────
function buildSpeakers(slide, s) {
  slide.background = { color: SLD_BG };
  var speakers = s.speakers || [];
  var sub = speakers.length + ' participant' + (speakers.length !== 1 ? 's' : '');
  hdr(slide, s.title || 'Speaker Breakdown', sub);
  ftr(slide);

  if (!speakers.length) return;

  var maxPct  = Math.max.apply(null, speakers.map(function(sp){ return sp.pct || 0; }));
  if (!maxPct) maxPct = 1;
  var maxBarW = 9.0;
  var startY  = 1.35;
  var avail   = 7.0 - startY - 0.4;
  var rowH    = avail / speakers.length;

  speakers.forEach(function(sp, i) {
    var y    = startY + i * rowH;
    var pct  = sp.pct || 0;
    var barW = Math.max((pct / maxPct) * maxBarW, 0.06);
    var barH = Math.min(rowH * 0.44, 0.38);
    var clr  = BAR_CLRS[i % BAR_CLRS.length];
    var barY = y + (rowH - barH) / 2;

    // Speaker name
    slide.addText(sp.name || 'Unknown', {
      x:0.4, y:barY, w:2.55, h:barH,
      fontSize:11, bold:true, color:TXT_DARK,
      fontFace:'Calibri', align:'right', valign:'middle',
      isTextBox:true, margin:0
    });

    // Track (background bar)
    slide.addShape('rect', {
      x:3.1, y:barY, w:maxBarW, h:barH,
      fill:{ color:'EDE4F0' }, line:{ type:'none' }
    });

    // Fill bar
    slide.addShape('rect', {
      x:3.1, y:barY, w:barW, h:barH,
      fill:{ color:clr }, line:{ type:'none' }
    });

    // Percentage label (right of bar)
    slide.addText(pct + '%', {
      x:3.15 + barW, y:barY, w:0.85, h:barH,
      fontSize:10, bold:true, color:clr,
      fontFace:'Calibri', align:'left', valign:'middle',
      isTextBox:true, margin:0
    });

    // Tone (below name)
    if (sp.tone) {
      slide.addText(sp.tone, {
        x:0.4, y:barY + barH + 0.03, w:2.55, h:0.22,
        fontSize:8, color:TXT_SOFT, italic:true,
        fontFace:'Calibri', align:'right',
        isTextBox:true, margin:0
      });
    }
  });
}

// ── Sentiment ─────────────────────────────────────────────────────────────────
function buildSentiment(slide, s) {
  slide.background = { color: SLD_BG };
  var entries = (s.entries || []).slice(0, 6);
  var sub = entries.length + ' speaker' + (entries.length !== 1 ? 's' : '') + ' analyzed';
  hdr(slide, s.title || 'Communication Sentiment', sub);
  ftr(slide);

  if (!entries.length) return;

  var cols   = entries.length <= 2 ? entries.length : 3;
  var cardW  = cols === 1 ? 5.0 : cols === 2 ? 5.6 : 3.8;
  var cardH  = 2.1;
  var gapX   = 0.3;
  var totalW = cols * cardW + (cols - 1) * gapX;
  var startX = (13.3 - totalW) / 2;
  var startY = 1.3;

  entries.forEach(function(e, i) {
    var col    = i % cols;
    var row    = Math.floor(i / cols);
    var x      = startX + col * (cardW + gapX);
    var y      = startY + row * (cardH + 0.28);
    var scClr  = SCORE_CLR[e.score] || '374151';
    var bdgW   = 1.55;
    var bdgX   = x + (cardW - bdgW) / 2;

    // Card background
    slide.addShape('rect', {
      x:x, y:y, w:cardW, h:cardH,
      fill:{ color:'FAFAFA' },
      line:{ pt:0.75, color:BORDER },
      shadow:{ type:'outer', color:'000000', blur:6, offset:3, angle:90, opacity:0.06 }
    });

    // Speaker name
    slide.addText(e.speaker || 'Unknown', {
      x:x+0.15, y:y+0.18, w:cardW-0.3, h:0.52,
      fontSize:13, bold:true, color:TXT_DARK,
      fontFace:'Calibri', align:'center',
      isTextBox:true, margin:0
    });

    // Score badge (coloured rectangle)
    slide.addShape('rect', {
      x:bdgX, y:y+0.78, w:bdgW, h:0.4,
      fill:{ color:scClr }, line:{ type:'none' }
    });
    slide.addText(e.score || 'NEUTRAL', {
      x:bdgX, y:y+0.78, w:bdgW, h:0.4,
      fontSize:9.5, bold:true, color:WHITE,
      fontFace:'Calibri', align:'center', valign:'middle',
      isTextBox:true, margin:0
    });

    // Note
    if (e.note) {
      slide.addText(e.note, {
        x:x+0.12, y:y+1.28, w:cardW-0.24, h:0.72,
        fontSize:8.5, color:TXT_SOFT,
        fontFace:'Calibri', align:'center', valign:'top',
        isTextBox:true, margin:0
      });
    }
  });
}

// ── Risk ──────────────────────────────────────────────────────────────────────
function buildRisk(slide, s) {
  slide.background = { color: SLD_BG };
  var rClr = RISK_CLR[s.risk_level] || RISK_CLR.NONE;
  var nsig = s.total_signals || 0;
  var sub  = nsig + ' signal' + (nsig !== 1 ? 's' : '') + ' detected';
  hdr(slide, s.title || 'Risk Assessment', sub);
  ftr(slide);

  // Risk level badge
  slide.addShape('rect', {
    x:0.5, y:1.3, w:2.6, h:1.15,
    fill:{ color:rClr }, line:{ type:'none' }
  });
  slide.addText(s.risk_level || 'UNKNOWN', {
    x:0.5, y:1.3, w:2.6, h:0.75,
    fontSize:24, bold:true, color:WHITE,
    fontFace:'Calibri', align:'center', valign:'middle',
    isTextBox:true, margin:0
  });
  slide.addText('Risk Level', {
    x:0.5, y:2.05, w:2.6, h:0.4,
    fontSize:9, color:WHITE, align:'center',
    fontFace:'Calibri', isTextBox:true, margin:0
  });

  // Signals list
  slide.addText('Detected signals:', {
    x:3.45, y:1.3, w:9.35, h:0.38,
    fontSize:10.5, bold:true, color:TXT_DARK,
    fontFace:'Calibri', isTextBox:true, margin:0
  });

  var signals = (s.signals || []).slice(0, 5);
  signals.forEach(function(sig, i) {
    var phrase  = typeof sig === 'string' ? sig : (sig.phrase || sig.english || '(signal)');
    var speaker = typeof sig === 'object'  ? (sig.speaker || '') : '';
    var txt     = phrase + (speaker ? '  \u2014  ' + speaker : '');
    slide.addText(txt, {
      x:3.45, y:1.75 + i * 0.48, w:9.35, h:0.42,
      fontSize:11, color:rClr, fontFace:'Calibri',
      isTextBox:true, margin:0
    });
  });

  // Cultural note
  if (s.cultural_note) {
    slide.addText(s.cultural_note, {
      x:0.5, y:5.55, w:12.3, h:1.45,
      fontSize:9.5, color:TXT_SOFT, italic:true,
      fontFace:'Calibri', valign:'top',
      isTextBox:true, margin:0
    });
  }
}

// ── Closing ───────────────────────────────────────────────────────────────────
function buildClosing(slide, s) {
  slide.background = { color: CLO_BG };

  slide.addText(s.title || 'Key Takeaways', {
    x:0.8, y:0.85, w:11.7, h:0.78,
    fontSize:30, bold:true, color:WHITE,
    fontFace:'Calibri', isTextBox:true, margin:0
  });

  var items = s.takeaways || [];
  if (items.length) {
    var runs = items.map(function(t, i) {
      return {
        text: '\u2713   ' + t,
        options: {
          fontSize:13.5, color:'D0C0CE', fontFace:'Calibri',
          paraSpaceAfter:16,
          breakLine: i < items.length - 1
        }
      };
    });
    slide.addText(runs, {
      x:0.8, y:2.05, w:11.7, h:4.8,
      valign:'top', isTextBox:true, margin:0
    });
  }

  slide.addText('Generated by TranscriptAI', {
    x:0.8, y:7.18, w:11.7, h:0.24,
    fontSize:8, color:'4A3856', align:'right',
    fontFace:'Calibri', isTextBox:true, margin:0
  });
}

// ── Dispatch ──────────────────────────────────────────────────────────────────
var BUILDERS = {
  cover:        buildCover,
  overview:     buildOverview,
  action_items: buildActionItems,
  speakers:     buildSpeakers,
  sentiment:    buildSentiment,
  risk:         buildRisk,
  closing:      buildClosing
};

(plan.slides || []).forEach(function(s) {
  var slide = prs.addSlide();
  var fn    = BUILDERS[s.type];
  if (fn) {
    fn(slide, s);
  } else {
    slide.addText('Unknown slide type: ' + s.type, {
      x:1, y:2.5, w:11, h:0.6,
      fontSize:13, color:'C0392B',
      isTextBox:true, margin:0
    });
  }
});

prs.writeFile({ fileName: '__OUT_PATH__' })
  .then(function() { process.exit(0); })
  .catch(function(err) { console.error('pptxgenjs:', err.message); process.exit(1); });
"""


# ── Public API ────────────────────────────────────────────────────────────────

def build_pptx(plan: dict, timeout: int = 90) -> bytes:
    """
    Convert a SlidePlan dict into raw PPTX bytes.

    Args:
        plan    : dict produced by SlideArchitectAgent.plan()
        timeout : seconds to wait for Node.js process (default 90)

    Returns:
        bytes — the complete .pptx file content

    Raises:
        RuntimeError  — if Node.js exits non-zero
        FileNotFoundError — if Node.js is not installed
        subprocess.TimeoutExpired — if generation takes too long

    DSA: O(S × C) + Node.js startup overhead (~1–3 s)
         S = slides, C = max content items per slide
    """
    with tempfile.TemporaryDirectory() as tmp:
        plan_path = os.path.join(tmp, "plan.json")
        js_path   = os.path.join(tmp, "gen.js")
        out_path  = os.path.join(tmp, "output.pptx")

        # Write plan as JSON
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False)

        # Inject paths into JS template
        script = (
            _JS
            .replace("__PLAN_PATH__", plan_path)
            .replace("__OUT_PATH__",  out_path)
        )
        with open(js_path, "w", encoding="utf-8") as f:
            f.write(script)

        # Execute
        proc = subprocess.run(
            ["node", js_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"pptxgenjs exited {proc.returncode}.\n"
                f"stderr: {proc.stderr.strip()}\n"
                f"stdout: {proc.stdout.strip()}"
            )

        # Read and return bytes
        with open(out_path, "rb") as f:
            return f.read()


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from agents.slide_architect import SlideArchitectAgent

    mock = {
        "meeting_title": "Q3 Budget Review — Acme Corp",
        "summary": [
            "Q3 budget exceeded by 12% due to unplanned engineering hires.",
            "Client raised concerns about delivery timeline and system stability.",
            "Team committed to written response within 2 hours.",
        ],
        "action_items": [
            {"task": "Send revised budget report", "description": "Send revised budget report",
             "owner": "Alice", "deadline": "by Friday", "urgency_tier": "this_week"},
            {"task": "Provide written response", "description": "Provide written response",
             "owner": "Kenji", "deadline": "within 2 hours", "urgency_tier": "immediate"},
            {"task": "Schedule follow-up review", "description": "Schedule follow-up review",
             "owner": "Alice", "deadline": "by next week", "urgency_tier": "standard"},
        ],
        "speakers": [
            {"name": "Alice",  "talk_time_pct": 45, "tone": "assertive"},
            {"name": "Kenji",  "talk_time_pct": 35, "tone": "deferential"},
            {"name": "Client", "talk_time_pct": 20, "tone": "concerned"},
        ],
        "sentiment": [
            {"speaker": "Alice",  "score": "PROFESSIONAL", "note": "Direct and solution-focused"},
            {"speaker": "Kenji",  "score": "DEFENSIVE",    "note": "Escalation + relationship preservation"},
            {"speaker": "Client", "score": "CONCERNED",    "note": "Trust fragile but recoverable"},
        ],
        "soft_rejections": {
            "risk_level": "HIGH", "total_signals": 3,
            "signals": [
                {"phrase": "上司に相談します", "english": "Need to consult manager", "speaker": "Kenji"},
            ],
            "cultural_note": "Escalation phrase signals authority deferral, not nemawashi.",
        },
        "_detected_language": "ja",
    }

    agent = SlideArchitectAgent()
    plan  = agent.plan(mock, lang="ja")

    print(f"Building PPTX for {len(plan['slides'])} slides...")
    pptx_bytes = build_pptx(plan)

    out = "/tmp/transcriptai_test.pptx"
    with open(out, "wb") as f:
        f.write(pptx_bytes)

    print(f"Written: {out}  ({len(pptx_bytes):,} bytes)")
    print("Run: python /mnt/skills/public/pptx/scripts/office/validate.py", out)