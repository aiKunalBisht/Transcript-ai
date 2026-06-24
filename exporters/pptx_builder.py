"""
exporters/pptx_builder.py — TranscriptAI Enterprise Edition
Design: Midnight Executive palette — dark cover/last, light content (sandwich)
Motif: numbered circle icons on every content slide (repeating, not stripes)
Layout variety:
  Slide 1 (cover):   dark bg, large title, exec summary, stat row
  Slide 2 (content): two-column — bullets left, stat callouts right
  Slide 3+ (content): icon+text rows filling full height
  Last slide:         dark bg, large CTA, next steps as numbered pills

NO accent stripes. NO color bars. NO cream backgrounds on content slides.
Safe fonts: Cambria titles, Calibri body. Margins=0 on all textboxes.
"""
import io
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from agents.slide_architect import PresentationPlan

# ── Midnight Executive palette ────────────────────────────────────────────────
DARK_BG     = RGBColor(0x1E, 0x27, 0x61)   # deep navy
DARK_CARD   = RGBColor(0x28, 0x33, 0x78)   # slightly lighter navy for cards
ICE_BLUE    = RGBColor(0xCA, 0xDC, 0xFC)   # ice blue — accent on dark slides
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
OFF_WHITE   = RGBColor(0xF8, 0xF9, 0xFF)   # content slide bg
CARD_BG     = RGBColor(0xEE, 0xF2, 0xFF)   # icon circle / card bg
NAVY_TEXT   = RGBColor(0x1E, 0x27, 0x61)   # body text on light slides
MID_TEXT    = RGBColor(0x3D, 0x4E, 0x8A)   # secondary text
SOFT_TEXT   = RGBColor(0x7A, 0x8A, 0xBB)   # captions
ACCENT      = RGBColor(0xCA, 0xDC, 0xFC)   # ice blue (on dark)
ACCENT_DARK = RGBColor(0x1E, 0x27, 0x61)   # navy (on light)
CORAL       = RGBColor(0xF9, 0x61, 0x67)   # warning / flagged items
GOLD        = RGBColor(0xF9, 0xE7, 0x95)   # highlight / stat

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)


def _bg(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def _tb(slide, l, t, w, h):
    shape = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.word_wrap = True
    return tf


def _run(tf, text, size, color, font="Calibri", bold=False, italic=False, align=PP_ALIGN.LEFT):
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.color.rgb = color
    r.font.name = font
    r.font.bold = bold
    r.font.italic = italic
    return r


_PML_NS = "{http://schemas.openxmlformats.org/presentationml/2006/main}"


def _strip_theme_style(shape):
    """
    Removes the <p:style> element python-pptx leaves on every autoshape.
    That element carries a theme effectRef (among other refs) which
    LibreOffice applies as a drop-shadow even when shadow.inherit=False
    sets an explicit empty effectLst — the two don't fully override each
    other in practice. Since fill/line are already set explicitly on every
    shape here, the style block's refs are redundant anyway.
    """
    sp = shape._element
    style = sp.find(_PML_NS + "style")
    if style is not None:
        sp.remove(style)


def _rect(slide, l, t, w, h, color):
    s = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = color; s.line.fill.background()
    s.shadow.inherit = False
    _strip_theme_style(s)
    return s


def _circle(slide, l, t, d, color):
    """Oval (circle when w==h) shape."""
    s = slide.shapes.add_shape(9, Inches(l), Inches(t), Inches(d), Inches(d))
    s.fill.solid(); s.fill.fore_color.rgb = color; s.line.fill.background()
    s.shadow.inherit = False
    _strip_theme_style(s)
    return s


def _num_in_circle(slide, number, l, t, d=0.44, bg=DARK_BG, fg=WHITE):
    """Numbered circle — the repeating motif."""
    _circle(slide, l, t, d, bg)
    tf = _tb(slide, l, t + 0.02, d, d - 0.04)
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    r = tf.paragraphs[0].add_run()
    r.text = str(number)
    r.font.size = Pt(13); r.font.bold = True
    r.font.color.rgb = fg; r.font.name = "Calibri"


def _slide_num(slide, n, total, fg=SOFT_TEXT):
    tf = _tb(slide, 12.3, 7.1, 0.9, 0.3)
    tf.paragraphs[0].alignment = PP_ALIGN.RIGHT
    r = tf.paragraphs[0].add_run()
    r.text = f"{n} / {total}"
    r.font.size = Pt(9); r.font.color.rgb = fg; r.font.name = "Calibri"


def _row_layout(start_y, band_h, n, ideal_row_h=1.6, min_row_h=0.9):
    """
    Returns (start_y, row_h) for n stacked rows within a vertical band.

    A 1-bullet slide dividing the full band by n=1 puts one line of text at
    the top and leaves the rest of the slide empty — that's what produced
    the "useless"-looking sparse slides. This caps row height at a sensible
    maximum and vertically centers the resulting block when content doesn't
    fill the band, instead of stretching it to fill regardless of count.
    Falls back to the old behavior (compress to fit) when there are enough
    bullets that ideal_row_h would overflow the band.
    """
    row_h = max(min_row_h, min(ideal_row_h, band_h / max(n, 1)))
    content_h = row_h * n
    if content_h < band_h:
        start_y = start_y + (band_h - content_h) / 2
    else:
        row_h = band_h / n
    return start_y, row_h


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — dark cover
# Layout: left half = title + exec summary + stats row
#          right half = large meeting name in light type
# ══════════════════════════════════════════════════════════════════════════════
def _cover_slide(prs, plan):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(slide, DARK_BG)

    # Right dark panel (slightly lighter) — visual depth, not a stripe
    _rect(slide, 7.8, 0, 5.53, 7.5, DARK_CARD)

    # Brand label — top left
    tf = _tb(slide, 0.5, 0.38, 7.0, 0.3)
    _run(tf, "TRANSCRIPTAI  ·  MEETING INTELLIGENCE", 8, ICE_BLUE, bold=True,
         align=PP_ALIGN.LEFT)

    # Main title — large, white
    tf_t = _tb(slide, 0.5, 0.82, 7.0, 2.4)
    p = tf_t.paragraphs[0]
    r = p.add_run()
    r.text = plan.meeting_title
    r.font.size = Pt(40); r.font.bold = True
    r.font.color.rgb = WHITE; r.font.name = "Cambria"

    # Exec summary — ice blue, italic
    tf_e = _tb(slide, 0.5, 3.4, 7.0, 1.5)
    tf_e.paragraphs[0].alignment = PP_ALIGN.LEFT
    r2 = tf_e.paragraphs[0].add_run()
    r2.text = plan.executive_summary
    r2.font.size = Pt(16); r2.font.color.rgb = ICE_BLUE
    r2.font.name = "Calibri"; r2.font.italic = True

    # Stats row — slides · duration · language
    dur_sec = sum(s.estimated_duration_seconds for s in plan.slides)
    dur_min = max(1, dur_sec // 60)
    stats = [
        (str(plan.total_slides), "SLIDES"),
        (f"{dur_min}m", "DURATION"),
        (plan.language.upper(), "LANGUAGE"),
    ]
    for i, (val, lbl) in enumerate(stats):
        x = 0.5 + i * 2.2
        # value
        tf_v = _tb(slide, x, 5.2, 2.0, 0.65)
        _run(tf_v, val, 28, WHITE, font="Cambria", bold=True)
        # label
        tf_l = _tb(slide, x, 5.85, 2.0, 0.3)
        _run(tf_l, lbl, 8, SOFT_TEXT, bold=True)

    # Right panel — decorative large meeting name watermark
    tf_w = _tb(slide, 7.95, 1.2, 5.2, 5.0)
    tf_w.word_wrap = True
    p_w = tf_w.paragraphs[0]
    p_w.alignment = PP_ALIGN.LEFT
    r_w = p_w.add_run()
    r_w.text = plan.meeting_title
    r_w.font.size = Pt(32); r_w.font.bold = True
    r_w.font.color.rgb = RGBColor(0x38, 0x45, 0x90)  # dim — purely decorative
    r_w.font.name = "Cambria"

    # Date label on right panel
    from datetime import datetime
    tf_d = _tb(slide, 7.95, 6.5, 5.0, 0.4)
    _run(tf_d, datetime.now().strftime("%B %d, %Y"), 11, SOFT_TEXT)

    _slide_num(slide, 1, plan.total_slides, fg=SOFT_TEXT)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — two-column: bullets left, stat callouts right
# ══════════════════════════════════════════════════════════════════════════════
def _two_col_slide(prs, slide_data, plan, idx):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(slide, OFF_WHITE)

    # Title
    tf_t = _tb(slide, 0.4, 0.22, 12.5, 0.9)
    _run(tf_t, slide_data.title, 32, NAVY_TEXT, font="Cambria", bold=True)

    # Light rule under title (shape not line — acceptable visual separator)
    _rect(slide, 0.4, 1.18, 12.5, 0.04, RGBColor(0xCC, 0xD4, 0xF0))

    bullets = (slide_data.bullets or ["Details in transcript"])[:4]
    n = len(bullets)
    font_sz = {1:22, 2:19, 3:16, 4:15}.get(n, 15)

    # LEFT column — numbered bullets (0.4" to 7.6")
    start_y, row_h = _row_layout(1.4, 5.5, n, ideal_row_h=1.7)
    for b_idx, bullet in enumerate(bullets):
        y = start_y + b_idx * row_h
        _num_in_circle(slide, b_idx + 1, 0.4, y + 0.05, 0.42, DARK_BG, WHITE)
        tf_b = _tb(slide, 0.98, y, 6.8, row_h - 0.1)
        _run(tf_b, bullet, font_sz, NAVY_TEXT)

    # RIGHT column — stat callout card (7.8" to 12.8")
    _rect(slide, 7.8, 1.35, 5.1, 5.7, CARD_BG)

    # Big number from slide number as visual anchor
    tf_big = _tb(slide, 8.0, 1.55, 4.7, 1.6)
    _run(tf_big, f"0{slide_data.slide_number}", 72, DARK_BG,
         font="Cambria", bold=True, align=PP_ALIGN.CENTER)

    tf_sub = _tb(slide, 8.0, 3.1, 4.7, 0.5)
    _run(tf_sub, slide_data.title.upper(), 9, SOFT_TEXT,
         bold=True, align=PP_ALIGN.CENTER)

    # Speaker notes excerpt in card
    notes_preview = (slide_data.speaker_notes or "")[:140]
    tf_np = _tb(slide, 8.1, 3.75, 4.6, 2.8)
    tf_np.word_wrap = True
    _run(tf_np, f'"{notes_preview}"', 11, MID_TEXT, italic=True)

    _slide_num(slide, slide_data.slide_number, plan.total_slides)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 3+ — icon+text rows, fills full height
# ══════════════════════════════════════════════════════════════════════════════
def _icon_row_slide(prs, slide_data, plan):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(slide, WHITE)

    # Inset header card (NOT edge-to-edge — a full-width band reads as
    # AI-generated filler; this stops short of both side margins instead)
    _rect(slide, 0.4, 0.3, 12.53, 0.95, DARK_BG)

    # Title inside the card
    tf_t = _tb(slide, 0.7, 0.52, 11.9, 0.6)
    _run(tf_t, slide_data.title, 26, WHITE, font="Cambria", bold=True)

    bullets = (slide_data.bullets or ["Details in transcript"])[:5]
    n = len(bullets)
    font_sz = {1:20, 2:18, 3:16, 4:15, 5:14}.get(n, 14)

    ICON_CHARS = ["◆", "▸", "●", "★", "◉"]

    start_y, row_h = _row_layout(1.45, 5.6, n, ideal_row_h=1.6)
    for b_idx, bullet in enumerate(bullets):
        y = start_y + b_idx * row_h
        is_even = b_idx % 2 == 0

        # Alternating subtle row tint
        if is_even:
            _rect(slide, 0.3, y + 0.04, 12.7, row_h - 0.08,
                  RGBColor(0xF4, 0xF6, 0xFF))

        # Numbered circle — the motif
        _num_in_circle(slide, b_idx + 1, 0.45, y + (row_h - 0.44) / 2,
                       0.44, DARK_BG, ICE_BLUE)

        # Icon accent
        tf_ic = _tb(slide, 1.05, y + (row_h - 0.44) / 2, 0.44, 0.44)
        _run(tf_ic, ICON_CHARS[b_idx % len(ICON_CHARS)], 14,
             RGBColor(0xCA, 0xDC, 0xFC), align=PP_ALIGN.CENTER)

        # Bullet text
        tf_b = _tb(slide, 1.55, y + (row_h - 0.55) / 2, 11.4, row_h * 0.85)
        _run(tf_b, bullet, font_sz, NAVY_TEXT)

    _slide_num(slide, slide_data.slide_number, plan.total_slides)


# ══════════════════════════════════════════════════════════════════════════════
# LAST SLIDE — dark, next steps as numbered pills
# ══════════════════════════════════════════════════════════════════════════════
def _closing_slide(prs, slide_data, plan):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(slide, DARK_BG)

    # Subtle lighter panel on right
    _rect(slide, 8.5, 0, 4.83, 7.5, DARK_CARD)

    # Title
    tf_t = _tb(slide, 0.5, 0.35, 7.7, 1.0)
    _run(tf_t, slide_data.title, 34, WHITE, font="Cambria", bold=True)

    # Sub-label
    tf_s = _tb(slide, 0.5, 1.35, 7.7, 0.4)
    _run(tf_s, "NEXT STEPS  ·  ACTION REQUIRED", 9, ICE_BLUE, bold=True)

    bullets = (slide_data.bullets or ["Follow up with all stakeholders"])[:5]
    n = len(bullets)
    font_sz = {1:20, 2:18, 3:15, 4:14, 5:13}.get(n, 13)

    start_y, row_h = _row_layout(1.9, 4.8, n, ideal_row_h=1.5, min_row_h=0.85)
    for b_idx, bullet in enumerate(bullets):
        y = start_y + b_idx * row_h
        # Pill background
        _rect(slide, 0.5, y, 7.7, row_h - 0.12,
              RGBColor(0x28, 0x33, 0x78))
        # Number
        _num_in_circle(slide, b_idx + 1, 0.6, y + 0.06, 0.42, ICE_BLUE,
                       DARK_BG)
        # Text
        tf_b = _tb(slide, 1.18, y + 0.06, 6.8, row_h - 0.18)
        _run(tf_b, bullet, font_sz, WHITE)

    # Right panel — brand watermark
    tf_br = _tb(slide, 8.65, 0.6, 4.5, 1.2)
    _run(tf_br, "TranscriptAI", 24, ICE_BLUE, font="Cambria",
         bold=True, align=PP_ALIGN.CENTER)

    tf_bl = _tb(slide, 8.65, 1.75, 4.5, 0.35)
    _run(tf_bl, "Meeting Intelligence Platform", 10, SOFT_TEXT,
         align=PP_ALIGN.CENTER)

    # Generated by line
    tf_g = _tb(slide, 8.65, 6.88, 4.5, 0.35)
    _run(tf_g, "github.com/aiKunalBisht/Transcript-ai", 8,
         SOFT_TEXT, align=PP_ALIGN.CENTER)

    _slide_num(slide, slide_data.slide_number, plan.total_slides, fg=SOFT_TEXT)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════
def build_pptx(plan) -> bytes:
    """
    Builds enterprise PPTX from PresentationPlan.
    Sandwich structure: dark cover → light content → dark closing.
    Returns raw bytes for st.download_button.
    Never raises.

    Accepts either a PresentationPlan instance OR a plain dict (e.g. the
    output of plan.model_dump(), or a dict that's been through a JSON
    round-trip via caching/storage). Whatever produced the dict upstream
    is worth tracking down separately, but build_pptx() should not crash
    just because its input took a detour through serialization —
    'dict' object has no attribute 'slides' was exactly this.
    """
    try:
        if isinstance(plan, dict):
            plan = PresentationPlan(**plan)

        prs = Presentation()
        prs.slide_width  = SLIDE_W
        prs.slide_height = SLIDE_H

        content_slides = plan.slides  # all slides from agent
        total = len(content_slides)

        for idx, slide_data in enumerate(content_slides):
            is_first = (idx == 0)
            is_last  = (idx == total - 1)

            if is_first:
                _cover_slide(prs, plan)
            elif is_last:
                _closing_slide(prs, slide_data, plan)
            elif idx == 1:
                # First content slide → two-column with stat card
                _two_col_slide(prs, slide_data, plan, idx)
            else:
                # All other content → icon+text rows
                _icon_row_slide(prs, slide_data, plan)

            # Speaker notes
            try:
                notes_slide = prs.slides[-1].notes_slide
                notes_slide.notes_text_frame.text = slide_data.speaker_notes or ""
            except Exception:
                pass

        buf = io.BytesIO()
        prs.save(buf)
        buf.seek(0)
        return buf.read()

    except Exception as e:
        # Emergency fallback
        import logging
        logging.error(f"pptx_builder failed: {e}", exc_info=True)
        try:
            prs = Presentation()
            prs.slide_width  = SLIDE_W
            prs.slide_height = SLIDE_H
            slide = prs.slides.add_slide(prs.slide_layouts[6])
            slide.background.fill.solid()
            slide.background.fill.fore_color.rgb = DARK_BG
            tb = slide.shapes.add_textbox(Inches(0.5), Inches(2), Inches(12), Inches(2))
            p  = tb.text_frame.paragraphs[0]
            r  = p.add_run()
            r.text = f"TranscriptAI — Build error: {str(e)[:200]}"
            r.font.size = Pt(18); r.font.color.rgb = WHITE
            buf = io.BytesIO(); prs.save(buf); buf.seek(0)
            return buf.read()
        except Exception:
            return b""