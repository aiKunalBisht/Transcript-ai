"""
exporters/pptx_builder.py
Takes a PresentationPlan and builds a .pptx file.
Returns raw bytes ready for st.download_button.
"""
import io
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from agents.slide_architect import PresentationPlan


# ── Brand colours matching app.py sakura palette ─────────────────────────────
C_INK        = RGBColor(0x3C, 0x24, 0x16)   # dark brown
C_INK_MID    = RGBColor(0x7A, 0x50, 0x40)   # mid brown
C_INK_SOFT   = RGBColor(0xA8, 0x78, 0x68)   # soft brown
C_SAKURA     = RGBColor(0xD9, 0x60, 0x80)   # sakura pink
C_SAKURA_DEEP= RGBColor(0xBE, 0x40, 0x60)   # deep sakura
C_WASHI      = RGBColor(0xFA, 0xF6, 0xF2)   # warm off-white background
C_SURFACE    = RGBColor(0xFF, 0xFE, 0xFB)   # card surface
C_BORDER     = RGBColor(0xEF, 0xE2, 0xD8)   # border
C_PEACH      = RGBColor(0xE8, 0x80, 0x60)   # accent peach
C_GOLD       = RGBColor(0xB8, 0x78, 0x30)   # gold


def _set_bg(slide, color: RGBColor):
    """Fill slide background with a solid colour."""
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def _add_text(tf, text: str, size: int, bold: bool, color: RGBColor,
              align=PP_ALIGN.LEFT, space_before: int = 0):
    """Add a paragraph to a text frame with full formatting."""
    p   = tf.add_paragraph()
    p.text            = text
    p.alignment       = align
    p.space_before    = Pt(space_before)
    run               = p.runs[0]
    run.font.size     = Pt(size)
    run.font.bold     = bold
    run.font.color.rgb= color
    run.font.name     = "Georgia" if bold else "Calibri"
    return p


def build_pptx(plan: PresentationPlan) -> bytes:
    prs = Presentation()
    prs.slide_width  = Inches(13.33)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]   # completely blank

    for idx, slide_data in enumerate(plan.slides):
        slide = prs.slides.add_slide(blank_layout)
        _set_bg(slide, C_WASHI)

        is_title_slide = (idx == 0)
        is_last_slide  = (idx == len(plan.slides) - 1)

        # ── Left accent bar ───────────────────────────────────────────────────
        bar = slide.shapes.add_shape(
            1,  # MSO_SHAPE_TYPE.RECTANGLE
            Inches(0), Inches(0),
            Inches(0.12), Inches(7.5),
        )
        bar.fill.solid()
        bar.fill.fore_color.rgb = C_SAKURA if not is_last_slide else C_PEACH
        bar.line.fill.background()

        # ── Top accent line ───────────────────────────────────────────────────
        top_line = slide.shapes.add_shape(
            1, Inches(0.12), Inches(0),
            Inches(13.21), Inches(0.04),
        )
        top_line.fill.solid()
        top_line.fill.fore_color.rgb = C_BORDER
        top_line.line.fill.background()

        # ── Slide number pill (bottom right) ──────────────────────────────────
        num_box = slide.shapes.add_textbox(
            Inches(12.2), Inches(7.1), Inches(1.0), Inches(0.3)
        )
        num_tf = num_box.text_frame
        num_tf.word_wrap = False
        p = num_tf.add_paragraph()
        p.text = f"{slide_data.slide_number} / {plan.total_slides}"
        p.alignment = PP_ALIGN.RIGHT
        run = p.runs[0]
        run.font.size  = Pt(9)
        run.font.color.rgb = C_INK_SOFT
        run.font.name  = "Calibri"

        # ── Language tag (top right) ──────────────────────────────────────────
        lang_box = slide.shapes.add_textbox(
            Inches(11.8), Inches(0.15), Inches(1.4), Inches(0.25)
        )
        lang_tf = lang_box.text_frame
        p2 = lang_tf.add_paragraph()
        p2.text = slide_data.language.upper()
        p2.alignment = PP_ALIGN.RIGHT
        r2 = p2.runs[0]
        r2.font.size = Pt(8)
        r2.font.color.rgb = C_INK_SOFT
        r2.font.name = "Calibri"

        if is_title_slide:
            # ── TITLE SLIDE LAYOUT ────────────────────────────────────────────
            # Executive summary label
            label_box = slide.shapes.add_textbox(
                Inches(0.5), Inches(1.4), Inches(12.0), Inches(0.3)
            )
            ltf = label_box.text_frame
            lp  = ltf.add_paragraph()
            lp.text = "MEETING INTELLIGENCE · TRANSCRIPT AI"
            lr = lp.runs[0]
            lr.font.size = Pt(9)
            lr.font.color.rgb = C_SAKURA
            lr.font.bold  = True
            lr.font.name  = "Calibri"
            lr.font.name  = "Calibri"

            # Main title
            title_box = slide.shapes.add_textbox(
                Inches(0.5), Inches(1.9), Inches(11.5), Inches(1.8)
            )
            ttf = title_box.text_frame
            ttf.word_wrap = True
            tp = ttf.add_paragraph()
            tp.text = plan.meeting_title
            tr = tp.runs[0]
            tr.font.size = Pt(42)
            tr.font.bold = True
            tr.font.color.rgb = C_INK
            tr.font.name = "Georgia"

            # Executive summary
            exec_box = slide.shapes.add_textbox(
                Inches(0.5), Inches(3.85), Inches(10.5), Inches(1.2)
            )
            etf = exec_box.text_frame
            etf.word_wrap = True
            ep = etf.add_paragraph()
            ep.text = plan.executive_summary
            er = ep.runs[0]
            er.font.size = Pt(18)
            er.font.color.rgb = C_INK_MID
            er.font.name = "Calibri"

            # Divider line (visual)
            div = slide.shapes.add_shape(
                1, Inches(0.5), Inches(3.7), Inches(5.0), Inches(0.025)
            )
            div.fill.solid()
            div.fill.fore_color.rgb = C_SAKURA
            div.line.fill.background()

            # Duration pill
            dur = sum(s.estimated_duration_seconds for s in plan.slides)
            dur_min = dur // 60
            dur_box = slide.shapes.add_textbox(
                Inches(0.5), Inches(5.2), Inches(4.0), Inches(0.3)
            )
            dtf = dur_box.text_frame
            dp  = dtf.add_paragraph()
            dp.text = f"Estimated duration: {dur_min} min  ·  {plan.total_slides} slides"
            dr = dp.runs[0]
            dr.font.size = Pt(11)
            dr.font.color.rgb = C_INK_SOFT
            dr.font.name = "Calibri"

        else:
            # ── CONTENT SLIDE LAYOUT ──────────────────────────────────────────
            # Slide title
            title_box = slide.shapes.add_textbox(
                Inches(0.35), Inches(0.3), Inches(12.5), Inches(0.85)
            )
            ttf = title_box.text_frame
            ttf.word_wrap = True
            tp = ttf.add_paragraph()
            tp.text = slide_data.title
            tr = tp.runs[0]
            tr.font.size = Pt(30)
            tr.font.bold = True
            tr.font.color.rgb = C_INK if not is_last_slide else C_SAKURA_DEEP
            tr.font.name = "Georgia"

            # Thin divider under title
            div = slide.shapes.add_shape(
                1, Inches(0.35), Inches(1.2), Inches(12.6), Inches(0.025)
            )
            div.fill.solid()
            div.fill.fore_color.rgb = C_BORDER
            div.line.fill.background()

            # Bullets
            bullet_box = slide.shapes.add_textbox(
                Inches(0.55), Inches(1.4), Inches(11.8), Inches(5.5)
            )
            btf = bullet_box.text_frame
            btf.word_wrap = True

            for b_idx, bullet in enumerate(slide_data.bullets):
                bp = btf.add_paragraph()
                bp.text = f"  {bullet}"
                bp.space_before = Pt(10 if b_idx > 0 else 0)
                br = bp.runs[0]
                br.font.size = Pt(22)
                br.font.color.rgb = C_INK_MID
                br.font.name = "Calibri"
                br.font.bold = False

                # Bullet dot (sakura pink)
                dot_box = slide.shapes.add_textbox(
                    Inches(0.3), Inches(1.38 + b_idx * 0.75),
                    Inches(0.25), Inches(0.4)
                )
                dtf2 = dot_box.text_frame
                dp2  = dtf2.add_paragraph()
                dp2.text = "◆"
                dr2 = dp2.runs[0]
                dr2.font.size = Pt(10)
                dr2.font.color.rgb = C_SAKURA if not is_last_slide else C_PEACH
                dr2.font.name = "Calibri"

        # ── Speaker notes (always) ────────────────────────────────────────────
        notes_slide = slide.notes_slide
        notes_tf    = notes_slide.notes_text_frame
        notes_tf.text = slide_data.speaker_notes

    # ── Save to bytes ─────────────────────────────────────────────────────────
    buf = io.BytesIO()
    prs.save(buf)
    buf.seek(0)
    return buf.getvalue()