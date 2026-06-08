"""
exporters/pptx_builder.py
Takes a PresentationPlan and builds a .pptx file.
Returns raw bytes ready for st.download_button.
"""
import io
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from agents.slide_architect import PresentationPlan

C_INK         = RGBColor(0x3C, 0x24, 0x16)
C_INK_MID     = RGBColor(0x7A, 0x50, 0x40)
C_INK_SOFT    = RGBColor(0xA8, 0x78, 0x68)
C_SAKURA      = RGBColor(0xD9, 0x60, 0x80)
C_SAKURA_DEEP = RGBColor(0xBE, 0x40, 0x60)
C_WASHI       = RGBColor(0xFA, 0xF6, 0xF2)
C_BORDER      = RGBColor(0xEF, 0xE2, 0xD8)
C_PEACH       = RGBColor(0xE8, 0x80, 0x60)


def _set_bg(slide, color: RGBColor):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def build_pptx(plan: PresentationPlan) -> bytes:
    prs = Presentation()
    prs.slide_width  = Inches(13.33)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]

    for idx, slide_data in enumerate(plan.slides):
        slide = prs.slides.add_slide(blank_layout)
        _set_bg(slide, C_WASHI)

        is_first = (idx == 0)
        is_last  = (idx == len(plan.slides) - 1)
        accent   = C_PEACH if is_last else C_SAKURA

        # Left accent bar
        bar = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(0.12), Inches(7.5))
        bar.fill.solid()
        bar.fill.fore_color.rgb = accent
        bar.line.fill.background()

        # Top line
        top = slide.shapes.add_shape(1, Inches(0.12), Inches(0), Inches(13.21), Inches(0.04))
        top.fill.solid()
        top.fill.fore_color.rgb = C_BORDER
        top.line.fill.background()

        # Slide number bottom right
        nb = slide.shapes.add_textbox(Inches(12.0), Inches(7.1), Inches(1.2), Inches(0.3))
        np = nb.text_frame.add_paragraph()
        np.text = f"{slide_data.slide_number} / {plan.total_slides}"
        np.alignment = PP_ALIGN.RIGHT
        nr = np.runs[0]
        nr.font.size = Pt(9)
        nr.font.color.rgb = C_INK_SOFT
        nr.font.name = "Calibri"

        if is_first:
            # Brand label
            lb = slide.shapes.add_textbox(Inches(0.5), Inches(1.2), Inches(12.0), Inches(0.3))
            lp = lb.text_frame.add_paragraph()
            lp.text = "MEETING INTELLIGENCE  ·  TRANSCRIPTAI"
            lr = lp.runs[0]
            lr.font.size = Pt(9)
            lr.font.bold = True
            lr.font.color.rgb = C_SAKURA
            lr.font.name = "Calibri"

            # Main title
            tb = slide.shapes.add_textbox(Inches(0.5), Inches(1.7), Inches(11.5), Inches(1.8))
            tb.text_frame.word_wrap = True
            tp2 = tb.text_frame.add_paragraph()
            tp2.text = plan.meeting_title
            tr2 = tp2.runs[0]
            tr2.font.size = Pt(40)
            tr2.font.bold = True
            tr2.font.color.rgb = C_INK
            tr2.font.name = "Georgia"

            # Divider
            div = slide.shapes.add_shape(1, Inches(0.5), Inches(3.6), Inches(4.5), Inches(0.025))
            div.fill.solid()
            div.fill.fore_color.rgb = C_SAKURA
            div.line.fill.background()

            # Executive summary
            eb = slide.shapes.add_textbox(Inches(0.5), Inches(3.75), Inches(10.5), Inches(1.2))
            eb.text_frame.word_wrap = True
            ep2 = eb.text_frame.add_paragraph()
            ep2.text = plan.executive_summary
            er2 = ep2.runs[0]
            er2.font.size = Pt(17)
            er2.font.color.rgb = C_INK_MID
            er2.font.name = "Calibri"

            # Duration
            dur_sec = sum(s.estimated_duration_seconds for s in plan.slides)
            dur_min = max(1, dur_sec // 60)
            db = slide.shapes.add_textbox(Inches(0.5), Inches(5.1), Inches(5.0), Inches(0.3))
            dp2 = db.text_frame.add_paragraph()
            dp2.text = f"Est. duration: {dur_min} min  ·  {plan.total_slides} slides"
            dr2 = dp2.runs[0]
            dr2.font.size = Pt(11)
            dr2.font.color.rgb = C_INK_SOFT
            dr2.font.name = "Calibri"

        else:
            # Slide title
            tb = slide.shapes.add_textbox(Inches(0.35), Inches(0.2), Inches(12.5), Inches(0.9))
            tb.text_frame.word_wrap = True
            tp2 = tb.text_frame.add_paragraph()
            tp2.text = slide_data.title
            tr2 = tp2.runs[0]
            tr2.font.size = Pt(30)
            tr2.font.bold = True
            tr2.font.color.rgb = C_SAKURA_DEEP if is_last else C_INK
            tr2.font.name = "Georgia"

            # Divider under title
            div = slide.shapes.add_shape(1, Inches(0.35), Inches(1.15), Inches(12.6), Inches(0.025))
            div.fill.solid()
            div.fill.fore_color.rgb = C_BORDER
            div.line.fill.background()

            # Bullets
            bullets = slide_data.bullets or ["See transcript for details"]
            for b_idx, bullet in enumerate(bullets[:5]):
                y_pos = 1.4 + b_idx * 0.82

                # Diamond dot
                dot = slide.shapes.add_textbox(Inches(0.3), Inches(y_pos), Inches(0.25), Inches(0.45))
                dp2 = dot.text_frame.add_paragraph()
                dp2.text = "◆"
                dr2 = dp2.runs[0]
                dr2.font.size = Pt(10)
                dr2.font.color.rgb = accent
                dr2.font.name = "Calibri"

                # Bullet text
                bb = slide.shapes.add_textbox(Inches(0.58), Inches(y_pos), Inches(12.0), Inches(0.7))
                bb.text_frame.word_wrap = True
                bp2 = bb.text_frame.add_paragraph()
                bp2.text = bullet
                br2 = bp2.runs[0]
                br2.font.size = Pt(21)
                br2.font.color.rgb = C_INK_MID
                br2.font.name = "Calibri"
                br2.font.bold = False

        # Speaker notes
        notes_slide = slide.notes_slide
        notes_slide.notes_text_frame.text = slide_data.speaker_notes

    buf = io.BytesIO()
    prs.save(buf)
    buf.seek(0)
    return buf.getvalue()
    