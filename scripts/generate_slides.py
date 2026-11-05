"""
Sylva Cortex — Ingénieux 2026 Hackathon Submission
Generate a professional 5-slide PowerPoint deck.

Install dependency:
    pip install python-pptx

Run from the project root:
    python scripts/generate_slides.py

Output:
    slides/sylva_cortex_submission.pptx
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


# ── Theme ────────────────────────────────────────────────────────────────────

BG = RGBColor(17, 21, 24)          # Dark charcoal
PANEL = RGBColor(25, 31, 34)       # Card background
PANEL_LIGHT = RGBColor(33, 41, 44)
WHITE = RGBColor(242, 246, 244)
MUTED = RGBColor(157, 171, 166)
GREEN = RGBColor(167, 255, 60)     # Neon green
GREEN_DARK = RGBColor(53, 91, 31)
BORDER = RGBColor(48, 60, 62)

FONT = "Aptos"
FONT_DISPLAY = "Aptos Display"

SLIDE_W = 13.333
SLIDE_H = 7.5


# ── Presentation setup ────────────────────────────────────────────────────────

prs = Presentation()
prs.slide_width = Inches(SLIDE_W)
prs.slide_height = Inches(SLIDE_H)

blank_layout = prs.slide_layouts[6]


# ── Helpers ──────────────────────────────────────────────────────────────────

def set_background(slide, color=BG):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, x, y, w, h, color, radius=False, line_color=None):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(
        shape_type,
        Inches(x),
        Inches(y),
        Inches(w),
        Inches(h),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color

    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()

    if radius:
        try:
            shape.adjustments[0] = 0.12
        except (IndexError, AttributeError):
            pass

    return shape


def add_text(
    slide,
    text,
    x,
    y,
    w,
    h,
    size=18,
    color=WHITE,
    bold=False,
    font=FONT,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    margin=0,
):
    box = slide.shapes.add_textbox(
        Inches(x),
        Inches(y),
        Inches(w),
        Inches(h),
    )
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(margin)
    tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign

    p = tf.paragraphs[0]
    p.alignment = align
    p.space_after = Pt(0)

    run = p.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color

    return box


def add_rich_text(slide, runs, x, y, w, h, size=18):
    box = slide.shapes.add_textbox(
        Inches(x),
        Inches(y),
        Inches(w),
        Inches(h),
    )
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0

    p = tf.paragraphs[0]
    p.space_after = Pt(0)

    for text, color, bold in runs:
        run = p.add_run()
        run.text = text
        run.font.name = FONT
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color

    return box


def add_header(slide, section, title, slide_num):
    add_text(
        slide,
        section.upper(),
        0.72,
        0.42,
        5.5,
        0.25,
        size=10,
        color=GREEN,
        bold=True,
    )

    add_text(
        slide,
        title,
        0.72,
        0.82,
        11.9,
        0.62,
        size=29,
        color=WHITE,
        bold=True,
        font=FONT_DISPLAY,
    )

    add_rect(slide, 0.72, 1.58, 0.65, 0.045, GREEN)

    add_text(
        slide,
        f"SYLVA CORTEX  /  INGÉNIEUX 2026                              {slide_num:02d}",
        0.72,
        7.12,
        11.9,
        0.18,
        size=8,
        color=MUTED,
        bold=True,
    )


def add_bullet(slide, text, x, y, w, size=17, color=WHITE):
    add_rect(slide, x, y + 0.12, 0.09, 0.09, GREEN, radius=True)
    add_text(
        slide,
        text,
        x + 0.25,
        y,
        w - 0.25,
        0.75,
        size=size,
        color=color,
    )


def add_card(slide, x, y, w, h, title=None):
    add_rect(slide, x, y, w, h, PANEL, radius=True, line_color=BORDER)
    if title:
        add_text(
            slide,
            title.upper(),
            x + 0.25,
            y + 0.2,
            w - 0.5,
            0.22,
            size=9,
            color=GREEN,
            bold=True,
        )


def add_connector(slide, x1, y1, x2, y2, color=GREEN):
    line = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        Inches(x1),
        Inches(y1),
        Inches(x2),
        Inches(y2),
    )
    line.line.color.rgb = color
    line.line.width = Pt(2)
    return line


# ── Slide 1: Title ───────────────────────────────────────────────────────────

slide = prs.slides.add_slide(blank_layout)
set_background(slide)

# Accent panels
add_rect(slide, 0, 0, 0.12, SLIDE_H, GREEN)
add_rect(slide, 9.7, 0.75, 2.9, 5.8, PANEL, radius=True)

# Decorative signal waveform
wave_x = 10.05
wave_y = 3.25
wave_heights = [
    0.25, 0.55, 0.35, 0.9, 1.3, 0.65, 1.7, 1.0,
    0.45, 1.25, 0.7, 0.35, 0.9, 0.5, 0.25,
]
for i, height in enumerate(wave_heights):
    add_rect(
        slide,
        wave_x + i * 0.14,
        wave_y - height / 2,
        0.055,
        height,
        GREEN if i % 3 == 0 else GREEN_DARK,
        radius=True,
    )

add_text(
    slide,
    "INGÉNIEUX 2026  /  TRACK 02",
    0.85,
    1.55,
    7.8,
    0.3,
    size=11,
    color=GREEN,
    bold=True,
)

add_text(
    slide,
    "Sylva Cortex:",
    0.85,
    2.15,
    8.2,
    0.75,
    size=36,
    color=WHITE,
    bold=True,
    font=FONT_DISPLAY,
)

add_text(
    slide,
    "Adaptive Bioacoustic\nThreat Detection",
    0.85,
    2.95,
    8.4,
    1.45,
    size=31,
    color=WHITE,
    bold=True,
    font=FONT_DISPLAY,
)

add_rect(slide, 0.85, 4.65, 0.7, 0.05, GREEN)

add_text(
    slide,
    "Real-time DSP + Agentic AI for Forest Protection",
    0.85,
    4.95,
    8.5,
    0.5,
    size=17,
    color=MUTED,
)

add_text(
    slide,
    "LISTEN  /  REASON  /  PROTECT",
    10.05,
    5.45,
    2.25,
    0.3,
    size=9,
    color=GREEN,
    bold=True,
    align=PP_ALIGN.CENTER,
)


# ── Slide 2: Problem & Solution ──────────────────────────────────────────────

slide = prs.slides.add_slide(blank_layout)
set_background(slide)
add_header(slide, "01 / Mission", "The Problem & The Solution", 2)

# Problem card
add_card(slide, 0.72, 2.0, 5.65, 3.75, "The challenge")

add_bullet(
    slide,
    "Forests are too large to patrol manually.",
    1.05,
    2.65,
    4.95,
    size=18,
)

add_bullet(
    slide,
    "Static classifiers fail because context matters.",
    1.05,
    3.55,
    4.95,
    size=18,
)

add_text(
    slide,
    "A chainsaw at 2 AM in a protected zone is not the same event as one at 2 PM in a logging concession.",
    1.3,
    4.45,
    4.65,
    0.95,
    size=13,
    color=MUTED,
)

# Solution card
add_card(slide, 6.75, 2.0, 5.85, 3.75, "The solution")

add_text(
    slide,
    "An intelligent agent\nthat listens, reasons,\nand acts.",
    7.1,
    2.75,
    5.0,
    1.65,
    size=27,
    color=WHITE,
    bold=True,
    font=FONT_DISPLAY,
)

add_rect(slide, 7.1, 4.75, 0.55, 0.05, GREEN)

add_text(
    slide,
    "Context-aware sensing for forest protection",
    7.1,
    5.0,
    4.8,
    0.4,
    size=13,
    color=MUTED,
)


# ── Slide 3: Technical Architecture ─────────────────────────────────────────

slide = prs.slides.add_slide(blank_layout)
set_background(slide)
add_header(slide, "02 / System", "Technical Architecture", 3)

card_y = 2.25
card_w = 3.55
card_h = 3.45
card_xs = [0.72, 4.9, 9.08]

# DSP
add_card(slide, card_xs[0], card_y, card_w, card_h, "01 / Signal processing")

add_text(
    slide,
    "DSP",
    card_xs[0] + 0.28,
    card_y + 0.65,
    card_w - 0.56,
    0.45,
    size=25,
    color=WHITE,
    bold=True,
    font=FONT_DISPLAY,
)

add_text(
    slide,
    "Butterworth bandpass\n80–8000 Hz\n\nSpectral subtraction\nα = 2.0\n\n40-Mel MFCCs",
    card_xs[0] + 0.28,
    card_y + 1.25,
    card_w - 0.56,
    1.75,
    size=14,
    color=MUTED,
)

# Classifier
add_card(slide, card_xs[1], card_y, card_w, card_h, "02 / Classification")

add_text(
    slide,
    "Classifier",
    card_xs[1] + 0.28,
    card_y + 0.65,
    card_w - 0.56,
    0.45,
    size=25,
    color=WHITE,
    bold=True,
    font=FONT_DISPLAY,
)

add_text(
    slide,
    "LightGBM\n\nTrained on\nSNR-degraded\nsynthetic data",
    card_xs[1] + 0.28,
    card_y + 1.25,
    card_w - 0.56,
    1.65,
    size=15,
    color=MUTED,
)

# Agent
add_card(slide, card_xs[2], card_y, card_w, card_h, "03 / Decision intelligence")

add_text(
    slide,
    "Agent",
    card_xs[2] + 0.28,
    card_y + 0.65,
    card_w - 0.56,
    0.45,
    size=25,
    color=WHITE,
    bold=True,
    font=FONT_DISPLAY,
)

add_text(
    slide,
    "Bayesian belief state\nwith temporal decay\n\nContextual multipliers\n\nAdaptive thresholds",
    card_xs[2] + 0.28,
    card_y + 1.25,
    card_w - 0.56,
    1.75,
    size=14,
    color=MUTED,
)

# Pipeline connectors
add_connector(slide, 4.35, 3.95, 4.72, 3.95)
add_connector(slide, 8.53, 3.95, 8.9, 3.95)

add_text(
    slide,
    "A replayable pipeline from raw audio to context-aware action",
    1.65,
    6.1,
    10.0,
    0.35,
    size=13,
    color=GREEN,
    bold=True,
    align=PP_ALIGN.CENTER,
)


# ── Slide 4: Experimental Results ────────────────────────────────────────────

slide = prs.slides.add_slide(blank_layout)
set_background(slide)
add_header(slide, "03 / Evaluation", "Experimental Results", 4)

# Metric cards
metric_cards = [
    (0.72, 2.0, 3.75, "98.82%", "E1 baseline accuracy"),
    (4.78, 2.0, 3.75, "95% F1", "Under heavy noise · −10 dB SNR"),
    (8.84, 2.0, 3.75, "1.0000", "E2 denoised F1 · 0 dB SNR"),
]

for x, y, w, value, label in metric_cards:
    add_card(slide, x, y, w, 1.55)
    add_text(
        slide,
        value,
        x + 0.25,
        y + 0.32,
        w - 0.5,
        0.55,
        size=27,
        color=GREEN,
        bold=True,
        font=FONT_DISPLAY,
    )
    add_text(
        slide,
        label,
        x + 0.25,
        y + 1.02,
        w - 0.5,
        0.3,
        size=11,
        color=MUTED,
    )

# Experiment details
add_card(slide, 0.72, 3.9, 7.45, 1.85, "DSP ablation · E2")

add_text(
    slide,
    "Denoiser improves F1-score at 0 dB SNR",
    1.0,
    4.45,
    6.8,
    0.4,
    size=16,
    color=WHITE,
    bold=True,
)

add_rich_text(
    slide,
    [
        ("0.9917", MUTED, True),
        ("   →   ", GREEN, True),
        ("1.0000", GREEN, True),
    ],
    1.0,
    4.95,
    5.8,
    0.5,
    size=23,
)

# Agent adaptation
add_card(slide, 8.55, 3.9, 4.04, 1.85, "Agent adaptation")

add_text(
    slide,
    "0.70  →  0.808",
    8.85,
    4.55,
    3.45,
    0.5,
    size=23,
    color=GREEN,
    bold=True,
    font=FONT_DISPLAY,
)

add_text(
    slide,
    "Threshold after simulated false-positive feedback",
    8.85,
    5.08,
    3.35,
    0.5,
    size=10,
    color=MUTED,
)


# ── Slide 5: Impact & Demo ───────────────────────────────────────────────────

slide = prs.slides.add_slide(blank_layout)
set_background(slide)
add_header(slide, "04 / Deployment", "Impact & Live Demo", 5)

# Impact points
add_card(slide, 0.72, 2.0, 5.1, 3.85, "Impact")

add_bullet(
    slide,
    "24/7 automated monitoring.",
    1.05,
    2.75,
    4.4,
    size=18,
)

add_bullet(
    slide,
    "Reduces alert fatigue via adaptive thresholds.",
    1.05,
    3.65,
    4.4,
    size=18,
)

add_text(
    slide,
    "Designed to support forest rangers with context-aware alerts.",
    1.3,
    4.7,
    4.1,
    0.75,
    size=13,
    color=MUTED,
)

# Demo panel
add_card(slide, 6.2, 2.0, 6.4, 3.85, "Live demo")

add_text(
    slide,
    "Listen.",
    6.55,
    2.75,
    1.6,
    0.45,
    size=24,
    color=WHITE,
    bold=True,
    font=FONT_DISPLAY,
)

add_text(
    slide,
    "Reason.",
    8.25,
    2.75,
    1.8,
    0.45,
    size=24,
    color=WHITE,
    bold=True,
    font=FONT_DISPLAY,
)

add_text(
    slide,
    "Act.",
    10.25,
    2.75,
    1.5,
    0.45,
    size=24,
    color=GREEN,
    bold=True,
    font=FONT_DISPLAY,
)

# Simple demo flow
flow_xs = [6.65, 8.65, 10.65]
flow_labels = ["Detect\nthreats", "Adjust\ncontext", "Dispatch\ndrones"]

for x, label in zip(flow_xs, flow_labels):
    add_rect(slide, x, 3.55, 1.45, 0.9, PANEL_LIGHT, radius=True, line_color=BORDER)
    add_text(
        slide,
        label,
        x + 0.1,
        3.72,
        1.25,
        0.5,
        size=12,
        color=WHITE,
        bold=True,
        align=PP_ALIGN.CENTER,
        valign=MSO_ANCHOR.MIDDLE,
    )

add_connector(slide, 8.12, 4.0, 8.55, 4.0)
add_connector(slide, 10.12, 4.0, 10.55, 4.0)

add_text(
    slide,
    "Watch the dashboard detect threats, adjust context, and dispatch drones in real-time.",
    6.55,
    4.9,
    5.55,
    0.65,
    size=13,
    color=MUTED,
)

# Closing footer accent
add_rect(slide, 0.72, 6.35, 11.88, 0.025, BORDER)
add_text(
    slide,
    "SYLVA CORTEX  ·  INTELLIGENT SENSING FOR FOREST PROTECTION",
    0.72,
    6.55,
    11.9,
    0.25,
    size=9,
    color=GREEN,
    bold=True,
)


# ── Save presentation ────────────────────────────────────────────────────────

OUTPUT = Path("slides/sylva_cortex_submission.pptx")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

prs.save(OUTPUT)
print(f"Created presentation: {OUTPUT.resolve()}")