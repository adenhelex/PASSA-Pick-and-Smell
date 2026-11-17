"""Build the executive deck from the dashboard screenshots."""

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).parent
SHOTS = HERE / "screenshots"
CROPS = HERE / "screenshots" / "crops"
OUT = HERE / "PASSA-Pick-and-Smell-Executive-Briefing.pptx"

BG = RGBColor(16, 20, 17)
PANEL = RGBColor(28, 34, 30)
LINE = RGBColor(58, 66, 60)
TEXT = RGBColor(232, 240, 234)
MUTED = RGBColor(160, 172, 164)
BLUE = RGBColor(86, 146, 214)
TEAL = RGBColor(52, 170, 140)
AMBER = RGBColor(226, 160, 64)
RED = RGBColor(220, 84, 66)
FONT = "Helvetica Neue"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = prs.slide_width, prs.slide_height


def crop(name, top=0.0, bottom=1.0, left=0.0, right=1.0):
    CROPS.mkdir(parents=True, exist_ok=True)
    image = Image.open(SHOTS / name)
    w, h = image.size
    box = (int(w * left), int(h * top), int(w * right), int(h * bottom))
    target = CROPS / name
    image.crop(box).save(target)
    return target


def background(slide):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = BG


def text(slide, x, y, w, h, value, size=18, color=TEXT, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(x, y, w, h)
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    lines = value if isinstance(value, list) else [value]
    for index, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = align
        run = paragraph.add_run()
        run.text = line
        run.font.name = FONT
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
    return box


def bullets(slide, x, y, w, h, items, size=17, gap=10, color=TEXT, marker_color=TEAL):
    box = slide.shapes.add_textbox(x, y, w, h)
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = 0
    for index, item in enumerate(items):
        head, _, rest = item.partition(" | ")
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.space_after = Pt(gap)
        mark = paragraph.add_run()
        mark.text = "■  "
        mark.font.size = Pt(size - 6)
        mark.font.color.rgb = marker_color
        mark.font.name = FONT
        first = paragraph.add_run()
        first.text = head
        first.font.name = FONT
        first.font.size = Pt(size)
        first.font.bold = bool(rest)
        first.font.color.rgb = color
        if rest:
            second = paragraph.add_run()
            second.text = "  " + rest
            second.font.name = FONT
            second.font.size = Pt(size)
            second.font.color.rgb = MUTED
    return box


def panel(slide, x, y, w, h, fill=PANEL, line=LINE, radius=True):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, x, y, w, h)
    if radius:
        shape.adjustments[0] = 0.06
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(1)
    shape.shadow.inherit = False
    return shape


def picture(slide, path, x, y, max_w, max_h, frame=True):
    image = Image.open(path)
    ratio = image.width / image.height
    w = max_w
    h = int(w / ratio)
    if h > max_h:
        h = max_h
        w = int(h * ratio)
    left = x + (max_w - w) // 2
    top = y + (max_h - h) // 2
    if frame:
        border = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left - Emu(9525 * 2), top - Emu(9525 * 2), w + Emu(9525 * 4), h + Emu(9525 * 4))
        border.fill.solid()
        border.fill.fore_color.rgb = LINE
        border.line.fill.background()
        border.shadow.inherit = False
    slide.shapes.add_picture(str(path), left, top, w, h)
    return left, top, w, h


def header(slide, kicker, title):
    background(slide)
    text(slide, Inches(0.6), Inches(0.42), Inches(12), Inches(0.35), kicker.upper(), size=12, color=TEAL, bold=True)
    text(slide, Inches(0.6), Inches(0.75), Inches(12.2), Inches(0.9), title, size=30, bold=True)


def footer(slide, number):
    text(slide, Inches(0.6), Inches(7.05), Inches(8), Inches(0.3), "PASSA Physical Robotics · Pick-and-Smell", size=10, color=MUTED)
    text(slide, Inches(11.7), Inches(7.05), Inches(1.0), Inches(0.3), str(number), size=10, color=MUTED, align=PP_ALIGN.RIGHT)


def notes(slide, value):
    slide.notes_slide.notes_text_frame.text = value


def arrow(slide, x1, y1, x2, y2, color=MUTED):
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    line.line.color.rgb = color
    line.line.width = Pt(1.75)
    ln = line.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    return line


def node(slide, x, y, w, h, title, detail, accent=BLUE):
    panel(slide, x, y, w, h)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y + Inches(0.12), Inches(0.06), h - Inches(0.24))
    bar.fill.solid()
    bar.fill.fore_color.rgb = accent
    bar.line.fill.background()
    text(slide, x + Inches(0.22), y + Inches(0.14), w - Inches(0.3), Inches(0.35), title, size=15, bold=True)
    text(slide, x + Inches(0.22), y + Inches(0.5), w - Inches(0.3), h - Inches(0.6), detail, size=11.5, color=MUTED)


def stat(slide, x, y, w, value, label, color=TEAL):
    text(slide, x, y, w, Inches(0.6), value, size=30, bold=True, color=color)
    text(slide, x, y + Inches(0.62), w, Inches(0.6), label, size=12, color=MUTED)


# 1. Title
slide = prs.slides.add_slide(BLANK)
background(slide)
picture(slide, crop("02-digital-twin.png", top=0.08, left=0.18, right=0.82), Inches(6.6), Inches(1.0), Inches(6.2), Inches(3.4))
text(slide, Inches(0.7), Inches(1.2), Inches(5.8), Inches(0.4), "EXECUTIVE BRIEFING", size=13, color=TEAL, bold=True)
text(slide, Inches(0.7), Inches(1.65), Inches(5.8), Inches(1.8), ["PASSA", "Pick-and-Smell"], size=48, bold=True)
text(
    slide,
    Inches(0.7),
    Inches(3.55),
    Inches(5.6),
    Inches(1.4),
    "A robot arm that sees the table, understands a plain-language request, moves to the object, grabs it, lifts, puts it down, and rests.",
    size=19,
    color=MUTED,
)
panel(slide, Inches(0.7), Inches(5.35), Inches(11.9), Inches(1.15))
stat(slide, Inches(1.0), Inches(5.45), Inches(2.6), "5 steps", "move, grab, lift, put down, rest")
stat(slide, Inches(3.9), Inches(5.45), Inches(2.6), "~3 s", "local AI picks the object", BLUE)
stat(slide, Inches(6.8), Inches(5.45), Inches(2.6), "4–5 / s", "live object detection updates", BLUE)
stat(slide, Inches(9.7), Inches(5.45), Inches(2.8), "100% local", "no cloud, no data leaves the site", AMBER)
notes(
    slide,
    "Open with the one-line idea: you tell the robot what to pick, it finds it, grabs it, lifts, puts it down, and rests. "
    "Smell runs only if the operator asks. Everything stays on one Mac on site.",
)

# 2. What it does
slide = prs.slides.add_slide(BLANK)
header(slide, "What it does today", "One sentence in, one complete pick cycle out")
picture(slide, SHOTS / "01-dashboard.png", Inches(0.6), Inches(1.75), Inches(7.6), Inches(5.1))
bullets(
    slide,
    Inches(8.6),
    Inches(1.85),
    Inches(4.2),
    Inches(5),
    [
        "See | The phone camera finds and labels every object on the table in real time.",
        "Understand | Thai or English, such as “pick the glass nearest the bottle.”",
        "Decide | A local AI model chooses which object the sentence means.",
        "Act | Move to it, grab, lift, put it down, then rest. Smell only if asked.",
        "Show | A live Braccio twin and a five-step log show exactly what the robot is doing.",
    ],
    size=16,
    gap=14,
)
footer(slide, 2)
notes(
    slide,
    "This is the live operator dashboard. Top: 3D twin of the arm and the table. Bottom left: live camera with detection boxes. "
    "Right: the command box. The operator only types a sentence and presses Send.",
)

# 3. See
slide = prs.slides.add_slide(BLANK)
header(slide, "Step 1 · See", "The camera finds and names each object on the table")
picture(slide, SHOTS / "03-camera-detections.png", Inches(0.6), Inches(1.75), Inches(7.0), Inches(5.1))
bullets(
    slide,
    Inches(8.0),
    Inches(1.85),
    Inches(4.8),
    Inches(5),
    [
        "On-device detector | YOLOv8 runs on the Mac, about 4–5 updates per second.",
        "Tight boxes with names | Each object gets its own box, label, and measured color.",
        "Tracks objects as they move | A chosen object keeps its identity frame to frame.",
        "Anything can be picked | If the detector has no name for it, the operator clicks it in the picture.",
        "Orange box = target | The object the robot is about to pick.",
    ],
    size=16,
    gap=13,
)
footer(slide, 3)
notes(
    slide,
    "The detector runs locally. It found the blue bottle and the glass (labelled cup by the detector). "
    "The orange box is the object the robot chose. The large box marked sink is the tabletop itself, a known detector quirk we filter out of the 3D view.",
)

# 4. Understand and decide
slide = prs.slides.add_slide(BLANK)
header(slide, "Step 2 · Understand and decide", "Plain language in. The AI only chooses the object.")
picture(slide, crop("08-command-result.png", top=0.0, bottom=0.5), Inches(0.6), Inches(1.75), Inches(5.6), Inches(5.1))
panel(slide, Inches(6.7), Inches(1.8), Inches(6.0), Inches(1.25))
text(slide, Inches(6.95), Inches(1.95), Inches(5.6), Inches(0.3), "REQUEST", size=11, color=TEAL, bold=True)
text(slide, Inches(6.95), Inches(2.3), Inches(5.6), Inches(0.6), "“pick the glass nearest the bottle”", size=20, bold=True)
panel(slide, Inches(6.7), Inches(3.2), Inches(6.0), Inches(1.25))
text(slide, Inches(6.95), Inches(3.35), Inches(5.6), Inches(0.3), "AI DECISION", size=11, color=BLUE, bold=True)
text(slide, Inches(6.95), Inches(3.7), Inches(5.6), Inches(0.6), "The model chose the white cup (the glass), not the bottle.", size=17)
bullets(
    slide,
    Inches(6.7),
    Inches(4.75),
    Inches(6.0),
    Inches(2.2),
    [
        "Understands position words | nearest, farthest, left, right, second from the left.",
        "The AI never moves the motors | It returns one object ID. The motion steps are fixed code.",
        "Runs on our own hardware | Ollama with gpt-oss or Llama, no internet needed.",
    ],
    size=15,
    gap=10,
)
footer(slide, 4)
notes(
    slide,
    "Key design decision for safety: the language model only answers one question, which object. "
    "It does not write the plan and it never sends motor commands. That keeps behaviour predictable and auditable.",
)

# 5. Act
slide = prs.slides.add_slide(BLANK)
header(slide, "Step 3 · Act", "A fixed 5-step cycle: move, grab, lift, put down, rest")
picture(slide, crop("08-command-result.png", top=0.5, bottom=1.0), Inches(0.6), Inches(1.75), Inches(5.0), Inches(5.1))
picture(slide, crop("09-mission-timeline.png", top=0.0, bottom=0.36), Inches(5.95), Inches(1.75), Inches(6.8), Inches(3.3))
bullets(
    slide,
    Inches(5.95),
    Inches(5.3),
    Inches(6.8),
    Inches(1.7),
    [
        "Same steps every time | move, grab, lift, put down, rest.",
        "Puts it back down | The arm returns to the pose where it gripped, then opens and rests.",
        "Every action is logged | Time-stamped record for each mission.",
    ],
    size=15,
    gap=9,
)
footer(slide, 5)
notes(
    slide,
    "Left: the 5 fixed steps. Right: the live mission log. "
    "The object is put down at the pose where the gripper closed, then the arm rests in the Begin pose.",
)

# 6. Digital twin
slide = prs.slides.add_slide(BLANK)
header(slide, "Digital twin", "A live Braccio model of the arm and the objects on the table")
picture(slide, crop("07-twin-reaching.png", top=0.08, left=0.15, right=0.85), Inches(0.6), Inches(1.75), Inches(8.2), Inches(4.3))
bullets(
    slide,
    Inches(9.1),
    Inches(1.85),
    Inches(3.7),
    Inches(5),
    [
        "Mirrors the camera | Left in the picture is left on the table. Right stays right.",
        "Braccio CAD | The twin uses the real arm meshes, not a generic stick model.",
        "Rest like the real arm | Park is the Begin pose: folded, gripper hanging, not pointing up.",
        "Grab stays folded | A pick leans forward to the object. It does not swing at the sky.",
        "Safe rehearsal | Every motion can be watched in dry-run before the arm is powered.",
    ],
    size=15,
    gap=12,
)
text(slide, Inches(0.6), Inches(6.25), Inches(8.2), Inches(0.5), "Captured mid-reach: the arm is moving toward the white cup (ringed in blue).", size=13, color=MUTED)
footer(slide, 6)
notes(
    slide,
    "This screenshot was taken while the arm was reaching for the cup in dry-run. "
    "The twin lets anyone verify what the robot will do before it is armed.",
)

# 7. How it works
slide = prs.slides.add_slide(BLANK)
header(slide, "How it works", "Everything runs on one Mac on site")
y1 = Inches(1.95)
nh = Inches(1.35)
nw = Inches(2.75)
xs = [Inches(0.6), Inches(3.75), Inches(6.9), Inches(10.05)]
node(slide, xs[0], y1, nw, nh, "iPhone camera", "Live video of the table.", BLUE)
node(slide, xs[1], y1, nw, nh, "Object detector", "YOLOv8 on the Mac. Boxes, names, colors.", BLUE)
node(slide, xs[2], y1, nw, nh, "Local AI model", "Reads the request. Returns one object ID.", BLUE)
node(slide, xs[3], y1, nw, nh, "Mission planner", "Fixed 5 steps: move, grab, lift, put down, rest.", TEAL)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(slide, a + nw, y1 + nh // 2, b, y1 + nh // 2)
y2 = Inches(4.15)
node(slide, xs[3], y2, nw, nh, "Motion controller", "Joint limits, one joint per command. Stop overrides everything.", AMBER)
node(slide, xs[2], y2, nw, nh, "Arduino Uno", "USB serial to the servo shield.", AMBER)
node(slide, xs[1], y2, nw, nh, "Braccio arm", "6 joints: base, shoulder, elbow, wrist ×2, gripper.", AMBER)
node(slide, xs[0], y2, nw, nh, "E-nose", "Smell sample at the wrist inlet.", TEAL)
arrow(slide, xs[3] + nw // 2, y1 + nh, xs[3] + nw // 2, y2)
arrow(slide, xs[3], y2 + nh // 2, xs[2] + nw, y2 + nh // 2)
arrow(slide, xs[2], y2 + nh // 2, xs[1] + nw, y2 + nh // 2)
arrow(slide, xs[1], y2 + nh // 2, xs[0] + nw, y2 + nh // 2)
panel(slide, Inches(0.6), Inches(5.9), Inches(12.2), Inches(0.85))
text(
    slide,
    Inches(0.85),
    Inches(6.08),
    Inches(11.8),
    Inches(0.6),
    "Operator dashboard (web): live camera, command box, 3D twin, mission log, joint angles, and the exact motor commands.",
    size=15,
    color=TEXT,
)
footer(slide, 7)
notes(
    slide,
    "Top row is perception and decision. Bottom row is physical action. "
    "The AI sits in the middle but can only pick an object; the planner and motion controller are deterministic code with hard limits.",
)

# 8. Safety
slide = prs.slides.add_slide(BLANK)
header(slide, "Safety and control", "Built to be predictable before it is fast")
picture(slide, SHOTS / "10-robot-uart.png", Inches(0.6), Inches(1.75), Inches(5.4), Inches(5.0))
bullets(
    slide,
    Inches(6.4),
    Inches(1.85),
    Inches(6.4),
    Inches(5.1),
    [
        "Dry-run by default | The arm only moves after someone explicitly presses Arm.",
        "Hard joint limits | Every pose is checked. Unsafe combinations are refused.",
        "Stop is instant | It goes straight to the motion controller and skips the planner.",
        "AI cannot drive motors | It chooses an object; fixed code does the rest.",
        "Full audit trail | Every motor command and step is shown and logged.",
        "No cloud | Camera images and data stay on the local machine.",
    ],
    size=16,
    gap=13,
)
footer(slide, 8)
notes(
    slide,
    "Left: the joint angles and the exact serial lines the controller would send to the Arduino. "
    "In dry-run nothing is sent, so we rehearse every mission before arming the hardware.",
)

# 9. Status
slide = prs.slides.add_slide(BLANK)
header(slide, "Where we are", "Working end to end in dry-run. Hardware validation is next.")
panel(slide, Inches(0.6), Inches(1.8), Inches(6.0), Inches(4.95))
text(slide, Inches(0.9), Inches(2.0), Inches(5.5), Inches(0.4), "WORKING NOW", size=13, color=TEAL, bold=True)
bullets(
    slide,
    Inches(0.9),
    Inches(2.5),
    Inches(5.5),
    Inches(4.2),
    [
        "Live detection on the iPhone camera",
        "Plain-language object choice by a local AI",
        "Move, grab, lift, put down, and rest in dry-run",
        "Smell only when the operator asks",
        "Live Braccio twin that follows the camera left and right",
        "47 automated tests passing",
        "Code on GitHub with version history",
    ],
    size=16,
    gap=9,
)
panel(slide, Inches(6.8), Inches(1.8), Inches(6.0), Inches(4.95))
text(slide, Inches(7.1), Inches(2.0), Inches(5.5), Inches(0.4), "NOT YET DONE", size=13, color=AMBER, bold=True)
bullets(
    slide,
    Inches(7.1),
    Inches(2.5),
    Inches(5.5),
    Inches(4.2),
    [
        "Camera-to-table calibration | Reach is still estimated from the image.",
        "Real E-nose readings | Smell data is simulated for now.",
        "Powered arm trials | Most runs so far are dry-run.",
        "Grip confirmation | The gripper does not report force.",
        "AI accuracy check | Small models misread complex sentences; larger ones do better.",
    ],
    size=15,
    gap=10,
    marker_color=AMBER,
)
footer(slide, 9)
notes(
    slide,
    "Be direct about limits. The software loop works end to end in simulation. "
    "The main risks are physical accuracy without calibration and the E-nose integration.",
)

# 10. Value
slide = prs.slides.add_slide(BLANK)
header(slide, "Why it matters", "Repeatable smell testing without a person at the bench")
cards = [
    ("Quality control", "Sample products for odor defects the same way every time.", TEAL),
    ("Lab sample triage", "Pick, sniff, and log many samples while staff do other work.", BLUE),
    ("Low-cost hardware", "Off-the-shelf arm, Arduino, phone camera, and one Mac.", AMBER),
    ("Easy to operate", "Type a sentence. No robot programming needed.", TEAL),
]
for index, (title, detail, color) in enumerate(cards):
    x = Inches(0.6) + index * Inches(3.08)
    panel(slide, x, Inches(1.95), Inches(2.9), Inches(2.6))
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, Inches(1.95), Inches(2.9), Inches(0.08))
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()
    text(slide, x + Inches(0.25), Inches(2.25), Inches(2.45), Inches(0.5), title, size=18, bold=True)
    text(slide, x + Inches(0.25), Inches(2.85), Inches(2.45), Inches(1.6), detail, size=14, color=MUTED)
panel(slide, Inches(0.6), Inches(4.85), Inches(12.2), Inches(1.85))
text(slide, Inches(0.9), Inches(5.05), Inches(11.6), Inches(0.4), "THE DIFFERENCE", size=12, color=TEAL, bold=True)
text(
    slide,
    Inches(0.9),
    Inches(5.45),
    Inches(11.6),
    Inches(1.2),
    "Most lab robots need a programmer for every new task. Here an operator describes the object in words, "
    "the robot finds it on its own, and every action stays inside fixed safety rules.",
    size=17,
)
footer(slide, 10)
notes(
    slide,
    "Use cases are candidates to validate, not committed markets. Ask the room which one to pilot first.",
)

# 11. Roadmap and asks
slide = prs.slides.add_slide(BLANK)
header(slide, "Next steps", "From working prototype to validated pilot")
phases = [
    ("Next 30 days", ["Calibrate camera to table", "Connect the real E-nose", "Supervised powered-arm runs"], TEAL),
    ("Next 60 days", ["Measure AI choice accuracy", "Grip check after each lift", "Safety checklist and operator guide"], BLUE),
    ("Next 90 days", ["Pilot on one real use case", "Report success rate and cycle time", "Decide on scale-up hardware"], AMBER),
]
for index, (title, items, color) in enumerate(phases):
    x = Inches(0.6) + index * Inches(4.1)
    panel(slide, x, Inches(1.85), Inches(3.9), Inches(2.85))
    text(slide, x + Inches(0.25), Inches(2.05), Inches(3.4), Inches(0.4), title.upper(), size=13, color=color, bold=True)
    bullets(slide, x + Inches(0.25), Inches(2.5), Inches(3.45), Inches(2.1), items, size=15, gap=9, marker_color=color)
panel(slide, Inches(0.6), Inches(4.95), Inches(12.2), Inches(1.8))
text(slide, Inches(0.9), Inches(5.12), Inches(11.6), Inches(0.4), "DECISIONS WE NEED", size=12, color=RED, bold=True)
bullets(
    slide,
    Inches(0.9),
    Inches(5.5),
    Inches(11.6),
    Inches(1.2),
    [
        "Pick the first pilot use case | quality control or lab sample triage.",
        "Approve bench time and E-nose integration support | for powered trials.",
    ],
    size=16,
    gap=8,
    marker_color=RED,
)
footer(slide, 11)
notes(
    slide,
    "Close with the asks. CEO: choose the pilot use case. COO: bench time and an operator. CTO: E-nose integration and calibration support.",
)

# 12. Q&A
slide = prs.slides.add_slide(BLANK)
background(slide)
picture(slide, crop("03-camera-detections.png", top=0.1, bottom=0.95), Inches(7.0), Inches(1.2), Inches(5.8), Inches(4.6))
text(slide, Inches(0.7), Inches(2.2), Inches(6), Inches(1.0), "Questions", size=48, bold=True)
text(
    slide,
    Inches(0.7),
    Inches(3.4),
    Inches(5.8),
    Inches(1.6),
    "Live demo available: type a request and watch the arm and its Braccio twin run move, grab, lift, put down, and rest in dry-run.",
    size=19,
    color=MUTED,
)
text(slide, Inches(0.7), Inches(5.3), Inches(6), Inches(0.4), "github.com/adenhelex/PASSA-Pick-and-Smell", size=14, color=TEAL)
notes(slide, "Offer the live demo. Keep it in dry-run unless the arm is supervised and cleared to move.")

prs.save(OUT)
print(OUT)
