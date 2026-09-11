"""Assemble PowerPoint decks from the per-frame exports.

Every frame is placed as its own picture shape, so the slides stay editable:
you can move, delete or recolour individual frames in PowerPoint.

  # one slide per control step of a single episode
  python scripts/make_memory_slides.py steps --frames <dir> --stats step_stats.json --out deck.pptx
  # one slide per task
  python scripts/make_memory_slides.py tasks --root <dir-of-per-task-frame-dirs> \
      --picks task_picks.json --out deck.pptx
"""
import argparse
import glob
import json
import os
import pathlib

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

DEMO = RGBColor(0x89, 0x83, 0xBF)   # demonstration-video frames
EXEC = RGBColor(0x54, 0xB3, 0x45)   # the robot's own rollout
ACC = RGBColor(0xF2, 0x79, 0x70)
INK = RGBColor(0x33, 0x33, 0x33)
GREY = RGBColor(0x77, 0x77, 0x77)
ROWS = ["deployment\ndeterministic", "training  λ=0", "training  λ=1"]


def new_deck():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    return prs


def tb(sl, x, y, w, h, text, size=13, color=INK, bold=False, align=PP_ALIGN.LEFT):
    t = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    f = t.text_frame
    f.word_wrap = True
    f.margin_left = f.margin_right = f.margin_top = f.margin_bottom = 0
    p = f.paragraphs[0]
    p.alignment = align
    for i, line in enumerate(text.split("\n")):
        par = p if i == 0 else f.add_paragraph()
        par.alignment = align
        r = par.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.color.rgb = color
        r.font.bold = bold
    return t


def draw_tree_slide(prs, man, frames_dir, heading, subtitle, footer=""):
    """Three rows (deployment, λ=0, λ=1) × the pooled frames, one picture each."""
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    exec_start = man["exec_start"]
    F = len(man["frames"])
    X0, Y0, CELL, GAPX, GAPY = 1.30, 1.70, min(1.36, 11.2 / max(F, 1)), 0.05, 0.42
    tb(sl, 0.5, 0.24, 9.5, 0.45, heading, 23, INK, True)
    tb(sl, 0.5, 0.72, 12.4, 0.28, subtitle, 11.5, GREY)
    for x, col, lab in ((0.5, DEMO, "demonstration video — the manner to imitate"),
                        (4.6, EXEC, "execution — the robot's own rollout")):
        s = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(1.075), Inches(0.16), Inches(0.11))
        s.fill.solid(); s.fill.fore_color.rgb = col
        s.line.fill.background(); s.shadow.inherit = False
        tb(sl, x + 0.22, 1.04, 3.6, 0.2, lab, 10.5, col)
    nd = sum(1 for fr in man["frames"] if fr["t"] < exec_start)
    for lo, hi, col in ((0, nd, DEMO), (nd, F, EXEC)):
        if hi <= lo:
            continue
        x = X0 + lo * (CELL + GAPX)
        w = (hi - lo) * CELL + (hi - lo - 1) * GAPX
        b = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(1.36), Inches(w), Inches(0.05))
        b.fill.solid(); b.fill.fore_color.rgb = col
        b.line.fill.background(); b.shadow.inherit = False
    for f_i, fr in enumerate(man["frames"]):
        col = DEMO if fr["t"] < exec_start else EXEC
        tb(sl, X0 + f_i * (CELL + GAPX), 1.45, CELL, 0.22, f"t = {fr['t']}", 10.5, col, True, PP_ALIGN.CENTER)
    for r_i, row in enumerate(man["rows"]):
        y = Y0 + r_i * (CELL + GAPY)
        tb(sl, 0.16, y + CELL / 2 - 0.26, 1.06, 0.6, ROWS[r_i], 10.5, INK, True, PP_ALIGN.RIGHT)
        for f_i, cell in enumerate(row["cells"]):
            x = X0 + f_i * (CELL + GAPX)
            col = DEMO if man["frames"][f_i]["t"] < exec_start else EXEC
            pic = sl.shapes.add_picture(os.path.join(frames_dir, cell["file"]),
                                        Inches(x), Inches(y), Inches(CELL), Inches(CELL))
            pic.line.color.rgb = col
            pic.line.width = Pt(1.75)
            tb(sl, x, y + CELL + 0.015, CELL, 0.2, f"{cell['kept']}/16 kept", 9.5, col, False, PP_ALIGN.CENTER)
        if F % 2 == 0:
            xd = X0 + (F // 2) * (CELL + GAPX) - GAPX / 2 - 0.015
            d = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(xd), Inches(y - 0.04),
                                    Inches(0.03), Inches(CELL + 0.08))
            d.fill.solid(); d.fill.fore_color.rgb = INK
            d.line.fill.background(); d.shadow.inherit = False
    tb(sl, 0.5, 6.98, 12.4, 0.24,
       "black divider = the round-1 chunk boundary: node A scores the older half, node B the newer half",
       10, GREY, False, PP_ALIGN.CENTER)
    if footer:
        tb(sl, 0.5, 7.20, 12.4, 0.24, footer, 10.5, ACC, False, PP_ALIGN.CENTER)
    return sl


def mode_steps(args):
    prs = new_deck()
    stats = {r["step"]: r for r in json.load(open(args.stats))} if args.stats else {}
    for man_path in sorted(glob.glob(os.path.join(args.frames, "manifest_step*.json"))):
        man = json.load(open(man_path))
        st = man["step"]
        r = stats.get(st, {})
        sub = (f"{r['n_demo_frames']}/8 pooled frames are still demonstration video   ·   "
               f"{r['demo_tokens']} of the 32 kept tokens come from the video   ·   "
               f"newest frame gets {r['newest']}   ·   round-1 nodes contribute "
               f"{r['chunkA']} and {r['chunkB']}") if r else ""
        foot = (f"training with λ=1 keeps {r['overlap']}/32 of the same tokens as deployment") if r else ""
        draw_tree_slide(prs, man, args.frames, f"Control step {st}", sub, foot)
    prs.save(args.out)
    print("wrote", args.out, len(prs.slides._sldIdLst), "slides")


def mode_tasks(args):
    prs = new_deck()
    picks = json.load(open(args.picks))
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    tb(sl, 0.7, 0.8, 12, 0.8, "The D&R memory tree on every RoboMME task", 30, INK, True)
    tb(sl, 0.7, 1.75, 12, 0.4, args.subtitle, 15, GREY)
    tb(sl, 0.7, 2.6, 12, 3.4,
       "One episode per task, shown at a late control step. Each row is the same memory pool cut by the same\n"
       "tree; the rows differ only in the selector's training-time Gumbel noise.\n\n"
       "black — dropped by its round-1 node (the lower level)     grey — dropped by the root cut     unmasked — kept\n\n"
       "Purple frame border = demonstration video the robot must imitate. Green = the robot's own rollout.", 14)
    for task in sorted(picks):
        d = os.path.join(args.root, task)
        mans = sorted(glob.glob(os.path.join(d, "manifest_step*.json")))
        if not mans:
            continue
        man = json.load(open(mans[-1]))
        v = picks[task]
        ok = "success" if v["success"] else "FAILED rollout"
        kept_demo = sum(c["kept"] for c, fr in zip(man["rows"][0]["cells"], man["frames"])
                        if fr["t"] < man["exec_start"])
        sub = (f"episode {v['chosen_episode']} · {ok} · control step {man['step']} · "
               f"execution starts at t={man['exec_start']} · "
               f"{kept_demo} of the 32 kept tokens come from the demonstration video")
        draw_tree_slide(prs, man, d, task, sub)
    prs.save(args.out)
    print("wrote", args.out, len(prs.slides._sldIdLst), "slides")


def _strip(sl, y, cells, x0, cell, gapx, label, sub, exec_start, labcol=INK):
    """One mechanism's frame row; every frame is placed as its own picture."""
    tb(sl, 0.16, y + cell / 2 - 0.34, 1.10, 0.8, label, 11, labcol, True, PP_ALIGN.RIGHT)
    tb(sl, 0.16, y + cell / 2 + 0.06, 1.10, 0.5, sub, 8.5, GREY, False, PP_ALIGN.RIGHT)
    for i, c in enumerate(cells):
        x = x0 + i * (cell + gapx)
        col = DEMO if c["t"] < exec_start else EXEC
        pic = sl.shapes.add_picture(c["path"], Inches(x), Inches(y), Inches(cell), Inches(cell))
        pic.line.color.rgb = col
        pic.line.width = Pt(1.6)
        tb(sl, x, y - 0.22, cell, 0.2, f"t = {c['t']}", 9.5, col, True, PP_ALIGN.CENTER)
        tb(sl, x, y + cell + 0.02, cell, 0.2, c["cap"], 9, col, False, PP_ALIGN.CENTER)


def mode_mechanisms(args):
    """One slide per task: FrameSamp, D&R and TokenDrop on the same rollout."""
    prs = new_deck()
    picks = json.load(open(args.picks))
    BLUE = RGBColor(0x05, 0xB9, 0xE2)
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    tb(sl, 0.7, 0.75, 12, 0.8, "Three memory mechanisms, every RoboMME task", 30, INK, True)
    tb(sl, 0.7, 1.7, 12, 0.4, args.subtitle, 15, GREY)
    tb(sl, 0.7, 2.5, 12, 3.8,
       "FrameSamp and TokenDrop pick their tokens with parameter-free rules that live in the memory buffer, so "
       "given the same\nframes they build the same memory whichever policy produced them. Each slide replays all "
       "three over one rollout.\n\n"
       "FrameSamp (budget 64)   4 evenly spaced frames, all 16 patches of each kept. Sparse in time, complete per frame.\n\n"
       "D&R (budget 64, pool 128)   8 evenly spaced frames, 128 candidates cut to 32 by the tree. Twice the time "
       "coverage,\nonly a few patches per frame. Shown at deployment settings (deterministic top-K).\n\n"
       "TokenDrop (budget 512)   8×8 patches scored by pixel difference every 8 frames; the highest scoring fill the "
       "budget.\nFrame 0 enters with a sentinel score of 1000 and can never be evicted. The 8 frames shown are those "
       "holding the\nmost tokens, out of the many the mechanism draws from.\n\n"
       "Black mask = not in memory.   Purple border = demonstration video · green = the robot's own rollout.", 14)
    X0, CELL, GAPX = 1.32, 1.30, 0.055
    for task in sorted(picks):
        fs = os.path.join(args.baselines, task)
        dn = os.path.join(args.dnr, task)
        fsm = sorted(glob.glob(os.path.join(fs, "manifest_framesamp*_step*.json")))
        tdm = sorted(glob.glob(os.path.join(fs, "manifest_tokendrop512_step*.json")))
        dnm = sorted(glob.glob(os.path.join(dn, "manifest_step*.json")))
        if not (fsm and tdm and dnm):
            print("skipping", task, "- missing manifests")
            continue
        fsj, tdj, dnj = (json.load(open(m[-1])) for m in (fsm, tdm, dnm))
        v = picks[task]
        ex = dnj["exec_start"]
        sl = prs.slides.add_slide(prs.slide_layouts[6])
        tb(sl, 0.5, 0.22, 9, 0.45, task, 23, INK, True)
        tb(sl, 0.5, 0.70, 12.4, 0.26,
           f"episode {v['chosen_episode']} · {'success' if v['success'] else 'FAILED rollout'} · "
           f"control step {dnj['step']} · "
           + (f"execution starts at t={ex}" if ex else "no demonstration video in this task"),
           11.5, GREY)
        _strip(sl, 1.30,
               [{"t": f["t"], "path": os.path.join(fs, f["file"]), "cap": f"{f['kept']}/16"}
                for f in fsj["frames"]],
               X0, CELL, GAPX, "FrameSamp", "budget 64\n4 frames × 16", ex)
        _strip(sl, 3.20,
               [{"t": dnj["frames"][i]["t"], "path": os.path.join(dn, c["file"]), "cap": f"{c['kept']}/16"}
                for i, c in enumerate(dnj["rows"][0]["cells"])],
               X0, CELL, GAPX, "D&R", "budget 64\npool 128 → 32", ex, ACC)
        _strip(sl, 5.10,
               [{"t": f["t"], "path": os.path.join(fs, f["file"]), "cap": f"{f['kept']}/64"}
                for f in tdj["frames"]],
               X0, CELL, GAPX, "TokenDrop",
               f"budget 512\ntop 8 of {tdj['n_source_frames']} frames", ex, BLUE)
        kept_demo = sum(c["kept"] for c, fr in zip(dnj["rows"][0]["cells"], dnj["frames"])
                        if fr["t"] < ex)
        tb(sl, 0.5, 6.74, 12.4, 0.5,
           f"FrameSamp spends its 64 tokens on 4 whole frames.   "
           f"D&R spreads the same budget over 8 frames"
           + (f" and keeps {kept_demo} of 32 from the video.   " if ex else ".   ")
           + f"TokenDrop draws from {tdj['n_source_frames']} frames, 64 of its slots being frame 0's sentinel.",
           11, GREY, False, PP_ALIGN.CENTER)
    prs.save(args.out)
    print("wrote", args.out, len(prs.slides._sldIdLst), "slides")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("steps"); a.add_argument("--frames", required=True)
    a.add_argument("--stats", default=None); a.add_argument("--out", required=True)
    a.set_defaults(fn=mode_steps)
    b = sub.add_parser("tasks"); b.add_argument("--root", required=True)
    b.add_argument("--picks", required=True); b.add_argument("--out", required=True)
    b.add_argument("--subtitle", default="dnr_bud64_pool128_abl3_slotrope_ab @ 80k · seed 7")
    b.set_defaults(fn=mode_tasks)
    c = sub.add_parser("mechanisms")
    c.add_argument("--dnr", required=True, help="root of per-task D&R frame exports")
    c.add_argument("--baselines", required=True, help="root of per-task FrameSamp/TokenDrop exports")
    c.add_argument("--picks", required=True); c.add_argument("--out", required=True)
    c.add_argument("--subtitle", default="dnr_bud64_pool128_abl3_slotrope_ab @ 80k · seed 7")
    c.set_defaults(fn=mode_mechanisms)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
