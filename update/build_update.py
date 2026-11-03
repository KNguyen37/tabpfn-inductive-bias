#!/usr/bin/env python
"""Build the one-page project update. Result numbers are read from committed files.

    .venv/bin/python update/build_update.py
"""
import json
import statistics
from pathlib import Path

import pandas as pd
from matplotlib import get_data_path
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path("update/project_update.pdf")
REPO = "https://github.com/KNguyen37/tabpfn-inductive-bias"

# Proposed in PR #17 and docs/DECISIONS.md; checked against the screen's output below.
PROPOSED_TREE = ("covertype", "run_or_walk", "kdd_ipums_la_97_small")
PROPOSED_SMOOTH = ("california", "eeg_eye_state", "kc1")

# Embedded so the page looks the same in every viewer. DejaVu ships with matplotlib.
_FONTS = Path(get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("Body", _FONTS / "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("Body-Bold", _FONTS / "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body", boldItalic="Body-Bold")

# Proposal Table 3, for comparison with the rerun.
PROPOSAL_ACCURACY = dict(logreg=62.9, knn=88.5, rf=89.7, gbdt=88.9, svm=88.7, mlp=91.7, gp=92.6)


def load_numbers() -> dict:
    surfaces = pd.read_csv("results/metric_sanity/known_surfaces.csv").set_index("surface")
    rerun = pd.read_csv("results/metrics/classic_pilot_rerun.csv").groupby("model").mean(numeric_only=True)
    tau = pd.read_csv("results/metric_sanity/jump_share_threshold_sensitivity.csv", index_col="model")
    classic = pd.read_csv("results/metrics/classic_synth.csv")
    if len(classic) != 8960 or classic.duplicated(["dataset", "variant", "n", "seed", "model"]).any():
        raise SystemExit("The classic sweep is not complete; rewrite the update's text.")
    screen = pd.read_csv("results/metrics/real_screen.csv")
    screened = screen[screen.status == "screened"]
    sizes = screened.groupby("dataset").n.nunique()
    missing = set(PROPOSED_TREE + PROPOSED_SMOOTH) - set(screened.dataset)
    if missing:
        raise SystemExit(f"Proposed datasets not in the screen: {missing}")
    smoke = json.loads(Path("results/smoke/tabpfn_smoke.json").read_text())
    warm = {
        m: statistics.median(r["total_seconds"] for r in smoke["runs"] if r["model"] == m and not r["cold"])
        for m in ("tabpfn_v2", "tabpfn_v35")
    }
    if smoke["verdict"] != "GO" or smoke["errors"]:
        raise SystemExit("The smoke report is not a clean GO; rewrite the update's text.")
    return {
        "ramp0": surfaces.loc["smooth ramp, 0 deg"],
        "ramp45": surfaces.loc["smooth ramp, 45 deg"],
        "edge0": surfaces.loc["sharp edge, 0 deg"],
        "edge45": surfaces.loc["sharp edge, 45 deg"],
        "acc_gap": max(abs(rerun.accuracy[m] * 100 - v) for m, v in PROPOSAL_ACCURACY.items()),
        "rf_low_tau": tau.loc["rf", "0.01"],
        "rf_high_tau": tau.loc["rf", "0.1"],
        "smoke": smoke,
        "warm": warm,
        "runs": len(pd.read_csv("results/metrics/classic_pilot_rerun.csv")),
        "classic_runs": len(classic),
        "classic": classic.groupby("model")[["jump_share", "axis_alignment"]].mean(),
        "nan_axis": int(classic.axis_alignment.isna().sum()),
        "nan_jump": int(classic.jump_share.isna().sum()),
        "screened": len(sizes),
        "both_sizes": int((sizes == 2).sum()),
        "one_size": int((sizes == 1).sum()),
        "fits": len(pd.read_csv("results/metrics/real_screen_runs.csv")),
    }


def build():
    x = load_numbers()
    ink = colors.HexColor("#1b1b1b")
    rule = colors.HexColor("#c9c8c3")
    base = dict(fontName="Body", fontSize=8.2, leading=10.8, textColor=ink)
    body = ParagraphStyle("body", **base, spaceAfter=2.5)
    bullet = ParagraphStyle("bullet", **base, leftIndent=9, bulletIndent=0, spaceAfter=1.8)
    cell = ParagraphStyle("cell", fontName="Body", fontSize=7.7, leading=9.8, textColor=ink)
    head = ParagraphStyle("head", fontName="Body-Bold", fontSize=9.8, leading=12, textColor=ink,
                          spaceBefore=5, spaceAfter=2.5)
    title = ParagraphStyle("title", fontName="Body-Bold", fontSize=13, leading=16, alignment=TA_CENTER, textColor=ink)
    sub = ParagraphStyle("sub", fontName="Body", fontSize=8.2, leading=11, alignment=TA_CENTER, textColor=ink, spaceAfter=1)

    def P(text, style=cell):
        return Paragraph(text, style)

    def B(text):
        return Paragraph(text, bullet, bulletText="•")

    def table(rows, widths):
        t = Table([[P(c) for c in r] for r in rows], colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.7, ink),
            ("LINEBELOW", (0, 1), (-1, -1), 0.3, rule),
            ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 1.4), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
        ]))
        return t

    c, smoke, warm = x["classic"], x["smoke"], x["warm"]
    gpu = smoke["gpu"].replace("NVIDIA ", "").split("-")[0]
    story = [
        P("What Has TabPFN’s Prior Learned?", title),
        P("Project update · DS-GA 3001: Visualization for Machine Learning · November 3, 2026", sub),
        P("Kiet Nguyen (kn2827@nyu.edu) and Nhan Doan (nsd8221@nyu.edu)", sub),

        P("Progress Since the Proposal", head),
        B("<b>Foundation (Kiet).</b> Built the shared repository: a data format for all runs, corrected metrics, nested synthetic "
          "datasets, a resumable sweep runner for local and Colab use, and a test suite."),
        B(f"<b>Metric test (Kiet).</b> On surfaces of known shape, the proposal\u2019s step score S gives a smooth ramp and a sharp edge the same score "
          f"({x['ramp0'].axis_alignment_formerly_S:.2f} at 0°, {x['ramp45'].axis_alignment_formerly_S:.2f} at 45°). "
          f"Our new jump share separates them ({x['ramp0'].jump_share:.2f} vs. {x['edge0'].jump_share:.2f} at 0°; "
          f"{x['ramp45'].jump_share:.2f} vs. {x['edge45'].jump_share:.2f} at 45°)."),
        B(f"<b>Classic sweep (Nhan).</b> All {x['classic_runs']:,} synthetic runs are complete: 8 datasets, 4 variants, 4 context "
          f"sizes, 10 seeds, 7 models. Mean jump share: logistic regression {c.jump_share['logreg']:.2f}, GP {c.jump_share['gp']:.2f}, "
          f"SVM {c.jump_share['svm']:.2f}, random forest {c.jump_share['rf']:.2f}, MLP {c.jump_share['mlp']:.2f}, gradient boosting "
          f"{c.jump_share['gbdt']:.2f}, k-NN {c.jump_share['knn']:.2f}. Random forest and MLP are close on jump share but differ on "
          f"axis alignment ({c.axis_alignment['rf']:.2f} vs. {c.axis_alignment['mlp']:.2f}), so we read the two metrics together. "
          f"An earlier {x['runs']}-run rerun at the pilot’s size landed within {x['acc_gap']:.1f} accuracy points of the proposal’s table."),
        B(f"<b>Real-data screen (Nhan).</b> {x['screened']} OpenML datasets screened with classic models only over 10 seeds "
          f"({x['fits']:,} fits; {x['both_sizes']} at n=100 and 500, {x['one_size']} at n=100 only). TabPFN has not seen any of them."),
        B(f"<b>Model access (Kiet).</b> TabPFN v2 and 3.5 both ran on a Colab {gpu} without errors, taking about "
          f"{warm['tabpfn_v2']:.1f} s and {warm['tabpfn_v35']:.1f} s per full 200×200 grid job. This single-problem "
          f"test gives a rough estimate of {smoke['estimated_gpu_hours']:.0f} GPU hours for the planned runs."),
        B(f"<b>Flagged for the metric freeze.</b> Near-flat surfaces leave shape metrics undefined in some classic runs "
          f"({x['nan_axis']} for axis alignment, {x['nan_jump']} for jump share), and SVM probabilities are unreliable at very small n."),

        P("Response to Feedback", head),
        table([
            ["<b>Feedback</b>", "<b>Decision</b>", "<b>What changed</b>"],
            ["1. Step score", "Replaced",
             "S is renamed axis alignment; jump share now measures sharpness. Jump share depends on its threshold "
             f"(random forest: {x['rf_low_tau']:.2f} at 0.01, {x['rf_high_tau']:.2f} at 0.1), so we will fix the value before "
             "reading sweep results and report sensitivity."],
            ["2. Statistical strength", "Accepted",
             "10 seeds, with paired differences, 95% CIs, effect sizes and pre-set margins. Inconclusive results are reported as such."],
            ["3. Two datasets", "Both options",
             f"Framed as a case study, with roles assigned using classic models only. Proposed: {', '.join(PROPOSED_TREE)} (tree-favoring) and "
             f"{', '.join(PROPOSED_SMOOTH)} (smooth-favoring). Smooth-favoring data proved scarce, so we relaxed that side’s rule after "
             "seeing classic results, before TabPFN, and will disclose this. eeg_eye_state needs a re-screen with time-block splits. "
             "Fallback: a smaller case study."],
            ["4. Classmate study", "Removed", "Replaced by 3–5 expert walkthroughs reported qualitatively, plus our worked examples."],
            ["5. Visualization claim", "Accepted",
             "D3 only; Streamlit removed. Claim: a slider shows one boundary evolving as context grows, and compare mode shows "
             "|pA − pB|, with a worked example of each. A static page cannot run TabPFN, so the slider uses precomputed nested contexts."],
            ["6. Model access", "Checked", "Go for H5, H6 and RQ3, on the result above. TabPFN runs remain on Kiet’s Colab."],
        ], [1.15 * inch, 0.72 * inch, 5.36 * inch]),

        P("Scope and Plan", head),
        P("<b>Completed:</b> the foundation, metric test, classic sweep, dataset screen and model access check above. "
          "<b>Remaining:</b> TabPFN sweeps, real-data tests, statistical analysis, the D3 interface, walkthroughs and the paper.", body),
        table([
            ["<b>Dates</b>", "<b>Kiet</b>", "<b>Nhan</b>"],
            ["Nov 4–10", "Freeze metrics, hypotheses and margins; run TabPFN synthetic sweep, results unread until the freeze", "Finalize datasets and re-screen eeg_eye_state; D3 canvas and controls"],
            ["Nov 11–17", "RQ1 and RQ3 analysis; paper skeleton", "Small multiples and metrics panel"],
            ["Nov 18–24", "TabPFN real-data runs; RQ2 analysis", "Real-data pipeline and baselines; slider and compare mode"],
            ["Nov 25–Dec 1", "Full paper draft; walkthroughs", "Story mode; walkthroughs"],
            ["Dec 2–8", "Freeze results Dec 5; revise paper", "Clean-clone reproducibility test; polish interface; revise paper"],
            ["Dec 9–12", "Final fixes; submit by Dec 11", "Final fixes; Dec 12 is buffer"],
        ], [0.95 * inch, 3.08 * inch, 3.2 * inch]),
        Spacer(1, 3),
        P("<b>If time runs short, we cut in order:</b> ensemble-size ablation, noise-column test, H6, the additional datasets, story mode.", body),

        P("GitHub Repository", head),
        P(f'<link href="{REPO}" color="#1d5ea9">{REPO}</link> (public; the README lists how to reproduce current results)', body),

        P("AI Usage Disclosure", head),
        P("The project\u2019s direction and research ideas originated with us. We used Claude (Claude Code) to help with coding, "
          "experiment orchestration, writing and organization. It also suggested some technical options, such as the jump share "
          "metric, which we reviewed before adopting. We are responsible for all content and results.", body),
    ]

    OUT.parent.mkdir(exist_ok=True)
    SimpleDocTemplate(
        str(OUT), pagesize=letter, leftMargin=0.55 * inch, rightMargin=0.55 * inch,
        topMargin=0.45 * inch, bottomMargin=0.4 * inch,
        title="TabPFN project update", author="Kiet Nguyen and Nhan Doan",
    ).build(story)

    pages = len(PdfReader(OUT).pages)
    print(f"{OUT}: {pages} page(s)")
    if pages != 1:
        raise SystemExit("The update must be exactly one page.")


if __name__ == "__main__":
    build()
