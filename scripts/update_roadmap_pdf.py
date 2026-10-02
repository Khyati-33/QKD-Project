"""Append a dated evidence/status addendum to the QKD research roadmap PDF."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
BURGUNDY = colors.HexColor("#9E1735")
PALE = colors.HexColor("#F6F1F2")
DARK = colors.HexColor("#242424")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, default=ROOT / "paper" / "supplementary" /
                        "corrected_route_comparison_20261002.json")
    parser.add_argument("--physics", type=Path, default=ROOT / "paper" / "supplementary" /
                        "physics_analytical_validation.json")
    parser.add_argument("--fso", type=Path, default=ROOT / "paper" / "supplementary" /
                        "fso_model_validation.json")
    parser.add_argument("--sensitivity", type=Path, default=ROOT / "paper" / "supplementary" /
                        "link_quality_sensitivity_20261002.json")
    parser.add_argument("--pair-metrics", type=Path, default=ROOT / "paper" / "supplementary" /
                        "current_checkpoint_pair_metrics_20261002.json")
    parser.add_argument("--test-report", type=Path, default=ROOT / "paper" / "supplementary" /
                        "test_results_20261002.json")
    parser.add_argument("--roadmap", type=Path, default=ROOT /
                        "QKD_Routing_Agent_Research_Roadmap_QSMS_Style.pdf")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.comparison.exists():
        raise FileNotFoundError(f"comparison report not found: {args.comparison}")
    if not args.physics.exists() or not args.fso.exists():
        raise FileNotFoundError("physics validation reports are required")
    comparison, physics, fso = (read_json(args.comparison), read_json(args.physics),
                                read_json(args.fso))
    sensitivity = read_json(args.sensitivity) if args.sensitivity.exists() else None
    pair_metrics = read_json(args.pair_metrics) if args.pair_metrics.exists() else None
    test_report = read_json(args.test_report) if args.test_report.exists() else None
    dated = date.today().isoformat()
    out_pdf = args.output or ROOT / "paper" / "roadmap_status_addendum.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="RoadTitle", parent=styles["Title"],
        fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=BURGUNDY,
        alignment=TA_LEFT, spaceAfter=3))
    styles.add(ParagraphStyle(name="RoadSubtitle", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=13, leading=17, textColor=DARK,
        spaceAfter=8))
    styles.add(ParagraphStyle(name="RoadSection", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=BURGUNDY,
        spaceBefore=6, spaceAfter=5))
    styles.add(ParagraphStyle(name="RoadBody", parent=styles["BodyText"],
        fontName="Helvetica", fontSize=8.5, leading=11, textColor=DARK,
        spaceAfter=4))
    styles.add(ParagraphStyle(name="RoadSmall", parent=styles["BodyText"],
        fontName="Helvetica", fontSize=7.3, leading=9, textColor=DARK,
        spaceAfter=2))
    styles.add(ParagraphStyle(name="RoadCell", parent=styles["BodyText"],
        fontName="Helvetica", fontSize=7.4, leading=9.1, textColor=DARK))
    styles.add(ParagraphStyle(name="RoadCellHead", parent=styles["RoadCell"],
        fontName="Helvetica-Bold", textColor=colors.white))

    story = [paragraph("RESEARCH ROADMAP", styles["RoadTitle"]),
             paragraph(f"STATUS UPDATE - {dated}", styles["RoadSubtitle"])]
    intro = Table([[paragraph(
        "<b>How to read this update.</b> This addendum supplements the original 22-page roadmap. "
        "It records work completed in the repository, generated evidence, and remaining dependencies. "
        "Earlier archived results remain historical records and are not silently replaced. All current "
        "outputs are simulator-derived; none is field validation or a QKD security proof.", styles["RoadBody"])]],
        colWidths=[7.18 * inch])
    intro.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("BOX", (0, 0), (-1, -1), 0.5, BURGUNDY), ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story += [intro, Spacer(1, 7), paragraph("1. Work completed in this update", styles["RoadSection"])]

    analytical = physics["independent_formula_max_abs_error"]
    ar1 = physics["fso_log_cn2_ar1"]
    fso_rows = fso["rows"]
    max_fso_error = max(abs(row["availability_error"]) for row in fso_rows)
    max_fso_z = max(abs(row.get("error_in_target_standard_errors", 0.0))
                     for row in fso_rows)
    rate_sensitivity = (sensitivity or {}).get("controlled_input_sensitivity", {}).get(
        "rate_ratio_sweep", [])
    qber_sensitivity = (sensitivity or {}).get("controlled_input_sensitivity", {}).get(
        "qber_sweep", [])
    sensitivity_result = "The controlled diagnostic includes a proxy-rate ratio sweep, a feasible QBER-margin sweep, and an outage-mask counterfactual with route context held fixed."
    if rate_sensitivity and qber_sensitivity:
        sensitivity_result += (f" The FSO action probability rose from "
            f"{rate_sensitivity[0]['fso_probability']:.3f} to "
            f"{rate_sensitivity[-1]['fso_probability']:.3f} as its proxy-rate ratio rose "
            f"from {rate_sensitivity[0]['fso_to_fiber_rate_ratio']:.2g}x to "
            f"{rate_sensitivity[-1]['fso_to_fiber_rate_ratio']:.2g}x; at QBER 0.111 it was masked.")
    work_rows = [
        ["Workstream", "Change / result", "Remarks and evidence"],
        ["Corrected route comparison",
         "Baseline selectors use the observation's feasible-action ordering; all methods use the same season and channel seeds. Per-episode output records action validity, no-feasible-action states, fallbacks, route, hops, revisits, and link validity.",
         "The comparison output is identified below. No-feasible-action states are separated from invalid policy actions."],
        ["Physics consistency",
         f"Independent formula transcription covered {physics['fiber_sweep_points']} fiber distances from 0 to 200 km. Maximum absolute difference: QBER {analytical['qber']:.2g}; normalized proxy {analytical['normalized_rate_proxy']:.2g}. Transmission and proxy decrease; QBER does not decrease.",
         "Internal consistency only. It does not validate assumptions against measurements or establish security."],
        ["FSO stochastic checks",
         f"All 16 configured season/distance anchors sampled at {fso.get('samples_per_condition', 2000):,} trials each; maximum absolute availability deviation {max_fso_error:.4f} ({max_fso_z:.2f} target standard errors). Log-Cn2 AR(1): configured {ar1['configured']:.2f}, sampled {ar1['sampled']:.4f} over {ar1['samples']:,} steps.",
         "These are model targets and samples, not validation against Indian weather observations."],
        ["Link-feature sensitivity",
         (sensitivity_result if sensitivity else "Controlled rate/QBER/outage sensitivity output is not yet present."),
         "A policy-input diagnostic does not establish link-measurement validity or route-level performance."],
    ]
    work_table = Table([[paragraph(str(cell), styles["RoadCellHead"] if i == 0 else styles["RoadCell"])
                         for cell in row] for i, row in enumerate(work_rows)],
                       colWidths=[1.05 * inch, 3.25 * inch, 2.88 * inch], repeatRows=1)
    work_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), BURGUNDY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9D0D2")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [work_table, Spacer(1, 6), paragraph("2. Paired route comparison", styles["RoadSection"])]

    summary = comparison["summary"]
    cmp_rows = [["Policy", "Success", "Successful mean hops", "Invalid actions", "No-valid-action states", "Revisits"]]
    for policy_name, row in summary.items():
        hops = "n/a" if row["successful_mean_hops"] is None else f"{row['successful_mean_hops']:.2f}"
        ci = row.get("success_rate_wilson_95pct", [None, None])
        ci_text = (f"; 95% CI {ci[0]:.1%}-{ci[1]:.1%}" if ci[0] is not None else "")
        cmp_rows.append([policy_name, f"{row['successes']}/{row['episodes']} ({row['success_rate']:.1%}{ci_text})",
                         hops, str(row["invalid_actions"]), str(row.get("no_valid_action_states", 0)),
                         str(row["revisits"])])
    cmp_table = Table([[paragraph(str(cell), styles["RoadCellHead"] if i == 0 else styles["RoadCell"])
                        for cell in row] for i, row in enumerate(cmp_rows)],
                      colWidths=[1.05 * inch, 1.12 * inch, 1.14 * inch, 0.83 * inch, 1.55 * inch, 0.65 * inch],
                      repeatRows=1)
    cmp_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), BURGUNDY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9D0D2")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story += [cmp_table, Spacer(1, 5), paragraph(
        f"Conditions: {comparison['conditions']['source']} to {comparison['conditions']['destination']}; "
        f"{comparison['conditions']['seeds_per_season']} shared environment seeds per each of "
        f"{len(comparison['conditions']['seasons'])} seasons; QBER/rate action mask; "
        f"{comparison['conditions']['max_steps']} step limit. Scope is single-route simulator behavior, "
        "not traffic blocking, network-wide key-pool balance, or delivered service capacity.", styles["RoadSmall"])]
    if pair_metrics:
        aggregate = pair_metrics["aggregate"]
        successful_pairs = sum(bool(row["success"]) for row in pair_metrics["pairs"])
        story += [paragraph("Checkpoint transfer across city pairs", styles["RoadSection"]),
            paragraph(f"The same 50-epoch checkpoint completed {successful_pairs}/"
              f"{aggregate['pairs']} ordered city-pair episodes ({aggregate['success_rate']:.1%}) "
              f"under the archived monsoon-night scenario. Median successful route length was "
              f"{aggregate['median_hops']:.0f} hops; localized decision latency was "
              f"{aggregate['decision_p50_ms']:.2f} ms median and {aggregate['decision_p95_ms']:.2f} ms "
              f"p95 on this host. This is one stochastic sample per ordered pair, not a held-out "
              "endpoint protocol or a matched baseline comparison.", styles["RoadBody"])]
    story.append(PageBreak())

    story += [paragraph("3. Roadmap status after this work", styles["RoadSection"])]
    status_rows = [
        ["Priority", "Status", "Evidence / remaining work"],
        ["P0-1 corrected route comparison", "DONE - route-level",
         "Paired environment seeds, shared feasibility rules, and action/fallback accounting. The separate demand-driven QKP comparison remains out of scope."],
        ["P0-2 physics checks", "DONE - internal only",
         "Fiber formula consistency, configured FSO anchor sampling, and temporal correlation. No matched device/channel calibration or measured traces."],
        ["P0-3 link sensitivity", "DONE - diagnostic" if sensitivity else "IN PROGRESS",
         "Rate-ratio, QBER-margin, and hard outage-mask tests vary simulator policy inputs only."],
        ["P0-4 five training seeds", "TODO",
         "Five independent full-budget training runs are not present. Evaluation episodes for one checkpoint do not substitute for training-seed replication."],
        ["P1 endpoint / weather holdouts", "PARTIAL",
         "The current checkpoint was run on 42 ordered pairs (31/42 success), plus earlier adverse-condition episodes. These are coverage diagnostics, not held-out endpoint/weather tests."],
        ["P1 architecture / feature ablations", "TODO",
         "No matched-budget GNN depth, attention, feature, or reward ablation set is archived."],
        ["P1 inference cost", "PARTIAL",
         "Current checkpoint: 42-pair local inference had 3.83 ms median and 4.34 ms p95 decision latency. It is one host/scenario; report a full hardware profile and scaling before deployment claims."],
        ["P1 exact small-graph oracle", "TODO",
         "No exact feasible-path regret study on generated small graphs is archived."],
        ["P2 network-service study", "BLOCKED BY MODEL SCOPE",
         "Requires absolute key generation and QKP capacity, time-indexed demand, consumption/refill, blocking, load balance, controller message semantics, prediction baselines, and executable demand-level ILP."],
        ["P3 external validation", "BLOCKED BY DATA / HARDWARE",
         "Requires matched device/channel measurements or hardware-in-the-loop and separate model-error analysis."],
    ]
    status_table = Table([[paragraph(str(cell), styles["RoadCellHead"] if i == 0 else styles["RoadCell"])
                           for cell in row] for i, row in enumerate(status_rows)],
                         colWidths=[1.6 * inch, 1.18 * inch, 4.4 * inch], repeatRows=1)
    status_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), BURGUNDY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9D0D2")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story += [status_table, Spacer(1, 8), paragraph("4. Interpretation and next work", styles["RoadSection"]),
        paragraph("The corrected comparison supports a route-level simulator statement only. "
          "Report success, successful-route hops, revisits, action-mask violations, and route outcomes; "
          "do not infer network capacity or superiority from reward alone. Preserve baseline failures as "
          "observed outcomes and state each baseline definition.", styles["RoadBody"]),
        paragraph("Next priorities are five independent policies under a fixed training budget, followed "
          "by held-out endpoint/weather experiments and matched-budget ablations. The Johann-style "
          "demand/QKP study is a separate extension and depends on absolute key-resource units and a "
          "time-indexed demand model. Do not present Johann's BB84 equations as implemented until that "
          "model is adopted and validated.", styles["RoadBody"]),
        *([paragraph(f"Regression check: {test_report['passed']} tests passed, "
          f"{test_report['failed']} failed in {test_report['duration_seconds']:.1f} s; "
          f"{len(test_report.get('warnings', []))} warning(s) recorded. Details: "
          "<font name='Courier'>test_results_20261002.json</font>.", styles["RoadSmall"])]
          if test_report else []),
        Spacer(1, 4),
        paragraph("<b>Provenance rule:</b> Label every numeric result as measured, independently calculated, "
          "model-sampled, or derived from a policy run. Internal consistency checks do not upgrade an "
          "assumption to calibrated physics.", styles["RoadSmall"])]

    def page(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(BURGUNDY)
        canvas.setLineWidth(0.7)
        canvas.line(0.62 * inch, 0.48 * inch, 7.88 * inch, 0.48 * inch)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(DARK)
        canvas.drawString(0.62 * inch, 0.32 * inch,
                         "QKD Routing Research Roadmap - dated evidence/status addendum")
        canvas.drawRightString(7.88 * inch, 0.32 * inch, f"{doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(str(out_pdf), pagesize=letter, rightMargin=0.62 * inch,
        leftMargin=0.62 * inch, topMargin=0.55 * inch, bottomMargin=0.62 * inch,
        title="QKD Routing Research Roadmap Status Addendum",
        author="QKD routing research project").build(story, onFirstPage=page, onLaterPages=page)

    original_snapshot = ROOT / "paper" / "roadmap_original_20261002.pdf"
    if not original_snapshot.exists():
        shutil.copy2(args.roadmap, original_snapshot)
    base = original_snapshot if args.roadmap.resolve() == (ROOT /
        "QKD_Routing_Agent_Research_Roadmap_QSMS_Style.pdf").resolve() else args.roadmap
    writer = PdfWriter()
    for source in (base, out_pdf):
        reader = PdfReader(str(source))
        for page_obj in reader.pages:
            writer.add_page(page_obj)
    temp = args.roadmap.with_suffix(".updated.tmp.pdf")
    with temp.open("wb") as stream:
        writer.write(stream)
    shutil.move(str(temp), str(args.roadmap))
    print(json.dumps({"roadmap_pdf": str(args.roadmap),
        "original_pages": len(PdfReader(str(base)).pages),
        "addendum_pages": len(PdfReader(str(out_pdf)).pages),
        "total_pages": len(PdfReader(str(args.roadmap)).pages),
        "comparison": str(args.comparison)}, indent=2))


if __name__ == "__main__":
    main()
