"""Append the completed matched GNN-PPO/LSTM-PPO study to the project roadmap."""
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
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
BURGUNDY = colors.HexColor("#9E1735")
PALE = colors.HexColor("#F6F1F2")
DARK = colors.HexColor("#242424")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "experiments" /
                        "final_gnn_lstm_holdout_20261007.json")
    parser.add_argument("--reward-audit", type=Path, default=ROOT / "experiments" /
                        "final_gnn_lstm_reward_audit_20261007" / "audit_summary.json")
    parser.add_argument("--roadmap", type=Path,
                        default=ROOT / "QKD_Routing_Agent_Research_Roadmap_QSMS_Style.pdf")
    parser.add_argument("--output", type=Path, default=ROOT / "paper" /
                        "roadmap_gnn_lstm_update_20261007.pdf")
    args = parser.parse_args()
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    audit_path = args.reward_audit if args.reward_audit.is_absolute() else ROOT / args.reward_audit
    roadmap = args.roadmap if args.roadmap.is_absolute() else ROOT / args.roadmap
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise SystemExit("The matched GNN/LSTM evaluation is not complete; roadmap not updated.")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="RoadTitle2", parent=styles["Title"],
        fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=BURGUNDY,
        alignment=TA_LEFT, spaceAfter=3))
    styles.add(ParagraphStyle(name="RoadSubtitle2", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=DARK, spaceAfter=8))
    styles.add(ParagraphStyle(name="RoadBody2", parent=styles["BodyText"],
        fontName="Helvetica", fontSize=9, leading=12, textColor=DARK, spaceAfter=5))
    styles.add(ParagraphStyle(name="RoadCell2", parent=styles["BodyText"],
        fontName="Helvetica", fontSize=8, leading=10, textColor=DARK))
    styles.add(ParagraphStyle(name="RoadHead2", parent=styles["RoadCell2"],
        fontName="Helvetica-Bold", textColor=colors.white))

    summaries = report["summary"]
    intervals = report["nested_training_scenario_ci95"]
    paired = report["paired_success_difference"]
    rows = [["Measure", "GNN-PPO", "LSTM-PPO / paired result"]]
    for method, label in (("GNN-PPO", "GNN-PPO"), ("LSTM-PPO", "LSTM-PPO")):
        rates = report["per_training_seed"][method]
        rate_values = [float(row["success_rate"]) for row in rates.values()]
        hops = [float(row["successful_mean_hops"]) for row in rates.values()
                if row["successful_mean_hops"] is not None]
        rows.append([f"Route completion ({label})",
            f"{summaries[method]['successes']}/{summaries[method]['episodes']} "
            f"({summaries[method]['success_rate']:.1%}); nested 95% CI "
            f"{intervals[method]['ci95'][0]:.1%}-{intervals[method]['ci95'][1]:.1%}",
            f"Per-seed range {min(rate_values):.1%}-{max(rate_values):.1%}; "
            f"successful-hop checkpoint mean {sum(hops)/len(hops):.2f}" if hops else
            f"Per-seed range {min(rate_values):.1%}-{max(rate_values):.1%}; no successful routes"])
    rows.append(["Paired completion effect", "",
        f"LSTM minus GNN {paired['mean_difference']:+.1%}; nested 95% CI "
        f"{paired['ci95'][0]:+.1%} to {paired['ci95'][1]:+.1%}"])
    discordant = report.get("paired_outcome_counts", {})
    rows.append(["Matched outcomes", "",
        f"Both succeeded {discordant.get('both_success', 0)}; GNN-only success "
        f"{discordant.get('GNN_only_success', 0)}; LSTM-only success "
        f"{discordant.get('LSTM_only_success', 0)}; both failed "
        f"{discordant.get('both_fail', 0)}"])
    hop_effect = report.get("paired_hops_on_joint_success", {})
    if hop_effect.get("mean_LSTM_minus_GNN") is not None:
        rows.append(["Successful-route hops", "",
            f"Among {hop_effect['episodes']} episodes where both succeeded, mean LSTM-minus-GNN "
            f"difference {hop_effect['mean_LSTM_minus_GNN']:+.2f} hops"])
    latencies = report.get("inference_latency", {})
    for method in ("GNN-PPO", "LSTM-PPO"):
        arch = method.removesuffix("-PPO")
        values = [float(x["p95_ms"]) for key, x in latencies.items()
                  if key.startswith(arch + "|") and x.get("p95_ms") is not None]
        if values:
            rows.append([f"CPU actor-call p95 ({method})", f"{sum(values)/len(values):.3f} ms",
                         f"Across-checkpoint SD {__import__('statistics').stdev(values):.3f} ms"])

    story = [Paragraph("RESEARCH ROADMAP", styles["RoadTitle2"]),
        Paragraph(f"MATCHED GNN-PPO / LSTM-PPO UPDATE - {date.today().isoformat()}",
                  styles["RoadSubtitle2"]),
        Paragraph("The revised manuscript now compares only GNN-PPO with the repository's "
            "matched LSTM-PPO node-sequence encoder. Five checkpoints per architecture were "
            "evaluated on the same 108 scenario IDs per checkpoint (540 episodes per method). "
            "The table reports simulator results with uncertainty across fitted checkpoints and "
            "shared scenario clusters; it is not field validation.", styles["RoadBody2"])]
    table = Table([[Paragraph(str(cell), styles["RoadHead2"] if i == 0 else styles["RoadCell2"])
                    for cell in row] for i, row in enumerate(rows)],
                  colWidths=[1.42*inch, 2.6*inch, 3.2*inch], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), BURGUNDY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9D0D2")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [table, Spacer(1, 10), Paragraph("Interpretation and remaining scope",
        styles["Heading2"]), Paragraph("The paired bootstrap is the primary uncertainty summary. "
        "Successful-hop differences are conditional on both methods completing the same scenario. "
        "The LSTM reads node-feature rows in a fixed topology order, has no graph-edge input, and "
        "resets its state on every action call; the comparison is therefore between these two "
        "implementations, not graph memory versus persistent temporal memory. Reward-component "
        "totals are descriptive because trajectories can differ in length. The manuscript figures "
        "are constrained to the sections that interpret them, and the publication bundle excludes "
        "older route-baseline and GNN/MLP comparisons retained in the local research archive.",
        styles["RoadBody2"]),
        Paragraph("The study is simulator-only. Field channel calibration, absolute key-pool and "
        "multi-demand service modeling, network blocking/capacity, broader topology tests, and a "
        "formal composable-security proof remain outside the completed paper revision.",
        styles["RoadBody2"])]
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") == "complete":
            story.append(Paragraph("The selected-seed reward audit is complete on 108 matched "
                "scenarios. Its cumulative reward-component report and episode ledger are included "
                "in the supplementary package; the plot is a descriptive diagnostic only.",
                styles["RoadBody2"]))

    output.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(output), pagesize=letter, rightMargin=0.62*inch,
        leftMargin=0.62*inch, topMargin=0.55*inch, bottomMargin=0.62*inch,
        title="QKD Routing GNN-PPO/LSTM-PPO Roadmap Update",
        author="QKD routing research project").build(story)

    original = ROOT / "paper" / "roadmap_original_20261002.pdf"
    previous = ROOT / "paper" / "roadmap_status_addendum.pdf"
    writer = PdfWriter()
    sources = [original, previous, output]
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(source)
        for page in PdfReader(str(source)).pages:
            writer.add_page(page)
    temp = roadmap.with_suffix(".updated.tmp.pdf")
    with temp.open("wb") as stream:
        writer.write(stream)
    shutil.move(str(temp), str(roadmap))
    print(json.dumps({"output": str(output), "roadmap": str(roadmap),
        "pages": len(PdfReader(str(roadmap)).pages),
        "comparison": report["comparison"]}, indent=2))


if __name__ == "__main__":
    main()
