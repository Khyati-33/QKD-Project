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
    parser.add_argument("--training-seeds", type=Path, default=ROOT / "paper" / "supplementary" /
                        "training_seed_stability_20261002.json")
    parser.add_argument("--training-progress", type=Path, default=ROOT / "paper" / "supplementary" /
                        "training_seed_stability_progress_20261002.json")
    parser.add_argument("--oracle-report", type=Path, default=ROOT / "paper" / "supplementary" /
                        "small_graph_oracle_20261002.json")
    parser.add_argument("--transfer-report", type=Path, default=ROOT / "paper" / "supplementary" /
                        "pair_season_matrix_20261002.json")
    parser.add_argument("--inference-profile", type=Path, default=ROOT / "paper" / "supplementary" /
                        "inference_profile_20261002.json")
    parser.add_argument("--ablation-report", type=Path, default=ROOT / "paper" / "supplementary" /
                        "matched_budget_ablations_20261002.json")
    parser.add_argument("--strict-report", type=Path, default=ROOT / "experiments" /
                        "final_holdout_20261007.json")
    parser.add_argument("--reward-audit", type=Path, default=ROOT / "experiments" /
                        "final_reward_audit_20261007" / "audit_summary.json")
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
    training_report = read_json(args.training_seeds) if args.training_seeds.exists() else None
    training_progress = read_json(args.training_progress) if args.training_progress.exists() else None
    oracle_report = read_json(args.oracle_report) if args.oracle_report.exists() else None
    transfer_report = read_json(args.transfer_report) if args.transfer_report.exists() else None
    inference_profile = read_json(args.inference_profile) if args.inference_profile.exists() else None
    ablation_report = read_json(args.ablation_report) if args.ablation_report.exists() else None
    strict_report = read_json(args.strict_report) if args.strict_report.exists() else None
    reward_audit = read_json(args.reward_audit) if args.reward_audit.exists() else None
    strict_complete = bool(strict_report and strict_report.get("status") == "complete")
    final_training_complete = strict_complete and len(
        strict_report.get("trained_checkpoint_manifest", [])) == 10
    ablation_progress_files = sorted((ROOT / "experiments" / "runs").glob(
        "matched_ablations_*/progress.json"), key=lambda path: path.stat().st_mtime)
    ablation_progress = read_json(ablation_progress_files[-1]) if ablation_progress_files else None
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
    cn2_sensitivity = (sensitivity or {}).get("controlled_input_sensitivity", {}).get(
        "fso_turbulence_cn2_multiplier_sweep", [])
    sensitivity_result = "The controlled diagnostic includes a proxy-rate ratio sweep, a feasible QBER-margin sweep, and an outage-mask counterfactual with route context held fixed."
    if rate_sensitivity and qber_sensitivity:
        sensitivity_result += (f" The FSO action probability rose from "
            f"{rate_sensitivity[0]['fso_probability']:.3f} to "
            f"{rate_sensitivity[-1]['fso_probability']:.3f} as its proxy-rate ratio rose "
            f"from {rate_sensitivity[0]['fso_to_fiber_rate_ratio']:.2g}x to "
            f"{rate_sensitivity[-1]['fso_to_fiber_rate_ratio']:.2g}x; at QBER 0.111 it was masked.")
    if cn2_sensitivity:
        sensitivity_result += (f" In a separate 32-sample FSO turbulence counterfactual, a 100x Cn2 "
            f"multiplier lowered median proxy from {cn2_sensitivity[2]['median_skr_proxy']:.3f} "
            f"to {cn2_sensitivity[-1]['median_skr_proxy']:.3f} and mean FSO action probability "
            f"from {cn2_sensitivity[2]['mean_fso_probability']:.3f} to "
            f"{cn2_sensitivity[-1]['mean_fso_probability']:.3f}; FSO remained selected in "
            f"{cn2_sensitivity[-1]['fso_selected_rate']:.0%} of samples.")
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
    if final_training_complete:
        seed_rates = strict_report["training_seed_uncertainty_success_rate"]
        nested = strict_report["nested_training_and_scenario_bootstrap_success_rate_ci95"]
        gnn_ci = nested["GNN-PPO"]["ci95"]
        mlp_ci = nested["MLP-PPO"]["ci95"]
        work_rows.append(["Independent training-seed stability",
            f"Five independent checkpoints per architecture were completed. GNN mean success "
            f"{seed_rates['GNN-PPO']['mean']:.1%} (SD {seed_rates['GNN-PPO']['std_sample']:.1%}; "
            f"nested 95% interval {gnn_ci[0]:.1%}-{gnn_ci[1]:.1%}); MLP mean success "
            f"{seed_rates['MLP-PPO']['mean']:.1%} (SD {seed_rates['MLP-PPO']['std_sample']:.1%}; "
            f"nested interval {mlp_ci[0]:.1%}-{mlp_ci[1]:.1%}).",
            "Variability is across fitted policies and scenario clusters in this simulator protocol; "
            "it is not external validation."])
    elif training_report and training_report.get("status") == "complete" and \
            training_report.get("training_seed_count") == 5:
        gnn = training_report["aggregate_across_independent_training_seeds"]["GNN-PPO"]
        success = gnn["success_rate"]
        hops = gnn["avg_hops_success"]
        work_rows.append(["Independent training-seed stability",
            f"{training_report['training_seed_count']} independent policies trained for "
            f"{training_report['fixed_training_budget']['ppo_epochs']} PPO epochs each. "
            f"Mean paired-set success rate {success['mean_across_training_seeds']:.1%} "
            f"(training-seed SD {success['sample_sd_across_training_seeds']:.1%}); "
            f"successful-route mean hops "
            f"{hops['mean_across_training_seeds']:.2f} "
            f"(SD {hops['sample_sd_across_training_seeds']:.2f}).",
            "Interval summarizes variability across fitted policies on one fixed simulator protocol; "
            "it is not an episode-level confidence interval or external validation."])
    elif training_progress:
        completed = training_progress.get("completed", [])
        pending = training_progress.get("not_yet_complete", [])
        active = pending[0]["training_seed"] if pending else "none"
        progress_text = (f"{len(completed)}/{len(training_progress['planned_seeds'])} independent "
                         f"policies have final evaluations. Seed {active} is the latest run without "
                         "a final summary.")
        if completed:
            observed = completed[0]
            progress_text += (f" Completed seed {observed['training_seed']}: "
                f"{observed['success_rate']:.1%} success, "
                f"{observed['avg_hops_success']:.2f} mean hops among successful episodes.")
        work_rows.append(["Five-seed stability campaign", progress_text,
            "Inter-seed spread and confidence summaries are withheld until all planned seeds finish. "
            "A training-seed interval is distinct from an episode-level interval."])
    if oracle_report:
        oracle_summary = oracle_report["summary"]
        work_rows.append(["Exact toy-graph route oracle",
            f"Exhaustive simple-path reward search covered {oracle_summary['episodes']} graph/channel "
            f"cases with a feasible oracle in {oracle_summary['with_feasible_oracle']}. The policy "
            f"succeeded in {oracle_summary['policy_success_rate']:.0%}; it exactly matched oracle "
            f"reward in {oracle_summary['policy_reward_optimal_fraction']:.0%} of all cases and "
            f"{oracle_summary['successful_policy_reward_optimal_fraction']:.0%} of successful cases.",
            "Five generated 7-node graphs; episode link states frozen with 5% per-edge outage. "
            "This exact result is limited to the toy static-channel protocol, not the full dynamic graph."])
    if transfer_report:
        gnn_transfer = transfer_report["summary_by_method"]["GNN-PPO"]
        bfs_transfer = transfer_report["summary_by_method"]["BFS-hop"]
        transfer_ci = gnn_transfer["success_rate_wilson_95pct"]
        paired_transfer = transfer_report.get("paired_analysis", {}).get("GNN-PPO_vs_BFS-hop")
        transfer_hops = ("n/a" if gnn_transfer["successful_mean_hops"] is None else
                         f"{gnn_transfer['successful_mean_hops']:.2f}")
        paired_text = (f" Paired exact McNemar p={paired_transfer['exact_mcnemar_two_sided_p']:.4f}; "
            f"among jointly successful episodes, GNN-minus-BFS hop difference was "
            f"{paired_transfer['mean_paired_hop_difference']:.2f} "
            f"(95% pair-cluster bootstrap CI "
            f"{paired_transfer['hop_difference_95pct_pair_cluster_bootstrap_ci'][0]:.2f} to "
            f"{paired_transfer['hop_difference_95pct_pair_cluster_bootstrap_ci'][1]:.2f}). "
            if paired_transfer else " GNN did not outperform BFS.")
        work_rows.append(["Paired endpoint / season / time transfer",
            f"Across {gnn_transfer['episodes']} GNN episodes, success was "
            f"{gnn_transfer['successes']}/{gnn_transfer['episodes']} "
            f"({gnn_transfer['success_rate']:.1%}; Wilson 95% CI "
            f"{transfer_ci[0]:.1%}-{transfer_ci[1]:.1%}); successful mean hops "
            f"{transfer_hops}. BFS-hop success was "
            f"{bfs_transfer['successes']}/{bfs_transfer['episodes']} "
            f"({bfs_transfer['success_rate']:.1%}), with "
            f"{bfs_transfer['successful_mean_hops']:.2f} successful mean hops. "
            + paired_text + "GNN did not outperform BFS.",
            "Six ordered pairs, four seasons and selected day/night times, paired seeds. "
            "BC endpoint randomization means these are not strict end-to-end endpoint holdouts. "
            "Hop interval resamples only six endpoint-pair clusters."])
    if inference_profile:
        thread_keys = sorted(inference_profile["summary_by_threads"], key=int)
        latency_text = "; ".join(
            f"{thread_key} thread(s): p50/p95/p99 "
            f"{inference_profile['summary_by_threads'][thread_key]['p50_ms']:.2f}/"
            f"{inference_profile['summary_by_threads'][thread_key]['p95_ms']:.2f}/"
            f"{inference_profile['summary_by_threads'][thread_key]['p99_ms']:.2f} ms"
            for thread_key in thread_keys)
        rss = inference_profile["memory"]["process_rss_after_bytes"] / (1024 * 1024)
        model_mb = inference_profile["memory"]["model_parameter_bytes"] / (1024 * 1024)
        work_rows.append(["Repeated inference profile",
            f"{latency_text}. Process RSS after run {rss:.1f} MiB; model parameters {model_mb:.1f} MiB.",
            "42 ordered pairs, deterministic single-decision calls, encoder cache disabled. "
            "One host/topology; graph-size scaling and deployment profiling remain open."])
    if ablation_progress:
        completed_variants = ablation_progress.get("completed_variants", [])
        ablation_state = ablation_progress.get("training_state") or {}
        running_detail = (f" Current condition: {ablation_progress.get('running_variant')}, "
            f"{ablation_state.get('phase', 'initializing')} phase, PPO "
            f"{ablation_state.get('ppo_epoch', 0)}/{ablation_state.get('target_ppo_epochs', 50)}."
            if ablation_progress.get("running_variant") else "")
        work_rows.append(["Matched-budget ablation campaign",
            f"{len(completed_variants)}/{len(ablation_progress.get('planned_variants', []))} "
            "conditions complete at the shared 50-epoch budget." + running_detail,
            "Single training seed; progress is not a completed comparative result."])
    if strict_report and strict_report.get("status") == "complete":
        seed_rates = strict_report["training_seed_uncertainty_success_rate"]
        nested = strict_report["nested_training_and_scenario_bootstrap_success_rate_ci95"]
        bfs_rate = strict_report["baseline_summary"]["BFS-hop"]["success_rate"]
        gnn = seed_rates["GNN-PPO"]
        mlp = seed_rates["MLP-PPO"]
        gnn_ci = nested["GNN-PPO"]["ci95"]
        mlp_ci = nested["MLP-PPO"]["ci95"]
        def mean_hops(method):
            values = [value["successful_mean_hops"] for key, value in
                strict_report["summary_by_method_training_seed"].items()
                if key.startswith(method + "|") and value["successful_mean_hops"] is not None]
            return (sum(values) / len(values)) if values else None
        gnn_hops, mlp_hops = mean_hops("GNN-PPO"), mean_hops("MLP-PPO")
        outage_level = max(strict_report["fiber_outage_stress_levels"])
        outage_rates = {}
        for method in ("GNN-PPO", "MLP-PPO"):
            values = [value["success_rate"] for key, value in
                strict_report["fiber_outage_stress_summary_by_training_seed"].items()
                if key.startswith(f"{method}|outage={outage_level:.2f}|")]
            outage_rates[method] = sum(values) / len(values) if values else None
        work_rows.append(["Strict matched held-out evaluation",
            f"Five GNN and five MLP policies were evaluated on the same 108 held-out scenarios. "
            f"Training-seed mean success: GNN {gnn['mean']:.1%} (SD {gnn['std_sample']:.1%}; "
            f"nested 95% interval {gnn_ci[0]:.1%}-{gnn_ci[1]:.1%}), MLP "
            f"{mlp['mean']:.1%} (SD {mlp['std_sample']:.1%}; nested interval "
            f"{mlp_ci[0]:.1%}-{mlp_ci[1]:.1%}), BFS {bfs_rate:.1%}. "
            f"Mean successful hops across seeds: GNN {gnn_hops:.2f}, MLP {mlp_hops:.2f}. "
            f"At {outage_level:.0%} fiber outage, GNN/MLP mean success was "
            f"{outage_rates['GNN-PPO']:.1%}/{outage_rates['MLP-PPO']:.1%}.",
            "Paired scenario IDs; five fitted policies per learned architecture; intervals resample "
            "training seeds outside scenario clusters. Simulator results, not field validation."])
    if reward_audit and reward_audit.get("status") == "complete":
        audit_summary = reward_audit["reward_term_summary"]
        work_rows.append(["Final-checkpoint reward decomposition",
            f"One final GNN seed and its matched MLP seed were decomposed over "
            f"{reward_audit['scenario_count_per_policy']} scenarios against BFS-hop. "
            f"Route success: GNN {audit_summary['GNN-PPO']['success_rate']:.1%}, "
            f"MLP {audit_summary['MLP-PPO']['success_rate']:.1%}, "
            f"BFS {audit_summary['BFS-hop']['success_rate']:.1%}.",
            "Reward components are descriptive shaped-return contributions; route metrics remain "
            "the primary comparison and this one-seed audit does not estimate seed uncertainty."])
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
        ["P0-4 QBER stress", "PARTIAL",
         "Controlled action-probability sweep covers QBER 0.01, 0.03, 0.05, 0.07, 0.09, 0.10, 0.109 and 0.111. The out-of-threshold candidate is masked. Route success, entropy, and preference curves across endpoint/weather conditions remain."],
        ["P0-5 FSO / outage stress", "DONE - paired route stress" if strict_complete else "PARTIAL",
          (f"Paired route outcomes are archived at outage probabilities "
           f"{', '.join(f'{p:.0%}' for p in strict_report['fiber_outage_stress_levels'])}. "
           "These are simulator stress scenarios; pointing/background parameters remain fixed and "
           "outage-persistence bursts are not separately calibrated." if strict_complete else
           "A 0.01x-100x Cn2 input sweep, season/hour quality samples, configured marginal availability, and an outage-mask counterfactual exist. Route-level outage outcomes are pending; pointing/background parameters are fixed and outage-persistence bursts are unsupported.")],
        ["P0-6 five training seeds", "DONE - five GNN + five MLP" if final_training_complete else ("DONE - fixed protocol" if training_report and
          training_report.get("status") == "complete" and training_report.get("training_seed_count") == 5
          else (f"IN PROGRESS - {training_progress.get('completed_seed_count', 0)}/"
                f"{len(training_progress.get('planned_seeds', []))} complete" if training_progress else "IN PROGRESS")),
          (f"Five independent policies, each trained for "
           f"{training_report['fixed_training_budget']['ppo_epochs']} PPO epochs and evaluated on "
           "the same held-out simulator seeds. Results vary across fitted policies; this does not "
           "establish endpoint/weather generalization." if training_report and
           training_report.get("status") == "complete" and training_report.get("training_seed_count") == 5 else
           ("Five GNN and five matched MLP full-budget runs are complete and their checkpoint hashes are recorded in the strict evaluation report." if final_training_complete else
            "Five independent full-budget training runs are not present. Evaluation episodes for one checkpoint do not substitute for training-seed replication."))],
         ["P1 endpoint / weather holdouts", "DONE - strict paired test" if strict_complete else ("PARTIAL - transfer matrix" if transfer_report else "PARTIAL"),
           (f"Final report covers {strict_report['unique_scenario_count']} paired held-out endpoint/condition/scenario IDs, "
              "five trained seeds each for GNN and MLP, four fiber-outage levels, and per-checkpoint CPU latency. "
              "See the final report and manuscript for estimates and uncertainty." if strict_complete else
              (f"Paired across-pair and season/time matrix is archived ({transfer_report['summary_by_method']['GNN-PPO']['episodes']} episodes per policy) with Wilson intervals. GNN success was {transfer_report['summary_by_method']['GNN-PPO']['success_rate']:.1%} versus BFS {transfer_report['summary_by_method']['BFS-hop']['success_rate']:.1%}; successful mean hops were {transfer_report['summary_by_method']['GNN-PPO']['successful_mean_hops']:.2f} versus {transfer_report['summary_by_method']['BFS-hop']['successful_mean_hops']:.2f}. BC endpoint randomization prevents a strict endpoint-holdout claim." if transfer_report else
               "The current checkpoint was run on 42 ordered pairs (31/42 success), plus earlier adverse-condition episodes. These are coverage diagnostics, not held-out endpoint/weather tests."))],
        ["P1 architecture / feature ablations",
          "DONE - single-seed scope" if ablation_report else "DEFERRED - outside current scope",
          ("Matched-budget results archive width, node-feature, DropEdge, and SKR reward sensitivities. "
           "Single-seed results do not quantify training-seed variability." if ablation_report else
           (f"An earlier campaign stopped during the baseline at PPO epoch "
            f"{(ablation_progress.get('training_state') or {}).get('ppo_epoch', 0)}/"
            f"{(ablation_progress.get('training_state') or {}).get('target_ppo_epochs', 50)}; "
            "no ablation comparison is used in the manuscript and the campaign is deferred." if ablation_progress else
            "No matched-budget GNN feature, architecture, or reward ablation set is used in the manuscript."))],
         ["P1 inference cost", "DONE - one-host protocol" if strict_complete else ("PARTIAL - profiled" if inference_profile else "PARTIAL"),
          ("Per-decision latency was measured for all ten final checkpoints with warm-up, four CPU threads, and caching disabled. Results characterize one host/topology only; graph-size scaling and deployment profiling remain open." if strict_complete else
           ("Repeated per-decision CPU latency, p50/p95/p99, thread counts, host details, and process/model memory are archived across 42 ordered pairs. This remains a one-host, one-topology simulator profile; no graph-size scaling or deployment claim." if inference_profile else
            "Current checkpoint: 42-pair local inference had 3.83 ms median and 4.34 ms p95 decision latency. It is one host/scenario; report tail latency, memory, and scaling before deployment claims."))],
        ["P1 exact small-graph oracle", "DONE - toy scope" if oracle_report else "TODO",
         ("Exhaustive simple-path reward oracle and policy regret archived for generated 7-node graphs with frozen episode channels. "
          "Does not establish optimality on the full dynamic network." if oracle_report else
          "No exact feasible-path regret study on generated small graphs is archived.")],
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
    next_work = ("The strict five-seed endpoint/weather and paired outage evaluation, selected-checkpoint "
        "reward decomposition, final figures, and revised manuscript are complete. Before submission, "
        "the authors should fill the author/funding block and verify the final package in their target "
        "venue template. The demand/QKP service-capacity study and external device/weather validation "
        "remain deferred extensions outside this completion." if strict_complete else
        "Complete the five-seed training and strict held-out endpoint/weather evaluation before making "
        "final comparative claims. Then update the manuscript and evidence package. The demand/QKP "
        "service-capacity study and external device/weather validation remain separate extensions.")
    story += [status_table, Spacer(1, 8), paragraph("4. Interpretation and next work", styles["RoadSection"]),
        paragraph("The corrected comparison supports a route-level simulator statement only. "
          "Report success, successful-route hops, revisits, action-mask violations, and route outcomes; "
          "do not infer network capacity or superiority from reward alone. Preserve baseline failures as "
          "observed outcomes and state each baseline definition.", styles["RoadBody"]),
        paragraph(next_work + " Do not present the demand-level BB84/QKP formulation as implemented "
          "until an absolute key-resource and demand model is adopted and validated.", styles["RoadBody"]),
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
