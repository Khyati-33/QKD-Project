"""Generate implementation-matched vector figures used by main.tex."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent / "figures"
NAVY = "#173B5E"
BLUE = "#DDEAF5"
TEAL = "#DCEFEA"
PALE = "#F4F6F8"
GOLD = "#F4E6C3"
INK = "#20303C"


def box(ax, xy, wh, title, body, *, fc=PALE, title_color=NAVY,
        fontsize=8.0, body_size=7.1, align="center"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.018,rounding_size=0.045",
        linewidth=0.9, edgecolor=NAVY, facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top",
            fontsize=fontsize, weight="bold", color=title_color)
    ax.text(x + (0.12 if align == "left" else w / 2), y + h - 0.36,
            body, ha=align, va="top", fontsize=body_size,
            color=INK, linespacing=1.25)


def arrow(ax, start, end, *, color=NAVY, lw=1.05, rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                 mutation_scale=9, linewidth=lw,
                                 color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


def physical_pipeline():
    fig, ax = plt.subplots(figsize=(7.16, 3.15))
    ax.set(xlim=(0, 7.16), ylim=(0, 3.15))
    ax.axis("off")
    ax.text(3.58, 3.02, "Physical channel and resource state enter the routing decision",
            ha="center", va="top", fontsize=11, weight="bold", color=NAVY)
    box(ax, (0.12, 1.84), (1.48, 0.84), "Fiber link",
        "$L$, $d$, $\\alpha$, detector\nefficiency, dark counts", fc=BLUE)
    box(ax, (0.12, 0.68), (1.48, 0.92), "FSO link",
        "$L$, season/time, correlated\n$C_n^2$, outage, pointing and\nbackground scenario terms", fc=TEAL)
    box(ax, (1.88, 1.20), (1.40, 1.02), "Link state",
        "$p_{\\rm det}$, QBER,\nnormalized rate proxy;\nFSO availability", fc=PALE)
    box(ax, (3.56, 1.20), (1.48, 1.02), "Route state",
        "normalized key-pool state,\nvisited history, link switch,\nchain-error projection", fc=GOLD)
    box(ax, (5.31, 1.20), (1.70, 1.02), "Constrained next hop",
        "graph observation → GNN\nscore over feasible neighbors\n→ selected adjacent node", fc=BLUE)
    arrow(ax, (1.60, 2.23), (1.87, 1.86))
    arrow(ax, (1.60, 1.14), (1.87, 1.56))
    arrow(ax, (3.28, 1.72), (3.55, 1.72))
    arrow(ax, (5.04, 1.72), (5.30, 1.72))
    ax.text(3.58, 0.53,
            "Candidate-edge vector: nine link and route features (see observation table)",
            ha="center", va="center", fontsize=7.2, color=INK)
    ax.text(3.58, 0.16,
            "Scenario channel and resource quantities are modeled inputs, not field measurements.",
            ha="center", va="center", fontsize=7.0, color="#56636D")
    fig.subplots_adjust(left=0.015, right=0.985, top=0.99, bottom=0.03)
    fig.savefig(OUT / "physical_state_pipeline.pdf", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def graph_policy():
    fig, ax = plt.subplots(figsize=(7.16, 3.55))
    ax.set(xlim=(0, 7.16), ylim=(0, 3.55))
    ax.axis("off")
    ax.text(3.58, 3.43, "GNN-PPO next-hop policy (features and constraints from the implementation)",
            ha="center", va="top", fontsize=10.4, weight="bold", color=NAVY)
    box(ax, (0.10, 1.72), (1.55, 1.27), "Graph observation",
        "Node (5): city, full-TN, STN,\ndestination-relative distance,\nnormalized degree\n\nGraph-edge attributes (4):\nQBER, rate, FSO flag, distance", fc=BLUE,
        body_size=6.5)
    box(ax, (1.91, 1.72), (1.42, 1.27), "Graph encoder",
        "Three message-passing\nlayers with learned\nmessage and update\nfunctions; invalid QBER/\nrate edges are excluded", fc=TEAL,
        body_size=6.8)
    box(ax, (3.60, 1.72), (1.48, 1.27), "Candidate scoring",
        "Current, destination and\nneighbor embeddings plus\n9 candidate-edge features\n\nAttention score per\nadjacent candidate", fc=PALE,
        body_size=6.8)
    box(ax, (5.34, 1.72), (1.68, 1.27), "Feasibility + actor",
        "$Q < Q_{\\max}$; rate $>0$;\nprojected chain error\n$< Q_{\\max}$; adjacent only\n\nMasked softmax selects\none feasible next hop", fc=GOLD,
        body_size=6.7)
    arrow(ax, (1.65, 2.34), (1.90, 2.34))
    arrow(ax, (3.33, 2.34), (3.59, 2.34))
    arrow(ax, (5.08, 2.34), (5.33, 2.34))
    box(ax, (1.08, 0.24), (2.19, 0.92), "PPO actor update",
        "Clipped policy objective; GAE;\nreward-normalized rollouts", fc=BLUE,
        body_size=6.8)
    box(ax, (3.70, 0.24), (2.19, 0.92), "Critic value estimate",
        "Mean graph embedding + current and\ndestination embeddings + two state scalars", fc=PALE,
        body_size=6.8)
    arrow(ax, (6.15, 1.70), (4.95, 1.18), color="#758796", rad=0.10)
    arrow(ax, (4.80, 1.18), (3.26, 0.78), color="#758796", rad=0.0)
    arrow(ax, (2.15, 1.18), (0.90, 1.65), color="#758796", rad=0.18)
    ax.text(3.58, 0.045, "A full trusted node clears the accumulated STN-chain QBER state. Episode ends at destination, security breach, pool depletion, no feasible action, or the step limit.",
            ha="center", va="bottom", fontsize=6.45, color="#56636D")
    fig.subplots_adjust(left=0.015, right=0.985, top=0.99, bottom=0.02)
    fig.savefig(OUT / "graph_policy_implementation.pdf", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def matched_architecture():
    """Matched GNN/LSTM encoder comparison and shared PPO decision path."""
    gnn = "#2878B5"
    lstm = "#D95F02"
    ink = "#263746"
    neutral = "#F4F6F8"
    fig, ax = plt.subplots(figsize=(7.16, 3.25))
    ax.set(xlim=(0, 7.16), ylim=(0, 3.25))
    ax.axis("off")
    ax.text(3.58, 3.15, "Matched encoders, shared constrained PPO decision path",
            ha="center", va="top", fontsize=10.2, weight="bold", color=ink)

    # Shared observation branches into the two encoder alternatives.
    box(ax, (0.12, 1.35), (1.25, 1.05), "Shared\nobservation",
        "node + graph state\ncurrent + destination\n9 candidate features",
        fc=neutral, fontsize=7.0, body_size=6.7)
    box(ax, (1.70, 1.90), (1.58, 0.82), "GNN encoder",
        "message passing over\nQBER/rate-valid edges",
        fc="#E7F0F7", title_color=gnn, body_size=7.2)
    box(ax, (1.70, 0.92), (1.58, 0.82), "LSTM encoder",
        "ordered node-feature rows;\nno edge-index input",
        fc="#FBEDE3", title_color=lstm, body_size=7.2)
    ax.text(2.49, 2.80, "GNN-PPO", ha="center", va="center", color=gnn,
            fontsize=8.0, weight="bold")
    ax.text(2.49, 0.73, "LSTM-PPO", ha="center", va="center", color=lstm,
            fontsize=8.0, weight="bold")
    arrow(ax, (1.37, 2.02), (1.68, 2.32), color=gnn, lw=1.25)
    arrow(ax, (1.37, 1.72), (1.68, 1.32), color=lstm, lw=1.25)

    box(ax, (3.56, 1.35), (1.23, 1.05), "Candidate\nscorer",
        "encoder embedding +\nroute-context features\nscore each adjacent node",
        fc=neutral, fontsize=7.0, body_size=6.6)
    arrow(ax, (3.28, 2.31), (3.54, 2.08), color=gnn, lw=1.25)
    arrow(ax, (3.28, 1.32), (3.54, 1.67), color=lstm, lw=1.25)

    box(ax, (5.03, 1.35), (1.05, 1.05), "Feasibility\nmask",
        "adjacent; QBER and\nchain error < threshold;\nrate > 0",
        fc="#F4E6C3", fontsize=7.0, body_size=6.35)
    arrow(ax, (4.79, 1.88), (5.01, 1.88))

    box(ax, (6.32, 1.92), (0.72, 0.66), "Actor",
        "masked\nsoftmax",
        fc="#E6EFF6", title_color=gnn, fontsize=7.4, body_size=6.7)
    box(ax, (6.32, 1.04), (0.72, 0.66), "Critic",
        "value\nestimate",
        fc=neutral, fontsize=7.4, body_size=6.7)
    arrow(ax, (6.08, 2.02), (6.30, 2.22))
    arrow(ax, (6.08, 1.72), (6.30, 1.37))
    ax.text(6.68, 2.88, "Selected feasible\nnext hop", ha="center", va="center",
            fontsize=7.0, color=ink)
    arrow(ax, (6.68, 2.59), (6.68, 2.70), color=ink)

    box(ax, (2.48, 0.05), (2.25, 0.46), "Shared PPO update",
        "clipped policy objective; value loss; entropy",
        fc=neutral, fontsize=7.5, body_size=6.5)
    arrow(ax, (6.67, 1.02), (4.73, 0.38), color="#758796", lw=0.9, rad=-0.08)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.03)
    fig.savefig(OUT / "matched_architecture.pdf", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    physical_pipeline()
    matched_architecture()
