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
            "Candidate-edge vector: QBER · normalized rate · FSO flag · normalized distance · projected chain error · pool state · link switch · visited flag · destination progress",
            ha="center", va="center", fontsize=6.65, color=INK, wrap=True)
    ax.text(3.58, 0.16,
            "The rate is a simulator utility; scenario weather and availability are modeled inputs, not field measurements.",
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


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    physical_pipeline()
    graph_policy()
