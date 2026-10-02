"""Regenerate figures and tables from the versioned QKD project artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
SUP = OUT / "supplementary"
import sys
sys.path.insert(0, str(ROOT))
FIG.mkdir(exist_ok=True)
SUP.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 9})


def architecture():
    fig, ax = plt.subplots(figsize=(7.1, 2.3))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    boxes = [
        (0.025, .35, .19, .38, "Topology + link profiles", "fiber, FSO, corridor\ngeometry, device inputs"),
        (.275, .35, .19, .38, "Time-stepped environment", "channel sampling, constraints,\nroute and key-pool state"),
        (.525, .35, .19, .38, "Routing policy", "GNN encoder, masked\naction head, PPO update"),
        (.775, .35, .19, .38, "Evaluation", "baselines, metrics,\nseeded reports"),
    ]
    for x, y, w, h, title, desc in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012",fc="#edf4fa",ec="#244a67",lw=1.0))
        ax.text(x+w/2,y+h*.68,title,ha="center",va="center",weight="bold",fontsize=8)
        ax.text(x+w/2,y+h*.34,desc,ha="center",va="center",fontsize=7)
    for x1,x2 in [(.215,.275),(.465,.525),(.715,.775)]:
        ax.add_patch(FancyArrowPatch((x1,.54),(x2,.54),arrowstyle="-|>",mutation_scale=10,color="#244a67",lw=1.1))
    ax.add_patch(FancyArrowPatch((.62,.32),(.37,.32),connectionstyle="arc3,rad=-.35",arrowstyle="-|>",mutation_scale=9,color="#b35a32",lw=1,linestyle="--"))
    ax.text(.495,.08,"observation / reward",ha="center",color="#8f4528",fontsize=7)
    ax.add_patch(FancyArrowPatch((.87,.32),(.13,.32),connectionstyle="arc3,rad=-.55",arrowstyle="-|>",mutation_scale=9,color="#666",lw=.9,linestyle="--"))
    ax.text(.5,.91,"versioned YAML configuration and fixed experiment seeds",ha="center",fontsize=7,color="#444")
    fig.tight_layout(pad=.2); fig.savefig(FIG/"architecture.pdf",bbox_inches="tight"); plt.close(fig)


def qkd_stack():
    fig, ax = plt.subplots(figsize=(7.0, 3.3)); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")
    rows=[("Application",.80,"request / delivered key"),
          ("Control plane",.62,"route policy: local observation or full-state comparator"),
          ("Key management",.44,"key pools and trusted relay abstraction"),
          ("QKD link layer",.26,"neighbor QKD modules, TN / STN roles"),
          ("Optical channels",.08,"fiber links and candidate FSO links")]
    colors=["#f3f0fa","#e5eff8","#edf4ef","#fff2e4","#f5eeee"]
    for (label,y,desc),color in zip(rows,colors):
        ax.add_patch(FancyBboxPatch((.08,y),.84,.12,boxstyle="round,pad=.008",fc=color,ec="#42566a",lw=.8))
        ax.text(.13,y+.075,label,ha="left",va="center",weight="bold",fontsize=8)
        ax.text(.44,y+.075,desc,ha="left",va="center",fontsize=7.5)
    for y1,y2 in [(.80,.74),(.62,.56),(.44,.38),(.26,.20)]:
        ax.add_patch(FancyArrowPatch((.5,y1),(.5,y2),arrowstyle="-|>",mutation_scale=8,color="#34495e"))
    ax.annotate("modeled simulator components",xy=(.96,.5),xytext=(.99,.5),ha="left",va="center",rotation=90,fontsize=7,color="#555")
    ax.text(.5,.98,"Conceptual QKD network stack and routing-control placement",ha="center",va="top",fontsize=9,weight="bold")
    fig.tight_layout(); fig.savefig(FIG/"qkd_stack.pdf",bbox_inches="tight"); plt.close(fig)


def meshed_key_relay_architecture():
    """Project-specific mesh and key-relay schematic, inspired by QKD relay diagrams."""
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(7.0, 3.25))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    ax.text(.34, .96, "Hybrid meshed QKD network", ha="center", va="top",
            fontsize=9, weight="bold")
    pos = {"A": (.07, .55), "T1": (.27, .78), "S1": (.47, .55),
           "T2": (.27, .28), "D": (.66, .55)}
    # Alternate candidate paths are light gray. The red path is illustrative.
    alternatives = [("A", "T2", "fiber"), ("T2", "D", "fiber"),
                    ("T1", "D", "fso"), ("T2", "S1", "fso")]
    for u, v, kind in alternatives:
        style = ":" if kind == "fso" else "-"
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="#aeb8c2", lw=1.1, ls=style, zorder=1)
    selected = [("A", "T1", r"$K_{AT_1}$"),
                ("T1", "S1", r"$K_{T_1S_1}$"),
                ("S1", "D", r"$K_{S_1D}$")]
    for u, v, key in selected:
        a, b = pos[u], pos[v]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#b43c35", lw=2.0,
                zorder=2)
        mx, my = (a[0]+b[0])/2, (a[1]+b[1])/2
        ax.text(mx, my+.045, key, ha="center", va="center", fontsize=7,
                color="#9b2929", bbox={"fc":"white", "ec":"none", "pad":.5}, zorder=3)
    node_specs = {
        "A": ("Alice", "#2865a5", "endpoint"),
        "T1": ("Full TN", "#bf4b42", "trusted"),
        "S1": ("STN", "#7651a1", "stn"),
        "T2": ("Full TN", "#bf4b42", "trusted"),
        "D": ("Bob", "#2865a5", "endpoint"),
    }
    for node, (label, color, role) in node_specs.items():
        x, y = pos[node]
        ax.add_patch(Circle((x, y), .045, fc=color, ec="white", lw=1.0, zorder=4))
        ax.text(x, y-.075, label, ha="center", va="top", fontsize=7, weight="bold")
    ax.plot([], [], color="#b43c35", lw=2, label="illustrative route")
    ax.plot([], [], color="#aeb8c2", lw=1.1, label="alternate candidate links")
    ax.plot([], [], color="#aeb8c2", lw=1.1, ls=":", label="candidate FSO link")
    ax.legend(loc="lower left", bbox_to_anchor=(.01, .025), ncol=1,
              frameon=False, fontsize=5.9, handlelength=1.5,
              labelspacing=.25, borderaxespad=0)

    # The simulated policy is centralized over its graph observation. The
    # diagram marks the controller boundary without implying a deployed SDN.
    ctrl = FancyBboxPatch((.75, .52), .22, .30,
                          boxstyle="round,pad=.012", fc="#eaf1f7",
                          ec="#35566f", lw=1.0, zorder=2)
    ax.add_patch(ctrl)
    ax.text(.86, .75, "Routing agent", ha="center", va="center",
            fontsize=8, weight="bold")
    ax.text(.86, .65, "GNN + PPO", ha="center", va="center", fontsize=7.5)
    ax.text(.86, .56, "graph state, link features,\nvalid-action mask",
            ha="center", va="center", fontsize=6.4)
    ax.add_patch(FancyArrowPatch((.70, .62), (.75, .62), arrowstyle="<|-|>",
                                 mutation_scale=7, color="#35566f", lw=.9))
    ax.text(.86, .42, "Chooses a valid\nnext hop",
            ha="center", va="center", fontsize=6.4, color="#34495e")
    ax.text(.26, .12,
            "Key relay: pairwise keys are relayed hop by hop through trusted nodes.",
            ha="left", va="center", fontsize=6.2)
    ax.text(.26, .055,
            "Model: normalized pools only; no operational OTP/XOR key-transfer plane.",
            ha="left", va="center", fontsize=6.0, color="#8a4f35")
    fig.tight_layout(pad=.15)
    fig.savefig(FIG/"meshed_key_relay_architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def tn_stn_relay():
    fig,axs=plt.subplots(2,1,figsize=(7,3.5))
    for ax,title,stn in [(axs[0],"Conventional trusted-node chain",False),(axs[1],"Simplified trusted-node chain",True)]:
        ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off"); ax.set_title(title,fontsize=9,pad=0)
        xs=[.10,.36,.64,.90]; names=["Alice","TN 1","TN 2","Bob"]
        for x,n in zip(xs,names):
            ax.add_patch(FancyBboxPatch((x-.065,.52),.13,.22,boxstyle="round,pad=.01",fc="#eaf0f7",ec="#34495e"))
            ax.text(x,.63,n,ha="center",va="center",fontsize=8,weight="bold")
        for i in range(3):
            ax.annotate("pairwise QKD",xy=((xs[i]+xs[i+1])/2,.64),ha="center",va="bottom",fontsize=6.5)
            ax.add_patch(FancyArrowPatch((xs[i]+.07,.55),(xs[i+1]-.07,.55),arrowstyle="<->",mutation_scale=7,color="#2f7592",lw=1))
        if stn:
            ax.text(.50,.33,"STNs announce raw-key parities; Alice and Bob perform final EC and PA",ha="center",fontsize=7.5)
            ax.annotate("authenticated parity announcements",xy=(.50,.40),xytext=(.50,.40),ha="center",fontsize=6.5,color="#985b2c")
        else:
            ax.text(.50,.33,"Each TN relays key material and is trusted with the keys it handles",ha="center",fontsize=7.5)
            ax.text(.50,.20,"Link-level post-processing is performed on each adjacent QKD link",ha="center",fontsize=6.5,color="#555")
    fig.tight_layout(h_pad=.4); fig.savefig(FIG/"tn_stn_relay.pdf",bbox_inches="tight"); plt.close(fig)


def simulation_flow():
    fig,ax=plt.subplots(figsize=(4.6,4.2)); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")
    nodes=[(.50,.88,"Load versioned scenario and graph"),(.50,.70,"Reset route, channel, and pool state"),
           (.50,.52,"Policy or baseline selects next hop"),(.50,.34,"Apply QBER, SKR, trust, and pool checks"),
           (.50,.16,"Update route, reward, and metrics")]
    for x,y,text in nodes:
        ax.add_patch(FancyBboxPatch((.22,y-.055),.56,.11,boxstyle="round,pad=.012",fc="#edf4fa",ec="#34566f"))
        ax.text(x,y,text,ha="center",va="center",fontsize=7.5)
    for (_,y1,_),(_,y2,_) in zip(nodes,nodes[1:]):
        ax.add_patch(FancyArrowPatch((.50,y1-.06),(.50,y2+.06),arrowstyle="-|>",mutation_scale=8,color="#34566f"))
    ax.text(.50,.055,"Repeat until destination or episode termination; aggregate seeded evaluations",ha="center",fontsize=6.5)
    ax.add_patch(FancyArrowPatch((.78,.34),(.82,.70),connectionstyle="arc3,rad=.48",arrowstyle="-|>",mutation_scale=8,color="#a45b3a",lw=.9))
    fig.tight_layout(); fig.savefig(FIG/"simulation_flow.pdf",bbox_inches="tight"); plt.close(fig)


def policy_architecture():
    fig,ax=plt.subplots(figsize=(7.0,2.8)); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")
    def box(x,y,w,h,title,desc,fc="#edf4fa"):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.01",fc=fc,ec="#34566f",lw=.8))
        ax.text(x+w/2,y+h*.68,title,ha="center",va="center",fontsize=7.5,weight="bold")
        ax.text(x+w/2,y+h*.31,desc,ha="center",va="center",fontsize=6.5)
    box(.02,.60,.16,.25,"Observation","node features\nedge features + mask")
    box(.24,.67,.20,.22,"GNN encoder","3 message-passing\nlayers")
    box(.24,.30,.20,.22,"LSTM encoder","node-array sequence\ngraph-blind comparator",fc="#f8f1e9")
    box(.51,.60,.18,.25,"Candidate scoring","current, destination,\nneighbor + edge data")
    box(.76,.68,.20,.18,"Actor","masked next-hop\nprobabilities")
    box(.76,.40,.20,.18,"Critic","state value")
    for a,b in [((.18,.72),(.24,.78)),((.18,.68),(.24,.41)),((.44,.78),(.51,.72)),((.44,.41),(.51,.66)),((.69,.72),(.76,.77)),((.69,.66),(.76,.49))]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=8,color="#34566f",lw=.9))
    ax.text(.50,.14,"PPO updates actor and critic from masked rollouts; invalid links remain unavailable",ha="center",fontsize=7.2,color="#555")
    fig.tight_layout(); fig.savefig(FIG/"gnn_ppo.pdf",bbox_inches="tight"); plt.close(fig)


def physics_to_policy():
    fig,ax=plt.subplots(figsize=(7.0,2.35)); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")
    blocks=[(.02,.58,.16,.24,"Scenario inputs","season, time,\nlink distance"),
            (.23,.58,.18,.24,"Channel model","fiber loss or FSO\nlog-normal $C_n^2$"),
            (.46,.58,.16,.24,"Link estimates","transmission, detector\nnoise, QBER"),
            (.67,.58,.14,.24,"Rate model","asymptotic proxy or\nfinite-key estimator"),
            (.86,.58,.12,.24,"Policy","features +\nhard mask")]
    for x,y,w,h,title,desc in blocks:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.01",fc="#edf4fa",ec="#36566e",lw=.8))
        ax.text(x+w/2,y+h*.68,title,ha="center",va="center",fontsize=7.2,weight="bold")
        ax.text(x+w/2,y+h*.29,desc,ha="center",va="center",fontsize=6.2)
    for (x1,w1),(x2,_) in zip([(b[0],b[2]) for b in blocks],[(b[0],b[2]) for b in blocks][1:]):
        ax.add_patch(FancyArrowPatch((x1+w1,.70),(x2,.70),arrowstyle="-|>",mutation_scale=7,color="#36566e",lw=.8))
    ax.text(.50,.32,"Current FSO model: correlated log-normal turbulence, atmospheric and pointing terms, background and detector noise",ha="center",fontsize=7,color="#555")
    ax.text(.50,.17,"No Gamma-Gamma irradiance model or generalized amplitude-damping channel is implemented",ha="center",fontsize=7.2,color="#a14f2d",weight="bold")
    fig.tight_layout(); fig.savefig(FIG/"physics_to_policy.pdf",bbox_inches="tight"); plt.close(fig)


def rate_distance():
    from physics import sample_link_state
    import random
    fig,axs=plt.subplots(1,2,figsize=(6.8,2.65))
    for ax,link_type,distances,season,hour,title in [
        (axs[0],"fiber",range(5,106,5),"normal",22,"Fiber, 1550 nm"),
        (axs[1],"fso",range(2,31,2),"monsoon",22,"FSO, 785 nm, monsoon night")]:
        med=[]; lo=[]; hi=[]
        for distance in distances:
            values=[]
            for seed in range(120):
                state=sample_link_state(link_type,float(distance),season,hour,random.Random(seed+distance*100),fiber_outage_prob=0.0,outage_uniform=0.0)
                if not state.get("outage",False): values.append(float(state["skr"]))
            if not values: values=[0.0]
            med.append(float(__import__('numpy').median(values)))
            lo.append(float(__import__('numpy').quantile(values,.05)))
            hi.append(float(__import__('numpy').quantile(values,.95)))
        ax.plot(list(distances),med,color="#226b8b",lw=1.4)
        ax.fill_between(list(distances),lo,hi,color="#73a6bd",alpha=.25,label="5th to 95th percentile")
        ax.set_title(title); ax.set_xlabel("Modeled link distance (km)"); ax.grid(alpha=.2)
        ax.legend(frameon=False,fontsize=6.5)
    axs[0].set_ylabel("Normalized rate proxy")
    fig.tight_layout(); fig.savefig(FIG/"rate_distance.png",dpi=300,bbox_inches="tight"); plt.close(fig)


def corridor_map():
    import sys
    from matplotlib.collections import LineCollection
    sys.path.insert(0,str(ROOT))
    from topology import CITY_COORDS, build_topology
    graph=build_topology()
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    fiber=[]; fso=[]
    for u,v,attrs in graph.edges(data=True):
        if attrs["link_type"] == "fiber":
            geo=attrs.get("geometry") or [graph.nodes[u]["pos"],graph.nodes[v]["pos"]]
            fiber.append([(lon,lat) for lat,lon in geo])
        else:
            p1=graph.nodes[u]["pos"]; p2=graph.nodes[v]["pos"]
            fso.append([(p1[1],p1[0]),(p2[1],p2[0])])
    ax.add_collection(LineCollection(fiber,colors="#465f78",linewidths=.72,alpha=.58,zorder=1,label="Fiber"))
    ax.add_collection(LineCollection(fso,colors="#d45519",linewidths=.42,alpha=.48,zorder=2,label="Candidate FSO"))
    hubs={name:(coord[1],coord[0]) for name,coord in CITY_COORDS.items()}
    for city,(lon,lat) in hubs.items():
        ax.scatter(lon,lat,s=26,color="#a54632",edgecolor="white",linewidth=.55,zorder=3)
        ax.annotate(city,(lon,lat),xytext=(3,3),textcoords="offset points",fontsize=7,zorder=4)
    ax.scatter([],[],s=22,color="#a54632",edgecolor="white",label="City access node")
    ax.add_patch(FancyArrowPatch(hubs["Mumbai"],hubs["Kolkata"],connectionstyle="arc3,rad=-.18",
                                 arrowstyle="-|>",mutation_scale=9,color="#7b3f91",lw=1.2,
                                 linestyle="--",zorder=5))
    ax.plot([],[],color="#7b3f91",lw=1.2,ls="--",label="Example request: Mumbai to Kolkata")
    ax.text(.02,.02,f"{graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} links\nRoad geometry is a corridor proxy",transform=ax.transAxes,fontsize=6.5,va="bottom",bbox={"fc":"white","ec":"none","alpha":.8})
    ax.set_xlabel("Longitude (degrees east)"); ax.set_ylabel("Latitude (degrees north)")
    ax.set_title("Modeled India corridor graph")
    ax.autoscale(); ax.grid(alpha=.2); ax.set_aspect(1.25); ax.legend(loc="upper left",frameon=True,fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG/"corridor_map.png",dpi=300,bbox_inches="tight"); plt.close(fig)


def model_summary():
    d=json.loads((ROOT/"experiments/physics_profile_report_idq.json").read_text(encoding="utf-8"))
    fiber=d["fiber"]["80"]; fso=d["fso"]["monsoon_22h"]
    vals=[fiber["normalized_skr_proxy"],fso["normalized_skr_proxy_median_conditional_on_samples"]]
    labels=["80 km fiber\n1550 nm","10 km FSO\nmonsoon, 22:00"]
    fig,axs=plt.subplots(1,2,figsize=(6.7,2.5))
    axs[0].bar(labels,vals,color=["#386b8a","#c17a3d"])
    axs[0].set_ylabel("Normalized rate proxy"); axs[0].set_title("Conditional model output")
    axs[0].grid(axis="y",alpha=.2)
    av=[d["fso"][k]["empirical_availability"]*100 for k in ["normal_22h","summer_22h","winter_22h","monsoon_22h"]]
    axs[1].bar(["Normal","Summer","Winter","Monsoon"],av,color="#619184")
    axs[1].set_ylim(0,100); axs[1].set_ylabel("Sampled availability (%)"); axs[1].set_title("10 km FSO scenario")
    axs[1].grid(axis="y",alpha=.2)
    fig.tight_layout(); fig.savefig(FIG/"link_model_summary.png",dpi=300,bbox_inches="tight"); plt.close(fig)


def comparison_table():
    d=json.loads((ROOT/"experiments/defence_monsoon_night_200ep_mumbai_kolkata/comparison.json").read_text(encoding="utf-8"))
    with (SUP/"preliminary_comparison.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["policy","success_rate","episodes","mean_hops_success","reward_median","reward_std","checked_edges","valid_edges"])
        for name, item in d["comparison"].items():
            monsoon=item["per_season"]["monsoon"]
            q=item.get("qber_skr_validity",{})
            w.writerow([name,item["success_rate"],item["episodes"],item["avg_hops_success"],item["overall_reward"],item.get("seed_reward_std"),q.get("checked_edges"),q.get("valid_edges")])


def adverse_scenarios():
    d=json.loads((ROOT/"experiments/adverse_conditions_comparison.json").read_text(encoding="utf-8"))
    keys=list(d["scenarios"])
    labels=[k.replace("_","\n") for k in keys]
    fig,ax=plt.subplots(figsize=(6.8,2.8))
    x=__import__('numpy').arange(len(keys)); width=.34
    colors={"GNN-200ep":"#356f91","BFS-hop":"#bf7544"}
    for offset,method in [(-width/2,"GNN-200ep"),(width/2,"BFS-hop")]:
        means=[]; stds=[]
        for key in keys:
            runs=d["scenarios"][key][method]["per_season"]
            hops=[r["hops"] for group in runs.values() for r in group["seed_rewards"] if r["success"]]
            means.append(float(__import__('numpy').mean(hops)) if hops else 0)
            stds.append(float(__import__('numpy').std(hops)) if hops else 0)
        ax.bar(x+offset,means,width,yerr=stds,capsize=2,color=colors[method],label=method)
    ax.set_xticks(x,labels,fontsize=6.4); ax.set_ylabel("Successful route hops")
    ax.set_title("Exploratory adverse-scenario comparison")
    ax.grid(axis="y",alpha=.2); ax.legend(frameon=False,ncol=2,fontsize=7)
    fig.tight_layout(); fig.savefig(FIG/"adverse_route_hops.png",dpi=300,bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    architecture(); qkd_stack(); meshed_key_relay_architecture(); tn_stn_relay(); simulation_flow(); policy_architecture(); physics_to_policy(); rate_distance()
    corridor_map(); model_summary(); comparison_table(); adverse_scenarios()
