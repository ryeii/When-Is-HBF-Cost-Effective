"""Build the preprint's vector figures from the released experiment tables."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

BLUE = "#1C6590"
OCHRE = "#94611F"
INK = "#20262D"
GRAY = "#626B72"
GRID = "#DCE1E5"
LIGHT = "#F0F2F4"
NEAR = "#DCEBF3"
FLASH = "#F2E2C7"
ARCHES = ("baseline", "shared_site", "h3_cascaded")
DENSE_MODELS = (
    ("qwen3_8b", "Qwen3-8B"), ("qwen3_32b", "Qwen3-32B"),
    ("llama31_70b", "Llama 3.1-70B"), ("llama31_405b", "Llama 3.1-405B"),
)
MOE_MODELS = (
    ("qwen3_30b_a3b", "Qwen3-30B-A3B"), ("qwen3_235b_a22b", "Qwen3-235B-A22B"),
    ("deepseek_v3", "DeepSeek V3"), ("kimi_k2", "Kimi K2"),
    ("muse_spark_open_816b_a42b", "Muse Spark Open"),
)
MODELS = DENSE_MODELS + MOE_MODELS
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": GRAY, "axes.linewidth": .5,
    "xtick.color": INK, "ytick.color": INK, "xtick.major.size": 2.4,
    "ytick.major.size": 2.4, "pdf.fonttype": 42, "savefig.facecolor": "white",
})


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.pdf")
    plt.close(fig)


def workflow(directory):
    fig = plt.figure(figsize=(250/72, 99/72))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 250), ylim=(0, 99))
    ax.set_axis_off()

    for x, y, width, title, detail, edge, fill in (
        (21, 74, 208, "Simulate across the test matrix",
         "Scenario × layout combinations (Table I)", GRAY, LIGHT),
        (21, 40, 208, "Keep traces that satisfies at least relaxed SLO",
         "", GRAY, LIGHT),
        (1, 1, 120, "For each demand",
         "compare layout cost", BLUE, NEAR),
        (129, 1, 120, "For each GPU count",
         "compare layout throughput", OCHRE, FLASH),
    ):
        ax.add_patch(Rectangle((x, y), width, 23, facecolor=fill,
                               edgecolor=edge, linewidth=.65))
        ax.text(x+width/2, y+(16 if detail else 11.5), title, ha="center", va="center",
                fontweight="bold", color=INK if edge == GRAY else edge)
        ax.text(x+width/2, y+7, detail, ha="center", va="center", fontsize=7)

    ax.plot([125, 125], [40, 32], color=GRAY, linewidth=.75)
    ax.plot([61, 189], [32, 32], color=GRAY, linewidth=.75)
    for x, top, bottom in ((125, 74, 63), (61, 32, 24), (189, 32, 24)):
        ax.add_patch(FancyArrowPatch((x, top), (x, bottom), arrowstyle="-|>",
                                    shrinkA=0, shrinkB=1, mutation_scale=7,
                                    linewidth=.75, color=GRAY))
    save(fig, directory, "workflow")


def decision_examples(designs, directory):
    fig = plt.figure(figsize=(3.48, 2.62))
    cost = fig.add_axes([.18, .60, .80, .27])
    rate = fig.add_axes([.18, .16, .80, .27])
    counts = (1, 2, 4, 8, 16, 32)
    loads = (1, 4, 16)
    offsets = dict(zip(ARCHES, (-.17, 0, .17)))
    costs = {}
    rates = {}
    for d in designs:
        if d["gpu"] != "rtx4090":
            continue
        n, view, arch = d["n"], d["view"], d["architecture"]
        if (d["model"], d["workload"], view) == ("qwen3_235b_a22b", "short_chat", "fixed_users"):
            if d["metrics"]["balanced"]["feasible"]:
                value = n + d["installed_stacks"]  # One GPU acquisition cost per stack.
                costs[arch, n] = min(costs.get((arch, n), value), value)
        if (d["model"], d["workload"], n) == ("kimi_k2", "offline_batch", 100) and view != "fixed_users":
            result = d["metrics"]["offline"]
            if result["feasible"]:
                load = int(view.rsplit("_", 1)[1])
                value = result["good_tokens_per_second"]
                rates[arch, load] = max(rates.get((arch, load), value), value)

    for ax, keys, values, xlabel, ylabel, maximum, ticks in (
        (cost, counts, costs, "Installed GPUs", "Cost (GPU units)", 106, (0, 25, 50, 75, 100)),
        (rate, loads, rates, "Users per GPU", "Output tokens/s", 765, (0, 200, 400, 600)),
    ):
        for arch, color, marker, label in zip(
            ARCHES, (GRAY, OCHRE, BLUE), ("o", "s", "D"),
            ("Conventional", "Shared site", "Cascaded"),
        ):
            for i, key in enumerate(keys):
                value = values.get((arch, key))
                ax.plot(i+offsets[arch], 0 if value is None else value,
                        marker="x" if value is None else marker,
                        markersize=3.5, markeredgewidth=.85, color=color,
                        linestyle="none", clip_on=False)
            if ax is cost:
                ax.plot([], [], marker=marker, color=color, linestyle="none", markersize=3.5, label=label)
        ax.set(xlim=(-.42, len(keys)-.62), ylim=(-.06*maximum, maximum),
               xticks=range(len(keys)), xticklabels=keys, yticks=ticks)
        ax.set_xlabel(xlabel, labelpad=1)
        ax.set_ylabel(ylabel, labelpad=2)
        ax.tick_params(labelsize=7, pad=1.5)
        ax.grid(axis="y", color=GRID, linewidth=.5)
        ax.spines[["top", "right"]].set_visible(False)

    cost.set_title("(a) Fix demand: 4 balanced chat users", loc="left", fontweight="bold", fontsize=7.7)
    rate.set_title("(b) Fix GPU count: 100 GPUs, offline jobs", loc="left", fontweight="bold", fontsize=7.7)
    cost_arch, cost_n = min(costs, key=costs.get)
    rate_arch, rate_load = max(rates, key=rates.get)
    for ax, x, y, label, text_xy in (
        (cost, counts.index(cost_n)+offsets[cost_arch], costs[cost_arch, cost_n],
         f"{costs[cost_arch, cost_n]:g} min. cost", (.23, 52)),
        (rate, loads.index(rate_load)+offsets[rate_arch], rates[rate_arch, rate_load],
         f"{rates[rate_arch, rate_load]:.1f} max. rate", (.97, 700)),
    ):
        ax.plot(x, y, marker="o", markersize=8, markerfacecolor="none",
                markeredgecolor=BLUE, markeredgewidth=1, clip_on=False)
        ax.annotate(label, (x, y), text_xy, fontsize=7, color=BLUE, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=BLUE, linewidth=.65, shrinkB=4))
    cost.plot([], [], marker="x", color=GRAY, linestyle="none", markersize=3.5, label="No pass")
    handles, labels = cost.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.55, 1.01),
               frameon=False, ncol=4, fontsize=7, handletextpad=.25, columnspacing=.75)
    save(fig, directory, "decision-examples")


def minimum_matrix(rows, directory):
    keys = {(r["model"], r["gpu"], r["architecture"]): r for r in rows
            if (r["workload"], r["slo"]) == ("short_chat", "balanced")}
    fig = plt.figure(figsize=(3.48, 2.06))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 7.97), ylim=(0, 11.6))
    ax.set_axis_off()
    for j, (gpu, label) in enumerate((("rtx4090", "RTX 4090 x N"), ("h200", "H200 x N"))):
        x0 = 2.45 + j*2.72
        ax.text(x0+1.29, 11.15, label, ha="center", va="center", fontweight="bold")
        for i, name in enumerate(("Conv.", "Shared", "Casc.")):
            ax.text(x0+(i+.5)*.86, 10.34, name, ha="center", va="center", fontsize=7.2)
        for row, (model, label) in enumerate(MODELS):
            y = 9.61-row*.91
            if j == 0:
                ax.text(.10, y, label, va="center", fontsize=7.15)
            base = keys[model, gpu, "shared_site"]["baseline_n"]
            vals = [base] + [keys[model, gpu, a]["hbf_n"] for a in ARCHES[1:]]
            for i, n in enumerate(vals):
                if not n:
                    color, ink = LIGHT, GRAY
                elif i and not base:
                    color, ink = FLASH, OCHRE
                elif i and int(n) < int(base):
                    color, ink = NEAR, BLUE
                else:
                    color, ink = "white", INK
                x = x0+i*.86
                ax.add_patch(Rectangle((x, y-.425), .86, .85, facecolor=color,
                                       edgecolor=GRID, linewidth=.45))
                ax.text(x+.43, y, str(n) if n else "–", ha="center", va="center",
                        color=ink, fontweight="bold" if color in (FLASH, NEAR) else "normal")
    ax.text(.10, .83, "Blue  fewer GPUs", color=BLUE, va="center", fontweight="bold")
    ax.text(2.53, .83, "Ochre  HBF-only pass", color=OCHRE, va="center", fontweight="bold")
    ax.text(5.58, .83, "–  no passing design", color=GRAY, va="center", fontsize=7.1)
    save(fig, directory, "minimum-gpus")


def purchase(rows, prices, directory):
    margins = Counter((r["workload"], int(r["baseline_n"])-int(r["hbf_n"])) for r in rows
                      if r["classification"] == "positive_headroom_within_pool")
    totals = Counter(r["workload"] for r in rows)
    savings = Counter()
    for r in prices:
        net = r["net_savings_gpu_units"]
        if net not in (None, "") and float(net) > 1e-9:
            savings[float(r["premium_gpu_units_per_stack"])] += 1
    assert margins.total() == savings[0] == 98
    fig = plt.figure(figsize=(3.48, 1.55))
    a = fig.add_axes([.145, .32, .30, .53])
    b = fig.add_axes([.575, .23, .40, .62])
    workloads = (
        ("paused_personal", "Paused"),
        ("short_chat", "Chat"),
        ("long_rag", "RAG"),
        ("tool_agent", "Agent"),
        ("offline_batch", "Offline"),
    )
    order = [key for key, _ in workloads]
    left = [0]*len(order)
    lower = 1
    for upper, label, color in (
        (4, "1–4", "#A9CFE2"), (8, "5–8", "#609EBC"),
        (16, "9–16", BLUE), (float("inf"), "17+", "#163C56"),
    ):
        widths = [sum(count for (workload, margin), count in margins.items()
                      if workload == key and lower <= margin <= upper) for key in order]
        a.barh(range(len(order)), widths, left=left, color=color, edgecolor="white",
               linewidth=.25, height=.58, label=label)
        left = [prior+width for prior, width in zip(left, widths)]
        lower = upper+1
    for i, (key, wins) in enumerate(zip(order, left)):
        a.text(wins+1.1, i, f"{wins}/{totals[key]}", va="center", fontsize=7)
    a.set(xlim=(0, 62), ylim=(4.6, -.6), xticks=[], yticks=range(5),
          yticklabels=[label for _, label in workloads])
    a.tick_params(length=0)
    for spine in a.spines.values():
        spine.set_visible(False)
    a.set_title("(a) Fewer GPUs", loc="left", fontsize=7.7, pad=5, fontweight="bold")
    a.legend(loc="upper left", bbox_to_anchor=(-.45, -.05), ncol=4, frameon=False,
             fontsize=6.5, title="GPUs saved per win", title_fontsize=6.5,
             handlelength=1, handletextpad=.3, columnspacing=.7, borderaxespad=0)
    x = (0, .25, .5, 1, 2, 4)
    retained = {p: 100*savings[p]/savings[0] for p in x}
    b.plot(x, [retained[p] for p in x], color=BLUE, marker="o", markersize=2.6, linewidth=1)
    for p, offset, align in ((0, (4, 2), "left"), (.5, (4, 3), "left"),
                             (1, (4, 3), "left"), (4, (-2, 5), "right")):
        b.annotate(f"{retained[p]:.0f}% ({savings[p]})", (p, retained[p]),
                   xytext=offset, textcoords="offset points", ha=align,
                   fontsize=6.5, color=BLUE)
    b.set(xlim=(-.15, 4.15), ylim=(-4, 114), xticks=(0, .5, 1, 2, 4),
          xticklabels=("0", "0.5", "1", "2", "4"),
          yticks=(0, 25, 50, 75, 100), yticklabels=("0%", "25%", "50%", "75%", "100%"))
    b.grid(axis="y", color=GRID, linewidth=.5)
    b.tick_params(labelsize=6.8, pad=1.5)
    for side in ("top", "right"):
        b.spines[side].set_visible(False)
    b.set_xlabel("GPU costs / stack", fontsize=7, labelpad=1)
    b.set_title("(b) Cost wins retained", loc="left", fontsize=7.7, pad=5, fontweight="bold")
    save(fig, directory, "cost-workloads")


def single_user(designs, directory):
    slos = ("responsive", "balanced", "relaxed")
    passing = {}
    for r in designs:
        if r["n"] != 1 or r["view"] != "users_per_gpu_1" or r["workload"] != "short_chat":
            continue
        for slo in slos:
            if r["metrics"][slo]["feasible"]:
                passing.setdefault((r["gpu"], slo, r["model"]), set()).add(r["architecture"])
    fig = plt.figure(figsize=(3.48, 2.06))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 7.97), ylim=(0, 11.6))
    ax.set_axis_off()
    for j, (gpu, label) in enumerate((("rtx4090", "RTX 4090 x 1"), ("h200", "H200 x 1"))):
        x0 = 2.45 + j*2.72
        ax.text(x0+1.29, 11.15, label, ha="center", va="center", fontweight="bold")
        for i, label in enumerate(("Resp.", "Bal.", "Relax.")):
            ax.text(x0+(i+.5)*.86, 10.34, label, ha="center", va="center", fontsize=7.2)
        for row, (model, label) in enumerate(MODELS):
            y = 9.61-row*.91
            if j == 0:
                ax.text(.10, y, label, va="center", fontsize=7.15)
            for i, slo in enumerate(slos):
                options = passing.get((gpu, slo, model), set())
                marks = " ".join(mark for arch, mark in zip(ARCHES, ("B", "S", "C")) if arch in options)
                if not options:
                    color, ink = LIGHT, GRAY
                elif "baseline" not in options:
                    color, ink = FLASH, OCHRE
                else:
                    color, ink = "white", INK
                x = x0+i*.86
                ax.add_patch(Rectangle((x, y-.425), .86, .85, facecolor=color,
                                       edgecolor=GRID, linewidth=.45))
                ax.text(x+.43, y, marks or "–", ha="center", va="center", color=ink,
                        fontweight="bold" if color == FLASH else "normal", fontsize=7.1)
    ax.text(.10, .83, "Ochre  HBF-only pass", color=OCHRE, va="center", fontweight="bold")
    ax.text(3.08, .83, "–  no passing design", color=GRAY, va="center", fontsize=7.1)
    save(fig, directory, "single-user")


def scale(rows, directory):
    lookup = {}
    for r in rows:
        val = r["tested_useful_good_tokens_per_second"]
        lookup[r["model"], r["gpu"], int(r["n"]), r["workload"], r["slo"], r["architecture"]] = (
            float(val) if val not in ("", None) else None)
    sizes = (1, 2, 4, 8, 16, 32, 64, 72, 100)
    fig = plt.figure(figsize=(3.48, 4.65))
    top_axes = [fig.add_axes([x, .796, .38, .136]) for x in (.145, .595)]
    workloads = (
        ("short_chat", "balanced", "Balanced chat", BLUE, "o"),
        ("offline_batch", "offline", "Offline", OCHRE, "s"),
    )
    chat_at_100 = Counter()
    for top, (family, models) in zip(top_axes, (("Dense", DENSE_MODELS), ("MoE", MOE_MODELS))):
        for workload, slo, label, color, marker in workloads:
            rates = []
            for n in sizes:
                counts = Counter()
                for model, _ in models:
                    for gpu in ("rtx4090", "h200"):
                        k = model, gpu, n, workload, slo
                        base = lookup[k + ("baseline",)]
                        for arch in ARCHES[1:]:
                            value = lookup[k + (arch,)]
                            if base and value:
                                ratio = value/base
                                outcome = "tie" if abs(ratio-1) <= 1e-6 else "lower" if ratio < 1 else "higher"
                                counts[outcome] += 1
                rates.append(100*counts["higher"]/counts.total() if counts else float("nan"))
                if workload == "short_chat" and n == 100:
                    chat_at_100.update(counts)
            top.plot(range(len(sizes)), rates, color=color, marker=marker, markersize=2.4,
                     linewidth=.9, label=label)
        top.set(xlim=(-.4, 8.5), ylim=(-4, 104), xticks=range(len(sizes)), xticklabels=sizes,
                yticks=(0, 25, 50, 75, 100))
        top.tick_params(axis="x", labelsize=6, pad=1)
        top.tick_params(axis="y", labelsize=7, pad=1.5)
        top.grid(axis="y", color=GRID, linewidth=.5)
        top.spines[["top", "right"]].set_visible(False)
        top.set_title(family, fontsize=7.5, fontweight="bold", pad=3)
    assert chat_at_100 == {"tie": 16, "lower": 9}
    top_axes[0].set_ylabel("HBF wins (%)", labelpad=2)
    top_axes[1].tick_params(axis="y", labelleft=False)
    handles, labels = top_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.56, 1),
               frameon=False, ncol=2, fontsize=6.8)
    fig.text(.03, .978, "(a)", va="center", fontweight="bold")
    fig.text(.56, .736, "Installed GPUs", ha="center", fontsize=7)
    fig.text(.03, .705, "(b)", va="center", fontweight="bold")
    for y, n in ((.406, 100), (.067, 1)):
        for x, gpu, title in ((.328, "rtx4090", "4090"), (.669, "h200", "H200")):
            a = fig.add_axes([x, y, .308, .283])
            for i, (model, _) in enumerate(MODELS):
                for offset, (workload, slo, _, color, marker) in zip((-.23, .23), workloads):
                    k = model, gpu, n, workload, slo
                    base, cascaded = (lookup[k + (arch,)] for arch in ("baseline", "h3_cascaded"))
                    row = i+offset
                    if base is not None and cascaded is not None:
                        a.hlines(row, min(base, cascaded), max(base, cascaded),
                                 color=color, linewidth=.75, alpha=.65)
                    if base is not None:
                        a.scatter(base, row, s=15, marker=marker, facecolors="white",
                                  edgecolors=color, linewidth=.7, zorder=3)
                    if cascaded is not None:
                        a.scatter(cascaded, row, s=6, marker=marker, color=color,
                                  linewidth=0, zorder=4)
                    if base and cascaded is not None and cascaded >= 2*base:
                        a.annotate(f"{cascaded/base:.1f}×", (cascaded, row),
                                   xytext=(3, 0), textcoords="offset points",
                                   va="center", fontsize=6, color=color, zorder=5)
            a.set(xscale="log", xlim=(.45, 160000), ylim=(len(MODELS)-.5, -.5),
                  yticks=range(len(MODELS)), xticks=(1, 10, 100, 1000, 10000, 100000),
                  xticklabels=("1", "10", "100", "1k", "", "100k"))
            a.set_yticklabels([name for _, name in MODELS] if gpu == "rtx4090" else [])
            a.minorticks_off()
            a.grid(axis="x", color=GRID, linewidth=.5)
            a.hlines([i+.5 for i in range(len(MODELS)-1)], .45, 160000,
                     color=GRID, linewidth=.35)
            a.tick_params(axis="x", labelsize=6, pad=1.5)
            a.tick_params(axis="y", labelsize=7, length=0, pad=2.5)
            a.spines[["top", "right", "left"]].set_visible(False)
            a.set_title(f"{title} x {n}", fontsize=7.5, fontweight="bold", pad=3)
    fig.text(.65, .012, "Useful output tokens/s (log scale)", ha="center", fontsize=7)
    save(fig, directory, "scale")


def sensitivity_audit(rows, directory):
    wins = defaultdict(set)
    for r in rows:
        if r["classification"] == "positive":
            key = tuple(r[field] for field in ("model", "gpu", "workload", "slo", "architecture"))
            wins[r["scenario"]].add(key)
    reference = wins["central_seed17"]
    settings = (
        ("link_64GBs", "64 GB/s HBF link"),
        ("combined_adverse", "Combined adverse"),
        ("hbf_adverse", "Slower HBF and GPU"),
        ("link_256GBs", "256 GB/s HBF link"),
        ("conventional_fast", "Faster conventional service"),
        ("moe_skewed", "Skewed MoE routing"),
        ("thermal_10W", "10 W per HBF stack"),
        ("program_slow", "Slower flash programming"),
        ("program_fast", "Faster flash programming"),
    )
    retained = [reference & wins[key] for key, _ in settings]
    assert len(reference) == 14

    fig = plt.figure(figsize=(3.48, 1.74))
    ax = fig.add_axes([.46, .055, .53, .865])
    positions = range(len(settings))
    counts = [len(group) for group in retained]
    ax.barh(positions, [len(reference)]*len(settings), height=.55, color=LIGHT)
    ax.barh(positions, counts, height=.55, color=BLUE)
    for i, count in enumerate(counts):
        ax.text(count+.35, i, f"{count}/{len(reference)}", fontsize=7, va="center")
    ax.set(xlim=(0, 17.7), ylim=(len(settings)-.5, -.5), xticks=[],
           yticks=positions, yticklabels=[label for _, label in settings])
    ax.tick_params(axis="y", length=0, pad=4, labelsize=7)
    ax.spines[:].set_visible(False)
    fig.text(.46, .95, "Reference wins retained", fontsize=7.5, fontweight="bold")
    save(fig, directory, "sensitivity-audit")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "evidence")
    parser.add_argument("--out", type=Path, default=root / "runs/figures")
    args = parser.parse_args()
    path = args.results / "adoption"

    def read_json(file):
        if file.exists():
            return json.loads(file.read_text())
        with gzip.open(file.with_suffix(".json.gz"), "rt") as stream:
            return json.load(stream)

    def read(name):
        with (path / name).open() as f:
            return list(csv.DictReader(f))
    muse = read_json(args.results / "muse.json")
    fixed = read("minimum-frontiers.csv") + muse["fixed_demand_frontiers"]
    prices = read("price-curves.csv") + muse["price_curves"]
    throughput = read("scale-frontiers.csv") + muse["scale_frontiers"]
    assert len(fixed) == 468
    args.out.mkdir(parents=True, exist_ok=True)
    workflow(args.out)
    minimum_matrix(fixed, args.out)
    purchase(fixed, prices, args.out)
    original = read_json(path / "aggregate.json")
    decision_examples(original["designs"], args.out)
    single_user(original["designs"] + muse["paired_designs"], args.out)
    scale(throughput, args.out)
    with (path.parent / "uncertainty/sensitivity-frontiers.csv").open() as f:
        sensitivity_audit(list(csv.DictReader(f)), args.out)


if __name__ == "__main__":
    main()
