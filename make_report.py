"""
Builds the results report from the training logs: tables, figures, and the summary CSV.
Run it after any new training run:
    python make_report.py

Reads:
    results/<level>/log.txt     runs made by hand (practice mode), or log_part1*.txt + log_part2*.txt
                                for a run that got stuck and was resumed
    results/runs/*.log          runs made with run_batch.py (first line describes the run)
Writes:
    results/summary.csv         one line per run
    docs/figures/*.png          figures used by the README and docs/RESULTS.md
    docs/RESULTS.md             the part between the "results:start" / "results:end" markers
"""
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {
    "stereo_madness": "Stereo Madness", "back_on_track": "Back on Track", "polargeist": "Polargeist",
    "dry_out": "Dry Out", "base_after_base": "Base After Base", "cant_let_go": "Can't Let Go",
    "jumper": "Jumper", "time_machine": "Time Machine", "cycles": "Cycles", "xstep": "xStep",
    "clutterfunk": "Clutterfunk", "theory_of_everything": "Theory of Everything",
    "electroman_adventures": "Electroman Adventures", "clubstep": "Clubstep",
    "electrodynamix": "Electrodynamix", "hexagon_force": "Hexagon Force",
    "blast_processing": "Blast Processing", "theory_of_everything_2": "Theory of Everything 2",
    "geometrical_dominator": "Geometrical Dominator", "deadlocked": "Deadlocked",
    "fingerdash": "Fingerdash", "dash": "Dash",
}
ORDER = list(NAMES)

# Colours (validated categorical palette: blue, orange, aqua), text stays in neutral inks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
# Official difficulty (stars) of each level, for comparison with the AI's own ranking
STARS = {
    "stereo_madness": 1, "back_on_track": 2, "polargeist": 3, "dry_out": 4, "base_after_base": 5,
    "cant_let_go": 6, "jumper": 7, "time_machine": 8, "cycles": 9, "xstep": 10, "clutterfunk": 11,
    "theory_of_everything": 12, "electroman_adventures": 10, "clubstep": 14, "electrodynamix": 12,
    "hexagon_force": 12, "blast_processing": 10, "theory_of_everything_2": 14,
    "geometrical_dominator": 10, "deadlocked": 15, "fingerdash": 12, "dash": 12,
}

# Levels trained with the finer state (--obs fine) instead of the default one
FINE = {"clubstep", "deadlocked"}

# Levels trained in two parts (part 1 stuck or interrupted, then resumed): total time unknown
RESUMED = {
    "time_machine": "interrupted at 73.8% then resumed: total time unknown",
    "clutterfunk": "stuck at 15.5% (cube lost in the sky counted as alive): rule added, then resumed; "
                   "total time unknown",
    "clubstep": "stuck at 70.6% in a tight UFO corridor: the state was too coarse to tell close positions "
                "apart; resumed from the best run with a finer state (--obs fine); total time unknown",
    "theory_of_everything": "stuck at 63.2% by a bug in that rule (too strict in UFO mode): fixed, then "
                            "resumed; total time unknown",
}
MODE_COLOURS = {("practice", "default"): BLUE, ("normal", "default"): ORANGE}

EP_LINE = re.compile(r"^ep\.\s+(\d+).*\(best\s+([\d.]+)%\)")
CLOCK = re.compile(r"^ep\.\s+\d+\s+(?:(\d+)h)?(?:(\d+)m)?(\d+)s\s")   # elapsed time (newer logs)
BEATEN = re.compile(r"BEATEN after (\d+) episodes, (\d+) steps simulated, ([\d.]+)s")
LIMIT = re.compile(r"(Time limit reached|Stopped by the user) after (\d+) episodes")


def parse_log(path):
    text = open(path, encoding="utf-8", errors="ignore").read().lstrip("﻿")
    curve, timed, run = [], [], {"won": False, "episodes": None, "steps": None, "seconds": None,
                      "resumed": "Resumed agent" in text or "Learned the saved run" in text,
                      "set_aside": text.count("set aside")}
    for line in text.splitlines():
        m = EP_LINE.match(line.strip())
        if m:
            curve.append((int(m.group(1)), float(m.group(2))))
            t = CLOCK.match(line.strip())
            if t:
                h, mi, se = (int(g or 0) for g in t.groups())
                timed.append((3600 * h + 60 * mi + se, float(m.group(2))))
    m = BEATEN.search(text)
    if m:
        run.update(won=True, episodes=int(m.group(1)), steps=int(m.group(2)), seconds=float(m.group(3)))
    else:
        m = LIMIT.search(text)
        if m:
            run["episodes"] = int(m.group(2))
    run["best"] = 100.0 if run["won"] else max([b for _, b in curve] or [0.0])
    run["curve"] = run["curve_adj"] = curve
    run["curve_time"] = timed
    run["episodes_adj"] = run["episodes"]
    header = re.search(r"^RUN (.*)$", text, re.M)
    if header:
        run.update(dict(kv.split("=", 1) for kv in header.group(1).split()))
    return run


def collect():
    runs = {}
    # 1. Runs made by hand, one folder per level: results/<level>/log.txt, or log_part1*.txt +
    #    log_part2*.txt when the run got stuck (or was interrupted) and was resumed
    for key in NAMES:
        folder = os.path.join("results", key)
        single = os.path.join(folder, "log.txt")
        p1 = sorted(glob.glob(os.path.join(folder, "log_part1*.txt")))
        p2 = sorted(glob.glob(os.path.join(folder, "log_part2*.txt")))
        if os.path.exists(single):
            r = parse_log(single)
            r.update(level=key, mode="practice", variant="default", source=single,
                     note="finer state (--obs fine)" if key in FINE else "")
            runs[(key, "practice", "default")] = r
        elif p1 and p2:
            a, b = parse_log(p1[0]), parse_log(p2[0])
            last_ep = a["curve"][-1][0] if a["curve"] else 0
            curve = a["curve"] + [(e + last_ep, p) for e, p in b["curve"]]
            # For the difficulty ranking: when part 1 was stuck because of a bug, its final plateau
            # says nothing about the level. Drop it: part 1 up to its last record, then part 2.
            cut = last_ep
            if "stuck" in p1[0] and a["curve"]:
                top = max(p for _, p in a["curve"])
                cut = next(e for e, p in a["curve"] if p == top)
            curve_adj = [(e, p) for e, p in a["curve"] if e <= cut] + [(e + cut, p) for e, p in b["curve"]]
            b.update(level=key, mode="practice", variant="default", source=p2[0],
                     episodes=(b["episodes"] or 0) + last_ep, seconds=None, steps=None, curve=curve,
                     episodes_adj=(b["episodes"] or 0) + cut, curve_adj=curve_adj,
                     note=RESUMED.get(key, "trained in two parts: total time unknown"))
            runs[(key, "practice", "default")] = b
    # 2. Runs made with run_batch.py: they replace the hand-made ones
    for path in glob.glob(os.path.join("results", "runs", "*.log")):
        r = parse_log(path)
        if "level" not in r:
            continue
        r.update(source=path, note="time limit reached" if not r["won"] and r["episodes"] else "")
        runs[(r["level"], r["mode"], r["variant"])] = r
    return runs


def minutes(r):
    return None if r["seconds"] is None else r["seconds"] / 60


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_minutes(runs, path):
    configs = [c for c in [("practice", "default"), ("normal", "default")]
               if any((l, *c) in runs and runs[(l, *c)]["seconds"] for l in ORDER)]
    levels = [l for l in ORDER if any((l, *c) in runs and runs[(l, *c)]["seconds"] for c in configs)]
    if not levels:
        return False
    h = 0.8 / len(configs)
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(levels) * len(configs) + 1.4))
    for i, c in enumerate(configs):
        for j, l in enumerate(levels):
            r = runs.get((l, *c))
            if not r or r["seconds"] is None:
                continue
            y = len(levels) - 1 - j + (len(configs) - 1 - i) * h - (len(configs) - 1) * h / 2
            m = minutes(r)
            ax.barh(y, m, height=h * 0.9, color=MODE_COLOURS.get(c, AQUA),
                    label=f"{c[0]}" if j == 0 else None)
            ax.text(m + 0.15, y, f"{m:.1f} min" + ("" if r["won"] else f" ({r['best']:.0f}%)"),
                    va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels([NAMES[l] for l in reversed(levels)], color=INK)
    ax.set_xlabel("Training time to beat the level (minutes)", color=INK2)
    style(ax)
    if len(configs) > 1:
        ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def fig_curves(runs, path, mode="practice", variant="default"):
    levels = [l for l in ORDER if (l, mode, variant) in runs and runs[(l, mode, variant)]["curve"]
              and not runs[(l, mode, variant)].get("note", "").startswith("development")]
    if not levels:
        return False
    cols = 3
    rows = (len(levels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.3 * rows), sharey=True, squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k >= len(levels):
            ax.axis("off")
            continue
        r = runs[(levels[k], mode, variant)]
        best, xs, ys = 0.0, [], []
        for e, p in r["curve"]:
            best = max(best, p)
            xs.append(e)
            ys.append(best)
        ax.step(xs, ys, where="post", color=BLUE, linewidth=2)
        ax.set_title(NAMES[levels[k]], fontsize=10, color=INK, loc="left")
        ax.set_ylim(0, 105)
        style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    for ax in axes[:, 0]:
        ax.set_ylabel("best %", color=INK2, fontsize=9)
    for ax in axes[-1, :]:
        ax.set_xlabel("episodes", color=INK2, fontsize=9)
    fig.suptitle(f"Best progress during training ({mode} mode)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def fig_compare(runs, path):
    """Practice vs normal mode on the same level: best progress against attempts, then against time."""
    levels = [l for l in ORDER if (l, "practice", "default") in runs and (l, "normal", "default") in runs
              and runs[(l, "normal", "default")]["curve"]]
    if not levels:
        return False
    fig, axes = plt.subplots(2, len(levels), figsize=(4.6 * len(levels) + 0.6, 6.2), sharey=True, squeeze=False)
    approx = False
    for k, l in enumerate(levels):
        for mode, colour in (("practice", BLUE), ("normal", ORANGE)):
            r = runs[(l, mode, "default")]
            curve = r["curve"] + ([(r["episodes"], 100.0)] if r["won"] and r["episodes"] else [])
            best, xs, ys = 0.0, [], []
            for e, p in curve:
                best = max(best, p)
                xs.append(e)
                ys.append(best)
            axes[0, k].step(xs, ys, where="post", color=colour, linewidth=2, label=mode)
            # Against time: the real clock when the log has one, otherwise spread the total time
            # evenly over the attempts (older practice logs; practice attempts all last about as long)
            if r["curve_time"]:
                pts = r["curve_time"] + ([(r["seconds"], 100.0)] if r["won"] and r["seconds"] else [])
                tx = [t / 60 for t, _ in pts]
                ty = [max(p for _, p in pts[:i + 1]) for i in range(len(pts))]
            elif r["seconds"] and r["episodes"]:
                approx = True
                tx = [x * r["seconds"] / r["episodes"] / 60 for x in xs]
                ty = ys
            else:
                continue
            axes[1, k].step(tx, ty, where="post", color=colour, linewidth=2,
                            linestyle="-" if r["curve_time"] else (0, (4, 2)), label=mode)
        axes[0, k].set_title(NAMES[l], fontsize=10, color=INK, loc="left")
        axes[0, k].set_xlabel("attempts", color=INK2, fontsize=9)
        axes[1, k].set_xlabel("training time (minutes)", color=INK2, fontsize=9)
    for ax in axes.flat:
        ax.set_ylim(0, 105)
        style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    for ax in axes[:, 0]:
        ax.set_ylabel("best %", color=INK2, fontsize=9)
    axes[0, -1].legend(frameon=False, fontsize=9, loc="lower right")
    title = "With checkpoints (practice) or without (normal): fewer attempts, but much more time"
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    if approx:
        fig.text(0.01, 0.005, "Dashed: time spread evenly over the attempts (older log without a clock).",
                 fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def table(runs):
    lines = ["| Level | Mode | Variant | Result | Episodes | Steps played | Time | Notes |",
             "|---|---|---|---|---|---|---|---|"]
    for l in ORDER:
        for (lv, mode, variant), r in sorted(runs.items()):
            if lv != l:
                continue
            res = "beaten" if r["won"] else f"{r['best']:.1f}%"
            ep = f"{r['episodes']:,}" if r["episodes"] else "—"
            st = f"{r['steps']:,}" if r["steps"] else "—"
            tm = f"{minutes(r):.1f} min" if r["seconds"] else "—"
            note = r.get("note", "") + (f"; {r['set_aside']} checkpoint(s) set aside" if r["set_aside"] else "")
            lines.append(f"| {NAMES[l]} | {mode} | {variant} | {res} | {ep} | {st} | {tm} | {note.strip('; ')} |")
    return "\n".join(lines)


def hardest_passage(curve):
    """Longest stretch without a new record: (episodes stuck, % where the agent was stuck)."""
    best, since, stuck = 0.0, 0, (0, 0.0)
    for e, p in curve:
        if p > best:
            if e - since > stuck[0]:
                stuck = (e - since, best)
            best, since = p, e
    return stuck


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    var = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return cov / var if var else 0.0


def beaten_practice(runs):
    return [r for (l, m, v), r in runs.items() if m == "practice" and v == "default" and r["won"]
            and r["episodes_adj"] and not r.get("note", "").startswith("development")]


def ranking(runs):
    rs = sorted(beaten_practice(runs), key=lambda r: -r["episodes_adj"])
    if len(rs) < 2:
        return ""
    lines = ["| Rank | Level | Official stars | Attempts to beat it | Hardest passage |",
             "|---|---|---|---|---|"]
    for i, r in enumerate(rs, 1):
        n, at = hardest_passage(r["curve_adj"])
        star = "*" if r["episodes_adj"] != r["episodes"] else ""
        lines.append(f"| {i} | {NAMES[r['level']]} | {STARS[r['level']]} | {r['episodes_adj']:,}{star} | "
                     f"stuck {n:,} attempts at {at:.0f}% |")
    rho = spearman([STARS[r["level"]] for r in rs], [r["episodes_adj"] for r in rs])
    lines += ["", f"Agreement with the official difficulty (Spearman rank correlation, 1 = same order, "
                  f"0 = unrelated): **{rho:.2f}** over {len(rs)} levels."]
    if any(r["episodes_adj"] != r["episodes"] for r in rs):
        lines += ["", "\\* without the attempts lost to a bug of the training code (see the notes of the "
                      "results table)."]
    return "\n".join(lines)


def place_labels(fig, ax, points, labels, fontsize=8.5):
    """Put each label next to its point without overlapping other labels or points: try a few
    positions around the point (right first) and keep the first free one."""
    candidates = [(7, -3, "left", "baseline"), (-7, -3, "right", "baseline"), (0, 8, "center", "bottom"),
                  (0, -8, "center", "top"), (7, 6, "left", "bottom"), (7, -8, "left", "top"),
                  (-7, 6, "right", "bottom"), (-7, -8, "right", "top")]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    taken = []
    for x, y in points:                                     # the markers themselves (about 6 px wide)
        px, py = ax.transData.transform((x, y))
        taken.append((px - 6, py - 6, px + 6, py + 6))
    axes_box = ax.get_window_extent(renderer)
    for (x, y), label in zip(points, labels):
        best = None
        for dx, dy, ha, va in candidates:
            t = ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                            fontsize=fontsize, color=INK, ha=ha, va=va)
            b = t.get_window_extent(renderer).padded(2)
            box = (b.x0, b.y0, b.x1, b.y1)
            free = all(box[2] < o[0] or box[0] > o[2] or box[3] < o[1] or box[1] > o[3] for o in taken)
            inside = b.x0 >= axes_box.x0 and b.x1 <= axes_box.x1 and b.y0 >= axes_box.y0 and b.y1 <= axes_box.y1
            if free and inside:
                best = (t, box)
                break
            t.remove()
        if best is None:                                    # no free spot: keep the default position
            t = ax.annotate(label, (x, y), xytext=(7, -3), textcoords="offset points",
                            fontsize=fontsize, color=INK)
            b = t.get_window_extent(renderer)
            best = (t, (b.x0, b.y0, b.x1, b.y1))
        taken.append(best[1])


def fig_ranking(runs, path):
    rs = beaten_practice(runs)
    if len(rs) < 2:
        return False
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    xs = [STARS[r["level"]] for r in rs]
    ys = [r["episodes_adj"] for r in rs]
    exact = [r["episodes_adj"] == r["episodes"] for r in rs]
    ax.scatter([x for x, e in zip(xs, exact) if e], [y for y, e in zip(ys, exact) if e],
               s=70, color=BLUE, edgecolor="white", linewidth=2, zorder=3, label="one clean run")
    if not all(exact):
        ax.scatter([x for x, e in zip(xs, exact) if not e], [y for y, e in zip(ys, exact) if not e],
                   s=60, facecolor="white", edgecolor=BLUE, linewidth=2, zorder=3,
                   label="trained in two parts (estimate)")
        ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.set_xlabel("Official difficulty (stars)", color=INK2)
    ax.set_ylabel("Attempts needed by the AI", color=INK2)
    ax.set_xticks(range(1, max(xs) + 1))
    ax.set_xlim(0.3, max(xs) + 2.6)
    ax.set_ylim(0, max(ys) * 1.15)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_title("Is a level hard for the AI when it is hard for humans?", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    place_labels(fig, ax, list(zip(xs, ys)), [NAMES[r["level"]] for r in rs])
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def main():
    runs = collect()
    os.makedirs(os.path.join("docs", "figures"), exist_ok=True)
    with open(os.path.join("results", "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["level", "mode", "variant", "won", "best_percent", "episodes", "steps_played",
                    "seconds", "note", "source"])
        for (l, mode, variant), r in sorted(runs.items(), key=lambda kv: ORDER.index(kv[0][0])):
            w.writerow([l, mode, variant, r["won"], r["best"], r["episodes"], r["steps"], r["seconds"],
                        r.get("note", ""), r["source"]])
    figs = []
    if fig_minutes(runs, os.path.join("docs", "figures", "minutes_per_level.png")):
        figs.append("minutes_per_level.png")
    if fig_curves(runs, os.path.join("docs", "figures", "curves_practice.png")):
        figs.append("curves_practice.png")
    if fig_curves(runs, os.path.join("docs", "figures", "curves_normal.png"), mode="normal"):
        figs.append("curves_normal.png")
    if fig_compare(runs, os.path.join("docs", "figures", "practice_vs_normal.png")):
        figs.append("practice_vs_normal.png")

    block = ["<!-- results:start (generated by make_report.py, do not edit by hand) -->", "",
             table(runs), ""]
    block += [f"![{f}](figures/{f})" for f in figs]
    rank = ranking(runs)
    if rank:
        block += ["", "### Difficulty according to the AI / Difficulté selon l'IA", "",
                  "Levels ranked by the number of attempts (practice mode) the agent needed to beat them. "
                  "\"Hardest passage\": the longest time it went without beating its record, and where.", "",
                  rank, ""]
        if fig_ranking(runs, os.path.join("docs", "figures", "ai_vs_official_difficulty.png")):
            block += ["![ai_vs_official_difficulty.png](figures/ai_vs_official_difficulty.png)"]
    block += ["", "<!-- results:end -->"]
    block = "\n".join(block)
    path = os.path.join("docs", "RESULTS.md")
    if os.path.exists(path):
        doc = open(path, encoding="utf-8").read()
        if "<!-- results:start" in doc and "<!-- results:end -->" in doc:
            doc = re.sub(r"<!-- results:start.*?<!-- results:end -->", lambda m: block, doc, flags=re.S)
        else:
            doc += "\n\n" + block + "\n"
    else:
        doc = "# Results\n\n" + block + "\n"
    open(path, "w", encoding="utf-8").write(doc)
    print(f"{len(runs)} runs -> results/summary.csv, docs/RESULTS.md, docs/figures/ ({', '.join(figs)})")


if __name__ == "__main__":
    main()
