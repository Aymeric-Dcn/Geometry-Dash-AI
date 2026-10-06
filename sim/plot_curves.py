"""
Plot the learning curves (agent progress over episodes) for each variant.
Usage: python -m sim.plot_curves [level_name]  ->  curves_<level>.png
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_qlearning import train

level = sys.argv[1] if len(sys.argv) > 1 else "stereo_lite"
configs = [("practice", "state", "Practice mode, knows its own movement", "#2a9d8f"),
           ("normal", "state", "Restarts from the beginning, knows its own movement", "#e9a03b"),
           ("practice", "x", "Only knows its x position", "#c0504d")]

fig, ax = plt.subplots(figsize=(9, 4.5))
for mode, obs, label, color in configs:
    res = train(level, mode, obs, episodes=20000, seed=0, verbose=False)
    eps, prog = zip(*res["history"])
    best, run = [], 0
    for p in prog:
        run = max(run, p)
        best.append(100 * run)
    ax.plot(eps, best, label=label + (" ✓" if res["won"] else ""), color=color, linewidth=2)

ax.set_xlabel("Episodes (attempts)")
ax.set_ylabel("Best progress (%)")
ax.set_title(f"Blind agent on '{level}'")
ax.set_ylim(0, 105)
ax.grid(alpha=0.3)
ax.legend(frameon=False)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(f"curves_{level}.png", dpi=130)
print(f"curves_{level}.png written")
