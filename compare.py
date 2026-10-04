"""
Compare the blind agent variants over several random seeds.
Usage: python compare.py [level_name]
"""
import sys
from statistics import median

from train_qlearning import train
from search_bot import solve

level = sys.argv[1] if len(sys.argv) > 1 else "stereo_lite"
SEEDS = range(5)

acts, sims, _ = solve(level)
print(f"{'Search bot (reference)':38s} beaten, {sims:>9,} steps simulated")

for mode in ["practice", "normal"]:
    for obs in ["state", "x"]:
        runs = [train(level, mode, obs, episodes=20000, seed=s, verbose=False) for s in SEEDS]
        wins = [r for r in runs if r["won"]]
        name = f"Q-learning mode={mode}, obs={obs}"
        if wins:
            steps = median(r["sim_steps"] for r in wins)
            print(f"{name:38s} {len(wins)}/{len(runs)} beaten, median {steps:>9,.0f} steps simulated"
                  f" (~{steps / 60 / 60:.0f} min of real-time play)")
        else:
            print(f"{name:38s} 0/{len(runs)} beaten within 20,000 episodes")
