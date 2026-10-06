"""
Compares the agent's current choices with its best run (debugging tool).
Stop the training (Ctrl+C), keep the level open, then:
    python diag_greedy.py theory-of-everything_clean
Writes diag_greedy.txt.
"""
import pickle
import sys
from collections import defaultdict

from realenv import GAMEMODES, RealGDEnv
from train_qlearning import greedy, make_key, upgrade_qtable

tag = sys.argv[1]
obs = sys.argv[2] if len(sys.argv) > 2 else "state"
Q = defaultdict(lambda: [0.0, 0.0], upgrade_qtable(pickle.load(open(f"qtable_{tag}_practice_{obs}.pkl", "rb"))))
best = [int(c) for c in open(f"solution_qlearning_{tag}_practice_{obs}.txt").read().strip()]
env = RealGDEnv(speed=20)
env.normal_mode()
out = open("diag_greedy.txt", "w")

# 1. Follow the best run; note every step where the agent would now choose differently
env.reset()
diffs, held = 0, 0
for step, a in enumerate(best):
    key = make_key(obs, step, env.state, held)
    held = a
    g = greedy(Q, key)
    if g != a and diffs < 15:
        diffs += 1
        out.write(f"step {step:5d} {env.raw.percent:5.1f}% {GAMEMODES[env.raw.mode]:<5} key={key} "
                  f"Q={[round(v, 2) for v in Q[key]]} best run does {a}, agent now does {g}\n")
    env.step(a)
    if env.raw.dead or env.raw.won:
        break
out.write(f"best run replayed: ends at {env.raw.percent:.1f}% after {env.steps} steps\n\n")

# 2. The agent's own run, without exploration
env.reset()
acts = []
while True:
    a = greedy(Q, make_key(obs, len(acts), env.state, acts[-1] if acts else 0))
    acts.append(a)
    env.step(a)
    if env.raw.dead or env.raw.won or len(acts) > 20000:
        break
first = next((i for i in range(min(len(acts), len(best))) if acts[i] != best[i]), None)
out.write(f"agent's run: ends at {env.raw.percent:.1f}% after {len(acts)} steps; "
          f"first different action from the best run: step {first}\n")
out.close()
env.close()
print("Written to diag_greedy.txt")
