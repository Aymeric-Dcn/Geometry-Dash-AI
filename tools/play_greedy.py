"""
Plays the trained agent (no exploration) in the real game and saves its run.
Open the level in GD, then:
    python -m tools.play_greedy                    # real time, so you can watch it
    python -m tools.play_greedy --speed 20         # fast
"""
import argparse
import pickle
from collections import defaultdict

from realenv import RealGDEnv
from train_qlearning import run_greedy

p = argparse.ArgumentParser()
p.add_argument("--qtable", default="qtable_real_practice_state.pkl")
p.add_argument("--speed", type=int, default=1)
p.add_argument("--out", default="greedy_run.txt")
args = p.parse_args()

Q = defaultdict(lambda: [0.0, 0.0])
with open(args.qtable, "rb") as f:
    Q.update(pickle.load(f))
env = RealGDEnv(speed=args.speed)
acts, states, info = run_greedy(env, Q, "state")
with open(args.out, "w") as f:
    f.write("".join(map(str, acts)))
print(f"{'LEVEL BEATEN' if info['won'] else 'Died'} at {100 * info['progress']:.1f}% "
      f"after {len(acts)} steps. Run saved to {args.out}")
env.close()
