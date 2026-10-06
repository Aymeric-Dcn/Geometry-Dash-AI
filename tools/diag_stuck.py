"""
Looks at what happens at the place where training is stuck (debugging tool).
Stop the training first (Ctrl+C), keep the level open in GD, then:
    python -m tools.diag_stuck clubstep
It replays the best run, then tries random moves from a bit before its end, and writes
what it sees (position, %, game mode, death) to diag_stuck_<level>.txt.
"""
import random
import sys

from realenv import GAMEMODES, RealGDEnv

level = sys.argv[1]
import os
path = next(f for f in (f"solution_qlearning_{level}_practice_fine.txt", f"solution_qlearning_{level}_practice_state.txt")
            if os.path.exists(f))
actions = [int(c) for c in open(path).read().strip()]
env = RealGDEnv(speed=20, max_fall=0)
env.normal_mode()
out = open(f"diag_stuck_{level}.txt", "w")


def line(tag=""):
    r, s = env.raw, env.state
    return (f"{tag}step {env.steps:5d}  x={s.x:8.2f}  y={s.y:6.2f}  vy={s.vy:7.2f}  ground={int(s.grounded)}  "
            f"mode={GAMEMODES[r.mode]:<6} {r.percent:6.2f}%  dead={int(r.dead)}")


# 1. The best run, step by step over its last 120 steps, then doing nothing
env.reset()
out.write(f"Best run: {len(actions)} actions\n")
for i, a in enumerate(actions + [0] * 600):
    env.step(a)
    if i >= len(actions) - 120 or env.raw.dead:
        out.write(line("press " if a else "      ") + "\n")
    if env.raw.dead or env.raw.won:
        break
out.write("\n")

# 2. Random tries from 60 steps before the end of the best run
start = max(0, len(actions) - 60)
rng = random.Random(0)
for t in range(40):
    env.reset()
    for a in actions[:start]:
        env.step(a)
    a, n = 0, 0
    while not (env.raw.dead or env.raw.won) and n < 1200:
        if rng.random() < 0.1:
            a = 1 - a
        env.step(a)
        n += 1
    out.write(line(f"try {t:2d}: ") + "\n")
    out.flush()
env.close()
out.close()
print(f"Written to diag_stuck_{level}.txt")
