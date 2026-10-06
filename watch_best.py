"""
Replays the agent's best run in the real game, at real speed, so you can see where it dies.
Open the level in GD, then:
    python watch_best.py                                          # practice-mode run
    python watch_best.py solution_qlearning_real_normal_state.txt
"""
import sys

from realenv import GAMEMODES, RealGDEnv

path = sys.argv[1] if len(sys.argv) > 1 else "solution_qlearning_real_practice_state.txt"
actions = [int(c) for c in open(path).read().strip()]
env = RealGDEnv(speed=1)                     # 1 step per frame = real time
env.reset()
top, cut_at = 0.0, None


def track():
    global top, cut_at
    top = max(top, abs(env.state.y))
    if cut_at is None and env.state.dead and not env.raw.dead:   # stopped by the "lost in the sky" rule
        cut_at = (env.steps, env.raw.percent)


for a in actions:
    env.step(a)
    track()
    if env.raw.dead or env.raw.won:
        break
else:
    # The saved run ends at its death: keep going without pressing to show what happens next
    while not (env.raw.dead or env.raw.won):
        env.step(0)
        track()
r = env.raw
print(f"{'WON' if r.won else 'Died'} at {r.percent:.1f}% (step {env.steps}, mode {GAMEMODES[r.mode]}, "
      f"x = {r.x / 30:.1f} blocks, y = {env.state.y:.1f} blocks above the start)")
print(f"Highest point: {top:.1f} blocks from the start")
if cut_at:
    print(f"-> training stops this run at step {cut_at[0]} ({cut_at[1]:.1f}%): cube lost in the sky "
          f"(more than {env.max_fall} s at full vertical speed)")
env.close()
