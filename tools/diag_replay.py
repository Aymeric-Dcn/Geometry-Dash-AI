"""
Replays a saved run and writes every step around a given point to a file (debugging tool).
    python -m tools.diag_replay backup_clubstep/solution_qlearning_clubstep_practice_state.txt 3164
Writes diag_replay.txt: position, game mode, ground contact, real death and "training death"
(the "lost in the sky" rule) for the 120 steps before that point and 40 after.
"""
import sys

from realenv import GAMEMODES, RealGDEnv

actions = [int(c) for c in open(sys.argv[1]).read().strip()]
around = int(sys.argv[2])
env = RealGDEnv(speed=20)
env.normal_mode()
env.reset()
with open("diag_replay.txt", "w") as out:
    first_cut = None
    for i, a in enumerate(actions):
        env.step(a)
        r, s = env.raw, env.state
        if first_cut is None and s.dead and not r.dead:
            first_cut = env.steps
        if around - 120 <= env.steps <= around + 40 or r.dead:
            out.write(f"{'press ' if a else '      '}step {env.steps:5d}  x={s.x:8.2f}  y={s.y:6.2f}  "
                      f"vy={s.vy:7.2f}  ground={int(s.grounded)}  mode={GAMEMODES[r.mode]:<6} "
                      f"full_speed={env._full_speed_steps}  dead={int(r.dead)}  training_dead={int(s.dead)}\n")
        if r.dead or r.won:
            break
    out.write(f"\nfirst step stopped by the 'lost in the sky' rule: {first_cut}\n")
env.close()
print("Written to diag_replay.txt")
