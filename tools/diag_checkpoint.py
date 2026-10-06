"""
Checks that respawning at a checkpoint reproduces the run from the start (debugging tool).
Stop the training first (Ctrl+C), keep the level open in GD, then:
    python -m tools.diag_checkpoint clubstep
For checkpoints placed along the end of the best run, it compares the run from the start with
the same actions played after a respawn, and writes the result to diag_checkpoint_<level>.txt.
"""
import sys

from realenv import GAMEMODES, RealGDEnv

level = sys.argv[1]
actions = [int(c) for c in open(f"solution_qlearning_{level}_practice_state.txt").read().strip()]
env = RealGDEnv(speed=20, max_fall=0)
env.normal_mode()
out = open(f"diag_checkpoint_{level}.txt", "w")


def play(from_step):
    trace = {}
    for i in range(from_step, len(actions) + 300):
        env.step(actions[i] if i < len(actions) else 0)
        r = env.raw
        trace[i + 1] = (env.state.x, env.state.y, env.state.vy, r.mode, r.dead)
        if r.dead or r.won:
            break
    return trace


env.reset()
states = [env.state]
ref = {}
for i, a in enumerate(actions):
    env.step(a)
    states.append(env.state)
    ref[i + 1] = (env.state.x, env.state.y, env.state.vy, env.raw.mode, env.raw.dead)
    if env.raw.dead:
        break
out.write(f"Reference run from the start: {len(ref)} steps, ends at x={states[-1].x:.2f}\n\n")

n = len(actions)
for target in range(n - 600, n - 20, 60):
    cp = env.make_checkpoint(target, actions, states)
    if cp is None:
        out.write(f"near step {target}: no checkpoint position (not on the ground)\n")
        continue
    try:
        env.reset(start_state=cp)
    except RuntimeError as e:
        out.write(f"checkpoint at step {cp.step}: respawn failed ({e})\n")
        continue
    mode = GAMEMODES[env.raw.mode]
    tr = play(cp.step)
    first = None
    for s, v in tr.items():
        if s not in ref or max(abs(v[0] - ref[s][0]), abs(v[1] - ref[s][1])) > 1e-3 or v[3] != ref[s][3]:
            first = s
            break
    last = max(tr)
    if first is None:
        out.write(f"checkpoint at step {cp.step} ({mode}): IDENTICAL to the run from the start "
                  f"(dies at step {last} like the reference)\n")
    else:
        a, b = tr[first], ref.get(first)
        out.write(f"checkpoint at step {cp.step} ({mode}): DIFFERENT from step {first} "
                  f"(after respawn: x={a[0]:.2f} y={a[1]:.2f} vy={a[2]:.2f} {GAMEMODES[a[3]]} | "
                  + (f"from start: x={b[0]:.2f} y={b[1]:.2f} vy={b[2]:.2f} {GAMEMODES[b[3]]}" if b else "from start: already dead")
                  + f"), after respawn it lasts until step {last}\n")
    out.flush()
env.normal_mode()
env.close()
out.close()
print(f"Written to diag_checkpoint_{level}.txt")
