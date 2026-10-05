"""
Checks practice mode (GD checkpoints) on the real game, BEFORE training with it.

Open a level in GD (Stereo Madness), stop any training, then:
    python test_practice.py                     # uses a simple run (no jump until the first death)
    python test_practice.py solution_qlearning_real_normal_state.txt   # uses a saved run (longer test)

The key question: does restarting from a checkpoint reproduce EXACTLY the original run?
If it does not, the Q-table would learn from situations that change at every attempt.
"""
import sys
import time

from realenv import RealGDEnv

ok_all = True


def check(name, ok, detail=""):
    global ok_all
    ok_all &= ok
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def compare(env, actions, ref_raw, start):
    """Play actions[start:] from the current state, return the max position difference vs the reference."""
    diff = max(abs(env.raw.x - ref_raw[start].x), abs(env.raw.y - ref_raw[start].y))
    n = 0
    for i in range(start, len(actions)):
        env.step(actions[i])
        r, ref = env.raw, ref_raw[i + 1]
        diff = max(diff, abs(r.x - ref.x), abs(r.y - ref.y))
        n += 1
        if r.dead or r.won:
            break
    same_end = env.raw.dead == ref_raw[min(start + n, len(ref_raw) - 1)].dead
    return diff, n, same_end


speed = 20
env = RealGDEnv(speed=speed)

# 1. Reference run from the start of the level
if len(sys.argv) > 1:
    actions = [int(c) for c in open(sys.argv[1]).read().strip()]
    print(f"Reference run: {sys.argv[1]} ({len(actions)} actions)")
else:
    actions = [0] * 3000
    print("Reference run: never jump (until the first death)")
env.reset()
ref_raw, ref_states = [env.raw], [env.state]
for a in actions:
    env.step(a)
    ref_raw.append(env.raw)
    ref_states.append(env.state)
    if env.raw.dead or env.raw.won:
        break
actions = actions[:len(ref_raw) - 1]
print(f"  the run lasts {len(actions)} steps, ends {'dead' if env.raw.dead else 'alive'} "
      f"at {env.raw.percent:.1f}%")

# 2. Checkpoint at ~60% of the run
target = int(len(actions) * 0.6)
cp = env.make_checkpoint(target, actions, ref_states)
print(f"\n1. Placing a checkpoint near step {target}")
check("valid checkpoint position found (on the ground)", cp is not None)
if cp is None:
    sys.exit(1)
t0 = time.time()
env.reset(start_state=cp)
t_place = time.time() - t0
print(f"  checkpoint at step {cp.step}, placed in {t_place:.2f} s (replay of {len(cp.prefix)} steps)")

print("\n2. Respawning at the checkpoint reproduces the original run")
diff, n, same_end = compare(env, actions, ref_raw, cp.step)
check("same positions as the original run", diff < 1e-3, f"max difference = {diff:.5f} units over {n} steps")
check("same ending (death at the same step)", same_end)

print("\n3. Respawning again (no replay needed)")
for attempt in range(3):
    t0 = time.time()
    env.reset(start_state=cp)
    t_respawn = time.time() - t0
    diff, n, _ = compare(env, actions, ref_raw, cp.step)
    check(f"respawn {attempt + 1}: identical", diff < 1e-3, f"diff = {diff:.5f}, respawn took {t_respawn:.3f} s")

print("\n4. GD's automatic checkpoints are ignored")
env.reset(start_state=cp)
for _ in range(300):                       # play a while so GD could place its own checkpoints
    env.step(0)
    if env.raw.dead:
        break
env.reset(start_state=cp)
d = max(abs(env.raw.x - ref_raw[cp.step].x), abs(env.raw.y - ref_raw[cp.step].y))
check("still respawning at OUR checkpoint", d < 1e-3, f"diff = {d:.5f}")

print("\n5. Back to the start of the level")
env.reset()
d = max(abs(env.raw.x - ref_raw[0].x), abs(env.raw.y - ref_raw[0].y))
check("reset() without checkpoint goes back to the start", d < 1e-3, f"diff = {d:.5f}")

env.close()
print("\nALL GOOD: practice mode can be used for training." if ok_all else
      "\nSome checks failed: send me this output. Training can still use --mode normal.")
