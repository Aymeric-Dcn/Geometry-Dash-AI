"""
Diagnoses a determinism problem between normal mode and practice mode (debugging tool).

Open the level in GD, then: python -m tools.diag_practice
1. A short training in normal mode finds a run that survives a little while.
2. That run is replayed several times, alternating normal and practice mode.
3. For each replay, we print where it starts to differ from the first one.
"""
from realenv import RealGDEnv
from train_qlearning import train

env = RealGDEnv(speed=20)
print("1. Short training in normal mode (about a minute)...")
res = train("diag", "normal", "state", episodes=400, env=env, verbose=False, focus_margin=30)
actions = res["actions"]
print(f"   reference run: {len(actions)} steps")


def play(practice):
    env._set_practice(practice)
    env.reset()
    trace = []
    for a in actions:
        env.step(a)
        trace.append((env.raw.x, env.raw.y, env.raw.dead, env.raw.percent))
        if env.raw.dead or env.raw.won:
            break
    return trace


print("\n2. Replays (N = normal mode, P = practice mode)")
ref = play(False)
print(f"   N reference: {len(ref)} steps, ends {'dead' if ref[-1][2] else 'alive'} at {ref[-1][3]:.1f}%")
for label, practice in [("N", False), ("P", True), ("P", True), ("N", False), ("P", True)]:
    tr = play(practice)
    first_diff = None
    for i, (a, b) in enumerate(zip(ref, tr)):
        if abs(a[0] - b[0]) > 1e-3 or abs(a[1] - b[1]) > 1e-3:
            first_diff = i
            break
    if first_diff is None and len(tr) == len(ref):
        print(f"   {label}: identical")
    else:
        i = first_diff if first_diff is not None else min(len(tr), len(ref)) - 1
        print(f"   {label}: DIFFERENT from step {i} (x={ref[i][0]:.2f} vs {tr[i][0]:.2f}, "
              f"y={ref[i][1]:.2f} vs {tr[i][1]:.2f}, at {ref[i][3]:.1f}%), lasts {len(tr)} steps")
env._set_practice(False)
env.close()
