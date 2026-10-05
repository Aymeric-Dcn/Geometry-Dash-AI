"""
Checks that the game connection works, BEFORE training anything.

1. Open Geometry Dash (with the GD AI Bridge mod), open Stereo Madness (or any level).
2. Run: python test_bridge.py

Tests:
  1. connection and restart
  2. running without pressing: x must grow by about 10.4 blocks per second
  3. one jump: y must go up then down, and the cube must land
  4. determinism: the same random inputs played twice must give exactly the same positions
  5. speed: how many steps per second the game can do
"""
import random
import sys
import time

from realenv import RealGDEnv

ok_all = True


def check(name, ok, detail=""):
    global ok_all
    ok_all &= ok
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def play(env, actions):
    env.reset()
    xs, ys = [], []
    for a in actions:
        env.step(a)
        xs.append(env.raw.x)
        ys.append(env.raw.y)
        if env.state.dead or env.state.won:
            break
    return xs, ys


speed = int(sys.argv[1]) if len(sys.argv) > 1 else 20
print(f"Connecting (speed = {speed} steps per frame)...")
env = RealGDEnv(speed=speed)

print("\n1. Connection and restart")
env.reset()
r = env.raw
print(f"  spawn: x={r.x:.1f} y={r.y:.1f} grounded={r.grounded} mode={r.mode} percent={r.percent:.1f}")
check("player is alive at spawn", not r.dead)

print("\n2. Running without pressing (30 steps = 0.5 s)")
env.reset()
for _ in range(30):
    env.step(0)
dx = env.state.x
check("x grows by ~5.2 blocks in 0.5 s", 4.0 < dx < 6.5, f"dx = {dx:.2f} blocks")
check("y stays on the ground", abs(env.state.y) < 0.05, f"y = {env.state.y:.3f} blocks")

print("\n3. One jump")
env.reset()
env.step(1)
heights, landed_at = [], None
for i in range(60):
    env.step(0)
    heights.append(env.state.y)
    if i > 3 and env.state.grounded and landed_at is None:
        landed_at = i
peak = max(heights)
check("the cube goes up", peak > 1.0, f"peak = {peak:.2f} blocks")
check("the cube lands again", landed_at is not None,
      f"landed after {landed_at + 1 if landed_at is not None else '?'} steps")

print("\n4. Determinism (same inputs twice)")
rng = random.Random(0)
actions = [1 if rng.random() < 0.15 else 0 for _ in range(300)]
xs1, ys1 = play(env, actions)
xs2, ys2 = play(env, actions)
n = min(len(xs1), len(xs2))
diff = max(max(abs(a - b) for a, b in zip(xs1[:n], xs2[:n])),
           max(abs(a - b) for a, b in zip(ys1[:n], ys2[:n])))
check("both runs last the same number of steps", len(xs1) == len(xs2), f"{len(xs1)} vs {len(xs2)}")
check("positions are identical", diff < 1e-3, f"max difference = {diff:.5f} units")

print("\n5. Speed")
env.reset()
t0 = time.time()
steps = 0
while time.time() - t0 < 3:
    env.step(0)
    steps += 1
    if env.state.dead or env.state.won:
        env.reset()
rate = steps / (time.time() - t0)
print(f"  {rate:.0f} steps/s = {rate / 60:.1f}x real time")

env.close()
print("\nALL GOOD, the game is ready for training." if ok_all else
      "\nSome checks failed: send me this output and we will fix it.")
