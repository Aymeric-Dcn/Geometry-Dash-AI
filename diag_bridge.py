"""
Step-by-step trace of the start of an attempt on the real game (debugging tool).
Open a level in GD, then: python diag_bridge.py
"""
from realenv import RealGDEnv

env = RealGDEnv(speed=1)

print("A. Running without pressing")
print(" step        x        y     yVel  ground  percent")
env.reset()
r = env.raw
print(f" {0:4d} {r.x:8.2f} {r.y:8.2f} {r.y_velocity:8.3f}  {int(r.grounded):6d}  {r.percent:7.2f}")
prev_x = r.x
for i in range(1, 41):
    env.step(0)
    r = env.raw
    print(f" {i:4d} {r.x:8.2f} {r.y:8.2f} {r.y_velocity:8.3f}  {int(r.grounded):6d}  {r.percent:7.2f}"
          f"   dx={r.x - prev_x:6.2f}")
    prev_x = r.x
    if r.dead:
        print(" (dead)")
        break

print("\nB. Wait 20 steps, then hold for 1 step and watch the jump")
env.reset()
for _ in range(20):
    env.step(0)
for i in range(30):
    env.step(1 if i == 0 else 0)
    r = env.raw
    print(f" {i:4d} {r.x:8.2f} {r.y:8.2f} {r.y_velocity:8.3f}  {int(r.grounded):6d}")
    if r.dead:
        print(" (dead)")
        break
env.close()
