"""
Where does a restart put the player right now? (debugging tool)
Stop the training with Ctrl+C WITHOUT leaving the level, then: python -m tools.diag_reset
"""
from realenv import GAMEMODES, RealGDEnv

env = RealGDEnv(speed=20)


def show(label):
    r = env.raw
    print(f"{label:<38} x={r.x / 30:8.2f} blocks  y={r.y / 30:6.2f}  {r.percent:5.1f}%  "
          f"mode={GAMEMODES[r.mode]}  dead={int(r.dead)}")


env.raw = None
env._restart()
show("1. RESET as left by the training:")
env._restart()
show("2. RESET again:")
env._expect_ok("CLEARCP")
env._restart()
show("3. after CLEARCP + RESET:")
env._expect_ok("CLEARCP")
env.close()
