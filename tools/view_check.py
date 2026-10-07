"""
Checks what the neural network will "see" in the real game (needs mod v0.5.0+).

Open a level in GD, then:
    python -m tools.view_check                                   # the player does nothing (dies early)
    python -m tools.view_check results/runs/stereo_madness__normal__default_win.txt   # follow a run
    python -m tools.view_check RUN_FILE --every 20

Every `--every` steps (30 = half a second) the game stops and the terminal shows the grid around
the player: compare it with the frozen game screen, then press Enter to continue.
    @ the player (1 block)   # solid   ^ hazard (kills)   o orb / pad / portal   . empty
The grid covers 1 block behind the player to 10 ahead, 5 blocks below its feet to 4 above; it is
aligned on the level's blocks, so the @ marks the block the player is (mostly) in.
"""
import argparse
from collections import Counter

from deep.realdqn import frac_x
from deep.realview import channel, draw, grid
from realenv import GAMEMODES, RealGDEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", nargs="?", help="a saved run to follow (file of 0/1); default: do nothing")
    p.add_argument("--every", type=int, default=30)
    args = p.parse_args()
    actions = [int(c) for c in open(args.run).read().strip()] if args.run else []

    env = RealGDEnv(speed=1)
    env.normal_mode()
    env.reset()
    unknown = Counter()
    step = 0
    while not (env.raw.dead or env.raw.won):
        a = actions[step] if step < len(actions) else 0
        env.step(a)
        step += 1
        if step % args.every == 0 or env.raw.dead:
            objs = env.view()
            for t, *_ in objs:
                if channel(t) is None:
                    unknown[t] += 1
            print(f"\nstep {step}  {env.raw.percent:.1f}%  mode={GAMEMODES[env.raw.mode]}  "
                  f"height={env.state.y:.2f} blocks  {len(objs)} objects nearby")
            print(draw(grid(objs, env.state.y, frac_x(env))))
            for t, x, y, w, h in sorted(objs, key=lambda o: o[1])[:12]:
                print(f"   type {t:3d}  x={x:6.2f}  y={y:6.2f}  size {w:.2f} x {h:.2f}")
            if env.raw.dead:
                print("(dead)")
                break
            input("Enter = continue, Ctrl+C = stop ")
    if unknown:
        print("\nObject types ignored by the grid:", dict(unknown))
    env.close()


if __name__ == "__main__":
    main()
