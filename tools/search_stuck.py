"""
Backtracking search (no AI) at the place where training is stuck (debugging tool).

Tells whether the agent's wall is passable at all from a bit earlier in its best run, and if it
is, finds a way through. Stop the training first (Ctrl+C), keep the level open in GD, then:
    python -m tools.search_stuck grief                 # search from 2 s before the death
    python -m tools.search_stuck grief --back 300      # from further back (5 s)
Any way past the old death is written to solution_search_<level>.txt as soon as it is found (the
whole run from the start of the level, kept even if the game crashes), ready to be taught to the agent:
    python train_qlearning.py --env real --level grief --obs fine --air-checkpoints --resume --seed-run solution_search_grief.txt

How it works: like sim/search_bot.py, depth first. At each step it tries the best run's action
first, then the other one; a state that only leads to death is remembered and never tried again.
The real game cannot save and restore a state, so each backtrack respawns at a practice checkpoint
and replays the moves since then.
"""
import argparse
import os
import time

from realenv import RealGDEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("level")
    p.add_argument("--back", type=int, default=120, help="start the search this many steps before the death (60 = 1 s)")
    p.add_argument("--ahead", type=int, default=180, help="success = this many steps past the old death")
    p.add_argument("--max-minutes", type=float, default=30)
    args = p.parse_args()

    path = next((f for f in (f"solution_qlearning_{args.level}_practice_fine.txt",
                             f"solution_qlearning_{args.level}_practice_state.txt") if os.path.exists(f)), None)
    if path is None:
        raise SystemExit(f"No best run found for '{args.level}'.")
    best = [int(c) for c in open(path).read().strip()]
    print(f"Best run: {path}, {len(best)} steps")

    env = RealGDEnv(speed=20, max_fall=0)
    env.air_checkpoints = True
    env.normal_mode()

    # 1. Replay the best run once to get its states, and place a checkpoint `back` steps before its end
    env.reset()
    states = [env.state]
    for a in best:
        env.step(a)
        states.append(env.state)
        if env.state.dead or env.state.won:
            break
    print(f"Best run replayed: {'dies' if env.state.dead else 'ends'} at {env.raw.percent:.2f}% (step {env.steps})")
    cp = env.make_checkpoint(max(1, len(best) - args.back), best, states)
    if cp is None:
        raise SystemExit("No checkpoint could be placed there; try another --back.")
    goal = len(best) + args.ahead
    print(f"Search from step {cp.step} ({len(best) - cp.step} steps before the death), "
          f"goal: step {goal}")

    def goto(actions):
        env.reset(start_state=cp)
        for a in actions:
            env.step(a)

    def key(held):
        # `held` (the button is already pressed) matters: an orb only reacts to a NEW press, so
        # the same position and speed with the button held or released are different situations
        s = env.state
        return (env.steps, round(s.y, 3), round(s.vy, 2), held)

    out = f"solution_search_{args.level}.txt"
    dead = set()                 # states known to always lead to death
    stack = [("checkpoint", [])]   # (state key, actions already tried from it)
    actions = []                 # actions since the checkpoint
    synced = False               # is the game exactly at the end of `actions`?
    resets = steps = 0
    furthest = 0
    t0 = last_print = time.time()
    found = None
    while stack:
        if time.time() - t0 > 60 * args.max_minutes:
            print("Time limit reached.")
            break
        k, tried = stack[-1]
        if len(tried) == 2 or k in dead:
            dead.add(k)
            stack.pop()
            if actions:
                actions.pop()
            synced = False
            continue
        step = cp.step + len(actions)
        first = best[step] if step < len(best) else 0
        a = first if first not in tried else 1 - first
        tried.append(a)
        if not synced:
            goto(actions)
            resets += 1
            steps += len(actions)
        env.step(a)
        steps += 1
        synced = True
        if env.state.dead or key(a) in dead:
            synced = False
            continue
        actions.append(a)
        if env.steps > furthest:
            furthest = env.steps
            if furthest > len(best):          # already better than the agent: save it right away
                with open(out, "w") as f:
                    f.write("".join(map(str, best[:cp.step] + actions)))
        if env.state.won or env.steps >= goal:
            found = actions
            break
        stack.append((key(a), []))
        if time.time() - last_print > 10:
            last_print = time.time()
            print(f"  {time.time() - t0:5.0f}s  at step {env.steps}  furthest {furthest} "
                  f"({furthest - len(best):+d} vs the old death)  dead ends {len(dead)}  respawns {resets}")

    dt = time.time() - t0
    # No RESET here: on Grief, restarting the level while the player is alive past ~21% crashed
    # GD (PlayLayer::updateTimeLabel). Just disconnect: the game goes on by itself.
    if found is None:
        print(f"\nNo way through found from step {cp.step} ({dt:.0f}s, {resets} respawns, furthest step {furthest}, "
              f"{furthest - len(best):+d} vs the old death).")
        if not stack:
            print("The search was exhaustive: from this checkpoint the passage is impossible, "
                  "the fatal mistake is earlier. Try a larger --back.")
        env.close()
        return
    run = best[:cp.step] + found
    with open(out, "w") as f:
        f.write("".join(map(str, run)))
    print(f"\nWay through found in {dt:.0f}s ({resets} respawns): reaches step {env.steps} "
          f"({env.raw.percent:.2f}%), {env.steps - len(best)} steps past the old death.")
    print(f"Written to {out}. Teach it to the agent with --resume --seed-run {out}")
    env.close()


if __name__ == "__main__":
    main()
