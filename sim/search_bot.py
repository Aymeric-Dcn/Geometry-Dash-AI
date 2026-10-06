"""
Reference bot WITHOUT any AI: backtracking search.
At each decision it tries "do nothing", and if that eventually leads to death it backtracks
and tries "jump". This is how many GD bots work.
Used to (1) prove a level is beatable, (2) compare against the learning agent.

Usage: python -m sim.search_bot [level_name]
"""
import sys
import time

from sim.gdsim import GDEnv


def key(s):
    return (round(s.x, 3), round(s.y, 3), round(s.vy, 2), s.grounded)


def solve(level_name="stereo_lite"):
    env = GDEnv(level_name)
    env.reset()
    dead_states = set()              # states known to always lead to death
    stack = [(env.snapshot(), [])]   # (state, actions already tried at this depth)
    actions = []
    sims = 0
    t0 = time.time()
    while stack:
        state, tried = stack[-1]
        if len(tried) == 2 or key(state) in dead_states:
            dead_states.add(key(state))
            stack.pop()
            if actions:
                actions.pop()
            continue
        a = 0 if 0 not in tried else 1
        tried.append(a)
        env.reset(start_state=state)
        env.step(a)
        sims += 1
        s = env.snapshot()
        if s.won:
            actions.append(a)
            return actions, sims, time.time() - t0
        if s.dead or key(s) in dead_states:
            continue
        actions.append(a)
        stack.append((s, []))
    return None, sims, time.time() - t0


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "stereo_lite"
    acts, sims, dt = solve(name)
    if acts is None:
        print(f"Level '{name}' is IMPOSSIBLE ({sims} steps simulated)")
    else:
        print(f"Level '{name}' beaten in {len(acts)} decisions, {sims} steps simulated, {dt:.2f}s")
        with open(f"solution_search_{name}.txt", "w") as f:
            f.write("".join(map(str, acts)))
