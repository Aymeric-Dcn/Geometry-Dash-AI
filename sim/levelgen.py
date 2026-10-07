"""
Random level generator for the simulator, to train and test an agent that SEES the level.

A level is a row of obstacle patterns (the ones of the hand-made levels: spikes, blocks, platforms,
pillars) separated by random gaps. Every generated level is checked with a backtracking search:
only levels that can actually be beaten are kept.

    from sim.levelgen import generate
    rows = generate(seed=3)            # ASCII rows, same format as sim/levels.py
    python -m sim.levelgen 3           # print level 3
"""
import random
import sys

from sim.gdsim import GDEnv

HEIGHT = 4          # rows of the ASCII map (the last one is on the ground)

# Patterns, written top row first, bottom row on the ground (missing rows on top are empty)
PATTERNS = [
    ["^"], ["^^"], ["^^^"], ["^.^"], ["^..^"],
    ["#"], ["##"], ["###"], ["#^"], ["##^^"], ["####^^"], ["#..#"],
    ["##", "##"], [".##", "###"], ["..#", ".##", "###"], [".#.", "###"],
    ["##^^.##^^"], ["#.^"], ["^#"],
    ["...##", "##..."], ["....###", "###...."],
]


def _place(rows, pattern, x):
    for k, line in enumerate(reversed(pattern)):          # k = 0: ground row
        r = HEIGHT - 1 - k
        for i, c in enumerate(line):
            if c != ".":
                rows[r][x + i] = c


def solvable(rows, max_sims=200_000):
    """Backtracking search (like sim/search_bot.py): can this level be beaten at all?"""
    env = GDEnv(rows)
    env.reset()
    dead, stack, sims = set(), [(env.snapshot(), [])], 0

    def key(s):
        return (round(s.x, 3), round(s.y, 3), round(s.vy, 2), s.grounded)

    while stack and sims < max_sims:
        state, tried = stack[-1]
        if len(tried) == 2 or key(state) in dead:
            dead.add(key(state))
            stack.pop()
            continue
        a = 0 if 0 not in tried else 1
        tried.append(a)
        env.reset(start_state=state)
        env.step(a)
        sims += 1
        s = env.snapshot()
        if s.won:
            return True
        if not s.dead and key(s) not in dead:
            stack.append((s, []))
    return False


def generate(seed, n_patterns=14, min_gap=4, max_gap=10, check=True):
    """A random level that is guaranteed to be beatable (same seed -> same level)."""
    rng = random.Random(seed)
    while True:
        chosen = [rng.choice(PATTERNS) for _ in range(n_patterns)]
        gaps = [rng.randint(min_gap, max_gap) for _ in range(n_patterns)]
        length = 8 + sum(len(p[0]) for p in chosen) + sum(gaps) + 6
        rows = [["."] * length for _ in range(HEIGHT)]
        x = 8                                              # a calm start
        for p, g in zip(chosen, gaps):
            _place(rows, p, x)
            x += len(p[0]) + g
        rows = ["".join(r) for r in rows]
        if not check or solvable(rows):
            return rows


if __name__ == "__main__":
    for line in generate(int(sys.argv[1]) if len(sys.argv) > 1 else 0):
        print(line)
