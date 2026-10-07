"""
What an agent that SEES the level gets: a small grid around the player, like a low-resolution
camera, plus its own vertical speed. Unlike the blind agent, it knows nothing about WHERE it is
in the level: only what is around it. That is what lets it play a level it has never seen.

Grid: COLS columns from 1 block behind the player to COLS-2 blocks ahead, ROWS rows from 2 blocks
below its feet to ROWS-3 blocks above. Two channels: solid blocks and spikes. The ground counts as
solid blocks. The grid is aligned on the level's blocks; the player's offset inside its block
(fractional x and y) is given separately so that the agent knows exactly where it stands.
"""
import math

import numpy as np

from sim.gdsim import JUMP_V

COLS, ROWS = 10, 6
SIZE = 2 * COLS * ROWS + 4


def grid_obs(level, s):
    x0 = int(math.floor(s.x)) - 1
    y0 = int(math.floor(s.y)) - 2
    g = np.zeros((2, ROWS, COLS), dtype=np.float32)
    for c in range(COLS):
        col = x0 + c
        for bx, by in level.blocks.get(col, ()):
            r = by - y0
            if 0 <= r < ROWS:
                g[0, r, c] = 1.0
        for sx, sy in level.spikes.get(col, ()):
            r = sy - y0
            if 0 <= r < ROWS:
                g[1, r, c] = 1.0
    for r in range(ROWS):
        if y0 + r < 0:                                   # below the ground: solid
            g[0, r, :] = 1.0
    extra = np.array([s.x - math.floor(s.x), s.y - math.floor(s.y), s.vy / JUMP_V, float(s.grounded)],
                     dtype=np.float32)
    return np.concatenate([g.ravel(), extra])
