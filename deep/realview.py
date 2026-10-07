"""
The real game seen as a grid, for the neural network (same idea as sim/vision.py).

The mod's VIEW command lists the objects around the player with their hitbox. This module draws
them on a small grid around the player: COLS columns from 1 block behind to COLS-2 ahead, ROWS rows
from BELOW blocks under the player's feet to ROWS-BELOW-1 above (GD levels use the height much more
than the simulator: spike pits, platforms several blocks below a jump). Three channels:
    0  solid    blocks, platforms, slopes, breakable blocks (the ground too)
    1  hazard   spikes, saws and other things that kill
    2  special  orbs, pads and portals (things that change the movement)
Everything else (coins, triggers) is ignored.

The grid is aligned on the level's block grid, not on the player: it only jumps by one column when
the player crosses into the next block. In between, the obstacles stay still in the grid and the
player's position inside its block (frac_x, frac_y) tells the network exactly how far they are.
A grid that moved with the player would keep a small spike in the same cell for 4 or 5 steps in a
row: too coarse to time a jump (tested on the simulator: it learned three times worse).
"""
import math

import numpy as np

COLS, ROWS, BELOW = 12, 10, 5
SOLID = {0, 21, 25}
HAZARD = {2, 48}
SPECIAL = {3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 26, 27, 28, 29, 32, 33, 34,
           35, 36, 37, 38, 41, 42, 43, 44, 47}
SHOW = 0.05                 # draw(): a cell is shown as occupied from this value


def channel(gd_type):
    if gd_type in SOLID:
        return 0
    if gd_type in HAZARD:
        return 1
    if gd_type in SPECIAL:
        return 2
    return None


def grid(objects, height_above_ground, frac_x):
    """objects: list from RealGDEnv.view() (positions relative to the player's bottom-left corner).
    height_above_ground: the player's height in blocks (state.y); frac_x: the player's position inside
    its block along x (state.x % 1). Each cell holds how much of an object lies in it (0 to 1; a small
    spike entirely inside a cell counts 1, like a full block)."""
    frac_y = height_above_ground % 1.0
    g = np.zeros((3, ROWS, COLS), dtype=np.float32)
    for t, x, y, w, h in objects:
        ch = channel(t)
        if ch is None:
            continue
        gx, gy = x + frac_x + 1, y + frac_y + BELOW        # in grid units: column c covers [c, c + 1)
        for c in range(max(0, math.floor(gx)), min(COLS, math.ceil(gx + w))):
            ox = min(gx + w, c + 1) - max(gx, c)
            if ox <= 0:
                continue
            for r in range(max(0, math.floor(gy)), min(ROWS, math.ceil(gy + h))):
                oy = min(gy + h, r + 1) - max(gy, r)
                if oy > 0:
                    g[ch, r, c] = min(1.0, g[ch, r, c] + (ox / min(w, 1.0)) * (oy / min(h, 1.0)))
    ground_top = BELOW + frac_y - height_above_ground      # the ground's top, in grid units
    for r in range(ROWS):
        g[0, r, :] = np.maximum(g[0, r, :], min(1.0, max(0.0, ground_top - r)))
    return g


def player_cell(frac_x, height_above_ground):
    return 1, BELOW


def draw(g):
    """The grid as text, top row first: # solid, ^ hazard, o special, @ the player's cell."""
    lines = []
    for r in reversed(range(ROWS)):
        row = ""
        for c in range(COLS):
            if (c, r) == (1, BELOW):
                row += "@"
            elif g[1, r, c] >= SHOW:
                row += "^"
            elif g[0, r, c] >= 0.25:
                row += "#"
            elif g[2, r, c] >= SHOW:
                row += "o"
            else:
                row += "."
        lines.append(row)
    return "\n".join(lines)
