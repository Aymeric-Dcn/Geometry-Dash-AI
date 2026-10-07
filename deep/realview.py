"""
The real game seen as a grid, for the neural network (same idea as sim/vision.py).

The mod's VIEW command lists the objects around the player with their hitbox. This module draws
them on a small grid that moves with the player: COLS columns from 1 block behind to COLS-2 ahead,
ROWS rows from BELOW blocks under the player's feet to ROWS-BELOW-1 above (GD levels use the height
much more than the simulator: spike pits, platforms several blocks below a jump). Three channels:
    0  solid    blocks, platforms, slopes, breakable blocks (the ground too)
    1  hazard   spikes, saws and other things that kill
    2  special  orbs, pads and portals (things that change the movement)
Everything else (coins, triggers) is ignored.
"""
import numpy as np

COLS, ROWS, BELOW = 12, 10, 5
SOLID = {0, 21, 25}
HAZARD = {2, 48}
SPECIAL = {3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 26, 27, 28, 29, 32, 33, 34,
           35, 36, 37, 38, 41, 42, 43, 44, 47}
MIN_OVERLAP = 0.05          # an object must cover at least this much of a cell (in blocks) to count


def channel(gd_type):
    if gd_type in SOLID:
        return 0
    if gd_type in HAZARD:
        return 1
    if gd_type in SPECIAL:
        return 2
    return None


def grid(objects, height_above_ground=None):
    """objects: list from RealGDEnv.view(). height_above_ground: the player's height in blocks
    (state.y), to draw the ground; None = no ground drawn."""
    g = np.zeros((3, ROWS, COLS), dtype=np.float32)
    for t, x, y, w, h in objects:
        ch = channel(t)
        if ch is None:
            continue
        for c in range(COLS):
            x0 = c - 1                                    # this column covers [x0, x0 + 1) blocks
            if min(x + w, x0 + 1) - max(x, x0) < MIN_OVERLAP:
                continue
            for r in range(ROWS):
                y0 = r - BELOW
                if min(y + h, y0 + 1) - max(y, y0) >= MIN_OVERLAP:
                    g[ch, r, c] = 1.0
    if height_above_ground is not None:
        ground_top = -height_above_ground                # relative to the player's feet
        for r in range(ROWS):
            if r - BELOW <= ground_top - MIN_OVERLAP:      # same rule as objects: enough overlap
                g[0, r, :] = 1.0
    return g


def draw(g):
    """The grid as text, top row first: # solid, ^ hazard, o special, @ the player."""
    lines = []
    for r in reversed(range(ROWS)):
        row = ""
        for c in range(COLS):
            if (c, r) == (1, BELOW):
                row += "@"
            elif g[1, r, c]:
                row += "^"
            elif g[0, r, c]:
                row += "#"
            elif g[2, r, c]:
                row += "o"
            else:
                row += "."
        lines.append(row)
    return "\n".join(lines)
