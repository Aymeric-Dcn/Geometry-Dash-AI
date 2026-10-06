"""
ASCII levels. Each character = 1 block. The last row sits on the ground.
    .  empty
    ^  spike
    #  solid block (you can run on it, touching its side = death)
Add your own levels here: just add a new entry to LEVELS.
"""

LEVELS = {
    # Warm-up level: a few isolated spikes
    "tuto": [
        "..........^.........^.........^^........^.........",
    ],

    # Main level: single, double and triple spikes, platforms, stairs
    "stereo_lite": [
        "..........................................................................................................................",
        "...........................................................................##.......................^.....................",
        "..........^.......^.....^^......^...^.....^^^.....####^^......#.^.....##^^.##^^.....^...^^^.....########^^....^^...^...^^^",
    ],
}
