"""
Mini Geometry Dash simulator (cube mode only).

Units: 1 block = 1.0 (in GD, 1 block = 30 units).
Fixed-step physics (240 Hz), the agent decides every 4 frames (60 Hz).
Fully deterministic: same inputs -> same outcome, every time.

Gymnasium-like API:
    env = GDEnv("stereo_lite")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)   # action 0 = nothing, 1 = button held
"""
from dataclasses import dataclass, replace
import math

from sim.levels import LEVELS

# --- Physics constants (approximation of the cube at normal speed) ---
SPEED = 10.4          # blocks / second (GD normal speed)
GRAVITY = 105.0       # blocks / s²
JUMP_V = 21.5         # jump velocity -> max height ~2.2 blocks, ~4.3 blocks travelled in the air
PHYS_HZ = 240
DT = 1.0 / PHYS_HZ
TICKS_PER_STEP = 4    # the agent decides 60 times per second
PLAYER = 1.0          # player hitbox: 1x1 block square
EPS = 1e-6

# Spike hitbox (smaller than the block, like in GD)
SPIKE_X0, SPIKE_X1 = 0.35, 0.65
SPIKE_Y0, SPIKE_Y1 = 0.15, 0.55


@dataclass(frozen=True)
class PlayerState:
    x: float = 0.0
    y: float = 0.0
    vy: float = 0.0
    grounded: bool = True
    dead: bool = False
    won: bool = False
    tick: int = 0
    mode: int = 0          # game mode (real game only: 0 cube, 1 ship, 3 UFO...)
    yv: float = 0.0        # real game only: GD's own vertical speed (to check checkpoints in the air)


class Level:
    """Level built from an ASCII map (see levels.py)."""

    def __init__(self, rows):
        h = len(rows)
        self.length = max(len(r) for r in rows)
        self.blocks = {}   # column -> list of (x, y)
        self.spikes = {}   # column -> list of (x, y)
        for row_i, row in enumerate(rows):
            y = h - 1 - row_i                      # the last row sits on the ground
            for x, c in enumerate(row):
                if c == "#":
                    self.blocks.setdefault(x, []).append((x, y))
                elif c == "^":
                    self.spikes.setdefault(x, []).append((x, y))
        self.end_x = self.length + 2               # you win by leaving the level

    def near(self, table, x):
        c0 = int(math.floor(x)) - 1
        for c in range(c0, c0 + 3):
            yield from table.get(c, ())


def physics_tick(level: Level, s: PlayerState, hold: bool) -> PlayerState:
    if s.dead or s.won:
        return s
    x, y, vy, grounded = s.x, s.y, s.vy, s.grounded

    # Jump: while the button is held, the cube jumps again as soon as it lands (like in GD)
    if grounded and hold:
        vy = JUMP_V
        grounded = False

    prev_y = y
    vy -= GRAVITY * DT
    x += SPEED * DT
    y += vy * DT
    grounded = False

    # Ground
    if y <= 0.0:
        y, vy, grounded = 0.0, 0.0, True

    # Blocks: you can land on top, but hitting a side or the bottom = death
    for bx, by in level.near(level.blocks, x):
        if x + PLAYER > bx + EPS and x < bx + 1 - EPS and y + PLAYER > by + EPS and y < by + 1 - EPS:
            top = by + 1.0
            if prev_y >= top - 0.05 and vy <= 0:
                y, vy, grounded = top, 0.0, True
            else:
                return replace(s, x=x, y=y, vy=vy, grounded=False, dead=True, tick=s.tick + 1)

    # Spikes
    for sx, sy in level.near(level.spikes, x):
        if (x + PLAYER > sx + SPIKE_X0 and x < sx + SPIKE_X1 and
                y + PLAYER > sy + SPIKE_Y0 and y < sy + SPIKE_Y1):
            return replace(s, x=x, y=y, vy=vy, grounded=False, dead=True, tick=s.tick + 1)

    won = x >= level.end_x
    return PlayerState(x, y, vy, grounded, False, won, s.tick + 1)


class GDEnv:
    """Training environment. The observation is 'blind': the agent sees no obstacle,
    only its own state (position, height, vertical speed, grounded or not)."""

    def __init__(self, level_name="stereo_lite", max_seconds=60):
        self.level_name = level_name
        self.level = Level(LEVELS[level_name])
        self.max_steps = int(max_seconds * PHYS_HZ / TICKS_PER_STEP)
        self.state = PlayerState()
        self.steps = 0

    # Blind observation
    def obs(self, s=None):
        s = s or self.state
        return (s.x, s.y, s.vy, float(s.grounded))

    def progress(self, s=None):
        s = s or self.state
        return min(1.0, s.x / self.level.end_x)

    def reset(self, start_state: PlayerState | None = None):
        """start_state lets you restart from a checkpoint (like GD's practice mode)."""
        self.state = start_state or PlayerState()
        self.steps = 0
        return self.obs(), {}

    def step(self, action):
        s0 = self.state
        s = s0
        for _ in range(TICKS_PER_STEP):
            s = physics_tick(self.level, s, bool(action))
            if s.dead or s.won:
                break
        self.state = s
        self.steps += 1
        # Reward: progress in blocks, big penalty on death, bonus at the end
        reward = s.x - s0.x
        if s.dead:
            reward -= 10.0
        if s.won:
            reward += 50.0
        terminated = s.dead or s.won
        truncated = self.steps >= self.max_steps and not terminated
        info = {"progress": self.progress(s), "dead": s.dead, "won": s.won}
        return self.obs(), reward, terminated, truncated, info

    def snapshot(self):
        return self.state       # the state is immutable: no copy needed


def play(actions, level_name="stereo_lite"):
    """Replay a list of actions and return the trajectory (list of states)."""
    env = GDEnv(level_name)
    env.reset()
    traj = [env.state]
    for a in actions:
        env.step(a)
        traj.append(env.state)
        if env.state.dead or env.state.won:
            break
    return traj
