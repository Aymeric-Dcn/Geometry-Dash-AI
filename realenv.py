"""
Environment for the REAL Geometry Dash, through the "GD AI Bridge" Geode mod (see mod/).

Same API as GDEnv in gdsim.py, so the training code does not change:
    env = RealGDEnv()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)   # 0 = nothing, 1 = button held

The mod runs the game in lockstep: one step = exactly 1/60 s of game time (4 physics ticks),
like TICKS_PER_STEP = 4 in the simulator. Units are converted to blocks (1 block = 30 GD units)
so the numbers look like the simulator's.
"""
import socket
from dataclasses import dataclass, replace

from gdsim import PlayerState

UNITS_PER_BLOCK = 30.0
CHECKPOINT_SEARCH = 30      # how far back we look for a valid checkpoint position (steps)
STEP_SECONDS = 1.0 / 60.0
MAX_FALL_SECONDS = 1.5      # cube mode only: flying at full vertical speed for longer than this = lost
                            # in the sky (upside down with no ceiling), counted as a death
TERMINAL_VY = 26.5          # blocks/s: GD caps the cube's vertical speed at about 27
GAMEMODES = ["cube", "ship", "ball", "ufo", "wave", "robot", "spider", "swing"]


@dataclass
class RawState:
    """State exactly as sent by the mod (GD units)."""
    x: float
    y: float
    y_velocity: float
    grounded: bool
    dead: bool
    won: bool
    percent: float
    mode: int

    @staticmethod
    def parse(line: str) -> "RawState":
        parts = line.split()
        if not parts or parts[0] != "S":
            raise RuntimeError(f"Unexpected answer from the game: {line!r}")
        x, y, vy, ground, dead, won, percent, mode = parts[1:9]
        return RawState(float(x), float(y), float(vy), ground == "1", dead == "1", won == "1",
                        float(percent), int(mode))


@dataclass(frozen=True)
class Checkpoint:
    """A place to restart from in practice mode.

    step:   index of the agent's step right after the respawn (same numbering as an attempt
            started from the beginning, so the Q-table keys stay consistent)
    prefix: the actions that lead from the start of the level to the checkpoint. The game is
            deterministic, so replaying them always brings the player to the same place.
    x, y:   expected position (blocks) right after the respawn, to detect a lost or badly
            restored checkpoint (e.g. wrong gravity)
    """
    step: int
    prefix: tuple
    x: float = 0.0
    y: float = 0.0
    mode: int = 0


class RealGDEnv:
    def __init__(self, host="127.0.0.1", port=22222, speed=20, max_seconds=120, max_fall=MAX_FALL_SECONDS):
        """speed = game steps per rendered frame (1 = real time, 20 = up to 20x faster).
        max_fall = in cube mode, how long (seconds) the player may move up or down at full speed
        before the attempt is stopped and counted as a death. Upside down with no ceiling, GD
        lets the cube fly away for a long time, and that useless run could look like the best one
        (Clutterfunk). A height limit does not work: legit runs climb 15+ blocks with orbs
        (Clubstep) or fall a long way. Other modes (ship, UFO...): no limit. 0 = off."""
        self.max_fall = max_fall
        self._full_speed_steps = 0
        try:
            self.sock = socket.create_connection((host, port), timeout=5)
        except OSError as e:
            raise SystemExit(
                f"Cannot connect to the game on {host}:{port} ({e}).\n"
                "Is Geometry Dash running with the GD AI Bridge mod enabled?") from None
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._buffer = b""
        self.max_steps = int(max_seconds / STEP_SECONDS)
        self.raw = None
        self.state = PlayerState()
        self.steps = 0
        self._origin = (0.0, 0.0)
        self._waiting_hint = False
        self._practice = False
        self._active_cp = None          # the Checkpoint currently placed in GD, if any
        self.set_speed(speed)

    # --- low-level protocol ------------------------------------------------------------
    def _command(self, text: str) -> str:
        self.sock.sendall((text + "\n").encode("ascii"))
        return self._read_line()

    def _read_line(self) -> str:
        self.sock.settimeout(3)
        while b"\n" not in self._buffer:
            try:
                chunk = self.sock.recv(4096)
            except socket.timeout:
                # The mod only answers while a level is open and not paused
                if not self._waiting_hint:
                    print("Waiting for the game... (open a level in GD and make sure it is not paused)")
                    self._waiting_hint = True
                continue
            if not chunk:
                raise ConnectionError("The game closed the connection")
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        return line.decode("ascii").strip()

    def _expect_ok(self, text: str) -> str:
        answer = self._command(text)
        if not answer.startswith("OK"):
            raise RuntimeError(f"{text!r} refused by the game: {answer!r}")
        return answer

    def set_speed(self, steps_per_frame: int):
        self._expect_ok(f"SPEED {int(steps_per_frame)}")

    def load_level(self, level_id: int):
        """Open an official level (1 = Stereo Madness ... 22 = Dash). Needs mod v0.4.0+."""
        self._expect_ok(f"LEVEL {int(level_id)}")
        self._practice = False          # a new level starts in normal mode, without checkpoints
        self._active_cp = None
        self._waiting_hint = False

    # --- conversion to the simulator's units ------------------------------------------
    def _to_state(self, raw: RawState, prev: PlayerState | None) -> PlayerState:
        x = (raw.x - self._origin[0]) / UNITS_PER_BLOCK
        y = (raw.y - self._origin[1]) / UNITS_PER_BLOCK
        # Vertical speed in blocks/s, measured from the movement (same meaning as in the simulator)
        vy = 0.0 if prev is None else (y - prev.y) / STEP_SECONDS
        return PlayerState(x, y, vy, raw.grounded, raw.dead, self._real_win(raw), self.steps * 4, raw.mode)

    @staticmethod
    def _real_win(raw: RawState) -> bool:
        # GD's end-of-level animation calls "level complete" about a second later. If the level is
        # restarted in between, that call can land in the NEXT attempt and look like a win.
        # A win only counts if the game also says we are (almost) at 100%.
        return raw.won and raw.percent >= 99.0

    def obs(self, s=None):
        s = s or self.state
        return (s.x, s.y, s.vy, float(s.grounded))

    def progress(self, s=None):
        return 1.0 if self.state.won else min(0.99, self.raw.percent / 100.0)

    # --- restarts ------------------------------------------------------------------------
    def _restart(self):
        """RESET, then wait until the player actually moves forward.

        After a restart (or a respawn at a checkpoint), GD keeps the player still for a few
        rendered frames. The number of frozen steps depends on real frames, not on game time,
        so it varies. Waiting for the first step that moves the player forward makes every
        attempt start from exactly the same state (the first step is always "do nothing").
        """
        self.raw = RawState.parse(self._command("RESET"))
        prev_x = self.raw.x
        for _ in range(1000):
            self.raw = RawState.parse(self._command("STEP 0"))
            if self.raw.x > prev_x:
                return
            prev_x = self.raw.x
        raise RuntimeError("The player never started moving after the restart")

    def _set_practice(self, on: bool):
        if self._practice != on:
            if self.raw is not None and self.raw.dead:
                # Never switch practice mode while the player is dead: it crashed Geometry Dash
                self._command("RESET")
            self._expect_ok(f"PRACTICE {1 if on else 0}")
            self._practice = on
            self._active_cp = None

    def normal_mode(self):
        """Leave practice mode whatever the game's state (e.g. after a training stopped with Ctrl+C):
        restart first (the player may be dead), then switch, then remove our checkpoints."""
        self._command("RESET")
        self._expect_ok("PRACTICE 0")
        self._practice = False
        self._active_cp = None

    def _place_checkpoint(self, cp: Checkpoint):
        """Replay the prefix from the start of the level, then place a GD checkpoint there."""
        self._expect_ok("CLEARCP")
        self._active_cp = None
        self._restart()
        self._origin = (self.raw.x, self.raw.y)
        for a in cp.prefix:
            self.raw = RawState.parse(self._command(f"STEP {a}"))
            if self.raw.dead:
                raise RuntimeError("Died while replaying the way to a checkpoint (game not deterministic?)")
        self._expect_ok("CHECKPOINT")
        self._active_cp = cp

    def make_checkpoint(self, step, actions, states):
        """Checkpoint near `step` along a known run (actions + states from the same attempt).

        Only positions where the cube is on the ground and the next action is "do nothing" are
        used: a checkpoint in the air could be a hopeless spot (already falling into a spike),
        and the "do nothing" condition makes the respawn reproduce the original run exactly.
        Returns None if no valid position is found.
        """
        for k in range(min(step, len(actions) - 1), max(0, step - CHECKPOINT_SEARCH), -1):
            if states[k].grounded and actions[k] == 0:
                return Checkpoint(step=k + 1, prefix=tuple(actions[:k]), x=states[k + 1].x, y=states[k + 1].y,
                                  mode=states[k + 1].mode)
        return None

    # --- Gym-like API -------------------------------------------------------------------
    def reset(self, start_state=None):
        """start_state=None: start of the level. start_state=Checkpoint: practice-mode respawn."""
        self._full_speed_steps = 0
        if start_state is None:
            if self._practice:
                self._expect_ok("CLEARCP")      # with no checkpoint, RESET goes back to the start
                self._active_cp = None
            self._restart()
            self._origin = (self.raw.x, self.raw.y)     # positions are measured from the spawn
            self.steps = 0
        else:
            cp = start_state
            self._set_practice(True)
            for attempt in range(3):
                if self._active_cp != cp:
                    self._place_checkpoint(cp)
                self._restart()                          # respawn at the checkpoint
                x = (self.raw.x - self._origin[0]) / UNITS_PER_BLOCK
                y = (self.raw.y - self._origin[1]) / UNITS_PER_BLOCK
                if abs(x - cp.x) < 1e-3 and abs(y - cp.y) < 1e-3 and self.raw.mode == cp.mode:
                    break
                # Not where we expected: the checkpoint was lost (or the game is not deterministic).
                # Place it again; if it keeps failing, stop rather than learn from wrong data.
                self._active_cp = None
            else:
                raise RuntimeError(f"Respawn at x={x:.3f}, y={y:.3f} ({GAMEMODES[self.raw.mode]}) instead of "
                                   f"{cp.x:.3f}, {cp.y:.3f} ({GAMEMODES[cp.mode]}): "
                                   "checkpoints are not reproducible on this level")
            self.steps = cp.step
        self.state = self._to_state(self.raw, None)
        return self.obs(), {}

    def step(self, action):
        prev = self.state
        self.raw = RawState.parse(self._command(f"STEP {1 if action else 0}"))
        self.steps += 1
        s = self._to_state(self.raw, prev)
        if self.lost_in_the_sky(s):
            s = replace(s, dead=True)
        self.state = s
        # Same reward as the simulator: progress in blocks, -10 on death, +50 on win
        reward = s.x - prev.x
        if s.dead:
            reward -= 10.0
        if s.won:
            reward += 50.0
        terminated = s.dead or s.won
        truncated = self.steps >= self.max_steps and not terminated
        info = {"progress": self.progress(), "dead": s.dead, "won": s.won,
                "mode": GAMEMODES[self.raw.mode] if 0 <= self.raw.mode < len(GAMEMODES) else "?"}
        return self.obs(), reward, terminated, truncated, info

    def lost_in_the_sky(self, s):
        """Cube mode: True if the player has been moving at full vertical speed for too long."""
        if self.raw.mode == 0 and not s.grounded and abs(s.vy) >= TERMINAL_VY:
            self._full_speed_steps += 1
        else:
            self._full_speed_steps = 0
        return (bool(self.max_fall) and not s.dead and not s.won
                and self._full_speed_steps * STEP_SECONDS > self.max_fall)

    def snapshot(self):
        return self.state

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass
