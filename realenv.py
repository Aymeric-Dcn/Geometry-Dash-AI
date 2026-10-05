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
from dataclasses import dataclass

from gdsim import PlayerState

UNITS_PER_BLOCK = 30.0
STEP_SECONDS = 1.0 / 60.0
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


class RealGDEnv:
    def __init__(self, host="127.0.0.1", port=22222, speed=20, max_seconds=120):
        """speed = game steps per rendered frame (1 = real time, 20 = up to 20x faster)."""
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

    def set_speed(self, steps_per_frame: int):
        answer = self._command(f"SPEED {int(steps_per_frame)}")
        if answer != "OK":
            raise RuntimeError(f"SPEED refused: {answer!r}")

    # --- conversion to the simulator's units ------------------------------------------
    def _to_state(self, raw: RawState, prev: PlayerState | None) -> PlayerState:
        x = (raw.x - self._origin[0]) / UNITS_PER_BLOCK
        y = (raw.y - self._origin[1]) / UNITS_PER_BLOCK
        # Vertical speed in blocks/s, measured from the movement (same meaning as in the simulator)
        vy = 0.0 if prev is None else (y - prev.y) / STEP_SECONDS
        return PlayerState(x, y, vy, raw.grounded, raw.dead, raw.won, self.steps * 4)

    def obs(self, s=None):
        s = s or self.state
        return (s.x, s.y, s.vy, float(s.grounded))

    def progress(self, s=None):
        return 1.0 if self.state.won else min(1.0, self.raw.percent / 100.0)

    # --- Gym-like API -------------------------------------------------------------------
    def reset(self, start_state=None):
        if start_state is not None:
            raise NotImplementedError("Practice mode (start_state) is not supported on the real game yet")
        self.raw = RawState.parse(self._command("RESET"))
        self._origin = (self.raw.x, self.raw.y)     # positions are measured from the spawn point
        self.steps = 0
        self.state = self._to_state(self.raw, None)
        return self.obs(), {}

    def step(self, action):
        prev = self.state
        self.raw = RawState.parse(self._command(f"STEP {1 if action else 0}"))
        self.steps += 1
        s = self._to_state(self.raw, prev)
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

    def snapshot(self):
        return self.state

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass
