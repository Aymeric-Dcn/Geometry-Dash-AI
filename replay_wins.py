"""
Replays the winning runs, one level after another, at real speed (60 steps per second, so the
music stays in sync). Made for recording videos: start your recorder (OBS, or Win+Alt+R with
the Xbox Game Bar), then run this script.

Open ANY level in GD first (the mod only listens while a level is open), then for example:
    python replay_wins.py                    # every level that has a winning run
    python replay_wins.py --levels 1-3       # only some levels
    python replay_wins.py --manual           # mod older than v0.4.0: you open each level yourself

Each run is replayed from the start of the level in NORMAL mode: no checkpoint, no death.
The script writes videos/chapters.txt: the time at which each level starts, ready to paste
into a YouTube description (chapters), counted from the moment the script started.

Automatic recording, one video per level (videos/01_stereo_madness.mp4, ...):
    pip install obsws-python
    In OBS: Tools > WebSocket Server Settings > Enable, note the password. Leave OBS open, with
    a "Game Capture" source on Geometry Dash. Then:
    python replay_wins.py --obs --obs-password <password>
"""
import argparse
import glob
import os
import shutil
import time

from realenv import RealGDEnv
from make_report import NAMES
from run_batch import LEVELS, parse_levels

STEPS_PER_SECOND = 60


def find_win(name):
    """The winning run of a level, wherever it was saved (newest conventions first)."""
    names = {name, name.replace("_", "-")}
    candidates = []
    for n in names:
        candidates += sorted(glob.glob(os.path.join("results", "runs", f"{n}__practice__*_win.txt")))
        candidates += sorted(glob.glob(os.path.join("results", "runs", f"{n}__normal__*_win.txt")))
        candidates.append(os.path.join("results", f"{n}_win.txt"))
    for n in names:   # not copied to results/ yet: the training's best run (a win if the level was beaten)
        candidates += [f"solution_qlearning_{n}_practice_fine.txt", f"solution_qlearning_{n}_practice_state.txt"]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def stamp(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--levels", default="1-22", help="e.g. 1-9 or 1,3,5 (1 = Stereo Madness)")
    p.add_argument("--manual", action="store_true", help="open each level yourself (mod older than v0.4.0)")
    p.add_argument("--pause", type=float, default=2.0, help="seconds to wait before each run")
    p.add_argument("--obs", action="store_true", help="start/stop OBS recording for each level (one video each)")
    p.add_argument("--obs-port", type=int, default=4455)
    p.add_argument("--obs-password", default="")
    args = p.parse_args()

    todo = []
    for level_id in parse_levels(args.levels):
        path = find_win(LEVELS[level_id])
        if path:
            todo.append((level_id, LEVELS[level_id], path))
    if not todo:
        raise SystemExit("No winning run found.")
    print("Runs to replay:")
    for level_id, name, path in todo:
        print(f"  {level_id:2d}. {name:<24} {path}")

    obs = None
    if args.obs:
        try:
            import obsws_python
        except ImportError:
            raise SystemExit("Install the OBS library first: pip install obsws-python") from None
        obs = obsws_python.ReqClient(host="localhost", port=args.obs_port, password=args.obs_password, timeout=5)

    env = RealGDEnv(speed=1)
    os.makedirs("videos", exist_ok=True)
    chapters = []
    t_start = time.time()
    for level_id, name, path in todo:
        if not args.manual:
            try:
                env.load_level(level_id)
            except RuntimeError:
                print("This version of the mod cannot open levels by itself (needs v0.4.0): "
                      "you will open each level yourself.")
                args.manual = True
        if args.manual:
            input(f"\nOpen {name} in GD (leave it running, not paused), then press Enter...")
        env.normal_mode()                     # one attempt, no checkpoint
        if obs:
            obs.start_record()
        time.sleep(args.pause)
        actions = [int(c) for c in open(path).read().strip()]
        env.reset()
        chapters.append(f"{stamp(time.time() - t_start)} {NAMES[name]}")
        t0 = time.time()
        for i, a in enumerate(actions):
            # Real speed: one step every 1/60 s of wall-clock time, whatever the screen's refresh rate
            delay = t0 + i / STEPS_PER_SECOND - time.time()
            if delay > 0:
                time.sleep(delay)
            env.step(a)
            if env.raw.dead or env.raw.won:
                break
        r = env.raw
        print(f"{name}: {'WON' if r.won else f'died at {r.percent:.1f}%'}")
        time.sleep(args.pause)
        if obs:
            recorded = obs.stop_record().output_path
            target = os.path.join("videos", f"{level_id:02d}_{name}{os.path.splitext(recorded)[1]}")
            for _ in range(20):                # OBS needs a moment to finish writing the file
                try:
                    shutil.move(recorded, target)
                    print(f"   video: {target}")
                    break
                except OSError:
                    time.sleep(0.5)
            else:
                print(f"   video left where OBS saved it: {recorded}")

    with open(os.path.join("videos", "chapters.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(chapters) + "\n")
    print("\nChapters written to videos/chapters.txt")
    env.close()


if __name__ == "__main__":
    main()
