"""
Trains the agent on several official levels in a row, opening each level automatically
(needs the GD AI Bridge mod v0.4.0 or newer).

Open ANY level in GD first (the mod only listens while a level is open), then for example:
    python run_batch.py --levels 1-9 --mode normal
    python run_batch.py --levels 1,3,5 --no-reverse --max-minutes 60 --variant no-reverse
    python run_batch.py --levels 1-9 --mode normal --skip-done      # continue an interrupted batch

For each level it writes, in results/runs/:
    <level>__<mode>__<variant>.log         the training log (first line describes the run)
    <level>__<mode>__<variant>_win.txt     the winning run (or _best.txt if not beaten)
and adds one line to results/runs/summary.csv.
Ctrl+C stops the batch cleanly (the current Q-table is saved).
"""
import argparse
import csv
import datetime
import os
import pickle
import sys

from realenv import RealGDEnv
from train_qlearning import train

LEVELS = {
    1: "stereo_madness", 2: "back_on_track", 3: "polargeist", 4: "dry_out", 5: "base_after_base",
    6: "cant_let_go", 7: "jumper", 8: "time_machine", 9: "cycles", 10: "xstep", 11: "clutterfunk",
    12: "theory_of_everything", 13: "electroman_adventures", 14: "clubstep", 15: "electrodynamix",
    16: "hexagon_force", 17: "blast_processing", 18: "theory_of_everything_2",
    19: "geometrical_dominator", 20: "deadlocked", 21: "fingerdash", 22: "dash",
}
RUNS_DIR = os.path.join("results", "runs")
SUMMARY = os.path.join(RUNS_DIR, "summary.csv")
FIELDS = ["level_id", "level", "mode", "variant", "won", "best_progress", "episodes",
          "steps_played", "seconds", "finished_at"]


class Tee:
    """Print to the console AND to a log file."""
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.console = sys.stdout

    def write(self, text):
        self.console.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def parse_levels(text):
    ids = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            ids += list(range(int(a), int(b) + 1))
        else:
            ids.append(int(part))
    unknown = [i for i in ids if i not in LEVELS]
    if unknown:
        raise SystemExit(f"Unknown level ids: {unknown} (official levels are 1 to 22)")
    return ids


def done_runs():
    if not os.path.exists(SUMMARY):
        return set()
    with open(SUMMARY, newline="", encoding="utf-8") as f:
        return {(r["level"], r["mode"], r["variant"]) for r in csv.DictReader(f)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--levels", default="1-9", help="e.g. 1-9 or 1,3,5 (1 = Stereo Madness)")
    p.add_argument("--mode", choices=["practice", "normal"], default="practice")
    p.add_argument("--variant", default=None, help="name for this configuration (default: 'default' "
                                                  "or 'no-reverse' / 'no-hold')")
    p.add_argument("--episodes", type=int, default=200000)
    p.add_argument("--max-minutes", type=float, default=120, help="time limit per level")
    p.add_argument("--speed", type=int, default=20)
    p.add_argument("--eps-start", type=float, default=0.3)
    p.add_argument("--no-reverse", action="store_true")
    p.add_argument("--no-hold", action="store_true")
    p.add_argument("--skip-done", action="store_true", help="skip runs already in summary.csv")
    args = p.parse_args()

    variant = args.variant or ("no-reverse" if args.no_reverse else "no-hold" if args.no_hold else "default")
    os.makedirs(RUNS_DIR, exist_ok=True)
    already = done_runs() if args.skip_done else set()
    env = RealGDEnv(speed=args.speed)

    for level_id in parse_levels(args.levels):
        name = LEVELS[level_id]
        if (name, args.mode, variant) in already:
            print(f"--- {name}: already done, skipped")
            continue
        base = os.path.join(RUNS_DIR, f"{name}__{args.mode}__{variant}")
        tee = Tee(base + ".log")
        sys.stdout = tee
        print(f"RUN level={name} id={level_id} mode={args.mode} variant={variant} "
              f"reverse={not args.no_reverse} hold={not args.no_hold} max_minutes={args.max_minutes} "
              f"started={datetime.datetime.now().isoformat(timespec='seconds')}")
        res = None
        try:
            env.load_level(level_id)
            qtable = f"qtable_{name}_{args.mode}_{variant}.pkl"

            def save(Q, best_acts, history):
                with open(qtable + ".tmp", "wb") as f:
                    pickle.dump(dict(Q), f)
                os.replace(qtable + ".tmp", qtable)

            res = train(name, args.mode, "state", args.episodes, env=env, verbose=True,
                        eps_start=args.eps_start, print_every=200, save_every=200, save_fn=save,
                        focus_margin=30, explore_hold=not args.no_hold,
                        reverse_replay=not args.no_reverse, max_seconds=args.max_minutes * 60)
            save(res["Q"], res["actions"], res["history"])
        except RuntimeError as e:                      # e.g. the game is not deterministic here
            print(f"\nRun failed: {e}")
        finally:
            sys.stdout = tee.console
            tee.close()

        if res is None:
            continue
        best = max([p for _, p in res["history"]] or [0.0])
        if res["won"]:
            best = 1.0
        with open(base + ("_win.txt" if res["won"] else "_best.txt"), "w") as f:
            f.write("".join(map(str, res["actions"])))
        if res.get("interrupted"):                     # not a finished run: not in the summary
            print("Batch stopped by the user.")
            break
        new_file = not os.path.exists(SUMMARY)
        with open(SUMMARY, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
            w.writerow({"level_id": level_id, "level": name, "mode": args.mode, "variant": variant,
                        "won": res["won"], "best_progress": round(best, 4), "episodes": res["episodes"],
                        "steps_played": res["sim_steps"], "seconds": round(res["seconds"], 1),
                        "finished_at": datetime.datetime.now().isoformat(timespec="seconds")})
        print(f"=== {name}: {'BEATEN' if res['won'] else f'best {100 * best:.1f}%'} "
              f"in {res['episodes']} episodes, {res['seconds'] / 60:.1f} min")
    env.close()


if __name__ == "__main__":
    main()
