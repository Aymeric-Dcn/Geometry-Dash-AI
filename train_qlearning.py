"""
"Blind" agent trained with tabular Q-learning: it sees no obstacle at all.
It learns purely by trial and error: "here, jumping killed me / didn't kill me".

Usage:
    python train_qlearning.py                         # stereo_lite level, practice mode
    python train_qlearning.py --mode normal           # always restarts from the beginning
    python train_qlearning.py --obs x                 # only knows its x position (even blinder)
    python train_qlearning.py --level tuto --seed 3
    python train_qlearning.py --env real               # the REAL game, practice mode (GD checkpoints)
    python train_qlearning.py --env real --mode normal # the REAL game, always from the start
    python train_qlearning.py --env real --resume      # continue from the saved Q-table
    python train_qlearning.py --env real --resume-from qtable_real_normal_state.pkl

Ctrl+C stops the training cleanly: the Q-table and the best run so far are saved.

"Practice" mode mimics GD checkpoints: when the agent gets stuck somewhere,
it restarts from a checkpoint placed a bit earlier instead of from the very start.
On the real game, GD's own practice mode is used. Checkpoints are only placed on the ground,
along the best run so far. Exploration is focused on the last steps before the death; when the
record does not move for 500 episodes, this zone doubles and the checkpoint moves back with it.
"""
import argparse
import json
import pickle
import random
import time
from collections import defaultdict

from gdsim import GDEnv, PlayerState


def make_key(obs_mode, step, s: PlayerState):
    if obs_mode == "x":
        return step                                   # only "where am I in the level"
    # "state": its position + its own movement (still blind to obstacles)
    return (step, round(s.y * 4), round(s.vy), s.grounded)


def greedy(Q, key):
    q = Q[key]
    return 1 if q[1] > q[0] else 0


def run_greedy(env, Q, obs_mode):
    """One run without exploration: this is what the agent really knows."""
    env.reset()
    acts, states = [], [env.state]
    step = 0
    while True:
        a = greedy(Q, make_key(obs_mode, step, env.state))
        _, _, term, trunc, info = env.step(a)
        acts.append(a)
        states.append(env.state)
        step += 1
        if term or trunc:
            return acts, states, info


def train(level="stereo_lite", mode="practice", obs_mode="state", episodes=20000,
          alpha=0.5, gamma=0.99, eps_start=0.3, eps_end=0.02, eval_every=50, seed=0, verbose=True,
          env=None, Q_init=None, print_every=1000, save_every=None, save_fn=None,
          focus_margin=None):
    random.seed(seed)
    env = env or GDEnv(level)                 # simulator by default, or any env with the same API
    Q = defaultdict(lambda: [0.0, 0.0])
    if Q_init:
        Q.update(Q_init)                      # resume from a saved table
    history = []                 # (episode, progress of the no-exploration run)
    checkpoints = []             # (step, state) along the best known run
    # Real game: GD checkpoints. One "frontier" checkpoint at a time (placing one costs a replay).
    real_practice = mode == "practice" and hasattr(env, "make_checkpoint")
    frontier = []                # candidate checkpoints along the best run, oldest first
    prev_margin = None
    best_greedy = 0.0
    best_acts = []
    last_record_ep = 0           # episode of the last new record
    total_sim_steps = 0
    t0 = time.time()
    ep = 0

    if Q_init:
        # Resuming: measure the current best run first, so that exploration and checkpoints are
        # focused on the right place from the first episode (instead of exploring everywhere).
        acts, states, info = run_greedy(env, Q, obs_mode)
        best_greedy, best_acts = info["progress"], acts
        if real_practice:
            cands = [env.make_checkpoint(i, acts, states) for i in range(30, len(acts) - 1, 30)]
            frontier = list(dict.fromkeys(c for c in cands if c is not None))
        if verbose:
            print(f"Resumed agent reaches {100 * best_greedy:.1f}% without exploration")

    try:
        for ep in range(1, episodes + 1):
            eps = max(eps_end, eps_start * (1 - ep / (0.5 * episodes)))

            # Focused exploration: no random moves on the part of the level already mastered,
            # explore only in the last `margin` steps of the best run so far. If the record does
            # not move for 500 episodes, the fatal decision may be earlier: the margin doubles
            # (up to 8x, i.e. 4 s with the default 30 steps: GD mistakes are never older than that).
            margin = focus_margin * min(8, 2 ** ((ep - last_record_ep) // 500)) if focus_margin else None
            explore_from = max(0, len(best_acts) - margin) if margin and best_acts else 0
            if verbose and margin and best_acts and prev_margin and margin > prev_margin:
                print(f"ep. {ep:6d}  no new record for a while: exploring the last {margin} steps")
            prev_margin = margin

            # Starting point: start of the level, or a checkpoint (practice mode)
            step, start = 0, None
            if real_practice:
                # Latest checkpoint at least 30 steps (0.5 s) before the exploration zone
                limit = explore_from - 30 if margin else len(best_acts) - 60
                eligible = [c for c in frontier if c.step <= limit]
                if eligible:
                    start = eligible[-1]
                    step = start.step
            elif mode == "practice" and checkpoints and random.random() < 0.8:
                step, start = random.choice(checkpoints[-3:])   # one of the last checkpoints before getting stuck
            env.reset(start_state=start)

            while True:
                key = make_key(obs_mode, step, env.state)
                explore = step >= explore_from and random.random() < eps
                a = random.randint(0, 1) if explore else greedy(Q, key)
                _, r, term, trunc, _ = env.step(a)
                total_sim_steps += 1
                step += 1
                nkey = make_key(obs_mode, step, env.state)
                target = r if term else r + gamma * max(Q[nkey])
                Q[key][a] += alpha * (target - Q[key][a])
                if term or trunc:
                    break

            if ep % eval_every == 0:
                acts, states, info = run_greedy(env, Q, obs_mode)
                total_sim_steps += len(acts)
                prog = info["progress"]
                history.append((ep, prog))
                new_best = prog > best_greedy
                if new_best:
                    best_greedy = prog
                    best_acts = acts
                    last_record_ep = ep
                    # Place checkpoints every ~0.5 s along this run, like in practice mode
                    checkpoints = [(i, states[i]) for i in range(0, len(states) - 1, 30)]
                    if real_practice:
                        cands = [env.make_checkpoint(i, acts, states) for i in range(30, len(acts) - 1, 30)]
                        frontier = list(dict.fromkeys(c for c in cands if c is not None))
                if verbose and (ep % print_every == 0 or new_best or info["won"]):
                    print(f"ep. {ep:6d}  eps={eps:.3f}  agent progress = {100*prog:5.1f}%  "
                          f"(best {100*best_greedy:5.1f}%)  known states={len(Q)}")
                if info["won"]:
                    dt = time.time() - t0
                    if verbose:
                        print(f"\n>>> LEVEL BEATEN after {ep} episodes, {total_sim_steps} steps simulated, {dt:.1f}s")
                    return {"won": True, "episodes": ep, "sim_steps": total_sim_steps, "seconds": dt,
                            "actions": acts, "history": history, "Q": Q}
            if save_fn and save_every and ep % save_every == 0:
                save_fn(Q, best_acts, history)
    except KeyboardInterrupt:
        print(f"\nStopped by the user after {ep} episodes.")

    return {"won": False, "episodes": ep, "sim_steps": total_sim_steps,
            "seconds": time.time() - t0, "actions": best_acts, "history": history, "Q": Q}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--level", default="stereo_lite")
    p.add_argument("--mode", choices=["practice", "normal"], default="practice")
    p.add_argument("--obs", choices=["state", "x"], default="state")
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env", choices=["sim", "real"], default="sim")
    p.add_argument("--speed", type=int, default=20, help="real game only: steps per rendered frame")
    p.add_argument("--resume", action="store_true", help="start from the saved Q-table")
    p.add_argument("--resume-from", help="start from this Q-table file (e.g. one trained in normal mode)")
    p.add_argument("--eps-start", type=float, default=0.3, help="initial exploration (lower it when resuming)")
    p.add_argument("--focus", type=int, default=None,
                   help="explore only in the last N steps of the best run (default: 30 on the real game, off on the simulator; doubles when stuck)")
    args = p.parse_args()

    env = None
    if args.env == "real":
        from realenv import RealGDEnv
        args.level = "real"
        env = RealGDEnv(speed=args.speed)

    tag = f"{args.level}_{args.mode}_{args.obs}"
    qtable_path = f"qtable_{tag}.pkl"

    def save(Q, best_acts, history):
        with open(qtable_path, "wb") as f:
            pickle.dump(dict(Q), f)
        with open(f"solution_qlearning_{tag}.txt", "w") as f:
            f.write("".join(map(str, best_acts)))
        with open(f"history_{tag}.json", "w") as f:
            json.dump(history, f)

    Q_init = None
    if args.resume or args.resume_from:
        path = args.resume_from or qtable_path
        with open(path, "rb") as f:
            Q_init = pickle.load(f)
        print(f"Resuming from {path} ({len(Q_init)} known states)")

    real = args.env == "real"
    res = train(args.level, args.mode, args.obs, args.episodes, seed=args.seed, env=env,
                eps_start=args.eps_start, Q_init=Q_init,
                print_every=200 if real else 1000, save_every=200 if real else None, save_fn=save,
                focus_margin=args.focus if args.focus is not None else (30 if real else None))
    save(res["Q"], res["actions"], res["history"])
    print(f"Saved: {qtable_path}, solution_qlearning_{tag}.txt")
    if not res["won"]:
        print(f"Not beaten yet after {res['episodes']} episodes (best run saved anyway).")
