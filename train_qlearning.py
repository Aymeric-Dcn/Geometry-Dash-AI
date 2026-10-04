"""
"Blind" agent trained with tabular Q-learning: it sees no obstacle at all.
It learns purely by trial and error: "here, jumping killed me / didn't kill me".

Usage:
    python train_qlearning.py                         # stereo_lite level, practice mode
    python train_qlearning.py --mode normal           # always restarts from the beginning
    python train_qlearning.py --obs x                 # only knows its x position (even blinder)
    python train_qlearning.py --level tuto --seed 3

"Practice" mode mimics GD checkpoints: when the agent gets stuck somewhere,
it restarts from a checkpoint placed a bit earlier instead of from the very start.
"""
import argparse
import json
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
          alpha=0.5, gamma=0.99, eps_start=0.3, eps_end=0.02, eval_every=50, seed=0, verbose=True):
    random.seed(seed)
    env = GDEnv(level)
    Q = defaultdict(lambda: [0.0, 0.0])
    history = []                 # (episode, progress of the no-exploration run)
    checkpoints = []             # (step, state) along the best known run
    best_greedy = 0.0
    total_sim_steps = 0
    t0 = time.time()

    for ep in range(1, episodes + 1):
        eps = max(eps_end, eps_start * (1 - ep / (0.5 * episodes)))

        # Starting point: start of the level, or a checkpoint (practice mode)
        step, start = 0, None
        if mode == "practice" and checkpoints and random.random() < 0.8:
            step, start = random.choice(checkpoints[-3:])   # one of the last checkpoints before getting stuck
        env.reset(start_state=start)

        while True:
            key = make_key(obs_mode, step, env.state)
            a = random.randint(0, 1) if random.random() < eps else greedy(Q, key)
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
            if prog > best_greedy:
                best_greedy = prog
                # Place checkpoints every ~0.5 s along this run, like in practice mode
                checkpoints = [(i, states[i]) for i in range(0, len(states) - 1, 30)]
            if verbose and (ep % (eval_every * 20) == 0 or info["won"]):
                print(f"ep. {ep:6d}  eps={eps:.3f}  agent progress = {100*prog:5.1f}%  "
                      f"(best {100*best_greedy:5.1f}%)  known states={len(Q)}")
            if info["won"]:
                dt = time.time() - t0
                if verbose:
                    print(f"\n>>> LEVEL BEATEN after {ep} episodes, {total_sim_steps} steps simulated, {dt:.1f}s")
                return {"won": True, "episodes": ep, "sim_steps": total_sim_steps, "seconds": dt,
                        "actions": acts, "history": history}

    return {"won": False, "episodes": episodes, "sim_steps": total_sim_steps,
            "seconds": time.time() - t0, "actions": run_greedy(env, Q, obs_mode)[0], "history": history}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--level", default="stereo_lite")
    p.add_argument("--mode", choices=["practice", "normal"], default="practice")
    p.add_argument("--obs", choices=["state", "x"], default="state")
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    res = train(args.level, args.mode, args.obs, args.episodes, seed=args.seed)
    tag = f"{args.level}_{args.mode}_{args.obs}"
    with open(f"solution_qlearning_{tag}.txt", "w") as f:
        f.write("".join(map(str, res["actions"])))
    with open(f"history_{tag}.json", "w") as f:
        json.dump(res["history"], f)
    if not res["won"]:
        print(f"Not beaten yet after {res['episodes']} episodes (best run saved anyway).")
