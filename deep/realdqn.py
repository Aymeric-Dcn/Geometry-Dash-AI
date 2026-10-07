"""
Phase 2 in the REAL game: the neural network that sees (deep/realview.py) learns to play official
levels, then is tested on a level it has never played. Needs mod v0.5.1+.

Open ANY level in GD first (the mod only listens while a level is open), then for example:
    python -m deep.realdqn --train 1,2,3,4,5 --valid 6 --test 7
    python -m deep.realdqn --train 1-5 --valid 6 --test 7 --steps 400000
    python -m deep.realdqn --eval deep/dqn_real.npz --test 7        # only test a saved network

Where attempts start: the blind agent already beat these levels, so we know a path through each
of them. Practice-mode checkpoints are placed at random points of that path, and the network plays
a few attempts from each: it sees every part of a level early on, not only the beginning (the
same trick as the random starts on the simulator). The path is only used to place checkpoints:
the network never sees the actions of the blind agent.

Levels are chosen by their number (1 = Stereo Madness ... 22 = Dash). Training levels must have a
winning run (results/<level>/win.txt or results/runs/...); validation and test levels do not need one,
so the network can be tested on levels the blind agent never beat (15, 17, 18, 19, 21, 22).
"""
import argparse
import random
import time

import numpy as np

from deep.dqn import Replay
from deep.mlp import MLP
from deep.realview import COLS, ROWS, grid
from realenv import GAMEMODES, RealGDEnv
from replay_wins import find_win
from run_batch import LEVELS, parse_levels

SIZE = 3 * ROWS * COLS + 5 + len(GAMEMODES)


def frac_x(env):
    """The player's position inside its block along x, on GD's own block grid (objects sit on
    multiples of 30 units; env.raw.x is the centre of the player, whose hitbox is 30 wide)."""
    return ((env.raw.x - 15.0) / 30.0) % 1.0


def observe(env, held):
    """The network's input: the grid around the player, its position inside its block, its vertical
    speed, whether it touches the ground, whether the button is already held (orbs need a NEW press),
    and its game mode."""
    fx, fy = frac_x(env), env.state.y % 1.0
    g = grid(env.view(), env.state.y, fx)
    mode = np.zeros(len(GAMEMODES), np.float32)
    if 0 <= env.raw.mode < len(GAMEMODES):
        mode[env.raw.mode] = 1.0
    extra = np.array([fx, fy, env.state.vy / 30.0, float(env.state.grounded), float(held)], np.float32)
    return np.concatenate([g.ravel(), extra, mode])


class Course:
    """One official level: its known path and the checkpoints that can be placed along it."""

    def __init__(self, env, level_id, need_path=True):
        self.id, self.name = level_id, LEVELS[level_id]
        self.checkpoints, self.bad, self.length = [], set(), 0
        path = find_win(self.name)
        if path is None:
            if need_path:
                raise SystemExit(f"No winning run for {self.name}: beat it with the blind agent first "
                                 "(training levels need one; validation and test levels do not).")
            return
        self.actions = [int(c) for c in open(path).read().strip()]
        env.load_level(level_id)
        env.normal_mode()
        env.reset()
        states = [env.state]
        for a in self.actions:
            env.step(a)
            states.append(env.state)
            if env.raw.dead or env.raw.won:
                break
        self.length = len(states) - 1
        cands = [env.make_checkpoint(i, self.actions, states) for i in range(60, self.length - 60, 45)]
        self.checkpoints = [c for c in dict.fromkeys(cands) if c is not None]


def greedy_run(env, net, course, max_steps=8000):
    """Play the level from the start without exploration: progress reached (0..1) and win."""
    env.load_level(course.id)
    env.normal_mode()
    env.reset()
    held, info = 0, {"progress": 0.0, "won": False}
    for _ in range(max_steps):
        q = net.forward(observe(env, held)[None])[0]
        held = int(q[1] > q[0])
        _, _, term, trunc, info = env.step(held)
        if term or trunc:
            break
    return info["progress"], info["won"]


def evaluate(env, net, courses):
    return [greedy_run(env, net, c) for c in courses]


def fmt(results, courses):
    return "  ".join(f"{c.name}: {'WON' if w else f'{100 * p:.0f}%'}" for c, (p, w) in zip(courses, results))


def train(train_ids, valid_ids, test_ids, steps=300_000, gamma=0.98, lr=5e-4, lr_end=5e-5, batch=64,
          eps_end=0.02, eps_steps=60_000, target_every=1000, warmup=3000, eval_every=25_000,
          attempts_per_checkpoint=6, attempt_steps=600, start_from_zero=0.15, seed=0,
          save="deep/dqn_real.npz", speed=20):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    t0 = time.time()
    env = RealGDEnv(speed=speed, max_fall=1.5)
    env.air_checkpoints = True
    print("Reading the known paths and placing the checkpoints...")
    courses = [Course(env, i) for i in train_ids]
    valid = [Course(env, i, need_path=False) for i in valid_ids]
    test = [Course(env, i, need_path=False) for i in test_ids]
    for c in courses:
        print(f"  {c.name}: {c.length} steps, {len(c.checkpoints)} possible starting points")

    net = MLP([SIZE, 128, 64, 2], seed=seed)
    target = MLP([SIZE, 128, 64, 2], seed=seed)
    target.copy_from(net)
    mem = Replay(150_000, SIZE)
    print(f"network: {SIZE} inputs -> 128 -> 64 -> 2, {net.n_params():,} weights ({time.time() - t0:.0f}s)")

    t, attempts, best_score = 0, 0, -1.0
    while t < steps:
        course = courses[rng.integers(len(courses))]
        env.load_level(course.id)
        env.normal_mode()
        ok = [c for c in course.checkpoints if c not in course.bad]
        start = None if (not ok or rng.random() < start_from_zero) else ok[rng.integers(len(ok))]
        for _ in range(attempts_per_checkpoint):
            try:
                env.reset(start_state=start)
            except RuntimeError:                       # GD could not reproduce this checkpoint
                course.bad.add(start)
                start = None
                env.normal_mode()
                env.reset()
            attempts += 1
            held = 0
            s = observe(env, held)
            for _ in range(attempt_steps):
                eps = max(eps_end, 1.0 - (1.0 - eps_end) * t / eps_steps)
                if rng.random() < eps:
                    a = int(rng.integers(2))
                else:
                    q = net.forward(s[None])[0]
                    a = int(q[1] > q[0])
                _, r, term, trunc, _ = env.step(a)
                held = a
                s2 = observe(env, held)
                mem.add(s, a, r, s2, float(term))
                s = s2
                t += 1
                if t > warmup:
                    bs, ba, br, bs2, bd = mem.sample(batch, rng)
                    a2 = net.forward(bs2).argmax(axis=1)
                    q2 = target.forward(bs2)[np.arange(batch), a2]
                    y = br + gamma * (1.0 - bd) * q2
                    q = net.forward(bs, keep=True)
                    err = np.clip(q[np.arange(batch), ba] - y, -1.0, 1.0)
                    grad = np.zeros_like(q)
                    grad[np.arange(batch), ba] = err / batch
                    net.backward(grad)
                    net.step(lr + (lr_end - lr) * min(1.0, t / steps))
                if t % target_every == 0:
                    target.copy_from(net)
                if t % eval_every == 0:
                    tr = evaluate(env, net, courses)
                    va = evaluate(env, net, valid)
                    score = sum(p + w for p, w in va) + 0.1 * sum(p for p, _ in tr)
                    mark = ""
                    if score > best_score:
                        best_score, mark = score, "  <- best so far, saved"
                        net.save(save)
                    print(f"step {t:7d}  {(time.time() - t0) / 60:5.1f} min  eps={eps:.2f}  attempts={attempts}\n"
                          f"    trained on: {fmt(tr, courses)}\n"
                          f"    never trained on (validation): {fmt(va, valid)}{mark}", flush=True)
                    env.load_level(course.id)          # back to the level being trained
                    env.normal_mode()
                    break                              # this attempt was interrupted by the evaluation
                if term or trunc:
                    break

    net.load(save)
    if test:
        res = evaluate(env, net, test)
        print(f"\nBest network on levels it has NEVER played: {fmt(res, test)}")
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="1-5", help="levels to train on, e.g. 1-5 or 1,3,5")
    p.add_argument("--valid", default="6", help="levels to pick the best network (never trained on)")
    p.add_argument("--test", default="7", help="levels for the final result (never trained on, never used to pick)")
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default="deep/dqn_real.npz")
    p.add_argument("--eval", help="only evaluate this saved network on --test")
    args = p.parse_args()
    if args.eval:
        env = RealGDEnv(speed=20, max_fall=1.5)
        net = MLP([SIZE, 128, 64, 2])
        net.load(args.eval)
        tests = [Course(env, i, need_path=False) for i in parse_levels(args.test)]
        print(fmt(evaluate(env, net, tests), tests))
        env.close()
        return
    ids = [parse_levels(x) if x else [] for x in (args.train, args.valid, args.test)]
    overlap = set(ids[0]) & (set(ids[1]) | set(ids[2]))
    if overlap:
        raise SystemExit(f"Levels {sorted(overlap)} are in the training set AND in validation/test.")
    train(*ids, steps=args.steps, seed=args.seed, save=args.save)


if __name__ == "__main__":
    main()
