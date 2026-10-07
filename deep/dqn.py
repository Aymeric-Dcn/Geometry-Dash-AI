"""
Phase 2: an agent that SEES the level (a small grid around the player, sim/vision.py) and learns
with a neural network (Deep Q-Network, deep/mlp.py), on the simulator.

The question: after training on random levels, can it beat levels it has NEVER seen?
(The blind Q-learning agent cannot, by construction: it memorises "at time t, jump".)

    python -m deep.dqn                          # train on 200 random levels, test on 50 others
    python -m deep.dqn --steps 300000 --train-levels 50
    python -m deep.dqn --eval deep/dqn_sim.npz  # only test a saved network

DQN in short: the network predicts, for the current view, the value Q of each action (total
future reward). The agent mostly plays the best action, sometimes a random one (exploration).
Every move is stored in a replay memory; the network is trained on random batches of past moves,
towards the target  reward + gamma * Q_target(next view, best next action)  (Double DQN), where
Q_target is a copy of the network that is only refreshed from time to time (stability).
"""
import argparse
import math
import random
import time

import numpy as np

from deep.mlp import MLP
from sim.gdsim import GDEnv, PlayerState
from sim.levelgen import generate
from sim.levels import LEVELS
from sim.vision import SIZE, grid_obs

TEST_SEED0 = 10_000          # test levels use seeds from here: never used for training
VALID_SEED0 = 20_000         # validation levels: used to pick the best network, not to report results


class Replay:
    def __init__(self, capacity, size):
        self.s = np.zeros((capacity, size), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.s2 = np.zeros((capacity, size), np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.capacity, self.n, self.i = capacity, 0, 0

    def add(self, s, a, r, s2, done):
        k = self.i
        self.s[k], self.a[k], self.r[k], self.s2[k], self.done[k] = s, a, r, s2, done
        self.i = (k + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self, batch, rng):
        idx = rng.integers(0, self.n, batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.done[idx]


def safe_starts(level):
    """Columns where the player can be put on the ground (nothing on or right around it)."""
    cols = []
    for c in range(2, level.length - 2):
        if all(not level.blocks.get(k) and not level.spikes.get(k) for k in range(c - 1, c + 2)):
            cols.append(c)
    return cols


def play(net, rows, max_steps=4000):
    """One greedy run (no exploration) from the start: progress reached (0..1) and win."""
    env = GDEnv(rows)
    env.reset()
    for _ in range(max_steps):
        q = net.forward(grid_obs(env.level, env.state)[None])[0]
        _, _, term, trunc, info = env.step(int(q[1] > q[0]))
        if term or trunc:
            break
    return info["progress"], info["won"]


def evaluate(net, levels):
    res = [play(net, rows) for rows in levels]
    return sum(w for _, w in res), float(np.mean([p for p, _ in res]))


def train(steps=200_000, n_train=200, n_test=50, gamma=0.98, lr=5e-4, lr_end=5e-5, batch=64, seed=0,
          eps_end=0.02, eps_steps=60_000, target_every=1000, warmup=2000, eval_every=20_000,
          random_start=0.5, save="deep/dqn_sim.npz", verbose=True):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    t0 = time.time()
    train_levels = [generate(s) for s in range(n_train)]
    test_levels = [generate(TEST_SEED0 + s) for s in range(n_test)]
    valid_levels = [generate(VALID_SEED0 + s) for s in range(n_test)]
    starts = [safe_starts(GDEnv(r).level) for r in train_levels]
    if verbose:
        print(f"{n_train} training levels, {n_test} test levels generated in {time.time() - t0:.1f}s")

    net = MLP([SIZE, 128, 64, 2], seed=seed)
    target = MLP([SIZE, 128, 64, 2], seed=seed)
    target.copy_from(net)
    mem = Replay(100_000, SIZE)
    if verbose:
        print(f"network: {SIZE} inputs -> 128 -> 64 -> 2 outputs, {net.n_params():,} weights")

    history = []
    best_valid = -1.0
    env, ep, ep_steps = None, 0, 0
    s = None
    for t in range(1, steps + 1):
        if s is None:                                      # new attempt
            k = rng.integers(n_train)
            env = GDEnv(train_levels[k])
            env.reset()
            if starts[k] and rng.random() < random_start:  # start somewhere in the level
                env.reset(start_state=PlayerState(x=float(rng.choice(starts[k]))))
            s = grid_obs(env.level, env.state)
            ep += 1
        eps = max(eps_end, 1.0 - (1.0 - eps_end) * t / eps_steps)
        if rng.random() < eps:
            a = int(rng.integers(2))
        else:
            q = net.forward(s[None])[0]
            a = int(q[1] > q[0])
        _, r, term, trunc, _ = env.step(a)
        s2 = grid_obs(env.level, env.state)
        mem.add(s, a, r, s2, float(term))
        s = None if (term or trunc) else s2

        if t > warmup:
            bs, ba, br, bs2, bd = mem.sample(batch, rng)
            # Double DQN target: the online network picks the next action, the target network rates it
            a2 = net.forward(bs2).argmax(axis=1)
            q2 = target.forward(bs2)[np.arange(batch), a2]
            y = br + gamma * (1.0 - bd) * q2
            q = net.forward(bs, keep=True)
            err = q[np.arange(batch), ba] - y
            err = np.clip(err, -1.0, 1.0)                  # Huber loss: gradient capped at 1
            grad = np.zeros_like(q)
            grad[np.arange(batch), ba] = err / batch
            net.backward(grad)
            net.step(lr + (lr_end - lr) * t / steps)          # learning rate going down linearly
        if t % target_every == 0:
            target.copy_from(net)

        if t % eval_every == 0 or t == steps:
            # DQN is unstable: keep the network that does best on the VALIDATION levels
            vwins, vprog = evaluate(net, valid_levels)
            twins, tprog = evaluate(net, train_levels[:n_test])
            score = vwins + vprog
            mark = ""
            if score > best_valid:
                best_valid, mark = score, "  <- best so far, saved"
                if save:
                    net.save(save)
            history.append((t, twins, vwins))
            if verbose:
                print(f"step {t:7d}  {time.time() - t0:6.0f}s  eps={eps:.2f}  attempts={ep:6d}  "
                      f"training levels: {twins}/{n_test} ({100 * tprog:.0f}%)  "
                      f"unseen (validation): {vwins}/{n_test} ({100 * vprog:.0f}%){mark}", flush=True)

    # Final, honest result: the saved best network on levels it has never seen and that were not
    # used to choose it either
    if save:
        net.load(save)
    wins, prog = evaluate(net, test_levels)
    hp, hw = play(net, LEVELS["stereo_lite"])
    if verbose:
        print(f"\nBest network on {n_test} NEW test levels: {wins}/{n_test} beaten "
              f"(average progress {100 * prog:.0f}%); hand-made stereo_lite: "
              f"{'WON' if hw else f'{100 * hp:.0f}%'}")
    return net, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--train-levels", type=int, default=200)
    p.add_argument("--test-levels", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--random-start", type=float, default=0.5,
                   help="share of attempts that start somewhere in the level instead of at the start")
    p.add_argument("--save", default="deep/dqn_sim.npz")
    p.add_argument("--eval", help="only evaluate this saved network")
    args = p.parse_args()
    if args.eval:
        net = MLP([SIZE, 128, 64, 2])
        net.load(args.eval)
        tests = [generate(TEST_SEED0 + s) for s in range(args.test_levels)]
        wins, prog = evaluate(net, tests)
        hp, hw = play(net, LEVELS["stereo_lite"])
        print(f"new levels: {wins}/{len(tests)} beaten ({100 * prog:.0f}%), "
              f"stereo_lite: {'WON' if hw else f'{100 * hp:.0f}%'}")
        return
    train(args.steps, args.train_levels, args.test_levels, seed=args.seed,
          random_start=args.random_start, save=args.save)


if __name__ == "__main__":
    main()
