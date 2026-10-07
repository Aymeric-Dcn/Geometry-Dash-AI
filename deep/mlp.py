"""
A small neural network written with NumPy only, so that every step is visible.

    net = MLP([124, 128, 64, 2])       # input size, two hidden layers, 2 outputs (one per action)
    q = net.forward(x)                 # x: (batch, 124) -> (batch, 2)
    net.backward(grad_q)               # gradient of the loss with respect to the outputs
    net.step(lr)                       # Adam update of all the weights

Hidden layers use ReLU (max(0, z)); the output layer is linear (Q-values can be any number).
"""
import numpy as np


class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            # He initialisation: keeps the signal at the same scale through ReLU layers
            self.W.append((rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)).astype(np.float32))
            self.b.append(np.zeros(n_out, dtype=np.float32))
        self.params = self.W + self.b
        self.m = [np.zeros_like(p) for p in self.params]      # Adam: running mean of the gradients
        self.v = [np.zeros_like(p) for p in self.params]      # Adam: running mean of their squares
        self.t = 0

    def n_params(self):
        return sum(p.size for p in self.params)

    def forward(self, x, keep=False):
        acts = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.maximum(x, 0.0)
            acts.append(x)
        if keep:
            self._acts = acts
        return x

    def backward(self, grad):
        """grad = dLoss/dOutput for the last forward(keep=True). Stores the weight gradients."""
        acts = self._acts
        self.gW, self.gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            if i < len(self.W) - 1:
                grad = grad * (acts[i + 1] > 0)               # ReLU lets the gradient through only where active
            self.gW[i] = acts[i].T @ grad
            self.gb[i] = grad.sum(axis=0)
            grad = grad @ self.W[i].T

    def step(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, clip=10.0):
        self.t += 1
        grads = self.gW + self.gb
        norm = np.sqrt(sum(float((g * g).sum()) for g in grads))
        scale = min(1.0, clip / (norm + 1e-12))               # gradient clipping: no huge jumps
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            g = g * scale
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            mh = m / (1 - b1 ** self.t)
            vh = v / (1 - b2 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + eps)

    def copy_from(self, other):
        for p, q in zip(self.params, other.params):
            p[...] = q

    def save(self, path):
        np.savez(path, *self.params)

    def load(self, path):
        data = np.load(path)
        for i, p in enumerate(self.params):
            p[...] = data[f"arr_{i}"]
