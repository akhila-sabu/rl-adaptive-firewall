"""Tabular Q-learning agent (epsilon-greedy) - simple enough to explain in a viva."""
import numpy as np


class QAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9,
                 eps=1.0, eps_min=0.02, eps_decay=0.998, seed=None):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha, self.gamma = alpha, gamma
        self.eps, self.eps_min, self.eps_decay = eps, eps_min, eps_decay
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, s, greedy=False):
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_actions))
        q = self.Q[s]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))  # random tie-break

    def learn(self, s, a, r, s2, done):
        target = r + (0 if done else self.gamma * self.Q[s2].max())
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])  # Bellman update

    def end_episode(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def save(self, path):
        np.save(path, self.Q)

    def load(self, path):
        self.Q = np.load(path)
