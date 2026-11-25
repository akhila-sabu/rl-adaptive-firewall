"""Simulated network-traffic environment for an RL-based adaptive firewall.

Each step the firewall sees ONE traffic window (a flow summary) and must pick an
action. Attacks come in bursts (campaigns), so history matters -> a real
sequential RL problem, not just classification.
"""
import numpy as np

TRAFFIC = ["normal", "port_scan", "syn_flood", "brute_force", "exfiltration"]
ACTIONS = ["allow", "rate_limit", "block"]

# (mean, std) of: packets/sec, unique dest ports, SYN ratio, failed logins, KB out
PROFILES = {
    "normal":       [(40, 25),    (2, 1.5),  (0.15, 0.08), (0.2, 0.4), (200, 150)],
    "port_scan":    [(120, 60),   (60, 25),  (0.90, 0.07), (0.0, 0.1), (20, 15)],
    "syn_flood":    [(2500, 900), (3, 2),    (0.95, 0.04), (0.0, 0.1), (50, 30)],
    "brute_force":  [(60, 30),    (1.5, 1),  (0.30, 0.10), (25, 10),   (30, 20)],
    "exfiltration": [(150, 70),   (3, 2),    (0.10, 0.06), (0.3, 0.5), (5000, 2000)],
}

# Reward table [traffic][action]: allow, rate_limit, block
#  - blocking good traffic is costly (false positive / outage)
#  - allowing an attack is very costly (breach)
#  - rate_limit is the "hedge": small cost on normal, partial credit on attacks
REWARDS = {
    "normal":       [ 1.0, -1.0, -5.0],
    "port_scan":    [-8.0,  2.0,  5.0],
    "syn_flood":    [-10.0, 4.0,  5.0],
    "brute_force":  [-8.0,  2.0,  5.0],
    "exfiltration": [-10.0, 2.0,  5.0],
}

EDGES = [  # bin edges per feature -> 4 bins each
    [30, 100, 500],      # pps
    [4, 15, 30],         # unique ports
    [0.25, 0.5, 0.8],    # syn ratio
    [1, 5, 15],          # failed logins
    [500, 2000, 4000],   # KB out
]
N_THREAT = 3  # recent-attack level: 0, 1-2, 3+ attacks in last 5 events
N_STATES = (4 ** 5) * N_THREAT


class FirewallEnv:
    def __init__(self, episode_len=200, attack_start=0.08, attack_persist=0.85,
                 stealth=0.0, seed=None):
        self.episode_len = episode_len
        self.p_start = attack_start      # chance a new attack campaign begins
        self.p_persist = attack_persist  # chance campaign continues next step
        self.stealth = stealth           # 0 = loud attacks, ->1 = look like normal
        self.rng = np.random.default_rng(seed)
        self.n_actions = len(ACTIONS)
        self.n_states = N_STATES

    # ---- traffic generation -------------------------------------------
    def _sample(self, kind):
        prof = PROFILES[kind]
        norm = PROFILES["normal"]
        feats = []
        for (m, s), (mn, sn) in zip(prof, norm):
            if kind != "normal":                      # attacker adapts: blend toward normal
                m = m * (1 - self.stealth) + mn * self.stealth
                s = s * (1 - self.stealth) + sn * self.stealth
            feats.append(max(0.0, self.rng.normal(m, s)))
        return feats

    def _next_kind(self):
        if self.kind != "normal" and self.rng.random() < self.p_persist:
            return self.kind
        if self.rng.random() < self.p_start:
            return self.rng.choice(TRAFFIC[1:])
        return "normal"

    def _encode(self, feats):
        idx = 0
        for f, edges in zip(feats, EDGES):
            idx = idx * 4 + int(np.searchsorted(edges, f))
        threat = sum(1 for k in self.history[-5:] if k != "normal")
        t = 0 if threat == 0 else (1 if threat < 3 else 2)
        return idx * N_THREAT + t

    # ---- gym-style API --------------------------------------------------
    def reset(self):
        self.t, self.kind, self.history = 0, "normal", []
        self.feats = self._sample(self.kind)
        return self._encode(self.feats)

    def step(self, action):
        reward = REWARDS[self.kind][action]
        info = {"kind": self.kind, "action": action}
        self.history.append(self.kind)
        self.t += 1
        self.kind = self._next_kind()
        self.feats = self._sample(self.kind)
        return self._encode(self.feats), reward, self.t >= self.episode_len, info
