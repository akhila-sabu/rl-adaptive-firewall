"""Compare RL agent vs static rules, then test adaptation to stealthier attackers."""
import numpy as np
from firewall_env import FirewallEnv
from agent import QAgent
from baseline import static_rules
from train import train


def evaluate(policy, stealth, episodes=200, seed=99):
    env = FirewallEnv(stealth=stealth, seed=seed)
    tp = fn = fp = tn = 0
    total = 0.0
    for _ in range(episodes):
        s, done = env.reset(), False
        while not done:
            a = policy(env, s)
            kind = env.kind
            s, r, done, _ = env.step(a)
            total += r
            if kind == "normal":
                fp += a == 2; tn += a != 2
            else:
                tp += a != 0; fn += a == 0   # rate-limit or block counts as "handled"
    n = episodes * env.episode_len
    return dict(reward=total / episodes,
                detection=tp / max(1, tp + fn),
                false_pos=fp / max(1, fp + tn))


def show(name, m):
    print(f"{name:<28} reward/ep {m['reward']:8.1f} | attacks handled {m['detection']*100:5.1f}% "
          f"| good traffic blocked {m['false_pos']*100:4.1f}%")


if __name__ == "__main__":
    agent = QAgent(1024 * 3, 3, seed=1)
    agent.load("qtable.npy"); agent.eps = 0.0
    rl = lambda env, s: agent.act(s, greedy=True)
    rules = lambda env, s: static_rules(env)

    print("=== Scenario 1: loud attacks (training conditions) ===")
    show("Static rules", evaluate(rules, 0.0))
    show("RL agent", evaluate(rl, 0.0))

    print("\n=== Scenario 2: attacker turns stealthy (stealth=0.6) ===")
    show("Static rules", evaluate(rules, 0.6))
    show("RL agent (before adapting)", evaluate(rl, 0.6))

    # Adaptation: keep learning online in the new environment
    agent.eps = 0.3
    agent.eps_min, agent.eps_decay = 0.02, 0.995
    train(FirewallEnv(stealth=0.6, seed=7), agent, episodes=1500)
    agent.eps = 0.0
    show("RL agent (after adapting)", evaluate(rl, 0.6))
