"""Train the Q-learning firewall agent and plot the learning curve."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from firewall_env import FirewallEnv
from agent import QAgent


def run_episode(env, agent, learn=True):
    s, total, done = env.reset(), 0.0, False
    while not done:
        a = agent.act(s, greedy=not learn)
        s2, r, done, _ = env.step(a)
        if learn:
            agent.learn(s, a, r, s2, done)
        s, total = s2, total + r
    return total


def train(env, agent, episodes):
    rewards = []
    for _ in range(episodes):
        rewards.append(run_episode(env, agent, learn=True))
        agent.end_episode()
    return rewards


if __name__ == "__main__":
    env = FirewallEnv(seed=1)
    agent = QAgent(env.n_states, env.n_actions, seed=1)
    rewards = train(env, agent, episodes=2000)
    agent.save("qtable.npy")
    ma = np.convolve(rewards, np.ones(50) / 50, mode="valid")
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, alpha=0.3, label="episode reward")
    plt.plot(range(49, len(rewards)), ma, label="50-episode average")
    plt.xlabel("Episode"); plt.ylabel("Total reward"); plt.title("Learning curve")
    plt.legend(); plt.tight_layout(); plt.savefig("learning_curve.png", dpi=150)
    print(f"Saved qtable.npy and learning_curve.png | last-100 avg reward: {np.mean(rewards[-100:]):.1f}")
