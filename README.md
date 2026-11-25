# Reinforcement Learning Adaptive Firewall

Skillpod internship project (30 days). A firewall agent that **learns** whether to
*allow*, *rate-limit* or *block* traffic from reward feedback, and **adapts** when
attackers change behaviour — something fixed rule-sets cannot do.

## How it works
| RL concept | In this project |
|---|---|
| Environment | `firewall_env.py` – simulated traffic: normal, port scan, SYN flood, brute force, data exfiltration. Attacks arrive in bursts (campaigns). |
| State | 5 discretised features (packets/s, unique ports, SYN ratio, failed logins, KB out) + a "recent threat level" from the last 5 events (3,072 states) |
| Actions | allow / rate-limit / block |
| Reward | +1 allow good traffic, -5 block good traffic (false positive), -8 to -10 allow an attack, +5 block an attack, +2 to +4 rate-limit an attack |
| Algorithm | Tabular Q-learning, epsilon-greedy, Bellman update (`agent.py`) |
| Baseline | Static threshold rules (`baseline.py`) |

## Run
```
pip install -r requirements.txt
python train.py       # trains, saves qtable.npy + learning_curve.png
python evaluate.py    # RL vs static rules, then stealthy-attacker adaptation test
```

## Results (from my run)
| Scenario | Static rules | RL agent |
|---|---|---|
| Loud attacks | 91.9% attacks handled, 404 reward | 99.1% handled, 467 reward |
| Stealthy attacks (before RL adapts) | 47.5%, -49 reward | 76.5%, 204 reward |
| Stealthy attacks (after online learning) | 47.5% (cannot adapt) | 96.5%, 415 reward |

Key point for the viva: the static firewall collapses when attackers change tactics;
the RL agent recovers by continuing to learn.

## 30-day plan
- **Days 1-5** Study: firewalls/IDS, attack types, RL basics (MDP, Q-learning). Literature on RL in network security.
- **Days 6-10** Build the simulated environment; later swap in a real dataset (NSL-KDD / CIC-IDS2017) for realism.
- **Days 11-16** Implement Q-learning agent, train, plot learning curve.
- **Days 17-21** Baseline comparison, metrics (detection rate, false positives), stealth/adaptation experiment.
- **Days 22-25** Extensions: DQN (PyTorch), tune rewards, optional live demo that applies actions as iptables/nftables rules in a VM.
- **Days 26-28** Final report (abstract, intro, literature, design, implementation, results, limitations, future work).
- **Days 29-30** Presentation + viva practice.

## Likely viva questions
1. Why RL instead of a supervised classifier? (No labelled data needed per environment, learns from feedback, adapts to drift, optimises long-term cost of mistakes.)
2. What are state, action, reward? Why is blocking normal traffic penalised?
3. Explain the Q-update and the roles of alpha, gamma, epsilon.
4. Why a "threat level" in the state? (Attacks are bursty; history makes it sequential.)
5. Limitations? (Simulated data, tabular Q-learning doesn't scale, reward design is manual, attacker could poison feedback.)
6. Future work? (DQN/PPO, real datasets, live integration with iptables, adversarial attackers.)

## Report outline
Abstract · Introduction · Literature review · System design (diagram) · Implementation · Experiments & results · Limitations · Conclusion & future work · References
