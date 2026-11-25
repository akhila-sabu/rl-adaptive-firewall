"""Static rule-based firewall (what the RL agent is compared against)."""
from firewall_env import FirewallEnv

ALLOW, RATE, BLOCK = 0, 1, 2


def static_rules(env):
    pps, ports, syn, fails, kb = env.feats
    if pps > 1000 or ports > 20 or fails > 10 or kb > 3000:
        return BLOCK
    return ALLOW
