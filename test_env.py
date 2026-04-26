"""Quick benchmark — tests 3 strategies against the environment."""
import sys
sys.path.insert(0, '.')
from daedalus.env import DaedalusEnvironment

strategies = {
    "Second-Price Default": {"auction_type": "second_price", "reserve_price": 0.1},
    "VCG + Anti-Collusion": {"auction_type": "vcg", "reserve_price": 0.1, "collusion_penalty": 2.0, "coalition_policy": "penalize_suspected", "reveal_winner_identity": False},
    "First-Price + High Reserve": {"auction_type": "first_price", "reserve_price": 0.35},
}

print("=" * 65)
print("DAEDALUS Baseline Benchmark — 10 episodes × 10 steps each")
print("=" * 65)

for name, action in strategies.items():
    rewards = []
    for ep in range(10):
        env = DaedalusEnvironment(episode_length=10)
        env.reset()
        ep_r = 0
        for s in range(10):
            _, r, done, info = env.step(action)
            ep_r += r
            if done:
                break
        rewards.append(ep_r / (s + 1))

    avg = sum(rewards) / len(rewards)
    mn = min(rewards)
    mx = max(rewards)
    print(f"\n  {name}:")
    print(f"    Avg Reward: {avg:.4f}  (min: {mn:.4f}, max: {mx:.4f})")

print("\n" + "=" * 65)
print("Benchmark COMPLETE ✓")
print("=" * 65)
