"""
analyze_failures.py : Characterize failure episodes 

"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import highway_env  

from dqn_agent import DQNAgent
from train import make_env, flatten_obs, HPARAMS
from config import SHARED_CORE_CONFIG

ACTION_NAMES  = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
ACTION_COLORS = ["#3498db", "#95a5a6", "#e74c3c", "#2ecc71", "#f39c12"]


def collect_episodes(agent, n=200, seed=0):
    env = make_env(seed)
    agent.eps = 0.0  # greedy

    failures, successes = [], []
    for ep in range(n):
        obs, _ = env.reset(seed=seed + ep)
        obs = flatten_obs(obs)
        history = []
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            history.append({"obs": obs.copy().reshape(10, 5), "action": action, "reward": reward})
            obs = flatten_obs(next_obs)
            done = terminated or truncated

        crashed = info.get("crashed", False)
        record = {
            "ep":           ep,
            "length":       len(history),
            "total_reward": sum(s["reward"] for s in history),
            "crashed":      crashed,
            "last_action":  history[-1]["action"],
            "last_obs":     history[-1]["obs"],
            "pre_crash_actions": [s["action"] for s in history[-5:]],
        }
        (failures if crashed else successes).append(record)

    env.close()
    return failures, successes


def plot_failure_analysis(failures, successes, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    n_total = len(failures) + len(successes)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Failure analysis — {n_total} greedy episodes  "
        f"({len(failures)} crashes = {100*len(failures)/n_total:.0f}%)",
        fontsize=12, fontweight="bold"
    )

    # Panel 1 : crash timing
    ax = axes[0, 0]
    lf = [f["length"] for f in failures]
    ls = [s["length"] for s in successes]
    all_l = lf + ls
    bins = range(0, max(all_l) + 5, 3) if all_l else range(0, 10)
    if lf:
        ax.hist(lf, bins=bins, color="#e74c3c", alpha=0.7, label=f"Crashes ({len(failures)})")
        ax.axvline(np.mean(lf), color="#c0392b", lw=1.5, linestyle="--",
                   label=f"Mean: {np.mean(lf):.1f} steps")
    if ls:
        ax.hist(ls, bins=bins, color="#2ecc71", alpha=0.7, label=f"Successes ({len(successes)})")
    ax.set_title("When do crashes happen?")
    ax.set_xlabel("Episode length (steps)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2 : last action before crash
    ax = axes[0, 1]
    ac = {i: 0 for i in range(5)}
    for f in failures:
        ac[f["last_action"]] += 1
    bars = ax.bar([ACTION_NAMES[i] for i in range(5)], [ac[i] for i in range(5)],
                  color=ACTION_COLORS, alpha=0.85, edgecolor="white")
    ax.set_title("Last action before crash")
    ax.set_ylabel("Number of crashes")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, [ac[i] for i in range(5)]):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, str(v), ha="center", fontsize=9)

    # Panel 3 : reward distribution
    ax = axes[1, 0]
    rf = [f["total_reward"] for f in failures]
    rs = [s["total_reward"] for s in successes]
    all_r = rf + rs
    if all_r:
        b2 = np.linspace(min(all_r), max(all_r), 25)
        if rf:
            ax.hist(rf, bins=b2, color="#e74c3c", alpha=0.7, label="Crashes")
        if rs:
            ax.hist(rs, bins=b2, color="#2ecc71", alpha=0.7, label="Successes")
    ax.set_title("Reward distribution")
    ax.set_xlabel("Total episode reward")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4 : pre-crash action sequence
    ax = axes[1, 1]
    step_counts = np.zeros((5, 5))
    for f in failures:
        for t, a in enumerate(f["pre_crash_actions"]):
            step_counts[t, a] += 1
    row_sums = step_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    step_freq = step_counts / row_sums * 100
    x = np.arange(5)
    w = 0.15
    for a in range(5):
        ax.bar(x + a * w, step_freq[:, a], w, label=ACTION_NAMES[a],
               color=ACTION_COLORS[a], alpha=0.85)
    ax.set_xticks(x + w * 2)
    ax.set_xticklabels([f"t-{4-i}" for i in range(5)])
    ax.set_title("Actions in last 5 steps before crash (%)")
    ax.set_ylabel("% of crash episodes")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(save_dir, "failure_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {path}")
    plt.show()


def print_report(failures, successes):
    n = len(failures) + len(successes)
    print("\n" + "=" * 65)
    print(" FAILURE ANALYSIS REPORT")
    print("=" * 65)
    print(f"  Crashes  : {len(failures)} / {n}  ({100*len(failures)/n:.1f}%)")
    print(f"  Successes: {len(successes)} / {n}  ({100*len(successes)/n:.1f}%)")

    if not failures:
        print("  No failures, very robust agent!")
        return

    lf = [f["length"] for f in failures]
    early = [f for f in failures if f["length"] <= 10]
    print(f"\n  Crash timing: mean={np.mean(lf):.1f} steps, "
          f"min={np.min(lf)}, max={np.max(lf)}")
    print(f"  Crashes within first 10 steps: {len(early)} ({100*len(early)/len(failures):.0f}%)")

    ac = {i: 0 for i in range(5)}
    for f in failures:
        ac[f["last_action"]] += 1
    print("\n  Last action before crash:")
    for a, cnt in sorted(ac.items(), key=lambda x: -x[1]):
        print(f"    {ACTION_NAMES[a]:<12} {cnt:>4}  ({100*cnt/len(failures):.0f}%)")

    rf = np.mean([f["total_reward"] for f in failures])
    rs = np.mean([s["total_reward"] for s in successes]) if successes else float("nan")
    print(f"\n  Mean reward — crashes  : {rf:.2f}")
    print(f"  Mean reward — successes: {rs:.2f}")
    print(f"  Reward gap: {rs - rf:.2f} (how much surviving is worth)")

    most_common = max(ac, key=ac.get)
    pct = 100 * ac[most_common] / len(failures)
    print("\n" + "-" * 65)
    print(" INTERPRETATION")
    print("-" * 65)
    print(f"\n  Most common crash action: {ACTION_NAMES[most_common]} ({pct:.0f}% of crashes)")
    if most_common in [0, 2]:
        print("  → Agent makes lane changes into occupied gaps.")
        print("    It hasn't learned to verify the target lane is clear before committing.")
    elif most_common == 3:
        print("  → Agent tailgates: accelerates into vehicles ahead.")
        print("    High-speed reward encourages FASTER but the agent hasn't fully")
        print("    learned to maintain safe following distance.")
    elif most_common == 1:
        print("  → Agent crashes while idle — it is being hit by other vehicles,")
        print("    not from its own aggression. May need evasive maneuvers.")

    if len(early) / len(failures) > 0.25:
        print(f"\n  {100*len(early)/len(failures):.0f}% of crashes occur in ≤10 steps.")
        print("  → Some failures may be caused by unavoidable spawn configurations,")
        print("    not poor policy. This is a known limitation of the environment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    tmp = gym.make("highway-v0", render_mode=None)
    tmp.unwrapped.configure(SHARED_CORE_CONFIG)
    obs, _ = tmp.reset()
    obs_dim = flatten_obs(obs).shape[0]
    n_actions = tmp.action_space.n
    tmp.close()

    agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, **HPARAMS)
    agent.load(args.checkpoint)
    print(f"Loaded: {args.checkpoint}")
    print(f"Collecting {args.n} greedy episodes...")

    failures, successes = collect_episodes(agent, n=args.n, seed=args.seed)
    print_report(failures, successes)
    plot_failure_analysis(failures, successes)