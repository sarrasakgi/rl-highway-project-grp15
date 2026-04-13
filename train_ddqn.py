"""
train_ddqn.py: Training loop, evaluation, and plotting for the Double DQN extension.

Identical to train.py except it uses DDQNAgent and writes to metrics_ddqn.json.
Hyperparameters and evaluation protocol are unchanged for a fair comparison.

Usage:
    python train_ddqn.py --seed 0
    python train_ddqn.py --seed 0 --eval_only --checkpoint checkpoints/ddqn_seed0_best.pt
    python train_ddqn.py --all_seeds
"""

import argparse
import os
import json
import random

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import highway_env

from ddqn_agent import DDQNAgent
from config import SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG


# Hyperparameters (identical to train.py for a controlled comparison)

HPARAMS = {
    "hidden_sizes": (256, 256),
    "buffer_capacity": 50_000,
    "batch_size": 64,
    "lr": 1e-4,
    "gamma": 0.99,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_steps": 12_000,
    "target_update_freq": 1_000,
    "min_buffer_size": 1_000,
}

TRAIN_EPISODES      = 1_500
EVAL_EPISODES       = 50
EVAL_MILESTONE_FREQ = 500
EVAL_SEEDS          = [0, 1, 2, 3, 4]
CHECKPOINT_DIR      = "checkpoints"
RESULTS_DIR         = "results"


# Helpers

def make_env(seed: int = 0) -> gym.Env:
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=None)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def flatten_obs(obs: np.ndarray) -> np.ndarray:
    return obs.flatten().astype(np.float32)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def smooth(values, window=20):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# Training

def train(seed: int, checkpoint_dir: str = CHECKPOINT_DIR) -> dict:
    os.makedirs(checkpoint_dir, exist_ok=True)
    set_seed(seed)

    env = make_env(seed)
    obs, _ = env.reset(seed=seed)
    obs_dim = flatten_obs(obs).shape[0]
    n_actions = env.action_space.n

    agent = DDQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        **HPARAMS,
    )

    episode_rewards = []
    episode_lengths = []
    episode_crashed = []
    milestone_evals = {}
    best_mean_reward = -float("inf")

    print(f"\n[DDQN Seed {seed}] Starting training — obs_dim={obs_dim}, n_actions={n_actions}")

    for episode in range(1, TRAIN_EPISODES + 1):
        obs, _ = env.reset()
        obs = flatten_obs(obs)
        ep_reward = 0.0
        ep_len = 0
        done = False
        crashed = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = flatten_obs(next_obs)
            done = terminated or truncated
            if info.get("crashed", False):
                crashed = True

            agent.store(obs, action, reward, next_obs, float(done))
            agent.update()

            obs = next_obs
            ep_reward += reward
            ep_len += 1

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)
        episode_crashed.append(crashed)

        # Log every 50 episodes
        if episode % 50 == 0:
            recent = episode_rewards[-50:]
            recent_crash = episode_crashed[-50:]
            mean_r = np.mean(recent)
            crash_pct = 100.0 * sum(recent_crash) / len(recent_crash)
            print(
                f"  Ep {episode:>4d} | "
                f"mean_reward(last50)={mean_r:.3f} | "
                f"crash%={crash_pct:.1f} | "
                f"eps={agent.eps:.3f} | "
                f"buffer={len(agent.buffer)}"
            )
            # Save best checkpoint
            if mean_r > best_mean_reward:
                best_mean_reward = mean_r
                agent.save(os.path.join(checkpoint_dir, f"ddqn_seed{seed}_best.pt"))

        # Milestone evaluation (greedy) every EVAL_MILESTONE_FREQ episodes
        if episode % EVAL_MILESTONE_FREQ == 0:
            print(f"  [Milestone eval at ep {episode}]")
            milestone_res = evaluate(agent, n_episodes=20, seed=seed)
            milestone_evals[episode] = milestone_res
            print(
                f"    milestone mean={milestone_res['mean']:.3f}  "
                f"crash%={milestone_res['collision_rate']*100:.1f}"
            )

    # Save final checkpoint
    agent.save(os.path.join(checkpoint_dir, f"ddqn_seed{seed}_final.pt"))
    env.close()

    metrics = {
        "seed": seed,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_crashed": episode_crashed,
        "milestone_evals": milestone_evals,
        "losses": agent.losses,
        "hparams": {k: str(v) for k, v in HPARAMS.items()},
    }
    return metrics, agent


# Evaluation

def evaluate(agent: DDQNAgent, n_episodes: int = EVAL_EPISODES, seed: int = 42) -> dict:

    env = make_env(seed)
    saved_eps = agent.eps
    agent.eps = 0.0  # greedy

    rewards = []
    collision_count = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        obs = flatten_obs(obs)
        ep_reward = 0.0
        done = False
        crashed = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            obs = flatten_obs(next_obs)
            ep_reward += reward
            done = terminated or truncated
            if info.get("crashed", False):
                crashed = True

        rewards.append(ep_reward)
        if crashed:
            collision_count += 1

    env.close()
    agent.eps = saved_eps  # restore

    return {
        "mean": float(np.mean(rewards)),
        "std":  float(np.std(rewards)),
        "min":  float(np.min(rewards)),
        "max":  float(np.max(rewards)),
        "collision_rate": collision_count / n_episodes,
        "n_episodes": n_episodes,
        "rewards": rewards,
    }


# Plotting

def plot_training_curves(all_metrics: list, results_dir: str = RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    for m in all_metrics:
        seed = m["seed"]
        rewards = m["episode_rewards"]
        smoothed = smooth(rewards, window=20)
        x = np.arange(len(smoothed)) + 20
        axes[0].plot(x, smoothed, alpha=0.8, label=f"seed {seed}")

    axes[0].set_title("DDQN Training reward (smoothed, window=20)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Episode reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for m in all_metrics:
        seed = m["seed"]
        crashed = m.get("episode_crashed", [])
        if crashed:
            cr_float = [float(c) for c in crashed]
            smoothed_cr = smooth(cr_float, window=50)
            x = np.arange(len(smoothed_cr)) + 50
            axes[1].plot(x, [v * 100 for v in smoothed_cr], alpha=0.8, label=f"seed {seed}")

    axes[1].set_title("DDQN Collision rate (rolling 50-ep window, %)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Collision rate (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    losses = all_metrics[0]["losses"]
    if losses:
        smoothed_loss = smooth(losses, window=200)
        axes[2].plot(smoothed_loss, color="crimson", alpha=0.8)
        axes[2].set_title(f"DDQN TD loss (seed {all_metrics[0]['seed']}, smoothed)")
        axes[2].set_xlabel("Update step")
        axes[2].set_ylabel("Huber loss")
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, "training_curves_ddqn.png")
    plt.savefig(path, dpi=150)
    print(f"Saved training curves → {path}")
    plt.close()


def print_milestone_table(all_metrics: list):
    print("\n## DDQN Milestone evaluations (greedy, 20 episodes)\n")
    milestones = sorted({ep for m in all_metrics for ep in m.get("milestone_evals", {}).keys()})
    if not milestones:
        return
    header = f"{'Seed':<6}" + "".join(f"  {'Ep'+str(ep):>10}" for ep in milestones)
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        row = f"{m['seed']:<6}"
        for ep in milestones:
            res = m.get("milestone_evals", {}).get(ep)
            if res:
                row += f"  {res['mean']:>7.2f}±{res['std']:.2f}"
            else:
                row += f"  {'N/A':>10}"
        print(row)


def print_eval_table(eval_results: dict):
    print("\n## DDQN Evaluation results (50 runs per seed)\n")
    print(f"{'Seed':<8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Collision%':>12}")
    print("-" * 56)
    for seed, res in eval_results.items():
        print(
            f"{seed:<8} {res['mean']:>8.3f} {res['std']:>8.3f} "
            f"{res['min']:>8.3f} {res['max']:>8.3f} "
            f"{res['collision_rate']*100:>11.1f}%"
        )


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Training seed")
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training and load a checkpoint for evaluation",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for eval_only mode",
    )
    parser.add_argument(
        "--all_seeds",
        action="store_true",
        help="Train over all EVAL_SEEDS and produce combined plots",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.eval_only:
        assert args.checkpoint, "Provide --checkpoint for eval_only mode"
        env = make_env(args.seed)
        obs, _ = env.reset()
        obs_dim = flatten_obs(obs).shape[0]
        n_actions = env.action_space.n
        env.close()

        agent = DDQNAgent(obs_dim=obs_dim, n_actions=n_actions, **HPARAMS)
        agent.load(args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")

        res = evaluate(agent, n_episodes=EVAL_EPISODES, seed=args.seed)
        print_eval_table({args.seed: res})

    elif args.all_seeds:
        all_metrics = []
        eval_results = {}

        for seed in EVAL_SEEDS:
            metrics, agent = train(seed)
            all_metrics.append(metrics)

            env = make_env(seed)
            obs, _ = env.reset()
            obs_dim = flatten_obs(obs).shape[0]
            n_actions = env.action_space.n
            env.close()

            best_agent = DDQNAgent(obs_dim=obs_dim, n_actions=n_actions, **HPARAMS)
            ckpt = os.path.join(CHECKPOINT_DIR, f"ddqn_seed{seed}_best.pt")
            best_agent.load(ckpt)

            eval_results[seed] = evaluate(best_agent, n_episodes=EVAL_EPISODES, seed=seed)
            print_eval_table({seed: eval_results[seed]})

        with open(os.path.join(RESULTS_DIR, "metrics_ddqn.json"), "w") as f:
            json.dump(
                [
                    {
                        **m,
                        "losses": m["losses"][:1000],
                        "eval": eval_results.get(m["seed"]),
                    }
                    for m in all_metrics
                ],
                f,
                indent=2,
            )

        plot_training_curves(all_metrics)
        print_milestone_table(all_metrics)
        print_eval_table(eval_results)

    else:
        metrics, agent = train(args.seed)

        res = evaluate(agent, n_episodes=EVAL_EPISODES, seed=args.seed)
        print_eval_table({args.seed: res})

        with open(os.path.join(RESULTS_DIR, f"metrics_ddqn_seed{args.seed}.json"), "w") as f:
            json.dump(
                {**metrics, "eval": res, "losses": metrics["losses"][:1000]}, f, indent=2
            )

        print_milestone_table([metrics])
        plot_training_curves([metrics])


if __name__ == "__main__":
    main()
