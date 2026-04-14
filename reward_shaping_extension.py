import copy
import csv
import json
import os
import random

import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.wrappers import RecordVideo

from config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
from dqn_agent import DQNAgent


DEFAULT_HPARAMS = {
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

DEFAULT_OUTPUT_DIR = os.path.join("results", "reward_shaping_extension")
DEFAULT_CHECKPOINT_DIR = os.path.join("checkpoints", "reward_shaping_extension")
DEFAULT_TRAIN_EPISODES = 1_000
DEFAULT_EVAL_EPISODES = 50
DEFAULT_SEEDS = (0, 1, 2)
STUDY_QUESTION = (
    "Does simple reward shaping produce a safer DQN policy on highway-v0 when "
    "evaluation stays on the original benchmark?"
)

VARIANT_SPECS = {
    "vanilla": {
        "label": "Vanilla DQN",
        "description": "Original shared reward from the core task.",
        "color": "#1f77b4",
        "overrides": {},
    },
    "reward_shaped": {
        "label": "Reward-Shaped DQN",
        "description": (
            "Safer reward design: stronger collision penalty, slightly lower speed "
            "incentive, higher lane-change penalty, and a small right-lane bonus."
        ),
        "color": "#d62728",
        "overrides": {
            "collision_reward": -2.5,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.5,
            "lane_change_reward": -0.08,
        },
    },
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def flatten_obs(obs: np.ndarray) -> np.ndarray:
    return obs.flatten().astype(np.float32)


def smooth(values, window: int = 20):
    if len(values) < window:
        return np.asarray(values, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="valid")


def save_json(path: str, payload: dict | list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)


def load_json(path: str) -> dict | list:
    with open(path, "r") as file:
        return json.load(file)


def get_variant_names() -> tuple[str, ...]:
    return tuple(VARIANT_SPECS.keys())


def get_variant_config(variant: str) -> dict:
    if variant not in VARIANT_SPECS:
        raise ValueError(f"Unknown variant: {variant}")
    config = copy.deepcopy(SHARED_CORE_CONFIG)
    config.update(VARIANT_SPECS[variant]["overrides"])
    return config


def make_env(seed: int = 0, env_config: dict | None = None) -> gym.Env:
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=None)
    env.unwrapped.configure(copy.deepcopy(env_config or SHARED_CORE_CONFIG))
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def get_reward_terms(env_config: dict) -> dict:
    keys = [
        "collision_reward",
        "right_lane_reward",
        "high_speed_reward",
        "lane_change_reward",
        "normalize_reward",
    ]
    return {key: env_config.get(key) for key in keys}


def checkpoint_path(checkpoint_dir: str, variant: str, seed: int, kind: str) -> str:
    filename = f"{variant}_seed{seed}_{kind}.pt"
    return os.path.join(checkpoint_dir, filename)


def checkpoint_exists(
    variant: str,
    seed: int,
    kind: str = "best",
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
) -> bool:
    return os.path.exists(checkpoint_path(checkpoint_dir, variant, seed, kind))


def build_agent(seed: int, env_config: dict, hparams: dict | None = None) -> DQNAgent:
    env = make_env(seed=seed, env_config=env_config)
    obs, _ = env.reset(seed=seed)
    obs_dim = flatten_obs(obs).shape[0]
    n_actions = env.action_space.n
    env.close()
    return DQNAgent(obs_dim=obs_dim, n_actions=n_actions, **(hparams or DEFAULT_HPARAMS))


def train_variant(
    seed: int,
    variant: str,
    train_episodes: int = DEFAULT_TRAIN_EPISODES,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    hparams: dict | None = None,
    log_every: int = 50,
) -> tuple[dict, DQNAgent]:
    if variant not in VARIANT_SPECS:
        raise ValueError(f"Unknown variant: {variant}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    set_seed(seed)
    hparams = copy.deepcopy(hparams or DEFAULT_HPARAMS)
    env_config = get_variant_config(variant)
    env = make_env(seed=seed, env_config=env_config)
    obs, _ = env.reset(seed=seed)
    obs_dim = flatten_obs(obs).shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, **hparams)
    episode_rewards = []
    episode_crashed = []
    episode_lengths = []
    best_mean_reward = -float("inf")

    print(f"\n[{variant} | seed {seed}] training for {train_episodes} episodes")

    for episode in range(1, train_episodes + 1):
        obs, _ = env.reset()
        obs = flatten_obs(obs)
        done = False
        crashed = False
        ep_reward = 0.0
        ep_length = 0

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = flatten_obs(next_obs)
            done = terminated or truncated
            crashed = crashed or info.get("crashed", False)

            agent.store(obs, action, reward, next_obs, float(done))
            agent.update()

            obs = next_obs
            ep_reward += reward
            ep_length += 1

        episode_rewards.append(ep_reward)
        episode_crashed.append(bool(crashed))
        episode_lengths.append(ep_length)

        if episode % log_every == 0:
            recent_rewards = episode_rewards[-log_every:]
            recent_crashes = episode_crashed[-log_every:]
            mean_reward = float(np.mean(recent_rewards))
            crash_rate = 100.0 * float(np.mean(recent_crashes))
            print(
                f"  ep {episode:>4d} | mean_reward={mean_reward:.3f} | "
                f"crash%={crash_rate:.1f} | eps={agent.eps:.3f}"
            )
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                agent.save(checkpoint_path(checkpoint_dir, variant, seed, "best"))

    if best_mean_reward == -float("inf"):
        agent.save(checkpoint_path(checkpoint_dir, variant, seed, "best"))
    agent.save(checkpoint_path(checkpoint_dir, variant, seed, "final"))
    env.close()

    metrics = {
        "variant": variant,
        "seed": seed,
        "train_episodes": train_episodes,
        "reward_terms": get_reward_terms(env_config),
        "episode_rewards": episode_rewards,
        "episode_crashed": episode_crashed,
        "episode_lengths": episode_lengths,
        "losses": agent.losses[:1000],
        "hparams": {key: str(value) for key, value in hparams.items()},
    }
    return metrics, agent


def load_trained_agent(
    variant: str,
    seed: int,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    kind: str = "best",
    hparams: dict | None = None,
) -> tuple[DQNAgent, str]:
    env_config = get_variant_config(variant)
    path = checkpoint_path(checkpoint_dir, variant, seed, kind)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    agent = build_agent(seed=seed, env_config=env_config, hparams=hparams)
    agent.load(path)
    return agent, path


def evaluate_agent(
    agent: DQNAgent,
    eval_config: dict | None = None,
    n_episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
) -> dict:
    env = make_env(seed=seed, env_config=eval_config or SHARED_CORE_CONFIG)
    saved_eps = agent.eps
    agent.eps = 0.0

    rewards = []
    crashes = []
    episode_lengths = []
    episode_lane_changes = []
    episode_mean_speeds = []

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        obs = flatten_obs(obs)
        done = False
        crashed = False
        total_reward = 0.0
        step_count = 0
        lane_changes = 0
        speed_samples = []
        vehicle = env.unwrapped.controlled_vehicles[0]
        previous_lane = getattr(vehicle, "lane_index", None)

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            obs = flatten_obs(next_obs)
            done = terminated or truncated
            crashed = crashed or info.get("crashed", False)
            total_reward += reward
            step_count += 1

            vehicle = env.unwrapped.controlled_vehicles[0]
            speed_samples.append(float(vehicle.speed))
            current_lane = getattr(vehicle, "lane_index", None)
            if (
                previous_lane is not None
                and current_lane is not None
                and previous_lane[2] != current_lane[2]
            ):
                lane_changes += 1
            previous_lane = current_lane

        rewards.append(total_reward)
        crashes.append(bool(crashed))
        episode_lengths.append(step_count)
        episode_lane_changes.append(lane_changes)
        episode_mean_speeds.append(float(np.mean(speed_samples)) if speed_samples else 0.0)

    env.close()
    agent.eps = saved_eps

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "collision_rate": float(np.mean(crashes)),
        "crash_free_rate": float(1.0 - np.mean(crashes)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_lane_changes": float(np.mean(episode_lane_changes)),
        "mean_speed": float(np.mean(episode_mean_speeds)),
        "rewards": rewards,
        "crashes": crashes,
        "episode_lengths": episode_lengths,
        "episode_lane_changes": episode_lane_changes,
        "episode_mean_speeds": episode_mean_speeds,
    }


def summarize_eval_results(eval_by_seed: dict[int, dict]) -> dict:
    rewards = np.array([result["mean_reward"] for result in eval_by_seed.values()], dtype=np.float32)
    collisions = np.array(
        [result["collision_rate"] for result in eval_by_seed.values()],
        dtype=np.float32,
    )
    crash_free = np.array(
        [result["crash_free_rate"] for result in eval_by_seed.values()],
        dtype=np.float32,
    )
    speeds = np.array([result["mean_speed"] for result in eval_by_seed.values()], dtype=np.float32)
    lane_changes = np.array(
        [result["mean_lane_changes"] for result in eval_by_seed.values()],
        dtype=np.float32,
    )
    episode_lengths = np.array(
        [result["mean_episode_length"] for result in eval_by_seed.values()],
        dtype=np.float32,
    )
    return {
        "mean_reward": float(np.mean(rewards)),
        "reward_std_across_seeds": float(np.std(rewards)),
        "mean_collision_rate": float(np.mean(collisions)),
        "mean_crash_free_rate": float(np.mean(crash_free)),
        "mean_speed": float(np.mean(speeds)),
        "mean_lane_changes": float(np.mean(lane_changes)),
        "mean_episode_length": float(np.mean(episode_lengths)),
    }


def summary_rows(study: dict) -> list[dict]:
    rows = []
    for variant in get_variant_names():
        variant_block = study["variants"][variant]
        summary = variant_block["summary"]
        rows.append(
            {
                "variant": variant,
                "label": VARIANT_SPECS[variant]["label"],
                "mean_reward": round(summary["mean_reward"], 3),
                "collision_rate": round(summary["mean_collision_rate"], 3),
                "crash_free_rate": round(summary["mean_crash_free_rate"], 3),
                "mean_speed": round(summary["mean_speed"], 3),
                "mean_lane_changes": round(summary["mean_lane_changes"], 3),
                "mean_episode_length": round(summary["mean_episode_length"], 3),
            }
        )
    return rows


def print_study_summary(study: dict):
    print("\nReward-shaping extension summary")
    print(STUDY_QUESTION)
    print()
    print(
        f"{'Variant':<18} {'Reward':>8} {'Collision%':>12} "
        f"{'CrashFree%':>12} {'Speed':>8} {'LaneChanges':>12}"
    )
    print("-" * 76)
    for row in summary_rows(study):
        print(
            f"{row['label']:<18} {row['mean_reward']:>8.3f} "
            f"{row['collision_rate']*100:>11.1f}% "
            f"{row['crash_free_rate']*100:>11.1f}% "
            f"{row['mean_speed']:>8.3f} "
            f"{row['mean_lane_changes']:>12.3f}"
        )


def save_summary_table(study: dict, output_dir: str = DEFAULT_OUTPUT_DIR) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    rows = summary_rows(study)
    csv_path = os.path.join(output_dir, "summary_table.csv")
    md_path = os.path.join(output_dir, "summary_table.md")

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Label | Mean Reward | Collision Rate | Crash-Free Rate | Mean Speed | Mean Lane Changes | Mean Episode Length |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['label']} | "
            f"{row['mean_reward']} | "
            f"{row['collision_rate']} | "
            f"{row['crash_free_rate']} | "
            f"{row['mean_speed']} | "
            f"{row['mean_lane_changes']} | "
            f"{row['mean_episode_length']} |"
        )
    with open(md_path, "w") as file:
        file.write("\n".join(lines) + "\n")

    return csv_path, md_path


def average_training_curve(train_metrics: list[dict], window: int = 20) -> tuple[np.ndarray, np.ndarray] | None:
    curves = [smooth(metrics["episode_rewards"], window=window) for metrics in train_metrics]
    curves = [curve for curve in curves if curve.size > 0]
    if not curves:
        return None
    min_len = min(curve.size for curve in curves)
    stacked = np.vstack([curve[:min_len] for curve in curves])
    x = np.arange(min_len) + window
    return x, stacked.mean(axis=0)


def plot_training_comparison(
    study: dict,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    show: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for variant in get_variant_names():
        train_metrics = study["variants"][variant]["train_metrics"]
        averaged = average_training_curve(train_metrics, window=20)
        if averaged is None:
            continue
        x, y = averaged
        color = VARIANT_SPECS[variant]["color"]
        axes[0].plot(x, y, color=color, linewidth=2, label=VARIANT_SPECS[variant]["label"])

        crash_curves = [
            smooth([float(value) for value in metrics["episode_crashed"]], window=20)
            for metrics in train_metrics
        ]
        crash_curves = [curve for curve in crash_curves if curve.size > 0]
        if crash_curves:
            min_len = min(curve.size for curve in crash_curves)
            crash_mean = np.vstack([curve[:min_len] for curve in crash_curves]).mean(axis=0)
            crash_x = np.arange(min_len) + 20
            axes[1].plot(
                crash_x,
                crash_mean * 100.0,
                color=color,
                linewidth=2,
                label=VARIANT_SPECS[variant]["label"],
            )

    axes[0].set_title("Average training reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Smoothed reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Average crash rate during training")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Crash rate (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "training_comparison.png")
    plt.savefig(path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
    return path


def plot_summary_comparison(
    study: dict,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    show: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metric_specs = [
        ("mean_reward", "Mean reward", 1.0),
        ("mean_collision_rate", "Collision rate (%)", 100.0),
        ("mean_speed", "Mean speed", 1.0),
        ("mean_lane_changes", "Mean lane changes", 1.0),
    ]

    for ax, (metric_key, title, scale) in zip(axes.flatten(), metric_specs):
        labels = []
        values = []
        colors = []
        for variant in get_variant_names():
            labels.append(VARIANT_SPECS[variant]["label"])
            values.append(study["variants"][variant]["summary"][metric_key] * scale)
            colors.append(VARIANT_SPECS[variant]["color"])
        ax.bar(labels, values, color=colors, alpha=0.85)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=10)

    plt.tight_layout()
    path = os.path.join(output_dir, "summary_comparison.png")
    plt.savefig(path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
    return path


def save_study_artifacts(study: dict, output_dir: str = DEFAULT_OUTPUT_DIR) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    training_plot = plot_training_comparison(study, output_dir=output_dir, show=False)
    summary_plot = plot_summary_comparison(study, output_dir=output_dir, show=False)
    csv_path, md_path = save_summary_table(study, output_dir=output_dir)
    artifact_paths = {
        "study_json": os.path.join(output_dir, "study.json"),
        "training_plot": training_plot,
        "summary_plot": summary_plot,
        "summary_csv": csv_path,
        "summary_markdown": md_path,
    }
    return artifact_paths


def collect_failure_episodes(
    agent: DQNAgent,
    env_config: dict,
    n_episodes: int = 200,
    seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    env = make_env(seed=seed, env_config=env_config)
    saved_eps = agent.eps
    agent.eps = 0.0

    failures = []
    successes = []

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        obs = flatten_obs(obs)
        history = []
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            history.append(
                {
                    "obs": obs.copy().reshape(10, 5),
                    "action": action,
                    "reward": reward,
                }
            )
            obs = flatten_obs(next_obs)
            done = terminated or truncated

        record = {
            "ep": episode,
            "length": len(history),
            "total_reward": float(sum(step["reward"] for step in history)),
            "crashed": bool(info.get("crashed", False)),
            "last_action": history[-1]["action"],
            "last_obs": history[-1]["obs"],
            "pre_crash_actions": [step["action"] for step in history[-5:]],
        }
        if record["crashed"]:
            failures.append(record)
        else:
            successes.append(record)

    env.close()
    agent.eps = saved_eps
    return failures, successes


def run_failure_analysis(
    variant: str = "reward_shaped",
    seed: int = 0,
    n_episodes: int = 200,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    kind: str = "best",
    hparams: dict | None = None,
) -> dict:
    from analyze_failures import plot_failure_analysis, print_report

    agent, checkpoint = load_trained_agent(
        variant=variant,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        kind=kind,
        hparams=hparams,
    )
    env_config = get_variant_config(variant)
    failures, successes = collect_failure_episodes(
        agent,
        env_config=env_config,
        n_episodes=n_episodes,
        seed=seed,
    )
    save_dir = os.path.join(output_dir, f"{variant}_failure_analysis")
    os.makedirs(save_dir, exist_ok=True)
    print_report(failures, successes)
    plot_failure_analysis(failures, successes, save_dir=save_dir)
    return {
        "variant": variant,
        "seed": seed,
        "checkpoint": checkpoint,
        "save_dir": save_dir,
        "plot_path": os.path.join(save_dir, "failure_analysis.png"),
        "n_failures": len(failures),
        "n_successes": len(successes),
    }


def record_variant_video(
    variant: str = "reward_shaped",
    seed: int = 0,
    n_episodes: int = 3,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    kind: str = "best",
    hparams: dict | None = None,
) -> dict:
    agent, checkpoint = load_trained_agent(
        variant=variant,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        kind=kind,
        hparams=hparams,
    )
    env_config = get_variant_config(variant)
    video_dir = os.path.join(output_dir, "video", variant)
    os.makedirs(video_dir, exist_ok=True)

    render_env = gym.make(SHARED_CORE_ENV_ID, render_mode="rgb_array")
    render_env.unwrapped.configure(copy.deepcopy(env_config))
    video_env = RecordVideo(
        render_env,
        video_folder=video_dir,
        episode_trigger=lambda _: True,
        name_prefix=f"{variant}_seed{seed}",
        disable_logger=True,
    )

    saved_eps = agent.eps
    agent.eps = 0.0
    rewards = []

    for episode in range(n_episodes):
        obs, _ = video_env.reset(seed=seed + episode)
        obs = flatten_obs(obs)
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = video_env.step(action)
            obs = flatten_obs(next_obs)
            total_reward += reward
            done = terminated or truncated

        rewards.append(float(total_reward))

    video_env.close()
    agent.eps = saved_eps
    return {
        "variant": variant,
        "seed": seed,
        "checkpoint": checkpoint,
        "video_dir": video_dir,
        "mean_reward": float(np.mean(rewards)) if rewards else None,
        "rewards": rewards,
    }


def run_reward_shaping_study(
    seeds: list[int] | tuple[int, ...] = DEFAULT_SEEDS,
    train_episodes: int = DEFAULT_TRAIN_EPISODES,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    hparams: dict | None = None,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    hparams = copy.deepcopy(hparams or DEFAULT_HPARAMS)
    core_eval_config = copy.deepcopy(SHARED_CORE_CONFIG)

    study = {
        "question": STUDY_QUESTION,
        "seeds": list(seeds),
        "train_episodes": train_episodes,
        "eval_episodes": eval_episodes,
        "variants": {},
    }

    for variant in get_variant_names():
        variant_train_metrics = []
        eval_by_seed = {}

        print(f"\n=== {VARIANT_SPECS[variant]['label']} ===")
        print(f"Training reward terms: {get_reward_terms(get_variant_config(variant))}")

        for seed in seeds:
            train_metrics, _ = train_variant(
                seed=seed,
                variant=variant,
                train_episodes=train_episodes,
                checkpoint_dir=checkpoint_dir,
                hparams=hparams,
            )
            variant_train_metrics.append(train_metrics)

            best_agent = build_agent(seed=seed, env_config=get_variant_config(variant), hparams=hparams)
            best_agent.load(checkpoint_path(checkpoint_dir, variant, seed, "best"))
            eval_by_seed[seed] = evaluate_agent(
                best_agent,
                eval_config=core_eval_config,
                n_episodes=eval_episodes,
                seed=seed,
            )

        study["variants"][variant] = {
            "label": VARIANT_SPECS[variant]["label"],
            "description": VARIANT_SPECS[variant]["description"],
            "reward_terms": get_reward_terms(get_variant_config(variant)),
            "train_metrics": variant_train_metrics,
            "eval_by_seed": eval_by_seed,
            "summary": summarize_eval_results(eval_by_seed),
        }

    study["artifacts"] = save_study_artifacts(study, output_dir=output_dir)
    save_json(os.path.join(output_dir, "study.json"), study)
    return study


def load_study(output_dir: str = DEFAULT_OUTPUT_DIR) -> dict:
    return load_json(os.path.join(output_dir, "study.json"))
