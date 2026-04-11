"""
record_video.py : Record a video of the trained DQN agent.

Usage:
    python record_video.py --checkpoint checkpoints/dqn_seed0_best.pt --seed 0
"""

import argparse
import os
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import highway_env  

from dqn_agent import DQNAgent
from train import make_env, flatten_obs, HPARAMS
from config import SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG


def record_agent_video(
    agent: DQNAgent,
    save_dir: str = "results/video",
    seed: int = 0,
    n_episodes: int = 3,
):

    os.makedirs(save_dir, exist_ok=True)

    render_env = gym.make("highway-v0", render_mode="rgb_array")
    render_env.unwrapped.configure({
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
            "normalize": True,
            "clip": True,
            "see_behind": True,
            "observe_intentions": False,
        },
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": [20, 25, 30],
        },
        "lanes_count": 4,
        "vehicles_count": 45,
        "controlled_vehicles": 1,
        "initial_lane_id": None,
        "duration": 30,
        "ego_spacing": 2,
        "vehicles_density": 1.0,
        "collision_reward": -1.5,
        "right_lane_reward": 0.0,
        "high_speed_reward": 0.7,
        "lane_change_reward": -0.02,
        "reward_speed_range": [22, 30],
        "normalize_reward": True,
        "offroad_terminal": True,
    })

    # Wrap with RecordVideo, records every episode
    video_env = RecordVideo(
        render_env,
        video_folder=save_dir,
        episode_trigger=lambda ep_id: True,  # record all episodes
        name_prefix="dqn_agent",
        disable_logger=True,
    )

    saved_eps = agent.eps
    agent.eps = 0.0  # greedy, no random actions during recording

    episode_rewards = []

    for ep in range(n_episodes):
        obs, _ = video_env.reset(seed=seed + ep)
        obs = flatten_obs(obs)
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = video_env.step(action)
            obs = flatten_obs(next_obs)
            total_reward += reward
            done = terminated or truncated

        crashed = info.get("crashed", False)
        episode_rewards.append(total_reward)
        print(f"  Episode {ep + 1}: reward = {total_reward:.2f}, crashed = {crashed}")

    video_env.close()
    agent.eps = saved_eps

    print(f"\nVideos saved to: {save_dir}/")
    print(f"Mean reward over {n_episodes} recorded episodes: {np.mean(episode_rewards):.2f}")
    return episode_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained checkpoint, e.g. checkpoints/dqn_seed0_best.pt"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_episodes", type=int, default=3,
                        help="Number of episodes to record")
    parser.add_argument("--save_dir", type=str, default="results/video")
    args = parser.parse_args()

    tmp_env = gym.make("highway-v0", render_mode=None)
    tmp_env.unwrapped.configure(SHARED_CORE_CONFIG)
    obs, _ = tmp_env.reset()
    obs_dim = flatten_obs(obs).shape[0]
    n_actions = tmp_env.action_space.n
    tmp_env.close()

    # Load the trained agent
    agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, **HPARAMS)
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    record_agent_video(
        agent,
        save_dir=args.save_dir,
        seed=args.seed,
        n_episodes=args.n_episodes,
    )
