# DQN — Highway-v0 Core Task

## File structure

```
dqn_highway/
├── config.py            # shared core config (do not modify)
├── network.py           # Q-network (MLP)
├── replay_buffer.py     # experience replay
├── dqn_agent.py         # DQN agent (target net, ε-greedy, Huber loss)
├── train.py             # training loop, evaluation, plotting
├── analyze_failures.py  # failure mode analysis
└── record_video.py      # record rollout videos of a trained agent
```

## Setup

```bash
pip install gymnasium highway-env torch matplotlib pillow
```

## Usage

### Train a single seed (quick start)
```bash
python train.py --seed 0
```

### Train all 5 seeds + generate full eval table and plots
```bash
python train.py --all_seeds
```

### Evaluate a saved checkpoint
```bash
python train.py --eval_only --checkpoint checkpoints/dqn_seed0_best.pt --seed 0
```

### Failure mode analysis
```bash
python analyze_failures.py --checkpoint checkpoints/dqn_seed0_best.pt --seed 0
```

### Record a video rollout
```bash
python record_video.py --checkpoint checkpoints/dqn_seed0_best.pt --seed 0 --n_episodes 3
```

## What gets saved

| Path | Contents |
|------|----------|
| `checkpoints/dqn_seedN_best.pt` | Best checkpoint (by mean reward over last 50 episodes) |
| `checkpoints/dqn_seedN_final.pt` | Final checkpoint after all training episodes |
| `results/training_curves.png` | 3-panel plot: reward, collision rate, TD loss |
| `results/metrics.json` | Episode rewards, lengths, crash flags, milestone evals, eval results |
| `results/failure_analysis.png` | 4-panel failure analysis plot |
| `results/video/` | Recorded rollout videos |

## Design choices 

| Choice | Value | Rationale |
|--------|-------|-----------|
| Network | MLP 256-256 | Kinematics obs is already low-dim; conv not needed |
| Loss | Huber (SmoothL1) | More robust to outlier rewards than MSE |
| Optimizer | Adam, lr=1e-4 | Conservative rate for stable convergence |
| Target update | Hard copy every 1 000 steps | Simple and effective |
| Replay buffer | 50 000 transitions | Balances memory and diversity |
| Gradient clipping | norm ≤ 10 | Prevents instability early in training |
| ε-decay | 1.0 → 0.05 over 12 000 steps | Full exploration for ~900 episodes, then exploit |
| Training length | 1 500 episodes per seed | Enough for policy to stabilize post-exploration |

