################################################################################
# task2_walker2d_cnn_transformer.py
#
# Task 2 Extended: CNN and Transformer state representations for Walker2d-v5
#
# This script trains SAC and TD3 (the two best algorithms from MLP experiments)
# with two alternative neural network architectures:
#
# 1. CNN (1D Convolution):
#    - Slides filters across the observation vector
#    - Detects LOCAL patterns between neighbouring joint values
#    - e.g. "knee position + knee velocity" patterns
#
# 2. Transformer (Self-Attention):
#    - Each observation value attends to all others
#    - Detects GLOBAL relationships between ANY joint values
#    - e.g. "torso angle relates to ankle position" for balance
#
# Comparison with MLP (baseline from task2_walker2d_drl.py):
#    - MLP treats all 17 values as a flat vector, no structure
#    - CNN imposes local structure (nearby values interact)
#    - Transformer allows global structure (any values interact)
#
#
# Uses different seeds from the MLP experiments to avoid bias.
# MLP seeds: [42, 123, 456]
# CNN/Transformer seeds: [55, 234, 678]
#
# Adapted from aml_continuous_drl_agents.py (CMP9137M workshop)
# Uses StableBaselines3 custom feature extractors:
#   https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
#
# Usage:
#   python task2_walker2d_cnn_transformer.py train           # train all
#   python task2_walker2d_cnn_transformer.py train SAC CNN   # train specific
#   python task2_walker2d_cnn_transformer.py test            # evaluate all
#   python task2_walker2d_cnn_transformer.py compare         # show results
################################################################################

import os
import sys
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
import gymnasium
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ==============================================================================
# CONFIGURATION
# ==============================================================================

ENVIRONMENT_ID = "Walker2d-v5"
ALGORITHMS = ["SAC", "TD3"]
ARCHITECTURES = ["CNN", "Transformer"]
SEEDS = [55, 234, 678]  # different from MLP seeds [42, 123, 456] to avoid bias
TRAINING_TIMESTEPS = int(1e6)  # 1M steps (more than MLP's 500K for better results)
NUM_EVAL_EPISODES = 20
POLICY_DIR = "./policies"
LOG_DIR = "./logs"
RESULTS_FILE = "./task2_cnn_transformer_results.json"


# ==============================================================================
# CNN FEATURE EXTRACTOR
#
# Uses 1D convolutions to process the observation vector.
# Treats the 17 observation values like a 1D signal and slides
# filters across it to detect patterns between neighbouring values.
#
# How 1D convolution works on Walker2d's 17 observations:
#   Input: [j1_pos, j1_vel, j2_pos, j2_vel, ..., torso_angle]
#
#   Filter of size 3 slides across:
#     [j1_pos, j1_vel, j2_pos] → detects pattern between joint 1 and 2
#          [j1_vel, j2_pos, j2_vel] → detects transition between joints
#               [j2_pos, j2_vel, j3_pos] → detects pattern in joint 2-3
#                    ...continues sliding...
#
# This captures LOCAL relationships — nearby joints that move together.
# ==============================================================================

class CNNFeatureExtractor(BaseFeaturesExtractor):
    """
    1D CNN feature extractor for vector observations.

    Architecture:
        Input (17,) → reshape to (1, 17) → Conv1d layers → flatten → Linear → features

    Conv1d processes the observation as a 1D sequence:
    - kernel_size=3: each filter sees 3 consecutive values
    - Multiple layers with increasing channels detect increasingly complex patterns
    - MaxPool reduces dimensionality between layers
    """
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)

        obs_size = observation_space.shape[0]  # 17 for Walker2d

        # Normalise raw observations BEFORE processing
        self.input_norm = nn.LayerNorm(obs_size)

        self.cnn = nn.Sequential(
            # Layer 1: 1 input channel, 32 filters, kernel size 3
            # Input: [batch, 1, 17]  → Output: [batch, 32, 15]
            # Each filter detects a pattern across 3 consecutive observation values
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=0),
            nn.ReLU(),

            # Layer 2: 32 input channels, 64 filters, kernel size 3
            # Input: [batch, 32, 15] → Output: [batch, 64, 13]
            # Combines patterns from layer 1 into higher-level features
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=0),
            nn.ReLU(),

            # Layer 3: 64 input channels, 64 filters, kernel size 3
            # Input: [batch, 64, 13] → Output: [batch, 64, 11]
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=0),
            nn.ReLU(),

            # Adaptive pool to fixed size regardless of input length
            nn.AdaptiveAvgPool1d(1),  # [batch, 64, 1]
        )

        # Final projection with LayerNorm for stability
        self.linear = nn.Sequential(
            nn.Flatten(),              # [batch, 64]
            nn.LayerNorm(64),          # normalise before output
            nn.Linear(64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        # observations: [batch, 17]
        x = self.input_norm(observations)  # [batch, 17] — normalise inputs
        x = x.unsqueeze(1)                 # [batch, 1, 17] — add channel dimension
        x = self.cnn(x)                    # [batch, 64, 1]
        x = self.linear(x)                 # [batch, features_dim]
        return x


# ==============================================================================
# TRANSFORMER FEATURE EXTRACTOR
#
# Uses self-attention to process the observation vector.
# Each of the 17 observation values becomes a "token" that can attend
# to all other tokens — discovering which joints are relevant to each other.
#
# How self-attention works on Walker2d's observations:
#   Each joint value asks: "which other values help me decide what to do?"
#
#   torso_angle attends to:  ALL joints (am I balanced?)
#   knee_velocity attends to: ankle_pos, hip_pos (coordinated movement)
#   ankle_pos attends to:    ground_contact, torso_angle (stability)
#
# This captures GLOBAL relationships — any joint can interact with any other.
# Unlike CNN which only sees local neighbours.
# ==============================================================================

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Transformer feature extractor for vector observations.

    Architecture:
        Input (17,) → LayerNorm → reshape to (17, 1) → project to d_model
        → add positional encoding → TransformerEncoder → mean pool
        → LayerNorm → Linear → features

    FIX: Added LayerNorm at input and output to prevent exploding gradients.
    Walker2d observations have very different scales (positions ~0-2,
    velocities ~-10 to 10). Without normalisation, the Transformer's
    attention scores blow up, causing critic loss to explode (7.57e+24).

    Each observation value becomes a token:
    - Token 0: joint1_position → "I am joint 1's position"
    - Token 1: joint1_velocity → "I am joint 1's velocity"
    - ...
    - Token 16: torso_angle → "I am the torso angle"

    Self-attention lets every token see every other token,
    learning which joints are most relevant to each other.
    """
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)

        obs_size = observation_space.shape[0]  # 17 for Walker2d
        d_model = 64  # embedding dimension for each token

        # Normalise raw observations BEFORE processing
        # Walker2d values range wildly: positions ~0-2, velocities ~-10 to 10
        # Without this, attention scores explode
        self.input_norm = nn.LayerNorm(obs_size)

        # Project each observation value (scalar) to d_model dimensions
        self.input_projection = nn.Linear(1, d_model)

        # Positional encoding: tells the Transformer WHICH joint each token represents
        # Without this, the Transformer can't distinguish joint 1 from joint 5
        # Small init (0.02) to prevent early instability
        self.pos_encoding = nn.Parameter(torch.randn(1, obs_size, d_model) * 0.02)

        # Transformer encoder: 2 layers, 4 attention heads
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,              # 4 attention heads (each sees different relationships)
            dim_feedforward=128,  # feed-forward hidden size
            dropout=0.1,
            batch_first=True,
            norm_first=True       # Pre-norm (more stable than post-norm)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Normalise output before passing to actor/critic
        # Prevents large values from destabilising SAC's entropy coefficient
        self.output_norm = nn.LayerNorm(d_model)

        # Project pooled output to features_dim
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # observations: [batch, 17]
        x = self.input_norm(observations)    # [batch, 17] — normalise inputs
        x = x.unsqueeze(-1)                  # [batch, 17, 1] — each value is a token
        x = self.input_projection(x)         # [batch, 17, 64] — project to d_model
        x = x + self.pos_encoding            # [batch, 17, 64] — add positional info
        x = self.transformer(x)              # [batch, 17, 64] — self-attention
        x = x.mean(dim=1)                    # [batch, 64] — average pool all tokens
        x = self.output_norm(x)              # [batch, 64] — normalise before output
        x = self.output_projection(x)        # [batch, features_dim]
        return x


# ==============================================================================
# HYPERPARAMETERS (same as MLP experiments for fair comparison)
# ==============================================================================

HYPERPARAMS = {
    "SAC": {
        "learning_rate": 1e-4,   # lower than MLP (3e-4) — custom extractors need gentler updates
        "gamma": 0.99,
        "buffer_size": 300000,
        "batch_size": 256,
        "tau": 0.005,
        "ent_coef": "auto",
        "learning_starts": 10000,
    },
    "TD3": {
        "learning_rate": 1e-4,   # lower than MLP (3e-4) — custom extractors need gentler updates
        "gamma": 0.99,
        "buffer_size": 300000,
        "batch_size": 256,
        "tau": 0.005,
        "policy_delay": 2,
        "learning_starts": 10000,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5,
    },
}


# ==============================================================================
# ENVIRONMENT AND MODEL CREATION
# ==============================================================================

def create_environment(env_id, seed, log_dir=None):
    """Creates a vectorised MuJoCo environment."""
    env = gymnasium.make(env_id)
    env = DummyVecEnv([lambda: env])
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        env = VecMonitor(env, log_dir)
    return env


def create_model(algorithm, architecture, env, seed, log_dir):
    """Creates a model with the specified algorithm and feature extractor."""
    params = HYPERPARAMS[algorithm]

    # Select feature extractor based on architecture
    if architecture == "CNN":
        policy_kwargs = {
            "features_extractor_class": CNNFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 64},
        }
    elif architecture == "Transformer":
        policy_kwargs = {
            "features_extractor_class": TransformerFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 64},
        }
    else:
        policy_kwargs = {}

    if algorithm == "SAC":
        model = SAC(
            "MlpPolicy", env, seed=seed,
            policy_kwargs=policy_kwargs,
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            buffer_size=params["buffer_size"],
            batch_size=params["batch_size"],
            tau=params["tau"],
            ent_coef=params["ent_coef"],
            learning_starts=params["learning_starts"],
            verbose=1,
            tensorboard_log=log_dir,
            device="auto"
        )
    elif algorithm == "TD3":
        model = TD3(
            "MlpPolicy", env, seed=seed,
            policy_kwargs=policy_kwargs,
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            buffer_size=params["buffer_size"],
            batch_size=params["batch_size"],
            tau=params["tau"],
            policy_delay=params["policy_delay"],
            learning_starts=params["learning_starts"],
            target_policy_noise=params["target_policy_noise"],
            target_noise_clip=params["target_noise_clip"],
            verbose=1,
            tensorboard_log=log_dir,
            device="auto"
        )

    return model


# ==============================================================================
# TRAINING
# ==============================================================================

def train_agent(algorithm, architecture, seed):
    """Trains a single agent and saves the policy."""
    print(f"\n{'='*60}")
    print(f"TRAINING: {algorithm} + {architecture} | Seed: {seed}")
    print(f"Timesteps: {TRAINING_TIMESTEPS:,}")
    print(f"{'='*60}")

    run_name = f"{algorithm}-{architecture}-{ENVIRONMENT_ID}-seed{seed}"
    run_log_dir = os.path.join(LOG_DIR, run_name)
    os.makedirs(POLICY_DIR, exist_ok=True)

    env = create_environment(ENVIRONMENT_ID, seed, run_log_dir)
    model = create_model(algorithm, architecture, env, seed, run_log_dir)

    # Print architecture
    print(f"\nFeature Extractor ({architecture}):")
    print(model.policy.features_extractor)
    print(f"\nTotal policy parameters: {sum(p.numel() for p in model.policy.parameters()):,}")

    # Train
    start_time = time.time()
    model.learn(total_timesteps=TRAINING_TIMESTEPS)
    train_time = time.time() - start_time

    # Save
    policy_path = os.path.join(POLICY_DIR, f"{run_name}.zip")
    model.save(policy_path)
    print(f"Policy saved to {policy_path}")
    print(f"Training time: {train_time:.1f}s")

    env.close()
    return train_time


def train_all(filter_alg=None, filter_arch=None):
    """Trains all or filtered combinations."""
    training_times = {}

    for alg in ALGORITHMS:
        if filter_alg and alg != filter_alg:
            continue
        for arch in ARCHITECTURES:
            if filter_arch and arch != filter_arch:
                continue
            for seed in SEEDS:
                key = f"{alg}-{arch}-seed{seed}"
                train_time = train_agent(alg, arch, seed)
                training_times[key] = train_time

    # Save training times
    times_file = "./task2_cnn_transformer_training_times.json"
    # Load existing if present
    existing = {}
    if os.path.exists(times_file):
        with open(times_file) as f:
            existing = json.load(f)
    existing.update(training_times)
    with open(times_file, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"\nTraining times saved to {times_file}")


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_agent(algorithm, architecture, seed):
    """Evaluates a trained agent."""
    run_name = f"{algorithm}-{architecture}-{ENVIRONMENT_ID}-seed{seed}"
    policy_path = os.path.join(POLICY_DIR, f"{run_name}.zip")

    if not os.path.exists(policy_path):
        print(f"  Policy not found: {policy_path}")
        return None

    print(f"\n--- Evaluating: {algorithm} + {architecture} | Seed: {seed} ---")

    env = create_environment(ENVIRONMENT_ID, seed)

    if algorithm == "SAC":
        model = SAC.load(policy_path, env=env, device="auto")
    elif algorithm == "TD3":
        model = TD3.load(policy_path, env=env, device="auto")

    # Run evaluation episodes
    start_time = time.time()
    episode_rewards = []
    episode_steps = []

    for ep in range(NUM_EVAL_EPISODES):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            ep_steps += 1

        episode_rewards.append(ep_reward)
        episode_steps.append(ep_steps)

    test_time = time.time() - start_time

    results = {
        "algorithm": algorithm,
        "architecture": architecture,
        "seed": seed,
        "avg_episode_reward": float(np.mean(episode_rewards)),
        "std_episode_reward": float(np.std(episode_rewards)),
        "avg_steps_per_episode": float(np.mean(episode_steps)),
        "std_steps_per_episode": float(np.std(episode_steps)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "test_time": round(test_time, 2),
    }

    print(f"  Avg Reward: {results['avg_episode_reward']:.2f} ± {results['std_episode_reward']:.2f}")
    print(f"  Avg Steps:  {results['avg_steps_per_episode']:.1f}")

    env.close()
    return results


def evaluate_all():
    """Evaluates all trained agents."""
    all_results = []

    # Load training times if available
    times_file = "./task2_cnn_transformer_training_times.json"
    training_times = {}
    if os.path.exists(times_file):
        with open(times_file) as f:
            training_times = json.load(f)

    for alg in ALGORITHMS:
        for arch in ARCHITECTURES:
            for seed in SEEDS:
                result = evaluate_agent(alg, arch, seed)
                if result:
                    key = f"{alg}-{arch}-seed{seed}"
                    result["train_time"] = training_times.get(key, 0)
                    all_results.append(result)

    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    print_comparison_table(all_results)
    return all_results


# ==============================================================================
# COMPARISON TABLE
# ==============================================================================

def print_comparison_table(all_results=None):
    """Prints comparison table combining MLP, CNN, and Transformer results."""
    if all_results is None:
        if not os.path.exists(RESULTS_FILE):
            print("No results file found.")
            return
        with open(RESULTS_FILE) as f:
            all_results = json.load(f)

    # Also load MLP results if available
    mlp_results = []
    if os.path.exists("./task2_results.json"):
        with open("./task2_results.json") as f:
            mlp_data = json.load(f)
            for r in mlp_data:
                r["architecture"] = "MLP"
                mlp_results.append(r)

    combined = mlp_results + all_results

    print(f"\n{'='*100}")
    print(f"TASK 2 COMPLETE RESULTS: DRL Agents for {ENVIRONMENT_ID}")
    print(f"{'='*100}")

    # Per-seed results
    print(f"\n--- Per-Seed Results ---")
    print(f"{'Algorithm':<8} {'Arch':<14} {'Seed':<6} {'Avg Reward':<14} "
          f"{'Std Reward':<14} {'Avg Steps':<12} {'Train Time':<12}")
    print("-" * 80)
    for r in sorted(combined, key=lambda x: (x["algorithm"], x.get("architecture", "MLP"), x["seed"])):
        arch = r.get("architecture", "MLP")
        print(f"{r['algorithm']:<8} {arch:<14} {r['seed']:<6} "
              f"{r['avg_episode_reward']:<14.2f} {r['std_episode_reward']:<14.2f} "
              f"{r['avg_steps_per_episode']:<12.1f} {r.get('train_time', 0):<12.1f}")

    # Averaged results
    print(f"\n--- Averaged Across Seeds (for report) ---")
    print(f"{'Algorithm':<8} {'Architecture':<14} {'Avg Reward':<14} {'Std (seeds)':<14} "
          f"{'Avg Steps':<12} {'Timesteps':<12}")
    print("-" * 74)

    # Group by algorithm + architecture
    groups = {}
    for r in combined:
        key = (r["algorithm"], r.get("architecture", "MLP"))
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    ranked = []
    for (alg, arch), results in sorted(groups.items()):
        rewards = [r["avg_episode_reward"] for r in results]
        steps = [r["avg_steps_per_episode"] for r in results]
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        avg_steps = np.mean(steps)

        # Determine timesteps
        if arch == "MLP":
            timesteps = "500K"
        else:
            timesteps = "1M"

        ranked.append((alg, arch, avg_reward, std_reward, avg_steps, timesteps))

        print(f"{alg:<8} {arch:<14} {avg_reward:<14.2f} {std_reward:<14.2f} "
              f"{avg_steps:<12.1f} {timesteps:<12}")

    # Rank by avg reward
    ranked.sort(key=lambda x: x[2], reverse=True)
    print(f"\n--- RANKING ---")
    for i, (alg, arch, reward, std, steps, ts) in enumerate(ranked, 1):
        print(f"  {i}. {alg} + {arch} (avg reward: {reward:.2f} ± {std:.2f}) [{ts} steps]")

    print(f"{'='*100}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("USAGE:")
        print("  python task2_walker2d_cnn_transformer.py train              # train all")
        print("  python task2_walker2d_cnn_transformer.py train SAC CNN      # train specific")
        print("  python task2_walker2d_cnn_transformer.py train TD3 Transformer")
        print("  python task2_walker2d_cnn_transformer.py test               # evaluate all")
        print("  python task2_walker2d_cnn_transformer.py compare            # show all results")
        sys.exit(0)

    mode = sys.argv[1]

    if mode == "train":
        filter_alg = sys.argv[2] if len(sys.argv) > 2 else None
        filter_arch = sys.argv[3] if len(sys.argv) > 3 else None
        train_all(filter_alg, filter_arch)

    elif mode == "test":
        evaluate_all()

    elif mode == "compare":
        print_comparison_table()

    else:
        print(f"Unknown mode: {mode}")
