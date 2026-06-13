################################################################################
# task2_walker2d_drl.py
#
#  Some of the functions here were adapted from the workshop
#
# Task 2: Deep Reinforcement Learning for Robot Learning
# Environment: Walker2d-v5 (MuJoCo — two-legged robot learns to walk)
#
# Trains 3 DRL agents with different learning algorithms:
# 1. PPO  — Proximal Policy Optimisation (on-policy, actor-critic)
# 2. SAC  — Soft Actor-Critic (off-policy, entropy-regularised)
# 3. TD3  — Twin Delayed DDPG (off-policy, twin critics)
#
# Each agent is trained with 3 different seeds and results are averaged.
# This follows the assessment brief requirement:
#   "you should train them with multiple seeds (3 at least) and average
#    their results — to reduce the noise (due to randomness)"
#
# Adapted from aml_continuous_drl_agents.py (CMP9137M workshop):
# - Same DRL_Agent class pattern (environment setup, model creation, training)
# - Same use of StableBaselines3 library
# - Same evaluation via evaluate_policy()
# - ADDED: multi-seed training, results averaging, comparison table
# - ADDED: learning curve logging for report plots
#
# Walker2d-v5 environment:
# - Observation: 17-d vector (joint positions, velocities, torso angle, etc.)
# - Action: 6-d continuous vector (torques applied to 6 joints)
# - Reward: forward velocity + healthy bonus - control cost
# - Goal: walk forward as fast as possible without falling
#
# Algorithms compared:
# - PPO: On-policy. Collects experience, updates policy, discards data. Stable but
#   needs more samples. Uses clipped surrogate objective to prevent too-large updates.
# - SAC: Off-policy. Stores experience in replay buffer and reuses it. Maximises
#   both reward AND entropy (randomness) — encourages exploration.
# - TD3: Off-policy. Uses TWO critic networks and takes the minimum to prevent
#   overestimation. Delays policy updates to stabilise training.
#
# Reference: Raffin et al. "Stable-Baselines3: Reliable Reinforcement Learning
#            Implementations", JMLR, 2021.
# Reference: Towers et al. "Gymnasium: A Standard Interface for RL Environments",
#            NeurIPS Datasets and Benchmarks, 2025.
#
# Usage:
#   python task2_walker2d_drl.py train         # train all agents (all seeds)
#   python task2_walker2d_drl.py train PPO     # train only PPO (all seeds)
#   python task2_walker2d_drl.py test          # evaluate all saved agents
#   python task2_walker2d_drl.py compare       # show comparison table from saved results
#
# Setup:
#   pip install stable-baselines3 gymnasium[mujoco]
################################################################################

import os
import sys
import json
import time
import pickle
import random
import numpy as np
import gymnasium
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback


# ==============================================================================
# CONFIGURATION
# ==============================================================================

ENVIRONMENT_ID = "Walker2d-v5"

# Algorithms to compare (the brief requires at least 3)
ALGORITHMS = ["PPO", "SAC", "TD3"]

# Seeds for each algorithm (the brief requires at least 3)
SEEDS = [77, 200, 431]

# Training timesteps (Walker2d typically needs 500K-1M for decent results)
TRAINING_TIMESTEPS = int(1.5e6)  # 1,500,000 steps per seed

# Evaluation settings
NUM_EVAL_EPISODES = 20  # episodes for final evaluation

# Output directories
POLICY_DIR = "./policies"
LOG_DIR = "./logs"
RESULTS_FILE = "./task2_results.json"


# ==============================================================================
# HYPERPARAMETERS FOR EACH ALGORITHM
#
# These are tuned for Walker2d based on StableBaselines3 documentation
# and RL literature. Each algorithm has different optimal settings.
# ==============================================================================

HYPERPARAMS = {
    "PPO": {
        # On-policy: collects n_steps of experience, then updates
        "learning_rate": 3e-4,
        "gamma": 0.99,           # discount factor
        "n_steps": 2048,         # steps before each update
        "batch_size": 64,        # minibatch size for updates
        "n_epochs": 10,          # number of passes through collected data
        "clip_range": 0.2,       # PPO clipping parameter
        "ent_coef": 0.0,         # entropy bonus (encourages exploration)
        "gae_lambda": 0.95,      # GAE lambda for advantage estimation
    },
    "SAC": {
        # Off-policy: stores experience in replay buffer
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "buffer_size": 300000,   # replay buffer size
        "batch_size": 256,       # batch size for training
        "tau": 0.005,            # soft update coefficient for target networks
        "ent_coef": "auto",      # auto-tune entropy coefficient
        "learning_starts": 10000, # random actions before training starts
    },
    "TD3": {
        # Off-policy with twin critics and delayed updates
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "buffer_size": 300000,
        "batch_size": 256,
        "tau": 0.005,
        "policy_delay": 2,       # update policy every N critic updates
        "learning_starts": 10000,
        # TD3 adds noise to target actions for smoothing
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5,
    },
}


# ==============================================================================
# ENVIRONMENT CREATION
# Adapted from create_environment() in aml_continuous_drl_agents.py
# ==============================================================================

def create_environment(env_id, seed, log_dir=None):
    """
    Creates a vectorised MuJoCo environment.
    Adapted from DRL_Agent.create_environment() in the workshop.
    
    Walker2d-v5 observations are a 17-d vector (not images), so we use
    MlpPolicy (not CnnPolicy). No image wrappers needed.
    """
    env = gymnasium.make(env_id)
    env = DummyVecEnv([lambda: env])

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        env = VecMonitor(env, log_dir)

    return env


# ==============================================================================
# MODEL CREATION
# Adapted from create_model() in aml_continuous_drl_agents.py
# ==============================================================================

def create_model(algorithm, env, seed, log_dir):
    """
    Creates a StableBaselines3 model with algorithm-specific hyperparameters.
    
    Workshop uses: PPO(policy, env, seed, lr, gamma, verbose=1)
    We add algorithm-specific hyperparameters for better performance.
    """
    params = HYPERPARAMS[algorithm]

    if algorithm == "PPO":
        model = PPO(
            "MlpPolicy", env, seed=seed,
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            n_steps=params["n_steps"],
            batch_size=params["batch_size"],
            n_epochs=params["n_epochs"],
            clip_range=params["clip_range"],
            ent_coef=params["ent_coef"],
            gae_lambda=params["gae_lambda"],
            verbose=1,
            tensorboard_log=log_dir
        )

    elif algorithm == "SAC":
        model = SAC(
            "MlpPolicy", env, seed=seed,
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            buffer_size=params["buffer_size"],
            batch_size=params["batch_size"],
            tau=params["tau"],
            ent_coef=params["ent_coef"],
            learning_starts=params["learning_starts"],
            verbose=1,
            tensorboard_log=log_dir
        )

    elif algorithm == "TD3":
        model = TD3(
            "MlpPolicy", env, seed=seed,
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
            tensorboard_log=log_dir
        )

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return model


# ==============================================================================
# TRAINING
# Adapted from train_or_load_model() in the workshop
# ==============================================================================

def train_agent(algorithm, seed):
    """
    Trains a single agent with a specific algorithm and seed.
    Returns training time.
    """
    print(f"\n{'='*60}")
    print(f"TRAINING: {algorithm} | Seed: {seed} | Env: {ENVIRONMENT_ID}")
    print(f"{'='*60}")

    # Setup directories
    run_log_dir = os.path.join(LOG_DIR, f"{algorithm}_seed{seed}")
    os.makedirs(POLICY_DIR, exist_ok=True)

    # Create environment and model
    env = create_environment(ENVIRONMENT_ID, seed, run_log_dir)
    model = create_model(algorithm, env, seed, run_log_dir)

    # Print architecture (same as workshop's print_neural_architectures)
    print(f"\nNeural Architecture for {algorithm}:")
    if algorithm == "PPO":
        print("Policy network:", model.policy.mlp_extractor)
        print("Actor:", model.policy.action_net)
        print("Critic:", model.policy.value_net)
    else:
        print("Actor:", model.policy.actor)
        print("Critic:", model.policy.critic)

    # Train
    start_time = time.time()
    model.learn(total_timesteps=TRAINING_TIMESTEPS)
    train_time = time.time() - start_time

    # Save policy (same pattern as workshop)
    policy_path = os.path.join(POLICY_DIR, f"{algorithm}-{ENVIRONMENT_ID}-seed{seed}.zip")
    model.save(policy_path)
    print(f"Policy saved to {policy_path}")
    print(f"Training time: {train_time:.1f}s")

    env.close()
    return train_time


# ==============================================================================
# EVALUATION
# Adapted from evaluate_policy() and render_policy() in aml_continuous_drl_agents.py
# ==============================================================================

def evaluate_agent(algorithm, seed):
    """
    Evaluates a trained agent. Returns metrics dict.
    Adapted from DRL_Agent.evaluate_policy() in the workshop.
    """
    print(f"\n--- Evaluating: {algorithm} | Seed: {seed} ---")

    # Load saved model
    policy_path = os.path.join(POLICY_DIR, f"{algorithm}-{ENVIRONMENT_ID}-seed{seed}.zip")
    if not os.path.exists(policy_path):
        print(f"  Policy not found: {policy_path}")
        return None

    env = create_environment(ENVIRONMENT_ID, seed)

    # Load model based on algorithm
    if algorithm == "PPO":
        model = PPO.load(policy_path, env=env)
    elif algorithm == "SAC":
        model = SAC.load(policy_path, env=env)
    elif algorithm == "TD3":
        model = TD3.load(policy_path, env=env)

    # Evaluate (same as workshop's evaluate_policy)
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
    print(f"  Avg Steps:  {results['avg_steps_per_episode']:.1f} ± {results['std_steps_per_episode']:.1f}")
    print(f"  Min/Max Reward: {results['min_reward']:.2f} / {results['max_reward']:.2f}")

    env.close()
    return results


# ==============================================================================
# MULTI-SEED TRAINING AND EVALUATION
# ==============================================================================

def train_all(algorithms=None):
    """Trains all algorithms with all seeds."""
    if algorithms is None:
        algorithms = ALGORITHMS

    all_results = []
    training_times = {}

    for alg in algorithms:
        training_times[alg] = {}
        for seed in SEEDS:
            train_time = train_agent(alg, seed)
            training_times[alg][seed] = train_time

    # Save training times
    with open("./task2_training_times.json", 'w') as f:
        json.dump(training_times, f, indent=2)
    print("\nTraining times saved to task2_training_times.json")


def evaluate_all():
    """Evaluates all trained agents and computes averaged results."""
    all_results = []

    # Load training times if available
    training_times = {}
    if os.path.exists("./task2_training_times.json"):
        with open("./task2_training_times.json") as f:
            training_times = json.load(f)

    for alg in ALGORITHMS:
        for seed in SEEDS:
            result = evaluate_agent(alg, seed)
            if result:
                # Add training time if available
                if alg in training_times and str(seed) in training_times[alg]:
                    result["train_time"] = training_times[alg][str(seed)]
                elif alg in training_times and seed in training_times[alg]:
                    result["train_time"] = training_times[alg][seed]
                else:
                    result["train_time"] = 0
                all_results.append(result)

    # Save all individual results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {RESULTS_FILE}")

    # Compute and display averaged results
    print_comparison_table(all_results)

    return all_results


# ==============================================================================
# COMPARISON TABLE
# Produces the averaged results table required by the brief
# ==============================================================================

def print_comparison_table(all_results=None):
    """
    Prints a comparison table with averaged results across seeds.
    This is what goes in your report.
    """
    if all_results is None:
        if not os.path.exists(RESULTS_FILE):
            print("No results file found. Run evaluation first.")
            return
        with open(RESULTS_FILE) as f:
            all_results = json.load(f)

    print(f"\n{'='*90}")
    print(f"TASK 2 RESULTS: DRL Agents for {ENVIRONMENT_ID}")
    print(f"Training timesteps: {TRAINING_TIMESTEPS:,} | Eval episodes: {NUM_EVAL_EPISODES}")
    print(f"{'='*90}")

    # Per-seed results
    print(f"\n--- Per-Seed Results ---")
    print(f"{'Algorithm':<10} {'Seed':<8} {'Avg Reward':<14} {'Std Reward':<14} "
          f"{'Avg Steps':<12} {'Train Time':<12}")
    print("-" * 70)
    for r in sorted(all_results, key=lambda x: (x["algorithm"], x["seed"])):
        print(f"{r['algorithm']:<10} {r['seed']:<8} "
              f"{r['avg_episode_reward']:<14.2f} {r['std_episode_reward']:<14.2f} "
              f"{r['avg_steps_per_episode']:<12.1f} {r.get('train_time', 0):<12.1f}")

    # Averaged results (across seeds)
    print(f"\n--- Averaged Across Seeds (for report) ---")
    print(f"{'Algorithm':<10} {'Avg Reward':<14} {'Std (seeds)':<14} "
          f"{'Avg Steps':<12} {'Avg Train(s)':<14} {'Avg Test(s)':<12}")
    print("-" * 76)

    for alg in ALGORITHMS:
        alg_results = [r for r in all_results if r["algorithm"] == alg]
        if not alg_results:
            continue

        rewards = [r["avg_episode_reward"] for r in alg_results]
        steps = [r["avg_steps_per_episode"] for r in alg_results]
        train_times = [r.get("train_time", 0) for r in alg_results]
        test_times = [r["test_time"] for r in alg_results]

        print(f"{alg:<10} "
              f"{np.mean(rewards):<14.2f} {np.std(rewards):<14.2f} "
              f"{np.mean(steps):<12.1f} {np.mean(train_times):<14.1f} "
              f"{np.mean(test_times):<12.2f}")

    print(f"{'='*90}")

    # Best algorithm
    avg_by_alg = {}
    for alg in ALGORITHMS:
        alg_results = [r for r in all_results if r["algorithm"] == alg]
        if alg_results:
            avg_by_alg[alg] = np.mean([r["avg_episode_reward"] for r in alg_results])

    if avg_by_alg:
        best_alg = max(avg_by_alg, key=avg_by_alg.get)
        print(f"\nBEST ALGORITHM: {best_alg} (avg reward: {avg_by_alg[best_alg]:.2f})")

    # Algorithm descriptions for report
    print(f"\n--- Algorithm Descriptions ---")
    print(f"PPO: On-policy, actor-critic. Clips policy updates to prevent instability.")
    print(f"     Uses collected experience once then discards. Stable but sample-inefficient.")
    print(f"SAC: Off-policy, entropy-regularised. Replay buffer for sample efficiency.")
    print(f"     Maximises reward AND entropy (exploration). Auto-tunes temperature.")
    print(f"TD3: Off-policy, twin critics. Two Q-networks prevent overestimation.")
    print(f"     Delayed policy updates and target smoothing for stability.")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("USAGE:")
        print("  python task2_walker2d_drl.py train          # train all agents")
        print("  python task2_walker2d_drl.py train PPO      # train only PPO")
        print("  python task2_walker2d_drl.py test           # evaluate all agents")
        print("  python task2_walker2d_drl.py compare        # show comparison table")
        sys.exit(0)

    mode = sys.argv[1]

    if mode == "train":
        if len(sys.argv) > 2:
            # Train specific algorithm only
            alg = sys.argv[2]
            if alg not in ALGORITHMS:
                print(f"Unknown algorithm: {alg}. Choose from {ALGORITHMS}")
                sys.exit(1)
            train_all([alg])
        else:
            train_all()

    elif mode == "test":
        evaluate_all()

    elif mode == "compare":
        print_comparison_table()

    else:
        print(f"Unknown mode: {mode}. Use 'train', 'test', or 'compare'.")
