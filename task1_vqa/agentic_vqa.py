################################################################################
# agentic_vqa.py
#
# Agentic AI for Visual Question Answering experimentation on SLAKE.
#
# Adapted from AgenticML_ForQuestionAnswering.py (CMP9137M workshop):
# - Same LangGraph state machine: perceive → select_action → run_experiment → update_history
# - Same EnvironmentState dataclass pattern
# - Same experiment grid generation and history tracking
# - REPLACED simulated experiments with actual training & evaluation
# - ADDED: GPU memory cleanup between experiments
# - ADDED: Resume capability (saves/loads progress to JSON)
#
# Requires: pip install langgraph
#
# Dataset: SLAKE (BoKelvin/SLAKE on HuggingFace)
################################################################################

import os
import gc
import json
import random
import time
import torch
import torch.nn as nn
from typing import List, Dict
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from langgraph.graph import StateGraph, END
from transformers import DistilBertTokenizer

from vqa_utils import (
    download_slake_images, load_slake_annotations, verify_images,
    build_answer_vocabulary, SLAKEDataset, SLAKEDataset_LargeImage,
    train_model, evaluate_model
)
from vqa_models import MODEL_REGISTRY


# ==============================================================================
# ENVIRONMENT STATE (adapted from workshop's EnvironmentState)
# ==============================================================================

@dataclass
class EnvironmentState():
    experiment_id: str = ""
    model_type: str = ""
    learning_rate: float = 3e-5
    layers_unfrozen: int = 2
    embedding_dim: int = 256
    action: str = ""
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    f1_score: float = 0.0
    mrr: float = 0.0
    ece: float = 0.0
    running_time: float = 0.0
    history: List[Dict] = field(default_factory=list)
    step: int = 0


# ==============================================================================
# ENVIRONMENT (adapted from workshop's Environment)
# ==============================================================================

class Environment:
    """Manages data, model building, training, and evaluation."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load data once (shared across all experiments)
        self.imgs_dir = download_slake_images("./data/slake")
        self.train_data, self.val_data, self.test_data = load_slake_annotations()
        verify_images(self.train_data, self.imgs_dir, "train")
        verify_images(self.val_data, self.imgs_dir, "val")
        verify_images(self.test_data, self.imgs_dir, "test")
        self.answer2idx, self.idx2answer = build_answer_vocabulary(self.train_data)
        self.num_classes = len(self.answer2idx)
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def build_model(self, config):
        """Builds a model from config using the MODEL_REGISTRY."""
        model_type = config["model_type"]
        model_class = MODEL_REGISTRY[model_type]

        # Map config to constructor arguments based on model type
        if "resnet" in model_type:
            layer_key = "num_resnet_layers_to_finetune"
        elif "efficientnet" in model_type:
            layer_key = "num_layers_to_finetune"
        elif "cross_attention" in model_type:
            layer_key = "num_vit_blocks_to_finetune"
        else:
            layer_key = "num_vit_blocks_to_finetune"

        kwargs = {
            "num_classes": self.num_classes,
            "embedding_dim": config["embedding_dim"],
            "pretrained": True,
            layer_key: config["layers_unfrozen"],
            "num_text_layers_to_finetune": 1,
        }

        return model_class(**kwargs).to(self.device)

    def get_data_loaders(self, config, batch_size=16):
        """Creates data loaders with appropriate image size."""
        if config["model_type"] == "efficientnet":
            dataset_class = SLAKEDataset_LargeImage
            batch_size = 12
        else:
            dataset_class = SLAKEDataset

        loaders = {}
        for name, data in [("train", self.train_data), ("val", self.val_data), ("test", self.test_data)]:
            ds = dataset_class(data, self.imgs_dir, self.answer2idx,
                               self.tokenizer, max_length=64, split=name)
            loaders[name] = DataLoader(ds, batch_size=batch_size,
                                       shuffle=(name == "train"))
            if name == "test":
                loaders["test_dataset"] = ds

        return loaders

    def cleanup_gpu(self):
        """Frees GPU memory between experiments."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"  GPU memory: {allocated:.0f}MB allocated, {reserved:.0f}MB reserved")

    def run_experiment(self, state, exp_config):
        """
        Runs a single experiment: build → train → evaluate → cleanup.

        This REPLACES the workshop's simulated experiment:
            state.performance = round(random.uniform(0.4, 0.7), 4)  # SIMULATED
        With actual model training and evaluation.
        """
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {state.experiment_id}")
        print(f"  Model: {exp_config['model_type']}")
        print(f"  LR: {exp_config['learning_rate']}")
        print(f"  Layers unfrozen: {exp_config['layers_unfrozen']}")
        print(f"  Embedding dim: {exp_config['embedding_dim']}")
        print(f"{'='*60}")

        start_time = time.time()

        try:
            # Build model
            model = self.build_model(exp_config)

            # Get data loaders
            loaders = self.get_data_loaders(exp_config)

            # Train
            criterion = nn.CrossEntropyLoss()
            optimiser = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=exp_config["learning_rate"], weight_decay=1e-4
            )

            checkpoint_path = f"./checkpoints/agentic_{state.experiment_id}.pth"
            train_model(
                model, loaders["train"], loaders["val"], criterion, optimiser,
                self.device, num_epochs=30, use_early_stopping=True,
                answer2idx=self.answer2idx, checkpoint_path=checkpoint_path
            )

            # Evaluate
            results = evaluate_model(model, loaders["test"], loaders["test_dataset"],
                                     self.device, self.idx2answer)

            state.accuracy = round(results["accuracy"], 4)
            state.balanced_accuracy = round(results["balanced_accuracy"], 4)
            state.f1_score = round(results["f1_weighted"], 4)
            state.mrr = round(results["mrr"], 4)
            state.ece = round(results["ece"], 4)

        except Exception as e:
            print(f"  EXPERIMENT FAILED: {e}")
            state.accuracy = 0.0
            state.balanced_accuracy = 0.0
            state.f1_score = 0.0
            state.mrr = 0.0
            state.ece = 1.0

        state.running_time = round(time.time() - start_time, 1)

        # GPU cleanup — free all model memory before next experiment
        print("  Cleaning up GPU memory...")
        try:
            del model, criterion, optimiser, loaders
        except:
            pass
        self.cleanup_gpu()

        return state


# ==============================================================================
# PROGRESS SAVE/LOAD (not in workshop — added for resume capability)
# ==============================================================================

PROGRESS_FILE = "./agentic_progress.json"

def save_progress(state):
    """Saves experiment history so we can resume later."""
    data = {
        "step": state.step,
        "history": state.history
    }
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Progress saved ({len(state.history)} experiments completed)")


def load_progress():
    """Loads previous progress. Returns (step, history) or (0, [])."""
    if not os.path.exists(PROGRESS_FILE):
        print("  No previous progress found — starting fresh")
        return 0, []

    with open(PROGRESS_FILE, 'r') as f:
        data = json.load(f)

    step = data.get("step", 0)
    history = data.get("history", [])
    print(f"  Loaded progress: {len(history)} experiments already completed (step {step})")
    return step, history


# ==============================================================================
# AGENTIC EXPERIMENTER (adapted from workshop's Agentic_ML_Experimenter)
# ==============================================================================

class Agentic_VQA_Experimenter:
    """
    Agentic AI experiment runner for VQA.

    Same LangGraph structure as workshop:
        perceive → select_action → run_experiment → update_history → END
    """
    def __init__(self):
        self.env = Environment()
        self.ml_experiments = self.initialise_list_of_experiments()
        print(f"\nTotal experiments in grid: {len(self.ml_experiments)}")

        # Build LangGraph (IDENTICAL structure to workshop)
        graph = StateGraph(EnvironmentState)
        graph.add_node("perceive", self.perceive)
        graph.add_node("select_action", self.select_action)
        graph.add_node("run_experiment", self.run_experiment)
        graph.add_node("update_history", self.update_history)
        graph.set_entry_point("perceive")
        graph.add_edge("perceive", "select_action")
        graph.add_edge("select_action", "run_experiment")
        graph.add_edge("run_experiment", "update_history")
        graph.add_edge("update_history", END)
        self.agent = graph.compile()

    def initialise_list_of_experiments(self):
        """
        Generates experiment grid.
        Workshop: models × prompts × RAG × LoRA
        Ours: model_types × learning_rates × layers_unfrozen
        """
        model_types = [
            "vit_concat", "vit_hadamard",
            "resnet_concat", "resnet_hadamard",
            "efficientnet", "cross_attention"
        ]
        learning_rates = [3e-5, 1e-5]
        layers_unfrozen = [2, 4]
        embedding_dims = [256]

        experiments = {}
        exp_id = 0
        for m in model_types:
            for lr in learning_rates:
                for lu in layers_unfrozen:
                    for ed in embedding_dims:
                        exp_id += 1
                        experiments[f"exp_{exp_id}"] = {
                            "model_type": m,
                            "learning_rate": lr,
                            "layers_unfrozen": lu,
                            "embedding_dim": ed
                        }
        return experiments

    def perceive(self, state):
        """Observe current state. (Same as workshop)"""
        done = len(state.history)
        remaining = len(self.ml_experiments) - done
        print(f"\n--- PERCEIVE --- Done: {done}, Remaining: {remaining}")
        if done > 0:
            best = max(state.history, key=lambda r: r["balanced_accuracy"])
            print(f"  Best so far: {best['experiment_id']} "
                  f"(bal_acc={best['balanced_accuracy']}, acc={best['accuracy']})")
        return state

    def select_action(self, state):
        """Pick next experiment. (Same random selection as workshop)"""
        done_ids = [h["experiment_id"] for h in state.history]
        undone = [eid for eid in self.ml_experiments if eid not in done_ids]

        if not undone:
            state.action = "none"
            return state

        chosen = random.choice(undone)
        state.action = "run_experiment"
        state.experiment_id = chosen
        return state

    def run_experiment(self, state):
        """Execute experiment. (Workshop called env.run_experiment)"""
        if state.action != "run_experiment":
            return state
        config = self.ml_experiments[state.experiment_id]
        self.env.run_experiment(state, config)
        return state

    def update_history(self, state):
        """Record results. (Adapted from workshop — we store more metrics)"""
        if state.action != "run_experiment":
            return state

        config = self.ml_experiments[state.experiment_id]
        record = {
            "experiment_id": state.experiment_id,
            "model_type": config["model_type"],
            "learning_rate": config["learning_rate"],
            "layers_unfrozen": config["layers_unfrozen"],
            "embedding_dim": config["embedding_dim"],
            "accuracy": state.accuracy,
            "balanced_accuracy": state.balanced_accuracy,
            "f1_score": state.f1_score,
            "mrr": state.mrr,
            "ece": state.ece,
            "runtime": state.running_time
        }
        state.history.append(record)

        # Save progress after each experiment (resume capability)
        save_progress(state)

        return state

    def print_results_table(self, state):
        """Print ranked comparison table."""
        if not state.history:
            return
        sorted_h = sorted(state.history, key=lambda r: r["balanced_accuracy"], reverse=True)
        print(f"\n{'='*110}")
        print("ALL EXPERIMENTS RANKED BY BALANCED ACCURACY")
        print(f"{'='*110}")
        print(f"{'Rank':<5} {'ID':<8} {'Model':<18} {'LR':<10} {'Unfrozen':<9} "
              f"{'Acc':<8} {'BalAcc':<8} {'F1':<8} {'MRR':<8} {'ECE':<8} {'Time':<8}")
        print("-" * 110)
        for rank, r in enumerate(sorted_h, 1):
            print(f"{rank:<5} {r['experiment_id']:<8} {r['model_type']:<18} "
                  f"{r['learning_rate']:<10.0e} {r['layers_unfrozen']:<9} "
                  f"{r['accuracy']:<8.4f} {r['balanced_accuracy']:<8.4f} "
                  f"{r['f1_score']:<8.4f} {r['mrr']:<8.4f} {r['ece']:<8.4f} "
                  f"{r['runtime']:<8.0f}")
        print(f"{'='*110}")

    def get_best_experiment(self, state):
        """Find and display best experiment. (Same as workshop)"""
        if not state.history:
            return
        best = max(state.history, key=lambda r: r["balanced_accuracy"])
        config = self.ml_experiments.get(best["experiment_id"], best)
        print(f"\n{'='*60}")
        print("BEST EXPERIMENT")
        print(f"{'='*60}")
        for k, v in best.items():
            print(f"  {k}: {v}")
        print(f"{'='*60}")

    def run(self, max_steps=None):
        """
        Main loop. (Same structure as workshop's run method)
        Supports resume: loads previous progress and skips completed experiments.
        """
        state = EnvironmentState()

        # Load previous progress (resume capability)
        prev_step, prev_history = load_progress()
        state.step = prev_step
        state.history = prev_history

        if max_steps is None:
            steps = len(self.ml_experiments)
        else:
            steps = min(max_steps, len(self.ml_experiments))

        remaining = steps - len(state.history)
        if remaining <= 0:
            print(f"\nAll {steps} experiments already completed!")
            self.print_results_table(state)
            self.get_best_experiment(state)
            return

        print(f"\nSTARTING agentic runner: {remaining} experiments to run "
              f"({len(state.history)} already done)")

        for i in range(remaining):
            state.step += 1
            print(f"\n{'#'*60}")
            print(f"STEP {state.step} (experiment {len(state.history)+1}/{steps})")
            print(f"{'#'*60}")

            state = self.agent.invoke(state)
            state = EnvironmentState(**state)

            if state.action == "none":
                break

            print(f"\n  Result: acc={state.accuracy}, bal_acc={state.balanced_accuracy}, "
                  f"time={state.running_time}s")

        print(f"\nFINISHED. {len(state.history)} experiments completed.")
        self.print_results_table(state)
        self.get_best_experiment(state)

        # Save final results
        with open("./agentic_results.json", 'w') as f:
            json.dump(state.history, f, indent=2)
        print("Results saved to ./agentic_results.json")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    experimenter = Agentic_VQA_Experimenter()

    # 6 models × 2 LR × 2 layers = 24 experiments
    # Each ~20-50 min → full run ~8-20 hours
    # Set max_steps to limit, or run overnight
    experimenter.run(max_steps=24)
