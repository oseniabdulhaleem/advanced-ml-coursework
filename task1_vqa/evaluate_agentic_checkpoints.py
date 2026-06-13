################################################################################
# evaluate_agentic_checkpoints.py
#
# Scans the checkpoints folder for agentic_exp_*.pth files, evaluates each one,
# and optionally continues training the promising ones.
#
# Usage:
#   python evaluate_agentic_checkpoints.py                  # evaluate all
#   python evaluate_agentic_checkpoints.py --resume exp_14  # resume specific
#   python evaluate_agentic_checkpoints.py --resume-best    # resume top 3
#
# Place this file in the same folder as vqa_utils.py and vqa_models.py
################################################################################

import os
import sys
import glob
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer

from vqa_utils import (
    download_slake_images, load_slake_annotations, verify_images,
    build_answer_vocabulary, SLAKEDataset, SLAKEDataset_LargeImage,
    train_model, evaluate_model
)
from vqa_models import MODEL_REGISTRY


# ==============================================================================
# EXPERIMENT GRID (must match agentic_vqa.py exactly)
# ==============================================================================

def rebuild_experiment_grid():
    """
    Rebuilds the same experiment grid as agentic_vqa.py.
    Must match initialise_list_of_experiments() exactly.
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


# ==============================================================================
# BUILD MODEL FROM CONFIG
# ==============================================================================

def build_model_from_config(config, num_classes, device):
    """Builds the correct model class based on experiment config."""
    model_type = config["model_type"]
    model_class = MODEL_REGISTRY[model_type]

    # Map config to correct constructor argument name
    if "resnet" in model_type:
        layer_key = "num_resnet_layers_to_finetune"
    elif "efficientnet" in model_type:
        layer_key = "num_layers_to_finetune"
    else:
        layer_key = "num_vit_blocks_to_finetune"

    kwargs = {
        "num_classes": num_classes,
        "embedding_dim": config["embedding_dim"],
        "pretrained": True,
        layer_key: config["layers_unfrozen"],
        "num_text_layers_to_finetune": 1,
    }

    model = model_class(**kwargs).to(device)
    return model


# ==============================================================================
# SCAN AND EVALUATE CHECKPOINTS
# ==============================================================================

def scan_checkpoints(checkpoint_dir="./checkpoints"):
    """Finds all agentic checkpoint files and returns their experiment IDs."""
    pattern = os.path.join(checkpoint_dir, "agentic_exp_*.pth")
    files = sorted(glob.glob(pattern))

    checkpoints = {}
    for f in files:
        # Extract experiment ID from filename: agentic_exp_14.pth → exp_14
        basename = os.path.basename(f).replace("agentic_", "").replace(".pth", "")
        checkpoints[basename] = f

    return checkpoints


# ==============================================================================
# SYNC TO PROGRESS.JSON
# Keeps agentic_vqa.py in sync so it doesn't re-run experiments
# ==============================================================================

PROGRESS_FILE = "./agentic_progress.json"

def sync_to_progress(result, config):
    """
    Saves a result to agentic_progress.json so agentic_vqa.py
    knows this experiment is done and won't re-run it.
    """
    # Load existing progress
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            progress = json.load(f)
    else:
        progress = {"step": 0, "history": []}

    # Check if this experiment is already in history
    existing_ids = [h["experiment_id"] for h in progress["history"]]
    exp_id = result["experiment_id"]

    # Build record matching agentic_vqa.py format
    record = {
        "experiment_id": exp_id,
        "model_type": config["model_type"],
        "learning_rate": config["learning_rate"],
        "layers_unfrozen": config["layers_unfrozen"],
        "embedding_dim": config["embedding_dim"],
        "accuracy": round(result["accuracy"], 4),
        "balanced_accuracy": round(result["balanced_accuracy"], 4),
        "f1_score": round(result["f1_weighted"], 4),
        "mrr": round(result["mrr"], 4),
        "ece": round(result["ece"], 4),
        "runtime": result.get("train_time", 0)
    }

    if exp_id in existing_ids:
        # Update existing entry
        for i, h in enumerate(progress["history"]):
            if h["experiment_id"] == exp_id:
                progress["history"][i] = record
                print(f"  Updated {exp_id} in {PROGRESS_FILE}")
                break
    else:
        # Add new entry
        progress["step"] += 1
        progress["history"].append(record)
        print(f"  Added {exp_id} to {PROGRESS_FILE}")

    # Save
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def evaluate_checkpoint(exp_id, checkpoint_path, config, data_bundle, device):
    """Loads a checkpoint and evaluates it."""
    imgs_dir, train_data, val_data, test_data, answer2idx, idx2answer, tokenizer = data_bundle
    num_classes = len(answer2idx)

    print(f"\n{'='*60}")
    print(f"EVALUATING: {exp_id}")
    print(f"  Model type:      {config['model_type']}")
    print(f"  Learning rate:   {config['learning_rate']}")
    print(f"  Layers unfrozen: {config['layers_unfrozen']}")
    print(f"  Embedding dim:   {config['embedding_dim']}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    epoch = checkpoint.get("epoch", "unknown")
    print(f"  Trained epochs:  {epoch}")

    # Build model
    try:
        model = build_model_from_config(config, num_classes, device)
        model.load_state_dict(checkpoint["model_state_dict"])
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return None

    # Create test data loader
    if config["model_type"] == "efficientnet":
        test_dataset = SLAKEDataset_LargeImage(
            test_data, imgs_dir, answer2idx, tokenizer, 64, "test"
        )
        batch_size = 12
    else:
        test_dataset = SLAKEDataset(
            test_data, imgs_dir, answer2idx, tokenizer, 64, "test"
        )
        batch_size = 16

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate
    results = evaluate_model(model, test_loader, test_dataset, device, idx2answer)
    results["experiment_id"] = exp_id
    results["model_type"] = config["model_type"]
    results["learning_rate"] = config["learning_rate"]
    results["layers_unfrozen"] = config["layers_unfrozen"]
    results["epochs_trained"] = epoch

    print(f"{'='*60}")

    # Free GPU
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


# ==============================================================================
# RESUME TRAINING
# ==============================================================================

def resume_training(exp_id, checkpoint_path, config, data_bundle, device,
                    additional_epochs=20):
    """Loads a checkpoint and continues training."""
    imgs_dir, train_data, val_data, test_data, answer2idx, idx2answer, tokenizer = data_bundle
    num_classes = len(answer2idx)

    print(f"\n{'='*60}")
    print(f"RESUMING TRAINING: {exp_id}")
    print(f"  Model type:        {config['model_type']}")
    print(f"  Additional epochs: {additional_epochs}")
    print(f"{'='*60}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    start_epoch = checkpoint.get("epoch", 0)
    print(f"  Resuming from epoch {start_epoch}")

    # Build model and load weights
    model = build_model_from_config(config, num_classes, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Create data loaders
    if config["model_type"] == "efficientnet":
        DatasetClass = SLAKEDataset_LargeImage
        batch_size = 12
    else:
        DatasetClass = SLAKEDataset
        batch_size = 16

    train_dataset = DatasetClass(train_data, imgs_dir, answer2idx, tokenizer, 64, "train")
    val_dataset = DatasetClass(val_data, imgs_dir, answer2idx, tokenizer, 64, "val")
    test_dataset = DatasetClass(test_data, imgs_dir, answer2idx, tokenizer, 64, "test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Fresh optimiser with the experiment's learning rate
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=1e-4
    )

    # Continue training
    total_epochs = start_epoch + additional_epochs
    train_start = time.time()
    train_model(
        model, train_loader, val_loader, criterion, optimiser,
        device, num_epochs=total_epochs, use_early_stopping=True,
        start_epoch=start_epoch, answer2idx=answer2idx,
        checkpoint_path=checkpoint_path
    )
    train_time = time.time() - train_start

    # Evaluate
    results = evaluate_model(model, test_loader, test_dataset, device, idx2answer)
    results["train_time"] = train_time
    results["experiment_id"] = exp_id
    results["model_type"] = config["model_type"]

    print(f"\nResume training complete. Additional time: {train_time:.1f}s")

    # Free GPU
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Rebuild experiment grid
    experiments = rebuild_experiment_grid()

    # Scan for checkpoints
    checkpoints = scan_checkpoints()
    if not checkpoints:
        print("No agentic checkpoints found in ./checkpoints/")
        print("Looking for files matching: ./checkpoints/agentic_exp_*.pth")
        return

    print(f"\nFound {len(checkpoints)} checkpoints:")
    for exp_id, path in checkpoints.items():
        config = experiments.get(exp_id, {})
        model_type = config.get("model_type", "unknown")
        print(f"  {exp_id}: {model_type} (lr={config.get('learning_rate')}, "
              f"unfrozen={config.get('layers_unfrozen')})")

    # Load data once (shared across all evaluations)
    print("\nLoading SLAKE dataset...")
    imgs_dir = download_slake_images("./data/slake")
    train_data, val_data, test_data = load_slake_annotations()
    verify_images(test_data, imgs_dir, "test")
    answer2idx, idx2answer = build_answer_vocabulary(train_data)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    data_bundle = (imgs_dir, train_data, val_data, test_data, answer2idx, idx2answer, tokenizer)

    # Parse command line
    mode = sys.argv[1] if len(sys.argv) > 1 else "evaluate"

    if mode == "evaluate" or mode == "--evaluate":
        # Evaluate all checkpoints
        all_results = []
        for exp_id, path in checkpoints.items():
            config = experiments.get(exp_id)
            if not config:
                print(f"  WARNING: {exp_id} not found in experiment grid, skipping")
                continue
            result = evaluate_checkpoint(exp_id, path, config, data_bundle, device)
            if result:
                all_results.append(result)
                sync_to_progress(result, config)

        # Print ranked results
        if all_results:
            all_results.sort(key=lambda r: r["balanced_accuracy"], reverse=True)
            print(f"\n{'='*80}")
            print("CHECKPOINT EVALUATION RESULTS (ranked by balanced accuracy)")
            print(f"{'='*80}")
            print(f"{'Rank':<5} {'ExpID':<8} {'Model':<18} {'LR':<10} {'Unfr':<6} "
                  f"{'Epochs':<8} {'Acc':<8} {'BalAcc':<8} {'F1':<8}")
            print("-" * 80)
            for rank, r in enumerate(all_results, 1):
                print(f"{rank:<5} {r['experiment_id']:<8} {r['model_type']:<18} "
                      f"{r['learning_rate']:<10.0e} {r['layers_unfrozen']:<6} "
                      f"{r['epochs_trained']:<8} {r['accuracy']:<8.4f} "
                      f"{r['balanced_accuracy']:<8.4f} {r['f1_weighted']:<8.4f}")
            print(f"{'='*80}")

            # Save results
            with open("./agentic_checkpoint_results.json", 'w') as f:
                json.dump(all_results, f, indent=2)
            print("Results saved to agentic_checkpoint_results.json")

    elif mode == "--resume":
        # Resume specific experiment
        if len(sys.argv) < 3:
            print("Usage: python evaluate_agentic_checkpoints.py --resume exp_14")
            return

        exp_id = sys.argv[2]
        epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 20

        if exp_id not in checkpoints:
            print(f"Checkpoint not found for {exp_id}")
            return

        config = experiments.get(exp_id)
        if not config:
            print(f"{exp_id} not found in experiment grid")
            return

        resume_result = resume_training(exp_id, checkpoints[exp_id], config, data_bundle, device, epochs)
        if resume_result:
            sync_to_progress(resume_result, config)

    elif mode == "--resume-best":
        # Evaluate all, then resume top 3
        all_results = []
        for exp_id, path in checkpoints.items():
            config = experiments.get(exp_id)
            if not config:
                continue
            result = evaluate_checkpoint(exp_id, path, config, data_bundle, device)
            if result:
                all_results.append(result)
                sync_to_progress(result, config)

        if not all_results:
            print("No valid checkpoints to resume")
            return

        # Sort by balanced accuracy and resume top 3
        all_results.sort(key=lambda r: r["balanced_accuracy"], reverse=True)
        top_n = min(3, len(all_results))

        print(f"\n{'='*60}")
        print(f"RESUMING TOP {top_n} CHECKPOINTS")
        print(f"{'='*60}")

        for r in all_results[:top_n]:
            exp_id = r["experiment_id"]
            config = experiments[exp_id]
            print(f"\n  Resuming {exp_id} ({r['model_type']}) — "
                  f"current acc: {r['accuracy']:.4f}, bal_acc: {r['balanced_accuracy']:.4f}")
            resume_result = resume_training(exp_id, checkpoints[exp_id], config, data_bundle, device, 20)
            if resume_result:
                sync_to_progress(resume_result, config)

    else:
        print("Usage:")
        print("  python evaluate_agentic_checkpoints.py                    # evaluate all")
        print("  python evaluate_agentic_checkpoints.py --resume exp_14    # resume specific")
        print("  python evaluate_agentic_checkpoints.py --resume exp_14 30 # resume with 30 extra epochs")
        print("  python evaluate_agentic_checkpoints.py --resume-best      # resume top 3")


if __name__ == "__main__":
    main()