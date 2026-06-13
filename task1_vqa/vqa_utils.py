################################################################################
# vqa_utils.py
#
# Shared utilities for all VQA models on the SLAKE dataset.
# Contains: data loading, dataset classes, training loop, evaluation metrics.
#
# Adapted from workshop materials:
# - Dataset pattern from ITM_CNN_CLIP_Classifier.py (ITM_Dataset)
# - Training loop from ITM_CNN_CLIP_Classifier.py (train_model)
# - Evaluation pattern from ITM_CNN_CLIP_Classifier.py (evaluate_model)
# - Early stopping from ITM_CNN_CLIP_Classifier.py (EarlyStopping)
#
# Dataset: SLAKE (BoKelvin/SLAKE on HuggingFace)
################################################################################

import os
import sys
import time
import zipfile
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import balanced_accuracy_score, f1_score
from datasets import load_dataset
from huggingface_hub import hf_hub_download


# ==============================================================================
# DATA DOWNLOAD & LOADING
# ==============================================================================

def download_slake_images(data_dir="./data/slake"):
    """Downloads and extracts SLAKE images from BoKelvin/SLAKE on HuggingFace."""
    os.makedirs(data_dir, exist_ok=True)
    imgs_dir = os.path.join(data_dir, "imgs")

    if not os.path.exists(imgs_dir):
        print("Downloading SLAKE images...")
        zip_path = hf_hub_download(
            repo_id="BoKelvin/SLAKE", filename="imgs.zip",
            repo_type="dataset", local_dir=data_dir
        )
        print("Extracting images...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(data_dir)
        print(f"Images extracted to {imgs_dir}")
    else:
        print(f"Images already exist at {imgs_dir}")
    return imgs_dir


def load_slake_annotations():
    """Loads SLAKE annotations via HuggingFace datasets package, filtered to English."""
    print("Loading SLAKE annotations...")
    dataset = load_dataset("BoKelvin/SLAKE")

    print(f"  Splits: {list(dataset.keys())}")
    print(f"  Columns: {dataset['train'].column_names}")

    train = dataset["train"].filter(lambda x: x["q_lang"] == "en")
    val = dataset["validation"].filter(lambda x: x["q_lang"] == "en")
    test = dataset["test"].filter(lambda x: x["q_lang"] == "en")

    for name, split in [("train", train), ("val", val), ("test", test)]:
        answer_types = {}
        for item in split:
            atype = item.get("answer_type", "UNKNOWN")
            answer_types[atype] = answer_types.get(atype, 0) + 1
        print(f"  {name}: {len(split)} samples, types: {answer_types}")

    return train, val, test


def verify_images(split_data, imgs_dir, split_name=""):
    """Checks that all referenced images exist on disk."""
    missing = 0
    for item in split_data:
        if not os.path.exists(os.path.join(imgs_dir, item["img_name"])):
            missing += 1
    if missing > 0:
        print(f"  WARNING [{split_name}]: {missing}/{len(split_data)} images not found")
    else:
        print(f"  [{split_name}]: All {len(split_data)} images verified")
    return missing


def build_answer_vocabulary(train_data):
    """Builds answer-to-index mapping from training data."""
    all_answers = sorted(set(train_data["answer"]))
    answer2idx = {ans: idx for idx, ans in enumerate(all_answers)}
    idx2answer = {idx: ans for ans, idx in answer2idx.items()}
    print(f"Answer vocabulary: {len(answer2idx)} unique answers")
    return answer2idx, idx2answer


# ==============================================================================
# DATASET CLASSES
# Adapted from ITM_Dataset in ITM_CNN_CLIP_Classifier.py
# ==============================================================================

class SLAKEDataset(Dataset):
    """Standard SLAKE dataset for 224×224 models (ViT, ResNet)."""
    def __init__(self, data, imgs_dir, answer2idx, tokenizer, max_length=64, split="train"):
        self.data = data
        self.imgs_dir = imgs_dir
        self.answer2idx = answer2idx
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        answers = data["answer"]
        unseen = sum(1 for ans in answers if ans not in self.answer2idx)
        if unseen > 0:
            print(f"  [{split}] {unseen}/{len(self.data)} samples have unseen answers")

        self.answer_types = data["answer_type"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.imgs_dir, item["img_name"])
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        image = self.transform(image)

        tokens = self.tokenizer(item["question"], padding="max_length",
                                truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        label = self.answer2idx.get(item["answer"], -1)

        return image, input_ids, attention_mask, torch.tensor(label, dtype=torch.long)


class SLAKEDataset_LargeImage(Dataset):
    """SLAKE dataset for 380×380 models (EfficientNet)."""
    def __init__(self, data, imgs_dir, answer2idx, tokenizer, max_length=64, split="train"):
        self.data = data
        self.imgs_dir = imgs_dir
        self.answer2idx = answer2idx
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

        self.transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        answers = data["answer"]
        unseen = sum(1 for ans in answers if ans not in self.answer2idx)
        if unseen > 0:
            print(f"  [{split}] {unseen}/{len(self.data)} samples have unseen answers")

        self.answer_types = data["answer_type"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.imgs_dir, item["img_name"])
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            image = Image.new("RGB", (380, 380), (0, 0, 0))
        image = self.transform(image)

        tokens = self.tokenizer(item["question"], padding="max_length",
                                truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        label = self.answer2idx.get(item["answer"], -1)

        return image, input_ids, attention_mask, torch.tensor(label, dtype=torch.long)


# ==============================================================================
# EARLY STOPPING (from ITM_CNN_CLIP_Classifier.py)
# ==============================================================================

class EarlyStopping:
    """Prevents overfitting. Reused from ITM_CNN_CLIP_Classifier.py."""
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float('inf')
        self.patience_count = 0

    def check_early_stopping(self, val_loss):
        if val_loss < self.best_loss - 1e-3:
            self.best_loss = val_loss
            self.patience_count = 0
            improved = True
            stop = False
        else:
            self.patience_count += 1
            improved = False
            stop = self.patience_count >= self.patience
        print("  Val loss=%.5f, improved=%s, patience=%s/%s" %
              (val_loss, improved, self.patience_count, self.patience))
        if stop:
            print("  Stopping early!")
        return stop


# ==============================================================================
# CHECKPOINTS
# ==============================================================================

def save_checkpoint(model, optimiser, epoch, answer2idx, path):
    """Saves model + optimiser state for resuming later."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "epoch": epoch,
        "answer2idx": answer2idx,
        "num_classes": len(answer2idx)
    }, path)
    print(f"  Checkpoint saved to {path} (epoch {epoch})")


def load_checkpoint(model, optimiser, path):
    """Loads checkpoint to resume training. Returns start epoch."""
    if not os.path.exists(path):
        print(f"  No checkpoint found at {path} — starting from scratch")
        return 0
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
    start_epoch = checkpoint["epoch"]
    print(f"  Checkpoint loaded from {path} (resuming from epoch {start_epoch})")
    return start_epoch


# ==============================================================================
# TRAINING (adapted from train_model in ITM_CNN_CLIP_Classifier.py)
# ==============================================================================

def train_model(model, train_loader, val_loader, criterion, optimiser,
                device, num_epochs=30, use_early_stopping=True,
                start_epoch=0, answer2idx=None,
                checkpoint_path="./checkpoints/checkpoint.pth"):
    """Training loop with early stopping and checkpoint saving."""
    print("TRAINING model...")
    if start_epoch > 0:
        print(f"  Resuming from epoch {start_epoch}")
    early_stopping = EarlyStopping(patience=5) if use_early_stopping else None

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (images, input_ids, attention_mask, labels) in enumerate(train_loader):
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            valid_mask = labels >= 0
            if valid_mask.sum() == 0:
                continue

            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs[valid_mask], labels[valid_mask])

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            predicted = outputs[valid_mask].argmax(dim=1)
            correct += (predicted == labels[valid_mask]).sum().item()
            total += valid_mask.sum().item()

            if batch_idx % 100 == 0:
                print('  Epoch [%s/%s], Batch [%s/%s], Loss: %.4f' %
                      (epoch+1, num_epochs, batch_idx, len(train_loader), loss.item()))

        avg_loss = running_loss / max(len(train_loader), 1)
        train_acc = correct / max(total, 1)
        elapsed = time.time() - start_time
        print('  Epoch [%s/%s] Avg Loss: %.4f, Train Acc: %.4f, Time: %.2fs' %
              (epoch+1, num_epochs, avg_loss, train_acc, elapsed))

        if answer2idx is not None and checkpoint_path:
            save_checkpoint(model, optimiser, epoch+1, answer2idx, checkpoint_path)

        if use_early_stopping and val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)
            if early_stopping.check_early_stopping(val_loss):
                break

    print("Training complete.\n")


def validate(model, val_loader, criterion, device):
    """Validation pass for early stopping."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for images, input_ids, attention_mask, labels in val_loader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            valid_mask = labels >= 0
            if valid_mask.sum() == 0:
                continue
            outputs = model(images, input_ids, attention_mask)
            total_loss += criterion(outputs[valid_mask], labels[valid_mask]).item()
            num_batches += 1
    return total_loss / max(num_batches, 1)


# ==============================================================================
# EVALUATION METRICS
# Extends balanced accuracy from ITM_CNN_CLIP_Classifier.py with
# F1, MRR, ECE as required by the assessment brief.
# ==============================================================================

def compute_mrr(all_probs, all_labels):
    """Mean Reciprocal Rank."""
    reciprocal_ranks = []
    for probs, label in zip(all_probs, all_labels):
        if label < 0:
            reciprocal_ranks.append(0.0)
            continue
        sorted_indices = probs.argsort(descending=True)
        rank = (sorted_indices == label).nonzero(as_tuple=True)[0].item() + 1
        reciprocal_ranks.append(1.0 / rank)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def compute_ece(all_probs, all_labels, n_bins=10):
    """Expected Calibration Error."""
    valid_mask = torch.tensor(all_labels) >= 0
    if valid_mask.sum() == 0:
        return 0.0
    valid_probs = all_probs[valid_mask]
    valid_labels = torch.tensor(all_labels)[valid_mask]

    confidences = valid_probs.max(dim=1).values
    predictions = valid_probs.argmax(dim=1)
    correct = (predictions == valid_labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = correct[mask].mean()
            ece += mask.float().mean() * abs(avg_acc - avg_conf)
    return ece.item()


def evaluate_model(model, test_loader, test_dataset, device, idx2answer=None):
    """Full evaluation with all metrics required by assessment brief."""
    print("EVALUATING model...")
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []
    start_time = time.time()

    with torch.no_grad():
        for images, input_ids, attention_mask, labels in test_loader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(images, input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            all_labels.extend(labels.numpy().tolist())
            all_predictions.extend(preds.cpu().numpy().tolist())
            all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    elapsed = time.time() - start_time
    total_samples = len(all_labels)
    unseen_count = sum(1 for l in all_labels if l < 0)

    valid_labels = [l for l in all_labels if l >= 0]
    valid_preds = [p for l, p in zip(all_labels, all_predictions) if l >= 0]

    correct_count = sum(1 for l, p in zip(all_labels, all_predictions) if l >= 0 and l == p)
    accuracy = correct_count / total_samples
    error_rate = 1.0 - accuracy

    if len(valid_labels) > 0:
        bal_acc = balanced_accuracy_score(valid_labels, valid_preds)
        f1 = f1_score(valid_labels, valid_preds, average="weighted", zero_division=0)
    else:
        bal_acc = 0.0
        f1 = 0.0

    mrr = compute_mrr(all_probs, all_labels)
    ece = compute_ece(all_probs, all_labels)

    # Per-category metrics
    answer_types = test_dataset.answer_types
    categories = sorted(set(answer_types))
    per_category = {}
    for cat in categories:
        cat_indices = [i for i, t in enumerate(answer_types) if t == cat]
        cat_labels = [all_labels[i] for i in cat_indices]
        cat_preds = [all_predictions[i] for i in cat_indices]
        cat_valid_l = [l for l in cat_labels if l >= 0]
        cat_valid_p = [cat_preds[j] for j, l in enumerate(cat_labels) if l >= 0]
        cat_correct = sum(1 for l, p in zip(cat_labels, cat_preds) if l >= 0 and l == p)
        cat_total = len(cat_labels)
        cat_acc = cat_correct / max(cat_total, 1)
        if len(cat_valid_l) > 1:
            cat_bal_acc = balanced_accuracy_score(cat_valid_l, cat_valid_p)
        else:
            cat_bal_acc = cat_acc
        per_category[cat] = {
            "total": cat_total, "accuracy": cat_acc,
            "error_rate": 1.0 - cat_acc, "balanced_accuracy": cat_bal_acc
        }

    # Print results
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total samples:         {total_samples}")
    print(f"  Unseen answers:        {unseen_count}")
    print(f"  Correct predictions:   {correct_count}/{total_samples}")
    print(f"  Accuracy:              {accuracy:.4f}")
    print(f"  Error Rate:            {error_rate:.4f}")
    print(f"  Balanced Accuracy:     {bal_acc:.4f}")
    print(f"  F1 Score (weighted):   {f1:.4f}")
    print(f"  Mean Reciprocal Rank:  {mrr:.4f}")
    print(f"  Expected Cal. Error:   {ece:.4f}")
    print(f"  Test Time:             {elapsed:.2f}s")
    for cat, m in per_category.items():
        print(f"  [{cat}] n={m['total']}, Acc={m['accuracy']:.4f}, BalAcc={m['balanced_accuracy']:.4f}")
    print("=" * 60)

    return {
        "accuracy": accuracy, "error_rate": error_rate,
        "balanced_accuracy": bal_acc, "f1_weighted": f1,
        "mrr": mrr, "ece": ece, "test_time": elapsed,
        "unseen_answers": unseen_count, "total_samples": total_samples,
        "per_category": per_category
    }
