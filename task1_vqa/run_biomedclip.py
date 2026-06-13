################################################################################
# run_biomedclip.py
#
# BiomedCLIP fine-tuned for VQA on SLAKE.
# Separate from other models because it uses open_clip instead of transformers.
#
# Requires: pip install open_clip_torch==2.23.0
################################################################################

import os
import sys
import time
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from open_clip import create_model_from_pretrained, get_tokenizer

from vqa_utils import (
    download_slake_images, load_slake_annotations, verify_images,
    build_answer_vocabulary, EarlyStopping, save_checkpoint, load_checkpoint,
    compute_mrr, compute_ece
)
from vqa_models import VQA_BiomedCLIP
from sklearn.metrics import balanced_accuracy_score, f1_score
import numpy as np


class SLAKEDataset_BiomedCLIP(Dataset):
    """Dataset using BiomedCLIP's own preprocessing and tokenizer."""
    def __init__(self, data, imgs_dir, answer2idx, preprocess, tokenizer,
                 context_length=256, split="train"):
        self.data = data
        self.imgs_dir = imgs_dir
        self.answer2idx = answer2idx
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.answer_types = data["answer_type"]

        unseen = sum(1 for ans in data["answer"] if ans not in answer2idx)
        if unseen > 0:
            print(f"  [{split}] {unseen}/{len(data)} unseen answers")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.imgs_dir, item["img_name"])
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        image = self.preprocess(image)
        text_tokens = self.tokenizer([item["question"]],
                                      context_length=self.context_length).squeeze(0)
        label = self.answer2idx.get(item["answer"], -1)
        return image, text_tokens, torch.tensor(label, dtype=torch.long)


def train_biomedclip(model, train_loader, val_loader, criterion, optimiser,
                     device, num_epochs=30, start_epoch=0, answer2idx=None, checkpoint_path=None):
    print("TRAINING BiomedCLIP...")
    if start_epoch > 0:
        print(f"  Resuming from epoch {start_epoch}")
    early_stopping = EarlyStopping(patience=5)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (images, text_tokens, labels) in enumerate(train_loader):
            images, text_tokens, labels = images.to(device), text_tokens.to(device), labels.to(device)
            valid_mask = labels >= 0
            if valid_mask.sum() == 0:
                continue

            outputs = model(images, text_tokens)
            loss = criterion(outputs[valid_mask], labels[valid_mask])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            predicted = outputs[valid_mask].argmax(dim=1)
            correct += (predicted == labels[valid_mask]).sum().item()
            total += valid_mask.sum().item()

            if batch_idx % 50 == 0:
                print(f'  Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

        avg_loss = running_loss / max(len(train_loader), 1)
        elapsed = time.time() - start_time
        print(f'  Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}, Acc: {correct/max(total,1):.4f}, Time: {elapsed:.1f}s')

        if answer2idx and checkpoint_path:
            save_checkpoint(model, optimiser, epoch+1, answer2idx, checkpoint_path)

        # Validation
        model.eval()
        val_loss = 0.0
        nb = 0
        with torch.no_grad():
            for images, text_tokens, labels in val_loader:
                images, text_tokens, labels = images.to(device), text_tokens.to(device), labels.to(device)
                valid_mask = labels >= 0
                if valid_mask.sum() == 0:
                    continue
                outputs = model(images, text_tokens)
                val_loss += criterion(outputs[valid_mask], labels[valid_mask]).item()
                nb += 1
        if early_stopping.check_early_stopping(val_loss / max(nb, 1)):
            break

    print("Training complete.\n")


def evaluate_biomedclip(model, test_loader, test_dataset, device):
    """Evaluation for BiomedCLIP."""
    print("EVALUATING BiomedCLIP...")
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, text_tokens, labels in test_loader:
            images, text_tokens = images.to(device), text_tokens.to(device)
            outputs = model(images, text_tokens)
            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(probs.argmax(dim=1).cpu().numpy().tolist())
            all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    total = len(all_labels)
    correct = sum(1 for l, p in zip(all_labels, all_preds) if l >= 0 and l == p)
    accuracy = correct / total
    valid_l = [l for l in all_labels if l >= 0]
    valid_p = [p for l, p in zip(all_labels, all_preds) if l >= 0]
    bal_acc = balanced_accuracy_score(valid_l, valid_p) if valid_l else 0
    f1 = f1_score(valid_l, valid_p, average="weighted", zero_division=0) if valid_l else 0
    mrr = compute_mrr(all_probs, all_labels)
    ece = compute_ece(all_probs, all_labels)

    print(f"  Accuracy: {accuracy:.4f}, BalAcc: {bal_acc:.4f}, F1: {f1:.4f}, MRR: {mrr:.4f}, ECE: {ece:.4f}")
    return {"accuracy": accuracy, "balanced_accuracy": bal_acc, "f1_weighted": f1, "mrr": mrr, "ece": ece}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 16
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 200
    CHECKPOINT_PATH = "./checkpoints/biomedclip_checkpoint.pth"

    # Load BiomedCLIP
    print("Loading BiomedCLIP...")
    biomedclip_model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = get_tokenizer(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    # Load data
    imgs_dir = download_slake_images("./data/slake")
    train_data, val_data, test_data = load_slake_annotations()
    answer2idx, idx2answer = build_answer_vocabulary(train_data)

    # Create datasets
    train_ds = SLAKEDataset_BiomedCLIP(train_data, imgs_dir, answer2idx, preprocess, tokenizer, split="train")
    val_ds = SLAKEDataset_BiomedCLIP(val_data, imgs_dir, answer2idx, preprocess, tokenizer, split="val")
    test_ds = SLAKEDataset_BiomedCLIP(test_data, imgs_dir, answer2idx, preprocess, tokenizer, split="test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    model = VQA_BiomedCLIP(len(answer2idx), biomedclip_model, freeze_encoders=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=LEARNING_RATE, weight_decay=1e-4)
    #
    
    # Resume from checkpoint
    RESUME_TRAINING = True
    start_epoch = 0
    if RESUME_TRAINING:
        start_epoch = load_checkpoint(model, optimiser, CHECKPOINT_PATH)
    
    # Train & evaluate
    train_start = time.time()
    train_biomedclip(model, train_loader, val_loader, criterion, optimiser,
                     device, NUM_EPOCHS, start_epoch, answer2idx, CHECKPOINT_PATH)
    print(f"Train time: {time.time()-train_start:.1f}s")

    evaluate_biomedclip(model, test_loader, test_ds, device)
