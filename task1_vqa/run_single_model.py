################################################################################
# run_single_model.py
#
# Run a single VQA model on SLAKE for manual testing.
# Usage: python run_single_model.py
#
# Change MODEL_TYPE and hyperparameters below to test different configurations.
################################################################################

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
# CONFIGURATION — change these to test different models
# ==============================================================================

# Available models: "vit_concat", "vit_hadamard", "resnet_concat",
#                   "resnet_hadamard", "efficientnet", "cross_attention"
MODEL_TYPE = "vit_hadamard"

BATCH_SIZE = 16
LEARNING_RATE = 3e-5
EMBEDDING_DIM = 256
MAX_TEXT_LENGTH = 64
NUM_EPOCHS = 30
LAYERS_UNFROZEN = 2
CHECKPOINT_PATH = f"./checkpoints/{MODEL_TYPE}_checkpoint.pth"
RESUME_TRAINING = False


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model: {MODEL_TYPE}")

    # Load data
    imgs_dir = download_slake_images("./data/slake")
    train_data, val_data, test_data = load_slake_annotations()
    verify_images(train_data, imgs_dir, "train")
    answer2idx, idx2answer = build_answer_vocabulary(train_data)
    num_classes = len(answer2idx)

    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Dataset (use larger images for EfficientNet)
    if MODEL_TYPE == "efficientnet":
        DatasetClass = SLAKEDataset_LargeImage
        BATCH_SIZE = 12
    else:
        DatasetClass = SLAKEDataset

    train_dataset = DatasetClass(train_data, imgs_dir, answer2idx, tokenizer, MAX_TEXT_LENGTH, "train")
    val_dataset = DatasetClass(val_data, imgs_dir, answer2idx, tokenizer, MAX_TEXT_LENGTH, "val")
    test_dataset = DatasetClass(test_data, imgs_dir, answer2idx, tokenizer, MAX_TEXT_LENGTH, "test")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    ModelClass = MODEL_REGISTRY[MODEL_TYPE]

    if "resnet" in MODEL_TYPE:
        layer_key = "num_resnet_layers_to_finetune"
    elif "efficientnet" in MODEL_TYPE:
        layer_key = "num_layers_to_finetune"
    else:
        layer_key = "num_vit_blocks_to_finetune"

    model = ModelClass(
        num_classes=num_classes,
        embedding_dim=EMBEDDING_DIM,
        pretrained=True,
        **{layer_key: LAYERS_UNFROZEN},
        num_text_layers_to_finetune=1
    ).to(device)

    print(f"\nTrainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}\n")

    # Train
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=1e-4
    )

    train_start = time.time()
    train_model(model, train_loader, val_loader, criterion, optimiser,
                device, num_epochs=NUM_EPOCHS, use_early_stopping=True,
                answer2idx=answer2idx, checkpoint_path=CHECKPOINT_PATH)
    train_time = time.time() - train_start

    # Evaluate
    results = evaluate_model(model, test_loader, test_dataset, device, idx2answer)
    results["train_time"] = train_time
    print(f"\nTrain time: {train_time:.1f}s")
