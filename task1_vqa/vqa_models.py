################################################################################
# vqa_models.py
#
# All VQA model architectures for the SLAKE dataset.
# Each model uses a different combination of vision encoder, text encoder,
# and fusion strategy.
#
# Models:
# 1. VQA_ViT_Concat:        ViT + DistilBERT + Concatenation
# 2. VQA_ViT_Hadamard:      ViT + DistilBERT + Hadamard product
# 3. VQA_ResNet_Concat:      ResNet50 + DistilBERT + Concatenation
# 4. VQA_ResNet_Hadamard:    ResNet50 + DistilBERT + Hadamard product
# 5. VQA_EfficientNet:       EfficientNet-B4 + DistilBERT + Hadamard product
# 6. VQA_CrossAttention:     ViT + DistilBERT + Cross-attention fusion
# 7. VQA_BiomedCLIP:         BiomedCLIP (unified biomedical VLM) + Concat+Similarity
#
# Adapted from CMP9137M workshop materials:
# - Vision encoder patterns from Flowers_Classifier-ViT.py and ITM_CNN_CLIP_Classifier.py
# - Text encoder pattern from QAL_Classifier-DistilBERT.py
# - Fusion patterns from ITM_Model and ITM_Model_CLIP in ITM_CNN_CLIP_Classifier.py
################################################################################

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_32
from torchvision import models
from transformers import DistilBertModel


# ==============================================================================
# MODEL 1: ViT + DistilBERT + CONCATENATION
# Adapted from ITM_Model in ITM_CNN_CLIP_Classifier.py
# ==============================================================================

class VQA_ViT_Concat(nn.Module):
    """
    Late fusion via concatenation: [img_features; text_features] → classifier
    Classifier input: embedding_dim * 2
    """
    def __init__(self, num_classes, embedding_dim=256, pretrained=True,
                 num_vit_blocks_to_finetune=2, num_text_layers_to_finetune=1):
        super().__init__()
        # Vision encoder
        if pretrained:
            self.vision_model = vit_b_32(weights="IMAGENET1K_V1")
            for param in self.vision_model.parameters():
                param.requires_grad = False
            if num_vit_blocks_to_finetune > 0:
                for block in self.vision_model.encoder.layers[-num_vit_blocks_to_finetune:]:
                    for param in block.parameters():
                        param.requires_grad = True
            for param in self.vision_model.heads.parameters():
                param.requires_grad = True
        else:
            self.vision_model = vit_b_32(weights=None)
        self.vision_model.heads = nn.Linear(self.vision_model.heads[0].in_features, embedding_dim)

        # Text encoder
        self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        for param in self.text_model.parameters():
            param.requires_grad = False
        if num_text_layers_to_finetune > 0:
            for layer in self.text_model.transformer.layer[-num_text_layers_to_finetune:]:
                for p in layer.parameters():
                    p.requires_grad = True
        self.text_projection = nn.Linear(768, embedding_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        img_features = self.vision_model(image)
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.text_projection(text_output.last_hidden_state[:, 0])
        combined = torch.cat([img_features, text_features], dim=1)
        return self.classifier(combined)


# ==============================================================================
# MODEL 2: ViT + DistilBERT + HADAMARD PRODUCT
# Fusion via element-wise multiplication: img ⊙ text → classifier
# ==============================================================================

class VQA_ViT_Hadamard(nn.Module):
    """
    Hadamard fusion: img_features * text_features → classifier
    Classifier input: embedding_dim (not *2, because multiplication preserves size)
    """
    def __init__(self, num_classes, embedding_dim=256, pretrained=True,
                 num_vit_blocks_to_finetune=2, num_text_layers_to_finetune=1):
        super().__init__()
        if pretrained:
            self.vision_model = vit_b_32(weights="IMAGENET1K_V1")
            for param in self.vision_model.parameters():
                param.requires_grad = False
            if num_vit_blocks_to_finetune > 0:
                for block in self.vision_model.encoder.layers[-num_vit_blocks_to_finetune:]:
                    for param in block.parameters():
                        param.requires_grad = True
            for param in self.vision_model.heads.parameters():
                param.requires_grad = True
        else:
            self.vision_model = vit_b_32(weights=None)
        self.vision_model.heads = nn.Linear(self.vision_model.heads[0].in_features, embedding_dim)

        self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        for param in self.text_model.parameters():
            param.requires_grad = False
        if num_text_layers_to_finetune > 0:
            for layer in self.text_model.transformer.layer[-num_text_layers_to_finetune:]:
                for p in layer.parameters():
                    p.requires_grad = True
        self.text_projection = nn.Linear(768, embedding_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        img_features = self.vision_model(image)
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.text_projection(text_output.last_hidden_state[:, 0])
        combined = img_features * text_features
        return self.classifier(combined)


# ==============================================================================
# MODEL 3: ResNet50 + DistilBERT + CONCATENATION
# Vision encoder from retrieve_vision_model() in ITM_CNN_CLIP_Classifier.py
# ==============================================================================

class VQA_ResNet_Concat(nn.Module):
    """ResNet50 (CNN) + DistilBERT + Concatenation fusion."""
    def __init__(self, num_classes, embedding_dim=256, pretrained=True,
                 num_resnet_layers_to_finetune=2, num_text_layers_to_finetune=1):
        super().__init__()
        if pretrained:
            self.vision_model = models.resnet50(weights="IMAGENET1K_V2")
            for param in self.vision_model.parameters():
                param.requires_grad = False
            if num_resnet_layers_to_finetune > 0:
                for child in list(self.vision_model.children())[-num_resnet_layers_to_finetune:]:
                    for param in child.parameters():
                        param.requires_grad = True
        else:
            self.vision_model = models.resnet50(weights=None)
        self.vision_model.fc = nn.Linear(self.vision_model.fc.in_features, embedding_dim)

        self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        for param in self.text_model.parameters():
            param.requires_grad = False
        if num_text_layers_to_finetune > 0:
            for layer in self.text_model.transformer.layer[-num_text_layers_to_finetune:]:
                for p in layer.parameters():
                    p.requires_grad = True
        self.text_projection = nn.Linear(768, embedding_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        img_features = self.vision_model(image)
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.text_projection(text_output.last_hidden_state[:, 0])
        combined = torch.cat([img_features, text_features], dim=1)
        return self.classifier(combined)


# ==============================================================================
# MODEL 4: ResNet50 + DistilBERT + HADAMARD
# ==============================================================================

class VQA_ResNet_Hadamard(nn.Module):
    """ResNet50 (CNN) + DistilBERT + Hadamard fusion."""
    def __init__(self, num_classes, embedding_dim=256, pretrained=True,
                 num_resnet_layers_to_finetune=2, num_text_layers_to_finetune=1):
        super().__init__()
        if pretrained:
            self.vision_model = models.resnet50(weights="IMAGENET1K_V2")
            for param in self.vision_model.parameters():
                param.requires_grad = False
            if num_resnet_layers_to_finetune > 0:
                for child in list(self.vision_model.children())[-num_resnet_layers_to_finetune:]:
                    for param in child.parameters():
                        param.requires_grad = True
        else:
            self.vision_model = models.resnet50(weights=None)
        self.vision_model.fc = nn.Linear(self.vision_model.fc.in_features, embedding_dim)

        self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        for param in self.text_model.parameters():
            param.requires_grad = False
        if num_text_layers_to_finetune > 0:
            for layer in self.text_model.transformer.layer[-num_text_layers_to_finetune:]:
                for p in layer.parameters():
                    p.requires_grad = True
        self.text_projection = nn.Linear(768, embedding_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        img_features = self.vision_model(image)
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.text_projection(text_output.last_hidden_state[:, 0])
        combined = img_features * text_features
        return self.classifier(combined)


# ==============================================================================
# MODEL 5: EfficientNet-B4 + DistilBERT + HADAMARD
# Uses 380×380 input (requires SLAKEDataset_LargeImage)
# ==============================================================================

class VQA_EfficientNet(nn.Module):
    """EfficientNet-B4 (380×380 input) + DistilBERT + Hadamard fusion."""
    def __init__(self, num_classes, embedding_dim=256, pretrained=True,
                 num_layers_to_finetune=4, num_text_layers_to_finetune=1):
        super().__init__()
        if pretrained:
            self.vision_model = models.efficientnet_b4(weights="IMAGENET1K_V1")
            for param in self.vision_model.parameters():
                param.requires_grad = False
            if num_layers_to_finetune > 0:
                for child in list(self.vision_model.features.children())[-num_layers_to_finetune:]:
                    for param in child.parameters():
                        param.requires_grad = True
        else:
            self.vision_model = models.efficientnet_b4(weights=None)

        eff_features = self.vision_model.classifier[1].in_features
        self.vision_model.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(eff_features, embedding_dim)
        )

        self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        for param in self.text_model.parameters():
            param.requires_grad = False
        if num_text_layers_to_finetune > 0:
            for layer in self.text_model.transformer.layer[-num_text_layers_to_finetune:]:
                for p in layer.parameters():
                    p.requires_grad = True
        self.text_projection = nn.Linear(768, embedding_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        img_features = self.vision_model(image)
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.text_projection(text_output.last_hidden_state[:, 0])
        combined = img_features * text_features
        return self.classifier(combined)


# ==============================================================================
# MODEL 6: ViT + DistilBERT + CROSS-ATTENTION
# Image features attend to question tokens for question-aware visual features
# ==============================================================================

class CrossAttentionModule(nn.Module):
    """Multi-head cross-attention: image queries attend to text keys/values."""
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_out = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, key_mask=None):
        B = query.size(0)
        Q = self.W_q(query).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if key_mask is not None:
            attn = attn.masked_fill(key_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.head_dim)
        return self.W_out(out).squeeze(1)


class VQA_CrossAttention(nn.Module):
    """ViT + DistilBERT + Cross-attention fusion."""
    def __init__(self, num_classes, embedding_dim=256, pretrained=True,
                 num_vit_blocks_to_finetune=2, num_text_layers_to_finetune=1,
                 num_attention_heads=4):
        super().__init__()
        if pretrained:
            self.vision_model = vit_b_32(weights="IMAGENET1K_V1")
            for param in self.vision_model.parameters():
                param.requires_grad = False
            if num_vit_blocks_to_finetune > 0:
                for block in self.vision_model.encoder.layers[-num_vit_blocks_to_finetune:]:
                    for param in block.parameters():
                        param.requires_grad = True
            for param in self.vision_model.heads.parameters():
                param.requires_grad = True
        else:
            self.vision_model = vit_b_32(weights=None)
        self.vision_model.heads = nn.Linear(self.vision_model.heads[0].in_features, embedding_dim)

        self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        for param in self.text_model.parameters():
            param.requires_grad = False
        if num_text_layers_to_finetune > 0:
            for layer in self.text_model.transformer.layer[-num_text_layers_to_finetune:]:
                for p in layer.parameters():
                    p.requires_grad = True
        self.text_projection = nn.Linear(768, embedding_dim)

        self.cross_attention = CrossAttentionModule(embedding_dim, num_attention_heads)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        img_features = self.vision_model(image)
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_tokens = self.text_projection(text_output.last_hidden_state)
        img_query = img_features.unsqueeze(1)
        cross_attended = self.cross_attention(img_query, text_tokens, text_tokens, attention_mask)
        cross_attended = self.layer_norm(cross_attended + img_features)
        combined = torch.cat([img_features, cross_attended], dim=1)
        return self.classifier(combined)


# ==============================================================================
# MODEL 7: BiomedCLIP (unified biomedical VLM) + Concat+Similarity
# Requires: pip install open_clip_torch==2.23.0
# ==============================================================================

class VQA_BiomedCLIP(nn.Module):
    """BiomedCLIP with Concat+Similarity fusion. Requires separate dataset class."""
    def __init__(self, num_classes, biomedclip_model, freeze_encoders=True):
        super().__init__()
        self.biomedclip = biomedclip_model
        if freeze_encoders:
            for param in self.biomedclip.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 + 1, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, text_tokens):
        image_features, text_features, logit_scale = self.biomedclip(image, text_tokens)
        similarity = (image_features * text_features).sum(dim=-1, keepdim=True)
        combined = torch.cat([image_features, text_features, similarity], dim=1)
        return self.classifier(combined)


# ==============================================================================
# MODEL REGISTRY — maps model names to classes for the agentic framework
# ==============================================================================

MODEL_REGISTRY = {
    "vit_concat": VQA_ViT_Concat,
    "vit_hadamard": VQA_ViT_Hadamard,
    "resnet_concat": VQA_ResNet_Concat,
    "resnet_hadamard": VQA_ResNet_Hadamard,
    "efficientnet": VQA_EfficientNet,
    "cross_attention": VQA_CrossAttention,
}
