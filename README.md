# Encoding Strategies for Multimodal Medical VQA and Continuous Robot Control

Two machine-learning systems built for the _Advanced Machine Learning_ module, exploring a single question across two very
different domains: **does the encoding strategy matter more than architectural
complexity?**

- **Task 1 — Medical Visual Question Answering (VQA)** on the SLAKE dataset:
  10+ multimodal classifiers comparing vision encoders, text encoders and fusion
  strategies, with a [LangGraph](https://github.com/langchain-ai/langgraph) agentic
  framework driving the hyperparameter search.
- **Task 2 — Continuous robot control** on `Walker2d-v5` (MuJoCo): PPO, SAC and
  TD3 compared across MLP, 1D-CNN and Transformer state representations over
  multiple seeds.

Across both tasks the finding is the same: **inductive biases must match the data
structure — more complex architectures are not inherently better.**

> Full write-up: [`report/ML_Report.pdf`](report/ML_Report.pdf)

---

## Task 1 — Medical VQA on SLAKE

Given a medical image (CT / MRI / X-ray) and a natural-language question, predict
the answer from 228 unique training answers. SLAKE has ~4,900 English training
pairs and 1,061 test pairs, with heavy class imbalance ("Yes"/"No" make up 34% of
samples; 60 answer classes appear only once).

All models use a **late-fusion** architecture: a vision encoder and a text encoder
produce features that a fusion module combines before classification. Three axes
were varied:

| Axis               | Options explored                                                    |
| ------------------ | ------------------------------------------------------------------- |
| **Vision encoder** | ViT-B/32, ResNet50, EfficientNet-B4, BiomedCLIP                     |
| **Text encoder**   | DistilBERT, BiomedBERT (PubMedBERT)                                 |
| **Fusion**         | Concatenation, Hadamard product, Cross-attention, Concat+Similarity |

Training: AdamW (weight decay 1e-4), CrossEntropyLoss, early stopping (patience 5),
and selective unfreezing of 2–4 vision blocks to balance adaptation vs. overfitting.

### Results (held-out test set)

| Model                         | Acc       | Bal. Acc  | F1        | MRR       | ECE       |
| ----------------------------- | --------- | --------- | --------- | --------- | --------- |
| **BiomedCLIP + Concat+Sim**   | **80.77** | **57.50** | **80.72** | **88.12** | **0.038** |
| ResNet + Hadamard _(agentic)_ | 80.02     | 54.02     | 78.99     | 87.39     | 0.072     |
| ViT + Hadamard                | 79.83     | 54.79     | 79.07     | 87.24     | 0.059     |
| EfficientNet + Hadamard       | 79.55     | 52.31     | 79.22     | 87.23     | 0.079     |
| ViT + BiomedBERT + Hadamard   | 79.08     | 53.83     | 78.72     | 86.85     | 0.086     |
| ViT + Concat                  | 78.98     | 52.85     | 78.52     | 86.74     | 0.059     |
| ResNet + Concat _(agentic)_   | 78.89     | 55.66     | 78.70     | 86.69     | 0.062     |
| ViT + Cross-attention         | 75.12     | 46.65     | 73.43     | 83.79     | 0.064     |
| ViT + CLIP contrastive loss   | 63.52     | 17.57     | 57.45     | 73.08     | 0.100     |
| BiomedCLIP + Hadamard         | 36.00     | 4.36      | 25.52     | 48.81     | 0.134     |

All values in %. _(agentic)_ = configuration discovered by the LangGraph search.

### Key findings

- **Hadamard fusion beats concatenation** in every encoder pairing — the
  element-wise product acts as mutual gating (a feature must fire in both
  modalities to survive).
- **Biomedical pretraining wins.** BiomedCLIP gave the best balanced accuracy
  (57.50%), confirming domain-pretrained features beat ImageNet ones.
- **Failure modes are instructive.** CLIP contrastive loss collapsed (63.5%)
  because SLAKE reuses images across questions, producing contradictory gradients;
  BiomedCLIP + Hadamard collapsed to 36% because its L2-normalised features yield
  near-zero element-wise products.

### Agentic AI framework

[`agentic_vqa.py`](task1_vqa/agentic_vqa.py) adapts the workshop's LangGraph state
machine — `perceive → select action → run experiment → update history` — but
replaces _simulated_ experiments with **real training and evaluation**. It searched
24 configurations (6 model types × 2 learning rates × 2 unfreezing depths), added
GPU memory cleanup between runs, and supports resume-from-checkpoint. It confirmed
learning rate as the dominant hyperparameter (3e-5 > 1e-5 everywhere) and surfaced
a competitive ResNet+Hadamard config absent from the manual search.

---

## Task 2 — Robot Locomotion on Walker2d-v5

A bipedal robot learns forward locomotion in MuJoCo. Observations are 17-D
(joint angles, velocities, torso orientation); actions are 6-D continuous torques;
reward = forward velocity + healthy bonus − control cost over up to 1,000 steps.

Three algorithms (via [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3))
were compared across MLP, 1D-CNN and Transformer feature extractors, each over
multiple seeds to reduce variance.

### Results

| Algorithm | Architecture | Avg Reward  | Avg Steps | Timesteps |
| --------- | ------------ | ----------- | --------- | --------- |
| **SAC**   | **MLP**      | **5142.98** | **987.6** | 1.5M      |
| TD3       | MLP          | 4423.32     | 956.9     | 1.5M      |
| SAC       | CNN          | 3343.07¹    | 806.4     | 3M        |
| TD3       | CNN          | 2890.34     | 751.8     | 3M        |
| PPO       | MLP          | 2068.64     | 586.5     | 1.5M      |
| PPO       | CNN          | 596.92      | 278.8     | 600K      |

¹ Single seed; remaining seeds did not complete within available compute.

### TensorBoard Plot

[TensorBoard Plot](https://www.youtube.com/watch?v=JBKg4J9a06s)

### Key findings

- **SAC + MLP is the clear winner** (5142.98 reward), with two seeds sustaining the
  full 1,000-step episode — the robot walks the entire episode without falling.
  Off-policy replay makes it sample-efficient and entropy regularisation prevents
  premature convergence to poor gaits.
- **CNN hurts here.** Walker2d's observation vector has no spatial ordering — a
  joint's position and velocity can be many indices apart — so a 1D-CNN's
  local-neighbour assumption adds difficulty without useful structure, and needs
  double the training budget.
- **PPO lags** as an on-policy method that discards experience after each update;
  one seed scored only ~1,098, highlighting initialisation sensitivity.
- A **Transformer** variant was prototyped but showed critic instability from
  unnormalised inputs (fixed with input/output LayerNorm); full multi-seed results
  were not obtained due to time constraints.

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── report/
│   └── ML_Report.pdf                      # Full technical report
├── task1_vqa/                             # Task 1 — Medical VQA
│   ├── vqa_models.py                      # 7 model architectures
│   ├── vqa_utils.py                       # Data loading, training loop, metrics
│   ├── agentic_vqa.py                     # LangGraph agentic hyperparameter search
│   ├── run_single_model.py               # Train/eval one model
│   ├── run_biomedclip.py                 # BiomedCLIP (separate pipeline)
│   ├── evaluate_agentic_checkpoints.py    # Re-evaluate saved checkpoints
│   └── results/                           # Saved experiment results (JSON)
└── task2_drl/                             # Task 2 — Deep RL
    ├── task2_walker2d_drl.py              # PPO / SAC / TD3 with MLP policy
    ├── task2_walker2d_cnn_transformer.py  # CNN & Transformer feature extractors
    └── results/                           # Saved reward results (JSON)
```

> Trained model weights, RL policies, TensorBoard logs and the downloaded SLAKE
> dataset are intentionally **not** committed (see [`.gitignore`](.gitignore)).
> They regenerate by running the scripts below.

---

## Setup

```bash
git clone https://github.com/oseniabdulhaleem/advanced-ml-coursework.git
cd advanced-ml-coursework

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended (especially for Task 1).
Task 2 requires MuJoCo, installed via the `gymnasium[mujoco]` / `mujoco` packages.

---

## Running

### Task 1 — Medical VQA

```bash
cd task1_vqa

# Train/evaluate a single model — set MODEL_TYPE inside the script first
python run_single_model.py            # vit_concat | vit_hadamard | resnet_concat |
                                      # resnet_hadamard | efficientnet | cross_attention

python run_biomedclip.py              # BiomedCLIP (best model)

python agentic_vqa.py                 # Agentic search over all configs (resumable)
```

The SLAKE dataset downloads automatically from `BoKelvin/SLAKE` on Hugging Face on
first run.

### Task 2 — Deep RL

```bash
cd task2_drl

python task2_walker2d_drl.py train            # train PPO + SAC + TD3 (MLP, multi-seed)
python task2_walker2d_drl.py test             # evaluate trained policies
python task2_walker2d_drl.py compare          # print comparison table

python task2_walker2d_cnn_transformer.py train SAC CNN   # train a specific combo
python task2_walker2d_cnn_transformer.py compare
```

---

## Acknowledgements

Built for **Advanced Machine Learning**, MSc Robotics & AI, University of
Lincoln. Model and training patterns were adapted from the module workshop
materials (vision/text encoders, fusion patterns, the DRL agent wrapper, and the
LangGraph agentic template), then extended with real experiment execution,
multi-seed evaluation and additional encoders/fusion strategies. I also used Claude Opus for some tasks.

### References

- Liu et al., _SLAKE: A Semantically-Labeled Knowledge-Enhanced Dataset for Medical VQA_, IEEE ISBI, 2021.
- Zhang et al., _BiomedCLIP: A Multimodal Biomedical Foundation Model_, arXiv:2303.00915, 2023.
- Dosovitskiy et al., _An Image is Worth 16×16 Words_ (ViT), ICLR, 2021.
- He et al., _Deep Residual Learning for Image Recognition_ (ResNet), CVPR, 2016.
- Tan & Le, _EfficientNet_, ICML, 2019.
- Haarnoja et al., _Soft Actor-Critic_, ICML, 2018.
- Fujimoto et al., _Addressing Function Approximation Error in Actor-Critic Methods_ (TD3), ICML, 2018.
- Schulman et al., _Proximal Policy Optimization Algorithms_, arXiv:1707.06347, 2017.
- Raffin et al., _Stable-Baselines3: Reliable RL Implementations_, JMLR, 2021.
- Towers et al., _Gymnasium: A Standard Interface for RL Environments_, NeurIPS, 2025.

---

## Author

**Abdulhaleem Oseni**
