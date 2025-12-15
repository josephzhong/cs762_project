# cs762_project
For the project of CS762, 2025 Fall

# R2T: Reward Hacking Detection with Multi-Vector Representations

Code for the CS762 project **R2T**, an inference-time method to detect reward hacking in post-trained LLMs using **multi-vector latent representations** (trainable retrieval tokens + contrastive learning).

## What’s inside
- **Dataset generation:** build labeled (hacked vs. non-hacked) prompt–response pairs via GRPO using  
  (1) a verifiable reward function and (2) a modified reward that injects false-positive rewards.
- **R2T training:** append R2T tokens, run a LoRA-adapted backbone, extract multi-vector embeddings, and train with an InfoNCE objective.
- **Inference:** compute similarity scores over learned vectors to estimate reward hacking probability.

  ## Results
Best detection performance: **AUC = 0.76** (vector group length 16).

