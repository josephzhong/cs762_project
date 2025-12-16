# cs762_project
For the project of CS762, 2025 Fall

# R2T: Reward Hacking Detection with Multi-Vector Representations

Code for the CS762 project **R2T**, an inference-time method to detect reward hacking in post-trained LLMs using **multi-vector latent representations** (trainable retrieval tokens + contrastive learning).

## What’s inside
- **Setup:**
  1. pip install -r requirements.txt
  2. install [flash attention](https://github.com/Dao-AILab/flash-attention)
  3. Get access to the [dataset](https://huggingface.co/datasets/josephzhong/text-math-RewardHacking)
- **R2T training:** append R2T tokens, run a LoRA-adapted backbone, extract multi-vector embeddings, and train with an InfoNCE objective on the training dataset. 
  - To train R2T with a Qwen3-8B backbone: \
  export HF_TOKEN=<YOUR_HF_TOKEN>;
  python train.py 
  --model-name Qwen/Qwen3-8B 
  --dataset josephzhong/text-math-RewardHacking 
  --attn-implementation flash_attention_2
  --groups-q 1 2 4 8 16 --groups-c 1 2 4 8 16 
  --num-epochs 4 --max-length 12288 
  --batch-size 4 
  --lr 1e-4 --temperature 0.7 
  --lora-r 32 --lora-alpha 64 

- **R2T evaluation:** evaluate the trained R2T on a test dataset. 
  - To evaluate a saved R2T checkpoint: \
    python eval.py 
  --model-name 
   <BACKBONE_MODEL_NAME> 
  --save-path 
   <THE_MODEL_SAVE_PATH> 
  --dataset josephzhong/text-math-RewardHacking 
  --attn-implementation flash_attention_2 
  --threshold-precision 0.05
  --batch-size 4
  --max-length 12288 
  --temperature 0.7

## Notes
  The script has been tested for a 8B backbone with a single 80GB-RAM-GPU. Otherwise, try to use a smaller backbone or set to a smaller batch_size.
  The script should be able to reproduce **AUC > 0.73**.

