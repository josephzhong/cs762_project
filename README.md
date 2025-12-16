# cs762_project
For the project of CS762, 2025 Fall

# R2T: Reward Hacking Detection with Multi-Vector Representations

Code for the CS762 project **R2T**, an inference-time method to detect reward hacking in post-trained LLMs using **multi-vector latent representations** (trainable retrieval tokens + contrastive learning).

## What’s inside
- **Setup:**
  1. pip install -r requirements.txt
  2. install flash attention
- **R2T training:** append R2T tokens, run a LoRA-adapted backbone, extract multi-vector embeddings, and train with an InfoNCE objective on the training dataset.
  
  E.x. To train R2T with a Qwen3-8B backbone:
  
  python train.py 
    --model-name Qwen3/Qwen3-8B 
    --dataset josephzhong/text-math-RewardHacking 
    --groups-q 1 2 4 8 16 --groups-c 1 2 4 8 16 
    --num-epochs 4 --max-length 12288 
    --batch-size 4 
    --lr 1e-4 --temperature 0.7 
    --lora-r 32 --lora-alpha 64 

- **R2T evaluation:** evaluate the trained R2T on a test dataset.
  
  To evaluate a saved R2T checkpoint:
  
  python eval.py 
    --model-name 
     <BACKBONE_MODEL_NAME> 
    --save-path 
     <THE_MODEL_SAVE_PATH> 
    --dataset josephzhong/text-math-RewardHacking 
    --attn-implementation flash_attention_2 
    --threshold-precision 0.05 
    --dtype bfloat16 
    --batch-size 4 
    --limit-ratio 1.0 
    --max-length 12288 
    --temperature 0.7

- ## Notes
  The script has been tested in a single GPU with 80GB RAM. Otherwise, try to set a smaller batch_size.
  The script should be able to reproduce **AUC > 0.7**.

