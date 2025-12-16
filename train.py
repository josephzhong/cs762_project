import argparse
from datetime import datetime
import math
import os
import torch
from datasets import tqdm
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    set_seed
)
from peft import LoraConfig
from accelerate import Accelerator

from data_utils import make_dataloader
from model_utils import load_base_model, R2TMetaEmbed


def log_step(loss_group, lr, grad, step, accel, mode):
    if mode == "train":
        log_data = {
            "train/loss": loss_group.sum().item(),
            "train/lr": lr,
            "train/gradient_norm": grad
        }
        for g_index in range(loss_group.size(0)):
            log_data[f"train/loss_group_{g_index}"] = loss_group[g_index].item()
        accel.log(
            log_data,
            step=step
        )


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="R2T with LoRA")
    # Model / LoRA
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--soft-weight", type=float, default=0.2)

    # R2T tokens
    parser.add_argument("--groups-q", type=int, nargs="+", default=[1, 2, 4, 8], help="groups for query")
    parser.add_argument("--groups-c", type=int, nargs="+", default=[1, 2, 4, 8], help="groups for candidate")

    # Train
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)

    # Data
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--dataset-split-train", type=str, default="train")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="qwen3_r2t_lora")

    args = parser.parse_args()

    set_seed(args.seed)
    orig_proj_dir = os.getenv("ACCEL_LOG_DIR", "./logs")
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    project_dir = os.path.join(orig_proj_dir, timestamp + "_r2t")

    accelerator = Accelerator(log_with=["tensorboard"],
                              gradient_accumulation_steps=2,
                              mixed_precision="bf16",
                              project_dir=project_dir)

    assert len(args.groups_q) == len(args.groups_c)

    if accelerator.is_main_process:
        print("Parsed Arguments:")
        for arg_name, arg_value in vars(args).items():
            print(f"  {arg_name}: {arg_value}")

    dtype = torch.bfloat16

    base = load_base_model(args.model_name, attn_implementation=args.attn_implementation, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True,
                                              attn_implementation=args.attn_implementation)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader = make_dataloader(tokenizer, args.batch_size, args.dataset_split_train, args)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

    model = R2TMetaEmbed(base, tokenizer, args.groups_q, args.groups_c, args.temperature, lora_cfg, args.soft_weight)

    model.backbone.config.use_cache = False
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    decay, nodecay = [], []
    for n, p in trainable_params:
        # typical rule: no decay on bias, LayerNorm/Norm params, embeddings, and our r2t tokens
        if (
                n.endswith(".bias")
                or "norm" in n.lower()
                or "ln_" in n.lower()
                or "embed" in n.lower()
                or n.startswith("r2t_q.")
                or n.startswith("r2t_c.")
        ):
            nodecay.append(p)
        else:
            decay.append(p)

    trainable_params = [p for _, p in trainable_params]

    def uniq(params):
        seen = {}
        for p in params:
            if id(p) in seen:
                print(f"{p} has already been seen.")
            seen[id(p)] = p
        return list(seen.values())

    decay = uniq(decay)
    nodecay = uniq(nodecay)

    decay_ids = {id(p) for p in decay}
    nodecay = [p for p in nodecay if id(p) not in decay_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ],
        betas=(args.adam_beta1, args.adam_beta2),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    total_steps = math.ceil(len(train_loader) // accelerator.gradient_accumulation_steps ) * args.num_epochs
    print(f"total_steps: {total_steps}")
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    print(f"warmup_steps: {warmup_steps}")
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model, optimizer, train_loader, scheduler= accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # Train
    model.train()
    global_step = 0
    device = model.backbone.get_input_embeddings().weight.device
    group_length = len(args.groups_q)

    # Create one tqdm only (main process only)
    pbar = tqdm(
        total=total_steps,
        disable=not accelerator.is_main_process,
        desc="Training:"
    )

    for epoch in range(args.num_epochs):
        running_loss = torch.zeros((group_length,), dtype=torch.float32, device=device)
        running_loss_cnt = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss_group = model(batch)
                    loss = loss_group.sum()
                    gathered = accelerator.gather(loss_group.detach())
                    running_loss += gathered
                    running_loss_cnt += 1
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    optimizer.step() # no use here
                    gradient = accelerator.clip_grad_norm_(trainable_params, 1.0)
                    global_step += 1
                    avg_loss = running_loss / running_loss_cnt
                    if accelerator.is_main_process:
                        pbar.update(1)
                        loss_info={
                            "loss@total": avg_loss.sum().item(),
                        }
                        for i, g in enumerate(args.groups_q):
                            loss_info[f"loss@{g}"] = avg_loss[i].item()

                        pbar.set_postfix(loss_info)
                        log_step(avg_loss, scheduler.get_last_lr()[0], gradient, global_step, accelerator, "train")

                    running_loss[:] = 0.0
                    running_loss_cnt = 0

                    scheduler.step()
                    optimizer.zero_grad() # no use here


        if accelerator.is_main_process:
            # Save: LoRA adapter + R2T tokens
            save_dir = os.path.join(project_dir, f"epoch_{epoch}")
            model.save_model(save_dir, args.groups_q, args.groups_c)

    if accelerator.is_main_process:
        save_dir = os.path.join(project_dir, f"final")
        model.save_model(save_dir, args.groups_q, args.groups_c)


if __name__ == "__main__":
    main()