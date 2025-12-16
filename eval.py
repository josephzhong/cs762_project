import argparse
import os
from datetime import datetime

import torch
from accelerate import Accelerator
from accelerate.utils import tqdm
from sklearn import metrics
from transformers import AutoTokenizer, AutoProcessor

from data_utils import make_dataloader
from model_utils import R2TMetaEmbed


def get_performance(model, data_loader, thresholds, rq, rc, accelerator, device, pbar):
    tp = torch.zeros_like(thresholds)
    tn = torch.zeros_like(thresholds)
    fp = torch.zeros_like(thresholds)
    fn = torch.zeros_like(thresholds)
    for step, batch in enumerate(data_loader):
        batch = {k: v.to(device=device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with accelerator.autocast():
            s, s_neg, _, __, ___ = model(batch, rq=rq, rc=rc)
        print("dtype:", s.dtype)

        B = s.shape[0]
        s_neg_reshaped = s_neg.view(B, -1)
        hard_index = torch.arange(start=0, end=B, step=1, dtype=torch.long, device=s.device).unsqueeze(1)
        s_neg_hard = torch.gather(s_neg_reshaped, dim=1, index=hard_index).flatten()
        temp_thresholds = thresholds.unsqueeze(dim=0).expand(B, -1)
        s = s.unsqueeze(dim=1).expand(-1, temp_thresholds.shape[1])
        s_neg_hard = s_neg_hard.unsqueeze(dim=1).expand(-1, temp_thresholds.shape[1])
        tp += (s >= temp_thresholds).sum(dim=0)
        tn += (s_neg_hard < temp_thresholds).sum(dim=0)
        fp += (s_neg_hard >= temp_thresholds).sum(dim=0)
        fn += (s < temp_thresholds).sum(dim=0)
        pbar.update(1)

    eps = 1e-8
    fpr = fp / (fp + tn + eps)
    tpr = tp / (tp + fn + eps)
    roc_auc = metrics.auc(fpr.cpu().numpy(), tpr.cpu().numpy())

    return fpr, tpr, roc_auc


def evaluation(model, data_loader, groups_q, groups_c, accelerator, threshold_precision=0.05):
    fpr_list, tpr_list, roc_auc_list = [], [], []
    group_length = len(groups_q)
    total_steps = len(data_loader) * group_length
    pbar = tqdm(
        total=total_steps,
        desc="Eval:"
    )
    model.eval()
    device = model.r2t_q.embedding.device
    thresholds = torch.arange(0, 1.0 + threshold_precision, threshold_precision, device=device, dtype=torch.float32)
    with torch.no_grad():
        for group_index in range(group_length):
            rq = groups_q[group_index]
            rc = groups_c[group_index]
            fpr, tpr, roc_auc = get_performance(model, data_loader, thresholds, rq, rc, accelerator, device, pbar)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            roc_auc_list.append(roc_auc)

    return fpr_list, tpr_list, roc_auc_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MetaEmbed R2T with LoRA + meta tokens")
    # Model / LoRA
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--soft-weight", type=float, default=0.03)

    # Eval
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--group-len", type=int, default=-1)
    parser.add_argument("--threshold-precision", type=float, default=0.05)

    # Data
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--dataset-split-eval", type=str, default="test")
    parser.add_argument("--problem-col-name", type=str, default="problem")

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    dtype = torch.bfloat16

    accelerator = Accelerator(mixed_precision="bf16")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    fpr_list, tpr_list, roc_auc_list = [], [], []
    model_name, save_path = args.model_name, args.save_path
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True,
                                              attn_implementation=args.attn_implementation)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, tokenizer = R2TMetaEmbed.load_model(model_name, save_path, tokenizer,
                                               args.temperature, args.attn_implementation,
                                               args.soft_weight, device, dtype)
    eval_loader = make_dataloader(tokenizer, args.batch_size, args.dataset_split_eval, args,
                                  model.groups_q, model.groups_c, shuffle=False)

    model  = accelerator.prepare_model(model)
    eval_loader = accelerator.prepare_data_loader(eval_loader)


    groups_q, groups_c = model.groups_q, model.groups_c
    group_length = max(model.groups_q)
    fpr, tpr, roc_auc = evaluation(model, eval_loader, groups_q, groups_c,
                                   accelerator=accelerator,
                                   threshold_precision=args.threshold_precision)
    print(f"fpr: {fpr}", f"tpr: {tpr}", f"roc_auc: {roc_auc}")



