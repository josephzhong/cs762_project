from typing import List, Any, Dict

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataclasses import dataclass


@dataclass
class DataCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int, Rq: int, Rc: int, problem_col_name: str = "prompt"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.Rq = Rq
        self.Rc = Rc
        self.problem_col_name = problem_col_name


    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        q_text = [ex[self.problem_col_name] for ex in batch]
        c_text = [ex["generations_rh"] for ex in batch]
        c_neg_text = [ex["generations"] for ex in batch]
        bsz = len(batch)

        q_enc = self.tokenizer(
            q_text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None,
        )
        c_enc = self.tokenizer(
            c_text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None,
        )
        c_neg_enc = self.tokenizer(
            c_neg_text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None,
        )

        pad_id = self.tokenizer.pad_token_id
        q_ids = q_enc["input_ids"]
        c_ids = c_enc["input_ids"]
        c_neg_ids = c_neg_enc["input_ids"]

        text_lens_q = torch.tensor([len(t) for t in q_ids], dtype=torch.long)
        max_text_len_q = int(text_lens_q.max().item())
        max_final_len_q = max_text_len_q + self.Rq

        c_ids_all = []
        c_ids_all.extend(c_ids)
        c_ids_all.extend(c_neg_ids)

        text_lens_c = torch.tensor([len(t) for t in c_ids_all], dtype=torch.long)
        max_text_len_c = int(text_lens_c.max().item())
        max_final_len_c = max_text_len_c + self.Rc

        max_final_len = max(max_final_len_q, max_final_len_c)

        attention_mask = torch.ones((3 * bsz, max_final_len), dtype=torch.long)

        for row in range(len(q_ids)):
            q_ids[row].extend([pad_id] * (max_final_len - text_lens_q[row].item()))
            attention_mask[row, text_lens_q[row].item() + self.Rq:] = 0
        for row in range(len(c_ids)):
            c_ids[row].extend([pad_id] * (max_final_len - text_lens_c[row].item()))
            attention_mask[row + len(q_ids), text_lens_c[row].item() + self.Rc:] = 0
        for row in range(len(c_neg_ids)):
            c_neg_ids[row].extend([pad_id] * (max_final_len - text_lens_c[row + len(c_ids)].item()))
            attention_mask[row + len(q_ids) + len(c_ids), text_lens_c[row + len(c_ids)].item() + self.Rc:] = 0

        return {
            "q_input_ids": torch.tensor(q_ids, dtype=torch.long),
            "c_input_ids": torch.tensor(c_ids, dtype=torch.long),
            "c_neg_input_ids": torch.tensor(c_neg_ids, dtype=torch.long),
            "r2t_starts_q": text_lens_q,
            "r2t_starts_c": text_lens_c[:len(c_ids)],
            "r2t_starts_c_neg": text_lens_c[len(c_ids):],
            "attention_mask": attention_mask
        }


def make_dataloader(tokenizer, batch_size, split, args, groups_q=None, groups_c=None, problem_col_name="prompt", shuffle=True):
    """
    Replace this with your own dataset mapping function if needed.
    """
    ds_tr = load_dataset(args.dataset, "default", split=split)
    ds = ds_tr.with_format("torch")

    if groups_q is None:
        groups_q = args.groups_q
    if groups_c is None:
        groups_c = args.groups_c
    Rq = max(groups_q)
    Rc = max(groups_c)
    collator = DataCollator(tokenizer, args.max_length, Rq, Rc, problem_col_name=problem_col_name)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)
    return loader
