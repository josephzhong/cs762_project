import os
import torch
from peft import get_peft_model, LoraConfig
from torch import nn
from typing import Optional, Unpack, List
from transformers import Qwen3PreTrainedModel, GenerationMixin, Qwen3Model, AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import can_return_tuple, auto_docstring, TransformersKwargs


class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states = False,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python=
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states = output_hidden_states,
            use_cache=use_cache,
            **kwargs,
        )
        return outputs


def load_base_model(model_name, save_path=None, attn_implementation=None, dtype=None, max_length=None):
    if save_path is None:
        save_path = model_name
    if dtype is None:
        dtype = torch.float32
    if model_name.find("Qwen3") > -1:
        base_cls = Qwen3ForCausalLM
    else:
        # TODO: needs some special processing to save GPU RAM
        base_cls = AutoModelForCausalLM
    return base_cls.from_pretrained(save_path, trust_remote_code=True,
                                    attn_implementation=attn_implementation, dtype=dtype)


class R2T(nn.Module):
    def __init__(self, num_tokens: int, hidden_size: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.embedding = nn.Parameter(torch.randn(num_tokens, hidden_size) * 0.02, requires_grad=True)

    def init_weights(self, tok_embedding_weight):
        mu = tok_embedding_weight.mean()
        sigma = tok_embedding_weight.std() + 1e-6
        with torch.no_grad():
            self.embedding.copy_(mu + torch.randn_like(self.embedding) * sigma)

    def forward(self, bsz: int, dtype):
        return self.embedding.to(dtype=dtype).unsqueeze(0).expand(bsz, -1, -1)


class R2TMetaEmbed(nn.Module):
    def __init__(self, base_model, tokenizer, groups_q: List[int], groups_c: List[int], temperature: float,
                 lora_cfg: LoraConfig | None, soft_weight: float,
                 load=False, backbone=None, r2t_q=None, r2t_c=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.groups_q = groups_q
        self.groups_c = groups_c
        self.temperature = temperature
        self.Rq = max(groups_q)
        self.Rc = max(groups_c)
        if load:
            self.soft_weight = soft_weight
            self.backbone = backbone
            self.hidden_size = self.backbone.config.hidden_size
            self.r2t_q = r2t_q
            self.r2t_c = r2t_c
        else:
            self.soft_weight = soft_weight
            print("get peft for transformer")
            self.backbone = get_peft_model(base_model, lora_cfg)
            self.hidden_size = self.backbone.config.hidden_size
            tok_emb = self.backbone.get_input_embeddings()
            weight = tok_emb.weight
            self.r2t_q = R2T(self.Rq, self.hidden_size).to(device=weight.device)
            self.r2t_c = R2T(self.Rc, self.hidden_size).to(device=weight.device)
            self.r2t_q.init_weights(weight)
            self.r2t_c.init_weights(weight)

    def save_model(self, save_dir, groups_q, groups_c, tokenizer=None):
        os.makedirs(save_dir, exist_ok=True)
        self.backbone.save_pretrained(save_dir, base_model=self.backbone.model.config._name_or_path)
        torch.save({"groups_q": groups_q, "state_dict": self.r2t_q.state_dict()}, os.path.join(save_dir, "r2t_q.pt"))
        torch.save({"groups_c": groups_c, "state_dict": self.r2t_c.state_dict()}, os.path.join(save_dir, "r2t_c.pt"))
        if tokenizer:
            tokenizer.save_pretrained(save_dir)
        print(f"Saved LoRA adapter + R2T to {save_dir}")

    @staticmethod
    def load_model(model_name, save_path, tokenizer, temperature, attn_implementation, soft_weight, device, dtype):
        if model_name.find("gpt-oss") > -1:
            backbone, tokenizer = load_base_model(model_name, save_path, attn_implementation=attn_implementation, dtype=dtype)
        else:
            backbone = load_base_model(model_name, save_path, attn_implementation=attn_implementation, dtype=dtype)
        groups_q_dict = torch.load(os.path.join(save_path, "r2t_q.pt"), map_location=device)
        groups_q = groups_q_dict["groups_q"]
        print(f"info: groups_q: {groups_q}")
        r2t_q = R2T(max(groups_q), backbone.config.hidden_size)
        r2t_q.load_state_dict(groups_q_dict["state_dict"])
        groups_c_dict = torch.load(os.path.join(save_path, "r2t_c.pt"), map_location=device)
        groups_c = groups_c_dict["groups_c"]
        print(f"info: groups_c: {groups_c}")
        r2t_c = R2T(max(groups_c), backbone.config.hidden_size)
        r2t_c.load_state_dict(groups_c_dict["state_dict"])
        model = R2TMetaEmbed(base_model=None, tokenizer=tokenizer, groups_q=groups_q, groups_c=groups_c, temperature=temperature,
                            lora_cfg=None, soft_weight=soft_weight, load=True, backbone=backbone, r2t_q=r2t_q, r2t_c=r2t_c)
        model.to(device)
        return model, tokenizer

    def forward(self, batch, neg=True, rq=None, rc=None, target_step_size=2):
        if neg:
            if self.training:
                r2t_span_q, r2t_span_c, r2t_span_c_neg = self.encode_query_candidate_and_neg(batch, target_step_size=target_step_size)
                return self.compute_loss_group(r2t_span_q, r2t_span_c, r2t_span_c_neg)
            else:
                r2t_span_q, r2t_span_c, r2t_span_c_neg = self.encode_query_candidate_and_neg(batch, target_step_size=target_step_size)
                s = (self.inference_score(r2t_span_q, r2t_span_c, rq, rc) / rq + 1) / 2.0 # normalize score for testing
                s_neg = (self.inference_score_neg(r2t_span_q, r2t_span_c_neg, rq, rc) / rq + 1) / 2.0 # normalize score for testing
                return s, s_neg, r2t_span_q, r2t_span_c, r2t_span_c_neg
        else:
            assert not self.training
            r2t_span_q, r2t_span_c = self.encode_query_candidate_and_neg(batch, neg=False, target_step_size=target_step_size)
            s = (self.inference_score(r2t_span_q, r2t_span_c, rq, rc) / rq + 1) / 2.0 # normalize score for testing
            return s, None, r2t_span_q, r2t_span_c


    def encode_query_candidate_and_neg(self, batch, neg=True, target_step_size=2):
        r2t_span_q_list, r2t_span_c_list, r2t_span_c_neg_list = [], [], []
        batch_size = batch["q_input_ids"].shape[0]
        for start_index in range(0, batch_size, target_step_size):
            step_size = min(batch_size - start_index, target_step_size)
            q_ids = batch["q_input_ids"][start_index:start_index + step_size]
            c_ids = batch["c_input_ids"][start_index:start_index + step_size]
            r2t_starts_q = batch["r2t_starts_q"][start_index:start_index + step_size]
            r2t_starts_c = batch["r2t_starts_c"][start_index:start_index + step_size]

            q_attention_mask = batch["attention_mask"][start_index:start_index + step_size]
            c_attention_mask = batch["attention_mask"][start_index + batch_size: start_index + batch_size + step_size]
            if neg:
                c_neg_ids = batch["c_neg_input_ids"][start_index:start_index + step_size]
                r2t_starts_c_neg = batch["r2t_starts_c_neg"][start_index:start_index + step_size]
                c_neg_attention_mask = batch["attention_mask"][
                    start_index + 2 * batch_size: start_index + 2 * batch_size + step_size]
                attention_mask = torch.cat([q_attention_mask, c_attention_mask, c_neg_attention_mask], dim=0)
            else:
                attention_mask = torch.cat([q_attention_mask, c_attention_mask], dim=0)

            bsz = q_ids.size(0)
            tok_emb_q = self.backbone.get_input_embeddings()(q_ids)  # (B, L, H)
            hidden_size = tok_emb_q.size(2)
            r2t_q_emb = self.r2t_q(bsz, dtype=tok_emb_q.dtype) # (B, R, H)
            pos = torch.arange(self.Rq, device=r2t_starts_q.device).unsqueeze(0)  # [1, R]
            cols_q = r2t_starts_q.unsqueeze(1) + pos  # [B, R] absolute positions
            cols_q = cols_q.unsqueeze(2).expand(-1, -1, hidden_size)
            tok_emb_q = tok_emb_q.scatter(dim=1, index=cols_q, src=r2t_q_emb)

            tok_emb_c = self.backbone.get_input_embeddings()(c_ids)  # (B, L, H)
            r2t_c_emb = self.r2t_c(bsz, dtype=tok_emb_c.dtype)
            pos = torch.arange(self.Rc, device=r2t_starts_c.device).unsqueeze(0)  # [1, R]
            cols_c = r2t_starts_c.unsqueeze(1) + pos  # [B, R] absolute positions
            cols_c = cols_c.unsqueeze(2).expand(-1, -1, hidden_size)
            tok_emb_c = tok_emb_c.scatter(dim=1, index=cols_c, src=r2t_c_emb)

            if neg:
                tok_emb_c_neg = self.backbone.get_input_embeddings()(c_neg_ids)  # (B, L, H)
                pos = torch.arange(self.Rc, device=r2t_starts_c_neg.device).unsqueeze(0)  # [1, R]
                cols_c_neg = r2t_starts_c_neg.unsqueeze(1) + pos  # [B, R] absolute positions
                cols_c_neg = cols_c_neg.unsqueeze(2).expand(-1, -1, hidden_size)
                tok_emb_c_neg = tok_emb_c_neg.scatter(dim=1, index=cols_c_neg, src=r2t_c_emb)
                inputs_embeds = torch.cat([tok_emb_q, tok_emb_c, tok_emb_c_neg], dim=0)  # (3*B, L, H)
            else:
                inputs_embeds = torch.cat([tok_emb_q, tok_emb_c], dim=0)  # (2*B, L, H)
                attention_mask = attention_mask[:2*bsz, :]
            out = self.backbone(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    use_cache=False,
            )
            last = out.last_hidden_state  # (3 * B, L, H) or (2 * B, L, H)
            r2t_span_q = torch.gather(last[:bsz], dim=1, index=cols_q)  # (B, Rq, H)
            r2t_span_q = nn.functional.normalize(r2t_span_q.float(), dim=-1).to(dtype=last.dtype)
            r2t_span_c = torch.gather(last[bsz:2*bsz], dim=1, index=cols_c)  # (B, Rc, H)
            r2t_span_c = nn.functional.normalize(r2t_span_c.float(), dim=-1).to(dtype=last.dtype)

            r2t_span_q_list.append(r2t_span_q)
            r2t_span_c_list.append(r2t_span_c)
            if neg:
                r2t_span_c_neg = torch.gather(last[2*bsz:], dim=1, index=cols_c_neg)  # (B, Rc, H)  # (B, Rc, H)
                r2t_span_c_neg = nn.functional.normalize(r2t_span_c_neg.float(), dim=-1).to(dtype=last.dtype)
                r2t_span_c_neg_list.append(r2t_span_c_neg)
            del tok_emb_c
            del tok_emb_q
            del r2t_q_emb
            del r2t_c_emb
            if neg:
                del tok_emb_c_neg
            del out
            del last
            del inputs_embeds
            del attention_mask
            torch.cuda.empty_cache()

        r2t_span_q = torch.cat(r2t_span_q_list, dim=0)
        r2t_span_c = torch.cat(r2t_span_c_list, dim=0)
        if neg:
            r2t_span_c_neg = torch.cat(r2t_span_c_neg_list, dim=0)
            return r2t_span_q, r2t_span_c, r2t_span_c_neg  # multi-vector embedding
        else:
            return r2t_span_q, r2t_span_c

    def matryoshka_groups(self, R: int, groups: List[int]) -> List[int]:
        uniq = sorted(list([g for g in groups if 1 <= g <= R]))
        return uniq

    def score_groups(self, Eq: torch.Tensor, Ec: torch.Tensor, Ec_neg: torch.Tensor, rq, rc) -> torch.Tensor:
        Eq_g = Eq[:, :rq, :]  # (B, rq, D)
        Ec_g = Ec[:, :rc, :]  # (B, rc, D)
        Ec_neg_g = Ec_neg[:, :rc, :]
        Ec_g_all = torch.cat([Ec_g, Ec_neg_g], dim=0)
        s = torch.sum(torch.max(torch.tensordot(Eq_g, Ec_g_all, dims=[[2], [2]]),  # (B, rq, 2 * B, rc)
                                dim=3)[0],  # (B, rq, 2 * B)
                      dim=1)  # (B, 2 * B)
        return s

    def info_nce_matryoshka(self, Eq: torch.Tensor, Ec: torch.Tensor, Ec_neg) -> torch.Tensor:
        B, Rq, D = Eq.shape
        _, Rc, _ = Ec.shape

        group_length = len(self.groups_q)

        loss_groups = torch.zeros((group_length,), device=Eq.device)

        for group_index in range(group_length):
            rq = self.groups_q[group_index]
            rc = self.groups_c[group_index]
            s_rq_rc = self.score_groups(Eq, Ec, Ec_neg, rq, rc) / self.temperature
            denominator_index = torch.cat([
                torch.arange(start=0, end=B, dtype=torch.long, device=Eq.device).unsqueeze(1),
                torch.arange(start=B, end=2 * B, dtype=torch.long, device=Eq.device).unsqueeze(1),
            ], dim=1)
            src_value = torch.ones_like(denominator_index, dtype=s_rq_rc.dtype)
            denominator_weight = torch.log(torch.ones_like(s_rq_rc) * self.soft_weight)
            denominator_weight = denominator_weight.scatter(dim=1, index=denominator_index, src=src_value)
            log_probs = (s_rq_rc * denominator_weight).log_softmax(dim=1)  # [B ,2*B]
            index = torch.tensor([[i] for i in range(B)], device=Eq.device)
            loss_groups[group_index] = -1 * torch.gather(log_probs, 1, index).mean()

        return loss_groups

    def compute_loss_group(self, Eq, Ec, Ec_neg):
        Eq = Eq.float()
        Ec = Ec.float()
        Ec_neg = Ec_neg.float()
        loss_group = self.info_nce_matryoshka(Eq, Ec, Ec_neg)
        return loss_group

    def score_func(self, Eq_g: torch.Tensor, Ec_g: torch.Tensor) -> torch.Tensor:
        Eq_g = Eq_g.float()
        Ec_g = Ec_g.float()
        return torch.sum(torch.max(torch.matmul(Eq_g, torch.transpose(Ec_g, 1, 2)),  # (B, rq, rc)
                                   dim=2)[0],  # (B, rq)
                         dim=1)  # (B,)

    def inference_score(self, Eq: torch.Tensor, Ec: torch.Tensor, group_q_length: int, group_c_length: int) -> torch.Tensor:
        Eq_g = Eq[:, :group_q_length, :]  # (B, rq, D)
        Ec_g = Ec[:, :group_c_length, :]  # (B, rc, D)

        return self.score_func(Eq_g, Ec_g)

    def inference_score_neg(self, Eq: torch.Tensor, Ec: torch.Tensor, group_q_length: int, group_c_length: int):
        Eq_g = Eq[:, :group_q_length, :]  # (B, rq, D)
        Ec_g = Ec[:, :group_c_length, :]  # (B, rc, D)
        Eq_g = Eq_g.float()
        Ec_g = Ec_g.float()
        s = torch.sum(torch.max(torch.tensordot(Eq_g, Ec_g, dims=[[2], [2]]),  # (B, rq, B, rc)
                                dim=3)[0],  # (B, rq, B)
                      dim=1)  # (B, B)
        return s.flatten()

