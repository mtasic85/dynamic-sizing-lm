'''
Qwen3ForCausalLM(
  (model): Qwen3Model(
    (embed_tokens): Embedding(151936, 1024)
    (layers): ModuleList(
      (0-27): 28 x Qwen3DecoderLayer(
        (self_attn): Qwen3Attention(
          (q_proj): Linear(in_features=1024, out_features=2048, bias=False)
          (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
          (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
          (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
        )
        (mlp): Qwen3MLP(
          (gate_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (up_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (down_proj): Linear(in_features=3072, out_features=1024, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
        (post_attention_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
      )
    )
    (norm): Qwen3RMSNorm((1024,), eps=1e-06)
    (rotary_emb): Qwen3RotaryEmbedding()
  )
  (lm_head): Linear(in_features=1024, out_features=151936, bias=False)
)
Original model total parameters: 596049920
Original model generation:
The future of AI is not just about the technology itself, but about how we use it to solve real-world problems. As AI continues to evolve, it's important to consider the ethical implications of its use. AI has the potential to bring about significant changes in society, but it also has the power to create new challenges. Therefore, it's crucial to develop a comprehensive approach to AI that takes into account both the benefits and the risks associated with its use. This includes addressing issues such as bias, privacy, and accountability.
Tokens per second (original): 1.97
You're using a Qwen2TokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
Evaluation Original Results: {'wikitext_ppl': 34.81682997450476, 'hellaswag_acc': 0.415, 'boolq_acc': 0.575}
============================================================
Starting Wanda pruning...
Step 1/5: Setting up device and model
Step 2/5: Loading calibration dataset
Resolving data files: 100%|█████████████████████████████████████████████████████████████| 1024/1024 [00:00<00:00, 46010.28it/s]
Resolving data files: 100%|█████████████████████████████████████████████████████████████| 1024/1024 [00:00<00:00, 46819.29it/s]
Step 3/5: Preparing data loader
Preparing calibration texts...
Tokenizing 128 samples...
Step 4/5: Collecting activation statistics
Identifying linear layers...
Collecting activation stats over 32 batches...
Processing batch 1/32
Processing batch 2/32
Processing batch 3/32
Processing batch 4/32
Processing batch 5/32
Processing batch 6/32
Processing batch 7/32
Processing batch 8/32
Processing batch 9/32
Processing batch 10/32
Processing batch 11/32                                                                                                         Processing batch 12/32
Processing batch 13/32
Processing batch 14/32
Processing batch 15/32
Processing batch 16/32
Processing batch 17/32
Processing batch 18/32
Processing batch 19/32
Processing batch 20/32
Processing batch 21/32
Processing batch 22/32
Processing batch 23/32                                                                                                         Processing batch 24/32
Processing batch 25/32
Processing batch 26/32
Processing batch 27/32                                                                                                         Processing batch 28/32
Processing batch 29/32
Processing batch 30/32                                                                                                         Processing batch 31/32
Processing batch 32/32
Computing norms...
Step 5/5: Applying pruning                                                                                                     Applying pruning to 196 layers...
Pruning layer 1/196: model.layers.0.self_attn.q_proj
Pruning layer 2/196: model.layers.0.self_attn.k_proj                                                                           Pruning layer 3/196: model.layers.0.self_attn.v_proj
Pruning layer 4/196: model.layers.0.self_attn.o_proj
Pruning layer 5/196: model.layers.0.mlp.gate_proj
Pruning layer 6/196: model.layers.0.mlp.up_proj
Pruning layer 7/196: model.layers.0.mlp.down_proj
Pruning layer 8/196: model.layers.1.self_attn.q_proj
Pruning layer 9/196: model.layers.1.self_attn.k_proj                                                                           Pruning layer 10/196: model.layers.1.self_attn.v_proj
Pruning layer 11/196: model.layers.1.self_attn.o_proj
Pruning layer 12/196: model.layers.1.mlp.gate_proj
Pruning layer 13/196: model.layers.1.mlp.up_proj
Pruning layer 14/196: model.layers.1.mlp.down_proj
Pruning layer 15/196: model.layers.2.self_attn.q_proj
Pruning layer 16/196: model.layers.2.self_attn.k_proj
Pruning layer 17/196: model.layers.2.self_attn.v_proj
Pruning layer 18/196: model.layers.2.self_attn.o_proj
Pruning layer 19/196: model.layers.2.mlp.gate_proj
Pruning layer 20/196: model.layers.2.mlp.up_proj
Pruning layer 21/196: model.layers.2.mlp.down_proj                                                                             Pruning layer 22/196: model.layers.3.self_attn.q_proj
Pruning layer 23/196: model.layers.3.self_attn.k_proj
Pruning layer 24/196: model.layers.3.self_attn.v_proj
Pruning layer 25/196: model.layers.3.self_attn.o_proj
Pruning layer 26/196: model.layers.3.mlp.gate_proj
Pruning layer 27/196: model.layers.3.mlp.up_proj
Pruning layer 28/196: model.layers.3.mlp.down_proj
Pruning layer 29/196: model.layers.4.self_attn.q_proj
Pruning layer 30/196: model.layers.4.self_attn.k_proj
Pruning layer 31/196: model.layers.4.self_attn.v_proj
Pruning layer 32/196: model.layers.4.self_attn.o_proj
Pruning layer 33/196: model.layers.4.mlp.gate_proj
Pruning layer 34/196: model.layers.4.mlp.up_proj
Pruning layer 35/196: model.layers.4.mlp.down_proj
Pruning layer 36/196: model.layers.5.self_attn.q_proj
Pruning layer 37/196: model.layers.5.self_attn.k_proj
Pruning layer 38/196: model.layers.5.self_attn.v_proj
Pruning layer 39/196: model.layers.5.self_attn.o_proj
Pruning layer 40/196: model.layers.5.mlp.gate_proj
Pruning layer 41/196: model.layers.5.mlp.up_proj
Pruning layer 42/196: model.layers.5.mlp.down_proj                                                                             Pruning layer 43/196: model.layers.6.self_attn.q_proj
Pruning layer 44/196: model.layers.6.self_attn.k_proj
Pruning layer 45/196: model.layers.6.self_attn.v_proj
Pruning layer 46/196: model.layers.6.self_attn.o_proj
Pruning layer 47/196: model.layers.6.mlp.gate_proj
Pruning layer 48/196: model.layers.6.mlp.up_proj
Pruning layer 49/196: model.layers.6.mlp.down_proj                                                                             Pruning layer 50/196: model.layers.7.self_attn.q_proj
Pruning layer 51/196: model.layers.7.self_attn.k_proj
Pruning layer 52/196: model.layers.7.self_attn.v_proj
Pruning layer 53/196: model.layers.7.self_attn.o_proj
Pruning layer 54/196: model.layers.7.mlp.gate_proj
Pruning layer 55/196: model.layers.7.mlp.up_proj
Pruning layer 56/196: model.layers.7.mlp.down_proj
Pruning layer 57/196: model.layers.8.self_attn.q_proj
Pruning layer 58/196: model.layers.8.self_attn.k_proj
Pruning layer 59/196: model.layers.8.self_attn.v_proj
Pruning layer 60/196: model.layers.8.self_attn.o_proj                                                                          Pruning layer 61/196: model.layers.8.mlp.gate_proj
Pruning layer 62/196: model.layers.8.mlp.up_proj
Pruning layer 63/196: model.layers.8.mlp.down_proj                                                                             Pruning layer 64/196: model.layers.9.self_attn.q_proj
Pruning layer 65/196: model.layers.9.self_attn.k_proj
Pruning layer 66/196: model.layers.9.self_attn.v_proj
Pruning layer 67/196: model.layers.9.self_attn.o_proj
Pruning layer 68/196: model.layers.9.mlp.gate_proj
Pruning layer 69/196: model.layers.9.mlp.up_proj
Pruning layer 70/196: model.layers.9.mlp.down_proj                                                                             Pruning layer 71/196: model.layers.10.self_attn.q_proj
Pruning layer 72/196: model.layers.10.self_attn.k_proj
Pruning layer 73/196: model.layers.10.self_attn.v_proj
Pruning layer 74/196: model.layers.10.self_attn.o_proj
Pruning layer 75/196: model.layers.10.mlp.gate_proj
Pruning layer 76/196: model.layers.10.mlp.up_proj
Pruning layer 77/196: model.layers.10.mlp.down_proj
Pruning layer 78/196: model.layers.11.self_attn.q_proj
Pruning layer 79/196: model.layers.11.self_attn.k_proj
Pruning layer 80/196: model.layers.11.self_attn.v_proj
Pruning layer 81/196: model.layers.11.self_attn.o_proj
Pruning layer 82/196: model.layers.11.mlp.gate_proj
Pruning layer 83/196: model.layers.11.mlp.up_proj
Pruning layer 84/196: model.layers.11.mlp.down_proj
Pruning layer 85/196: model.layers.12.self_attn.q_proj
Pruning layer 86/196: model.layers.12.self_attn.k_proj
Pruning layer 87/196: model.layers.12.self_attn.v_proj
Pruning layer 88/196: model.layers.12.self_attn.o_proj
Pruning layer 89/196: model.layers.12.mlp.gate_proj
Pruning layer 90/196: model.layers.12.mlp.up_proj
Pruning layer 91/196: model.layers.12.mlp.down_proj
Pruning layer 92/196: model.layers.13.self_attn.q_proj
Pruning layer 93/196: model.layers.13.self_attn.k_proj
Pruning layer 94/196: model.layers.13.self_attn.v_proj
Pruning layer 95/196: model.layers.13.self_attn.o_proj
Pruning layer 96/196: model.layers.13.mlp.gate_proj
Pruning layer 97/196: model.layers.13.mlp.up_proj
Pruning layer 98/196: model.layers.13.mlp.down_proj
Pruning layer 99/196: model.layers.14.self_attn.q_proj
Pruning layer 100/196: model.layers.14.self_attn.k_proj
Pruning layer 101/196: model.layers.14.self_attn.v_proj
Pruning layer 102/196: model.layers.14.self_attn.o_proj
Pruning layer 103/196: model.layers.14.mlp.gate_proj
Pruning layer 104/196: model.layers.14.mlp.up_proj
Pruning layer 105/196: model.layers.14.mlp.down_proj
Pruning layer 106/196: model.layers.15.self_attn.q_proj
Pruning layer 107/196: model.layers.15.self_attn.k_proj
Pruning layer 108/196: model.layers.15.self_attn.v_proj
Pruning layer 109/196: model.layers.15.self_attn.o_proj
Pruning layer 110/196: model.layers.15.mlp.gate_proj
Pruning layer 111/196: model.layers.15.mlp.up_proj
Pruning layer 112/196: model.layers.15.mlp.down_proj
Pruning layer 113/196: model.layers.16.self_attn.q_proj
Pruning layer 114/196: model.layers.16.self_attn.k_proj
Pruning layer 115/196: model.layers.16.self_attn.v_proj
Pruning layer 116/196: model.layers.16.self_attn.o_proj
Pruning layer 117/196: model.layers.16.mlp.gate_proj
Pruning layer 118/196: model.layers.16.mlp.up_proj
Pruning layer 119/196: model.layers.16.mlp.down_proj
Pruning layer 120/196: model.layers.17.self_attn.q_proj
Pruning layer 121/196: model.layers.17.self_attn.k_proj
Pruning layer 122/196: model.layers.17.self_attn.v_proj
Pruning layer 123/196: model.layers.17.self_attn.o_proj
Pruning layer 124/196: model.layers.17.mlp.gate_proj
Pruning layer 125/196: model.layers.17.mlp.up_proj
Pruning layer 126/196: model.layers.17.mlp.down_proj
Pruning layer 127/196: model.layers.18.self_attn.q_proj
Pruning layer 128/196: model.layers.18.self_attn.k_proj
Pruning layer 129/196: model.layers.18.self_attn.v_proj
Pruning layer 130/196: model.layers.18.self_attn.o_proj
Pruning layer 131/196: model.layers.18.mlp.gate_proj
Pruning layer 132/196: model.layers.18.mlp.up_proj
Pruning layer 133/196: model.layers.18.mlp.down_proj
Pruning layer 134/196: model.layers.19.self_attn.q_proj
Pruning layer 135/196: model.layers.19.self_attn.k_proj
Pruning layer 136/196: model.layers.19.self_attn.v_proj
Pruning layer 137/196: model.layers.19.self_attn.o_proj
Pruning layer 138/196: model.layers.19.mlp.gate_proj
Pruning layer 139/196: model.layers.19.mlp.up_proj
Pruning layer 140/196: model.layers.19.mlp.down_proj
Pruning layer 141/196: model.layers.20.self_attn.q_proj
Pruning layer 142/196: model.layers.20.self_attn.k_proj
Pruning layer 143/196: model.layers.20.self_attn.v_proj
Pruning layer 144/196: model.layers.20.self_attn.o_proj
Pruning layer 145/196: model.layers.20.mlp.gate_proj
Pruning layer 146/196: model.layers.20.mlp.up_proj
Pruning layer 147/196: model.layers.20.mlp.down_proj
Pruning layer 148/196: model.layers.21.self_attn.q_proj
Pruning layer 149/196: model.layers.21.self_attn.k_proj
Pruning layer 150/196: model.layers.21.self_attn.v_proj
Pruning layer 151/196: model.layers.21.self_attn.o_proj
Pruning layer 152/196: model.layers.21.mlp.gate_proj
Pruning layer 153/196: model.layers.21.mlp.up_proj
Pruning layer 154/196: model.layers.21.mlp.down_proj
Pruning layer 155/196: model.layers.22.self_attn.q_proj
Pruning layer 156/196: model.layers.22.self_attn.k_proj
Pruning layer 157/196: model.layers.22.self_attn.v_proj
Pruning layer 158/196: model.layers.22.self_attn.o_proj
Pruning layer 159/196: model.layers.22.mlp.gate_proj
Pruning layer 160/196: model.layers.22.mlp.up_proj
Pruning layer 161/196: model.layers.22.mlp.down_proj
Pruning layer 162/196: model.layers.23.self_attn.q_proj
Pruning layer 163/196: model.layers.23.self_attn.k_proj
Pruning layer 164/196: model.layers.23.self_attn.v_proj
Pruning layer 165/196: model.layers.23.self_attn.o_proj
Pruning layer 166/196: model.layers.23.mlp.gate_proj
Pruning layer 167/196: model.layers.23.mlp.up_proj
Pruning layer 168/196: model.layers.23.mlp.down_proj
Pruning layer 169/196: model.layers.24.self_attn.q_proj
Pruning layer 170/196: model.layers.24.self_attn.k_proj
Pruning layer 171/196: model.layers.24.self_attn.v_proj
Pruning layer 172/196: model.layers.24.self_attn.o_proj
Pruning layer 173/196: model.layers.24.mlp.gate_proj
Pruning layer 174/196: model.layers.24.mlp.up_proj
Pruning layer 175/196: model.layers.24.mlp.down_proj
Pruning layer 176/196: model.layers.25.self_attn.q_proj
Pruning layer 177/196: model.layers.25.self_attn.k_proj
Pruning layer 178/196: model.layers.25.self_attn.v_proj
Pruning layer 179/196: model.layers.25.self_attn.o_proj
Pruning layer 180/196: model.layers.25.mlp.gate_proj
Pruning layer 181/196: model.layers.25.mlp.up_proj
Pruning layer 182/196: model.layers.25.mlp.down_proj
Pruning layer 183/196: model.layers.26.self_attn.q_proj
Pruning layer 184/196: model.layers.26.self_attn.k_proj
Pruning layer 185/196: model.layers.26.self_attn.v_proj
Pruning layer 186/196: model.layers.26.self_attn.o_proj
Pruning layer 187/196: model.layers.26.mlp.gate_proj
Pruning layer 188/196: model.layers.26.mlp.up_proj
Pruning layer 189/196: model.layers.26.mlp.down_proj
Pruning layer 190/196: model.layers.27.self_attn.q_proj
Pruning layer 191/196: model.layers.27.self_attn.k_proj
Pruning layer 192/196: model.layers.27.self_attn.v_proj
Pruning layer 193/196: model.layers.27.self_attn.o_proj
Pruning layer 194/196: model.layers.27.mlp.gate_proj
Pruning layer 195/196: model.layers.27.mlp.up_proj
Pruning layer 196/196: model.layers.27.mlp.down_proj
Wanda pruning completed.
Qwen3ForCausalLM(
  (model): Qwen3Model(
    (embed_tokens): Embedding(151936, 1024)
    (layers): ModuleList(
      (0-27): 28 x Qwen3DecoderLayer(
        (self_attn): Qwen3Attention(
          (q_proj): Linear(in_features=1024, out_features=2048, bias=False)
          (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
          (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
          (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
        )
        (mlp): Qwen3MLP(
          (gate_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (up_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (down_proj): Linear(in_features=3072, out_features=1024, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
        (post_attention_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
      )
    )
    (norm): Qwen3RMSNorm((1024,), eps=1e-06)
    (rotary_emb): Qwen3RotaryEmbedding()
  )
  (lm_head): Linear(in_features=1024, out_features=151936, bias=False)
)
Pruned model total parameters: 596049920
Number of pruned parameters: 220200960
Number of parameters that stayed unchanged: 375848960
Pruned model generation:
The future of AI is not just about the technology, but about the human. The human side is the key to the AI system. The human side is the key to the AI system. The human side is the key to the AI system. The human side is the key to the AI system. The human side is the key to the AI system. The human side is the key to the AI system. The human side is the key to the AI system. The human side is the key to the AI system. The
Tokens per second (pruned): 1.92
Pruned model saved to: Qwen-Qwen3-0.6B-pruned-375848960
Evaluation Pruned Results: {'wikitext_ppl': 67.41041543259905, 'hellaswag_acc': 0.395, 'boolq_acc': 0.64}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import math
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from typing import Dict, List, Optional, Union


class ListDataset(Dataset):
    def __init__(self, data_list: List[Dict]):
        self.data = data_list

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


def prepare_calib_loader(
    tokenizer,
    calib_ds,
    num_samples: int,
    batch_size: int,
    max_length: int,
    pad_token_id: int
) -> DataLoader:
    print("Preparing calibration texts...")
    texts = []
    for ex in calib_ds.take(num_samples):
        text = ex.get("text") or ex.get("sentence") or ""
        if len(text.strip()) > 10:
            texts.append(text)
    if not texts:
        raise ValueError("No valid calibration texts found.")
    num_samples = min(num_samples, len(texts))

    print(f"Tokenizing {num_samples} samples...")
    tokenized = tokenizer(
        texts[:num_samples],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None
    )

    data_list = [
        {"input_ids": tid, "attention_mask": tam}
        for tid, tam in zip(tokenized["input_ids"], tokenized["attention_mask"])
    ]

    def collate_fn(batch: List[Dict]) -> Dict:
        padded = tokenizer.pad(
            {
                "input_ids": [b["input_ids"] for b in batch],
                "attention_mask": [b["attention_mask"] for b in batch]
            },
            return_tensors="pt",
            padding=True
        )
        return padded

    return DataLoader(
        ListDataset(data_list),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )


def collect_activation_stats(
    model: nn.Module,
    loader: DataLoader,
    device: str
) -> tuple[Dict[str, torch.Tensor], Dict[str, nn.Linear]]:
    print("Identifying linear layers...")
    linears = {
        name: mod for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and mod.in_features > 0 and "lm_head" not in name
    }
    if not linears:
        raise ValueError("No linear layers found in the model.")

    accum_sq: Dict[str, torch.Tensor] = {
        name: torch.zeros(mod.in_features, device=device, dtype=torch.float32)
        for name, mod in linears.items()
    }

    def make_hook(nm: str):
        def hook(_: nn.Module, inputs: tuple) -> None:
            act = inputs[0].view(-1, inputs[0].size(-1))
            sq = (act ** 2).sum(dim=0).to(dtype=torch.float32)
            accum_sq[nm] += sq
        return hook

    handles = []
    for name, mod in linears.items():
        h = mod.register_forward_pre_hook(make_hook(name))
        handles.append(h)

    model.eval()
    total_batches = len(loader)
    print(f"Collecting activation stats over {total_batches} batches...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            print(f"Processing batch {i+1}/{total_batches}")
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(**batch)

    for h in handles:
        h.remove()

    print("Computing norms...")
    norms = {}
    for name, sq in accum_sq.items():
        norm = torch.sqrt(sq)
        norm[norm == 0] = 1e-8
        norms[name] = norm.to(linears[name].weight.dtype)

    return norms, linears


def apply_pruning(
    linears: Dict[str, nn.Linear],
    norms: Dict[str, torch.Tensor],
    sparsity_ratio: float,
    sparsity_type: str
) -> None:
    if not (0 <= sparsity_ratio <= 1):
        raise ValueError("sparsity_ratio must be in [0, 1]")

    total_layers = len(linears)
    print(f"Applying pruning to {total_layers} layers...")
    current_layer = 0
    for name, mod in linears.items():
        current_layer += 1
        print(f"Pruning layer {current_layer}/{total_layers}: {name}")
        W = mod.weight.data
        norm = norms[name]
        metric = W.abs() * norm.unsqueeze(0)  # (out, in)

        out_f, in_f = W.shape

        if sparsity_type == "unstructured":
            n_prune = math.ceil(sparsity_ratio * in_f)
            if n_prune == 0:
                continue
            for i in range(out_f):
                _, idx = torch.sort(metric[i])
                W[i, idx[:n_prune]] = 0
        else:
            # N:M format
            if ":" not in sparsity_type:
                raise ValueError("sparsity_type for structured must be 'N:M'")
            _, M = map(int, sparsity_type.split(":"))
            if M <= 1:
                raise ValueError("M in N:M must be >1")

            for i in range(out_f):
                pos = 0
                while pos < in_f:
                    end = min(pos + M, in_f)
                    sl_len = end - pos
                    n_prune_sl = math.ceil(sparsity_ratio * sl_len)
                    if n_prune_sl == 0:
                        pos = end
                        continue
                    sl_metric = metric[i, pos:end]
                    _, idx = torch.sort(sl_metric)
                    W[i, pos + idx[:n_prune_sl]] = 0
                    pos = end


def prune_wanda(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sparsity_ratio: float = 0.5,
    sparsity_type: str = "unstructured",
    calib_dataset: str = "allenai/c4",
    num_calib_samples: int = 128,
    batch_size_calib: int = 4,
    max_length: Optional[int] = None,
    device: Union[str, torch.device] = None
) -> AutoModelForCausalLM:
    print("Starting Wanda pruning...")
    print("Step 1/5: Setting up device and model")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device)

    if max_length is None:
        max_length = getattr(tokenizer, "model_max_length", 2048)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Step 2/5: Loading calibration dataset")
    config_name = "en" if "c4" in calib_dataset.lower() else None
    calib_ds = load_dataset(calib_dataset, config_name, streaming=True, split="train")

    print("Step 3/5: Preparing data loader")
    loader = prepare_calib_loader(
        tokenizer, calib_ds, num_calib_samples, batch_size_calib,
        max_length, tokenizer.pad_token_id
    )

    print("Step 4/5: Collecting activation statistics")
    norms, linears = collect_activation_stats(model, loader, str(device))

    print("Step 5/5: Applying pruning")
    apply_pruning(linears, norms, sparsity_ratio, sparsity_type)

    print("Wanda pruning completed.")
    return model


def compute_ppl(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "validation",
    num_samples: int = 1000,
    batch_size: int = 8,
    max_length: int = 512,
    device: Union[str, torch.device] = None
) -> float:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ds = load_dataset(dataset_name, dataset_config, streaming=True, split=split)
    texts = []
    for ex in ds.take(num_samples):
        txt = ex.get("text", "")
        if len(txt.strip()) > 10:
            texts.append(txt)
    if not texts:
        raise ValueError("No valid texts for PPL")

    tokenized = []
    for txt in texts:
        enc = tokenizer(txt, truncation=True, max_length=max_length, padding=False, return_tensors=None)
        tokenized.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})

    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        padded = tokenizer.pad(
            {"input_ids": [b["input_ids"] for b in batch], "attention_mask": [b["attention_mask"] for b in batch]},
            return_tensors="pt",
            padding=True
        )
        labels = padded["input_ids"].clone()
        labels[~padded["attention_mask"].bool()] = -100
        return {"input_ids": padded["input_ids"], "attention_mask": padded["attention_mask"], "labels": labels}

    dl = DataLoader(ListDataset(tokenized), batch_size=batch_size, shuffle=False, collate_fn=collate)

    model.eval()
    total_loss = total_tokens = 0.0
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            valid = batch["attention_mask"].sum().item()
            total_loss += loss.item() * valid
            total_tokens += valid

    if total_tokens == 0:
        raise ValueError("No valid tokens")
    return math.exp(total_loss / total_tokens)


def compute_accuracy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    task: str,
    num_samples: int = 1000,
    max_length: int = 512,
    device: Union[str, torch.device] = None
) -> float:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        if task == "hellaswag":
            ds = load_dataset("hellaswag", streaming=True, split="validation")
            for ex in ds.take(num_samples):
                ctx = ex["ctx"]
                endings = ex["endings"]
                label = int(ex["label"])
                prompt_tok = tokenizer(ctx, return_tensors="pt", truncation=True, max_length=max_length)
                len_prompt = prompt_tok["input_ids"].shape[1]

                min_loss = float("inf")
                best = -1
                for i, end in enumerate(endings):
                    full = ctx + " " + end
                    full_tok = tokenizer(full, return_tensors="pt", truncation=True, max_length=max_length)
                    ids = full_tok["input_ids"].to(device)
                    lbl = ids.clone()
                    lbl[0, :len_prompt] = -100
                    loss = model(input_ids=ids, labels=lbl).loss.item()
                    if loss < min_loss:
                        min_loss, best = loss, i
                if best == label:
                    correct += 1
                total += 1

        elif task == "boolq":
            ds = load_dataset("boolq", streaming=True, split="validation")
            for ex in ds.take(num_samples):
                passage = ex["passage"]
                question = ex["question"]
                answer = ex["answer"]
                prompt = f"passage: {passage}\nquestion: {question}\nanswer:"
                prompt_tok = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
                len_prompt = prompt_tok["input_ids"].shape[1]
                options = [" Yes", " No"]
                min_loss = float("inf")
                best = False
                for opt, tgt in zip(options, [True, False]):
                    full = prompt + opt
                    full_tok = tokenizer(full, return_tensors="pt", truncation=True, max_length=max_length)
                    ids = full_tok["input_ids"].to(device)
                    lbl = ids.clone()
                    lbl[0, :len_prompt] = -100
                    loss = model(input_ids=ids, labels=lbl).loss.item()
                    if loss < min_loss:
                        min_loss, best = loss, tgt
                if best == answer:
                    correct += 1
                total += 1
        else:
            raise ValueError(f"Unsupported task: {task}")

    return correct / total if total > 0 else 0.0


def evaluate_llm(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_config: Dict,
    device: Union[str, torch.device] = None
) -> Dict[str, float]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    results: Dict[str, float] = {}

    if eval_config.get("ppl"):
        cfg = eval_config["ppl"]
        dataset = cfg.get("dataset", "wikitext")
        config = cfg.get("config", "wikitext-2-raw-v1")
        split = cfg.get("split", "validation")
        ns = cfg.get("num_samples", 1000)
        results[f"{dataset}_ppl"] = compute_ppl(
            model, tokenizer, dataset, config, split, ns, device=device
        )

    if eval_config.get("tasks"):
        for task in eval_config["tasks"]:
            ns = eval_config.get("num_samples_per_task", 1000)
            results[f"{task}_acc"] = compute_accuracy(model, tokenizer, task, ns, device=device)

    return results


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_nonzero_parameters(model: nn.Module) -> int:
    return sum((p != 0).sum().item() for p in model.parameters())


def generate_text_token_by_token(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    device: torch.device = None
) -> tuple[str, float]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    start_time = time.time()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

    end_time = time.time()
    time_taken = end_time - start_time
    tokens_per_second = max_new_tokens / time_taken if time_taken > 0 else 0

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text, tokens_per_second


# -----------------------------
# Example / Demo
# -----------------------------
if __name__ == "__main__":
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(model)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model.to(device)

    total_params_orig = count_parameters(model)
    print(f"Original model total parameters: {total_params_orig}")

    prompt = "The future of AI is"
    generated_orig, tps_orig = generate_text_token_by_token(model, tokenizer, prompt, max_new_tokens=100, device=device)
    print("Original model generation:")
    print(generated_orig)
    print(f"Tokens per second (original): {tps_orig:.2f}")


    # Evaluate original
    eval_cfg = {
        "ppl": {"dataset": "wikitext", "config": "wikitext-2-raw-v1", "num_samples": 500},
        "tasks": ["hellaswag", "boolq"],
        "num_samples_per_task": 200
    }
    metrics_orig = evaluate_llm(model, tokenizer, eval_cfg, device=device)
    print("Evaluation Original Results:", metrics_orig)

    print('=' * 60)

    # Prune using Wanda
    pruned = prune_wanda(
        model,
        tokenizer,
        sparsity_ratio=0.5,
        sparsity_type="unstructured",  # or "2:4"
        num_calib_samples=128,
        device=device
    )
    print(pruned)

    total_params_pruned = count_parameters(pruned)
    nonzero_params = count_nonzero_parameters(pruned)
    pruned_params = total_params_pruned - nonzero_params
    unchanged_params = nonzero_params

    print(f"Pruned model total parameters: {total_params_pruned}")
    print(f"Number of pruned parameters: {pruned_params}")
    print(f"Number of parameters that stayed unchanged: {unchanged_params}")

    generated_pruned, tps_pruned = generate_text_token_by_token(pruned, tokenizer, prompt, max_new_tokens=100, device=device)
    print("Pruned model generation:")
    print(generated_pruned)
    print(f"Tokens per second (pruned): {tps_pruned:.2f}")

    # Save pruned model
    save_dir = model_name.replace('/', '-') + "-pruned-" + str(nonzero_params)
    pruned.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Pruned model saved to: {save_dir}")

    # Evaluate pruned
    metrics_pruned = evaluate_llm(pruned, tokenizer, eval_cfg, device=device)
    print("Evaluation Pruned Results:", metrics_pruned)
