import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import math
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


# -----------------------------
# Example / Demo
# -----------------------------
if __name__ == "__main__":
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print('=' * 60)
    print(model)

    # Evaluate (Wanda-style: PPL on WikiText + zero-shot tasks)
    eval_cfg = {
        "ppl": {"dataset": "wikitext", "config": "wikitext-2-raw-v1", "num_samples": 500},
        "tasks": ["hellaswag", "boolq"],
        "num_samples_per_task": 200
    }
    metrics = evaluate_llm(model, tokenizer, eval_cfg)
    print("Evaluation Original Results:", metrics)

    # Prune using Wanda
    pruned = prune_wanda(
        model,
        tokenizer,
        sparsity_ratio=0.5,
        sparsity_type="unstructured",  # or "2:4"
        num_calib_samples=128,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print('=' * 60)
    print(pruned)

    # Evaluate (Wanda-style: PPL on WikiText + zero-shot tasks)
    eval_cfg = {
        "ppl": {"dataset": "wikitext", "config": "wikitext-2-raw-v1", "num_samples": 500},
        "tasks": ["hellaswag", "boolq"],
        "num_samples_per_task": 200
    }
    metrics = evaluate_llm(pruned, tokenizer, eval_cfg)
    print("Evaluation Pruned Results:", metrics)
