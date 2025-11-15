"""
Downscaling logic for language models using structural pruning.
Based on the DepGraph method concepts from CVPR 2023.
"""

import os
import sys
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple
import math
import random
from datasets import load_dataset

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from common import count_parameters, format_parameter_count, get_model_size_suffix


# Dataset loading functions adapted from Torch-Pruning examples
def get_calibration_data(nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    """Get calibration data from multiple datasets for realistic pruning."""
    if tokenizer is None:
        raise ValueError("Tokenizer is required for calibration data")

    print("Loading calibration datasets...")

    all_samples = []
    random.seed(seed)

    # Try to load different datasets for diverse calibration
    datasets_tried = []

    # 1. Try WikiText-2
    print("  Loading WikiText-2...")
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    wikitext_samples = []
    for sample in traindata:  # type: ignore
        if len(wikitext_samples) >= nsamples // 4:
            break
        text = sample["text"]  # type: ignore
        if len(text.strip()) > 100:  # Only use substantial texts
            wikitext_samples.append(text)

    for text in wikitext_samples:
        trainenc = tokenizer(text, return_tensors="pt")  # type: ignore
        if trainenc.input_ids.shape[1] >= seqlen:
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            all_samples.append((inp, tar))

    datasets_tried.append(f"WikiText-2 ({len(wikitext_samples)} samples)")

    # 2. Try C4 dataset
    print("  Loading C4...")
    c4_samples = []
    # Use a smaller subset to avoid loading too much data
    traindata = load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    for i, sample in enumerate(traindata):
        if len(c4_samples) >= nsamples // 4:
            break
        if i >= 1000:  # Limit iterations
            break
        text = sample["text"]
        if len(text.strip()) > 200:  # Only use substantial texts
            c4_samples.append(text)

    for text in c4_samples:
        trainenc = tokenizer(text, return_tensors="pt")  # type: ignore
        if trainenc.input_ids.shape[1] >= seqlen:
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            all_samples.append((inp, tar))

    datasets_tried.append(f"C4 ({len(c4_samples)} samples)")

    # 3. Math/Code samples (synthetic for now)
    print("  Generating math/code samples...")
    math_code_texts = [
        "Solve the equation: ∫x² dx = x³/3 + C",
        "def quicksort(arr): if len(arr) <= 1: return arr; pivot = arr[0]; left = [x for x in arr[1:] if x <= pivot]; right = [x for x in arr[1:] if x > pivot]; return quicksort(left) + [pivot] + quicksort(right)",
        "The derivative of sin(x) is cos(x).",
        "class BinaryTree: def __init__(self, value): self.value = value; self.left = None; self.right = None",
        "Theorem: For any triangle with sides a, b, c, a² + b² = c² for right triangles.",
        "import torch; import torch.nn as nn; class Model(nn.Module): def __init__(self): super().__init__(); self.linear = nn.Linear(10, 1)",
        "The fundamental group of the circle is isomorphic to ℤ.",
        "function fibonacci(n) { if (n <= 1) return n; return fibonacci(n-1) + fibonacci(n-2); }",
        "Matrix multiplication: C[i][j] = Σ(A[i][k] * B[k][j])",
        "The time complexity of merge sort is O(n log n).",
    ]

    for text in math_code_texts:
        trainenc = tokenizer(text, return_tensors="pt")  # type: ignore
        if trainenc.input_ids.shape[1] >= seqlen:
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            all_samples.append((inp, tar))

    # 4. Fill remaining samples with diverse text
    diverse_texts = [
        "The history of artificial intelligence began in the 1950s with the development of the first neural networks.",
        "Climate change is one of the most pressing challenges facing humanity in the 21st century.",
        "The human brain contains approximately 86 billion neurons, each connected to thousands of others.",
        "Quantum computing has the potential to revolutionize cryptography and drug discovery.",
        "The Renaissance was a period of cultural, artistic, political and economic rebirth in Europe.",
        "Machine learning algorithms can be supervised, unsupervised, or reinforcement learning based.",
        "The universe is approximately 13.8 billion years old according to current cosmological models.",
        "Natural language processing combines linguistics, computer science, and artificial intelligence.",
        "The Industrial Revolution transformed societies from agrarian to industrial economies.",
        "Blockchain technology enables decentralized and transparent digital transactions.",
    ]

    while len(all_samples) < nsamples:
        text = random.choice(diverse_texts)
        trainenc = tokenizer(text, return_tensors="pt")  # type: ignore
        if trainenc.input_ids.shape[1] >= seqlen:
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            all_samples.append((inp, tar))

    datasets_tried.append(f"Math/Code ({len(math_code_texts)} samples)")

    # 4. Fill remaining samples with diverse text
    diverse_texts = [
        "The history of artificial intelligence began in the 1950s with the development of the first neural networks.",
        "Climate change is one of the most pressing challenges facing humanity in the 21st century.",
        "The human brain contains approximately 86 billion neurons, each connected to thousands of others.",
        "Quantum computing has the potential to revolutionize cryptography and drug discovery.",
        "The Renaissance was a period of cultural, artistic, political and economic rebirth in Europe.",
        "Machine learning algorithms can be supervised, unsupervised, or reinforcement learning based.",
        "The universe is approximately 13.8 billion years old according to current cosmological models.",
        "Natural language processing combines linguistics, computer science, and artificial intelligence.",
        "The Industrial Revolution transformed societies from agrarian to industrial economies.",
        "Blockchain technology enables decentralized and transparent digital transactions.",
    ]

    while len(all_samples) < nsamples:
        text = random.choice(diverse_texts)
        trainenc = tokenizer(text, return_tensors="pt")
        if trainenc.input_ids.shape[1] >= seqlen:
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            all_samples.append((inp, tar))

    print(f"  Datasets used: {', '.join(datasets_tried)}")
    print(f"  Total calibration samples: {len(all_samples)}")
    return all_samples


def prune_linear_layer(layer: nn.Linear, keep_indices: torch.Tensor) -> nn.Linear:
    """Prune a linear layer by keeping only specified output dimensions."""
    new_out_features = len(keep_indices)
    new_weight = layer.weight.data[keep_indices]
    new_bias = layer.bias.data[keep_indices] if layer.bias is not None else None

    new_layer = nn.Linear(
        layer.in_features, new_out_features, bias=layer.bias is not None
    )
    new_layer.weight.data = new_weight
    if new_bias is not None:
        new_layer.bias.data = new_bias

    return new_layer


def prune_linear_layer_input(layer: nn.Linear, keep_indices: torch.Tensor) -> nn.Linear:
    """Prune a linear layer by keeping only specified input dimensions."""
    new_in_features = len(keep_indices)
    new_weight = layer.weight.data[:, keep_indices]
    new_bias = layer.bias.data if layer.bias is not None else None

    new_layer = nn.Linear(
        new_in_features, layer.out_features, bias=layer.bias is not None
    )
    new_layer.weight.data = new_weight
    if new_bias is not None:
        new_layer.bias.data = new_bias

    return new_layer


def prune_layer_norm(layer_norm, keep_indices: torch.Tensor):
    """Prune a layer norm by keeping only specified dimensions."""
    # Handle different layer norm types (LayerNorm, RMSNorm, etc.)
    if hasattr(layer_norm, "weight") and layer_norm.weight is not None:
        new_weight = layer_norm.weight.data[keep_indices]
    else:
        new_weight = None

    if hasattr(layer_norm, "bias") and layer_norm.bias is not None:
        new_bias = layer_norm.bias.data[keep_indices]
    else:
        new_bias = None

    # Create new layer of the same type
    layer_type = type(layer_norm)
    if layer_type == nn.LayerNorm:
        new_normalized_shape = len(keep_indices)
        new_layer = layer_type(
            new_normalized_shape,
            eps=layer_norm.eps,
            elementwise_affine=layer_norm.elementwise_affine,
        )
    else:
        # For RMSNorm and similar, create with the new size
        new_layer = layer_type(len(keep_indices), eps=getattr(layer_norm, "eps", 1e-5))

    if new_weight is not None:
        new_layer.weight.data = new_weight
    if new_bias is not None and hasattr(new_layer, "bias"):
        new_layer.bias.data = new_bias

    return new_layer


def downscale_model(
    model_path: str,
    output_path: Optional[str] = None,
    pruning_ratio: float = 0.5,
    max_seq_len: int = 4096,
) -> Tuple[torch.nn.Module, str]:
    """
    Downscale a model using structural pruning.

    Args:
        model_path: Path or HuggingFace model identifier
        output_path: Optional output path for the downscaled model
        pruning_ratio: Ratio of hidden dimensions to keep (0.0 to 1.0)
        max_seq_len: Maximum sequence length for the model

    Returns:
        Tuple of (pruned_model, output_path)
    """
    print(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, low_cpu_mem_usage=True
    )

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = model.to(device)  # type: ignore
    model.eval()

    # Set sequence length attribute like in Torch-Pruning
    model.seqlen = min(max_seq_len, model.config.max_position_embeddings)  # type: ignore

    print(f"Using device: {device}")
    print(f"Pruning ratio: {pruning_ratio}")
    print(f"Max sequence length: {max_seq_len}")

    # Calculate dimensions to keep
    original_hidden_size = model.config.hidden_size
    keep_hidden_size = int(original_hidden_size * pruning_ratio)
    keep_hidden_size = max(keep_hidden_size, 8)  # Minimum size for stability

    print(f"Original hidden size: {original_hidden_size}")
    print(f"Pruned hidden size: {keep_hidden_size}")

    # Calculate new attention dimensions
    # Get head_dim from the actual layer dimensions
    sample_layer = model.model.layers[0]
    if hasattr(sample_layer.self_attn, "q_proj"):
        original_q_head_dim = (
            sample_layer.self_attn.q_proj.out_features
            // model.config.num_attention_heads
        )
        original_kv_head_dim = (
            sample_layer.self_attn.k_proj.out_features
            // model.config.num_key_value_heads
        )
        # Assume head_dim is the same for q and kv
        original_head_dim = original_q_head_dim
    else:
        # Fallback
        original_head_dim = 128  # Common default
    new_num_attention_heads = max(
        1, model.config.num_attention_heads * keep_hidden_size // original_hidden_size
    )
    new_num_key_value_heads = max(
        1, model.config.num_key_value_heads * keep_hidden_size // original_hidden_size
    )

    print(f"Original num_attention_heads: {model.config.num_attention_heads}")
    print(f"New num_attention_heads: {new_num_attention_heads}")
    print(f"Original num_key_value_heads: {model.config.num_key_value_heads}")
    print(f"New num_key_value_heads: {new_num_key_value_heads}")
    print(f"Head dim: {original_head_dim}")

    # Select dimensions to keep based on weight magnitude
    embed_weight = model.model.embed_tokens.weight.data
    # Compute importance scores for each dimension
    importance_scores = torch.norm(embed_weight, dim=0)
    _, keep_indices = torch.topk(importance_scores, keep_hidden_size)

    # Sort indices for consistency
    keep_indices = torch.sort(keep_indices)[0]

    print(f"Keeping dimensions: {keep_indices[:10].tolist()}... (showing first 10)")

    # Prune embedding layer
    print("Pruning embedding layer...")
    new_embed = nn.Embedding(
        model.config.vocab_size,
        keep_hidden_size,
        padding_idx=model.model.embed_tokens.padding_idx,
    )
    new_embed.weight.data = model.model.embed_tokens.weight.data[:, keep_indices]
    model.model.embed_tokens = new_embed

    # Prune each decoder layer
    print("Pruning decoder layers...")
    for i, layer in enumerate(model.model.layers):
        print(f"  Pruning layer {i + 1}/{len(model.model.layers)}")

        # Prune attention layers
        if hasattr(layer.self_attn, "q_proj"):
            # Separate QKV projections
            # First prune input dimensions
            layer.self_attn.q_proj = prune_linear_layer_input(
                layer.self_attn.q_proj, keep_indices
            )
            layer.self_attn.k_proj = prune_linear_layer_input(
                layer.self_attn.k_proj, keep_indices
            )
            layer.self_attn.v_proj = prune_linear_layer_input(
                layer.self_attn.v_proj, keep_indices
            )

            # Prune output dimensions proportionally
            q_keep_size = new_num_attention_heads * original_head_dim
            kv_keep_size = new_num_key_value_heads * original_head_dim

            q_keep_indices = torch.arange(q_keep_size)
            kv_keep_indices = torch.arange(kv_keep_size)

            layer.self_attn.q_proj = prune_linear_layer(
                layer.self_attn.q_proj, q_keep_indices
            )
            layer.self_attn.k_proj = prune_linear_layer(
                layer.self_attn.k_proj, kv_keep_indices
            )
            layer.self_attn.v_proj = prune_linear_layer(
                layer.self_attn.v_proj, kv_keep_indices
            )

            # Update o_proj (input from q_proj output, output to hidden)
            o_input_keep_indices = torch.arange(q_keep_size)
            layer.self_attn.o_proj = prune_linear_layer_input(
                layer.self_attn.o_proj, o_input_keep_indices
            )
            layer.self_attn.o_proj = prune_linear_layer(
                layer.self_attn.o_proj, keep_indices
            )

            # Prune attention normalization layers if they exist (Qwen3 specific)
            if hasattr(layer.self_attn, "q_norm"):
                q_norm_keep_indices = torch.arange(original_head_dim)
                layer.self_attn.q_norm = prune_layer_norm(
                    layer.self_attn.q_norm, q_norm_keep_indices
                )
            if hasattr(layer.self_attn, "k_norm"):
                k_norm_keep_indices = torch.arange(original_head_dim)
                layer.self_attn.k_norm = prune_layer_norm(
                    layer.self_attn.k_norm, k_norm_keep_indices
                )

        elif hasattr(layer.self_attn, "qkv_proj"):
            # Combined QKV projection
            layer.self_attn.qkv_proj = prune_linear_layer_input(
                layer.self_attn.qkv_proj, keep_indices
            )

        # Prune MLP layers
        if hasattr(layer.mlp, "gate_proj"):
            # Separate gate/up/down projections
            layer.mlp.gate_proj = prune_linear_layer_input(
                layer.mlp.gate_proj, keep_indices
            )
            layer.mlp.up_proj = prune_linear_layer_input(
                layer.mlp.up_proj, keep_indices
            )
            layer.mlp.down_proj = prune_linear_layer(layer.mlp.down_proj, keep_indices)

        elif hasattr(layer.mlp, "gate_up_proj"):
            # Combined gate+up projection
            layer.mlp.gate_up_proj = prune_linear_layer_input(
                layer.mlp.gate_up_proj, keep_indices
            )
            layer.mlp.down_proj = prune_linear_layer(layer.mlp.down_proj, keep_indices)

        elif hasattr(layer.mlp, "gate_up_proj"):
            # Combined gate+up projection
            layer.mlp.gate_up_proj = prune_linear_layer_input(
                layer.mlp.gate_up_proj, keep_indices
            )

            # Scale intermediate dimensions
            intermediate_keep_ratio = keep_hidden_size / original_hidden_size
            intermediate_keep_size = int(
                layer.mlp.gate_up_proj.out_features * intermediate_keep_ratio // 2
            )  # /2 because it's gate+up combined

            # Prune intermediate dimensions
            gate_up_keep_indices = torch.arange(intermediate_keep_size * 2)
            layer.mlp.gate_up_proj = prune_linear_layer(
                layer.mlp.gate_up_proj, gate_up_keep_indices
            )
            layer.mlp.down_proj = prune_linear_layer_input(
                layer.mlp.down_proj, gate_up_keep_indices
            )
            layer.mlp.down_proj = prune_linear_layer(layer.mlp.down_proj, keep_indices)
        elif hasattr(layer.mlp, "gate_up_proj"):
            # Combined gate+up projection
            layer.mlp.gate_up_proj = prune_linear_layer_input(
                layer.mlp.gate_up_proj, keep_indices
            )
            layer.mlp.down_proj = prune_linear_layer(layer.mlp.down_proj, keep_indices)

        # Prune layer norms
        layer.input_layernorm = prune_layer_norm(layer.input_layernorm, keep_indices)
        layer.post_attention_layernorm = prune_layer_norm(
            layer.post_attention_layernorm, keep_indices
        )

    # Prune final layer norm
    print("Pruning final layer norm...")
    model.model.norm = prune_layer_norm(model.model.norm, keep_indices)

    # Prune LM head
    print("Pruning LM head...")
    model.lm_head = prune_linear_layer_input(model.lm_head, keep_indices)

    # Update model configuration
    model.config.hidden_size = keep_hidden_size
    model.config.num_attention_heads = new_num_attention_heads
    model.config.num_key_value_heads = new_num_key_value_heads

    # Note: intermediate_size remains unchanged since we're not pruning intermediate dimensions

    # Generate output path if not provided
    if output_path is None:
        model_name = model_path.split("/")[-1]
        pruned_size = get_model_size_suffix(count_parameters(model))
        output_path = f"{model_name}-{pruned_size}"

    print(
        f"Pruning completed. New model size: {format_parameter_count(count_parameters(model))}"
    )

    return model, output_path
