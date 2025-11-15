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
import random
from datasets import load_dataset

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from common import count_parameters, format_parameter_count, get_model_size_suffix


# Dataset loading functions adapted from Torch-Pruning examples
def get_calibration_data(nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    """Get calibration data from a datasets for realistic pruning."""
    if tokenizer is None:
        raise ValueError("Tokenizer is required for calibration data")

    print("Loading calibration datasets...")
    random.seed(seed)

    all_samples = []

    # Use C4 dataset with multiple languages for diverse calibration data
    # Limit to 128 samples per language to avoid memory issues
    languages = ["en", "es", "fr"]
    # samples_per_language = nsamples

    for lang in languages:
        print(f"Loading C4 dataset for language: {lang}")
        # Load C4 dataset for this language
        dataset = load_dataset("allenai/c4", lang, split="train", streaming=True)
        dataset_iterator = iter(dataset)

        samples_from_lang = 0
        # max_samples_from_lang = min(
        #     samples_per_language, nsamples - len(all_samples)
        # )
        max_samples_from_lang = 64

        for _ in range(max_samples_from_lang):
            try:
                sample = next(dataset_iterator)
                # Extract text field
                try:
                    text = sample["text"]
                except (KeyError, TypeError):
                    text = str(sample)

                # Skip empty or very short texts
                if not text or len(text.strip()) < 2048:
                    continue

                # Tokenize the text
                tokens = tokenizer.encode(text, add_special_tokens=False)

                # Filter out any None values (safety check)
                tokens = [t for t in tokens if t is not None]

                # Limit tokens to first seqlen tokens
                tokens = tokens[:seqlen]

                # Pad if shorter than seqlen
                if len(tokens) < seqlen:
                    input_ids = tokens + [tokenizer.pad_token_id] * (
                        seqlen - len(tokens)
                    )
                else:
                    input_ids = tokens

                # Create target_ids (next token prediction)
                target_ids = input_ids[1:] + [tokenizer.eos_token_id]

                # Convert to tensors
                input_tensor = torch.tensor(input_ids, dtype=torch.long)
                target_tensor = torch.tensor(target_ids, dtype=torch.long)

                all_samples.append((input_tensor, target_tensor))
                samples_from_lang += 1
            except StopIteration:
                # No more samples in this language
                break
            except Exception:
                # Skip problematic samples
                continue

        print(f"Collected {samples_from_lang} samples from {lang}")

    print(f"Total collected calibration dataset: {len(all_samples)} samples")
    random.shuffle(all_samples)
    all_samples = all_samples[:nsamples]
    print(f"Final used calibration dataset: {len(all_samples)} samples")
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
    Downscale a model using structural pruning with Wanda importance scoring.

    Args:
        model_path: Path or HuggingFace model identifier
        output_path: Optional output path for the downscaled model
        pruning_ratio: Ratio of hidden dimensions to keep (0.0 to 1.0)
        max_seq_len: Maximum sequence length for the model

    Returns:
        Tuple of (pruned_model, output_path)
    """
    print("=== PRUNING PROCESS STARTED ===")
    print(f"Step 1/10: Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Step 2/10: Setting up device and model configuration")
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

    print("Step 3/10: Calculating pruning dimensions")
    # Calculate dimensions to keep
    original_hidden_size = model.config.hidden_size
    keep_hidden_size = int(original_hidden_size * pruning_ratio)
    keep_hidden_size = max(keep_hidden_size, 8)  # Minimum size for stability

    print(f"Original hidden size: {original_hidden_size}")
    print(f"Pruned hidden size: {keep_hidden_size}")

    print("Step 4/10: Calculating attention dimensions")
    # Calculate new attention dimensions
    # Get head_dim from the actual layer dimensions
    sample_layer = model.model.layers[0]
    if hasattr(sample_layer.self_attn, "q_proj"):
        original_q_head_dim = (
            sample_layer.self_attn.q_proj.out_features
            // model.config.num_attention_heads
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

    print("Step 5/10: Computing Wanda importance scores")
    # Use Wanda to compute importance scores
    print("Computing Wanda importance scores...")

    # Get calibration data
    nsamples = 128
    seqlen = model.seqlen  # Use model's sequence length for calibration
    calibration_data = get_calibration_data(
        nsamples=nsamples, seqlen=seqlen, tokenizer=tokenizer
    )

    # Initialize importance scores for each hidden dimension
    importance_scores = torch.zeros(original_hidden_size, device=device)

    # Hook to capture activations
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = input[0].detach()  # input[0] is the hidden states

        return hook

    # Register hooks for key layers
    hooks = []
    for i, layer in enumerate(model.model.layers):
        # Hook the input to each layer (after input_layernorm)
        hook = layer.register_forward_hook(get_activation(f"layer_{i}_input"))
        hooks.append(hook)

    print("Step 6/10: Running calibration and computing importance scores")
    # Run calibration
    print(f"Running calibration on {len(calibration_data)} samples...")
    with torch.no_grad():
        for sample_idx, (inp, tar) in enumerate(calibration_data):
            if sample_idx % 10 == 0:
                print(f"  Processing sample {sample_idx + 1}/{len(calibration_data)}")
            # Add batch dimension
            inp = inp.unsqueeze(0).to(device)
            tar = tar.unsqueeze(0).to(device)
            _ = model(inp)

            # For each layer, compute Wanda scores for input projections
            for i, layer in enumerate(model.model.layers):
                if f"layer_{i}_input" in activations:
                    x = activations[f"layer_{i}_input"]  # [batch, seq, hidden]
                    # Average over batch and sequence dimensions
                    x_mean = x.abs().mean(dim=[0, 1])  # [hidden]

                    # Compute importance for different projections
                    if hasattr(layer.self_attn, "q_proj"):
                        # Q projection importance: |W_q| * |x|
                        w_q = layer.self_attn.q_proj.weight.data.abs()  # [out, in]
                        importance_q = (w_q * x_mean.unsqueeze(0)).sum(dim=0)  # [in]
                        importance_scores += importance_q

                        # K and V projections
                        w_k = layer.self_attn.k_proj.weight.data.abs()
                        importance_k = (w_k * x_mean.unsqueeze(0)).sum(dim=0)
                        importance_scores += importance_k

                        w_v = layer.self_attn.v_proj.weight.data.abs()
                        importance_v = (w_v * x_mean.unsqueeze(0)).sum(dim=0)
                        importance_scores += importance_v

                    # MLP projections
                    if hasattr(layer.mlp, "gate_proj"):
                        w_gate = layer.mlp.gate_proj.weight.data.abs()
                        importance_gate = (w_gate * x_mean.unsqueeze(0)).sum(dim=0)
                        importance_scores += importance_gate

                        w_up = layer.mlp.up_proj.weight.data.abs()
                        importance_up = (w_up * x_mean.unsqueeze(0)).sum(dim=0)
                        importance_scores += importance_up

                    elif hasattr(layer.mlp, "gate_up_proj"):
                        w_gate_up = layer.mlp.gate_up_proj.weight.data.abs()
                        importance_gate_up = (w_gate_up * x_mean.unsqueeze(0)).sum(
                            dim=0
                        )
                        importance_scores += importance_gate_up

            activations.clear()  # Clear for next sample

    # Remove hooks
    for hook in hooks:
        hook.remove()

    print("Step 7/10: Selecting dimensions to keep")
    # Select dimensions to keep based on Wanda importance scores
    _, keep_indices = torch.topk(importance_scores, keep_hidden_size)

    # Sort indices for consistency
    keep_indices = torch.sort(keep_indices)[0]

    print(f"Keeping dimensions: {keep_indices[:10].tolist()}... (showing first 10)")

    print("Step 8/10: Pruning embedding layer")
    # Prune embedding layer
    print("Pruning embedding layer...")
    new_embed = nn.Embedding(
        model.config.vocab_size,
        keep_hidden_size,
        padding_idx=model.model.embed_tokens.padding_idx,
    )
    new_embed.weight.data = model.model.embed_tokens.weight.data[:, keep_indices]
    model.model.embed_tokens = new_embed

    print("Step 9/10: Pruning decoder layers")
    # Prune each decoder layer
    print("Pruning decoder layers...")
    total_layers = len(model.model.layers)
    for i, layer in enumerate(model.model.layers):
        print(f"  Pruning layer {i + 1}/{total_layers}")

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

    print("Step 10/10: Pruning final components")
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
    print("=== PRUNING PROCESS COMPLETED ===")

    return model, output_path
