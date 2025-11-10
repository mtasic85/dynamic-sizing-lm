"""
Downscaling logic for language models using Model Folding method.
Purely mathematical, fine-tuning-free parameter reduction via clustering and merging.
"""

import math
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM


def kmeans_torch(X, K, max_iter=100, device="cpu"):
    """Simple k-means implementation using PyTorch."""
    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(X)
    X = X.to(device)
    N, D = X.shape
    # Random init centers
    indices = torch.randperm(N, device=device)[:K]
    centers = X[indices].clone()
    for _ in range(max_iter):
        # Distances
        dist = torch.cdist(X, centers)  # [N, K]
        labels = dist.argmin(dim=1)
        # Update centers
        for k in range(K):
            mask = labels == k
            if mask.any():
                centers[k] = X[mask].mean(dim=0)
    return labels, centers


def downscale_model(
    model_path: str, output_path: Optional[str] = None, sparsity: float = 0.5
) -> Tuple[AutoModelForCausalLM, str]:
    """
    Downscale a model using Model Folding method.

    Args:
        model_path: Path to input model
        output_path: Output path for downscaled model
        sparsity: Target sparsity (0.5 means halve parameters)

    Returns:
        Tuple of (downscaled_model, output_path)
    """
    # Validate supported model
    supported_models = ["Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-0.6B"]
    if model_path not in supported_models:
        raise NotImplementedError(
            f"Downscaling currently only supports {supported_models}, got {model_path}"
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.float32
    )

    # Apply Model Folding
    downscaled_model = model_folding_llm(model, sparsity)

    # Generate output path if not provided
    if output_path is None:
        output_path = f"{model_path}-downscaled-{sparsity}"

    return downscaled_model, output_path


def model_folding_llm(
    model: AutoModelForCausalLM, target_sparsity: float
) -> AutoModelForCausalLM:
    """Apply Model Folding to reduce model parameters."""
    # Note: Modifies model in place
    num_layers = len(model.model.layers)
    total_steps = num_layers * 2 + 1  # attention + mlp per layer + embeddings
    current_step = 0

    print(f"Starting Model Folding downscaling with {total_steps} total steps")

    # For each transformer block
    for i, block in enumerate(model.model.layers):  # Assuming Qwen3 structure
        print(f"Processing layer {i + 1}/{num_layers}")

        # Handle attention
        current_step += 1
        print(f"Step {current_step}/{total_steps}: Downscaling attention heads")
        _downscale_attention(block, target_sparsity)

        # Handle MLP
        current_step += 1
        print(f"Step {current_step}/{total_steps}: Downscaling MLP")
        _downscale_mlp(block, target_sparsity)

        # Handle LayerNorm (skipped)
        _downscale_layernorm(block, target_sparsity)

    # Handle embeddings
    current_step += 1
    print(f"Step {current_step}/{total_steps}: Downscaling embeddings")
    _downscale_embeddings(model, target_sparsity)

    print("Model Folding downscaling completed")
    return model


def _downscale_attention(block, sparsity: float):
    """Downscale attention heads using clustering."""
    # Get attention projections
    q_proj = block.self_attn.q_proj
    k_proj = block.self_attn.k_proj
    v_proj = block.self_attn.v_proj
    o_proj = block.self_attn.o_proj

    d_model = q_proj.in_features
    head_dim_q = 128  # from q_norm shape
    num_heads = q_proj.out_features // head_dim_q
    head_dim_kv = k_proj.out_features // num_heads

    # Build head vectors: [Q_h; K_h; V_h] for each head h
    head_vectors = []
    for h in range(num_heads):
        q_h = q_proj.weight[h * head_dim_q : (h + 1) * head_dim_q, :].flatten()
        k_h = k_proj.weight[h * head_dim_kv : (h + 1) * head_dim_kv, :].flatten()
        v_h = v_proj.weight[h * head_dim_kv : (h + 1) * head_dim_kv, :].flatten()
        w_i = torch.cat([q_h, k_h, v_h])
        head_vectors.append(w_i)
    head_vectors = torch.stack(head_vectors)

    # k-Means clustering
    K = max(1, int(math.ceil(num_heads * (1 - sparsity))))
    labels, centers = kmeans_torch(head_vectors, K, device=str(head_vectors.device))

    # Create merge matrix M [K, num_heads]
    M = torch.zeros(K, num_heads, device=head_vectors.device)
    for k in range(K):
        mask = labels == k
        count = mask.sum()
        if count > 0:
            M[k, mask] = 1.0 / count

    # Merge heads
    q_merged = torch.zeros(K, head_dim_q, d_model)
    k_merged = torch.zeros(K, head_dim_kv, d_model)
    v_merged = torch.zeros(K, head_dim_kv, d_model)
    for h in range(num_heads):
        for k in range(K):
            q_merged[k] += (
                M[k, h] * q_proj.weight[h * head_dim_q : (h + 1) * head_dim_q, :]
            )
            k_merged[k] += (
                M[k, h] * k_proj.weight[h * head_dim_kv : (h + 1) * head_dim_kv, :]
            )
            v_merged[k] += (
                M[k, h] * v_proj.weight[h * head_dim_kv : (h + 1) * head_dim_kv, :]
            )

    # Fold-AR scaling
    pre_moment = (head_vectors**2).sum() / num_heads
    post_moment = (centers**2).sum() / K
    alpha = math.sqrt(pre_moment / post_moment) if post_moment > 0 else 1.0

    q_merged *= alpha
    k_merged *= alpha
    v_merged *= alpha

    # Set new weights
    block.self_attn.q_proj.weight.data = q_merged.view(K * head_dim_q, d_model)
    block.self_attn.k_proj.weight.data = k_merged.view(K * head_dim_kv, d_model)
    block.self_attn.v_proj.weight.data = v_merged.view(K * head_dim_kv, d_model)

    # Merge o_proj: [d_model, num_heads * head_dim_q] -> [d_model, K * head_dim_q]
    o_merged = torch.zeros(d_model, K * head_dim_q)
    for h in range(num_heads):
        for k in range(K):
            o_merged[:, k * head_dim_q : (k + 1) * head_dim_q] += (
                M[k, h] * o_proj.weight[:, h * head_dim_q : (h + 1) * head_dim_q]
            )
    o_merged *= alpha
    block.self_attn.o_proj.weight.data = o_merged


def _downscale_mlp(block, sparsity: float):
    """Downscale MLP using clustering."""
    # Up projections: gate_proj and up_proj have same shape [d_ffn, d_model]
    up_weight = block.mlp.up_proj.weight.data  # [d_ffn, d_model]
    up_vectors = up_weight  # keep as torch

    m = up_weight.shape[0]
    K_up = max(1, int(math.ceil(m * (1 - sparsity))))
    labels_up, centers_up = kmeans_torch(up_vectors, K_up, device=up_weight.device)

    M_up = torch.zeros(K_up, m, device=up_weight.device)
    for k in range(K_up):
        mask = labels_up == k
        count = mask.sum()
        if count > 0:
            M_up[k, mask] = 1.0 / count

    up_merged = M_up @ up_weight
    gate_merged = M_up @ block.mlp.gate_proj.weight.data

    # Fold-AR for up_proj (approximate SiLU variance)
    pre_moment_up = (up_vectors**2).sum() / m
    post_moment_up = (centers_up**2).sum() / K_up
    alpha_up = torch.sqrt(pre_moment_up / post_moment_up) if post_moment_up > 0 else 1.0
    up_merged *= alpha_up
    gate_merged *= alpha_up

    block.mlp.up_proj.weight.data = up_merged
    block.mlp.gate_proj.weight.data = gate_merged

    # Down projection after up merging
    down_weight = block.mlp.down_proj.weight.data  # [d_model, d_ffn]
    down_vectors = down_weight.T  # [d_ffn, d_model]

    K_down = K_up  # Match up projection
    labels_down, centers_down = kmeans_torch(
        down_vectors, K_down, device=down_weight.device
    )

    M_down = torch.zeros(K_down, down_weight.shape[1], device=down_weight.device)
    for k in range(K_down):
        mask = labels_down == k
        count = mask.sum()
        if count > 0:
            M_down[k, mask] = 1.0 / count

    down_merged = down_weight @ M_down.T

    # Fold-AR for down_proj
    pre_moment_down = (down_vectors**2).sum() / down_weight.shape[1]
    post_moment_down = (centers_down**2).sum() / K_down
    alpha_down = (
        torch.sqrt(pre_moment_down / post_moment_down) if post_moment_down > 0 else 1.0
    )
    down_merged *= alpha_down

    block.mlp.down_proj.weight.data = down_merged


def _downscale_layernorm(block, sparsity: float):
    """Merge LayerNorm parameters."""
    # For simplicity, assume no merging for LayerNorm, as per method
    pass


def _downscale_embeddings(model, sparsity: float):
    """Cluster embeddings."""
    embed = model.model.embed_tokens.weight.data
    embed_vectors = embed

    vocab_size, d_model = embed.shape
    K_embed = max(1, int(math.ceil(vocab_size * (1 - sparsity))))

    # Use kmeans on vectors
    labels_embed, centers_embed = kmeans_torch(
        embed_vectors, K_embed, device=embed.device
    )

    M_embed = torch.zeros(K_embed, vocab_size, device=embed.device)
    for k in range(K_embed):
        mask = labels_embed == k
        count = mask.sum()
        if count > 0:
            M_embed[k, mask] = 1.0 / count

    embed_merged = M_embed @ embed

    model.model.embed_tokens.weight.data = embed_merged
