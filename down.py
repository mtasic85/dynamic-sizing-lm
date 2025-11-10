"""
Downscaling logic for language models using the Folding method.
Based on "Forget the Data and Fine-Tuning! Just Fold the Network to Compress".
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from typing import Optional, Tuple


def torch_kmeans_plus_plus(data, k, max_iters=10, tol=1e-4, device="cuda"):
    """
    Efficient k-means++ on data: [num_points, dim].
    Returns: labels [num_points], centroids [k, dim].
    """
    num_points, dim = data.shape
    if k >= num_points:
        return torch.arange(num_points, device=device), data

    # Random projection for approximation (speed-up)
    if dim > 256:
        proj_dim = 256
        proj_matrix = torch.randn(dim, proj_dim, device=device) / (proj_dim**0.5)
        proj_data = data @ proj_matrix  # [num_points, proj_dim]
    else:
        proj_data = data
        proj_matrix = None

    # K-means++ init
    centroids = proj_data[torch.randint(0, num_points, (1,)).item()].unsqueeze(0)
    for _ in range(1, k):
        dists = torch.cdist(proj_data, centroids)
        probs = torch.min(dists, dim=1)[0] ** 2 / num_points
        next_idx = torch.multinomial(probs, 1).item()
        centroids = torch.cat([centroids, proj_data[next_idx].unsqueeze(0)])

    # Lloyd's iterations
    prev_labels = None
    for iter in range(max_iters):
        dists = torch.cdist(proj_data, centroids)  # [num_points, k]
        labels = torch.argmin(dists, dim=1)  # [num_points]

        if prev_labels is not None and (labels == prev_labels).all():
            break
        prev_labels = labels.clone()

        # Update centroids (exact mean)
        new_centroids = []
        for c in range(k):
            cluster_points = proj_data[labels == c]
            if cluster_points.numel() > 0:
                new_centroids.append(cluster_points.mean(0))
            else:
                new_centroids.append(centroids[c])
        centroids = torch.stack(new_centroids)

    # Project centroids back if approximated
    if proj_matrix is not None:
        centroids = centroids @ proj_matrix.T  # Back to original dim

    return labels, centroids


def build_projection_matrix(labels, k, num_points, device="cuda"):
    """
    Build U [num_points, k] binary assignment, C = U (U^T U)^{-1} U^T.
    Returns: U, C [num_points, num_points] (sparse-friendly).
    """
    U = F.one_hot(labels, num_classes=k).float()  # [num_points, k]
    sizes = torch.bincount(labels, minlength=k).float()  # Cluster sizes [k]
    D_inv = torch.diag(1.0 / (sizes + 1e-8))  # [k, k]
    C = U @ D_inv @ U.T  # [num_points, num_points]; use sparse for large n
    return U, C


def fold_generic_linears(prev_linear, curr_linear, next_linear, k, device="cuda"):
    """
    Fold curr_linear, considering prev/next for consistency.
    prev/next: Optional nn.Linear for concat.
    """
    W_curr = curr_linear.weight.data  # [n_out, n_in]
    rows_curr = W_curr.T.contiguous()  # [n_in, n_out] -> rows as points [n_out, n_in]

    # Concat with next for output folding (inter-layer)
    if next_linear is not None:
        W_next = next_linear.weight.data  # [n_next_out, n_curr_out]
        rows_next = W_next.T.contiguous()[
            :, : rows_curr.shape[1]
        ]  # Align dims if needed
        rows_concat = torch.cat(
            [rows_curr, rows_next], dim=0
        )  # [n_out + n_next_out, n_in]
    else:
        rows_concat = rows_curr

    # Cluster
    labels, centroids = torch_kmeans_plus_plus(rows_concat, k=k, device=device)

    # Split labels if concat
    if next_linear is not None:
        labels_curr = labels[: rows_curr.shape[0]]
    else:
        labels_curr = labels

    # Build projection
    n_out = rows_curr.shape[0]
    U, C = build_projection_matrix(labels_curr, k=k, num_points=n_out, device=device)

    # Fold current weight: C @ W_curr (project rows)
    folded_W = C @ W_curr  # [k, n_in]

    # Update module: Resize to k outputs
    with torch.no_grad():
        curr_linear.out_features = k
        curr_linear.weight.data = folded_W
        if curr_linear.bias is not None:
            curr_linear.bias.data = curr_linear.bias.data[
                labels_curr
            ]  # Average biases per cluster
            curr_linear.bias.data = torch.stack(
                [curr_linear.bias.data[labels_curr == c].mean(0) for c in range(k)]
            )

    # Adjust next input dim if exists
    if next_linear is not None:
        next_linear.in_features = k

    return U, labels_curr  # For repair/alignment


def fold_ffn_block(ffn_linears, r, device):  # e.g., [gate_proj, up_proj, down_proj]
    gate, up, down = ffn_linears

    # Output folding: Cluster gate/up outputs jointly for down_proj input
    rows_out_gate = gate.weight  # [out_gate, in]
    rows_out_up = up.weight  # [out_up, in]
    rows_out = torch.cat([rows_out_gate, rows_out_up], dim=0)  # [out_total, in]
    k_out = int(rows_out.shape[0] * r)
    _, centroids = torch_kmeans_plus_plus(rows_out, k=k_out, device=device)

    # Fold outputs by replacing with centroids
    gate.out_features = k_out // 2
    up.out_features = k_out - gate.out_features
    gate.weight.data = centroids[: gate.out_features]
    up.weight.data = centroids[gate.out_features :]

    # Adjust down_proj input
    down.in_features = k_out


def fold_attention_block(
    attn_linears, r, device
):  # e.g., [q_proj, k_proj, v_proj, o_proj]
    q, k, v, o = attn_linears

    # Output folding: o_proj
    rows_o = o.weight  # [out, in]
    k_out = int(rows_o.shape[0] * r)
    _, centroids = torch_kmeans_plus_plus(rows_o, k=k_out, device=device)
    o.weight.data = centroids
    o.out_features = k_out


def fold_ar_repair(linear, U, labels, norm_layer=None, device="cuda"):
    """
    Data-free repair for a folded linear (and optional LayerNorm).
    Adjusts scales to preserve Var(y) ≈ Var(original).
    """
    W = linear.weight.data  # Folded [k, in]

    # Normalize weights (mimic \Sigma_n)
    W_norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)  # Unit norm rows

    # For each cluster c, compute E[c] = avg cosine sim between pairs i≠j in c
    k = U.shape[1]
    for c in range(k):
        cluster_mask = labels == c
        if cluster_mask.sum() < 2:
            continue
        cluster_weights = W_norm[cluster_mask]  # [N_c, n_in]

        # Approx correlations: pairwise dot / norms (but norms=1)
        # Speed-up: Use (cluster_weights @ cluster_weights.T).sum() - diag, but off-diag avg
        corr_matrix = cluster_weights @ cluster_weights.T  # [N_c, N_c]
        diag = torch.diag(corr_matrix)
        E_c = (corr_matrix.sum() - diag.sum()) / (
            cluster_mask.sum() * (cluster_mask.sum() - 1)
        )

        N_c = cluster_mask.sum().float()
        scale_factor = N_c / torch.sqrt(N_c + (N_c**2 - N_c) * E_c)

        # Rescale cluster centroid
        linear.weight.data[c] *= scale_factor

    # If LayerNorm present, adjust its weight (scale)
    if norm_layer is not None and hasattr(norm_layer, "weight"):
        norm_weight = norm_layer.weight.data
        # Cluster norm scales similarly (treat as diag matrix rows)
        scale_labels, _ = torch_kmeans_plus_plus(
            norm_weight.unsqueeze(1), k=k, device=device
        )  # Dim=1
        scale_U, _ = build_projection_matrix(scale_labels, k, len(norm_weight), device)
        folded_scale = scale_U.T @ norm_weight  # Average per cluster
        # Apply similar repair
        for c in range(k):
            N_c = (scale_labels == c).sum().float()
            # E[c] for scales: simple avg similarity (1D)
            cluster_scales = norm_weight[scale_labels == c]
            E_c = (cluster_scales @ cluster_scales).sum() / (
                N_c * (N_c - 1)
            ) - 1.0 / N_c  # Approx
            scale_factor = N_c / torch.sqrt(N_c + (N_c**2 - N_c) * E_c)
            folded_scale[c] *= scale_factor
        norm_layer.weight.data = F.pad(
            folded_scale, (0, len(norm_weight) - k)
        )  # Truncate/resize
        norm_layer.weight.data[:k] = folded_scale  # Simplified; resize module if needed


def fold_transformer_model(model, compression_ratio=0.8, device="cuda"):
    """
    Fold all linear layers in a transformer model (e.g., Qwen).
    Assumes model is nn.Module with named_modules.
    Modifies in-place.
    """
    # Collect linear layers by block (assume standard HF structure)
    linears = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if "self_attn" in name or "mlp" in name:
                # For layers, group by 'model.layers.X'
                block_name = ".".join(name.split(".")[:3])
            else:
                block_name = ""  # For embed_tokens, lm_head, etc.
            if block_name not in linears:
                linears[block_name] = []
            linears[block_name].append((name, module))

    # Sequential folding per block
    def block_sort_key(x):
        if not x:
            return 999  # Put empty key (lm_head) last
        parts = x.split(".")
        last_part = parts[-1]
        if last_part.isdigit():
            return int(last_part)
        else:
            return -1  # Put non-numeric like 'model' first

    total_blocks = len([b for b in linears.keys() if b])
    for idx, block_idx in enumerate(sorted(linears.keys(), key=block_sort_key)):
        if not block_idx:
            continue  # Skip non-block linears like lm_head, embed_tokens
        block_linears = linears[block_idx]

        print(f"Folding block {block_idx} ({idx + 1}/{total_blocks})")

        # Group by type: FFN or Attention
        ffn_linears = []
        attn_linears = []
        for name, linear in block_linears:
            if "mlp" in name:
                if "gate_proj" in name:
                    ffn_linears.append(linear)
                elif "up_proj" in name:
                    ffn_linears.append(linear)
                elif "down_proj" in name:
                    ffn_linears.append(linear)
            elif "self_attn" in name:
                if "q_proj" in name:
                    attn_linears.append(linear)
                elif "k_proj" in name:
                    attn_linears.append(linear)
                elif "v_proj" in name:
                    attn_linears.append(linear)
                elif "o_proj" in name:
                    attn_linears.append(linear)

        if ffn_linears:
            print(f"  Folding FFN block with {len(ffn_linears)} layers")
            fold_ffn_block(ffn_linears, compression_ratio, device)
        if attn_linears:
            print(f"  Folding Attention block with {len(attn_linears)} layers")
            fold_attention_block(attn_linears, compression_ratio, device)

    return model


def downscale_model(
    model_path: str,
    output_path: Optional[str] = None,
    compression_ratio: float = 0.8,
    device: str = "auto",
) -> Tuple[torch.nn.Module, str]:
    """
    Downscale a model using the Folding method.

    Args:
        model_path: Path to input model
        output_path: Output path for downscaled model
        compression_ratio: Ratio to compress dimensions (0.0-1.0)
        device: Device to use ('auto', 'cuda', 'cpu')

    Returns:
        Tuple of (downscaled_model, output_path)
    """
    print(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.float32
    )

    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}")

    print(f"Compression ratio: {compression_ratio}")

    # Fold the model
    print("Starting folding process...")
    folded_model = fold_transformer_model(model, compression_ratio, device)
    print("Folding completed.")

    # Determine output path
    if output_path is None:
        base_name = model_path.split("/")[-1] if "/" in model_path else model_path
        output_path = f"{base_name}-folded-{compression_ratio}"

    return folded_model, output_path
