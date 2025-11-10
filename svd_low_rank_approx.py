"""
SVD-based low-rank approximation for model downscaling.

This module provides functions to compress weight matrices using Singular Value Decomposition (SVD)
to reduce model parameters while preserving performance.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List


def svd_low_rank_approx(
    weight: torch.Tensor,
    rank: Optional[int] = None,
    energy_threshold: Optional[float] = None,
    p_norm: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Approximate a weight matrix using low-rank SVD decomposition.

    Args:
        weight: Weight matrix to approximate (m x n)
        rank: Target rank for approximation. If None, determined by energy_threshold
        energy_threshold: Fraction of total energy to retain (0.0-1.0). Default 0.99
        p_norm: Norm for robust SVD (default 2.0 for standard Frobenius)

    Returns:
        Tuple of (W0, W1) where W0 @ W1 ≈ original weight
    """
    if energy_threshold is None:
        energy_threshold = 0.99

    # Compute SVD
    if p_norm == 2.0:
        U, S, V = torch.svd(weight)
    else:
        # For robust p-SVD, use low-rank approximation with q parameter
        # torch.svd_lowrank uses randomized SVD for efficiency
        U, S, V = torch.svd_lowrank(weight, q=min(rank or weight.shape[0] // 2, 100))

    # Determine rank if not specified
    if rank is None:
        # Cumulative energy
        total_energy = torch.sum(S**p_norm)
        cumulative_energy = torch.cumsum(S**p_norm, dim=0)
        rank = (
            int(torch.sum(cumulative_energy < energy_threshold * total_energy).item())
            + 1
        )
        rank = min(rank, len(S))

    # Truncate to rank
    U_r = U[:, :rank]
    S_r = S[:rank]
    V_r = V[:, :rank]

    # Original dimension
    d = weight.shape[1] if weight.shape[0] > weight.shape[1] else weight.shape[0]

    # Scale for variance preservation (similar to hypercloning)
    scale = torch.sqrt(torch.tensor(rank / d, dtype=weight.dtype, device=weight.device))

    # Create low-rank factors
    W0 = U_r @ torch.diag(torch.sqrt(S_r)) * scale
    W1 = torch.diag(torch.sqrt(S_r)) @ V_r.t() * scale

    return W0, W1


def replace_linear_with_low_rank(
    linear: nn.Linear,
    rank: Optional[int] = None,
    energy_threshold: Optional[float] = None,
    p_norm: float = 2.0,
) -> nn.Sequential:
    """
    Replace a linear layer with two low-rank linear layers.

    Args:
        linear: Original linear layer
        rank: Target rank
        energy_threshold: Energy retention threshold
        p_norm: Norm for SVD

    Returns:
        Sequential module with two linear layers approximating the original
    """
    W0, W1 = svd_low_rank_approx(linear.weight.data, rank, energy_threshold, p_norm)

    # Create new layers
    layer0 = nn.Linear(W1.shape[1], W0.shape[1], bias=False)
    layer1 = nn.Linear(W0.shape[1], linear.out_features, bias=linear.bias is not None)

    # Set weights
    layer0.weight.data = W1
    layer1.weight.data = W0

    if linear.bias is not None:
        layer1.bias.data = linear.bias.data

    return nn.Sequential(layer0, layer1)


def get_compressible_layers(model: nn.Module) -> List[Tuple[str, nn.Linear]]:
    """
    Get list of compressible linear layers in the model.

    Args:
        model: PyTorch model

    Returns:
        List of (layer_name, layer) tuples for compressible layers
    """
    compressible_layers = []

    def is_compressible_layer(name: str, module: nn.Module) -> bool:
        """Check if a layer should be compressed."""
        compressible_names = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",  # attention
            "gate_proj",
            "up_proj",
            "down_proj",  # FFN
        ]

        return isinstance(module, nn.Linear) and any(
            comp in name for comp in compressible_names
        )

    for name, module in model.named_modules():
        if is_compressible_layer(name, module):
            compressible_layers.append((name, module))

    return compressible_layers


def apply_low_rank_approximation(
    model: nn.Module,
    rank: Optional[int] = None,
    energy_threshold: Optional[float] = None,
    p_norm: float = 2.0,
    layer_names: Optional[List[str]] = None,
) -> nn.Module:
    """
    Apply low-rank approximation to compressible layers in the model.

    Args:
        model: Input model
        rank: Target rank for all layers
        energy_threshold: Energy retention threshold
        p_norm: Norm for SVD
        layer_names: Specific layers to compress (if None, compress all compressible)

    Returns:
        Model with low-rank approximated layers
    """
    compressible_layers = get_compressible_layers(model)

    if layer_names:
        compressible_layers = [
            (n, m) for n, m in compressible_layers if n in layer_names
        ]

    for name, linear in compressible_layers:
        # Get parent module and attribute name
        *parent_names, attr_name = name.split(".")
        parent = model

        for p_name in parent_names:
            parent = getattr(parent, p_name)

        # Replace with low-rank approximation
        low_rank_seq = replace_linear_with_low_rank(
            linear, rank, energy_threshold, p_norm
        )

        setattr(parent, attr_name, low_rank_seq)

    return model
