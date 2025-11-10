"""
Downscaling logic for language models using SVD-based low-rank approximation.
"""

from typing import Optional, Tuple
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from svd_low_rank_approx import apply_low_rank_approximation
from common import count_parameters, format_parameter_count


def downscale_model(
    model_path: str,
    output_path: Optional[str] = None,
    rank: Optional[int] = None,
    energy_threshold: Optional[float] = None,
    p_norm: float = 2.0,
) -> Tuple[torch.nn.Module, str]:
    """
    Downscale a model using SVD-based low-rank approximation.

    Args:
        model_path: Path to input model
        output_path: Output path for downscaled model
        rank: Target rank for low-rank approximation
        energy_threshold: Energy retention threshold (0.0-1.0)
        p_norm: Norm for robust SVD

    Returns:
        Tuple of (downscaled_model, output_path)
    """
    # Load model
    print(f"Loading model: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.float32
    )

    # Apply low-rank approximation
    print("Applying SVD low-rank approximation...")

    downscaled_model = apply_low_rank_approximation(
        model,
        rank=rank,
        energy_threshold=energy_threshold,
        p_norm=p_norm,
    )

    # Print parameter count
    print(
        f"Downscaled model has {format_parameter_count(count_parameters(downscaled_model))} parameters"
    )

    # Determine output path
    if output_path is None:
        base_name = os.path.basename(model_path)
        rank_str = f"_rank{rank}" if rank else f"_energy{energy_threshold}"
        output_path = f"{base_name}_downscaled{rank_str}"

    print(f"Output path: {output_path}")

    # Save model
    print(f"Saving downscaled model to: {output_path}")
    downscaled_model.save_pretrained(output_path)

    # Try to save tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.save_pretrained(output_path)
        print("Tokenizer saved successfully")
    except Exception as e:
        print(f"Warning: Could not save tokenizer: {e}")

    return downscaled_model, output_path
