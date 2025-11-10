"""
Common utilities for tensor manipulations and layer/block operations.
Based on HyperCloning method from https://arxiv.org/html/2409.12903v2
"""


def count_parameters(model):
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def format_parameter_count(num_params):
    """Format parameter count in human-readable format (B for billions, M for millions) with exact count."""
    if num_params >= 1e9:
        formatted = f"{num_params / 1e9:.1f}B"
    elif num_params >= 1e6:
        formatted = f"{num_params / 1e6:.1f}M"
    else:
        formatted = str(num_params)

    return f"{formatted} ({num_params:,})"


def get_model_size_suffix(num_params):
    """Get model size suffix based on parameter count (rounded only)."""
    if num_params >= 1e9:
        res = f"{num_params / 1e9:.1f}B"
    elif num_params >= 1e6:
        res = f"{num_params / 1e6:.1f}M"
    else:
        res = str(num_params)

    return res
