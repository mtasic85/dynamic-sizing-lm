"""
Common utilities for tensor manipulations and layer/block operations.
Based on HyperCloning method from https://arxiv.org/html/2409.12903v2
"""

import torch


def scale_linear_layer(layer: torch.nn.Linear, scaler: float):
    """
    Scales the parameters of 'layer' so that its output is multiplied by 'scaler'.
    """
    layer.weight.data *= scaler
    if layer.bias is not None:
        layer.bias.data *= scaler


def get_noise_with_snr(weight: torch.Tensor, snr_db: float):
    """
    Gaussian noise to be added to 'weight' so that the signal-to-noise
    ratio becomes 'snr_db'.
    """
    signal_power = torch.mean(weight**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(weight)
    current_noise_power = torch.mean(noise**2)
    noise = noise * torch.sqrt(noise_power / current_noise_power)
    return noise.to(weight.dtype)


def add_noise(weight, block_shape, snr_db):
    """
    Repeatedly adds and subtracts noise to 'block_shape' blocks within 'weight'.
    """
    assert weight.shape[0] % block_shape[0] == 0
    assert weight.shape[1] % block_shape[1] == 0
    n_repeat_0 = weight.shape[0] // block_shape[0]
    n_repeat_1 = weight.shape[1] // block_shape[1]
    if weight.ndim == 2:
        for n0 in range(n_repeat_0):
            start0 = n0 * block_shape[0]
            end0 = start0 + block_shape[0]
            for n1 in range(n_repeat_1 // 2):
                start1 = 2 * n1 * block_shape[1]
                end1 = start1 + block_shape[1]
                start2 = (2 * n1 + 1) * block_shape[1]
                end2 = start2 + block_shape[1]
                noise = get_noise_with_snr(weight[start0:end0, start1:end1], snr_db)
                weight[start0:end0, start1:end1] += noise
                weight[start0:end0, start2:end2] -= noise
        return weight
    else:
        for n0 in range(weight.shape[0]):
            weight[n0] = add_noise(weight[n0], block_shape[1:], snr_db)
        return weight


def clone_matrix(dst_weight_shape, src_weight, snr_db=None, normalize=True):
    """
    Clones a matrix from 'src_weight' into 'dst_weight_shape'.
    """
    out_features_old, in_features_old = src_weight.shape
    out_features_new, in_features_new = dst_weight_shape
    assert out_features_new >= out_features_old
    assert out_features_new % out_features_old == 0
    assert in_features_new >= in_features_old
    assert in_features_new % in_features_old == 0, (
        f"{in_features_new} does not divide {in_features_old}"
    )
    n_repeat_0 = out_features_new // out_features_old
    n_repeat_1 = in_features_new // in_features_old

    dst_weight = src_weight.data.repeat(n_repeat_0, n_repeat_1)
    if normalize:
        dst_weight = dst_weight / n_repeat_1
    if snr_db is not None:
        dst_weight = add_noise(dst_weight, src_weight.shape, snr_db)
    return dst_weight


def clone_vector(dst_vector_shape, src_vector):
    """
    Clones a vector from 'src_vector' into 'dst_vector_shape'.
    """
    assert src_vector.shape[0] <= dst_vector_shape[0]
    assert dst_vector_shape[0] % src_vector.shape[0] == 0
    n_repeat = dst_vector_shape[0] // src_vector.shape[0]
    dst_vector = src_vector.repeat(n_repeat)
    return dst_vector


def clone_linear_layer(dst_layer, src_layer, snr_db=None):
    """
    Clones linear layer parameters from 'src_layer' into 'dst_layer'.
    """
    dst_layer.weight.data = clone_matrix(
        dst_layer.weight.shape, src_layer.weight.data, snr_db=snr_db
    )
    if src_layer.bias is not None:
        assert dst_layer.bias is not None, (
            "source model has bias in its linear layers but destination model doesn't"
        )
        dst_layer.bias.data = clone_vector(dst_layer.bias.shape, src_layer.bias.data)


def clone_layer_norm(dst_layer, src_layer):
    """
    Clones normalization layer parameters from 'src_layer' into 'dst_layer'.
    """
    if src_layer.weight is None and src_layer.bias is None:
        assert dst_layer.weight is None and dst_layer.bias is None
        return
    assert dst_layer.eps == src_layer.eps, (
        f"eps should be the same for source and destination layer-norms, \
        got {src_layer.eps} and {dst_layer.eps}"
    )
    assert dst_layer.elementwise_affine == src_layer.elementwise_affine, (
        f"elementwise_affine should be the same for source and destination \
        layer-norms, got {src_layer.elementwise_affine} and {dst_layer.elementwise_affine}"
    )
    dst_layer.weight.data = clone_vector(dst_layer.weight.shape, src_layer.weight)
    dst_layer.bias.data = clone_vector(dst_layer.bias.shape, src_layer.bias.data)


def clone_rms_norm(dst_layer, src_layer):
    """
    Clones rms-normalization layer parameters from 'src_layer' into 'dst_layer'.
    """
    dst_layer.weight.data = clone_vector(dst_layer.weight.shape, src_layer.weight)


def rename_config(
    config, embedding_dim_multiplier: int = 1, up_project_multiplier: int = 1
):
    """
    adjusts the model name according to 'embedding_dim_multiplier' and 'up_project_multiplier'
    """
    if embedding_dim_multiplier > 1:
        config._name_or_path += f"-{embedding_dim_multiplier}xembedding"
    if up_project_multiplier > 1:
        config._name_or_path += f"-{up_project_multiplier}xffn"
    return config


class scaledLinear(torch.nn.Module):
    """
    Wrapper layer that scales the weights of a linear layer before applying
    the linear transformation.
    """

    def __init__(self, layer, scaler):
        super().__init__()
        self.layer = layer
        self.scaler = scaler
        self.weight = self.layer.weight
        self.bias = self.layer.bias

    def forward(self, x):
        weight = self.layer.weight * self.scaler
        if self.layer.bias is not None:
            bias = self.layer.bias * self.scaler
        else:
            bias = None
        return torch.nn.functional.linear(x, weight, bias)


def clone_gemma_qkv_layer(
    dst_layer, src_layer, dst_num_heads, src_num_heads, snr_db=None
):
    """
    Clones QKV layers for attention mechanisms, handling head expansion.
    """
    assert dst_num_heads % src_num_heads == 0
    head_multiplier = dst_num_heads // src_num_heads

    dst_layer.weight.data = clone_matrix(
        dst_layer.weight.shape, src_layer.weight.data, snr_db=snr_db
    )
    if src_layer.bias is not None:
        assert dst_layer.bias is not None
        dst_layer.bias.data = clone_vector(dst_layer.bias.shape, src_layer.bias.data)


def count_parameters(model):
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def format_parameter_count(num_params):
    """Format parameter count in human-readable format (B for billions, M for millions)."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.1f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.1f}M"
    else:
        return str(num_params)


def get_model_size_suffix(num_params):
    """Get model size suffix based on parameter count."""
    if num_params >= 1e9:
        return ".1f"
    elif num_params >= 1e6:
        return ".1f"
    else:
        return str(num_params)
