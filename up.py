"""
Upscaling logic using HyperCloning method.
Based on https://arxiv.org/html/2409.12903v2 and https://github.com/mtasic85/ml-hypercloning
"""

import copy
import torch
from transformers import AutoModelForCausalLM
from typing import Optional

from common import (  # type: ignore
    clone_matrix,
    clone_linear_layer,
    clone_rms_norm,
    clone_layer_norm,
    clone_qkv_layer,
    rename_config,
    count_parameters,
    format_parameter_count,
    get_model_size_suffix,
)


def upscale_model(
    model_path: str,
    embed_dim_multiplier: int,
    up_proj_multiplier: int,
    output_path: Optional[str] = None,
    snr_db: Optional[float] = None,
) -> tuple:
    """
    Upscale a model using HyperCloning method.

    Args:
        model_path: Path or HuggingFace model identifier
        embed_dim_multiplier: Integer multiplier for embedding dimensions
        up_proj_multiplier: Integer multiplier for FFN dimensions
        output_path: Optional output path for the upscaled model
        snr_db: Optional signal-to-noise ratio for adding noise

    Returns:
        Tuple of (upscaled_model, output_path)
    """
    print(f"Loading source model: {model_path}")

    src_model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.float32
    )

    # Determine cloning function based on config type
    config_type = str(type(src_model.config)).split(".")[-1][:-2].strip()

    print(f"Detected model type: {config_type}")

    if config_type not in REGISTERED_CLONING_FUNCTIONS:
        raise ValueError(
            f"Model type {config_type} is not supported. Supported types: {list(REGISTERED_CLONING_FUNCTIONS.keys())}"
        )

    cloning_function = REGISTERED_CLONING_FUNCTIONS[config_type]
    print(f"Using cloning function: {cloning_function.__name__}")

    # Clone the model
    dst_model = cloning_function(
        src_model,
        embedding_dim_multiplier=embed_dim_multiplier,
        up_project_multiplier=up_proj_multiplier,
        snr_db=snr_db,
    )

    # Determine output path if not provided
    if output_path is None:
        # Calculate parameter count and create name
        param_count = count_parameters(dst_model)
        size_suffix = get_model_size_suffix(param_count)

        # Extract base name from model path
        base_name = model_path.split("/")[-1] if "/" in model_path else model_path
        output_path = f"{base_name}-{size_suffix}"

    print(
        f"Upscaled model has {format_parameter_count(count_parameters(dst_model))} parameters"
    )
    print(f"Output path: {output_path}")

    return dst_model, output_path


def clone_qwen2_5(
    src_network,
    embedding_dim_multiplier: int = 1,
    up_project_multiplier: int = 1,
    **kwargs,
):
    """
    Cloning function for Qwen2.5 models.
    """
    snr_db = kwargs.get("snr_db", None)
    num_heads_multiplier = kwargs.get("num_heads_multiplier", embedding_dim_multiplier)
    assert num_heads_multiplier == embedding_dim_multiplier, (
        "head_dim expansion is not supported for Qwen2.5. The number of heads will \
        be automatically computed based on embedding dimension expansion."
    )

    # Set the destination network config
    config = copy.deepcopy(src_network.config)
    config.hidden_size = embedding_dim_multiplier * config.hidden_size
    config.intermediate_size = up_project_multiplier * config.intermediate_size
    if config.num_key_value_heads != 1:
        config.num_key_value_heads = (
            embedding_dim_multiplier * config.num_key_value_heads
        )
    config.num_attention_heads = embedding_dim_multiplier * config.num_attention_heads
    config.tie_word_embeddings = False

    # Rename config according to expansion factors
    config = rename_config(config, embedding_dim_multiplier, up_project_multiplier)

    # Create destination network
    dst_network = type(src_network)._from_config(config)

    # Clone embeddings
    dst_network.model.embed_tokens.weight.data = clone_matrix(
        dst_network.model.embed_tokens.weight.data.shape,
        src_network.model.embed_tokens.weight.data,
        normalize=False,
    )

    # Clone each layer
    for dst_layer, src_layer in zip(dst_network.model.layers, src_network.model.layers):
        clone_rms_norm(dst_layer.input_layernorm, src_layer.input_layernorm)
        clone_rms_norm(
            dst_layer.post_attention_layernorm, src_layer.post_attention_layernorm
        )
        clone_qwen_attention(dst_layer.self_attn, src_layer.self_attn, snr_db=snr_db)
        clone_linear_layer(
            dst_layer.mlp.gate_proj,
            src_layer.mlp.gate_proj,
            snr_db=snr_db,  # type: ignore
        )
        clone_linear_layer(dst_layer.mlp.up_proj, src_layer.mlp.up_proj, snr_db=snr_db)  # type: ignore
        clone_linear_layer(
            dst_layer.mlp.down_proj,
            src_layer.mlp.down_proj,
            snr_db=snr_db,  # type: ignore
        )

    clone_rms_norm(dst_network.model.norm, src_network.model.norm)
    clone_linear_layer(dst_network.lm_head, src_network.lm_head)
    return dst_network


def clone_qwen3(
    src_network,
    embedding_dim_multiplier: int = 1,
    up_project_multiplier: int = 1,
    **kwargs,
):
    """
    Cloning function for Qwen3 models.
    """
    snr_db = kwargs.get("snr_db", None)
    num_heads_multiplier = kwargs.get("num_heads_multiplier", embedding_dim_multiplier)
    assert num_heads_multiplier == embedding_dim_multiplier, (
        "head_dim expansion is not supported for Qwen3. The number of heads will \
        be automatically computed based on embedding dimension expansion."
    )

    # Set the destination network config
    config = copy.deepcopy(src_network.config)
    config.hidden_size = embedding_dim_multiplier * config.hidden_size
    config.intermediate_size = up_project_multiplier * config.intermediate_size
    if config.num_key_value_heads != 1:
        config.num_key_value_heads = (
            embedding_dim_multiplier * config.num_key_value_heads
        )
    config.num_attention_heads = embedding_dim_multiplier * config.num_attention_heads
    config.tie_word_embeddings = False

    # Rename config according to expansion factors
    config = rename_config(config, embedding_dim_multiplier, up_project_multiplier)

    # Create destination network
    dst_network = type(src_network)._from_config(config)

    # Clone embeddings
    dst_network.model.embed_tokens.weight.data = clone_matrix(
        dst_network.model.embed_tokens.weight.data.shape,
        src_network.model.embed_tokens.weight.data,
        normalize=False,
    )

    # Clone each layer
    for dst_layer, src_layer in zip(dst_network.model.layers, src_network.model.layers):
        clone_rms_norm(dst_layer.input_layernorm, src_layer.input_layernorm)
        clone_rms_norm(
            dst_layer.post_attention_layernorm, src_layer.post_attention_layernorm
        )
        clone_qwen_attention(dst_layer.self_attn, src_layer.self_attn, snr_db=snr_db)
        clone_linear_layer(
            dst_layer.mlp.gate_proj,
            src_layer.mlp.gate_proj,
            snr_db=snr_db,  # type: ignore
        )
        clone_linear_layer(dst_layer.mlp.up_proj, src_layer.mlp.up_proj, snr_db=snr_db)  # type: ignore
        clone_linear_layer(
            dst_layer.mlp.down_proj,
            src_layer.mlp.down_proj,
            snr_db=snr_db,  # type: ignore
        )

    clone_rms_norm(dst_network.model.norm, src_network.model.norm)
    clone_linear_layer(dst_network.lm_head, src_network.lm_head)
    return dst_network


def clone_smollm2(
    src_network,
    embedding_dim_multiplier: int = 1,
    up_project_multiplier: int = 1,
    **kwargs,
):
    """
    Cloning function for SmolLM2 models.
    """
    snr_db = kwargs.get("snr_db", None)

    # Set destination config
    config = copy.deepcopy(src_network.config)
    config.hidden_size = embedding_dim_multiplier * config.hidden_size
    config.intermediate_size = up_project_multiplier * config.intermediate_size
    config.num_attention_heads = embedding_dim_multiplier * config.num_attention_heads
    config.num_key_value_heads = embedding_dim_multiplier * config.num_key_value_heads
    config.tie_word_embeddings = False

    # Rename config
    config = rename_config(config, embedding_dim_multiplier, up_project_multiplier)

    # Create destination network
    dst_network = type(src_network)._from_config(config)

    # Clone embeddings
    dst_network.model.embed_tokens.weight.data = clone_matrix(
        dst_network.model.embed_tokens.weight.data.shape,
        src_network.model.embed_tokens.weight.data,
        normalize=False,
    )

    # Clone each layer
    for dst_layer, src_layer in zip(dst_network.model.layers, src_network.model.layers):
        clone_rms_norm(dst_layer.input_layernorm, src_layer.input_layernorm)
        clone_rms_norm(
            dst_layer.post_attention_layernorm, src_layer.post_attention_layernorm
        )
        clone_smollm_attention(dst_layer.self_attn, src_layer.self_attn, snr_db=snr_db)
        clone_linear_layer(  # type: ignore
            dst_layer.mlp.gate_proj, src_layer.mlp.gate_proj, snr_db=snr_db
        )
        clone_linear_layer(dst_layer.mlp.up_proj, src_layer.mlp.up_proj, snr_db=snr_db)
        clone_linear_layer(
            dst_layer.mlp.down_proj, src_layer.mlp.down_proj, snr_db=snr_db
        )

    clone_rms_norm(dst_network.model.norm, src_network.model.norm)
    clone_linear_layer(dst_network.lm_head, src_network.lm_head)
    return dst_network


def clone_smollm3(
    src_network,
    embedding_dim_multiplier: int = 1,
    up_project_multiplier: int = 1,
    **kwargs,
):
    """
    Cloning function for SmolLM3 models.
    """
    snr_db = kwargs.get("snr_db", None)

    # Set destination config
    config = copy.deepcopy(src_network.config)
    config.hidden_size = embedding_dim_multiplier * config.hidden_size
    config.intermediate_size = up_project_multiplier * config.intermediate_size
    config.num_attention_heads = embedding_dim_multiplier * config.num_attention_heads
    config.num_key_value_heads = embedding_dim_multiplier * config.num_key_value_heads
    config.tie_word_embeddings = False

    # Rename config
    config = rename_config(config, embedding_dim_multiplier, up_project_multiplier)

    # Create destination network
    dst_network = type(src_network)._from_config(config)

    # Clone embeddings
    dst_network.model.embed_tokens.weight.data = clone_matrix(
        dst_network.model.embed_tokens.weight.data.shape,
        src_network.model.embed_tokens.weight.data,
        normalize=False,
    )

    # Clone each layer
    for dst_layer, src_layer in zip(dst_network.model.layers, src_network.model.layers):
        clone_rms_norm(dst_layer.input_layernorm, src_layer.input_layernorm)
        clone_rms_norm(
            dst_layer.post_attention_layernorm, src_layer.post_attention_layernorm
        )
        clone_smollm_attention(dst_layer.self_attn, src_layer.self_attn, snr_db=snr_db)
        clone_linear_layer(  # type: ignore
            dst_layer.mlp.gate_proj, src_layer.mlp.gate_proj, snr_db=snr_db
        )
        clone_linear_layer(dst_layer.mlp.up_proj, src_layer.mlp.up_proj, snr_db=snr_db)
        clone_linear_layer(
            dst_layer.mlp.down_proj, src_layer.mlp.down_proj, snr_db=snr_db
        )

    clone_rms_norm(dst_network.model.norm, src_network.model.norm)
    clone_linear_layer(dst_network.lm_head, src_network.lm_head)
    return dst_network


def clone_llama(
    src_network,
    embedding_dim_multiplier: int = 1,
    up_project_multiplier: int = 1,
    **kwargs,
):
    """
    Cloning function for Llama models.
    """
    snr_db = kwargs.get("snr_db", None)

    # Set the destination network config
    config = copy.deepcopy(src_network.config)
    config.hidden_size = embedding_dim_multiplier * config.hidden_size
    config.intermediate_size = up_project_multiplier * config.intermediate_size
    config.num_attention_heads = embedding_dim_multiplier * config.num_attention_heads
    config.num_key_value_heads = embedding_dim_multiplier * config.num_key_value_heads
    config.tie_word_embeddings = False

    # Rename config according to expansion factors
    config = rename_config(config, embedding_dim_multiplier, up_project_multiplier)

    # Create destination network
    dst_network = type(src_network)._from_config(config)

    # Clone embeddings
    dst_network.model.embed_tokens.weight.data = clone_matrix(
        dst_network.model.embed_tokens.weight.data.shape,
        src_network.model.embed_tokens.weight.data,
        normalize=False,
    )

    # Clone each layer
    for dst_layer, src_layer in zip(dst_network.model.layers, src_network.model.layers):
        clone_rms_norm(dst_layer.input_layernorm, src_layer.input_layernorm)
        clone_rms_norm(
            dst_layer.post_attention_layernorm, src_layer.post_attention_layernorm
        )
        clone_llama_attention(dst_layer.self_attn, src_layer.self_attn, snr_db=snr_db)
        clone_linear_layer(
            dst_layer.mlp.gate_proj,
            src_layer.mlp.gate_proj,
            snr_db=snr_db,  # type: ignore
        )
        clone_linear_layer(dst_layer.mlp.up_proj, src_layer.mlp.up_proj, snr_db=snr_db)  # type: ignore
        clone_linear_layer(
            dst_layer.mlp.down_proj,
            src_layer.mlp.down_proj,
            snr_db=snr_db,  # type: ignore
        )

    clone_rms_norm(dst_network.model.norm, src_network.model.norm)
    clone_linear_layer(dst_network.lm_head, src_network.lm_head)
    return dst_network


def clone_phi_1(
    src_network,
    embedding_dim_multiplier: int = 1,
    up_project_multiplier: int = 1,
    **kwargs,
):
    """
    Cloning function for Phi-1.5 models.
    """
    snr_db = kwargs.get("snr_db", None)

    # Set destination config
    config = copy.deepcopy(src_network.config)
    config.hidden_size = embedding_dim_multiplier * config.hidden_size
    config.intermediate_size = up_project_multiplier * config.intermediate_size
    config.num_attention_heads = embedding_dim_multiplier * config.num_attention_heads
    config.num_key_value_heads = embedding_dim_multiplier * config.num_key_value_heads
    config.tie_word_embeddings = False

    # Rename config
    config = rename_config(config, embedding_dim_multiplier, up_project_multiplier)

    # Create destination network
    dst_network = type(src_network)._from_config(config)

    # Clone embeddings
    dst_network.model.embed_tokens.weight.data = clone_matrix(
        dst_network.model.embed_tokens.weight.data.shape,
        src_network.model.embed_tokens.weight.data,
        normalize=False,
    )

    # Clone each layer
    for dst_layer, src_layer in zip(dst_network.model.layers, src_network.model.layers):
        clone_layer_norm(dst_layer.input_layernorm, src_layer.input_layernorm)
        clone_phi_attention(dst_layer.self_attn, src_layer.self_attn, snr_db=snr_db)
        clone_linear_layer(
            dst_layer.mlp.fc1,
            src_layer.mlp.fc1,
            snr_db=snr_db,  # type: ignore
        )
        clone_linear_layer(
            dst_layer.mlp.fc2,
            src_layer.mlp.fc2,
            snr_db=snr_db,  # type: ignore
        )

    clone_layer_norm(
        dst_network.model.final_layernorm, src_network.model.final_layernorm
    )
    clone_linear_layer(dst_network.lm_head, src_network.lm_head)
    return dst_network


def clone_qwen_attention(dst_layer, src_layer, snr_db=None):
    """
    Clones the attention layer for Qwen models.
    """
    # Handle Qwen attention layers
    clone_qkv_layer(
        dst_layer.q_proj,
        src_layer.q_proj,
        dst_layer.config.num_attention_heads,
        src_layer.config.num_attention_heads,
        snr_db=snr_db,
    )
    clone_qkv_layer(
        dst_layer.k_proj,
        src_layer.k_proj,
        dst_layer.config.num_key_value_heads,
        src_layer.config.num_key_value_heads,
        snr_db=snr_db,
    )
    clone_qkv_layer(
        dst_layer.v_proj,
        src_layer.v_proj,
        dst_layer.config.num_key_value_heads,
        src_layer.config.num_key_value_heads,
        snr_db=snr_db,
    )
    clone_linear_layer(dst_layer.o_proj, src_layer.o_proj, snr_db=snr_db)

    # Clone Qwen-specific norms if they exist
    if hasattr(src_layer, "q_norm") and hasattr(dst_layer, "q_norm"):
        clone_rms_norm(dst_layer.q_norm, src_layer.q_norm)

    if hasattr(src_layer, "k_norm") and hasattr(dst_layer, "k_norm"):
        clone_rms_norm(dst_layer.k_norm, src_layer.k_norm)


def clone_llama_attention(dst_layer, src_layer, snr_db=None):
    """
    Clones the attention layer for Llama models.
    """
    clone_qkv_layer(
        dst_layer.q_proj,
        src_layer.q_proj,
        dst_layer.config.num_attention_heads,
        src_layer.config.num_attention_heads,
        snr_db=snr_db,
    )

    clone_qkv_layer(
        dst_layer.k_proj,
        src_layer.k_proj,
        dst_layer.config.num_key_value_heads,
        src_layer.config.num_key_value_heads,
        snr_db=snr_db,
    )

    clone_qkv_layer(
        dst_layer.v_proj,
        src_layer.v_proj,
        dst_layer.config.num_key_value_heads,
        src_layer.config.num_key_value_heads,
        snr_db=snr_db,
    )

    clone_linear_layer(dst_layer.o_proj, src_layer.o_proj, snr_db=snr_db)


def clone_smollm_attention(dst_layer, src_layer, snr_db=None):
    """
    Clones attention layer for SmolLM models.
    """
    clone_linear_layer(dst_layer.q_proj, src_layer.q_proj, snr_db=snr_db)
    clone_linear_layer(dst_layer.k_proj, src_layer.k_proj, snr_db=snr_db)
    clone_linear_layer(dst_layer.v_proj, src_layer.v_proj, snr_db=snr_db)
    clone_linear_layer(dst_layer.o_proj, src_layer.o_proj, snr_db=snr_db)


def clone_phi_attention(dst_layer, src_layer, snr_db=None):
    """
    Clones the attention layer for Phi models.
    """
    clone_linear_layer(dst_layer.q_proj, src_layer.q_proj, snr_db=snr_db)
    clone_linear_layer(dst_layer.k_proj, src_layer.k_proj, snr_db=snr_db)
    clone_linear_layer(dst_layer.v_proj, src_layer.v_proj, snr_db=snr_db)
    clone_linear_layer(dst_layer.dense, src_layer.dense, snr_db=snr_db)


def clone_olmo_attention(dst_layer, src_layer, snr_db=None):
    """
    Clones the attention layer for OLMo-2 models.
    """
    clone_linear_layer(dst_layer.q_proj, src_layer.q_proj, snr_db=snr_db)
    clone_linear_layer(dst_layer.k_proj, src_layer.k_proj, snr_db=snr_db)
    clone_linear_layer(dst_layer.v_proj, src_layer.v_proj, snr_db=snr_db)
    clone_linear_layer(dst_layer.o_proj, src_layer.o_proj, snr_db=snr_db)

    # Clone Q/K norms if they exist
    if hasattr(src_layer, "q_norm") and hasattr(dst_layer, "q_norm"):
        clone_rms_norm(dst_layer.q_norm, src_layer.q_norm)

    if hasattr(src_layer, "k_norm") and hasattr(dst_layer, "k_norm"):
        clone_rms_norm(dst_layer.k_norm, src_layer.k_norm)


def clone_olmo_2(
    src_network,
    embedding_dim_multiplier: int = 1,
    up_project_multiplier: int = 1,
    **kwargs,
):
    """
    Cloning function for OLMo-2 models.
    """
    snr_db = kwargs.get("snr_db", None)

    # Set the destination network config
    config = copy.deepcopy(src_network.config)
    config.hidden_size = embedding_dim_multiplier * config.hidden_size
    config.intermediate_size = up_project_multiplier * config.intermediate_size
    config.num_attention_heads = embedding_dim_multiplier * config.num_attention_heads
    config.num_key_value_heads = embedding_dim_multiplier * config.num_key_value_heads
    config.tie_word_embeddings = False

    # Rename config according to expansion factors
    config = rename_config(config, embedding_dim_multiplier, up_project_multiplier)

    # Create destination network
    dst_network = type(src_network)._from_config(config)

    # Clone embeddings
    dst_network.model.embed_tokens.weight.data = clone_matrix(
        dst_network.model.embed_tokens.weight.data.shape,
        src_network.model.embed_tokens.weight.data,
        normalize=False,
    )

    # Clone each layer
    for dst_layer, src_layer in zip(dst_network.model.layers, src_network.model.layers):
        clone_rms_norm(
            dst_layer.post_attention_layernorm,
            src_layer.post_attention_layernorm,
        )

        clone_rms_norm(
            dst_layer.post_feedforward_layernorm,
            src_layer.post_feedforward_layernorm,
        )

        clone_olmo_attention(
            dst_layer.self_attn,
            src_layer.self_attn,
            snr_db=snr_db,
        )

        clone_linear_layer(
            dst_layer.mlp.gate_proj,
            src_layer.mlp.gate_proj,
            snr_db=snr_db,
        )

        clone_linear_layer(
            dst_layer.mlp.up_proj,
            src_layer.mlp.up_proj,
            snr_db=snr_db,
        )

        clone_linear_layer(
            dst_layer.mlp.down_proj,
            src_layer.mlp.down_proj,
            snr_db=snr_db,
        )

    clone_rms_norm(dst_network.model.norm, src_network.model.norm)
    clone_linear_layer(dst_network.lm_head, src_network.lm_head)
    return dst_network


# Registry of supported cloning functions
REGISTERED_CLONING_FUNCTIONS = {
    "Qwen2Config": clone_qwen2_5,
    "Qwen3Config": clone_qwen3,
    "SmolLM2Config": clone_smollm2,
    "SmolLM3Config": clone_smollm3,
    "LlamaConfig": clone_llama,
    "PhiConfig": clone_phi_1,
    "Olmo2Config": clone_olmo_2,
}
