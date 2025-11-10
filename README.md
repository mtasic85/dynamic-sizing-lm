# dynamic-sizing-lm

Upscaling and Downscaling Language Models

## Upscaling

As introduced in the research paper "Scaling Smart: Accelerating Large Language Model Pre-training with Small Model Initialization," employs the HyperCloning method to efficiently transfer knowledge from a smaller pre-trained transformer-based LLM to a larger target LLM.

This function-preserving transformation expands the hidden dimensions of the source model, precisely mapping its linear layer parameters into the destination network so that the larger model exactly replicates the source’s hidden representations and output logits at initialization.

As a result, the upscaled LLM starts with the full accuracy of the smaller model, enabling faster convergence and improved performance through subsequent fine-tuning or continued pre-training—achieving superior results under limited computational budgets compared to training from scratch (see https://arxiv.org/html/2409.12903v2).

## Downscaling

Downscaling uses Singular Value Decomposition (SVD) for low-rank approximation to compress model weights while preserving performance. This mathematical approach decomposes weight matrices into low-rank components that capture the dominant variance, effectively reducing hidden dimensions while maintaining functional similarity to the original model.

The method applies layer-wise compression to attention and FFN layers, restructuring single linear layers into sequential low-rank approximations. It supports both fixed-rank truncation and energy-based thresholding (retaining 99% of variance by default), with optional robust SVD variants for improved quality retention.

## Limitations

- Dense transformer LLMs only
- Text-to-text
- Supported model architectures:
  - Qwen3
  - Qwen2/Qwen2.5
  - SmolLM3
  - SmolLM2
  - OLMo-2
  - Phi-1/Phi-1.5/Phi-2
  - Llama/TinyLlama

### Parameter Scaling

The HyperCloning method scales model parameters approximately as follows:
- Hidden dimensions (embeddings, attention projections): scale with n² where n = embed_dim_multiplier
- FFN intermediate dimensions: scale with m² where m = up_proj_multiplier
- Total parameters ≈ original × (n² × weight_attention + m² × weight_ffn + linear_terms)

For ~2x parameter increase, use asymmetric scaling like embed_dim_multiplier=1, up_proj_multiplier=2 (scales FFN quadratically while keeping attention size constant).

### Upscaling Limitations

- The current implementation requires `embed_dim_multiplier` and `up_proj_multiplier` to be integers; fractional values are not supported
- Although the destination network's output is valid, it may not be perfectly aligned with the source network due to numerical precision issues
- For attention layers, we recommend modifying only the number of attention heads while keeping `head_size` unchanged, as altering `head_size` would significantly increase implementation complexity

## CLI Tool

The `dslm.py` script provides a command-line interface for upscaling and downscaling language models.

### Installation

Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

### Usage

#### Describing Models

Get detailed information about a model's architecture and parameters:

```bash
# Describe Qwen3-0.6B model
python dslm.py desc --input Qwen/Qwen3-0.6B

# Describe Qwen2.5-0.5B model
python dslm.py desc --input Qwen/Qwen2.5-0.5B

# Describe SmolLM3-3B model
python dslm.py desc --input HuggingFaceTB/SmolLM3-3B

# Describe SmolLM2-360M model
python dslm.py desc --input HuggingFaceTB/SmolLM2-360M

# Describe OLMo-2-0425-1B model
python dslm.py desc --input allenai/OLMo-2-0425-1B

# Describe Phi-1.5 model
python dslm.py desc --input microsoft/phi-1_5

# Describe TinyLlama_v1.1 model
python dslm.py desc --input TinyLlama/TinyLlama_v1.1
```

#### Text Generation

Generate text using a model:

```bash
# Generate text with Qwen3-0.6B (default prompt)
python dslm.py gen --input Qwen/Qwen3-0.6B --prompt "The future of AI is"

# Generate text with Qwen2.5-0.5B (custom prompt)
python dslm.py gen --input Qwen/Qwen2.5-0.5B --prompt "The future of AI is"

# Generate text with SmolLM3-3B
python dslm.py gen --input HuggingFaceTB/SmolLM3-3B --prompt "The future of AI is"

# Generate text with SmolLM2-360M
python dslm.py gen --input HuggingFaceTB/SmolLM2-360M --prompt "The future of AI is"

# Generate text with OLMo-2-0425-1B
python dslm.py gen --input allenai/OLMo-2-0425-1B --prompt "The future of AI is"

# Generate text with Phi-1.5
python dslm.py gen --input microsoft/phi-1_5 --prompt "The future of AI is"

# Generate text with TinyLlama_v1.1
python dslm.py gen --input TinyLlama/TinyLlama_v1.1 --prompt "The future of AI is"
```

#### Upscaling Models

Upscale a model using the HyperCloning method. Note: Parameter count scales approximately with n² where n is the multiplier for hidden dimensions. For ~2x parameter increase, use asymmetric scaling (e.g., embed-dim-multiplier 1, up-proj-multiplier 2).

```bash
# Upscale Qwen3-0.6B for ~2x parameters (keep hidden size, double FFN)
python dslm.py up --input Qwen/Qwen3-0.6B --embed-dim-multiplier 1 --up-proj-multiplier 2
python dslm.py gen --input Qwen3-0.6B-1.0B --prompt "The future of AI is"

# Upscale Qwen2.5-0.5B for ~4x parameters (double both hidden and FFN dimensions)
python dslm.py up --input Qwen/Qwen2.5-0.5B --embed-dim-multiplier 2 --up-proj-multiplier 2
python dslm.py gen --input Qwen2.5-0.5B-2.0B --prompt "The future of AI is"

# Upscale SmolLM3-3B for ~4x parameters
python dslm.py up --input HuggingFaceTB/SmolLM3-3B --embed-dim-multiplier 2 --up-proj-multiplier 2 --output SmolLM3-3B-12.3B
python dslm.py gen --input SmolLM3-3B-12.3B --prompt "The future of AI is"

# Upscale SmolLM2-360M for ~4x parameters with custom output path
python dslm.py up --input HuggingFaceTB/SmolLM2-360M --embed-dim-multiplier 2 --up-proj-multiplier 2 --output SmolLM2-360M-1.4B
python dslm.py gen --input SmolLM2-360M-1.4B --prompt "The future of AI is"

# Upscale OLMo-2-0425-1B for ~4x parameters
python dslm.py up --input allenai/OLMo-2-0425-1B --embed-dim-multiplier 2 --up-proj-multiplier 2
python dslm.py gen --input OLMo-2-0425-1B-5.1B --prompt "The future of AI is"

# Upscale Phi-1.5 for ~4x parameters
python dslm.py up --input microsoft/phi-1_5 --embed-dim-multiplier 2 --up-proj-multiplier 2
python dslm.py gen --input phi-1_5-5.3B --prompt "The future of AI is"

# Upscale TinyLlama_v1.1 for ~4x parameters
python dslm.py up --input TinyLlama/TinyLlama_v1.1 --embed-dim-multiplier 2 --up-proj-multiplier 2
python dslm.py gen --input TinyLlama_v1.1-4.1B --prompt "The future of AI is"
```

#### Downscaling Models

Downscale models using SVD-based low-rank approximation. The method compresses attention and FFN layers while preserving 99% of variance by default.

```bash
# Downscale Qwen3-4B-Instruct to ~2B parameters using energy threshold
python dslm.py down --input Qwen/Qwen3-4B-Instruct-2507 --energy-threshold 0.99
# python dslm.py gen --input Qwen3-4B-Instruct-2507_downscaled_energy0.99 --prompt "The future of AI is"

# Downscale Qwen3-1.2B to ~0.6B parameters with fixed rank
python dslm.py down --input Qwen/Qwen3-0.6B --rank 512
# python dslm.py gen --input Qwen3-1.2B-0.6B --prompt "The future of AI is"

# Downscale Qwen2.5-1B to ~0.5B parameters
python dslm.py down --input Qwen/Qwen2.5-1B --energy-threshold 0.95
# python dslm.py gen --input Qwen2.5-1B_downscaled_energy0.95 --prompt "The future of AI is"

# Downscale SmolLM3-3B to ~1.5B parameters with custom rank
python dslm.py down --input HuggingFaceTB/SmolLM3-3B --rank 768 --output SmolLM3-3B-1.5B
# python dslm.py gen --input SmolLM3-3B-1.5B --prompt "The future of AI is"

# Downscale SmolLM2-1.4B to ~0.7B parameters
python dslm.py down --input HuggingFaceTB/SmolLM2-1.4B --energy-threshold 0.98
# python dslm.py gen --input SmolLM2-1.4B_downscaled_energy0.98 --prompt "The future of AI is"

# Downscale TinyLlama_v1.1 to ~2B parameters from 4.1B
python dslm.py down --input TinyLlama_v1.1-4.1B --rank 1024 --output TinyLlama_v1.1-2B
# python dslm.py gen --input TinyLlama_v1.1-2B --prompt "The future of AI is"
```

### Command Reference

- `desc`: Describe model architecture and parameters
  - `--input, -i`: Input model path or HuggingFace identifier

- `gen`: Generate text using the model
  - `--input, -i`: Input model path or HuggingFace identifier
  - `--prompt, -p`: Prompt for text generation (default: "The capital of France is")
  - `--n-predict, -n`: Number of tokens to generate (default: 100)
  - `--temperature`: Sampling temperature (default: 0.8)
  - `--top-k`: Top-k sampling (default: 40)
  - `--top-p`: Top-p sampling (default: 0.9)

- `up`: Upscale a model using HyperCloning
  - `--input, -i`: Input model path or HuggingFace identifier
  - `--output, -o`: Output path (auto-generated if not provided)
  - `--embed-dim-multiplier, -edm`: Integer multiplier for embedding dimensions
  - `--up-proj-multiplier, -upm`: Integer multiplier for FFN dimensions
  - `--snr-db`: Optional signal-to-noise ratio for adding noise

- `down`: Downscale a model using SVD low-rank approximation
  - `--input, -i`: Input model path or HuggingFace identifier
  - `--output, -o`: Output path for downscaled model
  - `--rank`: Target rank for low-rank approximation
  - `--energy-threshold`: Energy retention threshold (0.0-1.0, default: 0.99)
  - `--p-norm`: Norm for robust SVD (default: 2.0)
