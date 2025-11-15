import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from safetensors.torch import save_file, load_file
import time
from copy import deepcopy
import json

# Configuration
MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
COMPRESSION_ENERGY_THRESHOLD = 0.15  # epsilon: lower = more aggressive compression
PROMPT = "The future of AI is"
MAX_NEW_TOKENS = 20

def compress_linear_layer(layer, epsilon=0.15):
    """
    Compress a Linear layer using SVD low-rank approximation.
    Replaces Linear(in_features, out_features) with Sequential(
        Linear(in_features, rank, bias=False),
        Linear(rank, out_features, bias=True)
    )
    """
    if not isinstance(layer, nn.Linear):
        return layer, 0, 0

    # Get original dtype and device
    original_dtype = layer.weight.dtype
    device = layer.weight.device

    # Get weight matrix and convert to float32 for SVD
    W = layer.weight.data.to(torch.float32)
    out_features, in_features = W.shape

    # Perform SVD on CPU if needed (more stable for some operations)
    try:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    except NotImplementedError:
        # Fallback: move to CPU for SVD
        W_cpu = W.cpu()
        U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
        U = U.to(device)
        S = S.to(device)
        Vh = Vh.to(device)

    # Determine rank based on energy threshold
    energy = torch.cumsum(S**2, dim=0) / torch.sum(S**2)
    rank = torch.searchsorted(energy, 1 - epsilon).item() + 1
    rank = max(1, min(rank, min(in_features, out_features) - 1))

    # Calculate parameter counts
    old_params = W.numel() + (layer.bias.numel() if layer.bias is not None else 0)
    new_params = rank * (in_features + out_features) + (layer.bias.numel() if layer.bias is not None else 0)

    # Only compress if it reduces parameters
    if new_params >= old_params:
        return layer, old_params, old_params

    # Create low-rank factorization: W ≈ U_r @ diag(S_r) @ Vh_r
    U_r = U[:, :rank] @ torch.diag(S[:rank])
    V_r = Vh[:rank, :]

    # Convert back to original dtype
    U_r = U_r.to(original_dtype)
    V_r = V_r.to(original_dtype)

    # Build compressed layer as two sequential linear layers
    compressed_layer = nn.Sequential(
        nn.Linear(in_features, rank, bias=False),
        nn.Linear(rank, out_features, bias=layer.bias is not None)
    )

    # Move to device and set dtype
    compressed_layer = compressed_layer.to(device)
    compressed_layer = compressed_layer.to(original_dtype)

    # Assign weights
    compressed_layer[0].weight.data = V_r
    compressed_layer[1].weight.data = U_r

    # Copy bias if present
    if layer.bias is not None:
        compressed_layer[1].bias.data = layer.bias.data

    return compressed_layer, old_params, new_params

def compress_model(model, epsilon=0.15):
    """
    Recursively compress all Linear layers in the model using SVD.
    """
    total_original = 0
    total_compressed = 0
    modifications = []

    def recursive_replace(module, parent_name=""):
        nonlocal total_original, total_compressed

        for name, child in list(module.named_children()):
            full_name = f"{parent_name}.{name}" if parent_name else name

            if isinstance(child, nn.Linear):
                print(f"Compressing Linear layer: {full_name}")
                try:
                    new_layer, old_size, new_size = compress_linear_layer(child, epsilon)
                    setattr(module, name, new_layer)
                    total_original += old_size
                    total_compressed += new_size
                    saved = old_size - new_size
                    pct = (saved / old_size) * 100 if old_size > 0 else 0
                    modifications.append({
                        'name': full_name,
                        'original': old_size,
                        'compressed': new_size,
                        'saved': saved,
                        'pct_saved': pct
                    })
                except Exception as e:
                    print(f"  Error compressing {full_name}: {e}")
                    # Skip this layer on error
                    total_original += child.weight.numel() + (child.bias.numel() if child.bias is not None else 0)
                    total_compressed += child.weight.numel() + (child.bias.numel() if child.bias is not None else 0)
            else:
                # Recurse into nested modules
                recursive_replace(child, full_name)

    recursive_replace(model)

    # Add embedding and lm_head parameters (which we don't compress)
    for name, param in model.named_parameters():
        if "embed" in name or "lm_head" in name:
            total_original += param.numel()
            total_compressed += param.numel()

    return model, total_original, total_compressed, modifications

def print_model_info(model, title="Model Structure"):
    """Print a summary of the model structure."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            print(f"{name}: {tuple(param.shape)} = {param.numel():,} params")

    print(f"\nTotal parameters: {total_params:,}")
    print(f"{'='*60}\n")

def count_trainable_params(model):
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def generate_and_benchmark(model, tokenizer, prompt, max_new_tokens=20):
    """
    Generate text token-by-token and measure tokens per second.
    """
    model.eval()

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids

    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    # Measure generation time
    start_time = time.time()

    tokens_generated = 0
    generated_text = prompt

    # Generate token by token
    current_input_ids = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(current_input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append token and decode
            current_input_ids = torch.cat([current_input_ids, next_token_id], dim=-1)
            token_str = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            generated_text += token_str
            tokens_generated += 1

            # Stop if EOS token
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    end_time = time.time()
    elapsed_time = end_time - start_time
    tokens_per_second = tokens_generated / elapsed_time if elapsed_time > 0 else 0

    return {
        'generated_text': generated_text,
        'tokens_generated': tokens_generated,
        'elapsed_time': elapsed_time,
        'tokens_per_second': tokens_per_second
    }

def main():
    print(f"Loading model: {MODEL_NAME}")
    print(f"Using device: {DEVICE}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in float32 initially for SVD stability, then convert to float16
    print("Loading model in float32 for compression...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True
    )

    # Convert to float16 after loading for memory efficiency
    if DEVICE == "cuda":
        model = model.half()

    model.to(DEVICE)

    print(f"Model loaded successfully!")
    print_model_info(model, "ORIGINAL MODEL STRUCTURE")

    original_params = count_trainable_params(model)
    print(f"Total parameters before pruning: {original_params:,}")

    # Benchmark original model
    print(f"\nRunning inference on prompt: '{PROMPT}'")
    original_result = generate_and_benchmark(model, tokenizer, PROMPT, MAX_NEW_TOKENS)
    print(f"Generated text: {original_result['generated_text']}")
    print(f"Tokens generated: {original_result['tokens_generated']}")
    print(f"Elapsed time: {original_result['elapsed_time']:.3f}s")
    print(f"Tokens per second: {original_result['tokens_per_second']:.2f}")

    # Compress model
    print(f"\n{'='*60}")
    print(f"COMPRESSING MODEL WITH SVD (epsilon={COMPRESSION_ENERGY_THRESHOLD})")
    print(f"{'='*60}")

    compressed_model = deepcopy(model)
    compressed_model, total_original, total_compressed, modifications = compress_model(
        compressed_model, epsilon=COMPRESSION_ENERGY_THRESHOLD
    )

    # Move compressed model to device
    compressed_model.to(DEVICE)

    # Print compression summary
    print(f"\nCompression Summary:")
    print(f"{'-'*60}")
    for mod in modifications:
        print(f"{mod['name']}: {mod['original']:,} → {mod['compressed']:,} "
              f"(saved {mod['saved']:,}, {mod['pct_saved']:.1f}%)")

    print(f"\n{'='*60}")
    print_model_info(compressed_model, "COMPRESSED MODEL STRUCTURE")

    compressed_params = count_trainable_params(compressed_model)
    print(f"Total parameters after pruning: {compressed_params:,}")

    # Calculate overall compression
    compression_ratio = original_params / compressed_params
    size_reduction_pct = ((original_params - compressed_params) / original_params) * 100
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Size reduction: {size_reduction_pct:.2f}%")

    # Benchmark compressed model
    print(f"\nRunning inference on prompt: '{PROMPT}' (Compressed Model)")
    compressed_result = generate_and_benchmark(compressed_model, tokenizer, PROMPT, MAX_NEW_TOKENS)
    print(f"Generated text: {compressed_result['generated_text']}")
    print(f"Tokens generated: {compressed_result['tokens_generated']}")
    print(f"Elapsed time: {compressed_result['elapsed_time']:.3f}s")
    print(f"Tokens per second: {compressed_result['tokens_per_second']:.2f}")

    # Performance comparison
    speedup = compressed_result['tokens_per_second'] / original_result['tokens_per_second']
    print(f"\nSpeedup: {speedup:.2f}x")

    # Save compressed model
    save_dir = "./qwen3_0.6b_svd_compressed"
    print(f"\nSaving compressed model to {save_dir}")

    # Save model weights in safetensors format
    compressed_model.save_pretrained(
        save_dir,
        safe_serialization=True,  # Use safetensors
        max_shard_size="5GB"
    )

    # Save tokenizer
    tokenizer.save_pretrained(save_dir)

    # Save compression info
    compression_info = {
        "original_model": MODEL_NAME,
        "compression_energy_threshold": COMPRESSION_ENERGY_THRESHOLD,
        "original_parameters": original_params,
        "compressed_parameters": compressed_params,
        "compression_ratio": compression_ratio,
        "size_reduction_percent": size_reduction_pct,
        "layer_modifications": modifications,
        "generation_speedup": speedup,
        "original_tps": original_result['tokens_per_second'],
        "compressed_tps": compressed_result['tokens_per_second']
    }
    with open(f"{save_dir}/compression_info.json", "w") as f:
        json.dump(compression_info, f, indent=2)

    print(f"Compressed model saved successfully!")
    print(f"Files saved:")
    print(f"  - Model weights (safetensors format)")
    print(f"  - Tokenizer files")
    print(f"  - config.json")
    print(f"  - compression_info.json")

    # Verify loading
    print(f"\n{'='*60}")
    print("Verifying compressed model can be loaded...")
    loaded_model = AutoModelForCausalLM.from_pretrained(
        save_dir,
        torch_dtype=torch.float16,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True
    )
    loaded_tokenizer = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=True)

    test_result = generate_and_benchmark(loaded_model, loaded_tokenizer, PROMPT, 10)
    print(f"Verification successful! Generated: {test_result['generated_text']}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
