import torch
import time
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file, load_file

# Load original model and tokenizer
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

# Move to CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Print model structure before pruning
print("Model structure before pruning:")
print(model)

# Count parameters before pruning
params_before = sum(p.numel() for p in model.parameters())
print(f"Number of parameters before pruning: {params_before}")

# Function to generate token by token and measure TPS
def generate_token_by_token(model, tokenizer, prompt, max_new_tokens=20):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    generated_ids = []
    start_time = time.time()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            generated_ids.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    end_time = time.time()
    generated_text = tokenizer.decode(generated_ids)
    tokens_generated = len(generated_ids)
    tps = tokens_generated / (end_time - start_time) if (end_time - start_time) > 0 else 0
    return generated_text, tps

# Generate before pruning
prompt = "The future of AI is"
before_output, before_tps = generate_token_by_token(model, tokenizer, prompt)
print("Generation before pruning:")
print(before_output)
print(f"Tokens per second before pruning: {before_tps}")

# Pruning function using SVD low-rank approximation
def prune_model(model, threshold=0.99):
    rs = {}
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.data
            U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
            energy = (S ** 2).cumsum() / (S ** 2).sum()
            try:
                r = ((energy > threshold).nonzero(as_tuple=True)[0][0] + 1).item()
            except IndexError:
                r = len(S)  # If no r satisfies, take full rank
            rs[name] = r
            U = U[:, :r]
            S = S[:r]
            Vh = Vh[:r, :]
            linear1 = torch.nn.Linear(module.in_features, r, bias=False, device=device)
            linear1.weight.data = Vh
            linear2 = torch.nn.Linear(r, module.out_features, bias=(module.bias is not None), device=device)
            linear2.weight.data = U * S[None, :]
            if module.bias is not None:
                linear2.bias.data = module.bias.data
            new_module = torch.nn.Sequential(linear1, linear2)
            # Replace in parent
            if '.' in name:
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            setattr(parent, child_name, new_module)
    return rs

# Apply pruning
rs = prune_model(model)

# Print model structure after pruning
print("Model structure after pruning:")
print(model)

# Count parameters after pruning
params_after = sum(p.numel() for p in model.parameters())
print(f"Number of parameters after pruning: {params_after}")

# Generate after pruning
after_output, after_tps = generate_token_by_token(model, tokenizer, prompt)
print("Generation after pruning:")
print(after_output)
print(f"Tokens per second after pruning: {after_tps}")

# Save pruned model
pruned_path = "pruned_qwen3_0.6b"
model.save_pretrained(pruned_path, safe_serialization=True)
tokenizer.save_pretrained(pruned_path)
with open(f"{pruned_path}/rs.json", 'w') as f:
    json.dump(rs, f)

# Function to load the pruned model
def load_pruned_model(pruned_path, original_model_name):
    tokenizer = AutoTokenizer.from_pretrained(pruned_path)
    model = AutoModelForCausalLM.from_pretrained(original_model_name, torch_dtype=torch.float32)
    model.to(device)
    with open(f"{pruned_path}/rs.json", 'r') as f:
        rs = json.load(f)
    for name, r in rs.items():
        module = dict(model.named_modules())[name]
        linear1 = torch.nn.Linear(module.in_features, r, bias=False, device=device)
        linear2 = torch.nn.Linear(r, module.out_features, bias=(module.bias is not None), device=device)
        new_module = torch.nn.Sequential(linear1, linear2)
        if '.' in name:
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
        setattr(parent, child_name, new_module)
    # Load state dict
    state_dict = load_file(f"{pruned_path}/model.safetensors")
    model.load_state_dict(state_dict)
    return model, tokenizer

# Example usage to load:
# pruned_model, pruned_tokenizer = load_pruned_model("pruned_qwen3_0.6b", "Qwen/Qwen3-0.6B")
