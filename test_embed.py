"""Quick script to analyze token embeddings between V1 and V3."""
from safetensors.torch import load_file
from pathlib import Path
import torch

# Check V1 (pytorch_model.bin)
print('=== V1 Checkpoint ===')
v1_path = Path('D:/Modeling_studio/outputs/ultrabert-gen-decoder-v1/pytorch_model.bin')
v1_dict = torch.load(str(v1_path), map_location='cpu', weights_only=True)
wte_v1 = v1_dict[[k for k in v1_dict if 'wte.weight' in k][0]]
print(f'GPT-2 tokens (0-50256) norm: mean={wte_v1[:50257].norm(dim=1).mean().item():.2f}')
print(f'New tokens (50257-50367) norm: mean={wte_v1[50257:].norm(dim=1).mean().item():.2f}')
print(f'BOS (50281) norm: {wte_v1[50281].norm().item():.2f}')

print('\n=== V3 Checkpoint ===')
v3_path = Path('D:/Modeling_studio/outputs/ultrabert-gen-decoder-v3/model.safetensors')
v3_dict = load_file(str(v3_path))
wte_v3 = v3_dict[[k for k in v3_dict if 'wte.weight' in k][0]]
print(f'GPT-2 tokens (0-50256) norm: mean={wte_v3[:50257].norm(dim=1).mean().item():.2f}')
print(f'New tokens (50257-50367) norm: mean={wte_v3[50257:].norm(dim=1).mean().item():.2f}')
print(f'BOS (50281) norm: {wte_v3[50281].norm().item():.2f}')

# Key observation: Did the special tokens change from v1 to v3?
print('\n=== V1 vs V3 difference ===')
bos_diff = (wte_v1[50281] - wte_v3[50281]).norm().item()
eos_diff = (wte_v1[50282] - wte_v3[50282]).norm().item()
pad_diff = (wte_v1[50283] - wte_v3[50283]).norm().item()
print(f'BOS diff norm: {bos_diff:.4f}')
print(f'EOS diff norm: {eos_diff:.4f}')
print(f'PAD diff norm: {pad_diff:.4f}')

# Compare a random GPT-2 token
rand_diff = (wte_v1[1000] - wte_v3[1000]).norm().item()
print(f'Random GPT-2 token (1000) diff: {rand_diff:.4f}')
