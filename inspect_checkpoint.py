"""Inspect checkpoint-500 to see what's inside."""

import torch

checkpoint_path = "d:/Modeling_studio/checkpoints/checkpoint-500/pytorch_model.bin"
state_dict = torch.load(checkpoint_path, weights_only=True, map_location="cpu")

print(f"Total keys: {len(state_dict)}")
print()

# Group by prefix
prefixes = {}
for key in state_dict.keys():
    prefix = key.split(".")[0]
    if prefix not in prefixes:
        prefixes[prefix] = []
    prefixes[prefix].append(key)

for prefix, keys in sorted(prefixes.items()):
    print(f"{prefix}: {len(keys)} keys")
    for k in keys[:3]:
        print(f"  {k}: {state_dict[k].shape}")
    if len(keys) > 3:
        print(f"  ... ({len(keys)-3} more)")
