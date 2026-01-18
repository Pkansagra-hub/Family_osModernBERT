# test_normalization.py
import torch

# Simulate UltraBERT output
raw_embeddings = torch.randn(1, 38, 768) * 1.2  # Similar to your stats
raw_embeddings[0, 10, 100] = 20.0  # Add extreme value like UltraBERT does
raw_embeddings[0, 15, 200] = -18.0

print("Raw embeddings:")
print(f"  Mean: {raw_embeddings.mean():.4f}")
print(f"  Std: {raw_embeddings.std():.4f}")
print(f"  Min/Max: {raw_embeddings.min():.4f}/{raw_embeddings.max():.4f}")
print(f"  Norm: {torch.norm(raw_embeddings, dim=-1).mean():.4f}")

# Normalize
norm = torch.norm(raw_embeddings, dim=-1, keepdim=True)
normalized = raw_embeddings / (norm + 1e-8)

print("\nNormalized embeddings:")
print(f"  Mean: {normalized.mean():.4f}")
print(f"  Std: {normalized.std():.4f}")
print(f"  Min/Max: {normalized.min():.4f}/{normalized.max():.4f}")
print(f"  Norm: {torch.norm(normalized, dim=-1).mean():.4f}")

# Check if direction is preserved
cosine_sim = torch.cosine_similarity(
    raw_embeddings.view(-1, 768),
    normalized.view(-1, 768),
    dim=-1
)
print(f"\nCosine similarity (should be ~1.0): {cosine_sim.mean():.4f}")
