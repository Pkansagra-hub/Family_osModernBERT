"""
Pooling Strategies for Encoder Outputs

This module contains various pooling strategies to convert
token-level encoder outputs into fixed-size representations.

Pooling Methods:
    - CLSPooler: Use [CLS] token representation
    - MeanPooler: Average all token representations
    - MaxPooler: Max pooling over tokens
    - WeightedMeanPooler: Attention-weighted mean
    - LastTokenPooler: Use last non-padding token

Each pooler handles attention masks properly to ignore padding tokens.

Usage:
    pooler = MeanPooler()
    sentence_embedding = pooler(
        hidden_states,      # (batch, seq_len, hidden_size)
        attention_mask      # (batch, seq_len)
    )  # -> (batch, hidden_size)
"""

# TODO: Implement BasePooler abstract class
#   - forward(hidden_states, attention_mask) -> pooled_output

# TODO: Implement CLSPooler
#   - Extract hidden_states[:, 0, :]
#   - Optionally pass through dense + activation

# TODO: Implement MeanPooler
#   - Masked mean over sequence dimension
#   - Handle attention_mask properly

# TODO: Implement MaxPooler
#   - Masked max over sequence dimension
#   - Set padding positions to -inf before max

# TODO: Implement WeightedMeanPooler
#   - Learn attention weights over tokens
#   - Compute weighted sum

# TODO: Implement LastTokenPooler
#   - Find last non-padding token per sequence
#   - Extract that token's representation
#   - Useful for causal/decoder models
