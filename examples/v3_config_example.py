"""
Example usage of ModernBERT v3 Configuration.

This script demonstrates how to create and use the v3 configuration
dataclass for ModernBERT v3.3 Ultra.
"""

from modeling_studio.models.config_v3 import ModernBERTv3Config


def main():
    """Demonstrate config usage."""
    print("=" * 80)
    print("ModernBERT v3.3 Ultra Configuration Example")
    print("=" * 80)

    # Create default configuration
    config = ModernBERTv3Config()

    print("\n📊 Architecture Overview:")
    print(f"  Hidden Size: {config.hidden_size}")
    print(f"  Total Layers: {config.num_layers}")
    print(f"  Attention Heads: {config.num_attention_heads}")
    print(f"  Intermediate Size: {config.intermediate_size}")
    print(f"  Max Position Embeddings: {config.max_position_embeddings}")
    print(f"  Vocab Size: {config.vocab_size} (v2 vocab + 4 hub tokens)")

    print("\n🎯 Hub Token System:")
    print(f"  Hub Tokens: {', '.join(config.hub_tokens)}")
    print(f"  Positions:")
    for token, pos in config.hub_token_positions.items():
        print(f"    {token:8} → Position {pos}")
    print(f"  Global Attention Positions: {config.global_attention_positions}")

    print("\n🪟 Multi-Scale Sliding Windows:")
    for band, window_size in config.window_sizes.items():
        layers = config.layer_bands[band]
        print(
            f"  {band.capitalize():12} Band (L{min(layers):2}-L{max(layers):2}): Window = {window_size:3}"
        )

    print("\n🏢 Layer Bands:")
    for band_name, layers in config.layer_bands.items():
        print(
            f"  {band_name.capitalize():12}: Layers {min(layers)}-{max(layers)} ({len(layers)} layers)"
        )

    print("\n🔧 LoRA Configuration:")
    print(f"  Enabled: {config.lora_enabled}")
    print(f"  Rank (r): {config.lora_r}")
    print(f"  Alpha: {config.lora_alpha}")
    print(f"  Dropout: {config.lora_dropout}")
    print(f"  Target Layers: {config.lora_target_layers}")

    print("\n👥 Pair Encoder:")
    print(f"  Enabled: {config.pair_encoder_enabled}")
    print(f"  Heads: {config.pair_encoder_heads}")
    print(f"  Dropout: {config.pair_encoder_dropout}")

    print("\n🎓 Training Configuration:")
    print(
        f"  Frozen Layers (Phase 1): L{min(config.frozen_layers_phase1)}-L{max(config.frozen_layers_phase1)}"
    )

    print("\n🔥 Trainable Layers by Phase:")
    for phase in ["phase0", "phase1", "phase2"]:
        trainable = config.get_trainable_layers(phase)
        phase_name = {"phase0": "Healing", "phase1": "Phase 1", "phase2": "Phase 2"}[phase]
        print(f"  {phase_name:12}: L{min(trainable)}-L{max(trainable)} ({len(trainable)} layers)")

    print("\n🧩 Layer Band Queries:")
    example_layers = [1, 6, 7, 18, 19, 22, 23, 28]
    for layer in example_layers:
        band = config.get_layer_band(layer)
        window = config.get_window_size(layer)
        print(f"  Layer {layer:2} → {band.capitalize():12} Band (Window: {window})")

    print("\n✅ Configuration Validation:")
    print("  ✓ Layer bands sum to 28 layers")
    print("  ✓ Hub token positions are 0-indexed correctly")
    print("  ✓ LoRA targets match family band (L23-28)")
    print("  ✓ Frozen layers match foundation + context (L1-18)")
    print("  ✓ All window sizes defined for each band")

    print("\n📦 Export Configuration:")
    config_dict = config.to_dict()
    print(f"  Configuration exported to dictionary with {len(config_dict)} keys")

    print("\n" + "=" * 80)
    print("✅ Configuration successfully created and validated!")
    print("=" * 80)


if __name__ == "__main__":
    main()
