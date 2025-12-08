# ModernBERT v3 Code Discovery Document

> Auto-generated from source code analysis
> Generated: December 2025

---

## Summary

- **Total Files:** 38
- **Total Classes:** 111
- **Total Top-Level Functions:** 162
- **Total Methods:** 460

---

## DATA

### `collators_v3.py`

**Path:** `src\modeling_studio\data\collators_v3.py`

**Constants:**
- `HUB_TOKEN_COUNT`
- `V3_SPECIAL_PREFIX_LEN`
- `POSITION_CLS`
- `POSITION_EMO`
- `POSITION_MEM`
- `POSITION_REL`
- `POSITION_TASK`
- `POSITION_TEXT_START`

**Classes:**

#### `class V3CollatorConfig` (line 57)
> Configuration for v3 collators.

Methods:
- `__post_init__()`

#### `class V3BaseCollator` (line 80)
> Base collator for v3 models with hub token support.

Handles the v3 token layout:
    [CLS] [EMO] [MEM] [REL] [TASK] <text tokens> [SEP] [PAD]...

All position-based labels (NER, etc.) must be offset ...

Methods:
- `__init__()`
- `_validate_tokenizer()`
- `_add_hub_tokens()`
- `_offset_labels()`
- `__call__()`

#### `class V3ClassificationCollator(V3BaseCollator)` (line 206)
> Collator for sequence classification tasks (sentiment, safety, etc.).

No label offsetting needed - just single label per sequence.

Methods:
- `__call__()`
- `_pad_batch()`

#### `class V3TokenClassificationCollator(V3BaseCollator)` (line 272)
> Collator for token classification tasks (NER, etc.).

Labels must be offset by +5 for hub token prefix.

Methods:
- `__call__()`
- `_pad_batch()`

#### `class V3MultiTaskCollator(V3BaseCollator)` (line 340)
> Collator for multi-task training with multiple label types.

Handles unified samples with multiple task labels.

Methods:
- `__init__()`
- `__call__()`
- `_pad_multitask_batch()`


**Functions:**

- `create_v3_collator(tokenizer, task_type) -> V3BaseCollator` (line 462)
  > Factory function to create appropriate v3 collator.

Args:
    tokenizer: Tokenizer with v3 hub tokens
    task_type: "classification", "token_classif...

---

### `extractors_v3.py`

**Path:** `src\modeling_studio\data\extractors_v3.py`

**Classes:**

#### `class LabelVocabulary` (line 27)
> Label vocabulary for a single task.

Methods:
- `__post_init__()`
- `encode()`
- `encode_multi()`
- `to_multi_hot()`
- `num_labels()`

#### `class V3LabelVocabularies` (line 65)
> Container for all label vocabularies used in v3 FamilyOS tasks.

#### `class ExtractedLabels` (line 222)
> Container for all extracted labels for a single sample.

#### `class MultiTaskExtractor` (line 237)
> Extracts labels for all tasks from unified samples.

Methods:
- `__init__()`
- `extract()`
- `_get_offset_mapping()`
- `_extract_bio_labels()`
- `_extract_relations()`
- `extract_batch()`


**Functions:**

- `collate_classification_labels(labels, ignore_index)` (line 376)
  > Collate single-label classification targets with ignore_index padding.
- `collate_multi_label(labels, num_labels)` (line 386)
  > Collate multi-label targets, replacing missing entries with zeros.
- `collate_token_labels(labels, max_len, ignore_index)` (line 398)
  > Collate token-level labels with padding or truncation.

---

### `loaders_v3.py`

**Path:** `src\modeling_studio\data\loaders_v3.py`

**Classes:**

#### `class TaskType(Enum)` (line 32)
> Supported task types in unified FamilyOS data.

#### `class HubType(Enum)` (line 45)
> Hub token routing types.

#### `class HubTaskMapping` (line 55)
> Maps hub routing to task activation.

Methods:
- `__post_init__()`

#### `class HubRoutingParser` (line 74)
> Parses hub routing to determine task activation and gradient masking.

Methods:
- `__init__()`
- `get_active_tasks()`
- `get_hub_gradient_mask()`
- `get_task_weights()`
- `parse_batch()`

#### `class HubRouting` (line 166)
> Hub routing configuration parsed from sample.

Methods:
- `from_dict()`
- `to_tensor()`
- `active_hubs()`

#### `class SpanAnnotation` (line 210)
> Span annotation for NER and temporal tasks.

Methods:
- `from_dict()`

#### `class RelationTriple` (line 231)
> Relation triple annotation.

Methods:
- `from_dict()`

#### `class UnifiedSample` (line 250)
> Parsed sample from unified FamilyOS JSONL.

Methods:
- `from_json()`
- `has_task()`

#### `class UnifiedFamilyOSDataset(Dataset)` (line 307)
> PyTorch Dataset for unified FamilyOS data (eager loading).

Methods:
- `__init__()`
- `_load_samples()`
- `_load_shard()`
- `_should_include()`
- `__len__()`
- `__getitem__()`
- `get_task_distribution()`
- `get_hub_distribution()`

#### `class IterableUnifiedFamilyOSDataset(IterableDataset)` (line 411)
> Streaming/Iterable Dataset for unified FamilyOS data (memory efficient).

Methods:
- `__init__()`
- `__iter__()`


---

### `replay_sampler_v3.py`

**Path:** `src\modeling_studio\data\replay_sampler_v3.py`

**Classes:**

#### `class ReplayConfig` (line 29)
> Configuration for replay sampling.

Attributes:
    replay_ratio: Fraction of replay samples relative to primary data.
    task_balanced: Whether to balance sampling across replay tasks.
    min_repla...

#### `class ReplaySampler(Sampler)` (line 49)
> Sampler that mixes primary training data with replay data.

The replay mechanism prevents catastrophic forgetting by replaying
Stage A benchmark samples during training.

Sampling strategy:
    1. For...

Methods:
- `__init__()`
- `_calculate_sample_counts()`
- `_build_task_indices()`
- `_sample_replay_indices()`
- `__iter__()`
- `__len__()`
- `update_replay_ratio()`

#### `class ReplayDataset(Dataset)` (line 226)
> Dataset wrapper that exposes interleaved primary/replay samples.

Methods:
- `__init__()`
- `_refresh_epoch()`
- `__len__()`
- `__getitem__()`
- `refresh()`


**Functions:**

- `create_replay_sampler(primary_dataset, replay_dataset, replay_ratio, batch_size, task_balanced, ...)` (line 268)
  > Factory to create replay-enabled dataset and sampler.

Args:
    primary_dataset: Main training dataset.
    replay_dataset: Stage A replay dataset.
 ...

---

### `shard_loader_v3.py`

**Path:** `src\modeling_studio\data\shard_loader_v3.py`

**Classes:**

#### `class ShardConfig` (line 38)
> Configuration for shard-based loading.

#### `class ShardStats` (line 74)
> Statistics for a single shard.

Methods:
- `merge()`

#### `class ShardIndex` (line 121)
> Index of available shards with metadata.

Methods:
- `build()`
- `get_worker_shards()`
- `save()`
- `load()`

#### `class ShardReader` (line 195)
> Reads samples from a single shard file.

Methods:
- `__init__()`
- `__iter__()`
- `_validate_sample()`
- `_update_task_stats()`
- `_update_hub_stats()`

#### `class StreamingShardDataset(IterableDataset)` (line 288)
> Memory-efficient streaming dataset over multiple shards.

Methods:
- `__init__()`
- `__iter__()`
- `__len__()`
- `set_epoch()`
- `get_stats()`
- `_worker_checkpoint_path()`
- `_load_resume_offset()`
- `_write_resume_checkpoint()`

#### `class BufferedShardDataset(IterableDataset)` (line 413)
> Buffered streaming dataset with prefetching.

Methods:
- `__init__()`
- `_prefetch_shard()`
- `__iter__()`
- `__len__()`
- `set_epoch()`
- `_worker_checkpoint_path()`
- `_load_resume_offset()`
- `_write_resume_checkpoint()`


**Functions:**

- `create_shard_dataset(data_dir, shard_pattern, streaming, buffered, transform)` (line 532)
  > Create a shard-based dataset.
- `get_shard_statistics(data_dir, shard_pattern) -> ShardStats` (line 553)
  > Compute aggregate statistics over all shards.

---

## MODELS

### `attention_v3.py`

**Path:** `src\modeling_studio\models\attention_v3.py`

**Constants:**
- `GLOBAL_TOKEN_POSITIONS`

**Classes:**

#### `class MultiScaleAttentionWithGlobals(nn.Module)` (line 517)
> Multi-head attention with:
- Sliding window for text tokens
- Global attention for hub tokens (positions 0-4)
- Layer-specific window sizes

This is the v3.3 solution to the "Blind Hub" problem.

Arch...

Methods:
- `__init__()`
- `_get_attention_mask()`
- `forward()`
- `extra_repr()`

#### `class FlashAttentionWithGlobals(nn.Module)` (line 737)
> Flash Attention 2 implementation with Global Hub Token support.

⚠️ MITIGATION STRATEGY:
1. Hub→Text Attention: ✅ Solved via manual calculation (Hubs see everything).
2. Text→Hub Attention: ❌ NOT nati...

Methods:
- `__init__()`
- `forward()`
- `extra_repr()`


**Functions:**

- `create_global_local_attention_mask(seq_len, window_size, global_positions, device, dtype)` (line 46)
  > Create attention mask with global tokens + sliding windows.

This is the v3.3 solution to the "Blind Hub" problem. Hub tokens need to see
the entire s...
- `create_causal_global_local_mask(seq_len, window_size, global_positions, device, dtype)` (line 130)
  > Create CAUSAL attention mask (for decoder-style, if needed).

Combines global attention + sliding window + causal masking.
Position i can only attend ...
- `expand_mask_for_batch(mask, batch_size, num_heads)` (line 173)
  > Expand 2D mask to 4D for multi-head attention.

Args:
    mask: [seq_len, seq_len] attention mask
    batch_size: Batch size
    num_heads: Number of ...
- `convert_mask_to_additive(mask, dtype)` (line 197)
  > Convert boolean mask to additive mask for scaled_dot_product_attention.

Args:
    mask: Boolean mask [True = can attend, False = masked]
    dtype: O...
- `get_window_size_for_layer(layer_idx) -> int` (line 271)
  > Get the sliding window size for a given layer.

Args:
    layer_idx: 1-indexed layer number (1-28)

Returns:
    Window size (64, 128, 256, or 512)

R...
- `get_layer_band_name(layer_idx) -> str` (line 308)
  > Get the band name for a layer.

Args:
    layer_idx: 1-indexed layer number (1-28)

Returns:
    Band name ("foundation", "context", "semantic", or "f...
- `get_attention_mask_for_layer(layer_idx, seq_len, device, dtype)` (line 339)
  > Get the appropriate attention mask for a specific layer.

Convenience function that combines window size lookup and mask creation.

Args:
    layer_id...
- `print_layer_config() -> None` (line 374)
  > Print the layer window configuration for debugging.

Example output:
    📊 Layer Window Configuration:
    -------------------------------------------...
- `get_layer_config_summary()` (line 394)
  > Get layer configuration as a dictionary for programmatic access.

Returns:
    Dictionary mapping band names to config dicts with keys:
    - start_la...
- `visualize_attention_mask(mask, max_display) -> None` (line 428)
  > Print a visual representation of an attention mask.

Args:
    mask: [seq_len, seq_len] attention mask (boolean or float)
    max_display: Maximum seq...
- `count_attention_patterns(mask)` (line 467)
  > Count attention patterns in a mask for analysis.

Args:
    mask: [seq_len, seq_len] boolean attention mask

Returns:
    Dictionary with counts:
    ...
- `create_attention_layer(hidden_size, num_attention_heads, attention_dropout, layer_idx, use_flash_attention)` (line 884)
  > Factory function implementing the DECISION MATRIX (Safety Switch).

Decision Logic:
1. If Flash Attention missing → Standard (SDPA optimized)
2. If us...

---

### `config_v3.py`

**Path:** `src\modeling_studio\models\config_v3.py`

**Classes:**

#### `class ModernBERTv3Config` (line 17)
> Configuration for ModernBERT v3.3 Ultra.

Methods:
- `__post_init__()`
- `get_layer_band()`
- `get_window_size()`
- `get_trainable_layers()`
- `get_lora_layers()`
- `to_dict()`

#### `class LayerSource(Enum)` (line 183)
> Source of layer weights during v3 initialization.

#### `class LayerMapping(NamedTuple)` (line 191)
> Mapping of v3 layer to its weight source.


**Functions:**

- `get_layer_source_mapping()` (line 199)
  > Get the complete layer source mapping for v3 initialization.

Returns:
    Dict mapping v3 layer index (1-28) to LayerMapping

Strategy:
    - Layers ...
- `print_layer_source_mapping()` (line 233)
  > Print a human-readable view of the layer source mapping.

---

### `embeddings_v3.py`

**Path:** `src\modeling_studio\models\embeddings_v3.py`

**Classes:**

#### `class ModernBERTEmbeddingsV3(nn.Module)` (line 26)
> Embeddings module for ModernBERT v3.3 Ultra.

Token layout:
    [CLS] [EMO] [MEM] [REL] [TASK] <text tokens> [SEP] [PAD]...
    pos 0   1     2     3     4        5+

Components:
    1. Word embedding...

Methods:
- `__init__()`
- `forward()`
- `get_hub_token_embeddings()`
- `resize_token_embeddings()`
- `get_num_params()`
- `extra_repr()`


---

### `encoder_v3.py`

**Path:** `src\modeling_studio\models\encoder_v3.py`

**Classes:**

#### `class ModernBERTEncoderV3(nn.Module)` (line 31)
> 28-layer encoder stack for ModernBERT v3.3 Ultra.

Layer Structure:
    - Layers 1-6:   Foundation Band (window=64, frozen in Phase 1)
    - Layers 7-18:  Context Band (window=128, frozen in Phase 1)
...

Methods:
- `__init__()`
- `forward()`
- `_checkpoint_forward()`
- `freeze_layers()`
- `unfreeze_layers()`
- `freeze_by_band()`
- `unfreeze_by_band()`
- `get_layers_by_band()`
- `print_layer_summary()`
- `get_num_params()`
- `extra_repr()`


---

### `ffn_v3.py`

**Path:** `src\modeling_studio\models\ffn_v3.py`

**Classes:**

#### `class GELUFFN(nn.Module)` (line 18)
> GELU Feed-Forward Network (same as v2).

Architecture:
    hidden → intermediate (4x) → GELU → hidden
    768 → 3072 → GELU → 768

This implementation matches v2 exactly to enable direct weight transf...

Methods:
- `__init__()`
- `_gelu_new()`
- `forward()`
- `extra_repr()`

#### `class SwiGLUFFN(nn.Module)` (line 115)
> SwiGLU Feed-Forward Network (DEPRECATED - R&D only).

⚠️ NOT used in v3 production. Kept for research experiments.

Architecture:
    hidden → gate (4x) → SiLU
    hidden → up (4x)
    gate * up → dow...

Methods:
- `__init__()`
- `forward()`
- `extra_repr()`


**Functions:**

- `create_ffn(hidden_size, intermediate_size, hidden_dropout_prob, ffn_type)` (line 194)
  > Factory function to create FFN module.

This function provides a unified interface for creating different
FFN variants. By default, it creates the pro...

---

### `heads_v3.py`

**Path:** `src\modeling_studio\models\heads_v3.py`

**Classes:**

#### `class HeadConfig` (line 43)
> Configuration for a task head.

Attributes:
    name: Task/capability name (e.g., "emotions", "ner_general")
    num_labels: Number of output labels
    head_type: Type of head ("classification", "tok...

#### `class HubAwareClassificationHead(nn.Module)` (line 73)
> Classification head that receives input from a specific hub token.

This head automatically extracts the correct hub token representation
and applies a simple linear classifier.

Used for: emotions, s...

Methods:
- `__init__()`
- `forward()`
- `extra_repr()`

#### `class HubAwareTokenClassificationHead(nn.Module)` (line 137)
> Token-level classification head for sequence labeling.

This head receives the full sequence output and applies classification
at each token position. Used for NER and temporal expression detection.

...

Methods:
- `__init__()`
- `forward()`
- `get_predictions()`
- `extra_repr()`

#### `class HubAwareHierarchicalHead(nn.Module)` (line 220)
> Hierarchical classification head for emotions.

This head implements a two-level hierarchy:
1. Primary: Ekman emotions (7 classes)
2. Secondary: GoEmotions (28 classes), conditioned on primary

Uses [...

Methods:
- `__init__()`
- `_build_hierarchy_mask()`
- `forward()`
- `extra_repr()`

#### `class HubAwareSafetyHead(nn.Module)` (line 336)
> Safety classification head with calibrated outputs.

Uses [EMO] hub token (safety correlates with emotional content).

Features:
    - Binary classification (safe/unsafe)
    - Temperature-based confi...

Methods:
- `__init__()`
- `forward()`
- `predict_with_threshold()`
- `extra_repr()`

#### `class HubAwareNLIHead(nn.Module)` (line 440)
> NLI head using [REL] hub token.

This head uses the [REL] hub token which captures relationship
information between premise and hypothesis in NLI tasks.

Labels: entailment (0), neutral (1), contradic...

Methods:
- `__init__()`
- `forward()`
- `extra_repr()`


**Functions:**

- `create_head_for_capability(capability, hidden_size, num_labels)` (line 535)
  > Factory function to create appropriate head for a capability.

Args:
    capability: Task/capability name (e.g., "emotions", "ner_general", "nli")
   ...
- `create_all_heads(hidden_size, capabilities)` (line 604)
  > Create heads for all (or specified) capabilities.

Args:
    hidden_size: Model hidden size (default: 768)
    capabilities: List of capabilities to c...

---

### `hub_initialization_v3.py`

**Path:** `src\modeling_studio\models\hub_initialization_v3.py`

**Functions:**

- `resize_token_embeddings_aligned(model, new_vocab_size, alignment) -> None` (line 80)
  > Resize token embeddings to align with hardware efficiency requirements.

ModernBERT-base has vocab_size=50265. After adding 4 hub tokens via
add_speci...
- `get_aligned_vocab_size(base_size, alignment) -> int` (line 166)
  > Calculate the next aligned vocabulary size.

Args:
    base_size: Current vocabulary size
    alignment: Alignment factor (default: 128)

Returns:
   ...
- `verify_padding_tokens_unreachable(tokenizer, model_vocab_size)` (line 188)
  > Verify that padding tokens are unreachable by the tokenizer.

This function confirms that the tokenizer can never produce token IDs
in the padding ran...
- `compute_semantic_centroid(word_list, tokenizer, embeddings)` (line 244)
  > Compute the semantic centroid of a list of words.

The centroid is computed as the mean of word embeddings. For words that
tokenize into multiple subw...
- `initialize_hub_tokens_semantic(model, v2_tokenizer, v2_embeddings) -> None` (line 308)
  > Initialize hub token embeddings using semantic centroids from v2.

This function updates the hub token embeddings in-place in the v3 model's
embedding...
- `verify_hub_token_initialization(model, v2_tokenizer, v2_embeddings)` (line 383)
  > Verify hub token initialization quality using cosine similarity.

Computes the cosine similarity between each hub token's embedding and its
expected s...

---

### `hub_tokens.py`

**Path:** `src\modeling_studio\models\hub_tokens.py`

**Classes:**

#### `class HubToken(Enum)` (line 18)
> Hub token identifiers.

#### `class HubTokenSpec` (line 29)
> Specification for a hub token.


**Functions:**

- `get_hub_for_capability(capability) -> str` (line 86)
  > Get the hub token that routes to a given capability.

Args:
    capability: Name of the capability (e.g., "emotions", "intent", "ner_family")

Returns...
- `get_capabilities_for_hub(hub_token)` (line 119)
  > Get all capabilities routed through a hub token.

Args:
    hub_token: Hub token string (e.g., "[EMO]", "[MEM]")

Returns:
    List of capability name...
- `get_hub_positions()` (line 140)
  > Get position indices for all hub tokens (including CLS).

Returns:
    Dictionary mapping hub token to position index

Note:
    Position 0 is reserve...
- `get_global_attention_positions()` (line 161)
  > Get positions that should have global attention (CLS + all hubs).

Returns:
    List of position indices with global attention

Note:
    These tokens...
- `get_semantic_seeds(hub_token)` (line 181)
  > Get semantic seed words for hub token initialization.

These words are used to initialize hub token embeddings as semantic centroids,
placing them in ...
- `get_hub_token_id(hub_token) -> int` (line 205)
  > Get the reserved token ID for a hub token.

Args:
    hub_token: Hub token string (e.g., "[EMO]", "[MEM]")

Returns:
    Token ID in the extended voca...
- `get_all_hub_tokens()` (line 229)
  > Get list of all hub token strings.

Returns:
    List of hub token strings (excluding [CLS])

Examples:
    >>> get_all_hub_tokens()
    ['[EMO]', '[M...
- `print_hub_token_registry()` (line 243)
  > Print a human-readable view of the hub token registry.

---

### `initialization_v3.py`

**Path:** `src\modeling_studio\models\initialization_v3.py`

**Classes:**

#### `class V2CheckpointInfo` (line 65)
> Information about a v2 checkpoint.

Attributes:
    path: Path to checkpoint file
    num_layers: Number of encoder layers (should be 22 for v2)
    hidden_size: Hidden dimension (should be 768)
    v...

#### `class WeightTransferStats` (line 95)
> Statistics from weight transfer operation.

Attributes:
    total_params: Total parameters in v3 model
    transferred_params: Parameters copied from v2
    initialized_params: New parameters (hub tok...

#### `class V2CheckpointLoader` (line 124)
> Loads and parses ModernBERT v2 checkpoints.

v2 Architecture (22 layers):
    - Foundation Band: L1-6 (window=64)
    - Core Band: L7-18 (window=128)
    - Family Band: L19-22 (window=256)

v3 Archite...

Methods:
- `__init__()`
- `load()`
- `_clean_state_dict()`
- `get_info()`
- `validate()`
- `get_layer_weights()`
- `get_embedding_weights()`
- `print_summary()`

#### `class LayerCopier` (line 576)
> Copies layer weights from v2 to v3 (direct 1:1 mapping).

Implements Issue 4.1.2: Layer 1-22 Direct Copy

Layer Mapping (v3 ← v2):
    - L1-6 (Foundation) ← L1-6: Direct copy (window 64)
    - L7-18 (...

Methods:
- `__init__()`
- `copy_layer()`
- `copy_layers_1_to_22()`
- `get_stats()`

#### `class LayerCloner` (line 802)
> Clones layer weights from v2 to new v3 layers.

Implements Issue 4.1.3: Layer 23-28 Cloning from L15-20

Clone Mapping (v3 ← v2):
    L23 ← L15: First Family Band layer
    L24 ← L16: Second layer
   ...

Methods:
- `__init__()`
- `clone_layer()`
- `clone_layers_23_to_28()`
- `get_stats()`

#### `class EmbeddingTransfer` (line 1188)
> Transfers embeddings from v2 to v3 with hub token slot creation.

Implements Issue 4.1.4: Embedding Transfer with Hub Token Slots

v2 Vocabulary: 50,368 tokens (ModernBERT-base)
v3 Vocabulary: 50,372 ...

Methods:
- `__init__()`
- `transfer_word_embeddings()`
- `_get_word_embedding_weight()`
- `transfer_position_embeddings()`
- `transfer_layer_norm()`
- `_get_layer_norm()`
- `transfer_all()`
- `get_stats()`

#### `class HubTokenSemanticInitializer` (line 1566)
> Initialize hub token embeddings with semantic meaning.

Implements Issue 4.1.5: Hub Token Semantic Initialization

Strategy: Average embeddings of semantically related tokens to create
a meaningful st...

Methods:
- `__init__()`
- `get_seed_token_ids()`
- `initialize_hub_token()`
- `initialize_all_hubs()`
- `_get_word_embeddings()`
- `_print_summary()`
- `get_stats()`


**Functions:**

- `load_v2_checkpoint(path) -> V2CheckpointLoader` (line 532)
  > Factory function to load v2 checkpoint.

Convenience function that creates a loader, validates it,
and prints a summary.

Args:
    path: Path to chec...
- `copy_layers_direct(v3_model, v2_checkpoint_path) -> int` (line 739)
  > Copy v2 layers 1-22 directly to v3 layers 1-22.

Main entry point for Issue 4.1.2: Layer 1-22 Direct Copy.

This function:
1. Loads v2 checkpoint
2. C...
- `clone_layers_for_growth(v3_model, v2_checkpoint_path, add_noise, noise_std) -> int` (line 1005)
  > Clone v2 layers 15-20 to v3 layers 23-28.

Main entry point for Issue 4.1.3: Layer 23-28 Cloning from L15-20.

This function:
1. Loads v2 checkpoint
2...
- `get_clone_source_for_layer(v3_layer_idx)` (line 1079)
  > Get the v2 layer that was cloned to create this v3 layer.

Only layers 22-27 (v3 L23-28) are cloned from v2 L14-19.
Layers 0-21 are direct copies.

Ar...
- `get_band_for_layer(v3_layer_idx) -> str` (line 1101)
  > Get the band name for a v3 layer index.

Args:
    v3_layer_idx: v3 layer index (0-indexed)

Returns:
    Band name: 'foundation', 'core', 'SEMANTIC', o...
- `get_layers_in_band(band_name)` (line 1128)
  > Get all layer indices in a band.

Args:
    band_name: One of 'foundation', 'core', 'SEMANTIC', 'family'

Returns:
    List of layer indices (0-indexed)...
- `print_layer_band_summary() -> None` (line 1150)
  > Print summary of v3 layer bands.

Example output:
    ══════════════════════════════════════════════════════════════
    📊 v3 Layer Band Configuration...
- `transfer_embeddings(v3_model, v2_checkpoint_path) -> int` (line 1518)
  > Transfer embeddings from v2 to v3 with hub token slots.

Main entry point for Issue 4.1.4: Embedding Transfer with Hub Token Slots.

This function:
1....
- `initialize_hub_tokens_semantic(v3_model, tokenizer_name) -> int` (line 1883)
  > Initialize hub token embeddings with semantic meaning.

Convenience function for Issue 4.1.5.

Creates a HubTokenSemanticInitializer and initializes a...
- `initialize_from_v2(v3_model, v2_checkpoint_path, add_clone_noise, clone_noise_std, tokenizer_name) -> WeightTransferStats` (line 1931)
  > Complete initialization of v3 model from v2 checkpoint.

This is the main orchestration function that performs all steps of
v2→v3 weight transfer as s...

---

### `layers_v3.py`

**Path:** `src\modeling_studio\models\layers_v3.py`

**Classes:**

#### `class ModernBERTLayerV3(nn.Module)` (line 34)
> Single transformer layer for ModernBERT v3.3 Ultra.

This layer implements the core transformer block with multi-scale attention,
feed-forward network, and optional LoRA adaptation for the Family Band...

Methods:
- `__init__()`
- `_init_lora()`
- `forward()`
- `freeze_base_weights()`
- `unfreeze_base_weights()`
- `merge_lora_weights()`
- `get_num_params()`
- `extra_repr()`


**Functions:**

- `create_layer_stack(num_layers, hidden_size, num_attention_heads, intermediate_size, hidden_dropout_prob, ...)` (line 309)
  > Create the full 28-layer transformer stack for ModernBERT v3.3 Ultra.

Layer Band Configuration:
    - Foundation Band (L1-6): Window=64, No LoRA, Fro...
- `freeze_layer_bands(layers, freeze_bands) -> None` (line 390)
  > Freeze specific layer bands.

Args:
    layers: ModuleList of layers
    freeze_bands: Bands to freeze (default: ["foundation", "context"])
          ...
- `unfreeze_layer_bands(layers, unfreeze_bands) -> None` (line 430)
  > Unfreeze specific layer bands.

Args:
    layers: ModuleList of layers
    unfreeze_bands: Bands to unfreeze (default: ["semantic", "family"])

Exampl...
- `get_layer_stats(layers) -> dict` (line 468)
  > Get statistics about the layer stack.

Returns:
    Dict with:
        - num_layers: Total layers
        - total_params: Total parameters
        - t...
- `print_layer_stack_summary(layers) -> None` (line 510)
  > Print detailed summary of the layer stack.

Args:
    layers: ModuleList of layers

---

### `lora_v3.py`

**Path:** `src\modeling_studio\models\lora_v3.py`

**Classes:**

#### `class LoRALayer(nn.Module)` (line 20)
> Low-Rank Adaptation layer for efficient fine-tuning.

Adds trainable low-rank matrices A and B to a frozen weight matrix W:
    output = (W + BA) @ x = W @ x + B @ (A @ x)

Where:
    - W: Original fr...

Methods:
- `__init__()`
- `forward()`
- `extra_repr()`

#### `class LinearWithLoRA(nn.Module)` (line 105)
> Linear layer with optional LoRA adapter.

This is a drop-in replacement for nn.Linear that optionally adds a LoRA
adapter. During training, the base linear weights can be frozen while
only the LoRA pa...

Methods:
- `__init__()`
- `forward()`
- `merge_lora()`
- `freeze_base()`
- `unfreeze_base()`
- `extra_repr()`


**Functions:**

- `apply_lora_to_layer(layer, r, alpha, dropout, target_modules)` (line 238)
  > Apply LoRA adapters to specific modules in a transformer layer.

This function searches for Linear modules in the layer that match the
target module n...
- `get_lora_parameters(model)` (line 291)
  > Get all LoRA parameters for optimizer.

This function collects all parameters with "lora" in their name,
which should be the only trainable parameters...
- `count_lora_parameters(model) -> int` (line 314)
  > Count trainable LoRA parameters.

Args:
    model: Model containing LoRA layers

Returns:
    Number of trainable LoRA parameters

Example:
    >>> pr...
- `freeze_non_lora_parameters(model) -> None` (line 332)
  > Freeze all non-LoRA parameters in the model.

This is a convenience function for setting up LoRA training where
only LoRA adapters should be trainable...
- `print_lora_info(model) -> None` (line 347)
  > Print summary of LoRA configuration in the model.

Args:
    model: Model containing LoRA layers

---

### `losses_v3.py`

**Path:** `src\modeling_studio\models\losses_v3.py`

**Classes:**

#### `class LossOutput` (line 40)
> Container for loss computation results.

Attributes:
    total_loss: Weighted sum of all task losses
    task_losses: Per-task individual losses
    task_weights: Weights applied to each task

#### `class HubAwareLossComputer(nn.Module)` (line 55)
> Computes losses for all tasks with hub routing awareness.

This loss computer handles multiple task types:
- Classification: Standard cross-entropy (emotions, sentiment, safety, intent, etc.)
- Token-...

Methods:
- `__init__()`
- `compute_task_loss()`
- `_compute_token_level_loss()`
- `_compute_focal_loss()`
- `_compute_hierarchical_loss()`
- `compute_multitask_loss()`
- `update_task_weight()`
- `extra_repr()`

#### `class UncertaintyWeightedLoss(nn.Module)` (line 369)
> Multi-task loss with learned uncertainty weighting.

Based on "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry
and Semantics" (Kendall et al., CVPR 2018).

The loss for each t...

Methods:
- `__init__()`
- `forward()`
- `get_task_weights()`
- `get_task_uncertainties()`
- `extra_repr()`


**Functions:**

- `create_loss_computer(task_configs, use_uncertainty_weighting)` (line 474)
  > Factory function to create loss computer.

Creates either a standard HubAwareLossComputer with fixed weights or
a combined module with UncertaintyWeig...

---

### `modernbert_v3.py`

**Path:** `src\modeling_studio\models\modernbert_v3.py`

**Classes:**

#### `class ModernBERTv3Output` (line 53)
> Output container for ModernBERT v3 forward pass.

Attributes:
    last_hidden_state: Final layer output [batch, seq, hidden]
    pooled_outputs: Dict of hub token representations
    hidden_states: Al...

#### `class ModernBERTv3Ultra(nn.Module)` (line 76)
> ModernBERT v3.3 Ultra - Unified FamilyOS Encoder.

Architecture:
    - 28 transformer layers (vs 22 in v2)
    - 4 hub tokens: [EMO], [MEM], [REL], [TASK]
    - Multi-scale sliding window attention (6...

Methods:
- `__init__()`
- `_init_weights()`
- `forward()`
- `get_representation_for_capability()`
- `get_embedding_representation()`
- `freeze_for_phase()`
- `merge_lora_weights()`
- `get_input_embeddings()`
- `set_input_embeddings()`
- `resize_token_embeddings()`
- `num_parameters()`
- `num_trainable_parameters()`
- `print_model_summary()`

#### `class ModernBERTv3ForMultiTask(ModernBERTv3Ultra)` (line 465)
> ModernBERT v3 with multi-task heads and hub routing.

Extends the base model with:
    - Task-specific classification/regression heads
    - Hub token routing to appropriate heads
    - Multi-task los...

Methods:
- `__init__()`
- `register_task_head()`
- `forward_for_task()`
- `forward_multitask()`
- `_compute_task_loss()`
- `get_hub_gradient_mask()`
- `set_task_loss_weight()`
- `print_routing_table()`

#### `class ClassificationHead(nn.Module)` (line 821)
> Simple classification head for hub-routed tasks.

Used for sequence-level classification tasks like:
- Emotion detection (7 classes)
- Sentiment analysis (3 classes)
- Safety classification (2-5 class...

Methods:
- `__init__()`
- `forward()`

#### `class TokenClassificationHead(nn.Module)` (line 855)
> Token-level classification head for NER/temporal.

Used for token-level tasks like:
- Named Entity Recognition (9 classes)
- Temporal expression detection (5 classes)

Example:
    >>> head = TokenCla...

Methods:
- `__init__()`
- `forward()`

#### `class RegressionHead(nn.Module)` (line 888)
> Regression head for similarity tasks.

Used for regression tasks like:
- Semantic similarity (STS-B)
- Relevance scoring

Example:
    >>> head = RegressionHead(768)
    >>> pooled = torch.randn(4, 76...

Methods:
- `__init__()`
- `forward()`


**Functions:**

- `create_modernbert_v3_ultra(from_v2_checkpoint) -> ModernBERTv3Ultra` (line 427)
  > Factory function to create ModernBERT v3 Ultra.

Args:
    from_v2_checkpoint: Path to v2 checkpoint for initialization
    **config_overrides: Overri...
- `create_v3_multitask_model(config, task_configs) -> ModernBERTv3ForMultiTask` (line 921)
  > Factory function to create v3 with task heads.

Args:
    config: Model config
    task_configs: Dict mapping task names to head configs
        Examp...

---

### `pair_encoder_v3.py`

**Path:** `src\modeling_studio\models\pair_encoder_v3.py`

**Classes:**

#### `class PairEncoderV3(nn.Module)` (line 35)
> Pair Encoder for sentence-pair tasks in v3.

Token Layout for Pairs:
    [CLS] [EMO] [MEM] [REL] [TASK] <text_a> [SEP] <text_b> [SEP] [PAD]...

Key Innovation: The [REL] hub token (position 3) capture...

Methods:
- `__init__()`
- `forward()`
- `get_rel_hub_representation()`
- `set_pooling_strategy()`
- `extra_repr()`

#### `class SiamesePairEncoderV3(nn.Module)` (line 236)
> Siamese-style pair encoder for semantic similarity.

Uses the [MEM] hub token for embedding representation
and [REL] hub for explicit relationship modeling.

Good for:
    - Semantic textual similarit...

Methods:
- `__init__()`
- `forward()`
- `extra_repr()`


---

### `poolers_v3.py`

**Path:** `src\modeling_studio\models\poolers_v3.py`

**Classes:**

#### `class HubTokenPooler(nn.Module)` (line 27)
> Extracts hub token representations from the final hidden states.

Given sequence: [CLS] [EMO] [MEM] [REL] [TASK] <text> [SEP] [PAD]...
Returns dict of hub token representations for routing to heads.

...

Methods:
- `__init__()`
- `forward()`
- `get_pooled_for_capability()`

#### `class CombinedPooler(nn.Module)` (line 130)
> Combined pooler that provides CLS, Mean, and Hub token pooling.

This pooler extracts multiple pooled representations in a single forward pass:
- [CLS] token representation (with projection)
- Hub tok...

Methods:
- `__init__()`
- `forward()`


---

### `registry_v3.py`

**Path:** `src\modeling_studio\models\registry_v3.py`

**Classes:**

#### `class TaskType(Enum)` (line 45)
> Types of tasks supported in v3.

- CLASSIFICATION: Standard classification (emotions, sentiment, intent, etc.)
- TOKEN_CLASSIFICATION: Token-level sequence labeling (NER, temporal)
- REGRESSION: Regre...

#### `class TaskSpec` (line 66)
> Complete specification for a task/capability.

Attributes:
    name: Task identifier (e.g., "emotions", "ner_general")
    task_type: Type of task (classification, token_classification, etc.)
    hub_...

#### `class TaskRegistry` (line 429)
> Registry for managing v3 task configurations.

Provides centralized access to task specifications, head creation,
hub routing information, and metric configuration for all 12 capabilities.

Features:
...

Methods:
- `__init__()`
- `get_task()`
- `get_all_tasks()`
- `get_tasks_by_hub()`
- `get_hub_routed_tasks()`
- `get_token_level_tasks()`
- `create_head()`
- `create_all_heads()`
- `get_loss_weights()`
- `get_metrics()`
- `print_registry()`
- `extra_repr()`


**Functions:**

- `get_registry() -> TaskRegistry` (line 680)
  > Get global task registry singleton.

Returns:
    Singleton TaskRegistry instance

Example:
    >>> registry = get_registry()
    >>> emotions_spec = ...

---

### `routing_v3.py`

**Path:** `src\modeling_studio\models\routing_v3.py`

**Classes:**

#### `class HubRouter(nn.Module)` (line 30)
> Routes hub token representations to capability heads.

For each capability, determines:
1. Which hub token provides the representation
2. Whether to use hub pooling or per-token representations

The r...

Methods:
- `__init__()`
- `get_representation_for_capability()`
- `get_hub_gradient_mask()`

#### `class CapabilityHead(nn.Module)` (line 166)
> Wrapper for a capability head that handles hub routing.

This wrapper automatically routes representations to the underlying head
based on the capability's routing type (hub or token-level).

Args:
  ...

Methods:
- `__init__()`
- `forward()`
- `extra_repr()`


**Functions:**

- `create_hub_routing_info(capability)` (line 244)
  > Get routing information for a capability.

Returns comprehensive routing information including pool type,
hub token, and hub description.

Args:
    c...
- `print_routing_table() -> None` (line 280)
  > Print a human-readable view of the routing table.

---

### `tokenization_v3.py`

**Path:** `src\modeling_studio\models\tokenization_v3.py`

**Classes:**

#### `class HubTokenizer` (line 20)
> Wrapper tokenizer that injects hub tokens after [CLS].

This tokenizer extends ModernBERT-base tokenizer to include 4 specialized
hub tokens that route to different capability heads in the v3 architec...

Methods:
- `__init__()`
- `vocab_size()`
- `__call__()`
- `get_hub_token_positions()`
- `get_text_start_position()`
- `decode()`
- `batch_decode()`
- `save_pretrained()`
- `from_pretrained()`
- `__len__()`
- `__repr__()`


---

### `verification_v3.py`

**Path:** `src\modeling_studio\models\verification_v3.py`

**Classes:**

#### `class LayerComparisonResult` (line 50)
> Comparison result for a single layer.

Attributes:
    layer_idx: Layer index (0-based)
    v2_norm: L2 norm of v2 layer output
    v3_norm: L2 norm of v3 layer output
    diff_norm: Maximum absolute ...

#### `class VerificationResult` (line 80)
> Complete results from function preserving verification.

Attributes:
    passed: Whether all verifications passed
    max_diff: Maximum difference across all layers
    mean_diff: Mean difference acro...

#### `class WeightComparisonResult` (line 111)
> Result of weight-level comparison between v2 and v3.

Attributes:
    passed: Whether all weight comparisons passed
    matched_params: Number of parameters that matched
    mismatched_params: Number ...

#### `class FunctionPreservingVerifier` (line 139)
> Verifies that v3 model preserves v2 function for layers 1-22.

Function Preserving Property:
    For layers L1-L22, given identical input embeddings,
    the layer outputs should be identical (within ...

Methods:
- `__init__()`
- `verify_embeddings()`
- `verify_layer()`
- `verify_all_layers()`
- `verify_weights_only()`
- `_get_embeddings()`
- `_get_layer()`
- `_forward_layer()`
- `_create_message()`


**Functions:**

- `verify_function_preserving(v2_model, v3_model, input_ids, attention_mask, tolerance, ...) -> VerificationResult` (line 597)
  > Verify v3 preserves v2 function for shared layers.

Convenience function that creates a verifier and runs full verification.

Args:
    v2_model: Orig...
- `verify_weight_transfer(v2_model, v3_model, tolerance, verbose) -> WeightComparisonResult` (line 629)
  > Verify weights transferred correctly from v2 to v3.

Quick verification that only checks weights, not forward passes.

Args:
    v2_model: Original v2...
- `create_verification_inputs(vocab_size, seq_length, batch_size, device)` (line 653)
  > Create random inputs for verification testing.

Args:
    vocab_size: Vocabulary size (default 50368 for ModernBERT)
    seq_length: Sequence length
 ...
- `verify_embedding_transfer(v2_model, v3_model, tolerance, verbose)` (line 680)
  > Verify word embeddings transferred correctly.

Compares the first 50,368 embeddings (v2 vocab) between v2 and v3.
Hub token embeddings (positions 5036...
- `_get_embedding_weight(model)` (line 720)
  > Get word embedding weight tensor from model.

---

## SCRIPTS

### `initialize_v3_from_v2.py`

**Path:** `scripts\initialize_v3_from_v2.py`

**Functions:**

- `parse_args()` (line 57)
  > Parse command line arguments.
- `create_v3_config(v2_loader) -> ModernBERTv3Config` (line 128)
  > Create v3 config based on v2 checkpoint info.

Args:
    v2_loader: Loaded v2 checkpoint

Returns:
    ModernBERTv3Config configured for v3 architectu...
- `create_mock_v2_model(v2_loader)` (line 154)
  > Create a mock v2 model structure for verification.

Since we don't have a standalone v2 model class, we create a simple
wrapper around the checkpoint ...
- `run_verification(v2_loader, v3_model, tolerance, device, verbose)` (line 244)
  > Run function preserving verification.

Args:
    v2_loader: Loaded v2 checkpoint
    v3_model: Initialized v3 model
    tolerance: Verification tolera...
- `save_model(model, config, output_dir, stats, v2_checkpoint_path, ...) -> None` (line 303)
  > Save initialized model and metadata.

Args:
    model: Initialized v3 model
    config: v3 configuration
    output_dir: Output directory
    stats: W...
- `main() -> int` (line 372)
  > Main entry point.

---

### `train_v3_phase0_5.py`

**Path:** `scripts\train_v3_phase0_5.py`

**Classes:**

#### `class Phase05Config` (line 77)
> Configuration specific to Phase 0.5 healing.

Methods:
- `to_dict()`
- `save()`
- `from_dict()`

#### `class SyntheticHealingDataset(Dataset)` (line 379)
> Synthetic dataset for smoke testing.

Methods:
- `__init__()`
- `__len__()`
- `__getitem__()`

#### `class HealingDataset(Dataset)` (line 480)
> Dataset for healing data from JSONL.

Methods:
- `__init__()`
- `__len__()`
- `__getitem__()`


**Functions:**

- `parse_args()` (line 173)
  > Parse command line arguments.
- `load_config(args) -> Phase05Config` (line 263)
  > Load and merge configuration from YAML and CLI args.
- `load_healing_data(config, tokenizer)` (line 432)
  > Load healing datasets.

Args:
    config: Training configuration
    tokenizer: Tokenizer with hub tokens

Returns:
    Tuple of (train_dataset, val_d...
- `create_dataloaders(train_dataset, val_dataset, config)` (line 530)
  > Create train and validation dataloaders.

Args:
    train_dataset: Training dataset
    val_dataset: Validation dataset
    config: Training configura...
- `setup_model(config, tokenizer)` (line 571)
  > Load or create model for training.

Args:
    config: Training configuration
    tokenizer: Tokenizer with hub tokens

Returns:
    Model instance
- `setup_layer_freezing(model, config)` (line 617)
  > Configure layer freezing for Phase 0.5.

NOTE: Does NOT freeze embeddings - that's handled via gradient masking
to allow hub tokens to train while fre...
- `setup_hub_gradient_masking(model, config)` (line 658)
  > Setup gradient masking for hub tokens.

Args:
    model: Model to configure
    config: Training configuration

Returns:
    Gradient hook manager
- `create_optimizer(model, config)` (line 685)
  > Create Zipper LR optimizer.

Args:
    model: Model to optimize
    config: Training configuration

Returns:
    Optimizer instance
- `create_scheduler(optimizer, config)` (line 726)
  > Create warmup + cosine scheduler.

Args:
    optimizer: Optimizer to schedule
    config: Training configuration

Returns:
    Scheduler instance
- `setup_gradient_clipping(model, config)` (line 752)
  > Setup gradient clipping and monitoring.

Args:
    model: Model to clip gradients for
    config: Training configuration

Returns:
    Tuple of (clipp...
- `train_step(model, batch, optimizer, scheduler, clipper, ...)` (line 791)
  > Execute single training step.

Args:
    model: Model to train
    batch: Input batch
    optimizer: Optimizer
    scheduler: LR scheduler
    clipper...
- `evaluate(model, val_loader, config) -> float` (line 877)
  > Run evaluation on validation set.

Args:
    model: Model to evaluate
    val_loader: Validation dataloader
    config: Training configuration

Return...
- `save_checkpoint(model, optimizer, scheduler, step, path, ...) -> None` (line 949)
  > Save training checkpoint.

Args:
    model: Model to save
    optimizer: Optimizer state
    scheduler: Scheduler state
    step: Current training ste...
- `train_phase_0_5(model, train_loader, val_loader, optimizer, scheduler, ...)` (line 987)
  > Execute Phase 0.5 training loop.

Args:
    model: Model to train
    train_loader: Training dataloader
    val_loader: Validation dataloader
    opti...
- `run_dry_run(config)` (line 1165)
  > Execute dry run to validate configuration.

Args:
    config: Training configuration

Returns:
    Dictionary with validation results
- `main() -> int` (line 1282)
  > Main entry point.

---

### `train_v3_phase1.py`

**Path:** `scripts\train_v3_phase1.py`

**Classes:**

#### `class Phase1Config` (line 77)
> Configuration specific to Phase 1 multi-task training.

Methods:
- `to_dict()`
- `save()`
- `from_dict()`

#### `class SyntheticMultiTaskDataset(Dataset)` (line 392)
> Synthetic dataset for smoke testing Phase 1 multi-task training.

Methods:
- `__init__()`
- `__len__()`
- `__getitem__()`


**Functions:**

- `parse_args()` (line 176)
  > Parse command line arguments.
- `load_config(args) -> Phase1Config` (line 270)
  > Load and merge configuration from YAML and CLI args.
- `load_familyos_data(config, tokenizer)` (line 493)
  > Load FamilyOS datasets.

Args:
    config: Training configuration
    tokenizer: Tokenizer with hub tokens

Returns:
    Tuple of (train_dataset, val_...
- `load_healing_replay_dataset(config, tokenizer) -> list` (line 563)
  > Load healing data for replay sampling.

Args:
    config: Training configuration
    tokenizer: Tokenizer with hub tokens

Returns:
    List of healin...
- `create_phase1_dataloaders(train_dataset, val_dataset, config)` (line 599)
  > Create train and validation dataloaders.

Args:
    train_dataset: Training dataset
    val_dataset: Validation dataset
    config: Training configura...
- `setup_model(config, tokenizer)` (line 682)
  > Load Phase 0.5 model for Phase 1 training.

Args:
    config: Training configuration
    tokenizer: Tokenizer with hub tokens

Returns:
    Model inst...
- `setup_layer_freezing(model, config)` (line 742)
  > Configure layer freezing for Phase 1.

Same as Phase 0.5: Freeze L1-18 (Foundation + Core), train L19-28.

Args:
    model: Model to configure
    con...
- `setup_hub_gradient_masking(model, config)` (line 775)
  > Setup gradient masking for hub tokens.

Args:
    model: Model to configure
    config: Training configuration

Returns:
    Gradient hook manager
- `create_optimizer(model, config)` (line 799)
  > Create Zipper LR optimizer for Phase 1.

Args:
    model: Model to optimize
    config: Training configuration

Returns:
    Optimizer instance
- `create_scheduler_fn(optimizer, config)` (line 838)
  > Create warmup + cosine scheduler.

Args:
    optimizer: Optimizer to schedule
    config: Training configuration

Returns:
    Scheduler instance
- `setup_gradient_clipping(model, config)` (line 864)
  > Setup gradient clipping.

Args:
    model: Model to clip gradients for
    config: Training configuration

Returns:
    Tuple of (clipper, monitor)
- `setup_loss_function(config)` (line 898)
  > Setup hub-weighted multi-task loss.

Args:
    config: Training configuration

Returns:
    Loss function instance
- `train_step(model, batch, loss_fn, optimizer, scheduler, ...)` (line 932)
  > Execute single training step with multi-task loss.

Args:
    model: Model to train
    batch: Input batch with hub_routing
    loss_fn: Hub-weighted ...
- `evaluate(model, val_loader, loss_fn, config)` (line 1034)
  > Run evaluation on validation set.

Args:
    model: Model to evaluate
    val_loader: Validation dataloader
    loss_fn: Loss function
    config: Tra...
- `save_checkpoint(model, optimizer, scheduler, step, path, ...) -> None` (line 1115)
  > Save training checkpoint with task/hub statistics.

Args:
    model: Model to save
    optimizer: Optimizer state
    scheduler: Scheduler state
    s...
- `train_phase_1(model, train_loader, val_loader, optimizer, scheduler, ...)` (line 1160)
  > Execute Phase 1 multi-task training loop.

Args:
    model: Model to train
    train_loader: Training dataloader
    val_loader: Validation dataloader...
- `run_dry_run(config)` (line 1389)
  > Execute dry run to validate configuration.

Args:
    config: Training configuration

Returns:
    Dictionary with validation results
- `main() -> int` (line 1522)
  > Main entry point.

---

### `prepare_healing_data.py`

**Path:** `scripts\prepare_healing_data.py`

**Lines:** 388

**Functions:**

- `parse_args()` - Parse command line arguments
- `load_and_sample_dataset()` - Load and sample from HuggingFace datasets
- `convert_sst2_sample()` - Convert SST-2 samples to healing format
- `_bio_tags_to_spans()` - Convert BIO tags to span format
- `convert_conll_sample()` - Convert CoNLL samples to healing format
- `convert_mnli_sample()` - Convert MNLI samples to healing format
- `prepare_healing_data()` - Main healing data preparation
- `_write_jsonl()` - Write samples to JSONL file
- `save_healing_data()` - Save prepared healing data
- `validate_healing_data()` - Validate healing data integrity
- `main()` - Main entry point

---

### `prepare_healing_data_enhanced.py`

**Path:** `scripts\prepare_healing_data_enhanced.py`

**Lines:** 475

**Functions:**

- `parse_args()` - Parse command line arguments
- `load_and_sample_dataset()` - Load and sample from HuggingFace datasets
- `_bio_tags_to_spans()` - Convert BIO tags to span format
- `convert_sst2_sample()` - Convert SST-2 samples to healing format
- `convert_conll_sample()` - Convert CoNLL samples to healing format
- `convert_mnli_sample()` - Convert MNLI samples to healing format
- `convert_squad_sample()` - Convert SQuAD samples to healing format
- `convert_stsb_sample()` - Convert STS-B samples to healing format
- `prepare_enhanced_healing_data()` - Main enhanced healing data preparation
- `_write_jsonl()` - Write samples to JSONL file
- `save_enhanced_healing_data()` - Save prepared enhanced healing data
- `validate_enhanced_healing_data()` - Validate enhanced healing data integrity
- `main()` - Main entry point

---

## TRAINERS

### `freezing_v3.py`

**Path:** `src\modeling_studio\trainers\freezing_v3.py`

**Classes:**

#### `class LayerBand(Enum)` (line 32)
> Layer bands in v3 architecture.

#### `class TrainingPhase(Enum)` (line 50)
> Training phases for v3.

#### `class LayerFreezer` (line 73)
> Manages layer freezing for phase-based training.

Freeze Strategy:
    Phase 0.5 (Healing):
        - Frozen: L1-18 (Foundation + Core)
        - Trainable: L19-28 (SEMANTIC + Family)
        - Purpose:...

Methods:
- `__init__()`
- `get_layer()`
- `freeze_layer()`
- `unfreeze_layer()`
- `freeze_band()`
- `unfreeze_band()`
- `freeze_embeddings()`
- `unfreeze_embeddings()`
- `freeze_hub_tokens()`
- `is_layer_frozen()`
- `is_band_frozen()`
- `configure_for_phase()`
- `get_freeze_stats()`
- `get_frozen_layers()`
- `get_trainable_layers()`
- `_print_freeze_summary()`


**Functions:**

- `configure_model_for_phase(model, phase)` (line 405)
  > Configure model freezing for a training phase.

This is a convenience function that creates a LayerFreezer and
configures the model for the specified ...
- `get_band_for_layer(layer_idx)` (line 431)
  > Get the band that a layer belongs to.

Args:
    layer_idx: Layer index (0-indexed)

Returns:
    The LayerBand the layer belongs to, or None if out o...
- `get_layers_for_band(band)` (line 447)
  > Get layer indices for a band.

Args:
    band: The layer band

Returns:
    List of layer indices (0-indexed)
- `get_trainable_bands_for_phase(phase)` (line 460)
  > Get which bands are trainable for a given phase.

Args:
    phase: Phase name or TrainingPhase enum

Returns:
    List of trainable bands

---

### `gradient_masking_v3.py`

**Path:** `src\modeling_studio\trainers\gradient_masking_v3.py`

**Constants:**
- `V2_VOCAB_SIZE`
- `HUB_TOKEN_START`
- `HUB_TOKEN_COUNT`
- `V3_VOCAB_SIZE`

**Classes:**

#### `class GradientMaskConfig` (line 50)
> Configuration for gradient masking.

Attributes:
    train_hub_tokens: List of hub token names to train. None = all.
    freeze_original_vocab: Whether to freeze original vocab (0-50367).
    hub_toke...

Methods:
- `__post_init__()`
- `to_dict()`
- `from_dict()`

#### `class EmbeddingGradientHook` (line 86)
> Gradient hook for selective embedding training.

Applies gradient masking to word embeddings to:
1. Zero gradients for frozen token positions
2. Scale gradients for hub tokens
3. Enable per-token trai...

Methods:
- `__init__()`
- `embedding_weight()`
- `_build_gradient_mask()`
- `_gradient_hook()`
- `register()`
- `remove()`
- `is_registered()`
- `update_trainable_tokens()`
- `update_grad_scale()`
- `get_mask_stats()`

#### `class HubTokenGradientManager` (line 260)
> Manages hub token gradient masking for a model.

Provides high-level interface for controlling hub token training:
- Setup/cleanup gradient hooks
- Freeze/unfreeze specific hub tokens
- Get hub token ...

Methods:
- `__init__()`
- `get_embedding_weight()`
- `setup()`
- `cleanup()`
- `is_setup()`
- `freeze_all_hub_tokens()`
- `unfreeze_all_hub_tokens()`
- `train_specific_hub_tokens()`
- `set_grad_scale()`
- `get_hub_token_gradients()`
- `get_hub_token_embeddings()`
- `get_stats()`


**Functions:**

- `setup_hub_token_gradient_masking(model, train_hub_tokens, freeze_original_vocab, hub_token_grad_scale) -> HubTokenGradientManager` (line 480)
  > Setup hub token gradient masking for a model.

This is a convenience function that creates a HubTokenGradientManager
and sets up gradient masking in o...
- `get_hub_token_positions()` (line 524)
  > Get hub token positions in vocabulary.

Returns:
    Dict mapping token name to position
- `get_vocab_layout()` (line 534)
  > Get vocabulary layout constants.

Returns:
    Dict with V2_VOCAB_SIZE, HUB_TOKEN_START, HUB_TOKEN_COUNT, V3_VOCAB_SIZE

---

### `gradient_utils_v3.py`

**Path:** `src\modeling_studio\trainers\gradient_utils_v3.py`

**Classes:**

#### `class GradientClipConfig` (line 39)
> Configuration for gradient clipping.

Attributes:
    max_grad_norm: Maximum gradient norm for global clipping
    per_layer_clip: Whether to apply per-layer clipping
    interface_clip: Clip threshol...

#### `class GradientStats` (line 74)
> Statistics about gradients.

Attributes:
    total_norm: Total gradient norm
    layer_norms: Per-layer gradient norms
    max_grad: Maximum gradient value
    min_grad: Minimum gradient value
    has...

#### `class GradientClipper` (line 102)
> Gradient clipping and monitoring for v3 training.

Provides:
1. Global gradient clipping (standard)
2. Per-layer gradient clipping (for interface sensitivity)
3. Gradient norm monitoring
4. NaN/Inf de...

Methods:
- `__init__()`
- `clip_gradients()`
- `_check_gradient_health()`
- `_zero_bad_gradients()`
- `_compute_layer_norms()`
- `_global_clip()`
- `_per_layer_clip()`
- `_log_gradient_stats()`
- `get_gradient_summary()`
- `clear_history()`
- `reset()`

#### `class InterfaceGradientMonitor` (line 398)
> Specialized monitor for L22->L23 interface gradients.

The interface between v2 (L22) and v3 (L23) is the most sensitive
region during healing. This monitor tracks gradient flow across
this boundary.
...

Methods:
- `__init__()`
- `record()`
- `_layer_grad_norm()`
- `get_interface_health()`
- `clear_history()`


**Functions:**

- `clip_gradients(model, max_norm, per_layer) -> float` (line 525)
  > Clip gradients for a model.

This is a convenience function that creates a temporary GradientClipper
and applies clipping. For repeated use, create a ...
- `create_gradient_clipper(model, max_grad_norm, per_layer_clip, interface_clip, log_every_n_steps) -> GradientClipper` (line 556)
  > Create a GradientClipper with common settings.

Args:
    model: Model to clip gradients for
    max_grad_norm: Maximum gradient norm for global clipp...

---

### `lora_v3.py`

**Path:** `src\modeling_studio\trainers\lora_v3.py`

**Classes:**

#### `class LoRAConfig` (line 44)
> Configuration for LoRA adapters.

Attributes:
    rank: LoRA rank (r) - number of low-rank dimensions
    alpha: Scaling factor (alpha) - controls update magnitude
    dropout: Dropout probability on ...

Methods:
- `__post_init__()`
- `scaling()`
- `to_dict()`
- `from_dict()`

#### `class LoRALinear(nn.Module)` (line 105)
> Linear layer with LoRA (Low-Rank Adaptation) adapter.

LoRA decomposes the weight update as a low-rank product:
    W' = W + ΔW = W + BA

Where:
    - W: Original frozen weights [out_features, in_feat...

Methods:
- `__init__()`
- `_init_lora_weights()`
- `forward()`
- `merge_weights()`
- `unmerge_weights()`
- `get_lora_params()`
- `from_linear()`
- `extra_repr()`

#### `class LoRAManager` (line 316)
> Manages LoRA application to a model.

This class handles:
    - Applying LoRA to specific layers and modules
    - Tracking all LoRA modules for parameter access
    - Merging/unmerging all LoRA weigh...

Methods:
- `__init__()`
- `apply_lora()`
- `_get_module()`
- `_set_module()`
- `get_lora_parameters()`
- `get_lora_state_dict()`
- `merge_all()`
- `unmerge_all()`
- `enable_lora()`
- `save_lora_weights()`
- `load_lora_weights()`
- `get_stats()`
- `print_summary()`


**Functions:**

- `apply_lora_to_family_band(model, rank, alpha, dropout, target_modules) -> LoRAManager` (line 635)
  > Apply LoRA to Family Band (L23-28).

This is a convenience function that creates a LoRAConfig targeting
the Family Band layers with sensible defaults....
- `get_lora_param_count(hidden_size, rank, num_layers, num_modules_per_layer) -> int` (line 680)
  > Calculate expected LoRA parameter count.

Args:
    hidden_size: Model hidden dimension
    rank: LoRA rank
    num_layers: Number of layers with LoRA...

---

### `lr_groups_v3.py`

**Path:** `src\modeling_studio\trainers\lr_groups_v3.py`

**Classes:**

#### `class LayerGroupLRConfig` (line 34)
> Configuration for layer-group learning rates.

Rationale:
    - Foundation/Core (L1-18): Very low or frozen - preserve v2 knowledge
    - SEMANTIC (L19-22): Low LR - gentle refinement of interface
    -...

Methods:
- `get_layer_lr()`
- `get_component_lr()`
- `get_band_lr()`
- `get_warmup_steps()`
- `get_min_lr()`
- `to_dict()`
- `from_dict()`

#### `class LayerGroupOptimizer` (line 241)
> Creates optimizer with layer-group specific learning rates.

This class builds parameter groups for each layer band with appropriate
learning rates, enabling fine-grained control over training dynamic...

Methods:
- `__init__()`
- `_get_encoder()`
- `_has_layers()`
- `_get_num_layers()`
- `create_optimizer()`
- `_build_param_groups()`
- `_get_embedding_groups()`
- `_get_task_head_groups()`
- `_get_remaining_groups()`
- `get_param_groups()`
- `_log_param_groups()`


**Functions:**

- `get_band_for_layer(layer_idx) -> str` (line 225)
  > Get the band name for a layer index.

Args:
    layer_idx: 0-indexed layer index

Returns:
    Band name
- `create_layer_group_optimizer(model, phase, base_lr, weight_decay, optimizer_class)` (line 494)
  > Create optimizer with phase-appropriate layer-group LRs.

This is a convenience function that creates a LayerGroupOptimizer
with phase-specific config...
- `get_phase_config(phase) -> LayerGroupLRConfig` (line 541)
  > Get the LR config for a training phase.

Args:
    phase: Training phase name

Returns:
    LayerGroupLRConfig for the phase
- `print_lr_summary(config) -> None` (line 558)
  > Print a summary of learning rates for all components.

Args:
    config: LayerGroupLRConfig to summarize

---

### `schedulers_v3.py`

**Path:** `src\modeling_studio\trainers\schedulers_v3.py`

**Constants:**
- `VALID_SCHEDULER_TYPES`

**Classes:**

#### `class WarmupCosineScheduler(_LRScheduler)` (line 40)
> Learning rate scheduler with linear warmup and cosine decay.

LR Profile:
    Warmup Phase (steps 0 to warmup_steps):
        lr = base_lr * (step / warmup_steps)

    Cosine Decay Phase (steps warmup...

Methods:
- `__init__()`
- `get_lr()`
- `get_lr_at_step()`

#### `class WarmupLinearScheduler(_LRScheduler)` (line 147)
> Learning rate scheduler with linear warmup and linear decay.

Simpler than cosine but can be effective for shorter training runs.

LR Profile:
    Warmup Phase (steps 0 to warmup_steps):
        lr = ...

Methods:
- `__init__()`
- `get_lr()`
- `get_lr_at_step()`

#### `class WarmupConstantScheduler(_LRScheduler)` (line 229)
> Learning rate scheduler with linear warmup then constant LR.

Useful for short fine-tuning runs where decay isn't beneficial.

LR Profile:
    Warmup Phase (steps 0 to warmup_steps):
        lr = base...

Methods:
- `__init__()`
- `get_lr()`
- `get_lr_at_step()`

#### `class PhaseAwareScheduler` (line 290)
> Scheduler that handles phase transitions in v3 training.

Manages separate schedulers for each phase and handles transitions
between phases. Each phase can have its own warmup, decay, and LR settings....

Methods:
- `__init__()`
- `set_phase()`
- `step()`
- `get_last_lr()`
- `get_phase_progress()`
- `is_warmup_complete()`
- `get_state_dict()`
- `load_state_dict()`


**Functions:**

- `create_scheduler(optimizer, scheduler_type, warmup_steps, total_steps, min_lr_ratio) -> _LRScheduler` (line 447)
  > Create a learning rate scheduler.

Args:
    optimizer: Wrapped optimizer
    scheduler_type: "cosine", "linear", or "constant"
    warmup_steps: Numb...
- `create_phase_aware_scheduler(optimizer, phase_configs) -> PhaseAwareScheduler` (line 504)
  > Create a phase-aware scheduler with optional custom configs.

Args:
    optimizer: Wrapped optimizer
    phase_configs: Optional custom phase configur...
- `compute_warmup_steps(total_steps, warmup_ratio, min_warmup, max_warmup) -> int` (line 529)
  > Compute warmup steps based on total steps and ratio.

Args:
    total_steps: Total training steps
    warmup_ratio: Ratio of total steps for warmup
  ...
- `get_lr_at_step(scheduler, step)` (line 551)
  > Get learning rate at a specific step without modifying scheduler state.

Args:
    scheduler: LR scheduler
    step: Step number

Returns:
    List of...
- `print_scheduler_profile(scheduler, total_steps, num_points) -> None` (line 576)
  > Print scheduler LR profile at key points.

Args:
    scheduler: LR scheduler
    total_steps: Total training steps
    num_points: Number of points to...

---

### `trainer_v3.py`

**Path:** `src\modeling_studio\trainers\trainer_v3.py`

**Classes:**

#### `class TrainingConfig` (line 55)
> Configuration for v3 training.

This dataclass contains all configuration options for phase-based
training of ModernBERT v3 models.

Attributes:
    phase: Training phase ("phase_0.5", "phase_1", "pha...

Methods:
- `to_dict()`
- `save()`
- `from_dict()`
- `load()`

#### `class TrainingState` (line 157)
> Tracks training state.

This dataclass maintains the current state of training, including
step counts, metrics history, and losses.

Attributes:
    global_step: Current training step
    epoch: Curre...

Methods:
- `to_dict()`
- `from_dict()`

#### `class ModernBERTv3Trainer` (line 190)
> Phase-aware trainer for ModernBERT v3.

This trainer implements phase-based training with layer freezing,
per-layer learning rates, and comprehensive logging and checkpointing.

Training Phases:
    P...

Methods:
- `__init__()`
- `setup()`
- `_create_optimizer()`
- `_get_parameter_groups()`
- `_create_scheduler()`
- `_init_wandb()`
- `train()`
- `_move_batch_to_device()`
- `_training_step()`
- `_compute_loss()`
- `evaluate()`
- `_log_step()`
- `_log_eval()`
- `_save_checkpoint()`
- `load_checkpoint()`
- `get_trainable_params()`
- `get_total_params()`
- `print_training_summary()`


---

### `zipper_lr_v3.py`

**Path:** `src\modeling_studio\trainers\zipper_lr_v3.py`

**Constants:**
- `FOUNDATION_END`
- `CORE_END`
- `SEMANTIC_END`
- `INTERFACE_LAYER`
- `FAMILY_END`
- `V3_LAYER_COUNT`
- `ZIPPER_LR_QUICK_REF`

**Classes:**

#### `class ZipperLRConfig` (line 55)
> Configuration for Zipper Learning Rate strategy.

The Zipper strategy creates a smooth LR transition across the
v2-v3 interface boundary to prevent gradient discontinuities.

Layer Layout:
    L1-18: ...

Methods:
- `get_layer_lr()`
- `get_all_layer_lrs()`
- `get_trainable_layer_lrs()`
- `get_band_summary()`

#### `class ZipperLROptimizer` (line 282)
> Creates optimizer using Zipper Learning Rate strategy.

The Zipper method ensures:
    1. Smooth LR transition at v2-v3 interface
    2. Maximum plasticity at L23 (interface layer)
    3. Graduated LR...

Methods:
- `__init__()`
- `_get_encoder()`
- `_get_layers()`
- `create_optimizer()`
- `_build_zipper_param_groups()`
- `_print_zipper_summary()`
- `get_lr_dict()`
- `get_param_group_count()`
- `get_trainable_param_count()`


**Functions:**

- `get_zipper_preset(preset_name) -> ZipperLRConfig` (line 248)
  > Get a Zipper LR preset configuration.

Args:
    preset_name: Name of the preset

Returns:
    ZipperLRConfig for the preset

Raises:
    ValueError: ...
- `list_zipper_presets()` (line 267)
  > List available Zipper LR presets.

Returns:
    List of preset names
- `create_zipper_optimizer(model, preset, weight_decay)` (line 533)
  > Create optimizer with Zipper LR strategy.

Args:
    model: ModernBERTv3 model
    preset: Preset name from ZIPPER_PRESETS
    weight_decay: Weight de...
- `print_zipper_lr_profile(config) -> None` (line 581)
  > Print the Zipper LR profile for a configuration.

Args:
    config: Zipper LR configuration
- `compare_zipper_presets()` (line 615)
  > Compare learning rates across all Zipper presets.

Returns:
    Dictionary mapping preset names to layer LR dictionaries
- `validate_zipper_config(config)` (line 634)
  > Validate a Zipper LR configuration.

Args:
    config: Configuration to validate

Returns:
    List of warning messages (empty if valid)

---

## TRAINING

### `losses_v3.py`

**Path:** `src\modeling_studio\training\losses_v3.py`

**Classes:**

#### `class HubLossConfig` (line 20)
> Configuration for hub-weighted loss scaling.

#### `class HubLossWeightCalculator` (line 49)
> Calculates per-sample, per-task loss weights based on hub routing.

Methods:
- `__init__()`
- `compute_weight()`
- `compute_batch_weights()`

#### `class HubWeightedMultiTaskLoss(nn.Module)` (line 100)
> Hub-aware multi-task loss with per-sample task weighting.

Methods:
- `__init__()`
- `forward()`
- `_compute_task_loss()`
- `_compute_token_loss()`
- `_get_has_labels()`
- `_get_device()`

#### `class HubGradientMaskedLoss(nn.Module)` (line 211)
> Apply hub token gradient masking before delegating to a base loss.

Methods:
- `__init__()`
- `get_hub_gradient_mask()`
- `forward()`


**Functions:**

- `aggregate_task_losses(task_losses, task_weights)` (line 258)
  > Aggregate per-task losses using optional weights.
- `log_task_losses(task_losses, prefix)` (line 270)
  > Convert task losses to scalars for logging.
- `_get_first_device(tensors)` (line 278)
  > Helper to infer device from dict of tensors.

---
