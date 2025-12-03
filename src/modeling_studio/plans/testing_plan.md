# FamilyOS Unified Encoder - Project Structure & Test Plan

---

## Milestone 1: Core Infrastructure

### Epic 1.1: Package Foundation

#### Issue 1.1.1: `__init__.py`

**Tests:**

- `test_version_exists` - Verify `__version__` is defined
- `test_author_exists` - Verify `__author__` is defined
- `test_config_import` - Verify `Config` class can be imported from package root

---

#### Issue 1.1.2: `cli.py`

**Tests:**

- `test_app_exists` - Verify Typer app is instantiated
- `test_train_command_exists` - Verify `train()` function is defined
- `test_evaluate_command_exists` - Verify `evaluate()` function is defined
- `test_cli_help` - Verify CLI responds to `--help` flag

---

#### Issue 1.1.3: `config.py`

**Tests:**

- `test_model_config_defaults` - Verify `ModelConfig` default values (type="encoder", name_or_path, torch_dtype)
- `test_training_config_defaults` - Verify `TrainingConfig` defaults (learning_rate=2e-5, num_train_epochs=3, etc.)
- `test_data_config_defaults` - Verify `DataConfig` defaults (max_length=512, truncation=True)
- `test_peft_config_defaults` - Verify `PEFTConfig` defaults (method="lora", r=16, lora_alpha=32)
- `test_config_from_yaml` - Load config from YAML file, verify all fields populated
- `test_config_from_yaml_missing_file` - Verify `FileNotFoundError` raised for missing file
- `test_config_from_dict` - Create config from dictionary
- `test_config_to_dict` - Convert config to dictionary, verify round-trip
- `test_config_save` - Save config to YAML, verify file created
- `test_config_partial_yaml` - Load YAML with only some sections, verify defaults used

---

### Epic 1.2: Utilities

#### Issue 1.2.1: `utils/__init__.py`

**Tests:**

- `test_all_exports` - Verify all functions in `__all__` are importable
- `test_setup_logging_export` - Verify `setup_logging` is exported
- `test_get_device_export` - Verify `get_device` is exported
- `test_set_seed_export` - Verify `set_seed` is exported

---

#### Issue 1.2.2: `utils/device.py`

**Tests:**

- `test_get_device_returns_device` - Verify returns `torch.device` instance
- `test_get_device_cpu_fallback` - On CPU-only system, returns `cpu` device
- `test_get_device_map_auto` - `get_device_map("auto")` returns `"auto"`
- `test_get_device_map_cpu` - `get_device_map("cpu")` returns `{"": "cpu"}`
- `test_get_device_map_cuda` - `get_device_map("cuda")` returns `{"": 0}`
- `test_get_device_map_balanced` - `get_device_map("balanced")` returns `"balanced"`
- `test_get_torch_dtype_float32` - `get_torch_dtype("float32")` returns `torch.float32`
- `test_get_torch_dtype_float16` - `get_torch_dtype("float16")` returns `torch.float16`
- `test_get_torch_dtype_bfloat16` - `get_torch_dtype("bfloat16")` returns `torch.bfloat16`
- `test_get_torch_dtype_alias` - `get_torch_dtype("fp16")` returns `torch.float16`
- `test_set_seed_reproducibility` - Set seed, generate random, reset seed, verify same output
- `test_get_num_gpus` - Returns integer >= 0
- `test_print_gpu_memory` - Runs without error (smoke test)
- `test_setup_environment` - Verifies `TOKENIZERS_PARALLELISM` env var set

---

#### Issue 1.2.3: `utils/logging.py`

**Tests:**

- `test_setup_logging_returns_logger` - Returns `logging.Logger` instance
- `test_setup_logging_level_info` - Logger level is `INFO` by default
- `test_setup_logging_level_debug` - Can set level to `DEBUG`
- `test_setup_logging_with_file` - Creates log file when path specified
- `test_setup_logging_rich_handler` - Uses `RichHandler` when `use_rich=True`
- `test_setup_logging_no_rich` - Uses `StreamHandler` when `use_rich=False`
- `test_get_logger_named` - `get_logger("test")` returns logger named `modeling_studio.test`
- `test_get_logger_unnamed` - `get_logger()` returns logger named `modeling_studio`

---

### Milestone 1 Integration Points

| From | To | Connection |
|------|-----|------------|
| `__init__.py` | `config.py` | Imports and exports `Config` class |
| `utils/__init__.py` | `utils/device.py` | Re-exports all device functions |
| `utils/__init__.py` | `utils/logging.py` | Re-exports logging functions |
| `cli.py` | `config.py` | CLI commands will use Config for loading settings |
| `cli.py` | `utils/logging.py` | CLI uses `setup_logging()` for output |
| `cli.py` | `utils/device.py` | CLI uses `get_device()` for hardware detection |

**Future Integration Tests (Milestone 1):**

- `test_cli_loads_config` - CLI can load config from file and use settings
- `test_logging_with_device_info` - Logger outputs device information correctly
- `test_full_init_sequence` - Package init → logging setup → device detection → config load

---

## Milestone 2: Data Pipeline

### Epic 2.1: Data Core

#### Issue 2.1.1: `data/__init__.py`

**Tests:**

- `test_label_schemas_exported` - Verify all label schemas in `__all__` are importable
- `test_capability_enum_exported` - Verify `Capability` enum is exported
- `test_get_labels_for_capability_exported` - Verify helper function exported
- `test_all_12_capabilities_available` - Verify all 12 capability label schemas accessible

---

#### Issue 2.1.2: `data/labels.py`

**Tests:**

- `test_label_schema_encode` - `schema.encode("B-PER")` returns correct ID
- `test_label_schema_decode` - `schema.decode(1)` returns correct label string
- `test_label_schema_num_labels` - `schema.num_labels` returns correct count
- `test_label_schema_id2label` - `schema.id2label` property works correctly
- `test_label_schema_to_dict` - Serialization to dict works
- `test_label_schema_from_dict` - Deserialization from dict works
- `test_ner_general_labels_count` - NER_GENERAL has 17 BIO tags
- `test_ner_family_labels_count` - NER_FAMILY has 21 BIO tags
- `test_sentiment_labels_count` - SENTIMENT has 5 classes
- `test_emotions_familyos_labels_count` - EMOTIONS_FAMILYOS has 44 classes
- `test_safety_generic_labels_count` - SAFETY_GENERIC has 8 types
- `test_safety_familyos_labels_count` - SAFETY_FAMILYOS has 4 bands
- `test_nli_labels_count` - NLI has 3 classes
- `test_ingress_labels_count` - INGRESS has 12 domains
- `test_temporal_labels_count` - TEMPORAL has 13 BIO tags
- `test_relation_labels_count` - RELATION has 15 relations
- `test_intent_labels_count` - INTENT has 8 intents
- `test_subcategory_to_band_mapping` - SUBCATEGORY_TO_BAND_ID maps correctly
- `test_band_to_subcategory_mapping` - BAND_TO_SUBCATEGORY_IDS maps correctly
- `test_capability_enum_values` - All 12 capabilities have correct string values
- `test_capability_to_labels_mapping` - CAPABILITY_TO_LABELS returns correct schema
- `test_get_labels_for_capability_string` - Works with string input
- `test_get_labels_for_capability_enum` - Works with Capability enum input
- `test_get_num_labels` - Returns correct label count for capability
- `test_embedding_capability_no_labels` - Embedding returns None for labels

---

#### Issue 2.1.3: `data/loaders.py`

**Tests:**

- `test_load_ner_dataset_conll2003` - Load CoNLL-2003, verify tokens and ner_tags columns
- `test_load_ner_dataset_label_remapping` - Labels remapped to NER_GENERAL schema
- `test_load_ner_from_jsonl` - Load from local JSONL file
- `test_load_ner_from_directory` - Load from directory with train/val/test splits
- `test_load_ner_string_to_id_conversion` - String labels converted to IDs
- `test_load_classification_dataset_sst2` - Load SST-2, verify text and label columns
- `test_load_classification_label_mapping` - Binary labels mapped to 5-class sentiment
- `test_load_classification_from_csv` - Load from local CSV file
- `test_load_classification_from_jsonl` - Load from local JSONL file
- `test_load_multilabel_dataset` - Load GoEmotions, verify multi-hot encoding
- `test_load_nli_dataset_mnli` - Load MNLI, verify premise/hypothesis/label
- `test_load_nli_dataset_snli` - Load SNLI, verify format
- `test_load_embedding_dataset` - Load STS-B, verify sentence pairs and scores
- `test_load_familyos_ner` - Load custom family NER data
- `test_load_familyos_ingress` - Load custom ingress data
- `test_load_familyos_safety` - Load custom safety band data
- `test_wikineural_label_mapping` - WikiNeural 33 labels mapped to CoNLL 9
- `test_keep_datasets_in_memory` - KEEP_DATASETS_IN_MEMORY flag works

---

#### Issue 2.1.4: `data/tokenization.py`

**Tests:**

- `test_load_tokenizer` - Load ModernBERT tokenizer successfully
- `test_load_tokenizer_vocab_size` - Tokenizer has expected vocab size
- `test_tokenize_for_classification` - Returns input_ids and attention_mask
- `test_tokenize_for_classification_truncation` - Truncates to max_length
- `test_tokenize_for_multilabel` - Returns multi-hot encoded labels
- `test_tokenize_for_multilabel_num_labels_required` - Raises error if num_labels missing
- `test_tokenize_for_token_classification` - Returns aligned labels
- `test_tokenize_for_token_classification_word_ids` - Returns word_ids for alignment
- `test_subword_alignment_first_only` - Only first subword gets label (default)
- `test_subword_alignment_all_tokens` - All subwords get label when flag set
- `test_tokenize_for_nli` - Encodes premise-hypothesis pairs correctly
- `test_tokenize_for_nli_separator` - Has separator token between sentences
- `test_tokenize_for_embedding` - Returns embeddings-ready input
- `test_tokenize_batch` - Batch tokenization works correctly
- `test_ignore_index_value` - IGNORE_INDEX is -100

---

#### Issue 2.1.5: `data/multitask_dataset.py`

**Tests:**

- `test_task_dataset_len` - `len(TaskDataset)` returns dataset size
- `test_task_dataset_getitem` - `TaskDataset[0]` returns sample with task field
- `test_task_dataset_task_field` - Sample has correct task name
- `test_task_dataset_preprocessing` - Preprocessing function applied
- `test_task_dataset_weight` - Weight attribute set correctly
- `test_task_dataset_select` - `select(indices)` returns subset
- `test_task_dataset_shuffle` - `shuffle()` randomizes order
- `test_task_dataset_column_names` - Includes "task" in column names
- `test_multitask_dataset_len` - Total length is sum of all task lengths
- `test_multitask_dataset_getitem` - Returns correct sample from correct task
- `test_multitask_dataset_task_names` - `task_names` property correct
- `test_multitask_dataset_task_sizes` - `task_sizes` dict correct
- `test_multitask_dataset_get_task_dataset` - Retrieves specific task dataset
- `test_multitask_dataset_get_task_samples` - Iterator for specific task
- `test_multitask_dataset_shuffle` - Shuffling works with reshuffle()
- `test_multitask_dataset_split_by_task` - Splits back into individual TaskDatasets
- `test_multitask_dataset_binary_search` - Index lookup uses binary search
- `test_multitask_dataset_empty_error` - Raises error for empty list
- `test_streaming_multitask_dataset` - StreamingMultiTaskDataset initializes
- `test_streaming_task_weights_normalized` - Weights sum to 1.0

---

### Epic 2.2: Data Processing

#### Issue 2.2.1: `data/preprocessing.py`

**Tests:**

- `test_clean_text_basic` - Basic cleaning (whitespace, control chars)
- `test_clean_text_unicode_normalization` - NFKC normalization applied
- `test_clean_text_lowercase` - Lowercase when flag set
- `test_preprocess_config_defaults` - Verify default config values
- `test_remove_urls` - URLs removed when flag set
- `test_remove_emails` - Emails removed when flag set
- `test_remove_mentions` - @mentions removed when flag set
- `test_remove_hashtags` - #hashtags removed when flag set
- `test_emoji_handling_keep` - Emojis preserved
- `test_emoji_handling_remove` - Emojis removed
- `test_emoji_handling_replace` - Emojis replaced with placeholder
- `test_collapse_punctuation` - "!!!" becomes "!"
- `test_truncation_head` - Keeps beginning of text
- `test_truncation_tail` - Keeps end of text
- `test_truncation_middle` - Keeps beginning and end
- `test_kinship_terms_mapping` - Indian kinship terms recognized
- `test_kinship_normalize_to_english` - Normalizes "nani" to "grandmother"
- `test_preserve_kinship_terms` - Kinship terms not altered when flag set
- `test_task_specific_preprocessing` - Different preprocessing for NER vs sentiment

---

#### Issue 2.2.2: `data/augmentation.py`

**Tests:**

- `test_kinship_variants_defined` - All kinship variant lists populated
- `test_mother_variants_complete` - ALL_MOTHER_VARIANTS has English and Indian terms
- `test_father_variants_complete` - ALL_FATHER_VARIANTS has English and Indian terms
- `test_grandmother_variants_multicultural` - Includes Indian, Spanish, Filipino
- `test_kinship_variants_mapping` - KINSHIP_VARIANTS maps standard to variants
- `test_augment_kinship_replacement` - "Mom" augmented to "Mum", "Amma", etc.
- `test_augment_kinship_case_preservation` - Preserves original casing
- `test_augment_nickname_generation` - Generates plausible nickname variations
- `test_back_translation_paraphrase` - Generates semantic paraphrases
- `test_random_masking` - Masks tokens for MLM training
- `test_synonym_replacement` - Replaces words with synonyms
- `test_character_augmentation` - Adds typos for robustness
- `test_augmentation_deterministic` - Same seed gives same augmentation

---

#### Issue 2.2.3: `data/cultural_mappings.py`

**Tests:**

- `test_indian_english_mappings_defined` - INDIAN_ENGLISH_MAPPINGS has entries
- `test_doing_needful_mapping` - "doing the needful" → "doing what's needed"
- `test_revert_back_mapping` - "revert back" → "respond"
- `test_passed_out_mapping` - "passed out from college" → "graduated from college"
- `test_today_morning_mapping` - "today morning" → "this morning"
- `test_lakh_crore_mapping` - Indian number words mapped
- `test_indian_venting_patterns_defined` - INDIAN_VENTING_PATTERNS is frozenset
- `test_venting_die_of_embarrassment` - "I'll die of embarrassment" recognized
- `test_venting_killing_me` - "this is killing me" recognized
- `test_venting_head_bursting` - "my head will burst" recognized
- `test_venting_patterns_not_crisis` - Venting patterns should NOT trigger CRISIS
- `test_indian_kinship_all_regions` - North, South, Bengali, Marathi, Gujarati covered
- `test_kinship_maternal_paternal` - Maternal/paternal distinctions preserved
- `test_normalizer_class_exists` - IndianEnglishNormalizer class defined
- `test_normalizer_apply` - Normalizer transforms text correctly
- `test_family_structure_types` - Family structure classifications defined

---

### Milestone 2 (Epic 2.1 + 2.2) Integration Points

| From | To | Connection |
|------|-----|------------|
| `data/__init__.py` | `data/labels.py` | Exports all label schemas and Capability enum |
| `data/loaders.py` | `data/labels.py` | Uses label schemas for dataset loading and remapping |
| `data/tokenization.py` | `data/labels.py` | Uses label schemas for multi-label encoding |
| `data/multitask_dataset.py` | `data/loaders.py` | TaskDataset wraps loaded HF datasets |
| `data/preprocessing.py` | `data/cultural_mappings.py` | Uses kinship terms for preservation |
| `data/preprocessing.py` | `data/augmentation.py` | Can apply augmentation during preprocessing |
| `data/augmentation.py` | `data/cultural_mappings.py` | Uses kinship variants for augmentation |
| `data/tokenization.py` | `data/multitask_dataset.py` | Tokenization applied via preprocessing_fn |
| `data/loaders.py` | `data/preprocessing.py` | Loaded data goes through preprocessing |

**Future Integration Tests (Milestone 2):**

- `test_load_and_tokenize_ner` - Load NER dataset, tokenize, verify alignment
- `test_load_and_preprocess_classification` - Load sentiment, preprocess, verify cleaning
- `test_multitask_dataset_from_loaders` - Create MultiTaskDataset from multiple loaders
- `test_augmentation_in_preprocessing` - Apply augmentation during TaskDataset preprocessing
- `test_cultural_mapping_in_pipeline` - Indian text normalized before tokenization
- `test_full_data_pipeline` - Raw data → Load → Preprocess → Augment → Tokenize → MultiTaskDataset

---

## Milestone 3: Models

### Epic 3.1: Model Core

#### Issue 3.1.1: `models/__init__.py`

**Tests:**

- `test_model_class_exports` - Verify `ModernBertMultiTaskModel` exported
- `test_multi_task_output_exported` - Verify `MultiTaskOutput` exported
- `test_capability_to_head_type_exported` - Verify mapping exported
- `test_all_head_classes_exported` - BaseHead, SequenceClassificationHead, TokenClassificationHead, etc.
- `test_pooler_classes_exported` - CLSPooler, MeanPooler, AttentionPooler exported
- `test_adapter_classes_exported` - BottleneckAdapter, TaskGroupAdapter, LoRAAdapter exported
- `test_pair_encoder_exported` - CrossAttentionPairEncoder, ConcatPairEncoder exported

---

#### Issue 3.1.2: `models/modernbert_multitask.py`

**Tests:**

- `test_model_init` - Model initializes with default capabilities
- `test_model_init_specific_capabilities` - Initialize with subset of capabilities
- `test_model_heads_created` - Heads created for each capability
- `test_normalize_capabilities_string` - String capabilities converted to Capability enum
- `test_normalize_capabilities_enum` - Capability enum passed through
- `test_task_groups_defined` - TASK_GROUPS has token_tasks, sequence_tasks, pair_tasks, embedding_tasks
- `test_get_task_group` - Returns correct group for capability
- `test_capability_to_head_type_mapping` - 12 capabilities mapped to correct head types
- `test_get_problem_type` - Returns correct problem type for each capability
- `test_multi_task_output_init` - MultiTaskOutput initializes with all fields
- `test_multi_task_output_to_dict` - Converts output to dictionary
- `test_forward_ner_general` - Forward pass with NER capability returns logits
- `test_forward_sentiment` - Forward pass with sentiment returns logits
- `test_forward_emotions` - Forward pass with emotions returns multi-label logits
- `test_forward_safety_familyos` - Forward pass with safety returns 4 bands
- `test_forward_embedding` - Forward pass returns normalized embeddings
- `test_forward_with_labels` - Forward pass computes loss when labels provided
- `test_freeze_encoder` - Encoder weights frozen when flag set
- `test_shared_pooler_integration` - Shared pooler used by sequence heads
- `test_adapter_integration` - Task adapters applied to encoder output
- `test_pair_encoder_integration` - Pair encoder used for NLI/Relation
- `test_from_pretrained` - Load from HuggingFace checkpoint
- `test_save_pretrained` - Save model to disk
- `test_emotions_uses_hierarchical_head` - Emotions capability uses HierarchicalEmotionHead
- `test_safety_uses_enhanced_head` - Safety FamilyOS uses EnhancedSafetyHead

---

#### Issue 3.1.3: `models/heads.py`

**Tests:**

- `test_base_head_init` - BaseHead initializes with correct parameters
- `test_base_head_compute_loss_single_label` - Cross-entropy for single-label
- `test_base_head_compute_loss_multi_label` - BCE for multi-label
- `test_base_head_compute_loss_regression` - MSE for regression
- `test_base_head_asymmetric_loss` - ASL computes correctly for multi-label
- `test_base_head_focal_loss` - Focal loss reduces easy example weight
- `test_base_head_freeze_unfreeze` - Parameters freeze/unfreeze correctly
- `test_sequence_classification_head_init` - Initializes with dense + classifier
- `test_sequence_classification_head_pool_cls` - CLS pooling extracts first token
- `test_sequence_classification_head_pool_mean` - Mean pooling averages tokens
- `test_sequence_classification_head_pool_max` - Max pooling takes max values
- `test_sequence_classification_head_forward` - Forward returns logits
- `test_sequence_classification_head_with_labels` - Returns loss with labels
- `test_sequence_classification_head_external_pooler` - Uses external pooler when provided
- `test_token_classification_head_init` - Initializes with classifier
- `test_token_classification_head_forward` - Forward returns per-token logits
- `test_token_classification_head_ignore_index` - Loss ignores -100 labels
- `test_embedding_head_init` - Initializes with optional projection
- `test_embedding_head_normalize` - L2 normalizes output when flag set
- `test_embedding_head_pooling` - Uses specified pooling strategy
- `test_nli_head_init` - Initializes with 3 labels
- `test_nli_head_forward` - Returns 3-class logits
- `test_nli_head_pair_encoder` - Uses pair encoder for cross-attention
- `test_safety_head_init` - Initializes with 4 bands
- `test_safety_head_temperature_scaling` - Temperature calibration applied
- `test_enhanced_safety_head_subcategories` - Returns 12 subcategories
- `test_enhanced_safety_head_keyword_override` - CRISIS keywords trigger override
- `test_relation_head_init` - Initializes with 15 relations
- `test_relation_head_entity_pairs` - Handles entity pair representations
- `test_intent_head_init` - Initializes with 8 intents
- `test_intent_head_forward` - Returns intent logits
- `test_temporal_head_init` - Initializes with 13 BIO tags
- `test_temporal_head_forward` - Returns per-token temporal tags
- `test_hierarchical_emotion_head_44_emotions` - Handles 44 FamilyOS emotions
- `test_hierarchical_emotion_head_asl` - Uses asymmetric loss

---

#### Issue 3.1.4: `models/losses.py`

**Tests:**

- `test_focal_loss_init` - Initializes with alpha and gamma
- `test_focal_loss_forward` - Computes focal loss correctly
- `test_focal_loss_downweights_easy` - Easy examples have lower weight
- `test_focal_loss_per_class_alpha` - Per-class weights applied
- `test_focal_loss_ignore_index` - Ignores -100 labels
- `test_label_smoothing_ce_init` - Initializes with epsilon
- `test_label_smoothing_ce_forward` - Applies label smoothing
- `test_multiple_negatives_ranking_loss` - Contrastive loss for embeddings
- `test_cosine_similarity_loss` - Regression on similarity scores
- `test_triplet_loss` - Anchor-positive-negative loss
- `test_crf_loss` - CRF for sequence labeling
- `test_family_contrastive_loss` - Family-aware contrastive learning
- `test_multi_task_loss` - Weighted combination of task losses
- `test_uncertainty_weighted_loss` - Learns task weights automatically
- `test_fgm_adversarial` - FGM perturbation applied
- `test_pgd_adversarial` - PGD iterative perturbation
- `test_rdrop_loss` - R-Drop KL divergence regularization
- `test_embedding_mixup` - Mixup in embedding space

---

### Epic 3.2: Model Components

#### Issue 3.2.1: `models/poolers.py`

**Tests:**

- `test_base_pooler_abstract` - BasePooler is abstract
- `test_base_pooler_expand_mask` - Mask expanded for broadcasting
- `test_cls_pooler_init` - Initializes with optional dense layer
- `test_cls_pooler_forward` - Extracts CLS token
- `test_cls_pooler_with_dense` - Applies dense + tanh
- `test_mean_pooler_forward` - Computes masked mean
- `test_mean_pooler_ignores_padding` - Padding tokens excluded
- `test_max_pooler_forward` - Computes masked max
- `test_weighted_mean_pooler` - Attention-weighted mean
- `test_last_token_pooler` - Extracts last non-padding token
- `test_cls_mean_pooler` - Concatenates CLS and mean
- `test_attention_pooler` - Learns attention weights
- `test_get_pooler_factory` - Factory returns correct pooler type

---

#### Issue 3.2.2: `models/adapters.py`

**Tests:**

- `test_adapter_config_init` - AdapterConfig initializes correctly
- `test_adapter_config_validation` - Validates bottleneck_size > 0
- `test_get_activation` - Returns correct activation module
- `test_bottleneck_adapter_init` - Initializes down/up projections
- `test_bottleneck_adapter_forward` - Applies bottleneck transformation
- `test_bottleneck_adapter_residual` - Residual connection added
- `test_bottleneck_adapter_near_identity_init` - Small init keeps near identity
- `test_bottleneck_adapter_freeze_unfreeze` - Parameters freeze/unfreeze
- `test_bottleneck_adapter_num_parameters` - Reports parameter count
- `test_task_group_adapter_init` - Initializes adapters per group
- `test_task_group_adapter_forward` - Routes to correct adapter by task group
- `test_task_group_adapter_shared_down` - Shares down-projection when flag set
- `test_parallel_adapter_forward` - Adds to residual stream
- `test_lora_adapter_init` - Initializes with r and alpha
- `test_lora_adapter_forward` - Applies low-rank update
- `test_lora_adapter_merge` - Merges LoRA weights into base
- `test_adapted_linear_forward` - AdaptedLinear applies adapter
- `test_create_adapter_factory` - Factory creates correct adapter type

---

#### Issue 3.2.3: `models/pair_encoder.py`

**Tests:**

- `test_pair_encoder_config_init` - PairEncoderConfig initializes correctly
- `test_pair_encoder_config_validation` - Validates hidden_size % num_heads == 0
- `test_cross_attention_layer_init` - Initializes Q, K, V projections
- `test_cross_attention_layer_forward` - Computes cross-attention
- `test_cross_attention_layer_mask` - Masks out padding in key-value
- `test_cross_attention_layer_residual` - Adds residual connection
- `test_feedforward_init` - Initializes with GELU activation
- `test_feedforward_forward` - Two linear transformations with activation
- `test_bidirectional_cross_attention_block` - Both directions attended
- `test_attention_pooling` - Learns attention weights for pooling
- `test_cross_attention_pair_encoder_init` - Initializes with layers
- `test_cross_attention_pair_encoder_forward` - Returns pair representation
- `test_cross_attention_pair_encoder_pooling_cls` - Uses CLS pooling
- `test_cross_attention_pair_encoder_pooling_mean` - Uses mean pooling
- `test_cross_attention_pair_encoder_pooling_attention` - Uses attention pooling
- `test_concat_pair_encoder_forward` - Concatenates pooled representations
- `test_create_pair_encoder_factory` - Factory creates correct encoder type

---

### Milestone 3 Integration Points

| From | To | Connection |
|------|-----|------------|
| `models/__init__.py` | `models/modernbert_multitask.py` | Exports main model class |
| `models/__init__.py` | `models/heads.py` | Exports all head classes |
| `models/__init__.py` | `models/poolers.py` | Exports pooler classes |
| `models/__init__.py` | `models/adapters.py` | Exports adapter classes |
| `models/__init__.py` | `models/pair_encoder.py` | Exports pair encoder classes |
| `models/modernbert_multitask.py` | `models/heads.py` | Creates heads based on capability |
| `models/modernbert_multitask.py` | `models/poolers.py` | Uses shared poolers |
| `models/modernbert_multitask.py` | `models/adapters.py` | Uses task-group adapters |
| `models/modernbert_multitask.py` | `models/pair_encoder.py` | Uses pair encoder for NLI/Relation |
| `models/heads.py` | `models/losses.py` | Heads use specialized loss functions |
| `models/heads.py` | `models/poolers.py` | Sequence heads use poolers |
| `models/modernbert_multitask.py` | `data/labels.py` | Gets num_labels and problem_type |

**Future Integration Tests (Milestone 3):**

- `test_model_with_all_heads` - Model with all 12 capabilities initializes correctly
- `test_model_forward_all_capabilities` - Forward pass works for each capability
- `test_model_with_adapters_and_poolers` - Epic 5.0 components work together
- `test_model_save_load_roundtrip` - Save and load preserves all weights
- `test_head_loss_backprop` - Loss backpropagates through correct head
- `test_encoder_shared_across_heads` - Encoder weights shared by all heads

---

## Milestone 4: Training

### Epic 4.1: Trainer Core

#### Issue 4.1.1: `trainers/__init__.py`

**Tests:**

- `test_task_sampler_exports` - All sampler classes exported
- `test_collator_exports` - All collator classes exported
- `test_ema_model_exported` - EMAModel exported
- `test_optimizer_functions_exported` - create_optimizer_with_head_lr exported
- `test_uncertainty_weighting_exported` - UncertaintyWeighting exported

---

#### Issue 4.1.2: `trainers/multitask_trainer.py`

**Tests:**

- `test_multi_task_data_loader_init` - Initializes with dataloaders and sampler
- `test_multi_task_data_loader_iter` - Yields batches with task field
- `test_multi_task_data_loader_len` - Returns total batch count
- `test_multi_task_data_loader_task_cycling` - Resets iterator on exhaustion
- `test_multi_task_iterable_dataset_init` - Initializes with datasets and sampler
- `test_multi_task_iterable_dataset_iter` - Yields samples with task field
- `test_multi_task_training_args_init` - Extended args with sampling_strategy
- `test_multi_task_training_args_rdrop` - R-Drop configuration
- `test_multi_task_training_args_adversarial` - Adversarial training config
- `test_multi_task_training_args_mixup` - Mixup configuration
- `test_multi_task_trainer_init` - Initializes with train/eval datasets
- `test_multi_task_trainer_create_sampler` - Creates correct sampler type
- `test_multi_task_trainer_get_train_dataloader` - Returns MultiTaskDataLoader
- `test_multi_task_trainer_compute_loss` - Computes task-specific loss
- `test_multi_task_trainer_task_weights` - Task weights applied to loss
- `test_multi_task_trainer_uncertainty_weighting` - Learned weights used
- `test_multi_task_trainer_evaluate` - Evaluates on all eval datasets
- `test_multi_task_trainer_per_task_metrics` - Reports per-task metrics
- `test_multi_task_trainer_rdrop_training` - R-Drop regularization applied
- `test_multi_task_trainer_adversarial_training` - FGM/PGD applied
- `test_multi_task_trainer_mixup_augmentation` - Mixup applied

---

#### Issue 4.1.3: `trainers/collators.py`

**Tests:**

- `test_base_collator_pad_token_id` - Returns correct pad token ID
- `test_base_collator_pad_sequence` - Pads to longest in batch
- `test_base_collator_pad_to_max_length` - Pads to fixed max_length
- `test_base_collator_pad_to_multiple` - Pads to multiple of value
- `test_sequence_classification_collator` - Pads input_ids and attention_mask
- `test_sequence_classification_collator_labels` - Labels tensor is long dtype
- `test_sequence_classification_collator_task` - Preserves task field
- `test_multi_label_collator_labels` - Labels tensor is float dtype
- `test_multi_label_collator_multi_hot` - Multi-hot encoding preserved
- `test_token_classification_collator` - Pads token labels with -100
- `test_token_classification_collator_alignment` - Labels aligned with tokens
- `test_nli_collator_premise_hypothesis` - Handles premise-hypothesis pairs
- `test_nli_collator_token_type_ids` - Token type IDs correct
- `test_embedding_collator_pairs` - Handles positive/negative pairs
- `test_embedding_collator_hard_negatives` - Includes hard negatives
- `test_relation_collator_entity_spans` - Handles entity span markers
- `test_multi_task_collator_routing` - Routes to correct collator by task
- `test_multi_task_collator_fallback` - Falls back to base collator

---

#### Issue 4.1.4: `trainers/optimizer.py`

**Tests:**

- `test_create_param_groups` - Creates groups for encoder and heads
- `test_create_param_groups_encoder_lr` - Encoder has lower LR
- `test_create_param_groups_head_lr` - Heads have higher LR
- `test_create_param_groups_token_head_lr` - Token heads have medium LR
- `test_create_param_groups_weight_decay` - Weight decay applied to correct params
- `test_create_param_groups_no_decay` - Bias and LayerNorm have no decay
- `test_create_param_groups_empty_filtered` - Empty groups filtered out
- `test_create_optimizer_with_head_lr` - Returns AdamW with param groups
- `test_optimizer_betas` - Betas configured correctly
- `test_optimizer_eps` - Epsilon configured correctly
- `test_layer_wise_lr_decay` - Optional layer-wise LR decay

---

### Epic 4.2: Training Strategies

#### Issue 4.2.1: `trainers/task_sampler.py`

**Tests:**

- `test_task_sampler_abstract` - TaskSampler is abstract
- `test_task_sampler_init` - Initializes with task names and weights
- `test_task_sampler_probabilities` - Probabilities property works
- `test_task_sampler_step_count` - Tracks sample count
- `test_task_sampler_reset` - Resets state and RNG
- `test_task_sampler_update_weights` - Updates weights and recomputes probs
- `test_task_sampler_state_checkpoint` - get_state/load_state work
- `test_proportional_sampler_init` - Initializes with task sizes
- `test_proportional_sampler_probabilities` - P(task) ∝ size × weight
- `test_proportional_sampler_sample` - Returns task name
- `test_temperature_sampler_init` - Initializes with temperature
- `test_temperature_sampler_high_temp` - High temp = more uniform
- `test_temperature_sampler_low_temp` - Low temp = more peaked
- `test_uniform_sampler_probabilities` - Equal probability for all tasks
- `test_sequential_sampler_order` - Round-robin through tasks
- `test_curriculum_sampler_stages` - Respects curriculum stages
- `test_create_sampler_factory` - Factory creates correct sampler type

---

#### Issue 4.2.2: `trainers/task_weighting.py`

**Tests:**

- `test_uncertainty_weighting_init` - Initializes with num_tasks
- `test_uncertainty_weighting_forward` - Computes weighted sum
- `test_uncertainty_weighting_learns` - Log vars are learnable
- `test_uncertainty_weighting_get_weights` - Returns current weights
- `test_uncertainty_weighting_get_log_vars` - Returns log variance values
- `test_uncertainty_weighting_none_loss` - Handles None losses
- `test_static_weighting_init` - Initializes with fixed weights
- `test_static_weighting_forward` - Applies fixed weights
- `test_dynamic_temperature_weighting` - Temperature-scaled weights
- `test_dynamic_temperature_learnable` - Temperature is learnable

---

#### Issue 4.2.3: `trainers/curriculum.py`

**Tests:**

- `test_task_difficulty_enum` - TaskDifficulty has EASY, MEDIUM, HARD, VERY_HARD
- `test_default_task_difficulty` - All 12 tasks have difficulty assigned
- `test_curriculum_stage_init` - CurriculumStage initializes correctly
- `test_curriculum_stage_get_task_list` - Expands "all" to all tasks
- `test_curriculum_config_init` - CurriculumConfig with defaults
- `test_curriculum_scheduler_init` - Initializes with stages
- `test_curriculum_scheduler_get_active_tasks` - Returns correct tasks for epoch
- `test_curriculum_scheduler_stage_progression` - Advances through stages
- `test_curriculum_scheduler_epoch_mapping` - Maps epochs to stages
- `test_curriculum_auto_difficulty_order` - Orders tasks by difficulty
- `test_curriculum_warmup` - Warmup epochs with subset of tasks
- `test_curriculum_dynamic_progression` - Loss-based stage advancement

---

#### Issue 4.2.4: `trainers/ema.py`

**Tests:**

- `test_ema_model_init` - Initializes with decay
- `test_ema_model_decay_validation` - Decay must be in [0, 1]
- `test_ema_model_shadow_init` - Shadow weights initialized from model
- `test_ema_model_update` - Shadow updated with EMA formula
- `test_ema_model_apply_shadow` - Applies EMA weights to model
- `test_ema_model_restore` - Restores original weights
- `test_ema_model_state_dict` - Returns shadow weights for saving
- `test_ema_model_load_state_dict` - Loads shadow weights
- `test_ema_model_copy_to` - Copies EMA weights permanently
- `test_ema_callback_on_step_end` - Updates EMA after each step
- `test_ema_callback_on_evaluate` - Applies shadow before eval
- `test_ema_callback_on_evaluate_end` - Restores weights after eval

---

#### Issue 4.2.5: `trainers/callbacks.py`

**Tests:**

- `test_task_metrics_state_init` - TaskMetricsState initializes
- `test_task_metrics_callback_init` - Initializes with log_every
- `test_task_metrics_callback_on_train_begin` - Resets state at start
- `test_task_metrics_callback_on_step_end` - Records per-task loss
- `test_task_metrics_callback_on_log` - Appends task metrics to logs
- `test_task_metrics_callback_reset_on_log` - Resets averages after logging
- `test_gradient_monitor_callback` - Monitors gradient norms
- `test_gradient_monitor_per_head` - Reports gradients per head
- `test_early_stopping_callback_init` - Initializes with patience
- `test_early_stopping_callback_check` - Stops when no improvement
- `test_early_stopping_callback_metric` - Uses specified metric
- `test_model_checkpoint_callback` - Saves model at intervals
- `test_model_checkpoint_save_best` - Saves best model by metric
- `test_safety_monitoring_callback` - Monitors CRISIS recall during training

---

### Milestone 4 Integration Points

| From | To | Connection |
|------|-----|------------|
| `trainers/__init__.py` | `trainers/multitask_trainer.py` | Exports MultiTaskTrainer |
| `trainers/__init__.py` | `trainers/collators.py` | Exports collator classes |
| `trainers/__init__.py` | `trainers/task_sampler.py` | Exports sampler classes |
| `trainers/__init__.py` | `trainers/ema.py` | Exports EMAModel |
| `trainers/__init__.py` | `trainers/optimizer.py` | Exports optimizer functions |
| `trainers/__init__.py` | `trainers/task_weighting.py` | Exports weighting classes |
| `trainers/multitask_trainer.py` | `trainers/collators.py` | Uses MultiTaskCollator |
| `trainers/multitask_trainer.py` | `trainers/task_sampler.py` | Uses task samplers |
| `trainers/multitask_trainer.py` | `trainers/task_weighting.py` | Uses uncertainty weighting |
| `trainers/multitask_trainer.py` | `trainers/callbacks.py` | Registers training callbacks |
| `trainers/multitask_trainer.py` | `trainers/ema.py` | Integrates EMA via callback |
| `trainers/multitask_trainer.py` | `trainers/optimizer.py` | Uses head-wise LR optimizer |
| `trainers/multitask_trainer.py` | `trainers/curriculum.py` | Uses curriculum scheduler |
| `trainers/collators.py` | `data/labels.py` | Gets problem_type for routing |
| `trainers/task_sampler.py` | `trainers/curriculum.py` | CurriculumSampler uses scheduler |

**Future Integration Tests (Milestone 4):**

- `test_trainer_with_model` - Trainer works with ModernBertMultiTaskModel
- `test_trainer_full_train_step` - Complete training step executes
- `test_trainer_multi_task_evaluation` - Evaluates all tasks correctly
- `test_trainer_with_ema` - EMA updates and applies during training
- `test_trainer_with_curriculum` - Curriculum stages respected
- `test_trainer_checkpoint_resume` - Training resumes from checkpoint
- `test_trainer_with_uncertainty_weighting` - Learned weights converge
- `test_full_training_pipeline` - Data → Model → Trainer → Checkpoint

---

### Epic 4.1: Trainer Core

- Issue 4.1.1: `trainers/__init__.py`
- Issue 4.1.2: `trainers/multitask_trainer.py`
- Issue 4.1.3: `trainers/collators.py`
- Issue 4.1.4: `trainers/optimizer.py`

### Epic 4.2: Training Strategies

- Issue 4.2.1: `trainers/task_sampler.py`
- Issue 4.2.2: `trainers/task_weighting.py`
- Issue 4.2.3: `trainers/curriculum.py`
- Issue 4.2.4: `trainers/ema.py`
- Issue 4.2.5: `trainers/callbacks.py`

---

## Milestone 5: Evaluation

### Epic 5.1: Evaluation Core

#### Issue 5.1.1: `evaluation/__init__.py`

**Tests:**

- `test_evaluation_exports` - All public APIs exported correctly
- `test_metrics_import` - Metrics functions importable
- `test_evaluator_import` - Evaluator class importable
- `test_benchmarks_import` - Benchmark classes importable

---

#### Issue 5.1.2: `evaluation/metrics.py`

**Tests:**

- `test_task_problem_types_mapping` - All 12 tasks mapped to problem types
- `test_task_primary_metrics_mapping` - Primary metrics defined for all tasks
- `test_compute_ner_metrics_entity_f1` - Entity-level F1 computation
- `test_compute_ner_metrics_precision_recall` - Per-entity precision/recall
- `test_compute_ner_metrics_with_seqeval` - Uses seqeval when available
- `test_compute_ner_metrics_fallback` - Falls back when seqeval missing
- `test_compute_ner_metrics_ignore_index` - Ignores -100 labels
- `test_compute_ner_metrics_per_entity_type` - Per-entity-type breakdown
- `test_compute_classification_metrics_accuracy` - Accuracy computed correctly
- `test_compute_classification_metrics_f1` - Weighted and macro F1
- `test_compute_classification_metrics_from_logits` - Handles 2D logits input
- `test_compute_classification_metrics_empty` - Handles empty predictions
- `test_compute_multilabel_metrics_micro_f1` - Multi-label micro F1
- `test_compute_multilabel_metrics_macro_f1` - Multi-label macro F1
- `test_compute_multilabel_metrics_hamming_loss` - Hamming loss computed
- `test_compute_multilabel_metrics_threshold` - Custom threshold applied
- `test_compute_multilabel_metrics_from_logits` - Sigmoid applied to logits
- `test_compute_embedding_metrics_spearman` - Spearman correlation
- `test_compute_embedding_metrics_pearson` - Pearson correlation
- `test_compute_embedding_metrics_handles_nan` - NaN values filtered
- `test_compute_nli_metrics_accuracy` - NLI accuracy
- `test_compute_nli_metrics_per_class_f1` - Per-class F1 (entail/neutral/contra)
- `test_compute_relation_metrics_ignore_no_relation` - Excludes no_relation from F1
- `test_compute_relation_metrics_micro_f1` - Micro F1 on relations
- `test_compute_relation_metrics_per_relation` - Per-relation breakdown
- `test_compute_intent_metrics_accuracy` - Intent accuracy
- `test_compute_intent_metrics_ece` - Expected Calibration Error
- `test_compute_ece_binned` - ECE binning logic
- `test_compute_safety_metrics` - Safety band metrics
- `test_compute_ingress_metrics` - Ingress classification metrics
- `test_compute_temporal_metrics` - Temporal NER metrics
- `test_compute_ner_family_metrics` - Family NER metrics
- `test_get_task_primary_metric` - Returns correct primary metric
- `test_aggregate_metrics_average` - Average across tasks
- `test_aggregate_metrics_weighted` - Weighted average

---

#### Issue 5.1.3: `evaluation/evaluator.py`

**Tests:**

- `test_task_results_init` - TaskResults dataclass initialization
- `test_task_results_primary_metric` - Extracts primary metric
- `test_task_results_to_dict` - Serialization to dict
- `test_eval_results_init` - EvalResults dataclass initialization
- `test_eval_results_summary` - Human-readable summary generation
- `test_eval_results_to_dict` - Serialization to dict
- `test_eval_results_save_json` - Save to JSON file
- `test_eval_results_save_markdown` - Save to markdown format
- `test_eval_results_to_markdown` - Markdown table generation
- `test_evaluator_init` - Evaluator initializes with model/tokenizer
- `test_evaluator_auto_device` - Auto-detects CUDA/CPU
- `test_evaluator_capabilities_from_model` - Gets capabilities from model
- `test_evaluator_get_label_list` - Retrieves label list for task
- `test_evaluator_prepare_batch` - Moves batch to device
- `test_evaluator_extract_labels` - Extracts labels from batch
- `test_evaluator_compute_predictions_token` - Token classification predictions
- `test_evaluator_compute_predictions_sequence` - Sequence classification predictions
- `test_evaluator_compute_predictions_multilabel` - Multi-label predictions
- `test_evaluator_compute_predictions_regression` - Regression predictions
- `test_evaluator_compute_task_metrics` - Routes to correct metric function
- `test_evaluator_evaluate_task` - Single task evaluation
- `test_evaluator_evaluate_task_with_progress` - Shows progress bar
- `test_evaluator_evaluate_all` - Multi-task evaluation
- `test_evaluator_aggregate_results` - Aggregates across tasks
- `test_evaluator_handles_errors` - Error handling in metric computation

---

#### Issue 5.1.4: `evaluation/benchmarks.py`

**Tests:**

- `test_latency_results_init` - LatencyResults dataclass initialization
- `test_latency_results_to_dict` - Serialization to dict
- `test_latency_results_summary` - Human-readable summary
- `test_latency_benchmark_init` - LatencyBenchmark initializes
- `test_latency_benchmark_auto_device` - Auto-detects device
- `test_latency_benchmark_get_memory_mb` - GPU memory tracking
- `test_latency_benchmark_reset_memory_stats` - Memory stats reset
- `test_latency_benchmark_tokenize_batch` - Tokenizes and moves to device
- `test_latency_benchmark_run_inference` - Single inference pass
- `test_latency_benchmark_run` - Full benchmark with warmup
- `test_latency_benchmark_warmup` - Warmup iterations not measured
- `test_latency_benchmark_percentiles` - P50/P95/P99 computed
- `test_latency_benchmark_throughput` - Samples per second
- `test_latency_benchmark_cuda_sync` - CUDA sync for accurate timing
- `test_benchmark_results_init` - BenchmarkResults initialization
- `test_benchmark_suite_init` - BenchmarkSuite initialization
- `test_base_benchmark_abstract` - BaseBenchmark is abstract
- `test_glue_benchmark_datasets` - GLUE dataset loading
- `test_ner_benchmark_conll` - CoNLL-2003 benchmark
- `test_embedding_benchmark_sts` - STS-B benchmark
- `test_familyos_benchmark` - FamilyOS-specific benchmarks
- `test_baseline_comparison` - Compare with baselines
- `test_benchmark_result_tracker` - Track results over time

---

### Epic 5.2: Safety Evaluation

#### Issue 5.2.1: `evaluation/safety_eval.py`

**Tests:**

- `test_safety_bands_constants` - GREEN/AMBER/RED/CRISIS indices
- `test_quality_targets_defaults` - Default quality targets defined
- `test_safety_scenarios_defined` - All scenarios with expected bands
- `test_calibration_results_init` - CalibrationResults dataclass
- `test_calibration_results_to_dict` - Serialization
- `test_calibration_results_summary` - Human-readable summary
- `test_safety_metrics_init` - SafetyMetrics dataclass
- `test_safety_metrics_to_dict` - Serialization
- `test_threshold_metrics_init` - ThresholdMetrics dataclass
- `test_threshold_results_init` - ThresholdResults dataclass
- `test_threshold_results_summary` - Threshold report
- `test_safety_eval_results_init` - SafetyEvalResults dataclass
- `test_safety_eval_results_summary` - Full summary with quality gates
- `test_safety_eval_results_save` - Save to JSON
- `test_safety_evaluator_init` - SafetyEvaluator initialization
- `test_safety_evaluator_auto_device` - Auto device detection
- `test_safety_evaluator_label_schema` - Correct schema loaded
- `test_safety_evaluator_evaluate` - Full evaluation pipeline
- `test_safety_evaluator_run_inference` - Batch inference
- `test_safety_evaluator_compute_safety_metrics` - Core metrics
- `test_safety_evaluator_confusion_matrix` - Confusion matrix computed
- `test_safety_evaluator_evaluate_scenarios` - Per-scenario evaluation
- `test_safety_evaluator_analyze_thresholds` - Threshold analysis
- `test_safety_evaluator_check_quality_gates` - Quality gate checks
- `test_safety_evaluator_compare_baseline` - Baseline comparison
- `test_crisis_recall_calculation` - CRISIS recall computed correctly
- `test_red_recall_calculation` - RED recall computed
- `test_green_fpr_calculation` - GREEN false positive rate
- `test_calibration_ece_computation` - Expected Calibration Error
- `test_calibration_mce_computation` - Maximum Calibration Error
- `test_calibration_reliability_diagram` - Reliability diagram data
- `test_temperature_scaling` - Optimal temperature finding
- `test_threshold_for_target_recall` - Find threshold for recall target
- `test_threshold_for_target_fnr` - Find threshold for FNR target
- `test_multi_label_safety_evaluation` - safety_generic evaluation
- `test_quality_gate_crisis_recall` - CRISIS recall ≥ 98%
- `test_quality_gate_red_recall` - RED recall ≥ 90%
- `test_quality_gate_macro_f1` - Macro F1 ≥ 80%

---

#### Issue 5.2.2: `evaluation/cultural_robustness.py`

**Tests:**

- `test_indian_venting_patterns_defined` - Patterns list populated
- `test_filipino_venting_patterns_defined` - Filipino patterns
- `test_adversarial_safe_examples_defined` - Edge cases defined
- `test_true_crisis_examples_defined` - True crisis examples
- `test_cultural_robustness_result_init` - Result dataclass
- `test_cultural_robustness_result_passed` - Pass/fail logic
- `test_cultural_robustness_result_to_dict` - Serialization
- `test_cultural_robustness_result_summary` - Human-readable summary
- `test_crisis_recall_result_init` - CrisisRecallResult dataclass
- `test_cultural_robustness_evaluator_init` - Evaluator initialization
- `test_cultural_robustness_evaluator_evaluate` - Full evaluation
- `test_evaluate_indian_hyperbole` - Indian expressions not flagged as CRISIS
- `test_evaluate_figurative_speech` - Metaphors not flagged
- `test_evaluate_true_crisis_detected` - True crises detected
- `test_false_positive_rate_calculation` - FPR computed correctly
- `test_false_positive_rate_target` - FPR ≤ 5% target
- `test_per_pattern_results` - Results per expression pattern
- `test_failed_examples_tracking` - Failed examples recorded

---

#### Issue 5.2.3: `evaluation/temporal_safety.py`

**Tests:**

- `test_safety_band_enum` - SafetyBand enum ordering
- `test_safety_band_from_string` - String to enum conversion
- `test_isolation_keywords_defined` - Isolation keywords set
- `test_escalation_keywords_defined` - Escalation keywords set
- `test_safety_signal_init` - SafetySignal dataclass
- `test_safety_signal_create_factory` - Factory method
- `test_safety_signal_band_name` - Band as string property
- `test_safety_signal_severity` - Numeric severity property
- `test_safety_escalation_init` - SafetyEscalation dataclass
- `test_temporal_safety_monitor_init` - Monitor initialization
- `test_monitor_add_signal` - Add signal to window
- `test_monitor_window_expiry` - Signals expire after window
- `test_monitor_amber_accumulation` - 3+ AMBER → RED escalation
- `test_monitor_isolation_escalation` - RED + isolation → CRISIS
- `test_monitor_rapid_escalation` - Quick escalation flagged
- `test_monitor_get_signals_in_window` - Retrieves recent signals
- `test_monitor_get_escalation_history` - Escalation log
- `test_monitor_clear_signals` - Clears signal history
- `test_multi_user_tracking` - Separate tracking per user_id
- `test_escalation_reason_recorded` - Escalation reason captured
- `test_escalation_callback` - Custom escalation handler

---

### Epic 5.3: Forgetting Evaluation

#### Issue 5.3.1: `evaluation/forgetting_eval.py`

**Tests:**

- `test_forgetting_thresholds_defined` - Default thresholds for all tasks
- `test_benchmark_datasets_mapping` - Task to dataset mapping
- `test_forgetting_result_init` - ForgettingResult dataclass
- `test_forgetting_result_repr` - String representation
- `test_forgetting_result_to_dict` - Serialization
- `test_forgetting_report_init` - ForgettingReport dataclass
- `test_forgetting_report_getitem` - Dict-style access by task
- `test_forgetting_report_summary` - Human-readable summary
- `test_forgetting_report_save` - Save to JSON
- `test_forgetting_evaluator_init` - Evaluator initialization
- `test_forgetting_evaluator_load_checkpoints` - Loads Stage A and B
- `test_forgetting_evaluator_evaluate_task` - Single task evaluation
- `test_forgetting_evaluator_evaluate_all` - All tasks evaluation
- `test_forgetting_evaluator_compute_drop` - Drop calculation
- `test_forgetting_evaluator_check_gates` - Gate pass/fail
- `test_forgetting_evaluator_recommendations` - Generates recommendations
- `test_ner_general_forgetting_gate` - ≤ 2% F1 drop
- `test_sentiment_forgetting_gate` - ≤ 2% accuracy drop
- `test_nli_forgetting_gate` - ≤ 2% accuracy drop
- `test_emotions_forgetting_gate` - ≤ 3% macro F1 drop
- `test_recommendation_reduce_lora_r` - LoRA r recommendation
- `test_recommendation_increase_replay` - Replay ratio recommendation
- `test_recommendation_freeze_layers` - Layer freezing recommendation

---

### Milestone 5 Integration Points

| From | To | Connection |
|------|-----|------------|
| `evaluation/__init__.py` | `evaluation/metrics.py` | Exports metric functions |
| `evaluation/__init__.py` | `evaluation/evaluator.py` | Exports Evaluator class |
| `evaluation/__init__.py` | `evaluation/benchmarks.py` | Exports benchmark classes |
| `evaluation/__init__.py` | `evaluation/safety_eval.py` | Exports SafetyEvaluator |
| `evaluation/evaluator.py` | `evaluation/metrics.py` | Uses compute_*_metrics functions |
| `evaluation/evaluator.py` | `trainers/collators.py` | Uses MultiTaskCollator |
| `evaluation/evaluator.py` | `data/labels.py` | Gets label schemas |
| `evaluation/benchmarks.py` | `models/modernbert_multitask.py` | Benchmarks model inference |
| `evaluation/safety_eval.py` | `evaluation/metrics.py` | Uses safety metric helpers |
| `evaluation/safety_eval.py` | `data/labels.py` | Uses SAFETY_FAMILYOS_LABELS |
| `evaluation/cultural_robustness.py` | `evaluation/safety_eval.py` | Extends safety evaluation |
| `evaluation/temporal_safety.py` | `evaluation/safety_eval.py` | Temporal safety tracking |
| `evaluation/forgetting_eval.py` | `evaluation/evaluator.py` | Uses Evaluator for benchmarks |
| `evaluation/forgetting_eval.py` | `evaluation/metrics.py` | Uses task metrics |

**Future Integration Tests (Milestone 5):**

- `test_full_evaluation_pipeline` - Load model → Evaluate all tasks → Generate report
- `test_safety_evaluation_with_calibration` - Safety eval + calibration
- `test_cultural_robustness_with_safety` - Cultural + safety combined
- `test_forgetting_before_after_stage_b` - Compare Stage A vs B
- `test_benchmark_suite_full_run` - All benchmarks executed
- `test_evaluation_report_export` - JSON/Markdown/HTML export

---

## Milestone 6: Inference & Runtime

### Epic 6.1: Inference

#### Issue 6.1.1: `inference/__init__.py`

**Tests:**

- `test_inference_exports` - All public APIs exported
- `test_entity_import` - Entity dataclass importable
- `test_relation_import` - Relation dataclass importable
- `test_unified_nlp_output_import` - UnifiedNLPOutput importable
- `test_sys_nlp_infer_import` - sys_nlp_infer function importable
- `test_get_unified_model_import` - get_unified_model importable

---

#### Issue 6.1.2: `inference/unified_output.py`

**Tests:**

- `test_entity_init` - Entity dataclass initialization
- `test_entity_to_dict` - Entity serialization
- `test_entity_with_token_offsets` - Token start/end tracked
- `test_relation_init` - Relation dataclass initialization
- `test_relation_to_dict` - Relation serialization
- `test_relation_with_spans` - Subject/object spans tracked
- `test_unified_nlp_output_init` - UnifiedNLPOutput initialization
- `test_unified_nlp_output_all_fields` - All 12 capability fields
- `test_unified_nlp_output_to_dict` - Full serialization
- `test_unified_nlp_output_from_dict` - Deserialization roundtrip
- `test_unified_nlp_output_partial` - None for unrequested capabilities
- `test_get_unified_model_default` - Loads default model
- `test_get_unified_model_custom_path` - Loads from custom path
- `test_get_unified_model_with_device` - Respects device parameter
- `test_get_unified_model_with_capabilities` - Subset of capabilities
- `test_get_unified_model_caching` - Singleton pattern works
- `test_clear_model_cache` - Cache cleared correctly
- `test_extract_entities_from_logits_bio` - BIO tag decoding
- `test_extract_entities_from_logits_spans` - Span aggregation
- `test_extract_entities_from_logits_confidence` - Confidence scores
- `test_extract_entities_from_logits_char_offsets` - Character offsets
- `test_process_sequence_classification_single` - Single-label processing
- `test_process_sequence_classification_multi` - Multi-label processing
- `test_process_sequence_classification_threshold` - Custom threshold
- `test_compute_valence` - Sentiment to valence mapping
- `test_compute_safety_score` - Safety band to score mapping
- `test_sys_nlp_infer_single_text` - Single text inference
- `test_sys_nlp_infer_batch` - Batch inference
- `test_sys_nlp_infer_all_capabilities` - All 12 capabilities
- `test_sys_nlp_infer_subset_capabilities` - Selected capabilities only
- `test_sys_nlp_infer_ner_general` - NER general extraction
- `test_sys_nlp_infer_ner_family` - Family NER extraction
- `test_sys_nlp_infer_temporal` - Temporal expression extraction
- `test_sys_nlp_infer_sentiment` - Sentiment classification
- `test_sys_nlp_infer_emotions` - Emotion detection (multi-label)
- `test_sys_nlp_infer_safety_generic` - Generic safety (multi-label)
- `test_sys_nlp_infer_safety_familyos` - FamilyOS safety bands
- `test_sys_nlp_infer_ingress` - Ingress classification
- `test_sys_nlp_infer_intent` - Intent classification
- `test_sys_nlp_infer_nli` - NLI with premise-hypothesis pairs
- `test_sys_nlp_infer_relation` - Relation extraction
- `test_sys_nlp_infer_embedding` - Embedding generation
- `test_sys_nlp_infer_with_pairs` - NLI pair input
- `test_sys_nlp_infer_with_entity_pairs` - Relation entity pairs
- `test_sys_nlp_infer_custom_model` - Pre-loaded model
- `test_sys_nlp_infer_custom_tokenizer` - Pre-loaded tokenizer
- `test_sys_nlp_infer_batch_size` - Respects batch_size
- `test_sys_nlp_infer_max_length` - Respects max_length
- `test_sys_nlp_infer_processing_time` - Tracks inference time
- `test_sys_nlp_infer_invalid_capability` - Raises on invalid capability

---

### Epic 6.2: K0 Runtime

#### Issue 6.2.1: `k0/__init__.py`

**Tests:**

- `test_k0_exports` - All public APIs exported
- `test_capability_enum_import` - Capability enum importable
- `test_model_info_import` - ModelInfo dataclass importable
- `test_head_info_import` - HeadInfo dataclass importable
- `test_registry_imports` - MODEL_REGISTRY, HEAD_REGISTRY importable

---

#### Issue 6.2.2: `k0/runtime/__init__.py`

**Tests:**

- `test_runtime_exports` - All runtime APIs exported
- `test_resolve_capability_import` - resolve_capability importable
- `test_get_model_info_import` - get_model_info importable
- `test_get_head_info_import` - get_head_info importable
- `test_get_unified_model_import` - get_unified_model importable
- `test_get_tokenizer_import` - get_tokenizer importable
- `test_migration_helpers_import` - migrate_legacy_model importable

---

#### Issue 6.2.3: `k0/runtime/model_registry.py`

**Tests:**

- `test_capability_enum_values` - All 12 capabilities defined
- `test_capability_enum_strings` - String values match expected
- `test_capability_aliases_defined` - Common aliases mapped
- `test_capability_alias_ner` - "ner" → NER_GENERAL
- `test_capability_alias_safety` - "safety" → SAFETY_FAMILYOS
- `test_capability_alias_embedding` - "embeddings" → EMBEDDING
- `test_model_info_init` - ModelInfo dataclass initialization
- `test_model_info_post_init` - Default head mappings set
- `test_model_info_all_capabilities` - All 12 capabilities mapped
- `test_head_info_init` - HeadInfo dataclass initialization
- `test_head_info_output_types` - Valid output types
- `test_model_registry_populated` - familyos_unified_v2 registered
- `test_model_registry_capabilities` - All capabilities supported
- `test_legacy_model_mapping` - Legacy models mapped to unified
- `test_head_registry_populated` - All heads registered
- `test_head_registry_label_counts` - Correct num_labels per head
- `test_resolve_capability_string` - Resolves string capability
- `test_resolve_capability_enum` - Resolves Capability enum
- `test_resolve_capability_alias` - Resolves via alias
- `test_resolve_capability_returns_tuple` - Returns (model, head)
- `test_resolve_capability_unknown` - Raises on unknown capability
- `test_get_model_info_valid` - Returns ModelInfo
- `test_get_model_info_legacy_warning` - Warns on legacy model
- `test_get_model_info_unknown` - Raises KeyError
- `test_get_head_info_valid` - Returns HeadInfo
- `test_get_head_info_unknown` - Raises KeyError
- `test_list_capabilities` - Returns all capability values
- `test_list_models` - Returns registered model names
- `test_list_heads` - Returns registered head names
- `test_get_unified_model_loads` - Model loads successfully
- `test_get_unified_model_device` - Respects device parameter
- `test_get_unified_model_cache` - Caches model instance
- `test_get_tokenizer_loads` - Tokenizer loads successfully
- `test_get_tokenizer_cache` - Caches tokenizer instance
- `test_clear_cache` - Clears model and tokenizer cache
- `test_migrate_legacy_model` - Returns unified model name
- `test_migrate_legacy_model_unknown` - Raises on unknown legacy
- `test_get_capability_for_module_m02` - M02 → NER_FAMILY
- `test_get_capability_for_module_m04` - M04 → EMOTIONS
- `test_get_capability_for_module_m10` - M10 → INGRESS
- `test_get_capability_for_module_p08` - P08 → EMBEDDING
- `test_get_capability_for_module_unknown` - Raises on unknown
- `test_register_model` - Adds new model to registry
- `test_register_model_overwrite` - Warns on overwrite
- `test_register_head` - Adds new head to registry
- `test_register_capability_alias` - Adds new alias

---

### Milestone 6 Integration Points

| From | To | Connection |
|------|-----|------------|
| `inference/__init__.py` | `inference/unified_output.py` | Exports all inference APIs |
| `inference/unified_output.py` | `models/modernbert_multitask.py` | Uses ModernBertMultiTaskModel |
| `inference/unified_output.py` | `data/labels.py` | Uses all label schemas |
| `k0/__init__.py` | `k0/runtime/__init__.py` | Re-exports runtime APIs |
| `k0/runtime/__init__.py` | `k0/runtime/model_registry.py` | Exports registry functions |
| `k0/runtime/model_registry.py` | `models/modernbert_multitask.py` | Loads model |
| `k0/runtime/model_registry.py` | `transformers.AutoTokenizer` | Loads tokenizer |
| `inference/unified_output.py` | `k0/runtime/model_registry.py` | Uses Capability enum |

**Future Integration Tests (Milestone 6):**

- `test_sys_nlp_infer_e2e` - Full end-to-end inference
- `test_k0_module_migration` - Legacy module to unified model
- `test_capability_resolution_chain` - Alias → Capability → Model → Head
- `test_model_loading_with_registry` - Registry-based model loading
- `test_batch_inference_performance` - Throughput benchmarking
- `test_concurrent_inference` - Thread-safe model access

---

## Milestone 7: Export & Optimization

### Epic 7.1: Export Utilities

#### Issue 7.1.1: `export_utility/__init__.py`

**Tests:**

- `test_export_utility_exports` - All public APIs exported
- `test_export_utility_version` - Version string defined
- `test_export_utility_dir` - EXPORT_UTILITY_DIR path correct

---

#### Issue 7.1.2: `export_utility/export_model.py`

**Tests:**

- `test_supported_formats_defined` - safetensors, pytorch, huggingface
- `test_model_card_template` - Template has all placeholders
- `test_load_model_and_tokenizer` - Loads from checkpoint path
- `test_load_model_and_tokenizer_heads` - Reports head count
- `test_filter_heads_subset` - Keeps only specified heads
- `test_filter_heads_none` - Keeps all when None
- `test_filter_heads_removes_correctly` - Removes unwanted heads
- `test_export_capabilities_json` - Writes capabilities.json
- `test_export_capabilities_json_with_calibration` - Includes calibration config
- `test_export_capabilities_json_structure` - Correct structure per head
- `test_export_training_config` - Copies training_config.json
- `test_export_training_args` - Copies training_args.json
- `test_load_calibration_config` - Loads YAML calibration file
- `test_load_calibration_config_missing` - Handles missing file
- `test_export_calibration_config` - Writes calibration_config.yaml
- `test_generate_model_card` - Creates README.md
- `test_generate_model_card_capabilities_list` - Lists all capabilities
- `test_generate_model_card_eval_results` - Includes evaluation results
- `test_export_model_safetensors` - Exports to safetensors format
- `test_export_model_safetensors_contiguous` - Tensors are contiguous
- `test_export_model_pytorch` - Exports pytorch_model.bin
- `test_export_model_huggingface` - Uses save_pretrained
- `test_export_tokenizer` - Tokenizer files saved
- `test_verify_export_loads` - Exported model loads correctly
- `test_verify_export_heads` - Exported model has expected heads
- `test_verify_export_inference` - Quick inference test passes

---

#### Issue 7.1.3: `export_utility/export_onnx.py`

**Tests:**

- `test_default_opset_version` - Uses opset 17
- `test_quantization_modes` - none, dynamic, static defined
- `test_onnx_export_wrapper_init` - Wrapper initializes with capability
- `test_onnx_export_wrapper_is_sequence_labeling` - Detects NER/temporal
- `test_onnx_export_wrapper_is_embedding` - Detects embedding capability
- `test_onnx_export_wrapper_forward` - Returns logits or embeddings
- `test_export_to_onnx_creates_file` - Creates .onnx file
- `test_export_to_onnx_validates` - ONNX model passes validation
- `test_export_to_onnx_dynamic_axes` - Supports variable batch/seq
- `test_export_to_onnx_sequence_labeling` - Correct output shape for NER
- `test_export_to_onnx_embedding` - Correct output shape for embedding
- `test_apply_dynamic_quantization` - Creates _quantized_dynamic.onnx
- `test_apply_dynamic_quantization_int8` - Uses INT8 weight type
- `test_apply_static_quantization` - Creates _quantized_static.onnx
- `test_apply_static_quantization_calibration` - Uses calibration data
- `test_calibration_data_reader` - Reads calibration samples correctly
- `test_convert_to_fp16` - Creates _fp16.onnx
- `test_convert_to_fp16_preserves_accuracy` - FP16 output close to FP32
- `test_validate_onnx_model` - Compares ONNX vs PyTorch outputs
- `test_validate_onnx_model_tolerance` - Within tolerance threshold
- `test_validate_onnx_model_reports_errors` - Reports validation failures
- `test_benchmark_onnx_model` - Measures ONNX latency

---

#### Issue 7.1.4: `export_utility/optimized_inference.py`

**Tests:**

- `test_inference_result_init` - InferenceResult dataclass
- `test_multi_capability_result_init` - MultiCapabilityResult dataclass
- `test_batch_inference_result_init` - BatchInferenceResult dataclass
- `test_encoder_cache_init` - EncoderCache initializes with max_size
- `test_encoder_cache_hash_text` - MD5 hash of text
- `test_encoder_cache_get_miss` - Returns None on cache miss
- `test_encoder_cache_put_and_get` - Stores and retrieves correctly
- `test_encoder_cache_lru_eviction` - Evicts oldest when full
- `test_encoder_cache_stats` - Returns hits, misses, hit_rate
- `test_encoder_cache_clear` - Clears all entries
- `test_encoder_cache_thread_safe` - Works under concurrent access
- `test_parallel_head_executor_init` - Initializes with heads
- `test_parallel_head_executor_cuda_streams` - Creates CUDA streams
- `test_parallel_head_executor_thread_pool` - Creates thread pool
- `test_parallel_head_executor_execute_parallel` - Runs heads in parallel
- `test_parallel_head_executor_cuda_streams_execution` - GPU parallel execution
- `test_parallel_head_executor_threaded_execution` - CPU threaded execution
- `test_parallel_head_executor_shutdown` - Shuts down cleanly
- `test_postprocess_token_classification` - Extracts entities from logits
- `test_postprocess_token_classification_bio` - Handles BIO tags
- `test_postprocess_token_classification_special_tokens` - Skips CLS/SEP/PAD
- `test_optimized_multitask_model_init` - OptimizedMultiTaskModel initializes
- `test_optimized_multitask_model_from_pretrained` - Loads from checkpoint
- `test_optimized_multitask_model_infer_single` - Single text inference
- `test_optimized_multitask_model_infer_batch` - Batch inference
- `test_optimized_multitask_model_infer_with_cache` - Uses encoder cache
- `test_optimized_multitask_model_parallel_heads` - Parallel head execution
- `test_optimized_multitask_model_latency_target` - < 15ms for 12 capabilities
- `test_optimized_multitask_model_throughput` - > 1000 samples/sec target

---

#### Issue 7.1.5: `export_utility/benchmark_latency.py`

**Tests:**

- `test_default_batch_sizes` - 1, 8, 32 defined
- `test_default_seq_lengths` - 64, 128, 256, 512 defined
- `test_benchmark_config_init` - BenchmarkConfig dataclass
- `test_benchmark_report_init` - BenchmarkReport dataclass
- `test_benchmark_report_to_dict` - Serialization
- `test_get_system_info` - Returns python, platform, torch versions
- `test_get_system_info_cuda` - Includes GPU info when available
- `test_get_system_info_cpu` - Includes CPU info
- `test_generate_markdown_report` - Creates human-readable report
- `test_generate_markdown_report_system_info` - Includes system info
- `test_generate_markdown_report_results` - Includes benchmark results
- `test_run_benchmark_single_capability` - Benchmarks one capability
- `test_run_benchmark_multiple_batch_sizes` - Tests different batch sizes
- `test_run_benchmark_multiple_seq_lengths` - Tests different seq lengths
- `test_run_benchmark_warmup` - Excludes warmup from measurements
- `test_run_benchmark_iterations` - Runs specified iterations
- `test_run_benchmark_cpu` - Works on CPU
- `test_run_benchmark_gpu` - Works on GPU
- `test_save_benchmark_json` - Saves results to JSON
- `test_save_benchmark_markdown` - Saves report to markdown

---

### Milestone 7 Integration Points

| From | To | Connection |
|------|-----|------------|
| `export_utility/__init__.py` | `export_utility/export_model.py` | Exports model export functions |
| `export_utility/__init__.py` | `export_utility/export_onnx.py` | Exports ONNX functions |
| `export_utility/__init__.py` | `export_utility/optimized_inference.py` | Exports optimized model |
| `export_utility/export_model.py` | `models/modernbert_multitask.py` | Loads model for export |
| `export_utility/export_onnx.py` | `models/modernbert_multitask.py` | Wraps model for ONNX |
| `export_utility/optimized_inference.py` | `models/modernbert_multitask.py` | Wraps for optimized inference |
| `export_utility/optimized_inference.py` | `data/labels.py` | Uses label schemas |
| `export_utility/benchmark_latency.py` | `evaluation/benchmarks.py` | Uses LatencyBenchmark |

**Future Integration Tests (Milestone 7):**

- `test_export_and_load_roundtrip` - Export → Load → Inference
- `test_onnx_vs_pytorch_accuracy` - ONNX matches PyTorch outputs
- `test_quantized_model_accuracy` - Quantization preserves accuracy
- `test_optimized_vs_standard_inference` - Same outputs, faster
- `test_full_export_pipeline` - Model → Export → Benchmark

---

## Milestone 8: Scripts

### Epic 8.1: Training Scripts

#### Issue 8.1.1: `scripts/train_stage_a.py`

**Tests:**

- `test_load_config` - Loads YAML config file
- `test_load_config_with_defaults` - Merges base configs
- `test_load_config_missing_file` - Raises FileNotFoundError
- `test_deep_merge` - Deep merges dictionaries correctly
- `test_deep_merge_override` - Override values take precedence
- `test_apply_overrides_simple` - Applies key=value overrides
- `test_apply_overrides_nested` - Applies key.subkey=value overrides
- `test_apply_overrides_type_parsing` - Parses int, float, bool values
- `test_parse_args_config` - Parses --config argument
- `test_parse_args_resume` - Parses --resume_from_checkpoint
- `test_parse_args_output_dir` - Parses --output_dir override
- `test_parse_args_seed` - Parses --seed override
- `test_parse_args_debug` - Parses --debug flag
- `test_parse_args_dry_run` - Parses --dry_run flag
- `test_parse_args_ignore_optimizer_state` - Parses optimizer state flag
- `test_init_model_from_config` - Initializes ModernBertMultiTaskModel
- `test_init_model_capabilities` - Creates heads for enabled capabilities
- `test_init_model_torch_dtype` - Uses correct dtype (bfloat16/float16)
- `test_init_model_attention_implementation` - SDPA or Flash Attention
- `test_configure_head_loss_asl` - Configures ASL for multi-label
- `test_configure_head_loss_focal` - Configures focal loss
- `test_configure_head_loss_class_weights` - Computes class weights
- `test_configure_head_loss_label_smoothing` - Sets label smoothing
- `test_configure_head_loss_pos_weight` - Sets positive sample weights
- `test_configure_head_loss_hierarchical_emotion` - SOTA emotion features
- `test_compute_class_weights_from_dataset` - Inverse frequency weights
- `test_compute_class_weights_multi_hot` - Handles multi-hot labels
- `test_compute_class_weights_clipping` - Clips extreme weights
- `test_stage_a_capabilities` - Correct Stage A tasks enabled
- `test_stage_a_datasets_loaded` - All Stage A datasets load
- `test_stage_a_trainer_init` - MultiTaskTrainer initializes
- `test_stage_a_ema_enabled` - EMA model configured when enabled
- `test_stage_a_dry_run` - Dry run validates without training
- `test_stage_a_checkpoint_saving` - Checkpoints saved correctly
- `test_stage_a_resume_from_checkpoint` - Training resumes correctly

---

#### Issue 8.1.2: `scripts/train_stage_b.py`

**Tests:**

- `test_load_config_stage_b` - Loads Stage B YAML config file
- `test_load_config_with_defaults_stage_b` - Merges base configs correctly
- `test_load_config_missing_file_stage_b` - Raises FileNotFoundError
- `test_apply_overrides_stage_b` - Applies key=value overrides
- `test_apply_overrides_nested_stage_b` - Applies key.subkey=value overrides
- `test_apply_overrides_peft_config` - Applies peft.lora.r override
- `test_parse_args_config_stage_b` - Parses --config argument
- `test_parse_args_data_config` - Parses --data_config argument
- `test_parse_args_output_dir_stage_b` - Parses --output_dir override
- `test_parse_args_checkpoint_dir` - Parses --checkpoint_dir override
- `test_parse_args_debug_stage_b` - Parses --debug flag
- `test_parse_args_dry_run_stage_b` - Parses --dry_run flag
- `test_parse_args_resume_from_checkpoint_stage_b` - Parses checkpoint path
- `test_parse_args_seed_stage_b` - Parses --seed override
- `test_parse_args_local_rank` - Parses --local_rank for distributed
- `test_parse_args_overrides_list` - Parses positional overrides
- `test_stage_a_capabilities_constant` - STAGE_A_CAPABILITIES defined correctly
- `test_stage_b_capabilities_constant` - STAGE_B_CAPABILITIES defined correctly
- `test_all_capabilities_constant` - ALL_CAPABILITIES is union of A and B
- `test_load_stage_a_model` - Loads model from checkpoint path
- `test_load_stage_a_model_exists_check` - Checks checkpoint path exists
- `test_load_stage_a_model_alt_path` - Falls back to alternative path
- `test_load_stage_a_model_file_not_found` - Raises FileNotFoundError if missing
- `test_load_stage_a_model_heads_present` - Verifies Stage A heads loaded
- `test_add_stage_b_heads` - Adds NER_FAMILY, INGRESS, SAFETY_FAMILYOS, RELATION, INTENT
- `test_add_stage_b_heads_with_shared_pooler` - Uses CLSMeanPooler when enabled
- `test_add_stage_b_heads_with_pair_encoder` - Uses CrossAttentionPairEncoder
- `test_add_stage_b_heads_pooler_type_cls` - Selects CLSPooler
- `test_add_stage_b_heads_pooler_type_mean` - Selects MeanPooler
- `test_add_stage_b_heads_pooler_type_cls_mean` - Selects CLSMeanPooler
- `test_add_stage_b_heads_pair_encoder_num_layers` - Respects num_layers config
- `test_add_stage_b_heads_task_group_adapters` - Uses TaskGroupAdapters when enabled
- `test_add_stage_b_heads_hierarchical_emotion` - Applies SOTA emotion features
- `test_add_stage_b_heads_loss_config` - Configures loss functions per head
- `test_freeze_stage_a_heads` - Freezes Stage A heads when enabled
- `test_freeze_stage_a_heads_disabled` - Does not freeze when disabled
- `test_freeze_stage_a_heads_selection` - Freezes only specified heads
- `test_freeze_stage_a_heads_requires_grad` - Sets requires_grad=False
- `test_apply_lora` - Applies LoRA adapters to model
- `test_apply_lora_target_modules` - Targets q_proj, k_proj, v_proj, o_proj
- `test_apply_lora_r_value` - Uses configured r (default 32)
- `test_apply_lora_alpha_value` - Uses configured lora_alpha (default 64)
- `test_apply_lora_dropout` - Uses configured dropout (default 0.1)
- `test_apply_lora_bias` - Uses bias setting (none/lora_only)
- `test_apply_lora_returns_peft_model` - Returns PeftModel instance
- `test_apply_lora_modules_to_save` - Saves new Stage B heads
- `test_apply_lora_trainable_params` - Reports trainable parameter count
- `test_load_datasets_for_stage_b` - Loads Stage B datasets
- `test_load_datasets_for_stage_b_familyos` - Loads FamilyOS domain data
- `test_load_datasets_for_stage_b_replay` - Loads Stage A replay data
- `test_load_datasets_for_stage_b_replay_ratio` - Respects replay_ratio config
- `test_load_datasets_for_stage_b_eval_split` - Loads eval splits
- `test_load_datasets_for_stage_b_debug_mode` - Limits data in debug mode
- `test_load_datasets_for_stage_b_tokenization` - Tokenizes correctly
- `test_apply_safety_oversampling` - Applies oversampling to safety data
- `test_apply_safety_oversampling_crisis_rate` - 20x oversampling for CRISIS
- `test_apply_safety_oversampling_red_rate` - 5x oversampling for RED
- `test_apply_safety_oversampling_disabled_debug` - Disabled in debug mode
- `test_apply_safety_oversampling_indices` - Returns oversampled indices
- `test_create_training_args` - Creates MultiTaskTrainingArguments
- `test_create_training_args_learning_rate` - Higher LR for LoRA (1e-4)
- `test_create_training_args_weight_decay` - Sets weight decay
- `test_create_training_args_max_grad_norm` - Sets gradient clipping
- `test_create_training_args_optimizer` - Uses adamw_torch_fused
- `test_create_training_args_scheduler` - Uses cosine scheduler
- `test_create_training_args_warmup_ratio` - Sets warmup ratio
- `test_create_training_args_epochs` - Sets num_train_epochs
- `test_create_training_args_batch_sizes` - Sets train/eval batch sizes
- `test_create_training_args_gradient_accumulation` - Sets accumulation steps
- `test_create_training_args_eval_strategy` - Sets evaluation strategy
- `test_create_training_args_save_strategy` - Sets save strategy
- `test_create_training_args_load_best_model` - Enables load best model
- `test_create_training_args_metric_for_best` - Uses safety_familyos_f1
- `test_create_training_args_bf16_supported` - Enables bf16 when supported
- `test_create_training_args_bf16_fallback` - Falls back when bf16 unsupported
- `test_create_training_args_gradient_checkpointing` - Enables checkpointing
- `test_create_training_args_debug_mode` - Reduces batch sizes in debug
- `test_create_training_args_sampling_strategy` - Uses temperature sampling
- `test_create_training_args_sampling_temperature` - Sets temperature to 2.0
- `test_create_training_args_resume_checkpoint` - Passes resume path
- `test_create_training_args_run_name` - Generates timestamped run name
- `test_save_merged_model` - Saves merged model to output dir
- `test_save_merged_model_lora_separate` - Saves LoRA adapters separately
- `test_save_merged_model_merge_and_unload` - Merges LoRA into base model
- `test_save_merged_model_tokenizer` - Saves tokenizer with model
- `test_save_merged_model_capabilities_json` - Creates capabilities.json
- `test_save_merged_model_epic5_config` - Includes Epic 5.0 config in caps
- `test_save_merged_model_stage_b_marker` - Marks stage as B
- `test_train_stage_b_full` - Complete Stage B training pipeline
- `test_train_stage_b_config_loading` - Loads config correctly
- `test_train_stage_b_seed_setting` - Sets random seed
- `test_train_stage_b_device_placement` - Moves model to correct device
- `test_train_stage_b_task_weights` - Sets task weights correctly
- `test_train_stage_b_safety_weight_boost` - Safety gets 1.5x weight
- `test_train_stage_b_replay_weight` - Replay tasks get 0.2 weight
- `test_train_stage_b_trainer_creation` - Creates MultiTaskTrainer
- `test_train_stage_b_training_execution` - Executes training
- `test_train_stage_b_checkpoint_saving` - Saves checkpoints
- `test_train_stage_b_best_model_save` - Saves best model
- `test_train_stage_b_config_save` - Saves training_config.json
- `test_train_stage_b_dry_run` - Validates without training
- `test_train_stage_b_dry_run_epic5_status` - Reports Epic 5.0 status
- `test_train_stage_b_resume` - Resumes from checkpoint

---

### Epic 8.2: Evaluation & Utility Scripts

#### Issue 8.2.1: `scripts/evaluate_stage_a.py`

**Tests:**

- `test_parse_args_evaluate` - Parses --model argument
- `test_parse_args_output` - Parses --output_dir argument
- `test_load_model_and_tokenizer` - Loads model for evaluation
- `test_load_conll_dataset` - Loads CoNLL-2003 for NER eval
- `test_load_sst2_dataset` - Loads SST-2 for sentiment eval
- `test_load_snli_dataset` - Loads SNLI for NLI eval
- `test_load_goemotions_dataset` - Loads GoEmotions for emotion eval
- `test_label_mapping_ner` - Correct CoNLL labels
- `test_label_mapping_sentiment` - Correct SST-2 labels
- `test_label_mapping_nli` - Correct SNLI labels
- `test_label_mapping_emotions` - Correct GoEmotions labels
- `test_evaluate_ner_general` - Evaluates NER metrics
- `test_evaluate_sentiment` - Evaluates sentiment metrics
- `test_evaluate_nli` - Evaluates NLI metrics
- `test_evaluate_emotions` - Evaluates emotion metrics
- `test_aggregate_results` - Aggregates all task results
- `test_save_results_json` - Saves results to JSON
- `test_print_summary` - Prints human-readable summary

---

#### Issue 8.2.2: `scripts/evaluate.py`

**Tests:**

- `test_parse_args_model` - Parses --model_path argument
- `test_parse_args_tasks` - Parses --tasks argument (comma-separated)
- `test_parse_args_output_dir` - Parses --output_dir argument
- `test_parse_args_batch_size` - Parses --batch_size argument
- `test_parse_args_max_samples` - Parses --max_samples argument
- `test_parse_args_all_tasks` - Parses --all flag
- `test_load_evaluation_config` - Loads evaluation config YAML
- `test_evaluate_single_task` - Evaluates one task
- `test_evaluate_multiple_tasks` - Evaluates multiple tasks
- `test_evaluate_all_capabilities` - Evaluates all 12 capabilities
- `test_metrics_computation` - Computes correct metrics per task
- `test_results_aggregation` - Aggregates results correctly
- `test_save_results` - Saves evaluation results
- `test_generate_report` - Generates markdown report
- `test_forgetting_comparison` - Compares Stage A vs B metrics
- `test_quality_gate_checks` - Validates quality gates

---

### Milestone 8 Integration Points

| From | To | Connection |
|------|-----|------------|
| `scripts/train_stage_a.py` | `models/modernbert_multitask.py` | Trains model |
| `scripts/train_stage_a.py` | `trainers/multitask_trainer.py` | Uses MultiTaskTrainer |
| `scripts/train_stage_a.py` | `data/loaders.py` | Loads Stage A datasets |
| `scripts/train_stage_a.py` | `trainers/ema.py` | Uses EMA model |
| `scripts/train_stage_b.py` | `models/modernbert_multitask.py` | Loads and extends model |
| `scripts/train_stage_b.py` | `models/heads.py` | Adds Stage B heads |
| `scripts/train_stage_b.py` | `models/poolers.py` | Uses shared poolers |
| `scripts/train_stage_b.py` | `models/pair_encoder.py` | Uses pair encoder |
| `scripts/train_stage_b.py` | `trainers/multitask_trainer.py` | Uses MultiTaskTrainer |
| `scripts/train_stage_b.py` | `data/loaders.py` | Loads Stage B + replay |
| `scripts/train_stage_b.py` | `peft.LoraConfig` | Applies LoRA adapters |
| `scripts/evaluate_stage_a.py` | `evaluation/evaluator.py` | Uses Evaluator |
| `scripts/evaluate_stage_a.py` | `models/modernbert_multitask.py` | Loads model |
| `scripts/evaluate.py` | `evaluation/evaluator.py` | Uses Evaluator |
| `scripts/evaluate.py` | `evaluation/metrics.py` | Uses metric functions |

**Future Integration Tests (Milestone 8):**

- `test_stage_a_train_and_evaluate` - Train Stage A → Evaluate
- `test_stage_b_train_from_stage_a` - Stage A checkpoint → Stage B training
- `test_stage_b_with_lora_merge` - Train → Merge → Validate heads
- `test_stage_b_epic5_full_pipeline` - Shared pooler + pair encoder training
- `test_forgetting_evaluation_pipeline` - Stage A vs Stage B comparison
- `test_safety_calibration_after_training` - Train → Calibrate thresholds

---

## Milestone 9: End-to-End Integration Testing

This milestone provides comprehensive integration tests covering all module interactions.
Tests are organized by interaction complexity: pairwise (2 components), triplets (3),
quadruples (4), and full pipeline (5+) integrations.

### Epic 9.1: Pairwise Integration Tests

#### Issue 9.1.1: Data + Model Integration

**Tests:**

- `test_labels_with_classification_head` - Label schemas work with ClassificationHead
- `test_labels_with_token_classification_head` - Label schemas work with TokenClassificationHead
- `test_labels_with_multilabel_head` - Label schemas work with MultiLabelClassificationHead
- `test_datasets_with_modernbert_input` - Datasets produce valid model inputs
- `test_datasets_with_tokenizer` - Dataset tokenization produces expected format
- `test_loaders_with_datasets` - DataLoaders wrap datasets correctly
- `test_loaders_with_collators` - DataLoaders use correct collation
- `test_collators_with_labels` - Collators handle label schemas
- `test_preprocessing_with_tokenizer` - Preprocessing produces tokenizable output
- `test_preprocessing_with_model_max_length` - Respects max_length constraint
- `test_label_mapping_with_model_output` - Model outputs map to labels correctly
- `test_vocab_with_embedding_layer` - Tokenizer vocab matches embedding dim

---

#### Issue 9.1.2: Model + Trainer Integration

**Tests:**

- `test_model_with_trainer_init` - Model initializes in trainer
- `test_model_with_training_step` - Model executes training step
- `test_model_with_evaluation_step` - Model executes evaluation step
- `test_model_gradients_flow` - Gradients flow through all heads
- `test_model_with_optimizer` - Optimizer updates model weights
- `test_model_with_scheduler` - Scheduler adjusts learning rate
- `test_model_with_collator` - Collator provides correct batch format
- `test_heads_with_loss_functions` - Each head computes loss
- `test_heads_with_gradient_checkpointing` - Heads work with checkpointing
- `test_ema_with_model_updates` - EMA tracks model updates
- `test_callbacks_with_trainer_events` - Callbacks fire on events
- `test_curriculum_with_trainer` - Curriculum stages work in trainer

---

#### Issue 9.1.3: Model + Evaluation Integration

**Tests:**

- `test_model_with_evaluator` - Model works with Evaluator class
- `test_model_predictions_with_metrics` - Predictions compute to metrics
- `test_model_with_benchmarks` - Model runs in benchmark suite
- `test_heads_with_metric_computation` - Each head's metrics computed
- `test_model_with_safety_eval` - Model works with SafetyEvaluator
- `test_model_with_cultural_robustness` - Model in cultural robustness eval
- `test_model_with_forgetting_eval` - Model in forgetting evaluation
- `test_model_inference_with_metrics` - Inference outputs score correctly
- `test_model_batch_eval_consistency` - Batch vs single eval consistent
- `test_model_with_latency_benchmark` - Model latency measured correctly

---

#### Issue 9.1.4: Inference + K0 Runtime Integration

**Tests:**

- `test_unified_output_with_k0_registry` - UnifiedNLPOutput uses K0 registry
- `test_capability_resolution_with_inference` - K0 capability resolves to inference
- `test_model_registry_with_model_loading` - Registry loads models for inference
- `test_head_registry_with_head_inference` - Registry matches head to inference
- `test_tokenizer_cache_with_inference` - Cached tokenizer used in inference
- `test_model_cache_with_inference` - Cached model used in inference
- `test_legacy_migration_with_inference` - Legacy modules migrate to unified
- `test_capability_aliases_with_inference` - Aliases resolve in inference calls

---

#### Issue 9.1.5: Export + Model Integration

**Tests:**

- `test_export_with_model_loading` - Export loads model correctly
- `test_export_preserves_heads` - Exported model retains all heads
- `test_export_safetensors_roundtrip` - Export → Load preserves weights
- `test_export_pytorch_roundtrip` - PyTorch export → Load works
- `test_onnx_export_with_model` - ONNX export produces valid model
- `test_onnx_export_preserves_output` - ONNX output matches PyTorch
- `test_quantization_with_model` - Quantized model produces outputs
- `test_optimized_inference_with_model` - Optimized wrapper works
- `test_encoder_cache_with_model` - Encoder cache stores embeddings
- `test_parallel_heads_with_model` - Parallel head execution works

---

### Epic 9.2: Triplet Integration Tests

#### Issue 9.2.1: Data → Model → Trainer

**Tests:**

- `test_data_model_trainer_forward` - Data flows through model in trainer
- `test_data_model_trainer_backward` - Gradients flow back through model
- `test_datasets_model_collator` - Datasets collate and feed model
- `test_loaders_model_batch_training` - Loaders provide batches for training
- `test_preprocessing_tokenization_model` - Preprocessing → Tokenize → Model
- `test_labels_heads_loss` - Labels match heads compute loss
- `test_multi_dataset_model_sampling` - Multiple datasets sample to model
- `test_data_augmentation_model_training` - Augmented data trains model
- `test_dataset_split_model_eval` - Eval splits evaluate model
- `test_task_datasets_multitask_trainer` - Per-task datasets in multitask trainer

---

#### Issue 9.2.2: Model → Trainer → Checkpoint

**Tests:**

- `test_model_trainer_save_checkpoint` - Model saves via trainer
- `test_model_trainer_load_checkpoint` - Model loads from checkpoint
- `test_model_trainer_resume_training` - Training resumes from checkpoint
- `test_model_trainer_best_model_save` - Best model saved correctly
- `test_model_ema_checkpoint` - EMA state saved in checkpoint
- `test_model_optimizer_checkpoint` - Optimizer state saved
- `test_model_scheduler_checkpoint` - Scheduler state saved
- `test_model_lora_checkpoint` - LoRA adapters saved
- `test_model_trainer_checkpoint_metrics` - Metrics saved with checkpoint
- `test_model_multi_gpu_checkpoint` - Distributed checkpoint works

---

#### Issue 9.2.3: Model → Evaluation → Metrics

**Tests:**

- `test_model_evaluator_per_task_metrics` - Per-task metrics computed
- `test_model_evaluator_aggregate_metrics` - Aggregated metrics correct
- `test_model_benchmark_latency_metrics` - Latency metrics measured
- `test_model_safety_eval_metrics` - Safety metrics computed
- `test_model_crisis_recall_metrics` - CRISIS recall correctly computed
- `test_model_cultural_robustness_metrics` - Cultural FPR computed
- `test_model_forgetting_metrics` - Forgetting drop computed
- `test_model_calibration_metrics` - ECE/MCE computed
- `test_model_confusion_matrix_metrics` - Confusion matrices generated
- `test_model_per_class_metrics` - Per-class breakdown computed

---

#### Issue 9.2.4: Config → Model → Training

**Tests:**

- `test_config_model_initialization` - Config initializes model correctly
- `test_config_head_configuration` - Head config creates correct heads
- `test_config_loss_configuration` - Loss config applied to heads
- `test_config_pooler_configuration` - Pooler config creates poolers
- `test_config_pair_encoder_configuration` - Pair encoder config applied
- `test_config_training_args` - Training config creates TrainingArguments
- `test_config_optimizer_params` - Optimizer params from config
- `test_config_scheduler_params` - Scheduler params from config
- `test_config_data_loading` - Data config loads correct datasets
- `test_config_overrides_applied` - CLI overrides modify training

---

#### Issue 9.2.5: Export → Load → Inference

**Tests:**

- `test_export_load_inference_cycle` - Export → Load → Run inference
- `test_export_load_all_capabilities` - All 12 capabilities work after export
- `test_export_load_batch_inference` - Batch inference after export
- `test_onnx_export_load_inference` - ONNX export → Load → Inference
- `test_quantized_export_load_inference` - Quantized → Load → Inference
- `test_optimized_export_inference` - Optimized model inferences correctly
- `test_export_load_accuracy_preserved` - Accuracy same after export
- `test_export_load_calibration_preserved` - Calibration config preserved
- `test_export_load_model_card` - Model card generated and valid
- `test_export_load_capabilities_json` - capabilities.json correct

---

### Epic 9.3: Quadruple Integration Tests

#### Issue 9.3.1: Data → Model → Trainer → Evaluation

**Tests:**

- `test_full_train_eval_cycle` - Train → Evaluate complete cycle
- `test_data_to_evaluation_pipeline` - Data loads → Model trains → Evaluate
- `test_multi_task_train_eval` - Multi-task training + evaluation
- `test_train_eval_metrics_consistency` - Train metrics match eval metrics
- `test_data_model_trainer_evaluator` - Full data-to-evaluation flow
- `test_validation_during_training` - Eval during training works
- `test_best_model_evaluation` - Best model evaluates correctly
- `test_task_weights_affect_evaluation` - Task weights impact results
- `test_curriculum_affects_evaluation` - Curriculum stages affect metrics
- `test_ema_evaluation` - EMA model evaluates correctly

---

#### Issue 9.3.2: Config → Data → Model → Trainer

**Tests:**

- `test_config_drives_full_training` - Config drives entire training
- `test_config_data_model_consistency` - Config keeps data/model aligned
- `test_config_overrides_full_pipeline` - Overrides propagate correctly
- `test_yaml_to_trained_model` - YAML config → Trained model
- `test_multitask_config_training` - Multi-task config trains correctly
- `test_stage_a_config_training` - Stage A config trains correctly
- `test_stage_b_config_training` - Stage B config trains correctly
- `test_lora_config_training` - LoRA config applies correctly
- `test_debug_config_training` - Debug mode works correctly
- `test_distributed_config_training` - Distributed config works

---

#### Issue 9.3.3: Model → Export → Load → Evaluation

**Tests:**

- `test_model_export_reload_evaluate` - Train → Export → Load → Evaluate
- `test_export_preserves_evaluation_metrics` - Metrics same after export
- `test_onnx_model_evaluation` - ONNX model evaluates correctly
- `test_quantized_model_evaluation` - Quantized model evaluates correctly
- `test_optimized_model_evaluation` - Optimized model evaluates correctly
- `test_export_safety_evaluation` - Exported model passes safety eval
- `test_export_cultural_robustness` - Exported model passes cultural eval
- `test_export_benchmark_comparison` - Exported model benchmarks match
- `test_lora_merge_evaluation` - Merged LoRA evaluates correctly
- `test_capability_subset_export_eval` - Subset export evaluates correctly

---

#### Issue 9.3.4: Stage A → LoRA → Stage B → Merge

**Tests:**

- `test_stage_a_to_stage_b_pipeline` - Full Stage A → B pipeline
- `test_lora_application_on_stage_a` - LoRA applies to Stage A model
- `test_stage_b_heads_with_lora` - Stage B heads train with LoRA
- `test_lora_merge_preserves_stage_a` - Stage A heads preserved
- `test_lora_merge_adds_stage_b` - Stage B heads added correctly
- `test_merged_model_all_capabilities` - All 12 capabilities work
- `test_merged_model_forgetting_check` - Stage A tasks not forgotten
- `test_merged_model_stage_b_quality` - Stage B tasks meet targets
- `test_adapter_save_load` - LoRA adapters save and reload
- `test_replay_prevents_forgetting` - Replay data prevents forgetting

---

### Epic 9.4: Full Pipeline Integration Tests

#### Issue 9.4.1: Complete Stage A Pipeline

**Tests:**

- `test_stage_a_full_pipeline` - Config → Data → Model → Train → Evaluate
- `test_stage_a_all_capabilities` - All Stage A capabilities train/eval
- `test_stage_a_ner_end_to_end` - NER_GENERAL full pipeline
- `test_stage_a_sentiment_end_to_end` - SENTIMENT full pipeline
- `test_stage_a_emotions_end_to_end` - EMOTIONS full pipeline
- `test_stage_a_safety_generic_end_to_end` - SAFETY_GENERIC full pipeline
- `test_stage_a_nli_end_to_end` - NLI full pipeline
- `test_stage_a_embedding_end_to_end` - EMBEDDING full pipeline
- `test_stage_a_temporal_end_to_end` - TEMPORAL full pipeline
- `test_stage_a_checkpoint_usable` - Checkpoint works for Stage B
- `test_stage_a_export_pipeline` - Train → Export → Validate
- `test_stage_a_benchmark_pipeline` - Train → Benchmark latency

---

#### Issue 9.4.2: Complete Stage B Pipeline

**Tests:**

- `test_stage_b_full_pipeline` - Stage A → Stage B complete pipeline
- `test_stage_b_all_capabilities` - All 12 capabilities functional
- `test_stage_b_ner_family_end_to_end` - NER_FAMILY full pipeline
- `test_stage_b_ingress_end_to_end` - INGRESS full pipeline
- `test_stage_b_safety_familyos_end_to_end` - SAFETY_FAMILYOS full pipeline
- `test_stage_b_relation_end_to_end` - RELATION full pipeline
- `test_stage_b_intent_end_to_end` - INTENT full pipeline
- `test_stage_b_with_epic5_pooler` - Epic 5.0 shared pooler pipeline
- `test_stage_b_with_epic5_pair_encoder` - Epic 5.0 pair encoder pipeline
- `test_stage_b_lora_merge_pipeline` - LoRA training → Merge → Validate
- `test_stage_b_export_pipeline` - Stage B → Export → Validate
- `test_stage_b_forgetting_pipeline` - Stage B → Forgetting eval

---

#### Issue 9.4.3: Production Deployment Pipeline

**Tests:**

- `test_production_model_pipeline` - Full production-ready pipeline
- `test_train_export_deploy_cycle` - Train → Export → Deploy simulation
- `test_onnx_production_pipeline` - PyTorch → ONNX → Validate
- `test_quantized_production_pipeline` - Quantize → Validate accuracy
- `test_optimized_inference_pipeline` - Optimize → Benchmark → Validate
- `test_model_card_generation` - README.md generated correctly
- `test_capabilities_json_complete` - All capabilities documented
- `test_calibration_config_export` - Calibration exported correctly
- `test_version_tracking` - Model version tracked correctly
- `test_reproducibility_pipeline` - Same seed → Same results
- `test_multi_gpu_to_single_gpu` - Multi-GPU train → Single-GPU export
- `test_inference_latency_target` - < 15ms for all 12 capabilities

---

#### Issue 9.4.4: Safety-Critical Pipeline

**Tests:**

- `test_safety_end_to_end_crisis` - CRISIS detection full pipeline
- `test_safety_end_to_end_red` - RED detection full pipeline
- `test_safety_calibration_pipeline` - Train → Calibrate → Validate
- `test_safety_quality_gates_pipeline` - All safety gates checked
- `test_cultural_robustness_pipeline` - Train → Cultural eval → Pass
- `test_temporal_safety_pipeline` - Temporal escalation tracking
- `test_safety_oversampling_pipeline` - Oversampling → Train → Validate
- `test_crisis_recall_target` - ≥ 98% CRISIS recall achieved
- `test_red_recall_target` - ≥ 90% RED recall achieved
- `test_green_fpr_target` - ≤ 5% GREEN false positive rate
- `test_safety_threshold_optimization` - Optimal thresholds found
- `test_safety_production_readiness` - All safety checks pass

---

#### Issue 9.4.5: K0 Runtime Integration Pipeline

**Tests:**

- `test_k0_full_migration_pipeline` - Legacy → Unified complete
- `test_k0_capability_resolution_all` - All capabilities resolve
- `test_k0_inference_all_capabilities` - All 12 inferences work
- `test_k0_batch_inference_pipeline` - Batch processing pipeline
- `test_k0_concurrent_inference` - Thread-safe inference
- `test_k0_model_registry_production` - Registry production-ready
- `test_k0_cache_management` - Cache correctly managed
- `test_k0_legacy_module_mapping` - All M/P modules map
- `test_k0_sys_nlp_infer_production` - sys_nlp_infer production-ready
- `test_k0_unified_output_complete` - UnifiedNLPOutput all fields

---

### Epic 9.5: Cross-Cutting Integration Tests

#### Issue 9.5.1: Multi-Task Interaction Tests

**Tests:**

- `test_all_12_capabilities_together` - All capabilities train together
- `test_task_interference` - Tasks don't negatively interfere
- `test_shared_encoder_all_tasks` - Shared encoder benefits all
- `test_task_sampling_balance` - Sampling balances task exposure
- `test_task_weighting_effect` - Task weights affect outcomes
- `test_head_isolation` - Heads don't leak gradients
- `test_multi_task_evaluation_consistency` - Eval consistent across tasks
- `test_catastrophic_forgetting_prevention` - No catastrophic forgetting
- `test_positive_transfer` - Positive transfer between tasks
- `test_embedding_quality_across_tasks` - Embeddings work for all tasks

---

#### Issue 9.5.2: Error Handling & Edge Cases

**Tests:**

- `test_empty_batch_handling` - Empty batches handled gracefully
- `test_malformed_input_handling` - Malformed input caught
- `test_missing_labels_handling` - Missing labels handled
- `test_oov_tokens_handling` - Out-of-vocab tokens handled
- `test_max_length_truncation` - Long sequences truncated
- `test_cuda_oom_recovery` - OOM errors handled gracefully
- `test_checkpoint_corruption_handling` - Corrupt checkpoints detected
- `test_config_validation_errors` - Invalid configs caught
- `test_missing_capability_error` - Unknown capabilities error
- `test_device_mismatch_handling` - Device mismatches handled
- `test_dtype_mismatch_handling` - Dtype mismatches handled
- `test_nan_loss_detection` - NaN losses detected and handled

---

#### Issue 9.5.3: Performance & Scalability Tests

**Tests:**

- `test_batch_scaling_performance` - Performance scales with batch size
- `test_sequence_length_performance` - Performance vs sequence length
- `test_num_capabilities_performance` - Performance vs num capabilities
- `test_gradient_accumulation_equivalence` - Accum equals larger batch
- `test_mixed_precision_accuracy` - BF16/FP16 matches FP32
- `test_throughput_benchmarks` - Throughput meets targets
- `test_memory_efficiency` - Memory usage within bounds
- `test_inference_parallelism` - Parallel inference works
- `test_encoder_caching_speedup` - Cache provides speedup
- `test_onnx_speedup` - ONNX faster than PyTorch
- `test_quantization_speedup` - Quantization provides speedup
- `test_large_dataset_handling` - Large datasets handled efficiently

---

#### Issue 9.5.4: Regression Test Suite

**Tests:**

- `test_regression_ner_general_f1` - NER F1 ≥ 90%
- `test_regression_sentiment_accuracy` - Sentiment accuracy ≥ 94%
- `test_regression_emotions_macro_f1` - Emotions macro F1 ≥ 50%
- `test_regression_nli_accuracy` - NLI accuracy ≥ 88%
- `test_regression_embedding_spearman` - Embedding ρ ≥ 0.85
- `test_regression_safety_generic_f1` - Safety generic F1 ≥ 85%
- `test_regression_ner_family_f1` - NER family F1 ≥ 85%
- `test_regression_ingress_accuracy` - Ingress accuracy ≥ 90%
- `test_regression_safety_familyos_f1` - Safety FamilyOS F1 ≥ 85%
- `test_regression_relation_f1` - Relation F1 ≥ 75%
- `test_regression_intent_accuracy` - Intent accuracy ≥ 90%
- `test_regression_temporal_f1` - Temporal F1 ≥ 80%
- `test_regression_latency_target` - Latency < 15ms
- `test_regression_memory_target` - Memory < 2GB

---

### Milestone 9 Integration Points

| From | To | Connection |
|------|-----|------------|
| Epic 9.1 | All Modules | Pairwise module connections validated |
| Epic 9.2 | Core Pipelines | Three-component flows validated |
| Epic 9.3 | Extended Pipelines | Four-component flows validated |
| Epic 9.4 | Full Pipelines | End-to-end production flows validated |
| Epic 9.5 | Cross-Cutting | System-wide properties validated |
| `data/*` | `models/*` | Data-Model interface verified |
| `models/*` | `trainers/*` | Model-Trainer interface verified |
| `trainers/*` | `evaluation/*` | Trainer-Eval interface verified |
| `scripts/*` | All Modules | Scripts orchestrate all modules |
| `export_utility/*` | `inference/*` | Export-Inference pipeline verified |
| `k0/*` | `inference/*` | K0-Inference integration verified |

**Summary Statistics:**

| Epic | Focus | Test Count |
|------|-------|------------|
| 9.1 | Pairwise Integration | 50 tests |
| 9.2 | Triplet Integration | 50 tests |
| 9.3 | Quadruple Integration | 40 tests |
| 9.4 | Full Pipeline | 60 tests |
| 9.5 | Cross-Cutting | 46 tests |
| **Total** | **Milestone 9** | **246 tests** |

---

## Testing Plan Summary

### Test Count by Milestone

| Milestone | Focus | Estimated Tests |
|-----------|-------|-----------------|
| M1: Core Config | Configuration loading | ~46 |
| M2: Data Layer | Datasets, labels, preprocessing | ~101 |
| M3: Model Layer | ModernBERT, heads, poolers | ~150 |
| M4: Training | Trainers, samplers, callbacks | ~130 |
| M5: Evaluation | Metrics, evaluators, safety | ~186 |
| M6: Inference | Runtime, K0 registry | ~114 |
| M7: Export | ONNX, optimization, benchmarks | ~100 |
| M8: Scripts | Training & evaluation scripts | ~170 |
| M9: Integration | End-to-end integration | ~246 |
| **Total** | | **~1,243 tests** |

### Priority Order

1. **Critical (M1-M2):** Core config and data layer must work first
2. **High (M3-M4):** Model and training are core functionality
3. **Medium (M5-M6):** Evaluation and inference for validation
4. **Standard (M7-M8):** Export and scripts for deployment
5. **Comprehensive (M9):** Integration tests validate everything

### Quality Gates

- **Unit Tests:** 100% of public APIs tested
- **Integration Tests:** All module pairs tested
- **End-to-End:** Full pipelines for Stage A and Stage B
- **Safety:** CRISIS recall ≥ 98%, RED recall ≥ 90%
- **Performance:** Latency < 15ms, Memory < 2GB
- **Regression:** All capabilities meet baseline metrics

---

| From | To | Connection |
|------|-----|------------|
| `scripts/train_stage_a.py` | `models/modernbert_multitask.py` | Trains model |
| `scripts/train_stage_a.py` | `trainers/multitask_trainer.py` | Uses MultiTaskTrainer |
| `scripts/train_stage_a.py` | `data/loaders.py` | Loads Stage A datasets |
| `scripts/train_stage_a.py` | `trainers/ema.py` | Uses EMA model |
| `scripts/evaluate_stage_a.py` | `evaluation/evaluator.py` | Uses Evaluator |
| `scripts/evaluate_stage_a.py` | `models/modernbert_multitask.py` | Loads model |
| `export_utility/benchmark_latency.py` | `evaluation/benchmarks.py` | Uses LatencyBenchmark |

---
