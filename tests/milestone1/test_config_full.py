"""
Issue 1.1.3: Tests for `modeling_studio/config.py`

Tests:
- test_model_config_defaults - Verify `ModelConfig` default values
- test_training_config_defaults - Verify `TrainingConfig` defaults
- test_data_config_defaults - Verify `DataConfig` defaults
- test_peft_config_defaults - Verify `PEFTConfig` defaults
- test_config_from_yaml - Load config from YAML file
- test_config_from_yaml_missing_file - Verify `FileNotFoundError` raised
- test_config_from_dict - Create config from dictionary
- test_config_to_dict - Convert config to dictionary
- test_config_save - Save config to YAML
- test_config_partial_yaml - Load YAML with only some sections
"""

import tempfile
from pathlib import Path

import pytest
import yaml


class TestModelConfig:
    """Test ModelConfig defaults and functionality."""

    def test_model_config_defaults(self):
        """Verify `ModelConfig` default values (type='encoder', name_or_path, torch_dtype)."""
        from modeling_studio.config import ModelConfig

        config = ModelConfig()

        assert config.type == "encoder"
        assert config.name_or_path == "bert-base-uncased"
        assert config.torch_dtype == "float32"
        assert config.architecture is None
        assert config.load_in_8bit is False
        assert config.load_in_4bit is False
        assert config.trust_remote_code is False
        assert config.use_flash_attention_2 is False
        assert config.quantization is None

    def test_model_config_custom_values(self):
        """Test ModelConfig with custom values."""
        from modeling_studio.config import ModelConfig

        config = ModelConfig(
            type="decoder",
            name_or_path="gpt2",
            torch_dtype="bfloat16",
            load_in_8bit=True,
        )

        assert config.type == "decoder"
        assert config.name_or_path == "gpt2"
        assert config.torch_dtype == "bfloat16"
        assert config.load_in_8bit is True


class TestTrainingConfig:
    """Test TrainingConfig defaults and functionality."""

    def test_training_config_defaults(self):
        """Verify `TrainingConfig` defaults (learning_rate=2e-5, num_train_epochs=3, etc.)."""
        from modeling_studio.config import TrainingConfig

        config = TrainingConfig()

        assert config.learning_rate == 2e-5
        assert config.weight_decay == 0.01
        assert config.num_train_epochs == 3
        assert config.max_steps == -1
        assert config.per_device_train_batch_size == 8
        assert config.per_device_eval_batch_size == 16
        assert config.gradient_accumulation_steps == 1
        assert config.gradient_checkpointing is False
        assert config.fp16 is False
        assert config.bf16 is False
        assert config.optim == "adamw_torch"
        assert config.lr_scheduler_type == "linear"
        assert config.warmup_ratio == 0.1
        assert config.warmup_steps == 0
        assert config.max_grad_norm == 1.0
        assert config.eval_strategy == "steps"
        assert config.eval_steps == 500
        assert config.save_strategy == "steps"
        assert config.save_steps == 500
        assert config.save_total_limit == 3
        assert config.logging_steps == 100
        assert config.seed == 42


class TestDataConfig:
    """Test DataConfig defaults and functionality."""

    def test_data_config_defaults(self):
        """Verify `DataConfig` defaults (max_length=512, truncation=True)."""
        from modeling_studio.config import DataConfig

        config = DataConfig()

        assert config.source == "huggingface"
        assert config.dataset_name is None
        assert config.data_dir is None
        assert config.train_split == "train"
        assert config.validation_split == "validation"
        assert config.max_length == 512
        assert config.truncation is True
        assert config.padding == "max_length"
        assert config.streaming is False
        assert config.preprocessing_num_workers == 4


class TestPEFTConfig:
    """Test PEFTConfig defaults and functionality."""

    def test_peft_config_defaults(self):
        """Verify `PEFTConfig` defaults (method='lora', r=16, lora_alpha=32)."""
        from modeling_studio.config import PEFTConfig

        config = PEFTConfig()

        assert config.method == "lora"
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05
        assert config.bias == "none"
        assert config.target_modules is None
        assert config.task_type == "CAUSAL_LM"


class TestConfig:
    """Test main Config class functionality."""

    def test_config_defaults(self):
        """Test Config has all sub-configs with defaults."""
        from modeling_studio.config import Config

        config = Config()

        assert config.model is not None
        assert config.training is not None
        assert config.data is not None
        assert config.peft is None  # Optional, None by default
        assert config.output_dir == "outputs"

    def test_config_from_yaml(self, tmp_path):
        """Load config from YAML file, verify all fields populated."""
        from modeling_studio.config import Config

        yaml_content = {
            "model": {
                "type": "encoder",
                "name_or_path": "roberta-base",
                "torch_dtype": "float16",
            },
            "training": {
                "learning_rate": 3e-5,
                "num_train_epochs": 5,
            },
            "data": {
                "max_length": 256,
            },
            "output_dir": "custom_outputs",
        }

        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = Config.from_yaml(yaml_path)

        assert config.model.name_or_path == "roberta-base"
        assert config.model.torch_dtype == "float16"
        assert config.training.learning_rate == 3e-5
        assert config.training.num_train_epochs == 5
        assert config.data.max_length == 256
        assert config.output_dir == "custom_outputs"

    def test_config_from_yaml_missing_file(self):
        """Verify `FileNotFoundError` raised for missing file."""
        from modeling_studio.config import Config

        with pytest.raises(FileNotFoundError):
            Config.from_yaml("/nonexistent/path/config.yaml")

    def test_config_from_dict(self):
        """Create config from dictionary."""
        from modeling_studio.config import Config

        config_dict = {
            "model": {
                "type": "encoder",
                "name_or_path": "distilbert-base-uncased",
            },
            "training": {
                "learning_rate": 1e-4,
                "num_train_epochs": 2,
            },
        }

        config = Config.from_dict(config_dict)

        assert config.model.name_or_path == "distilbert-base-uncased"
        assert config.training.learning_rate == 1e-4
        assert config.training.num_train_epochs == 2
        # Defaults should be used for missing fields
        assert config.data.max_length == 512  # default

    def test_config_to_dict(self):
        """Convert config to dictionary, verify round-trip."""
        from modeling_studio.config import Config

        config = Config()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "model" in config_dict
        assert "training" in config_dict
        assert "data" in config_dict
        assert config_dict["model"]["type"] == "encoder"
        assert config_dict["training"]["learning_rate"] == 2e-5

    def test_config_save(self, tmp_path):
        """Save config to YAML, verify file created."""
        from modeling_studio.config import Config

        config = Config()
        save_path = tmp_path / "saved_config.yaml"

        config.save(save_path)

        assert save_path.exists()

        # Verify content can be loaded back
        with open(save_path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["model"]["type"] == "encoder"
        assert loaded["training"]["learning_rate"] == 2e-5

    def test_config_partial_yaml(self, tmp_path):
        """Load YAML with only some sections, verify defaults used."""
        from modeling_studio.config import Config

        # Only specify model section
        yaml_content = {
            "model": {
                "name_or_path": "albert-base-v2",
            },
        }

        yaml_path = tmp_path / "partial_config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = Config.from_yaml(yaml_path)

        # Specified value
        assert config.model.name_or_path == "albert-base-v2"
        # Defaults for unspecified sections
        assert config.training.learning_rate == 2e-5
        assert config.data.max_length == 512

    def test_config_round_trip(self, tmp_path):
        """Test save and load round-trip preserves values."""
        from modeling_studio.config import Config, PEFTConfig

        original = Config()
        original.model.name_or_path = "custom-model"
        original.training.learning_rate = 5e-5
        # Ensure peft is not None for clean round-trip
        original.peft = PEFTConfig()

        save_path = tmp_path / "roundtrip.yaml"
        original.save(save_path)

        loaded = Config.from_yaml(save_path)

        assert loaded.model.name_or_path == original.model.name_or_path
        assert loaded.training.learning_rate == original.training.learning_rate
        assert loaded.peft is not None
        assert loaded.peft.method == original.peft.method

    def test_config_save_creates_parent_dirs(self, tmp_path):
        """Verify save creates parent directories if needed."""
        from modeling_studio.config import Config

        config = Config()
        save_path = tmp_path / "nested" / "dir" / "config.yaml"

        config.save(save_path)

        assert save_path.exists()
        assert save_path.parent.exists()
