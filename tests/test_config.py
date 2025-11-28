"""Test configuration module."""

from modeling_studio.config import Config, ModelConfig, TrainingConfig


class TestConfig:
    """Test configuration classes."""

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()
        assert config.type == "encoder"
        assert config.name_or_path == "bert-base-uncased"
        assert config.load_in_4bit is False

    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()
        assert config.learning_rate == 2e-5
        assert config.num_train_epochs == 3
        assert config.seed == 42

    def test_config_from_dict(self, sample_config):
        """Test Config creation from dictionary."""
        config = Config.from_dict(sample_config)
        assert config.model.name_or_path == "bert-base-uncased"
        assert config.training.learning_rate == 2e-5
