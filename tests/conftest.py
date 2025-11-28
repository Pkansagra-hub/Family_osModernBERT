# Test configuration
import pytest


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model": {
            "type": "encoder",
            "name_or_path": "bert-base-uncased",
        },
        "training": {
            "learning_rate": 2e-5,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 8,
        },
    }
