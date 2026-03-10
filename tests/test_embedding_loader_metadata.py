"""Regression tests for embedding-head metadata-driven reconstruction."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from transformers import BertConfig


def _tiny_config() -> BertConfig:
    """Create a tiny config for head-construction tests."""
    return BertConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
    )


def _load_release_model_class():
    """Load the patched release model class directly from the workspace file."""
    module_path = Path(__file__).resolve().parents[1] / "familyos_ultrabert" / "models" / "modernbert_multitask.py"
    module_name = "tests_release_modernbert_multitask"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.ModernBertMultiTaskModel


def test_release_model_uses_attentive_embedding_metadata() -> None:
    """Release model should honor embedding head metadata at construction time."""
    from familyos_ultrabert.data.labels import Capability

    ModernBertMultiTaskModel = _load_release_model_class()

    model = ModernBertMultiTaskModel(
        config=_tiny_config(),
        capabilities=[Capability.EMBEDDING],
        _embedding_config={
            "pooling": "attentive",
            "output_dim": 32,
            "normalize": True,
        },
    )

    head = model.get_head(Capability.EMBEDDING)
    assert head.pooling == "attentive"
    assert head.cross_attn is not None
    assert head.output_dim == 32


def test_studio_model_uses_attentive_embedding_metadata() -> None:
    """Training-side model should honor embedding head metadata at construction time."""
    from modeling_studio.data.labels import Capability
    from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

    model = ModernBertMultiTaskModel(
        config=_tiny_config(),
        capabilities=[Capability.EMBEDDING],
        _embedding_config={
            "pooling": "attentive",
            "output_dim": 32,
            "normalize": True,
        },
    )

    head = model.get_head(Capability.EMBEDDING)
    assert head.pooling == "attentive"
    assert head.cross_attn is not None
    assert head.output_dim == 32


def test_release_model_save_pretrained_writes_embedding_metadata(tmp_path) -> None:
    """Release model should persist embedding metadata for future reloads."""
    from familyos_ultrabert.data.labels import Capability

    ModernBertMultiTaskModel = _load_release_model_class()

    model = ModernBertMultiTaskModel(
        config=_tiny_config(),
        capabilities=[Capability.EMBEDDING],
        _embedding_config={
            "pooling": "attentive",
            "output_dim": 32,
            "normalize": True,
        },
    )

    model.save_pretrained(str(tmp_path))

    metadata = json.loads((tmp_path / "embedding_metadata.json").read_text())
    assert metadata["head_info"]["embedding"]["pooling"] == "attentive"
    assert metadata["head_info"]["embedding"]["output_dim"] == 32
