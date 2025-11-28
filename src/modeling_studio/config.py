"""
Configuration management for Modeling Studio.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf


@dataclass
class ModelConfig:
    """Model configuration."""

    type: str = "encoder"  # encoder, decoder, encoder_decoder
    name_or_path: str = "bert-base-uncased"
    architecture: str | None = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    torch_dtype: str = "float32"
    trust_remote_code: bool = False
    use_flash_attention_2: bool = False
    quantization: dict[str, Any] | None = None


@dataclass
class TrainingConfig:
    """Training configuration."""

    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 3
    max_steps: int = -1
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    fp16: bool = False
    bf16: bool = False
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 100
    seed: int = 42


@dataclass
class DataConfig:
    """Data configuration."""

    source: str = "huggingface"
    dataset_name: str | None = None
    data_dir: str | None = None
    train_split: str = "train"
    validation_split: str = "validation"
    max_length: int = 512
    truncation: bool = True
    padding: str = "max_length"
    streaming: bool = False
    preprocessing_num_workers: int = 4


@dataclass
class PEFTConfig:
    """PEFT/LoRA configuration."""

    method: str = "lora"
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] | None = None
    task_type: str = "CAUSAL_LM"


@dataclass
class Config:
    """Main configuration class."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    peft: PEFTConfig | None = None
    output_dir: str = "outputs"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        omega_conf = OmegaConf.create(config_dict)
        return cls._from_omega(omega_conf)

    @classmethod
    def _from_omega(cls, omega_conf: DictConfig) -> "Config":
        """Create configuration from OmegaConf."""
        config = cls()

        if "model" in omega_conf:
            config.model = ModelConfig(**OmegaConf.to_container(omega_conf.model))
        if "training" in omega_conf:
            config.training = TrainingConfig(**OmegaConf.to_container(omega_conf.training))
        if "data" in omega_conf:
            config.data = DataConfig(**OmegaConf.to_container(omega_conf.data))
        if "peft" in omega_conf:
            config.peft = PEFTConfig(**OmegaConf.to_container(omega_conf.peft))
        if "output_dir" in omega_conf:
            config.output_dir = omega_conf.output_dir

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return OmegaConf.to_container(OmegaConf.structured(self))

    def save(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
