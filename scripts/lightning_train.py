#!/usr/bin/env python3
"""
Lightning AI Training Script for Stage A Multi-Task Training

This script is designed to run on Lightning AI cloud GPUs (H100, H200, A100, L40S).
It handles:
- Automatic GPU detection and configuration
- Flash Attention 2 when available
- Efficient bf16 training on Ampere/Hopper/Ada GPUs
- Weights & Biases logging (optional)
- Checkpoint saving to cloud storage

Usage on Lightning AI:
    python scripts/lightning_train.py --config configs/training/multitask/stage_a_generic.yaml
    
With wandb logging:
    python scripts/lightning_train.py --config configs/training/multitask/stage_a_generic.yaml --wandb --wandb-project modernbert-multitask

For H100/H200, use full precision bf16:
    python scripts/lightning_train.py --config configs/training/multitask/stage_a_generic.yaml --bf16
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import yaml
from transformers import AutoTokenizer

from modeling_studio.models.multitask import MultiTaskModel
from modeling_studio.data.loaders import load_datasets_from_config
from modeling_studio.trainers.multitask_trainer import MultiTaskTrainer, MultiTaskTrainingArguments


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def detect_gpu_capabilities():
    """Detect GPU capabilities and return optimal settings."""
    if not torch.cuda.is_available():
        return {
            "device": "cpu",
            "bf16": False,
            "flash_attn": False,
            "gpu_name": "CPU",
            "vram_gb": 0,
        }
    
    gpu_name = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024**3)
    
    # Check for bf16 support (Ampere+)
    compute_capability = torch.cuda.get_device_capability(0)
    bf16_supported = compute_capability[0] >= 8  # SM 8.0+ (Ampere, Hopper, Ada)
    
    # Check for Flash Attention
    flash_attn_available = False
    try:
        import flash_attn
        flash_attn_available = True
    except ImportError:
        pass
    
    return {
        "device": "cuda",
        "bf16": bf16_supported,
        "flash_attn": flash_attn_available,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "compute_capability": f"{compute_capability[0]}.{compute_capability[1]}",
    }


def get_optimal_batch_size(vram_gb: float, model_size: str = "base") -> int:
    """Get optimal batch size based on VRAM."""
    # Conservative estimates for ModernBERT-base with multi-task heads
    if vram_gb >= 80:  # H100/H200 80GB
        return 64
    elif vram_gb >= 48:  # A100 40/80GB, L40S 48GB
        return 48
    elif vram_gb >= 24:  # A10, RTX 3090/4090
        return 32
    elif vram_gb >= 16:  # T4, RTX 4080
        return 16
    elif vram_gb >= 8:   # RTX 3070/4060
        return 8
    else:
        return 4


def main():
    parser = argparse.ArgumentParser(description="Lightning AI Multi-Task Training")
    parser.add_argument("--config", type=str, required=True, help="Path to training config")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint")
    
    # GPU/Performance options
    parser.add_argument("--bf16", action="store_true", help="Force bf16 training")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 instead of bf16")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--flash-attn", action="store_true", help="Enable Flash Attention 2")
    
    # Training options
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=None, help="Warmup ratio")
    
    # Logging options
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="modernbert-multitask", help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--log-steps", type=int, default=50, help="Logging interval")
    
    # Debug options
    parser.add_argument("--debug", action="store_true", help="Debug mode with small dataset")
    parser.add_argument("--dry-run", action="store_true", help="Just print config, don't train")
    
    args = parser.parse_args()
    logger = setup_logging()
    
    # =========================================================================
    # GPU Detection
    # =========================================================================
    gpu_info = detect_gpu_capabilities()
    
    logger.info("=" * 60)
    logger.info("Lightning AI Training Setup")
    logger.info("=" * 60)
    logger.info(f"GPU: {gpu_info['gpu_name']}")
    logger.info(f"VRAM: {gpu_info['vram_gb']:.1f} GB")
    if gpu_info['device'] == 'cuda':
        logger.info(f"Compute Capability: {gpu_info['compute_capability']}")
    logger.info(f"BF16 Supported: {gpu_info['bf16']}")
    logger.info(f"Flash Attention: {gpu_info['flash_attn']}")
    logger.info("=" * 60)
    
    # =========================================================================
    # Load Configuration
    # =========================================================================
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    data_config_path = config.get("data", {}).get("config_path", "configs/data/multitask/stage_a_datasets.yaml")
    
    # =========================================================================
    # Auto-configure based on GPU
    # =========================================================================
    
    # Batch size
    if args.batch_size:
        batch_size = args.batch_size
    else:
        batch_size = get_optimal_batch_size(gpu_info['vram_gb'])
    
    # Gradient accumulation - target effective batch size of 256
    target_effective_batch = 256
    if args.gradient_accumulation:
        grad_accum = args.gradient_accumulation
    else:
        grad_accum = max(1, target_effective_batch // batch_size)
    
    effective_batch = batch_size * grad_accum
    
    # Precision
    use_bf16 = args.bf16 or (gpu_info['bf16'] and not args.fp16)
    use_fp16 = args.fp16 and not use_bf16
    
    # Flash Attention
    use_flash_attn = args.flash_attn or gpu_info['flash_attn']
    attn_implementation = "flash_attention_2" if use_flash_attn else "sdpa"
    
    logger.info("Training Configuration:")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Gradient Accumulation: {grad_accum}")
    logger.info(f"  Effective Batch Size: {effective_batch}")
    logger.info(f"  Precision: {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}")
    logger.info(f"  Attention: {attn_implementation}")
    
    # =========================================================================
    # Output Directories
    # =========================================================================
    output_dir = Path(args.output_dir or training_config.get("output_dir", "outputs/lightning-run"))
    checkpoint_dir = Path(args.checkpoint_dir or training_config.get("checkpoint_dir", "checkpoints/lightning-run"))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"  Output Dir: {output_dir}")
    logger.info(f"  Checkpoint Dir: {checkpoint_dir}")
    
    if args.dry_run:
        logger.info("Dry run - exiting without training")
        return
    
    # =========================================================================
    # Initialize Tokenizer
    # =========================================================================
    model_name = model_config.get("base_model", "answerdotai/ModernBERT-base")
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # =========================================================================
    # Load Datasets
    # =========================================================================
    logger.info(f"Loading datasets from: {data_config_path}")
    
    train_datasets = load_datasets_from_config(
        data_config_path,
        tokenizer=tokenizer,
        split="train",
        max_length=model_config.get("max_length", 512),
    )
    
    eval_datasets = load_datasets_from_config(
        data_config_path,
        tokenizer=tokenizer,
        split="validation",
        max_length=model_config.get("max_length", 512),
    )
    
    # Debug mode - use small subsets
    if args.debug:
        logger.info("Debug mode: using smaller dataset subsets")
        for task_name in train_datasets:
            if len(train_datasets[task_name]) > 1000:
                train_datasets[task_name] = train_datasets[task_name].select(range(1000))
        for task_name in eval_datasets:
            if len(eval_datasets[task_name]) > 200:
                eval_datasets[task_name] = eval_datasets[task_name].select(range(200))
    
    logger.info(f"Loaded {len(train_datasets)} training datasets")
    total_train = sum(len(ds) for ds in train_datasets.values())
    total_eval = sum(len(ds) for ds in eval_datasets.values())
    for task_name, dataset in train_datasets.items():
        logger.info(f"  - {task_name}: {len(dataset):,} samples")
    logger.info(f"Total training samples: {total_train:,}")
    logger.info(f"Total eval samples: {total_eval:,}")
    
    # =========================================================================
    # Initialize Model
    # =========================================================================
    task_configs = {}
    for task_name, dataset in train_datasets.items():
        # Get num_labels from dataset features
        if hasattr(dataset, 'features') and 'labels' in dataset.features:
            feature = dataset.features['labels']
            if hasattr(feature, 'num_classes'):
                num_labels = feature.num_classes
            elif hasattr(feature, 'feature') and hasattr(feature.feature, 'num_classes'):
                num_labels = feature.feature.num_classes
            else:
                # Infer from data
                num_labels = len(set(dataset['labels'])) if 'labels' in dataset.column_names else 2
        else:
            num_labels = 2
        
        # Determine task type
        if 'ner' in task_name:
            task_type = 'ner'
            # NER has special label count (9 for CoNLL)
            num_labels = config.get("tasks", {}).get(task_name, {}).get("num_labels", 9)
        elif 'nli' in task_name:
            task_type = 'classification'
            num_labels = 3
        elif 'embedding' in task_name:
            task_type = 'embedding'
            num_labels = 0
        elif task_name in ['emotions', 'safety_generic']:
            task_type = 'multi_label'
            num_labels = config.get("tasks", {}).get(task_name, {}).get("num_labels", 28)
        else:
            task_type = 'classification'
        
        task_configs[task_name] = {
            "task_type": task_type,
            "num_labels": num_labels,
        }
    
    # Override from config if available
    for task_name, task_cfg in config.get("tasks", {}).items():
        if task_name in task_configs:
            task_configs[task_name].update(task_cfg)
    
    logger.info("Task configurations:")
    for task_name, task_cfg in task_configs.items():
        logger.info(f"  - {task_name}: {task_cfg}")
    
    logger.info(f"Loading model: {model_name}")
    model = MultiTaskModel(
        model_name_or_path=model_name,
        task_configs=task_configs,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32,
        attn_implementation=attn_implementation,
    )
    
    # Ensure model is in correct dtype
    if use_bf16:
        model = model.to(dtype=torch.bfloat16)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # =========================================================================
    # Training Arguments
    # =========================================================================
    num_epochs = args.epochs or training_config.get("num_epochs", 3)
    learning_rate = args.lr or training_config.get("learning_rate", 2e-5)
    warmup_ratio = args.warmup_ratio or training_config.get("warmup_ratio", 0.1)
    
    # W&B setup
    report_to = ["wandb"] if args.wandb else ["tensorboard"]
    
    if args.wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_run_name:
            os.environ["WANDB_NAME"] = args.wandb_run_name
    
    training_args = MultiTaskTrainingArguments(
        output_dir=str(output_dir),
        
        # Training schedule
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,  # Can use larger for eval
        gradient_accumulation_steps=grad_accum,
        
        # Optimizer
        learning_rate=learning_rate,
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=training_config.get("lr_scheduler", "cosine"),
        
        # Precision
        bf16=use_bf16,
        fp16=use_fp16,
        
        # Logging
        logging_dir=str(output_dir / "logs"),
        logging_steps=args.log_steps,
        report_to=report_to,
        
        # Evaluation
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_avg_f1",
        greater_is_better=True,
        
        # Performance
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        
        # Resume
        resume_from_checkpoint=args.resume_from,
        
        # Misc
        seed=training_config.get("seed", 42),
        remove_unused_columns=False,
    )
    
    logger.info("=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Learning Rate: {learning_rate}")
    logger.info(f"Warmup Ratio: {warmup_ratio}")
    logger.info(f"Report To: {report_to}")
    logger.info("=" * 60)
    
    # =========================================================================
    # Initialize Trainer
    # =========================================================================
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_datasets=train_datasets,
        eval_datasets=eval_datasets,
        tokenizer=tokenizer,
        task_configs=task_configs,
    )
    
    # =========================================================================
    # Train!
    # =========================================================================
    train_result = trainer.train(resume_from_checkpoint=args.resume_from)
    
    logger.info("Training completed!")
    logger.info(f"Training loss: {train_result.training_loss:.4f}")
    logger.info(f"Training steps: {train_result.global_step}")
    
    # =========================================================================
    # Final Evaluation
    # =========================================================================
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()
    
    logger.info("Final evaluation results:")
    for key, value in sorted(eval_results.items()):
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
    
    # =========================================================================
    # Save Model
    # =========================================================================
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Save training config
    with open(output_dir / "training_config.yaml", "w") as f:
        yaml.dump({
            "model": model_config,
            "training": {
                "epochs": num_epochs,
                "batch_size": batch_size,
                "effective_batch_size": effective_batch,
                "learning_rate": learning_rate,
                "warmup_ratio": warmup_ratio,
                "precision": "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32",
                "attention": attn_implementation,
            },
            "gpu": gpu_info,
            "results": {k: float(v) if isinstance(v, float) else v for k, v in eval_results.items()},
        }, f, default_flow_style=False)
    
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
