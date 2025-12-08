#!/usr/bin/env python
"""
Debug script to verify progressive regularization is working.
Runs 10 mini-epochs with tiny data to check feature toggling.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments
import numpy as np

from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel
from modeling_studio.data.labels import Capability
from modeling_studio.trainers.multitask_trainer import MultiTaskTrainer, MultiTaskTrainingArguments

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_dummy_data(n_samples: int = 150) -> dict[str, Dataset]:
    """Create tiny dummy datasets for each task."""

    # Dummy texts
    texts = [f"This is sample text number {i} for testing." for i in range(n_samples)]

    datasets = {}

    # Sentiment (5 classes)
    datasets["sentiment"] = Dataset.from_dict(
        {
            "text": texts,
            "labels": np.random.randint(0, 5, n_samples).tolist(),
            "task": ["sentiment"] * n_samples,
        }
    )

    # Emotions (multi-label, 44 classes)
    emotion_labels = []
    for _ in range(n_samples):
        vec = [0] * 44
        # Random 1-3 emotions per sample
        for idx in np.random.choice(44, size=np.random.randint(1, 4), replace=False):
            vec[idx] = 1
        emotion_labels.append(vec)
    datasets["emotions"] = Dataset.from_dict(
        {
            "text": texts,
            "labels": emotion_labels,
            "task": ["emotions"] * n_samples,
        }
    )

    # Safety generic (multi-label, 8 classes)
    safety_labels = []
    for _ in range(n_samples):
        vec = [0] * 8
        if np.random.random() > 0.5:  # 50% toxic
            for idx in np.random.choice(8, size=np.random.randint(1, 3), replace=False):
                vec[idx] = 1
        safety_labels.append(vec)
    datasets["safety_generic"] = Dataset.from_dict(
        {
            "text": texts,
            "labels": safety_labels,
            "task": ["safety_generic"] * n_samples,
        }
    )

    # NLI (3 classes)
    datasets["nli"] = Dataset.from_dict(
        {
            "text": texts,
            "labels": np.random.randint(0, 3, n_samples).tolist(),
            "task": ["nli"] * n_samples,
        }
    )

    return datasets


def main():
    print("=" * 60)
    print("DEBUG: Progressive Regularization Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create model
    print("\n1. Creating model...")
    model = ModernBertMultiTaskModel.from_pretrained(
        "answerdotai/ModernBERT-base",
        capabilities=[
            Capability.SENTIMENT,
            Capability.EMOTIONS,
            Capability.SAFETY_GENERIC,
            Capability.NLI,
        ],
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    # Create dummy data
    print("\n2. Creating dummy data (150 samples per task)...")
    datasets = create_dummy_data(150)

    # Tokenize
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128, padding=False)

    for name, ds in datasets.items():
        datasets[name] = ds.map(tokenize, batched=True)
        print(f"   {name}: {len(ds)} samples")

    # Training args with progressive regularization
    print("\n3. Setting up training with progressive regularization...")

    # Calculate steps: 150 samples, batch=16, 10 epochs = ~94 steps per epoch
    # We want to see epochs 1-3 (no features), 4-6 (rdrop+mixup), 7-10 (all)

    training_args = MultiTaskTrainingArguments(
        output_dir="outputs/debug_progressive",
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        # Progressive regularization
        progressive_regularization=True,
        rdrop_start_epoch=4,
        mixup_start_epoch=4,
        adversarial_start_epoch=7,
        # Enable the features (they'll be toggled by epoch)
        use_rdrop=True,
        rdrop_alpha=0.7,
        use_mixup=True,
        mixup_alpha=0.4,
        mixup_prob=0.8,
        use_adversarial=True,
        adversarial_type="fgm",
        adversarial_epsilon=1.0,
        # Fast settings
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        bf16=device == "cuda",
        dataloader_num_workers=0,
        report_to=[],
    )

    task_weights = {"sentiment": 1.0, "emotions": 1.5, "safety_generic": 2.0, "nli": 1.0}

    # Create trainer
    from modeling_studio.trainers.collators import MultiTaskCollator

    collator = MultiTaskCollator(
        tokenizer=tokenizer,
        capability_configs={
            Capability.SENTIMENT: {"num_labels": 5},
            Capability.EMOTIONS: {"num_labels": 44, "problem_type": "multi_label_classification"},
            Capability.SAFETY_GENERIC: {
                "num_labels": 8,
                "problem_type": "multi_label_classification",
            },
            Capability.NLI: {"num_labels": 3},
        },
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_datasets=datasets,
        eval_datasets=datasets,  # Use same for simplicity
        processing_class=tokenizer,
        data_collator=collator,
    )

    # Check initial state
    print("\n4. Checking initial feature state...")
    print(f"   progressive_regularization: {trainer.progressive_regularization}")
    print(f"   rdrop_loss: {trainer.rdrop_loss}")
    print(f"   mixup: {trainer.mixup}")
    print(f"   adversarial: {trainer.adversarial}")
    print(f"   _rdrop_loss_ref: {getattr(trainer, '_rdrop_loss_ref', 'NOT SET')}")
    print(f"   _mixup_ref: {getattr(trainer, '_mixup_ref', 'NOT SET')}")
    print(f"   _adversarial_ref: {getattr(trainer, '_adversarial_ref', 'NOT SET')}")

    # Run training
    print("\n5. Starting training (10 epochs)...")
    print("   Watch for '[Epoch X] <Feature> ENABLED' messages")
    print("=" * 60)

    trainer.train()

    print("\n" + "=" * 60)
    print("6. Final feature state:")
    print(f"   rdrop_loss: {trainer.rdrop_loss}")
    print(f"   mixup: {trainer.mixup}")
    print(f"   adversarial: {trainer.adversarial}")
    print("=" * 60)

    print("\n✅ Debug complete!")
    print("\nExpected behavior:")
    print("  - Epochs 1-3: All features = None")
    print("  - Epoch 4: R-Drop + Mixup ENABLED")
    print("  - Epoch 7: Adversarial ENABLED")


if __name__ == "__main__":
    main()
