"""
Task Head Registry for ModernBERT v3.3 Ultra.

This module provides a unified registry for v3 task heads that tracks:
- Hub routing for each task/capability
- Label schemas and counts
- Default loss weights
- Metric configurations
- Task types (classification, token-level, hierarchical, embedding)

The registry enables:
- Automatic head creation for any capability
- Task lookup by hub token
- Loss weight configuration
- Metric tracking per task

Registry Structure:
    - [EMO] Hub: emotions, sentiment, safety_generic, safety_familyos
    - [MEM] Hub: embedding (no head - raw output)
    - [REL] Hub: nli, relation
    - [TASK] Hub: intent, ingress
    - Token-level: ner_general, ner_family, temporal (full sequence)

Author: FamilyOS Team
Date: December 2025
Version: 3.3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch.nn as nn

from .heads_v3 import (
    HubAwareClassificationHead,
    HubAwareHierarchicalHead,
    HubAwareNLIHead,
    HubAwareSafetyHead,
    HubAwareTokenClassificationHead,
)


class TaskType(Enum):
    """
    Types of tasks supported in v3.

    - CLASSIFICATION: Standard classification (emotions, sentiment, intent, etc.)
    - TOKEN_CLASSIFICATION: Token-level sequence labeling (NER, temporal)
    - REGRESSION: Regression tasks (similarity scoring)
    - HIERARCHICAL: Hierarchical classification (Ekman → GoEmotions)
    - EMBEDDING: Embedding tasks (uses raw [MEM] output, no head)
    - MULTI_LABEL: Multi-label classification (safety_generic with 8 toxicity types)
    """

    CLASSIFICATION = "classification"
    TOKEN_CLASSIFICATION = "token_classification"
    REGRESSION = "regression"
    HIERARCHICAL = "hierarchical"
    EMBEDDING = "embedding"
    MULTI_LABEL = "multi_label"


@dataclass
class TaskSpec:
    """
    Complete specification for a task/capability.

    Attributes:
        name: Task identifier (e.g., "emotions", "ner_general")
        task_type: Type of task (classification, token_classification, etc.)
        hub_token: Which hub routes to this task (e.g., "[EMO]", "[REL]")
        head_class: Head class to instantiate (None for embedding)
        num_labels: Number of output labels
        label_names: List of label names
        loss_type: Type of loss ("cross_entropy", "hierarchical", "contrastive")
        loss_weight: Default weight for this task in multi-task training
        metrics: List of metrics to track (e.g., ["accuracy", "macro_f1"])
        description: Human-readable description

    Example:
        >>> emotions_spec = TaskSpec(
        ...     name="emotions",
        ...     task_type=TaskType.HIERARCHICAL,
        ...     hub_token="[EMO]",
        ...     head_class=HubAwareHierarchicalHead,
        ...     num_labels=7,
        ...     label_names=["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"],
        ...     loss_type="hierarchical",
        ...     metrics=["macro_f1", "accuracy"],
        ... )
    """

    name: str
    task_type: TaskType
    hub_token: str
    head_class: type[nn.Module] | None
    num_labels: int
    label_names: list[str]
    loss_type: str = "cross_entropy"
    loss_weight: float = 1.0
    metrics: list[str] = field(default_factory=list)
    description: str = ""


# ======================================================================
# Complete Task Registry for v3
# ======================================================================

TASK_REGISTRY_V3: dict[str, TaskSpec] = {
    # ═══════════════════════════════════════════════════════════════
    # [EMO] Hub Tasks - Emotional/Affective Understanding
    # ═══════════════════════════════════════════════════════════════
    "emotions": TaskSpec(
        name="emotions",
        task_type=TaskType.CLASSIFICATION,  # Flat classification for 44 FamilyOS emotions
        hub_token="[EMO]",
        head_class=HubAwareClassificationHead,  # Use standard classification head
        num_labels=44,  # EMOTIONS_FAMILYOS_LABELS: Core(8) + Positive(12) + Negative(10) + Family(14)
        label_names=[
            # Core Emotions (8)
            "neutral",
            "joy",
            "sadness",
            "anger",
            "fear",
            "surprise",
            "love",
            "disgust",
            # Positive Emotions (12)
            "admiration",
            "amusement",
            "approval",
            "caring",
            "curiosity",
            "desire",
            "excitement",
            "gratitude",
            "hope",
            "optimism",
            "pride",
            "tenderness",
            # Negative Emotions (10)
            "annoyance",
            "confusion",
            "disappointment",
            "disapproval",
            "embarrassment",
            "grief",
            "nervousness",
            "remorse",
            "worry",
            "emptiness",
            # Family-Specific Emotions (14)
            "nostalgia",
            "protectiveness",
            "relief",
            "contentment",
            "longing",
            "resentment",
            "guilt",
            "overwhelmed",
            "belonging",
            "abandonment",
            "jealousy",
            "trust",
            "vulnerability",
            "homesickness",
        ],
        loss_type="cross_entropy",
        loss_weight=1.0,
        metrics=["macro_f1", "accuracy"],
        description="Flat emotion classification (44 FamilyOS emotions)",
    ),
    "sentiment": TaskSpec(
        name="sentiment",
        task_type=TaskType.CLASSIFICATION,
        hub_token="[EMO]",
        head_class=HubAwareClassificationHead,
        num_labels=5,  # SENTIMENT_LABELS: 5-class scale
        label_names=["very_negative", "negative", "neutral", "positive", "very_positive"],
        loss_type="cross_entropy",
        loss_weight=0.8,
        metrics=["accuracy", "macro_f1"],
        description="5-class sentiment polarity classification",
    ),
    "safety_generic": TaskSpec(
        name="safety_generic",
        task_type=TaskType.MULTI_LABEL,  # Multi-label classification
        hub_token="[EMO]",
        head_class=HubAwareSafetyHead,
        num_labels=8,  # SAFETY_GENERIC_LABELS: 6 Jigsaw + 2 new
        label_names=[
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
            "self_harm",
            "dangerous_advice",
        ],
        loss_type="binary_cross_entropy",  # Multi-label uses BCE
        loss_weight=1.5,  # Higher weight for safety
        metrics=["recall", "precision", "f1"],
        description="Multi-label toxicity detection (8 types)",
    ),
    "safety_familyos": TaskSpec(
        name="safety_familyos",
        task_type=TaskType.CLASSIFICATION,
        hub_token="[EMO]",
        head_class=HubAwareSafetyHead,
        num_labels=4,  # SAFETY_FAMILYOS_LABELS: GREEN, AMBER, RED, CRISIS
        label_names=["GREEN", "AMBER", "RED", "CRISIS"],
        loss_type="cross_entropy",
        loss_weight=2.0,  # Highest weight - CRISIS recall is critical
        metrics=["recall", "precision", "f1", "crisis_recall"],
        description="FamilyOS safety policy bands (GREEN to CRISIS)",
    ),
    # ═══════════════════════════════════════════════════════════════
    # [MEM] Hub Tasks - Memory/Embedding
    # ═══════════════════════════════════════════════════════════════
    "embedding": TaskSpec(
        name="embedding",
        task_type=TaskType.EMBEDDING,
        hub_token="[MEM]",
        head_class=None,  # No head - uses raw [MEM] output
        num_labels=0,
        label_names=[],
        loss_type="contrastive",
        loss_weight=1.0,
        metrics=["recall@10", "mrr"],
        description="Sentence embedding for retrieval/similarity",
    ),
    # ═══════════════════════════════════════════════════════════════
    # [REL] Hub Tasks - Relationship Understanding
    # ═══════════════════════════════════════════════════════════════
    "nli": TaskSpec(
        name="nli",
        task_type=TaskType.CLASSIFICATION,
        hub_token="[REL]",
        head_class=HubAwareNLIHead,
        num_labels=3,
        label_names=["entailment", "neutral", "contradiction"],
        loss_type="cross_entropy",
        loss_weight=1.0,
        metrics=["accuracy"],
        description="Natural Language Inference",
    ),
    "relation": TaskSpec(
        name="relation",
        task_type=TaskType.CLASSIFICATION,
        hub_token="[REL]",
        head_class=HubAwareClassificationHead,
        num_labels=15,  # RELATION_LABELS: no_relation + 14 relations
        label_names=[
            "no_relation",
            "parent_of",
            "child_of",
            "spouse_of",
            "sibling_of",
            "grandparent_of",
            "grandchild_of",
            "aunt_uncle_of",
            "niece_nephew_of",
            "cousin_of",
            "pet_of",
            "friend_of",
            "colleague_of",
            "lives_at",
            "owns",
        ],
        loss_type="cross_entropy",
        loss_weight=1.2,
        metrics=["macro_f1", "accuracy"],
        description="Family relationship extraction (15 relations)",
    ),
    # ═══════════════════════════════════════════════════════════════
    # [TASK] Hub Tasks - Intent/Action Understanding
    # ═══════════════════════════════════════════════════════════════
    "intent": TaskSpec(
        name="intent",
        task_type=TaskType.CLASSIFICATION,
        hub_token="[TASK]",
        head_class=HubAwareClassificationHead,
        num_labels=8,  # INTENT_LABELS: 8 FamilyOS intents
        label_names=[
            "log_memory",
            "query_memory",
            "set_reminder",
            "express_feeling",
            "seek_advice",
            "share_news",
            "reflect",
            "other",
        ],
        loss_type="cross_entropy",
        loss_weight=1.0,
        metrics=["accuracy", "macro_f1"],
        description="FamilyOS user intent classification (8 intents)",
    ),
    "ingress": TaskSpec(
        name="ingress",
        task_type=TaskType.CLASSIFICATION,
        hub_token="[TASK]",
        head_class=HubAwareClassificationHead,
        num_labels=12,  # INGRESS_LABELS: 7 original + 5 extended
        label_names=[
            "DIARY",
            "TASK",
            "HEALTH",
            "FINANCE",
            "RELATIONSHIP",
            "WORK",
            "META",
            "MEMORY",
            "PLANNING",
            "CELEBRATION",
            "CONCERN",
            "GRATITUDE",
        ],
        loss_type="cross_entropy",
        loss_weight=1.0,
        metrics=["accuracy"],
        description="Extended domain classification (12 domains)",
    ),
    # ═══════════════════════════════════════════════════════════════
    # Token-Level Tasks (No Hub Pooling)
    # ═══════════════════════════════════════════════════════════════
    "ner_general": TaskSpec(
        name="ner_general",
        task_type=TaskType.TOKEN_CLASSIFICATION,
        hub_token="[CLS]",  # Not used - full sequence
        head_class=HubAwareTokenClassificationHead,
        num_labels=17,  # NER_GENERAL_LABELS: 17 BIO tags
        label_names=[
            "O",
            "B-PER",
            "I-PER",
            "B-ORG",
            "I-ORG",
            "B-LOC",
            "I-LOC",
            "B-MISC",
            "I-MISC",
            "B-DATE",
            "I-DATE",
            "B-TIME",
            "I-TIME",
            "B-EVENT",
            "I-EVENT",
            "B-PRODUCT",
            "I-PRODUCT",
        ],
        loss_type="cross_entropy",
        loss_weight=1.0,
        metrics=["entity_f1", "precision", "recall"],
        description="General named entity recognition (17 BIO tags)",
    ),
    "ner_family": TaskSpec(
        name="ner_family",
        task_type=TaskType.TOKEN_CLASSIFICATION,
        hub_token="[CLS]",
        head_class=HubAwareTokenClassificationHead,
        num_labels=21,  # NER_FAMILY_LABELS: 21 BIO tags
        label_names=[
            "O",
            "B-PERSON",
            "I-PERSON",
            "B-KINSHIP",
            "I-KINSHIP",
            "B-NICKNAME",
            "I-NICKNAME",
            "B-PET",
            "I-PET",
            "B-HOME_LOC",
            "I-HOME_LOC",
            "B-FAMILY_EVENT",
            "I-FAMILY_EVENT",
            "B-ROUTINE",
            "I-ROUTINE",
            "B-TRADITION",
            "I-TRADITION",
            "B-MILESTONE",
            "I-MILESTONE",
            "B-HEIRLOOM",
            "I-HEIRLOOM",
        ],
        loss_type="cross_entropy",
        loss_weight=1.2,
        metrics=["entity_f1", "precision", "recall"],
        description="Family-specific entity recognition (21 BIO tags)",
    ),
    "temporal": TaskSpec(
        name="temporal",
        task_type=TaskType.TOKEN_CLASSIFICATION,
        hub_token="[CLS]",
        head_class=HubAwareTokenClassificationHead,
        num_labels=13,  # TEMPORAL_LABELS: 13 BIO tags
        label_names=[
            "O",
            "B-DATE_ABS",
            "I-DATE_ABS",
            "B-DATE_REL",
            "I-DATE_REL",
            "B-TIME",
            "I-TIME",
            "B-DURATION",
            "I-DURATION",
            "B-FREQUENCY",
            "I-FREQUENCY",
            "B-AGE",
            "I-AGE",
        ],
        loss_type="cross_entropy",
        loss_weight=1.0,
        metrics=["entity_f1"],
        description="Temporal expression extraction (13 BIO tags)",
    ),
}


# ======================================================================
# Task Registry Class
# ======================================================================


class TaskRegistry:
    """
    Registry for managing v3 task configurations.

    Provides centralized access to task specifications, head creation,
    hub routing information, and metric configuration for all 12 capabilities.

    Features:
        - Task lookup by name
        - Head creation with automatic configuration
        - Hub routing queries
        - Loss weight management
        - Metric tracking

    Example:
        >>> registry = TaskRegistry()
        >>> emotions_spec = registry.get_task("emotions")
        >>> print(emotions_spec.hub_token)  # "[EMO]"
        >>> head = registry.create_head("emotions", hidden_size=768)
        >>> emo_tasks = registry.get_tasks_by_hub("[EMO]")
        >>> print(emo_tasks)  # ["emotions", "sentiment", "safety_generic", "safety_familyos"]
    """

    def __init__(self, custom_registry: dict[str, TaskSpec] | None = None):
        """
        Initialize task registry.

        Args:
            custom_registry: Optional custom task specifications to add/override
        """
        self.registry = TASK_REGISTRY_V3.copy()
        if custom_registry:
            self.registry.update(custom_registry)

    def get_task(self, name: str) -> TaskSpec:
        """
        Get task specification by name.

        Args:
            name: Task name (e.g., "emotions", "ner_general")

        Returns:
            TaskSpec for the task

        Raises:
            ValueError: If task name is not registered
        """
        if name not in self.registry:
            raise ValueError(f"Unknown task: {name}. Available: {list(self.registry.keys())}")
        return self.registry[name]

    def get_all_tasks(self) -> list[str]:
        """
        Get all registered task names.

        Returns:
            List of all task names
        """
        return list(self.registry.keys())

    def get_tasks_by_hub(self, hub_token: str) -> list[str]:
        """
        Get all tasks routed through a specific hub token.

        Args:
            hub_token: Hub token (e.g., "[EMO]", "[REL]", "[TASK]", "[MEM]")

        Returns:
            List of task names using this hub token
        """
        return [name for name, spec in self.registry.items() if spec.hub_token == hub_token]

    def get_hub_routed_tasks(self) -> list[str]:
        """
        Get tasks that use hub token pooling (excludes token-level and embedding).

        Token-level tasks (NER, temporal) use full sequence representations
        instead of hub pooling. Embedding task uses raw [MEM] output.

        Returns:
            List of hub-routed task names
        """
        return [
            name
            for name, spec in self.registry.items()
            if spec.task_type != TaskType.TOKEN_CLASSIFICATION
            and spec.task_type != TaskType.EMBEDDING
        ]

    def get_token_level_tasks(self) -> list[str]:
        """
        Get tasks that use token-level classification.

        These tasks receive full sequence representations and predict
        labels for each token (NER, temporal extraction).

        Returns:
            List of token-level task names
        """
        return [
            name
            for name, spec in self.registry.items()
            if spec.task_type == TaskType.TOKEN_CLASSIFICATION
        ]

    def create_head(
        self,
        task_name: str,
        hidden_size: int = 768,
        **kwargs,
    ) -> nn.Module | None:
        """
        Create head for a task with automatic configuration.

        Args:
            task_name: Name of the task
            hidden_size: Model hidden dimension (default: 768)
            **kwargs: Additional head-specific arguments

        Returns:
            Instantiated head module (None for embedding task)

        Example:
            >>> registry = TaskRegistry()
            >>> emotions_head = registry.create_head("emotions", hidden_size=768)
            >>> ner_head = registry.create_head("ner_general", hidden_size=768, dropout=0.2)
        """
        spec = self.get_task(task_name)

        if spec.head_class is None:
            return None  # Embedding task - no head

        # Build constructor arguments based on head type
        from .heads_v3 import (
            HubAwareHierarchicalHead,
            HubAwareTokenClassificationHead,
        )

        head_kwargs = {"hidden_size": hidden_size, **kwargs}

        # Handle special cases for different head signatures
        if spec.head_class == HubAwareHierarchicalHead:
            # Hierarchical head uses primary_labels/secondary_labels
            head_kwargs["primary_labels"] = spec.num_labels
            head_kwargs["secondary_labels"] = spec.num_labels * 4  # GoEmotions
            head_kwargs["hub_token"] = spec.hub_token
        elif spec.head_class == HubAwareTokenClassificationHead:
            # Token classification head doesn't take hub_token
            head_kwargs["num_labels"] = spec.num_labels
        else:
            # Standard heads (classification, safety, NLI)
            head_kwargs["num_labels"] = spec.num_labels
            head_kwargs["hub_token"] = spec.hub_token

        return spec.head_class(**head_kwargs)

    def create_all_heads(
        self,
        hidden_size: int = 768,
        tasks: list[str] | None = None,
    ) -> nn.ModuleDict:
        """
        Create heads for multiple tasks.

        Args:
            hidden_size: Model hidden dimension (default: 768)
            tasks: List of task names (default: all tasks)

        Returns:
            ModuleDict of task_name -> head module

        Example:
            >>> registry = TaskRegistry()
            >>> heads = registry.create_all_heads(hidden_size=768)
            >>> print(len(heads))  # 11 (all except embedding)
            >>> emo_heads = registry.create_all_heads(tasks=["emotions", "sentiment"])
        """
        if tasks is None:
            tasks = self.get_all_tasks()

        heads = nn.ModuleDict()
        for task in tasks:
            head = self.create_head(task, hidden_size)
            if head is not None:
                heads[task] = head

        return heads

    def get_loss_weights(self) -> dict[str, float]:
        """
        Get default loss weights for all tasks.

        Returns:
            Dict of task_name -> loss_weight
        """
        return {name: spec.loss_weight for name, spec in self.registry.items()}

    def get_metrics(self, task_name: str) -> list[str]:
        """
        Get metrics for a task.

        Args:
            task_name: Name of the task

        Returns:
            List of metric names
        """
        return self.get_task(task_name).metrics

    def print_registry(self) -> None:
        """
        Print registry summary organized by hub.

        Displays all tasks grouped by their hub token with type,
        label count, and loss weight information.
        """
        print("\n" + "=" * 80)
        print("📋 v3 Task Registry")
        print("=" * 80)

        # Group by hub token
        by_hub: dict[str, list[tuple[str, TaskSpec]]] = {}
        for name, spec in self.registry.items():
            hub = spec.hub_token
            if hub not in by_hub:
                by_hub[hub] = []
            by_hub[hub].append((name, spec))

        # Print each hub group
        for hub, tasks in sorted(by_hub.items()):
            print(f"\n  {hub} Hub:")
            for name, spec in tasks:
                type_str = spec.task_type.value[:12]
                print(
                    f"    {name:<18} {type_str:<15} "
                    f"labels={spec.num_labels:<3} weight={spec.loss_weight}"
                )

        print("\n" + "=" * 80)

    def extra_repr(self) -> str:
        return f"tasks={len(self.registry)}"


# ======================================================================
# Singleton Registry Instance
# ======================================================================

_registry: TaskRegistry | None = None


def get_registry() -> TaskRegistry:
    """
    Get global task registry singleton.

    Returns:
        Singleton TaskRegistry instance

    Example:
        >>> registry = get_registry()
        >>> emotions_spec = registry.get_task("emotions")
        >>> registry.print_registry()
    """
    global _registry
    if _registry is None:
        _registry = TaskRegistry()
    return _registry
