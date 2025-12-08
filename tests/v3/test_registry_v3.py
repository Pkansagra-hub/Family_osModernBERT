"""
Tests for Task Head Registry (registry_v3.py).

This test suite validates all acceptance criteria for Issue 3.2.3:
1. All 12 capabilities registered with complete specifications
2. TaskSpec includes hub_token, head_class, labels, metrics
3. get_tasks_by_hub() returns correct tasks per hub
4. create_head() instantiates correct head class
5. Loss weights configured (safety tasks have higher weight)
6. Token-level tasks correctly identified
7. print_registry() shows organized summary

Author: FamilyOS Team
Date: December 2025
Version: 3.3
"""

# ruff: noqa: F401
import torch

from modeling_studio.models.heads_v3 import (
    HubAwareClassificationHead,
    HubAwareHierarchicalHead,
    HubAwareNLIHead,
    HubAwareSafetyHead,
    HubAwareTokenClassificationHead,
)
from modeling_studio.models.registry_v3 import (
    TASK_REGISTRY_V3,
    TaskRegistry,
    TaskSpec,
    TaskType,
    get_registry,
)


# ======================================================================
# Test TaskSpec Dataclass
# ======================================================================


class TestTaskSpec:
    """Test TaskSpec dataclass."""

    def test_taskspec_creation(self):
        """Test creating a TaskSpec."""
        spec = TaskSpec(
            name="emotions",
            task_type=TaskType.HIERARCHICAL,
            hub_token="[EMO]",
            head_class=HubAwareHierarchicalHead,
            num_labels=7,
            label_names=["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"],
            loss_type="hierarchical",
            loss_weight=1.0,
            metrics=["macro_f1", "accuracy"],
            description="Hierarchical emotion classification",
        )

        assert spec.name == "emotions"
        assert spec.task_type == TaskType.HIERARCHICAL
        assert spec.hub_token == "[EMO]"
        assert spec.head_class == HubAwareHierarchicalHead
        assert spec.num_labels == 7
        assert len(spec.label_names) == 7
        assert spec.loss_type == "hierarchical"

        assert spec.loss_weight == 1.0
        assert len(spec.metrics) == 2

    def test_taskspec_defaults(self):
        """Test TaskSpec default values."""
        spec = TaskSpec(
            name="test",
            task_type=TaskType.CLASSIFICATION,
            hub_token="[EMO]",
            head_class=HubAwareClassificationHead,
            num_labels=2,
            label_names=["a", "b"],
        )

        assert spec.loss_type == "cross_entropy"
        assert spec.loss_weight == 1.0
        assert spec.metrics == []
        assert spec.description == ""


# ======================================================================
# Test TASK_REGISTRY_V3 Contents
# ======================================================================


class TestTaskRegistryContents:
    """Test the complete task registry."""

    def test_all_12_capabilities_registered(self):
        """AC1: All 12 capabilities registered with complete specifications."""
        expected_tasks = {
            "emotions",
            "sentiment",
            "safety_generic",
            "safety_familyos",
            "embedding",
            "nli",
            "relation",
            "intent",
            "ingress",
            "ner_general",
            "ner_family",
            "temporal",
        }

        assert set(TASK_REGISTRY_V3.keys()) == expected_tasks
        assert len(TASK_REGISTRY_V3) == 12

    def test_taskspec_completeness(self):
        """AC2: TaskSpec includes hub_token, head_class, labels, metrics."""
        for name, spec in TASK_REGISTRY_V3.items():
            assert isinstance(spec, TaskSpec)
            assert spec.name == name
            assert isinstance(spec.task_type, TaskType)
            assert spec.hub_token in ["[EMO]", "[MEM]", "[REL]", "[TASK]", "[CLS]"]
            # head_class can be None for embedding
            assert spec.num_labels >= 0
            assert isinstance(spec.label_names, list)
            assert spec.loss_type in [
                "cross_entropy",
                "hierarchical",
                "contrastive",
                "binary_cross_entropy",
            ]
            assert spec.loss_weight > 0
            assert isinstance(spec.metrics, list)
            assert isinstance(spec.description, str)

    def test_emo_hub_tasks(self):
        """Test [EMO] hub tasks are correctly registered."""
        emo_tasks = {name for name, spec in TASK_REGISTRY_V3.items() if spec.hub_token == "[EMO]"}
        assert emo_tasks == {"emotions", "sentiment", "safety_generic", "safety_familyos"}

    def test_mem_hub_tasks(self):
        """Test [MEM] hub tasks are correctly registered."""
        mem_tasks = {name for name, spec in TASK_REGISTRY_V3.items() if spec.hub_token == "[MEM]"}
        assert mem_tasks == {"embedding"}

    def test_rel_hub_tasks(self):
        """Test [REL] hub tasks are correctly registered."""
        rel_tasks = {name for name, spec in TASK_REGISTRY_V3.items() if spec.hub_token == "[REL]"}
        assert rel_tasks == {"nli", "relation"}

    def test_task_hub_tasks(self):
        """Test [TASK] hub tasks are correctly registered."""
        task_hub_tasks = {
            name for name, spec in TASK_REGISTRY_V3.items() if spec.hub_token == "[TASK]"
        }
        assert task_hub_tasks == {"intent", "ingress"}

    def test_token_level_tasks(self):
        """Test token-level tasks are correctly registered."""
        token_tasks = {
            name
            for name, spec in TASK_REGISTRY_V3.items()
            if spec.task_type == TaskType.TOKEN_CLASSIFICATION
        }
        assert token_tasks == {"ner_general", "ner_family", "temporal"}


# ======================================================================
# Test TaskRegistry Class
# ======================================================================


class TestTaskRegistryClass:
    """Test TaskRegistry class methods."""

    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = TaskRegistry()
        assert len(registry.registry) == 12

    def test_custom_registry_override(self):
        """Test custom registry can add/override tasks."""
        custom_spec = TaskSpec(
            name="custom_task",
            task_type=TaskType.CLASSIFICATION,
            hub_token="[EMO]",
            head_class=HubAwareClassificationHead,
            num_labels=3,
            label_names=["a", "b", "c"],
        )
        custom_registry = {"custom_task": custom_spec}

        registry = TaskRegistry(custom_registry=custom_registry)
        assert len(registry.registry) == 13  # 12 + 1 custom
        assert "custom_task" in registry.registry

    def test_get_task(self):
        """Test get_task method."""
        registry = TaskRegistry()

        emotions_spec = registry.get_task("emotions")
        assert emotions_spec.name == "emotions"
        assert (
            emotions_spec.task_type == TaskType.CLASSIFICATION
        )  # Flat classification for 44 FamilyOS emotions
        assert emotions_spec.hub_token == "[EMO]"

    def test_get_task_unknown(self):
        """Test get_task raises error for unknown task."""
        registry = TaskRegistry()

        try:
            registry.get_task("unknown_task")
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "Unknown task" in str(e)

    def test_get_all_tasks(self):
        """Test get_all_tasks returns all task names."""
        registry = TaskRegistry()
        tasks = registry.get_all_tasks()

        assert len(tasks) == 12
        assert "emotions" in tasks
        assert "ner_general" in tasks
        assert "embedding" in tasks

    def test_get_tasks_by_hub(self):
        """AC3: get_tasks_by_hub() returns correct tasks per hub."""
        registry = TaskRegistry()

        emo_tasks = registry.get_tasks_by_hub("[EMO]")
        assert set(emo_tasks) == {"emotions", "sentiment", "safety_generic", "safety_familyos"}

        mem_tasks = registry.get_tasks_by_hub("[MEM]")
        assert set(mem_tasks) == {"embedding"}

        rel_tasks = registry.get_tasks_by_hub("[REL]")
        assert set(rel_tasks) == {"nli", "relation"}

        task_hub = registry.get_tasks_by_hub("[TASK]")
        assert set(task_hub) == {"intent", "ingress"}

    def test_get_hub_routed_tasks(self):
        """Test get_hub_routed_tasks excludes token-level and embedding."""
        registry = TaskRegistry()
        hub_routed = registry.get_hub_routed_tasks()

        # Should have 8 hub-routed tasks (12 - 3 token - 1 embedding)
        assert len(hub_routed) == 8
        assert "emotions" in hub_routed
        assert "nli" in hub_routed
        assert "intent" in hub_routed

        # Should NOT include token-level or embedding
        assert "ner_general" not in hub_routed
        assert "ner_family" not in hub_routed
        assert "temporal" not in hub_routed
        assert "embedding" not in hub_routed

    def test_get_token_level_tasks(self):
        """AC6: Token-level tasks correctly identified."""
        registry = TaskRegistry()
        token_tasks = registry.get_token_level_tasks()

        assert set(token_tasks) == {"ner_general", "ner_family", "temporal"}


# ======================================================================
# Test Head Creation
# ======================================================================


class TestHeadCreation:
    """Test head creation methods."""

    def test_create_head_emotions(self):
        """AC4: create_head() instantiates correct head class."""
        registry = TaskRegistry()
        head = registry.create_head("emotions", hidden_size=768)

        assert isinstance(head, HubAwareClassificationHead)  # Now flat classification
        assert head.hidden_size == 768
        assert head.num_labels == 44  # EMOTIONS_FAMILYOS_LABELS: 44 classes
        assert head.hub_token == "[EMO]"

    def test_create_head_sentiment(self):
        """Test creating sentiment head."""
        registry = TaskRegistry()
        head = registry.create_head("sentiment", hidden_size=768)

        assert isinstance(head, HubAwareClassificationHead)
        assert head.num_labels == 5  # SENTIMENT_LABELS: 5 classes
        assert head.hub_token == "[EMO]"

    def test_create_head_safety(self):
        """Test creating safety head."""
        registry = TaskRegistry()
        head = registry.create_head("safety_generic", hidden_size=768)

        assert isinstance(head, HubAwareSafetyHead)
        assert head.num_labels == 8  # SAFETY_GENERIC_LABELS: 8 multi-label classes
        assert head.hub_token == "[EMO]"

    def test_create_head_nli(self):
        """Test creating NLI head."""
        registry = TaskRegistry()
        head = registry.create_head("nli", hidden_size=768)

        assert isinstance(head, HubAwareNLIHead)
        assert head.num_labels == 3
        assert head.hub_token == "[REL]"

    def test_create_head_intent(self):
        """Test creating intent head."""
        registry = TaskRegistry()
        head = registry.create_head("intent", hidden_size=768)

        assert isinstance(head, HubAwareClassificationHead)
        assert head.num_labels == 8  # INTENT_LABELS: 8 intents
        assert head.hub_token == "[TASK]"

    def test_create_head_ner(self):
        """Test creating NER head."""
        registry = TaskRegistry()
        head = registry.create_head("ner_general", hidden_size=768)

        assert isinstance(head, HubAwareTokenClassificationHead)
        assert head.num_labels == 17  # NER_GENERAL_LABELS: 17 BIO tags
        # Token-level heads don't use hub_token attribute

    def test_create_head_embedding_returns_none(self):
        """Test that embedding task returns None (no head)."""
        registry = TaskRegistry()
        head = registry.create_head("embedding", hidden_size=768)

        assert head is None

    def test_create_head_with_kwargs(self):
        """Test creating head with additional kwargs."""
        registry = TaskRegistry()
        head = registry.create_head("sentiment", hidden_size=768, dropout=0.2)

        assert isinstance(head, HubAwareClassificationHead)
        # Dropout is set internally, hard to verify without accessing private attrs

    def test_create_all_heads(self):
        """Test creating all heads at once."""
        registry = TaskRegistry()
        heads = registry.create_all_heads(hidden_size=768)

        # Should have 11 heads (12 tasks - 1 embedding)
        assert len(heads) == 11
        assert "emotions" in heads
        assert "sentiment" in heads
        assert "ner_general" in heads
        assert "embedding" not in heads  # No head for embedding

    def test_create_all_heads_subset(self):
        """Test creating heads for subset of tasks."""
        registry = TaskRegistry()
        heads = registry.create_all_heads(
            hidden_size=768, tasks=["emotions", "sentiment", "intent"]
        )

        assert len(heads) == 3
        assert "emotions" in heads
        assert "sentiment" in heads
        assert "intent" in heads
        assert "nli" not in heads

    def test_create_all_heads_with_embedding_excluded(self):
        """Test that embedding is properly excluded from create_all_heads."""
        registry = TaskRegistry()
        heads = registry.create_all_heads(hidden_size=768, tasks=["embedding", "emotions"])

        # Only emotions should be created
        assert len(heads) == 1
        assert "emotions" in heads
        assert "embedding" not in heads


# ======================================================================
# Test Loss Weights
# ======================================================================


class TestLossWeights:
    """Test loss weight configuration."""

    def test_get_loss_weights(self):
        """AC5: Loss weights configured (safety tasks have higher weight)."""
        registry = TaskRegistry()
        weights = registry.get_loss_weights()

        assert len(weights) == 12

        # Check specific weights
        assert weights["emotions"] == 1.0
        assert weights["sentiment"] == 0.8
        assert weights["safety_generic"] == 1.5  # Higher
        assert weights["safety_familyos"] == 2.0  # Highest
        assert weights["nli"] == 1.0
        assert weights["intent"] == 1.0
        assert weights["ner_family"] == 1.2
        assert weights["relation"] == 1.2

    def test_safety_weights_higher(self):
        """Verify safety tasks have higher weights than most tasks."""
        registry = TaskRegistry()
        weights = registry.get_loss_weights()

        # Safety weights should be higher than most tasks
        assert weights["safety_generic"] > weights["sentiment"]
        assert weights["safety_generic"] > weights["nli"]
        assert weights["safety_familyos"] > weights["safety_generic"]
        assert weights["safety_familyos"] == 2.0  # Highest weight


# ======================================================================
# Test Metrics
# ======================================================================


class TestMetrics:
    """Test metric configuration."""

    def test_get_metrics(self):
        """Test getting metrics for a task."""
        registry = TaskRegistry()

        emotions_metrics = registry.get_metrics("emotions")
        assert "macro_f1" in emotions_metrics
        assert "accuracy" in emotions_metrics

        safety_metrics = registry.get_metrics("safety_familyos")
        assert "recall" in safety_metrics
        assert "precision" in safety_metrics
        assert "f1" in safety_metrics
        assert "crisis_recall" in safety_metrics

        ner_metrics = registry.get_metrics("ner_general")
        assert "entity_f1" in ner_metrics
        assert "precision" in ner_metrics
        assert "recall" in ner_metrics


# ======================================================================
# Test Print Registry
# ======================================================================


class TestPrintRegistry:
    """Test print_registry functionality."""

    def test_print_registry_runs(self, capsys):
        """AC7: print_registry() shows organized summary."""
        registry = TaskRegistry()
        registry.print_registry()

        captured = capsys.readouterr()
        output = captured.out

        # Check for header
        assert "v3 Task Registry" in output

        # Check for hub sections
        assert "[EMO] Hub:" in output
        assert "[MEM] Hub:" in output
        assert "[REL] Hub:" in output
        assert "[TASK] Hub:" in output

        # Check for some task names
        assert "emotions" in output
        assert "sentiment" in output
        assert "nli" in output
        assert "ner_general" in output

        # Check for metadata
        assert "labels=" in output
        assert "weight=" in output


# ======================================================================
# Test Singleton
# ======================================================================


class TestSingleton:
    """Test singleton registry instance."""

    def test_get_registry_singleton(self):
        """Test get_registry returns singleton."""
        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2  # Same instance

    def test_singleton_has_all_tasks(self):
        """Test singleton has all registered tasks."""
        registry = get_registry()
        tasks = registry.get_all_tasks()

        assert len(tasks) == 12


# ======================================================================
# Test Head Forward Pass Integration
# ======================================================================


class TestHeadIntegration:
    """Test that created heads can perform forward passes."""

    def test_classification_head_forward(self):
        """Test classification head forward pass."""
        registry = TaskRegistry()
        head = registry.create_head("sentiment", hidden_size=768)
        assert head is not None

        batch_size = 4
        pooled_outputs = {"[EMO]": torch.randn(batch_size, 768)}

        logits = head(None, pooled_outputs)

        assert logits.shape == (batch_size, 5)  # 5 sentiment classes (SENTIMENT_LABELS)

    def test_token_classification_head_forward(self):
        """Test token classification head forward pass."""
        registry = TaskRegistry()
        head = registry.create_head("ner_general", hidden_size=768)
        assert head is not None

        batch_size, seq_len = 4, 20
        hidden_states = torch.randn(batch_size, seq_len, 768)

        logits = head(hidden_states, None)

        assert logits.shape == (batch_size, seq_len, 17)  # 17 NER BIO tags (NER_GENERAL_LABELS)

    def test_classification_head_emotions_forward(self):
        """Test emotions classification head forward pass (44 FamilyOS emotions)."""
        registry = TaskRegistry()
        head = registry.create_head("emotions", hidden_size=768)
        assert head is not None

        batch_size = 4
        pooled_outputs = {"[EMO]": torch.randn(batch_size, 768)}

        logits = head(None, pooled_outputs)

        assert logits.shape == (batch_size, 44)  # 44 FamilyOS emotion classes


# ======================================================================
# Acceptance Criteria Tests
# ======================================================================


class TestAcceptanceCriteria:
    """Comprehensive tests for all acceptance criteria."""

    def test_ac1_all_12_capabilities_registered(self):
        """AC1: All 12 capabilities registered with complete specifications."""
        expected = {
            "emotions",
            "sentiment",
            "safety_generic",
            "safety_familyos",
            "embedding",
            "nli",
            "relation",
            "intent",
            "ingress",
            "ner_general",
            "ner_family",
            "temporal",
        }
        assert set(TASK_REGISTRY_V3.keys()) == expected
        print("✓ AC1: All 12 capabilities registered")

    def test_ac2_taskspec_includes_required_fields(self):
        """AC2: TaskSpec includes hub_token, head_class, labels, metrics."""
        for _name, spec in TASK_REGISTRY_V3.items():
            assert hasattr(spec, "hub_token")
            assert hasattr(spec, "head_class")
            assert hasattr(spec, "label_names")
            assert hasattr(spec, "metrics")
            assert hasattr(spec, "num_labels")
            assert hasattr(spec, "loss_type")
            assert hasattr(spec, "loss_weight")
        print("✓ AC2: TaskSpec includes all required fields")

    def test_ac3_get_tasks_by_hub_correct(self):
        """AC3: get_tasks_by_hub() returns correct tasks per hub."""
        registry = TaskRegistry()

        emo = set(registry.get_tasks_by_hub("[EMO]"))
        assert emo == {"emotions", "sentiment", "safety_generic", "safety_familyos"}

        mem = set(registry.get_tasks_by_hub("[MEM]"))
        assert mem == {"embedding"}

        rel = set(registry.get_tasks_by_hub("[REL]"))
        assert rel == {"nli", "relation"}

        task = set(registry.get_tasks_by_hub("[TASK]"))
        assert task == {"intent", "ingress"}

        print("✓ AC3: get_tasks_by_hub() returns correct tasks")

    def test_ac4_create_head_instantiates_correct_class(self):
        """AC4: create_head() instantiates correct head class."""
        registry = TaskRegistry()

        emotions_head = registry.create_head("emotions")
        assert isinstance(emotions_head, HubAwareClassificationHead)  # Now flat classification

        sentiment_head = registry.create_head("sentiment")
        assert isinstance(sentiment_head, HubAwareClassificationHead)

        nli_head = registry.create_head("nli")
        assert isinstance(nli_head, HubAwareNLIHead)

        safety_head = registry.create_head("safety_generic")
        assert isinstance(safety_head, HubAwareSafetyHead)

        ner_head = registry.create_head("ner_general")
        assert isinstance(ner_head, HubAwareTokenClassificationHead)

        embedding_head = registry.create_head("embedding")
        assert embedding_head is None  # No head for embedding

        print("✓ AC4: create_head() instantiates correct head class")

    def test_ac5_loss_weights_configured(self):
        """AC5: Loss weights configured (safety tasks have higher weight)."""
        registry = TaskRegistry()
        weights = registry.get_loss_weights()

        # Safety tasks have higher weights
        assert weights["safety_generic"] == 1.5
        assert weights["safety_familyos"] == 2.0

        # Higher than most other tasks
        assert weights["safety_generic"] > weights["sentiment"]
        assert weights["safety_familyos"] > weights["safety_generic"]

        print("✓ AC5: Loss weights configured (safety tasks have higher weight)")

    def test_ac6_token_level_tasks_identified(self):
        """AC6: Token-level tasks correctly identified."""
        registry = TaskRegistry()
        token_tasks = set(registry.get_token_level_tasks())

        assert token_tasks == {"ner_general", "ner_family", "temporal"}

        # Verify they're not in hub-routed tasks
        hub_routed = set(registry.get_hub_routed_tasks())
        assert token_tasks.isdisjoint(hub_routed)

        print("✓ AC6: Token-level tasks correctly identified")

    def test_ac7_print_registry_shows_summary(self, capsys):
        """AC7: print_registry() shows organized summary."""
        registry = TaskRegistry()
        registry.print_registry()

        captured = capsys.readouterr()
        output = captured.out

        # Check for key elements
        assert "v3 Task Registry" in output
        assert "[EMO] Hub:" in output
        assert "[MEM] Hub:" in output
        assert "[REL] Hub:" in output
        assert "[TASK] Hub:" in output
        assert "emotions" in output
        assert "labels=" in output
        assert "weight=" in output

        print("✓ AC7: print_registry() shows organized summary")
