from __future__ import annotations

"""
Unified FamilyOS Dataset Loader for v3 Multi-Task Training.

Loads unified JSONL files with hub_routing and 8 task types:
- emotions (multi-label list)
- sentiment (single label)
- ner_family (span list)
- safety_familyos (single label)
- intent (single label)
- ingress (single label)
- relations (triple list)
- temporal (span list)
"""

import glob
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, IterableDataset

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Supported task types in unified FamilyOS data."""

    EMOTIONS = "emotions"
    SENTIMENT = "sentiment"
    NER_FAMILY = "ner_family"
    SAFETY_FAMILYOS = "safety_familyos"
    INTENT = "intent"
    INGRESS = "ingress"
    RELATIONS = "relations"
    TEMPORAL = "temporal"


class HubType(Enum):
    """Hub token routing types."""

    EMO = "EMO"
    REL = "REL"
    MEM = "MEM"
    TASK = "TASK"


@dataclass
class HubTaskMapping:
    """Maps hub routing to task activation."""

    hub_to_tasks: dict[str, list[TaskType]] = field(
        default_factory=lambda: {
            "EMO": [TaskType.EMOTIONS, TaskType.SENTIMENT, TaskType.SAFETY_FAMILYOS],
            "REL": [TaskType.RELATIONS],
            "MEM": [TaskType.TEMPORAL, TaskType.NER_FAMILY],
            "TASK": [TaskType.INTENT, TaskType.INGRESS],
        }
    )

    def __post_init__(self) -> None:
        self.task_to_hub: dict[TaskType, str] = {}
        for hub_name, tasks in self.hub_to_tasks.items():
            for task in tasks:
                self.task_to_hub[task] = hub_name


class HubRoutingParser:
    """Parses hub routing to determine task activation and gradient masking."""

    def __init__(
        self,
        hub_to_tasks: dict[str, list[TaskType]] | None = None,
        always_train_safety: bool = True,
        safety_weight_override: float = 2.0,
    ) -> None:
        self.mapping = HubTaskMapping() if hub_to_tasks is None else HubTaskMapping(hub_to_tasks)
        self.always_train_safety = always_train_safety
        self.safety_weight_override = safety_weight_override

    def get_active_tasks(self, hub_routing: HubRouting, sample: UnifiedSample) -> list[TaskType]:
        """Return tasks whose controlling hub is active and present in sample."""

        hub_active = {
            "EMO": hub_routing.emo,
            "REL": hub_routing.rel,
            "MEM": hub_routing.mem,
            "TASK": hub_routing.task,
        }

        active_tasks: list[TaskType] = []
        for task_type in TaskType:
            controlling_hub = self.mapping.task_to_hub.get(task_type)
            if controlling_hub and hub_active.get(controlling_hub, False):
                if sample.has_task(task_type):
                    active_tasks.append(task_type)

        if self.always_train_safety and sample.has_task(TaskType.SAFETY_FAMILYOS):
            if TaskType.SAFETY_FAMILYOS not in active_tasks:
                active_tasks.append(TaskType.SAFETY_FAMILYOS)

        return active_tasks

    def get_hub_gradient_mask(self, hub_routing: HubRouting) -> torch.Tensor:
        """Return hub gradient mask tensor [EMO, REL, MEM, TASK]."""

        return hub_routing.to_tensor()

    def get_task_weights(
        self, hub_routing: HubRouting, active_tasks: list[TaskType]
    ) -> dict[TaskType, float]:
        """Compute normalized weights per active task with safety override."""

        if not active_tasks:
            return {}

        weights: dict[TaskType, float] = {}
        base_weight = 1.0 / len(active_tasks)

        for task_type in active_tasks:
            weight = base_weight
            if task_type == TaskType.SAFETY_FAMILYOS:
                weight *= self.safety_weight_override
            weights[task_type] = weight

        return weights

    def parse_batch(self, samples: list[UnifiedSample]) -> dict[str, Any]:
        """Aggregate hub masks, active task indices, and task weights for a batch."""

        batch_size = len(samples)
        hub_masks = torch.zeros(batch_size, 4)
        task_active: dict[TaskType, list[int]] = {task: [] for task in TaskType}
        task_weights: dict[TaskType, list[float]] = {task: [] for task in TaskType}

        for idx, sample in enumerate(samples):
            mask = self.get_hub_gradient_mask(sample.hub_routing)
            hub_masks[idx] = mask

            active = self.get_active_tasks(sample.hub_routing, sample)
            weights = self.get_task_weights(sample.hub_routing, active)

            for task_type in TaskType:
                if task_type in active:
                    task_active[task_type].append(idx)
                    task_weights[task_type].append(weights[task_type])
                else:
                    task_weights[task_type].append(0.0)

        task_weight_tensors = {task: torch.tensor(w) for task, w in task_weights.items()}

        return {
            "hub_masks": hub_masks,
            "task_active": task_active,
            "task_weights": task_weight_tensors,
        }


@dataclass
class HubRouting:
    """Hub routing configuration parsed from sample."""

    emo: bool = False
    rel: bool = False
    mem: bool = False
    task: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, bool]) -> HubRouting:
        """Parse hub_routing dict from JSON data."""

        return cls(
            emo=bool(data.get("EMO", False)),
            rel=bool(data.get("REL", False)),
            mem=bool(data.get("MEM", False)),
            task=bool(data.get("TASK", False)),
        )

    def to_tensor(self) -> torch.Tensor:
        """Convert routing flags to float tensor [EMO, REL, MEM, TASK]."""

        return torch.tensor(
            [float(self.emo), float(self.rel), float(self.mem), float(self.task)],
            dtype=torch.float32,
        )

    @property
    def active_hubs(self) -> list[str]:
        """Return list of active hub names in deterministic order."""

        active: list[str] = []
        if self.emo:
            active.append("EMO")
        if self.rel:
            active.append("REL")
        if self.mem:
            active.append("MEM")
        if self.task:
            active.append("TASK")
        return active


@dataclass
class SpanAnnotation:
    """Span annotation for NER and temporal tasks."""

    start: int
    end: int
    label: str
    token: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpanAnnotation:
        """Create SpanAnnotation from dictionary."""

        return cls(
            start=int(data["start"]),
            end=int(data["end"]),
            label=str(data["label"]),
            token=str(data["token"]),
        )


@dataclass
class RelationTriple:
    """Relation triple annotation."""

    subject: str
    predicate: str
    object: str

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "RelationTriple | None":
        """Create RelationTriple from dictionary. Returns None if malformed."""
        # Handle malformed entries gracefully
        subject = data.get("subject")
        predicate = data.get("predicate")
        obj = data.get("object")

        if not all([subject, predicate, obj]):
            return None

        return cls(
            subject=str(subject),
            predicate=str(predicate),
            object=str(obj),
        )


@dataclass
class UnifiedSample:
    """Parsed sample from unified FamilyOS JSONL."""

    id: str
    text: str
    emotions: list[str] = field(default_factory=list)
    sentiment: str | None = None
    safety_familyos: str | None = None
    intent: str | None = None
    ingress: str | None = None
    ner_family: list[SpanAnnotation] = field(default_factory=list)
    temporal: list[SpanAnnotation] = field(default_factory=list)
    relations: list[RelationTriple] = field(default_factory=list)
    hub_routing: HubRouting = field(default_factory=HubRouting)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> UnifiedSample:
        """Parse sample from JSON dict."""

        tasks = data.get("tasks", {})

        # Filter out None values from malformed relation triples
        relations = [
            r
            for r in (RelationTriple.from_dict(triple) for triple in tasks.get("relations", []))
            if r is not None
        ]

        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            emotions=list(tasks.get("emotions", [])),
            sentiment=tasks.get("sentiment"),
            safety_familyos=tasks.get("safety_familyos"),
            intent=tasks.get("intent"),
            ingress=tasks.get("ingress"),
            ner_family=[SpanAnnotation.from_dict(span) for span in tasks.get("ner_family", [])],
            temporal=[SpanAnnotation.from_dict(span) for span in tasks.get("temporal", [])],
            relations=relations,
            hub_routing=HubRouting.from_dict(data.get("hub_routing", {})),
        )

    def has_task(self, task_type: TaskType) -> bool:
        """Check if sample has non-empty data for given task."""

        if task_type == TaskType.EMOTIONS:
            return len(self.emotions) > 0
        if task_type == TaskType.SENTIMENT:
            return self.sentiment is not None
        if task_type == TaskType.NER_FAMILY:
            return len(self.ner_family) > 0
        if task_type == TaskType.SAFETY_FAMILYOS:
            return self.safety_familyos is not None
        if task_type == TaskType.INTENT:
            return self.intent is not None
        if task_type == TaskType.INGRESS:
            return self.ingress is not None
        if task_type == TaskType.RELATIONS:
            return len(self.relations) > 0
        if task_type == TaskType.TEMPORAL:
            return len(self.temporal) > 0
        return False


class UnifiedFamilyOSDataset(Dataset):
    """PyTorch Dataset for unified FamilyOS data (eager loading)."""

    def __init__(
        self,
        data_dir: str | Path,
        shard_pattern: str = "shard_*.jsonl",
        max_samples: int | None = None,
        filter_tasks: list[TaskType] | None = None,
        require_hub_routing: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.shard_pattern = shard_pattern
        self.max_samples = max_samples
        self.filter_tasks = filter_tasks
        self.require_hub_routing = require_hub_routing

        self.samples: list[UnifiedSample] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Load all samples from shard files into memory."""

        shard_files = sorted(glob.glob(str(self.data_dir / self.shard_pattern)))
        if not shard_files:
            raise FileNotFoundError(
                f"No shard files found matching {self.data_dir / self.shard_pattern}"
            )

        logger.info("Found %d shard files in %s", len(shard_files), self.data_dir)

        for shard_path in shard_files:
            self._load_shard(shard_path)
            if self.max_samples is not None and len(self.samples) >= self.max_samples:
                self.samples = self.samples[: self.max_samples]
                break

        logger.info("Loaded %d samples from %s", len(self.samples), self.data_dir)

    def _load_shard(self, shard_path: str) -> None:
        """Load a single shard file and append samples."""

        with open(shard_path, encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                data = json.loads(line)
                sample = UnifiedSample.from_json(data)

                if self._should_include(sample):
                    self.samples.append(sample)

                if self.max_samples is not None and len(self.samples) >= self.max_samples:
                    return

    def _should_include(self, sample: UnifiedSample) -> bool:
        """Determine whether to include a sample based on filters."""

        if self.filter_tasks:
            has_required_task = any(sample.has_task(task) for task in self.filter_tasks)
            if not has_required_task:
                return False

        if self.require_hub_routing and not sample.hub_routing.active_hubs:
            return False

        return True

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> UnifiedSample:
        return self.samples[idx]

    def get_task_distribution(self) -> dict[str, int]:
        """Get count of samples per task type."""

        distribution = {task.value: 0 for task in TaskType}
        for sample in self.samples:
            for task_type in TaskType:
                if sample.has_task(task_type):
                    distribution[task_type.value] += 1
        return distribution

    def get_hub_distribution(self) -> dict[str, int]:
        """Get count of samples per hub routing value."""

        distribution = {"EMO": 0, "REL": 0, "MEM": 0, "TASK": 0, "none": 0}
        for sample in self.samples:
            routing = sample.hub_routing
            if routing.emo:
                distribution["EMO"] += 1
            if routing.rel:
                distribution["REL"] += 1
            if routing.mem:
                distribution["MEM"] += 1
            if routing.task:
                distribution["TASK"] += 1
            if not routing.active_hubs:
                distribution["none"] += 1
        return distribution


class IterableUnifiedFamilyOSDataset(IterableDataset):
    """Streaming/Iterable Dataset for unified FamilyOS data (memory efficient)."""

    def __init__(
        self,
        data_dir: str | Path,
        shard_pattern: str = "shard_*.jsonl",
        shuffle_shards: bool = True,
        filter_tasks: list[TaskType] | None = None,
        require_hub_routing: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.shard_pattern = shard_pattern
        self.shuffle_shards = shuffle_shards
        self.filter_tasks = filter_tasks
        self.require_hub_routing = require_hub_routing

        self.shard_files = sorted(glob.glob(str(self.data_dir / self.shard_pattern)))
        if not self.shard_files:
            raise FileNotFoundError(
                f"No shard files found matching {self.data_dir / self.shard_pattern}"
            )

    def __iter__(self) -> Iterator[UnifiedSample]:
        import random

        shard_files = self.shard_files.copy()
        if self.shuffle_shards:
            random.shuffle(shard_files)

        for shard_path in shard_files:
            with open(shard_path, encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue

                    data = json.loads(line)
                    sample = UnifiedSample.from_json(data)

                    if self.filter_tasks and not any(
                        sample.has_task(task) for task in self.filter_tasks
                    ):
                        continue

                    if self.require_hub_routing and not sample.hub_routing.active_hubs:
                        continue

                    yield sample
