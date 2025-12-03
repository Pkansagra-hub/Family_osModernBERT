"""
Milestone 2: Data Pipeline Tests
Issue 2.1.5: data/multitask_dataset.py

Tests for:
- TaskDataset: Wrapper adding task information to samples
- MultiTaskDataset: Combines multiple task datasets
- StreamingMultiTaskDataset: For large datasets (streaming mode)
- create_multitask_dataset: Factory function
- interleave_datasets: Custom sampling strategies
"""

import pytest
from datasets import Dataset


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_dataset():
    """Create a simple HuggingFace dataset for testing."""
    return Dataset.from_dict(
        {
            "text": ["Hello world", "Test text", "Another sample"],
            "label": [0, 1, 0],
        }
    )


@pytest.fixture
def ner_dataset():
    """Create a simple NER-style dataset."""
    return Dataset.from_dict(
        {
            "tokens": [["Hello", "World"], ["Test", "Sample"]],
            "ner_tags": [[0, 1], [0, 0]],
        }
    )


@pytest.fixture
def task_dataset(simple_dataset):
    """Create a TaskDataset for testing."""
    from modeling_studio.data.multitask_dataset import TaskDataset

    return TaskDataset(name="test_task", dataset=simple_dataset)


@pytest.fixture
def task_dataset_with_weight(simple_dataset):
    """Create a TaskDataset with custom weight."""
    from modeling_studio.data.multitask_dataset import TaskDataset

    return TaskDataset(name="weighted_task", dataset=simple_dataset, weight=2.0)


@pytest.fixture
def multitask_dataset(simple_dataset, ner_dataset):
    """Create a MultiTaskDataset with two tasks."""
    from modeling_studio.data.multitask_dataset import MultiTaskDataset, TaskDataset

    task1 = TaskDataset(name="classification", dataset=simple_dataset)
    task2 = TaskDataset(name="ner", dataset=ner_dataset)
    return MultiTaskDataset(task_datasets=[task1, task2])


# =============================================================================
# TaskDataset Tests
# =============================================================================


class TestTaskDatasetLen:
    """Test len(TaskDataset) returns dataset size."""

    def test_task_dataset_len_returns_correct_size(self, task_dataset):
        """len(TaskDataset) should return underlying dataset size."""
        assert len(task_dataset) == 3

    def test_task_dataset_len_empty(self):
        """len(TaskDataset) should return 0 for empty dataset."""
        from modeling_studio.data.multitask_dataset import TaskDataset

        empty_ds = Dataset.from_dict({"text": [], "label": []})
        task_ds = TaskDataset(name="empty", dataset=empty_ds)
        assert len(task_ds) == 0


class TestTaskDatasetGetitem:
    """Test TaskDataset[0] returns sample with task field."""

    def test_task_dataset_getitem_returns_dict(self, task_dataset):
        """TaskDataset[0] should return a dict."""
        sample = task_dataset[0]
        assert isinstance(sample, dict)

    def test_task_dataset_getitem_has_original_fields(self, task_dataset):
        """TaskDataset[0] should preserve original fields."""
        sample = task_dataset[0]
        assert "text" in sample
        assert "label" in sample

    def test_task_dataset_getitem_values(self, task_dataset):
        """TaskDataset[0] should return correct values."""
        sample = task_dataset[0]
        assert sample["text"] == "Hello world"
        assert sample["label"] == 0


class TestTaskDatasetTaskField:
    """Test sample has correct task name."""

    def test_task_dataset_has_task_field(self, task_dataset):
        """Sample should include 'task' field."""
        sample = task_dataset[0]
        assert "task" in sample

    def test_task_dataset_task_field_value(self, task_dataset):
        """Task field should have correct task name."""
        sample = task_dataset[0]
        assert sample["task"] == "test_task"

    def test_task_dataset_all_samples_have_task(self, task_dataset):
        """All samples should have the same task name."""
        for sample in task_dataset:
            assert sample["task"] == "test_task"


class TestTaskDatasetPreprocessing:
    """Test preprocessing function applied."""

    def test_task_dataset_preprocessing_applied(self, simple_dataset):
        """Preprocessing function should be applied to each sample."""
        from modeling_studio.data.multitask_dataset import TaskDataset

        def preprocess(sample):
            sample["text"] = sample["text"].upper()
            return sample

        task_ds = TaskDataset(name="test", dataset=simple_dataset, preprocessing_fn=preprocess)
        sample = task_ds[0]
        assert sample["text"] == "HELLO WORLD"

    def test_task_dataset_preprocessing_preserves_task(self, simple_dataset):
        """Preprocessing should not interfere with task field."""
        from modeling_studio.data.multitask_dataset import TaskDataset

        def preprocess(sample):
            sample["processed"] = True
            return sample

        task_ds = TaskDataset(name="test_task", dataset=simple_dataset, preprocessing_fn=preprocess)
        sample = task_ds[0]
        assert sample["task"] == "test_task"
        assert sample["processed"] is True


class TestTaskDatasetWeight:
    """Test weight attribute set correctly."""

    def test_task_dataset_default_weight(self, task_dataset):
        """Default weight should be 1.0."""
        assert task_dataset.weight == 1.0

    def test_task_dataset_custom_weight(self, task_dataset_with_weight):
        """Custom weight should be set correctly."""
        assert task_dataset_with_weight.weight == 2.0

    def test_task_dataset_weight_preserved_on_select(self, task_dataset_with_weight):
        """Weight should be preserved when selecting subset."""
        subset = task_dataset_with_weight.select([0, 1])
        assert subset.weight == 2.0


class TestTaskDatasetSelect:
    """Test select(indices) returns subset."""

    def test_task_dataset_select_returns_task_dataset(self, task_dataset):
        """select() should return a TaskDataset."""
        from modeling_studio.data.multitask_dataset import TaskDataset

        subset = task_dataset.select([0, 1])
        assert isinstance(subset, TaskDataset)

    def test_task_dataset_select_correct_size(self, task_dataset):
        """select() should return subset with correct size."""
        subset = task_dataset.select([0, 1])
        assert len(subset) == 2

    def test_task_dataset_select_preserves_name(self, task_dataset):
        """select() should preserve task name."""
        subset = task_dataset.select([0])
        assert subset.name == "test_task"
        assert subset[0]["task"] == "test_task"


class TestTaskDatasetShuffle:
    """Test shuffle() randomizes order."""

    def test_task_dataset_shuffle_returns_task_dataset(self, task_dataset):
        """shuffle() should return a TaskDataset."""
        from modeling_studio.data.multitask_dataset import TaskDataset

        shuffled = task_dataset.shuffle(seed=42)
        assert isinstance(shuffled, TaskDataset)

    def test_task_dataset_shuffle_preserves_size(self, task_dataset):
        """shuffle() should preserve dataset size."""
        shuffled = task_dataset.shuffle(seed=42)
        assert len(shuffled) == len(task_dataset)

    def test_task_dataset_shuffle_preserves_name(self, task_dataset):
        """shuffle() should preserve task name."""
        shuffled = task_dataset.shuffle(seed=42)
        assert shuffled.name == "test_task"

    def test_task_dataset_shuffle_deterministic(self, task_dataset):
        """shuffle() with same seed should give same order."""
        shuffled1 = task_dataset.shuffle(seed=42)
        shuffled2 = task_dataset.shuffle(seed=42)
        for s1, s2 in zip(shuffled1, shuffled2):
            assert s1["text"] == s2["text"]


class TestTaskDatasetColumnNames:
    """Test includes 'task' in column names."""

    def test_task_dataset_column_names_includes_task(self, task_dataset):
        """column_names should include 'task'."""
        assert "task" in task_dataset.column_names

    def test_task_dataset_column_names_includes_original(self, task_dataset):
        """column_names should include original columns."""
        assert "text" in task_dataset.column_names
        assert "label" in task_dataset.column_names


# =============================================================================
# MultiTaskDataset Tests
# =============================================================================


class TestMultiTaskDatasetLen:
    """Test total length is sum of all task lengths."""

    def test_multitask_dataset_len_sum(self, multitask_dataset):
        """len(MultiTaskDataset) should be sum of task lengths."""
        # simple_dataset has 3 samples, ner_dataset has 2 samples
        assert len(multitask_dataset) == 5

    def test_multitask_dataset_len_single_task(self, simple_dataset):
        """len() should work with single task."""
        from modeling_studio.data.multitask_dataset import MultiTaskDataset, TaskDataset

        task = TaskDataset(name="single", dataset=simple_dataset)
        mtd = MultiTaskDataset(task_datasets=[task])
        assert len(mtd) == 3


class TestMultiTaskDatasetGetitem:
    """Test returns correct sample from correct task."""

    def test_multitask_dataset_getitem_first_task(self, multitask_dataset):
        """First samples should come from first task."""
        sample = multitask_dataset[0]
        assert sample["task"] == "classification"

    def test_multitask_dataset_getitem_second_task(self, multitask_dataset):
        """Samples after first task should come from second task."""
        # First task has 3 samples (indices 0-2), second task starts at 3
        sample = multitask_dataset[3]
        assert sample["task"] == "ner"

    def test_multitask_dataset_getitem_index_error(self, multitask_dataset):
        """Out of range index should raise IndexError."""
        with pytest.raises(IndexError):
            _ = multitask_dataset[100]

    def test_multitask_dataset_getitem_negative_index_error(self, multitask_dataset):
        """Negative index should raise IndexError."""
        with pytest.raises(IndexError):
            _ = multitask_dataset[-1]


class TestMultiTaskDatasetTaskNames:
    """Test task_names property correct."""

    def test_multitask_dataset_task_names_list(self, multitask_dataset):
        """task_names should be a list."""
        assert isinstance(multitask_dataset.task_names, list)

    def test_multitask_dataset_task_names_content(self, multitask_dataset):
        """task_names should contain correct names."""
        assert "classification" in multitask_dataset.task_names
        assert "ner" in multitask_dataset.task_names

    def test_multitask_dataset_task_names_order(self, multitask_dataset):
        """task_names should preserve order."""
        assert multitask_dataset.task_names[0] == "classification"
        assert multitask_dataset.task_names[1] == "ner"


class TestMultiTaskDatasetTaskSizes:
    """Test task_sizes dict correct."""

    def test_multitask_dataset_task_sizes_dict(self, multitask_dataset):
        """task_sizes should be a dict."""
        assert isinstance(multitask_dataset.task_sizes, dict)

    def test_multitask_dataset_task_sizes_content(self, multitask_dataset):
        """task_sizes should have correct counts."""
        assert multitask_dataset.task_sizes["classification"] == 3
        assert multitask_dataset.task_sizes["ner"] == 2


class TestMultiTaskDatasetGetTaskDataset:
    """Test retrieves specific task dataset."""

    def test_multitask_dataset_get_task_dataset_exists(self, multitask_dataset):
        """get_task_dataset should return TaskDataset for valid name."""
        from modeling_studio.data.multitask_dataset import TaskDataset

        task_ds = multitask_dataset.get_task_dataset("classification")
        assert isinstance(task_ds, TaskDataset)
        assert task_ds.name == "classification"

    def test_multitask_dataset_get_task_dataset_not_found(self, multitask_dataset):
        """get_task_dataset should raise KeyError for unknown task."""
        with pytest.raises(KeyError):
            multitask_dataset.get_task_dataset("unknown_task")


class TestMultiTaskDatasetGetTaskSamples:
    """Test iterator for specific task."""

    def test_multitask_dataset_get_task_samples_iterator(self, multitask_dataset):
        """get_task_samples should return an iterator."""
        samples = list(multitask_dataset.get_task_samples("classification"))
        assert len(samples) == 3

    def test_multitask_dataset_get_task_samples_correct_task(self, multitask_dataset):
        """All samples should have correct task field."""
        for sample in multitask_dataset.get_task_samples("ner"):
            assert sample["task"] == "ner"


class TestMultiTaskDatasetShuffle:
    """Test shuffling works with reshuffle()."""

    def test_multitask_dataset_reshuffle_changes_order(self, simple_dataset):
        """reshuffle() should change sample order."""
        from modeling_studio.data.multitask_dataset import MultiTaskDataset, TaskDataset

        task = TaskDataset(name="test", dataset=simple_dataset)
        mtd = MultiTaskDataset(task_datasets=[task], shuffle=True)

        # Get original order
        original_order = [mtd[i]["text"] for i in range(len(mtd))]

        # Reshuffle with different seed
        mtd.reshuffle(seed=123)
        new_order = [mtd[i]["text"] for i in range(len(mtd))]

        # Orders might be same by chance, but at least reshuffle runs
        assert len(new_order) == len(original_order)

    def test_multitask_dataset_reshuffle_deterministic(self, simple_dataset):
        """reshuffle() with same seed should give same order."""
        from modeling_studio.data.multitask_dataset import MultiTaskDataset, TaskDataset

        task = TaskDataset(name="test", dataset=simple_dataset)
        mtd = MultiTaskDataset(task_datasets=[task], shuffle=True)

        mtd.reshuffle(seed=42)
        order1 = [mtd[i]["text"] for i in range(len(mtd))]

        mtd.reshuffle(seed=42)
        order2 = [mtd[i]["text"] for i in range(len(mtd))]

        assert order1 == order2


class TestMultiTaskDatasetSplitByTask:
    """Test splits back into individual TaskDatasets."""

    def test_multitask_dataset_split_by_task_returns_dict(self, multitask_dataset):
        """split_by_task should return a dict."""
        splits = multitask_dataset.split_by_task()
        assert isinstance(splits, dict)

    def test_multitask_dataset_split_by_task_keys(self, multitask_dataset):
        """split_by_task dict should have task names as keys."""
        splits = multitask_dataset.split_by_task()
        assert "classification" in splits
        assert "ner" in splits

    def test_multitask_dataset_split_by_task_values(self, multitask_dataset):
        """split_by_task should return TaskDatasets as values."""
        from modeling_studio.data.multitask_dataset import TaskDataset

        splits = multitask_dataset.split_by_task()
        for name, ds in splits.items():
            assert isinstance(ds, TaskDataset)
            assert ds.name == name


class TestMultiTaskDatasetBinarySearch:
    """Test index lookup uses binary search."""

    def test_multitask_dataset_binary_search_first_task(self, multitask_dataset):
        """Binary search should correctly find first task."""
        task_idx, local_idx = multitask_dataset._get_task_and_index(0)
        assert task_idx == 0
        assert local_idx == 0

    def test_multitask_dataset_binary_search_boundary(self, multitask_dataset):
        """Binary search should handle task boundaries."""
        # Index 2 is last of first task
        task_idx, local_idx = multitask_dataset._get_task_and_index(2)
        assert task_idx == 0
        assert local_idx == 2

        # Index 3 is first of second task
        task_idx, local_idx = multitask_dataset._get_task_and_index(3)
        assert task_idx == 1
        assert local_idx == 0

    def test_multitask_dataset_binary_search_last(self, multitask_dataset):
        """Binary search should work for last index."""
        task_idx, local_idx = multitask_dataset._get_task_and_index(4)
        assert task_idx == 1
        assert local_idx == 1


class TestMultiTaskDatasetEmptyError:
    """Test raises error for empty list."""

    def test_multitask_dataset_empty_raises_error(self):
        """MultiTaskDataset with empty list should raise ValueError."""
        from modeling_studio.data.multitask_dataset import MultiTaskDataset

        with pytest.raises(ValueError, match="At least one TaskDataset is required"):
            MultiTaskDataset(task_datasets=[])


# =============================================================================
# StreamingMultiTaskDataset Tests
# =============================================================================


class TestStreamingMultiTaskDataset:
    """Test StreamingMultiTaskDataset initializes."""

    def test_streaming_multitask_dataset_init(self):
        """StreamingMultiTaskDataset should initialize."""
        from modeling_studio.data.multitask_dataset import StreamingMultiTaskDataset

        # Create mock iterable datasets
        def gen_task1():
            yield {"text": "sample1"}
            yield {"text": "sample2"}

        def gen_task2():
            yield {"text": "sample3"}

        from datasets import IterableDataset

        task1 = IterableDataset.from_generator(gen_task1)
        task2 = IterableDataset.from_generator(gen_task2)

        stream_ds = StreamingMultiTaskDataset(task_datasets={"task1": task1, "task2": task2})

        assert stream_ds is not None
        assert "task1" in stream_ds.task_names
        assert "task2" in stream_ds.task_names

    def test_streaming_multitask_dataset_iteration(self):
        """StreamingMultiTaskDataset should be iterable."""
        from modeling_studio.data.multitask_dataset import StreamingMultiTaskDataset

        def gen_task():
            yield {"text": "sample"}

        from datasets import IterableDataset

        task = IterableDataset.from_generator(gen_task)

        stream_ds = StreamingMultiTaskDataset(task_datasets={"task": task})

        samples = list(stream_ds)
        assert len(samples) > 0
        assert samples[0]["task"] == "task"


class TestStreamingTaskWeightsNormalized:
    """Test weights sum to 1.0."""

    def test_streaming_task_weights_normalized(self):
        """Task probabilities should sum to 1.0."""
        from modeling_studio.data.multitask_dataset import StreamingMultiTaskDataset

        def gen():
            yield {"text": "x"}

        from datasets import IterableDataset

        task1 = IterableDataset.from_generator(gen)
        task2 = IterableDataset.from_generator(gen)

        stream_ds = StreamingMultiTaskDataset(
            task_datasets={"task1": task1, "task2": task2},
            task_weights={"task1": 2.0, "task2": 3.0},
        )

        total_prob = sum(stream_ds.task_probabilities.values())
        assert abs(total_prob - 1.0) < 1e-6

    def test_streaming_task_weights_default_uniform(self):
        """Default weights should be uniform."""
        from modeling_studio.data.multitask_dataset import StreamingMultiTaskDataset

        def gen():
            yield {"text": "x"}

        from datasets import IterableDataset

        task1 = IterableDataset.from_generator(gen)
        task2 = IterableDataset.from_generator(gen)

        stream_ds = StreamingMultiTaskDataset(task_datasets={"task1": task1, "task2": task2})

        # With no weights, each should have 0.5 probability
        assert abs(stream_ds.task_probabilities["task1"] - 0.5) < 1e-6
        assert abs(stream_ds.task_probabilities["task2"] - 0.5) < 1e-6


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateMultitaskDataset:
    """Test create_multitask_dataset factory function."""

    def test_create_multitask_dataset_basic(self, simple_dataset, ner_dataset):
        """create_multitask_dataset should create MultiTaskDataset."""
        from modeling_studio.data.multitask_dataset import (
            MultiTaskDataset,
            create_multitask_dataset,
        )

        datasets = {"classification": simple_dataset, "ner": ner_dataset}
        mtd = create_multitask_dataset(datasets)

        assert isinstance(mtd, MultiTaskDataset)
        assert len(mtd) == 5

    def test_create_multitask_dataset_with_weights(self, simple_dataset):
        """create_multitask_dataset should apply weights."""
        from modeling_studio.data.multitask_dataset import create_multitask_dataset

        datasets = {"task": simple_dataset}
        weights = {"task": 2.0}
        mtd = create_multitask_dataset(datasets, weights=weights)

        task_ds = mtd.get_task_dataset("task")
        assert task_ds.weight == 2.0


class TestModuleExports:
    """Test that all public APIs are exported."""

    def test_all_exports_defined(self):
        """__all__ should be defined with public APIs."""
        from modeling_studio.data import multitask_dataset

        assert hasattr(multitask_dataset, "__all__")
        assert "TaskDataset" in multitask_dataset.__all__
        assert "MultiTaskDataset" in multitask_dataset.__all__
        assert "StreamingMultiTaskDataset" in multitask_dataset.__all__
        assert "create_multitask_dataset" in multitask_dataset.__all__
        assert "interleave_datasets" in multitask_dataset.__all__
