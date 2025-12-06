from __future__ import annotations

import torch

from modeling_studio.data.extractors_v3 import (
    MultiTaskExtractor,
    collate_classification_labels,
    collate_multi_label,
    collate_token_labels,
)
from modeling_studio.data.loaders_v3 import RelationTriple, SpanAnnotation, UnifiedSample


class DummyTokenizer:
    """Simple tokenizer that returns space-based offset mappings."""

    def __call__(self, text: str, max_length: int, truncation: bool, return_offsets_mapping: bool):
        del max_length, truncation, return_offsets_mapping
        return {"offset_mapping": self._build_offsets(text)}

    def _build_offsets(self, text: str) -> list[tuple[int, int]]:
        offsets = [(0, 0)]  # Simulate [CLS] token
        start = 0
        for token in text.split():
            end = start + len(token)
            offsets.append((start, end))
            start = end + 1  # +1 for space
        return offsets


def test_multi_task_extractor() -> None:
    tokenizer = DummyTokenizer()
    extractor = MultiTaskExtractor(tokenizer=tokenizer, max_seq_length=128)

    sample = UnifiedSample(
        id="sample1",
        text="John and Mary met on Monday",
        emotions=["joy", "anger"],
        sentiment="positive",
        safety_familyos="GREEN",
        intent="share",
        ingress="DIARY",
        ner_family=[SpanAnnotation(start=0, end=4, label="PERSON", token="John")],
        temporal=[SpanAnnotation(start=21, end=27, label="DATE_ABS", token="Monday")],
        relations=[RelationTriple(subject="John", predicate="friend_of", object="Mary")],
    )

    labels = extractor.extract(sample)

    emotions_vocab = extractor.vocabs.EMOTIONS
    assert labels.emotions is not None
    assert labels.emotions.shape[0] == emotions_vocab.num_labels
    assert labels.emotions[emotions_vocab.encode("joy")] == 1.0
    assert labels.emotions[emotions_vocab.encode("anger")] == 1.0

    assert labels.sentiment == extractor.vocabs.SENTIMENT.encode("positive")
    assert labels.safety == extractor.vocabs.SAFETY.encode("GREEN")
    assert labels.intent == extractor.vocabs.INTENT.encode("share")
    assert labels.ingress == extractor.vocabs.INGRESS.encode("DIARY")

    assert labels.ner_family_labels is not None
    assert labels.ner_family_labels[0].item() == extractor.vocabs.NER_FAMILY.encode("O")
    assert labels.ner_family_labels[1].item() == extractor.vocabs.NER_FAMILY.encode("B-PERSON")

    assert labels.temporal_labels is not None
    assert labels.temporal_labels[-1].item() == extractor.vocabs.TEMPORAL.encode("B-DATE_ABS")

    assert labels.relation_triples is not None
    assert len(labels.relation_triples) == 1
    predicate_id = extractor.vocabs.RELATION_PREDICATES.encode("friend_of")
    assert labels.relation_triples[0][1] == predicate_id


def test_collate_helpers() -> None:
    class_labels = collate_classification_labels([1, None, 3])
    assert class_labels.tolist() == [1, -100, 3]

    multi_labels = collate_multi_label([torch.tensor([1.0, 0.0]), None], num_labels=2)
    expected_multi = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    assert torch.equal(multi_labels, expected_multi)

    token_labels = collate_token_labels([torch.tensor([1, 2, 3]), None], max_len=4)
    expected_token = torch.tensor([[1, 2, 3, -100], [-100, -100, -100, -100]])
    assert torch.equal(token_labels, expected_token)
