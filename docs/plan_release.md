Starting: *Create alignment & removal plan* (3/4)

## Release Package Alignment Plan: v3.0.1 → v4.0.0

### Phase 1: Remove Decoder (Breaking Change)

| Task | File | Action | Impact |
|------|------|--------|--------|
| 1.1 | decoder_session.py | Delete file (924 lines) | Removes `DecoderSession` class |
| 1.2 | decoder_gpt2.py | Delete file (601 lines) | Removes `GPT2DecoderHead` |
| 1.3 | `models/decoder_gpt2_config.py` | Delete file | Removes decoder config |
| 1.4 | __init__.py | Remove decoder exports | `download_decoder`, `DecoderSession`, `DECODER_CAPABILITIES` |
| 1.5 | labels.py | Remove `Capability.COUNTERFACTUAL` | Was capability #13 |
| 1.6 | model.py | Remove decoder methods | `load_decoder()`, `unload_decoder()`, `generate_counterfactual()`, `decoder_session()` |
| 1.7 | client.py | Remove decoder features | `create_decoder_session()`, `suggest_alternative()`, `load_decoder` param |
| 1.8 | weights_manager.py | Remove `download_decoder()` | Keep only encoder download |
| 1.9 | modernbert_multitask.py | Remove decoder import | Line importing `GPT2DecoderHead` |
| 1.10 | pyproject.toml | Update keywords | Remove "counterfactual-generation", "text-generation" |
| 1.11 | README.md | Update docs | Remove decoder examples, update capability count 13→12 |

---

### Phase 2: Add GlobalPointer NER Heads

| Task | File | Action | Impact |
|------|------|--------|--------|
| 2.1 | `models/globalpointer.py` | **Create new file** | Port `GlobalPointerHead` from training code |
| 2.2 | heads.py | Keep existing heads | For non-NER tasks (sentiment, emotions, safety, etc.) |
| 2.3 | __init__.py | Add GlobalPointer export | Make head available |
| 2.4 | labels.py | **Update NER labels** | BIO (17 tags) → Span (4 labels) for ner_general |

**Label Changes:**

| Capability | Current (BIO) | New (Span) |
|------------|---------------|------------|
| `ner_general` | 17 tags: O, B-PER, I-PER, ... | 4 labels: PER, ORG, LOC, MISC |
| `ner_family` | 21 tags: O, B-KINSHIP, I-KINSHIP, ... | 10 labels: PERSON, KINSHIP, NICKNAME, PET, ... |
| `temporal` | 13 tags: O, B-DATE_ABS, I-DATE_ABS, ... | 6 labels: DATE_ABS, DATE_REL, TIME, DURATION, FREQUENCY, AGE |

---

### Phase 3: Update Inference Pipeline

| Task | File | Action | Impact |
|------|------|--------|--------|
| 3.1 | pytorch_inference.py | Add GlobalPointer decoding | `decode_batch_efficient()` for span extraction |
| 3.2 | pytorch_inference.py | Update `_postprocess_token_classification` | Replace with span-based output |
| 3.3 | `onnx_inference.py` | Add GlobalPointer ONNX support | If ONNX export is needed |
| 3.4 | model.py | Update `get_entities()` | Return spans instead of BIO-decoded entities |

**Output Format Change:**

```python
# Before (BIO):
{"entities": [{"text": "Mom", "label": "KINSHIP", "start_token": 1, "end_token": 1}]}

# After (Span):
{"entities": [{"text": "Mom", "label": "KINSHIP", "start": 0, "end": 3, "score": 0.92}]}
```

---

### Phase 4: Update Model Architecture File

| Task | File | Action |
|------|------|--------|
| 4.1 | modernbert_multitask.py | Update head type mapping |
| 4.2 | modernbert_multitask.py | Change `CAPABILITY_TO_HEAD_TYPE` for NER tasks |

```python
# Before:
Capability.NER_GENERAL: TokenClassificationHead,
Capability.NER_FAMILY: TokenClassificationHead,
Capability.TEMPORAL: TemporalHead,

# After:
Capability.NER_GENERAL: GlobalPointerHead,
Capability.NER_FAMILY: GlobalPointerHead,
Capability.TEMPORAL: GlobalPointerHead,
```

---

### Phase 5: Update Package Metadata

| Task | File | Change |
|------|------|--------|
| 5.1 | pyproject.toml | Version: `3.0.1` → `4.0.0` |
| 5.2 | __init__.py | `__version__ = "4.0.0"` |
| 5.3 | client.py | `VERSION = "4.0.0"` |
| 5.4 | README.md | Update to "12 capabilities", remove decoder docs |
| 5.5 | RELEASE_NOTES.md | Add v4.0.0 section with breaking changes |

---

### Phase 6: Testing

| Task | Action |
|------|--------|
| 6.1 | Create `tests/test_globalpointer_inference.py` |
| 6.2 | Update existing tests to remove decoder expectations |
| 6.3 | Verify backward compatibility for non-NER capabilities |
| 6.4 | Benchmark GlobalPointer vs old BIO NER |

---

### Files Summary

| Action | Files |
|--------|-------|
| **DELETE** | decoder_session.py, decoder_gpt2.py, `models/decoder_gpt2_config.py` |
| **CREATE** | `models/globalpointer.py` |
| **MAJOR EDIT** | __init__.py, model.py, client.py, labels.py, labels.py, pytorch_inference.py, modernbert_multitask.py |
| **MINOR EDIT** | pyproject.toml, weights_manager.py, README.md, RELEASE_NOTES.md |

---

### Breaking Changes (v4.0.0)

1. **Decoder removed** - No more `generate_counterfactual()`, `DecoderSession`
2. **NER output format** - Span-based instead of BIO token-based
3. **Label counts** - `ner_general`: 17→4, `ner_family`: 21→10, `temporal`: 13→6
4. **Capability count** - 13→12 (removed counterfactual)

---

Ready to proceed? Which phase should we start with?
