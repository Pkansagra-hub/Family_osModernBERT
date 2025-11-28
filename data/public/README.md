# Public Datasets Directory

This directory contains or links to public datasets used in Stage A training.

## Datasets

### NER
- **CoNLL-2003**: Standard NER benchmark
  - Source: HuggingFace `conll2003`
  - License: [Check license]
  
- **OntoNotes 5.0**: Larger NER dataset (optional)
  - Source: HuggingFace `tner/ontonotes5`

### Sentiment
- **SST-2**: Stanford Sentiment Treebank
  - Source: HuggingFace `stanfordnlp/sst2`
  - License: [Check license]

### Emotions
- **GoEmotions**: Multi-label emotion classification
  - Source: HuggingFace `google-research-datasets/go_emotions`
  - License: Apache 2.0

### Safety
- **Jigsaw Toxicity**: Toxicity classification
  - Source: HuggingFace `jigsaw_toxicity_pred`
  - License: [Check Kaggle terms]

- **Civil Comments**: Toxicity with identity annotations
  - Source: HuggingFace `civil_comments`

### NLI
- **MNLI**: Multi-Genre NLI
  - Source: HuggingFace `nli` (multi_nli config)
  
- **SNLI**: Stanford NLI
  - Source: HuggingFace `stanfordnlp/snli`

### Embeddings
- **STS-B**: Semantic Textual Similarity Benchmark
  - Source: HuggingFace `sentence-transformers/stsb`

## Notes

Most datasets are loaded directly from HuggingFace Hub.
This directory is for:
- Local caches
- Preprocessed versions
- Custom splits
- License documentation

## License Compliance

Before using any dataset, verify:
1. License allows commercial use
2. Proper attribution is provided
3. Any restrictions are documented
