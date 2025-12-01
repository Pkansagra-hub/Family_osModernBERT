# Research Paper Requirements: FamilyOS-BERT

> **Working Title:** "FamilyOS-BERT: A Culturally-Aware Multi-Task Encoder for Family Conversation Understanding"
>
> **Target Venues:** ACL, EMNLP, NAACL, EACL, or AAAI (NLP track)

---

## 1. Core Research Contributions (Must Validate)

### 1.1 Primary Claims to Prove

| # | Claim | Validation Method | Target Metric |
|---|-------|-------------------|---------------|
| **C1** | First unified encoder for family domain with 12 capabilities | Literature survey + ablation | No prior work exists |
| **C2** | Culturally-aware safety classification reduces false positives on Indian English | Comparative evaluation | ≥50% FP reduction vs baselines |
| **C3** | Family-specific NER outperforms generic NER on family conversations | Benchmark comparison | ≥5% F1 improvement |
| **C4** | Multi-task learning improves over single-task models | Ablation study | ≥2% average improvement |
| **C5** | Hierarchical safety bands enable better triage than flat classification | User study or simulation | Faster response time, better routing |

---

## 2. Experimental Requirements

### 2.1 Baselines to Compare Against

#### Safety Classification

- [ ] **Perspective API** (Google) - Industry standard toxicity
- [ ] **Jigsaw Toxicity** (Kaggle winner) - Open source baseline
- [ ] **HateBERT** - Hate speech detection
- [ ] **ToxiGen** - Implicit toxicity
- [ ] **ModerateHatespeech** - Multi-class toxicity

#### NER

- [ ] **SpaCy en_core_web_trf** - Industrial NER
- [ ] **Flair NER** - Sequence labeling SOTA
- [ ] **BERT-NER** (CoNLL-2003 fine-tuned) - Standard baseline
- [ ] **GLiNER** - Zero-shot NER

#### Multi-Task

- [ ] **MT-DNN** (Microsoft) - Multi-task benchmark
- [ ] **Single-task BERT** × N tasks - Ablation baseline

#### Embeddings

- [ ] **Sentence-BERT** - Standard embedding baseline
- [ ] **BGE-base** - SOTA embedding model
- [ ] **E5-base** - Microsoft embedding model

### 2.2 Ablation Studies Required

| Ablation | What It Tests | Expected Outcome |
|----------|---------------|------------------|
| **A1:** Remove EMA | Value of model averaging | -0.5 to -1.5% across tasks |
| **A2:** Remove uncertainty weighting | Auto task balancing benefit | -1 to -2% on minority tasks |
| **A3:** Single LR vs head-wise LR | Head-wise LR benefit | -1 to -3% on heads |
| **A4:** Remove Stage A replay in Stage B | Forgetting prevention | >2% drop on Stage A tasks |
| **A5:** Remove cultural patterns from safety | Cultural awareness value | +20-50% FP on Indian English |
| **A6:** Freeze encoder in Stage B | LoRA vs full fine-tune | Trade-off analysis |
| **A7:** Remove family-specific emotions | Family emotion value | Lower recall on family sentiments |

### 2.3 Statistical Significance

- [ ] Report **mean ± std** across 3-5 random seeds
- [ ] Compute **p-values** for all baseline comparisons (paired t-test or bootstrap)
- [ ] Use **confidence intervals** (95%) for key metrics
- [ ] Report **effect size** (Cohen's d) for major claims

---

## 3. Dataset Requirements

### 3.1 Datasets to Create/Curate

| Dataset | Size Target | Current Status | Required For |
|---------|-------------|----------------|--------------|
| **FamilyNER** | ≥5,000 annotated | ~12K silver | Claim C3 |
| **FamilySafety** | ≥3,000 annotated | ~14K silver | Claim C2 |
| **IndianHyperbole** | ≥500 examples | ~50 seed | Claim C2 |
| **FamilyEmotions** | ≥2,000 annotated | ~1K | Claim C4 |
| **FamilyIngress** | ≥3,000 annotated | ~12K silver | Domain routing |
| **FamilyRelations** | ≥2,000 annotated | ~12K silver | Relation extraction |
| **FamilyTemporal** | ≥2,000 annotated | ~14K silver | Timeline feature |

### 3.2 Dataset Quality Requirements

- [ ] **Inter-annotator agreement (IAA)**: Cohen's κ ≥ 0.7 for all tasks
- [ ] **Gold standard test set**: ≥500 human-annotated examples per task
- [ ] **Annotation guidelines**: Documented with examples
- [ ] **Demographic diversity**: Multiple annotators, varied backgrounds
- [ ] **Error analysis**: Document common annotation errors

### 3.3 Dataset Documentation (Datasheets)

For each dataset, document:

- [ ] Motivation (why created)
- [ ] Composition (size, splits, label distribution)
- [ ] Collection process (how gathered)
- [ ] Preprocessing steps
- [ ] Annotator information (number, training, compensation)
- [ ] Intended use cases
- [ ] Limitations and biases
- [ ] Licensing terms

---

## 4. Evaluation Metrics

### 4.1 Per-Task Metrics

| Task | Primary Metric | Secondary Metrics | Threshold |
|------|----------------|-------------------|-----------|
| NER General | Entity F1 | Precision, Recall, per-entity F1 | ≥88% |
| NER Family | Entity F1 | Precision, Recall, per-entity F1 | ≥85% |
| Sentiment | Accuracy | Macro F1, per-class F1 | ≥92% |
| Emotions | Macro F1 | Accuracy, per-emotion F1 | ≥45% |
| Safety Generic | Macro F1 | AUROC, per-class recall | ≥70% |
| Safety FamilyOS | CRISIS Recall | Overall Accuracy, FPR | ≥98% |
| Ingress | Accuracy | Macro F1, confusion matrix | ≥85% |
| NLI | Accuracy | Macro F1 | ≥84% |
| Embedding | Spearman | Pearson, retrieval MRR | ≥0.80 |
| Relation | F1 | Precision, Recall | ≥80% |
| Intent | Accuracy | Macro F1 | ≥85% |
| Temporal | Entity F1 | Precision, Recall | ≥82% |

### 4.2 System-Level Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Inference latency** | ms per example (batch=1) | ≤50ms on GPU |
| **Throughput** | examples/second (batch=32) | ≥500/s on A100 |
| **Model size** | Parameters | ~150M |
| **Memory footprint** | GPU RAM at inference | ≤2GB |
| **Forgetting rate** | Drop on Stage A after Stage B | ≤2% |

### 4.3 Cultural Robustness Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Indian English FPR** | False CRISIS on hyperbole | ≤2% |
| **Kinship NER recall** | Recognition of Indian kinship terms | ≥90% |
| **Joint family handling** | Correct relationship inference | ≥85% |

---

## 5. Reproducibility Requirements

### 5.1 Code Release

- [ ] **Full training code** on GitHub (Apache 2.0 or MIT license)
- [ ] **Configuration files** for all experiments
- [ ] **Evaluation scripts** for all baselines
- [ ] **Pre-trained model weights** on HuggingFace Hub
- [ ] **Docker container** for environment reproducibility
- [ ] **README** with clear instructions

### 5.2 Training Details to Document

| Detail | What to Report |
|--------|----------------|
| **Hardware** | GPU type, count, memory |
| **Training time** | Wall-clock hours per phase |
| **Hyperparameters** | All values, how selected |
| **Random seeds** | Seeds used for reproducibility |
| **Software versions** | PyTorch, Transformers, etc. |
| **Compute cost** | Estimated cloud cost ($) |
| **Carbon footprint** | CO2 emissions estimate |

### 5.3 Model Card

Create HuggingFace model card with:

- [ ] Model description
- [ ] Intended use cases
- [ ] Limitations
- [ ] Training data summary
- [ ] Evaluation results
- [ ] Ethical considerations
- [ ] Bias analysis

---

## 6. Human Evaluation (Optional but Strengthens Paper)

### 6.1 Safety Classification Study

| Study | Participants | Method | Metric |
|-------|--------------|--------|--------|
| **Indian English understanding** | 20+ Indian English speakers | Rate safety predictions | Agreement rate |
| **Triage speed** | 10+ support agents | Compare flat vs hierarchical | Time to correct action |
| **Cultural false positive** | 30+ native speakers | Rate if CRISIS appropriate | FPR on hyperbole |

### 6.2 NER Quality Study

| Study | Participants | Method | Metric |
|-------|--------------|--------|--------|
| **Family entity recognition** | 20+ users | Review predictions | Precision/recall rating |
| **Kinship term coverage** | 30+ multilingual users | Missing terms survey | Coverage percentage |

---

## 7. Ethical Considerations

### 7.1 Privacy & Safety

- [ ] **No real user data** in published datasets (synthetic only, or anonymized)
- [ ] **IRB approval** if using any real user conversations
- [ ] **Consent** documentation if crowdsourcing annotations
- [ ] **Safety testing** before deployment (red-teaming)
- [ ] **Bias audit** across demographics

### 7.2 Potential Harms to Address

| Risk | Mitigation | Documentation |
|------|------------|---------------|
| False negatives on CRISIS | Conservative thresholds | Calibration report |
| Privacy leakage | No real names in data | Data collection protocol |
| Cultural bias | Multi-cultural annotators | Annotator demographics |
| Misuse for surveillance | Use case guidelines | Model card |

### 7.3 Limitations Section

Document explicitly:

- [ ] Languages supported (English only initially)
- [ ] Cultural contexts tested (Indian English focus)
- [ ] Family structures assumed (nuclear + joint)
- [ ] Age groups covered
- [ ] Failure modes observed

---

## 8. Writing & Submission Checklist

### 8.1 Paper Structure

| Section | Page Target | Status |
|---------|-------------|--------|
| Abstract | 0.25 pages | ⬜ |
| Introduction | 1 page | ⬜ |
| Related Work | 1 page | ⬜ |
| FamilyOS-BERT Architecture | 1.5 pages | ⬜ |
| Training Methodology | 1 page | ⬜ |
| Datasets | 1 page | ⬜ |
| Experiments & Results | 2 pages | ⬜ |
| Ablation Studies | 0.5 pages | ⬜ |
| Analysis & Discussion | 0.5 pages | ⬜ |
| Ethical Considerations | 0.25 pages | ⬜ |
| Conclusion | 0.25 pages | ⬜ |
| **Total** | **8 pages** (+ references, appendix) | |

### 8.2 Supplementary Materials

- [ ] Appendix A: Full hyperparameter tables
- [ ] Appendix B: Dataset statistics and examples
- [ ] Appendix C: Per-entity/class detailed results
- [ ] Appendix D: Annotation guidelines
- [ ] Appendix E: Error analysis examples
- [ ] Appendix F: Cultural expression examples

### 8.3 Pre-Submission Checklist

- [ ] **Anonymous submission** (no author names in PDF)
- [ ] **Supplementary materials** prepared
- [ ] **Code repository** anonymized (for double-blind)
- [ ] **Conflict of interest** declarations
- [ ] **Page limit** compliance
- [ ] **Formatting** per venue guidelines (ACL/EMNLP template)
- [ ] **Spellcheck** and grammar review
- [ ] **All claims backed by experiments**
- [ ] **Figures are readable** in black & white

---

## 9. Timeline & Milestones

### 9.1 Research Milestones

| Milestone | Target Date | Deliverable | Status |
|-----------|-------------|-------------|--------|
| **M1: Stage A Training** | Dec 2025 | Trained model, eval results | 🔄 In Progress |
| **M2: Stage B Training** | Dec 2025 | FamilyOS-adapted model | ⬜ |
| **M3: Baseline Comparisons** | Jan 2026 | All baseline results | ⬜ |
| **M4: Ablation Studies** | Jan 2026 | Ablation table complete | ⬜ |
| **M5: Human Evaluation** | Feb 2026 | User study results | ⬜ |
| **M6: Paper Draft v1** | Feb 2026 | Complete first draft | ⬜ |
| **M7: Internal Review** | Mar 2026 | Feedback incorporated | ⬜ |
| **M8: Submission** | Mar-Apr 2026 | Submit to ACL/EMNLP | ⬜ |

### 9.2 Venue Deadlines (2026 Estimated)

| Venue | Submission Deadline | Notification | Conference |
|-------|---------------------|--------------|------------|
| ACL 2026 | ~Feb 2026 | ~May 2026 | Aug 2026 |
| EMNLP 2026 | ~Jun 2026 | ~Sep 2026 | Nov 2026 |
| NAACL 2026 | ~Dec 2025 | ~Mar 2026 | Jun 2026 |
| EACL 2026 | ~Oct 2025 | ~Jan 2026 | Apr 2026 |
| AAAI 2026 | ~Aug 2025 | ~Nov 2025 | Feb 2026 |

---

## 10. Open Questions to Resolve

### 10.1 Technical Questions

| Question | Options | Decision Needed By |
|----------|---------|-------------------|
| Release real family data? | Synthetic only vs anonymized real | Before M5 |
| Open-source model weights? | Full release vs API only | Before M8 |
| Include FamilyOS branding? | Generic "Family-BERT" vs "FamilyOS-BERT" | Before M6 |
| Multi-lingual extension? | English only vs Hindi+English | Future work |

### 10.2 Scope Questions

| Question | Options | Recommendation |
|----------|---------|----------------|
| How many capabilities? | All 12 vs core 8 | Start with core, ablate others |
| Human evaluation scope? | Safety only vs all tasks | Safety + NER minimum |
| Baseline breadth? | 3-5 vs 10+ baselines | 5-7 strong baselines |

---

## 11. Success Criteria

### 11.1 Minimum Viable Paper (Must Have)

- [ ] Stage A + Stage B training complete
- [ ] All 12 capabilities evaluated
- [ ] 3+ baselines per major task (Safety, NER, Emotions)
- [ ] Ablation on EMA, head-wise LR, uncertainty weighting
- [ ] Cultural robustness (Indian English) demonstrated
- [ ] Forgetting gates pass (≤2% drop)
- [ ] Code + model released

### 11.2 Strong Paper (Should Have)

- [ ] Human evaluation on safety classification
- [ ] 5+ baselines per major task
- [ ] Cross-lingual analysis (Hindi-English code-switching)
- [ ] Real user study on triage speed
- [ ] Detailed error analysis
- [ ] Bias audit across demographics

### 11.3 Outstanding Paper (Nice to Have)

- [ ] Published dataset with annotation guidelines
- [ ] Deployed system with usage statistics
- [ ] Follow-up improvements based on deployment
- [ ] Industry adoption case study
- [ ] Extension to other cultural contexts

---

## 12. References to Cite

### 12.1 Foundation Models

- ModernBERT (2024) - Base architecture
- BERT (Devlin et al., 2019) - Original transformer encoder
- RoBERTa (Liu et al., 2019) - Improved pretraining

### 12.2 Multi-Task Learning

- MT-DNN (Liu et al., 2019) - Multi-task NLU
- Uncertainty Weighting (Kendall et al., 2018) - Task balancing
- Curriculum Learning (Bengio et al., 2009) - Training strategy

### 12.3 Safety & Toxicity

- Perspective API (Jigsaw, 2017) - Toxicity baseline
- HateBERT (Caselli et al., 2021) - Hate speech model
- ToxiGen (Hartvigsen et al., 2022) - Implicit toxicity

### 12.4 NER

- CoNLL-2003 (Tjong Kim Sang, 2003) - NER benchmark
- OntoNotes (Weischedel et al., 2013) - Extended NER
- Few-NERD (Ding et al., 2021) - Fine-grained NER

### 12.5 Emotion Detection

- GoEmotions (Demszky et al., 2020) - Emotion taxonomy
- EmoNet (Abdul-Mageed & Ungar, 2017) - Emotion classification

### 12.6 Cultural NLP

- Indian English variations (Kachru, 1983)
- Cross-cultural NLP (Hershcovich et al., 2022)

---

## Summary: Pre-Publication Checklist

```
BEFORE SUBMITTING:

□ All experiments complete with statistical significance
□ All baselines implemented and compared
□ All ablation studies done
□ Dataset documentation complete (datasheets)
□ Code repository ready for release
□ Model weights on HuggingFace
□ Ethics section written
□ Limitations documented
□ Paper formatted per venue guidelines
□ Anonymous submission prepared
□ Supplementary materials ready
□ Co-author approvals obtained
```

---

**Document Version:** 1.0
**Created:** November 30, 2025
**Last Updated:** November 30, 2025
**Status:** Requirements Gathering Phase
