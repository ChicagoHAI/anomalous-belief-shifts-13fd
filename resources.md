# Resources Catalog

## Summary
Collected papers, datasets, and code to study anomalous belief shifts in language models.

### Papers (6)
| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Belief in the Machine: Investigating Epistemological Blind Spots of LMs | Suzgun et al. | 2024 | papers/2410.21195_belief_in_the_machine.pdf | KaBLE benchmark for belief vs knowledge reasoning |
| Language Models Don't Always Say What They Think | Turpin et al. | 2023 | papers/2305.04388_cot_unfaithful_explanations.pdf | Biased CoT explanations and accuracy drops |
| BeliefBank: Adding Memory to a Pre-Trained LM | Kassner et al. | 2021 | papers/2109.14723_beliefbank.pdf | Memory + MaxSAT to enforce consistency |
| Calibrated Language Models Must Hallucinate | Kalai, Vempala | 2023 | papers/2311.14648_calibrated_lms_hallucinate.pdf | Theoretical link between calibration and hallucination |
| TruthfulQA: Measuring How Models Mimic Human Falsehoods | Lin et al. | 2021 | papers/2109.07958_truthfulqa.pdf | Truthfulness benchmark across 38 categories |
| From Yes-Men to Truth-Tellers: Addressing Sycophancy with Pinpoint Tuning | Chen et al. | 2024 | papers/2409.01658_sycophancy_pinpoint_tuning.pdf | Fine-tuning method to mitigate sycophancy |

See papers/README.md for details.

### Datasets (2)
| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| TruthfulQA (generation) | HuggingFace | 817 prompts | Truthfulness generation | datasets/truthful_qa/ | Small benchmark; samples in samples.json |
| ANLI | HuggingFace | ~166k train across rounds | Adversarial NLI | datasets/anli/ | Robustness stress-test; samples in samples.json |

See datasets/README.md for download instructions.

### Code Repositories (2)
| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| belief-in-the-machine | https://github.com/suzgunmirac/belief-in-the-machine | KaBLE benchmark + evaluation scripts | code/belief-in-the-machine | Includes prompts and analysis notebooks |
| sycophancy-interpretability | https://github.com/yellowtownhz/sycophancy-interpretability | Pinpoint Tuning for sycophancy mitigation | code/sycophancy-interpretability | Fine-tuning/interpretability utilities |

### Resource Gathering Notes
- **Search strategy**: arXiv API queries for belief, sycophancy, calibration; targeted paper IDs; selected datasets from HuggingFace aligned to belief robustness (TruthfulQA, ANLI); cloned repos referenced in papers.
- **Selection criteria**: Recency (2023-2024 emphasis), direct relevance to belief shifts, availability of code/datasets, diversity of methods (evaluation, mitigation, theory).
- **Challenges encountered**: None significant; TruthfulQA required versionless PDF link.
- **Gaps and workarounds**: KaBLE dataset accessed via repo; not mirrored separately here.

### Recommendations for Experiment Design
1. **Primary dataset(s)**: TruthfulQA for truthfulness shifts; ANLI and KaBLE (from repo) for adversarial consistency and belief reasoning.
2. **Baseline methods**: Zero/few-shot prompting; self-consistency decoding; BeliefBank-style memory reranking; Pinpoint-tuned model (if checkpoints available) vs vanilla model.
3. **Evaluation metrics**: Truthfulness accuracy/human grading; sycophancy rate (agreement with user’s stated belief); contradiction rate across paraphrases; ECE for confidence calibration.
4. **Code to adapt/reuse**: KaBLE evaluation scripts; Pinpoint Tuning pipeline for targeted mitigation; BeliefBank MaxSAT consistency approach (from paper code links).
