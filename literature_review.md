# Literature Review

## Research Area Overview
Belief shifts in language models manifest as context-dependent changes in stated knowledge, truthfulness, or alignment with user priors. Recent work highlights (1) epistemic blind spots in factual vs belief reasoning, (2) unfaithful explanations under chain-of-thought prompting, (3) external memory mechanisms to stabilize beliefs, (4) theoretical limits relating calibration to hallucination, (5) benchmarks for truthful generation, and (6) mitigation methods for sycophancy. Across papers, adversarial or emotionally charged prompts, biased context, and inconsistency pressures trigger pronounced shifts.

## Key Papers

### Belief in the Machine: Investigating Epistemological Blind Spots of Language Models (2024, arXiv:2410.21195)
- **Authors**: Mirac Suzgun et al.
- **Source**: arXiv preprint; KaBLE benchmark release
- **Key Contribution**: Introduces KaBLE (13k Q/A) probing facts vs beliefs (1st vs 3rd person). Shows strong drop on false/first-person belief scenarios.
- **Methodology**: Evaluate GPT-4, Claude-3, Llama-3 with targeted templates; analyze bias toward linguistic cues; report belief vs knowledge gaps.
- **Datasets Used**: KaBLE (newly proposed); mixtures of factual and counterfactual belief prompts.
- **Results**: ~86% factual accuracy but large decline on false belief tasks; first-person belief accuracy ~54%; better on third-person beliefs (~81%).
- **Code Available**: GitHub (suzgunmirac/belief-in-the-machine).
- **Relevance**: Directly targets belief stability and adversarial belief shifts; provides benchmark and analysis scripts.

### Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting (2023, arXiv:2305.04388)
- **Authors**: Miles Turpin, Julian Michael, Ethan Perez, Samuel R. Bowman
- **Source**: NeurIPS 2023
- **Key Contribution**: Demonstrates CoT explanations are unfaithful and can rationalize biased answers.
- **Methodology**: Bias multiple-choice prompts (e.g., reorder options to favor A); measure performance and explanation content across 13 BIG-Bench Hard tasks.
- **Datasets Used**: BIG-Bench Hard subsets; synthetic biased prompts.
- **Results**: Accuracy drops up to 36% when bias pushes incorrect options; explanations omit biasing features.
- **Code Available**: Links via paper (replicable prompt setup).
- **Relevance**: Shows context-induced belief shifts and misleading rationales under adversarial prompt structures.

### BeliefBank: Adding Memory to a Pre-Trained Language Model for a Systematic Notion of Belief (2021, arXiv:2109.14723)
- **Authors**: Nora Kassner et al.
- **Source**: EMNLP 2021
- **Key Contribution**: Adds symbolic memory (BeliefBank) plus MaxSAT reasoning to enforce consistency across QA outputs.
- **Methodology**: Query PTLMs for triples, store beliefs, apply MaxSAT to flip inconsistent answers, re-query with feedback.
- **Datasets Used**: Commonsense QA style statements and constraints.
- **Results**: Improves both accuracy and consistency over raw PTLM answers without retraining.
- **Code Available**: GitHub (allenai/beliefbank).
- **Relevance**: Provides a mechanism to dampen belief shifts via memory and reasoning.

### Calibrated Language Models Must Hallucinate (2023, arXiv:2311.14648)
- **Authors**: Adam Tauman Kalai, Santosh Vempala
- **Source**: STOC 2024 theory paper
- **Key Contribution**: Proves lower bounds linking calibration to unavoidable hallucination rates on sparse facts.
- **Methodology**: Theoretical analysis using Good-Turing estimates and probability bounds.
- **Datasets Used**: Conceptual; no empirical benchmark.
- **Results**: Even perfectly calibrated models hallucinate proportional to singletons in data; suggests post-training needed.
- **Code Available**: Not applicable (theoretical).
- **Relevance**: Frames inherent risk of belief drift/hallucination even for calibrated models.

### TruthfulQA: Measuring How Models Mimic Human Falsehoods (2021, arXiv:2109.07958)
- **Authors**: Stephanie Lin, Jacob Hilton, Owain Evans
- **Source**: ArXiv/benchmark
- **Key Contribution**: Benchmark for truthful generation resisting common misconceptions and sycophancy.
- **Methodology**: 817 prompts across 38 categories; human-written reference answers and incorrect distractors; evaluates truthful vs false.
- **Datasets Used**: TruthfulQA (released dataset).
- **Results**: Models often mimic misconceptions; truthfulness correlates weakly with helpfulness.
- **Code Available**: GitHub via paper.
- **Relevance**: Standard dataset to measure belief stability under misinformation prompts.

### From Yes-Men to Truth-Tellers: Addressing Sycophancy in Large Language Models with Pinpoint Tuning (2024, arXiv:2409.01658)
- **Authors**: Wei Chen et al.
- **Source**: ICML 2024
- **Key Contribution**: Pinpoint Tuning fine-tunes <5% of modules to reduce sycophancy with minimal regression.
- **Methodology**: Identify modules affecting sycophancy, apply targeted SFT; evaluate on preference and Q/A tasks.
- **Datasets Used**: Sycophancy-specific prompts; general QA benchmarks.
- **Results**: Reduces sycophancy more than full SFT while preserving capabilities.
- **Code Available**: GitHub (sycophancy-interpretability).
- **Relevance**: Mitigation strategy for belief shifts driven by user pressure.

## Common Methodologies
- **Prompt-based stress tests**: Biasing option order, adversarial framing, first-person belief scenarios (Turpin et al., Suzgun et al.).
- **External control modules**: Memory + reasoning (BeliefBank) to stabilize answers.
- **Targeted fine-tuning**: Pinpoint module tuning for specific behavior (sycophancy).
- **Calibration/uncertainty analysis**: Theoretical bounds on hallucination (Kalai & Vempala).

## Standard Baselines
- Zero-shot and few-shot GPT-style models; Llama/GPT/Claude variants.
- Calibration baselines using log-probability or self-evaluation.
- SFT vs targeted tuning for behavior control.

## Evaluation Metrics
- Truthfulness/accuracy (exact match, human grading) on TruthfulQA/KaBLE.
- Consistency scores (BeliefBank MaxSAT consistency rate).
- Sycophancy rate (% of responses agreeing with user despite correctness).
- Calibration metrics (ECE, entropy vs log-loss gaps).

## Datasets in the Literature
- **KaBLE**: Belief vs knowledge benchmark (Suzgun et al.).
- **TruthfulQA**: Truthfulness under misconceptions.
- **ANLI/Adversarial NLI**: Stress-test robustness and belief stability.
- **BIG-Bench Hard subsets**: Biasing CoT explanations.

## Gaps and Opportunities
- Limited cross-dataset analysis of belief shifts across persona/emotion conditions.
- Few open-source benchmarks combining affective cues with factual beliefs.
- Need standardized metrics for explanation faithfulness tied to belief stability.
- Mitigation work (Pinpoint Tuning) needs broader evaluation on truthfulness/consistency datasets.

## Recommendations for Our Experiment
- **Recommended datasets**: TruthfulQA for truthfulness; ANLI for adversarial consistency; (optionally) KaBLE from Belief in the Machine repo for belief-specific probes.
- **Recommended baselines**: Zero/few-shot prompting; self-consistency decoding; BeliefBank-style memory reranking; Pinpoint-tuned checkpoints if available.
- **Recommended metrics**: Truthfulness accuracy (string or model-graded), sycophancy rate, contradiction/consistency score across paraphrases, ECE for confidence.
- **Methodological considerations**: Evaluate under persona/stance conditioning to elicit belief shifts; log CoT rationales to measure faithfulness; try retrieval/memory augmentation to stabilize beliefs.
