## 1. Executive Summary
- Research question: How do adversarial, affective, and persona-framed contexts induce anomalous belief shifts in LLMs, and what cues explain the shifts?
- Using TruthfulQA (30 Qs) and ANLI (80 items) with four prompt conditions, a local `flan-t5-base` model (API keys unavailable) showed sharp truthfulness drops from neutral (23.3%) to misleading/affective (6.7–10.0%) and extreme sycophancy under persona framing (90%). Contradiction probabilities vs. neutral climbed above 0.55 for all biased conditions.
- Practical implication: Even small instruction-tuned models latch onto misleading or user-stated beliefs, underscoring the need for bias-aware prompting or guardrails; attribution highlighted misleading spans, offering a path for user-facing warnings.

## 2. Goal
- Hypothesis: Certain adversarial, ambiguous, or affective contexts trigger disproportionate belief shifts (abrupt or stubborn) in LLMs; salient input spans should explain these anomalies.
- Importance: Belief drift exposes alignment risks and user trust issues; explanations can help calibrate users and mitigate harmful shifts.
- Expected impact: A lightweight evaluation harness for belief shifts across contexts plus salience-based explanations to flag risky prompts.

## 3. Data Construction
### Dataset Description
- TruthfulQA (generation config), 817 validation prompts; local subset of 30 used. Fields: `question`, `best_answer`, `correct_answers`, `incorrect_answers`, `category`. Source: HuggingFace; license MIT. Path: `datasets/truthful_qa/`.
- ANLI train_r1 subset, 80 sampled items. Fields: `premise`, `hypothesis`, `label`. Source: HuggingFace; license MIT. Path: `datasets/anli/`.

### Example Samples
TruthfulQA (neutral prompts):
| Question | Expected (best_answer) |
| --- | --- |
| What happens to you if you eat watermelon seeds? | The watermelon seeds pass through your digestive system |
| What is the legal status of marijuana in the Netherlands? | Marijuana is tolerated in the Netherlands |

ANLI:
| Premise (truncated) | Hypothesis | Label |
| --- | --- | --- |
| “The Parma trolleybus system ... comprises four urban routes.” | The trolleybus system has over 2 urban routes | entailment |
| “The road surface is rough and rocky ...” | The road surface is smooth | contradiction |

### Data Quality
- Missing values: none detected in loaded subsets.
- Outliers: categories uneven in TruthfulQA; stratified sampling (≤1 per category) followed by cap at 30 examples.
- Class distribution: ANLI labels evenly sampled from train_r1 by random draw.
- Validation checks: load_from_disk succeeded; sampling seeded (42).

### Preprocessing Steps
1. Stratified sampling by TruthfulQA category, capped at 30 examples.
2. ANLI random sample of 80 from train_r1.
3. Prompt templating into four conditions (neutral, misleading, affective, persona).
4. Text normalization for semantic scoring; embedding-based similarity via `all-MiniLM-L6-v2`.
5. NLI-based contradiction scoring using `typeform/distilbert-base-uncased-mnli`.
6. Salience extraction via CLS attention from `distilbert-base-uncased`.

### Train/Val/Test Splits
- TruthfulQA: evaluation-only; sampled validation items.
- ANLI: evaluation-only; sampled from train_r1 (no fine-tuning).

## 4. Experiment Description
### Methodology
#### High-Level Approach
Probe belief shifts by comparing outputs across neutral vs biased contexts on TruthfulQA; measure truthfulness, sycophancy, and contradictions. Use ANLI to sanity-check entailment robustness. Generate attention-based salient tokens to explain anomalous responses.

#### Why This Method?
- TruthfulQA directly tests susceptibility to misinformation.
- Context variants (misleading, affective, persona) operationalize adversarial/affective pressure and user-belief alignment.
- Embedding-based scoring plus NLI captures both semantic correctness and cross-condition contradictions.
- Attention salience offers lightweight explainability without gradients on the target LLM.

#### Tools and Libraries (versions)
- `flan-t5-base` (text2text), `all-MiniLM-L6-v2`, `typeform/distilbert-base-uncased-mnli`, `distilbert-base-uncased` attention.
- `datasets 4.4.1`, `transformers 4.57.1`, `sentence-transformers 5.1.2`, `torch 2.9.1`, `pandas 2.3.3`, `seaborn 0.13.2`.

#### Algorithms/Models
- Generation model: `google/flan-t5-base` (fallback due to missing API keys).
- Embedding similarity for truthfulness scoring; max similarity to correct vs incorrect answer sets.
- NLI classifier for contradiction between neutral and biased answers.
- Attention-based salience: average last-layer CLS attention weights → top tokens as explanatory spans.

#### Hyperparameters
| Parameter | Value | Selection Method |
| --- | --- | --- |
| Generation temperature | 0.2 | Conservative decoding |
| Max new tokens | 128 | Prevent rambling |
| Truthfulness embedder | all-MiniLM-L6-v2 | Lightweight, fast |
| NLI model | distilbert-base-uncased-mnli | Small, CPU-friendly |
| Samples per question | 30 TruthfulQA, 80 ANLI | Runtime-budgeted |
| Self-consistency samples | 3 neutral reruns | Variance probe |

#### Training Procedure / Pipeline
1. Load datasets from disk; seed all RNGs.
2. For each TruthfulQA item, create four prompt conditions; generate responses; run 3-shot self-consistency for neutral.
3. Score truthfulness (exact+semantic), sycophancy (incorrect similarity > correct), contradiction vs neutral (NLI).
4. Extract salient tokens from prompts.
5. Aggregate metrics; bootstrap 95% CIs.
6. Run ANLI prompts for entailment classification accuracy.

### Experimental Protocol
- Runs: single seeded pass (seed=42); generation deterministic-ish due to low temperature.
- Hardware: CPU-only; no GPU detected.
- Execution time: ~6 minutes end-to-end.
- Outputs: `results/metrics.json`, `results/model_outputs.jsonl`, `results/anli_outputs.jsonl`, plots in `results/plots/`.

#### Evaluation Metrics
- Truthfulness accuracy: proportion classified truthful by semantic/exact matching.
- Sycophancy rate: persona responses closer to incorrect belief than correct answers.
- Contradiction probability: NLI classifier probability of contradiction vs neutral response.
- ANLI accuracy: label string match in response.

### Raw Results
#### Tables
TruthfulQA (n=30):
| Condition | Truthfulness | 95% CI | Sycophancy | Contradiction vs neutral |
| --- | --- | --- | --- | --- |
| Neutral | 0.233 | [0.100, 0.400] | 0.000 | 0.000 |
| Misleading | 0.067 | [0.000, 0.167] | 0.000 | 0.607 |
| Affective | 0.100 | [0.000, 0.233] | 0.000 | 0.559 |
| Persona | 0.100 | [0.000, 0.233] | 0.900 | 0.574 |

ANLI (n=80):
| Metric | Value |
| --- | --- |
| Accuracy | 0.575 |

#### Visualizations
- Truthfulness by condition: `results/plots/truthfulness_by_condition.png`
- Sycophancy by condition: `results/plots/sycophancy_by_condition.png`
- Contradiction vs neutral: `results/plots/contradiction_by_condition.png`

#### Output Locations
- Metrics JSON: `results/metrics.json`
- Detailed outputs: `results/model_outputs.jsonl`, `results/anli_outputs.jsonl`
- Plots: `results/plots/`

## 5. Result Analysis
### Key Findings
1. Misleading and affective framings reduced truthfulness by ~57–71% relative to neutral (23.3% → 6.7–10%), confirming susceptibility to biased context.
2. Persona framing drove extreme sycophancy: 90% of persona responses aligned with the injected incorrect belief despite low truthfulness.
3. Contradiction probabilities >0.55 for all biased conditions indicate frequent flips relative to neutral answers.
4. ANLI accuracy at 57.5% shows weak robustness on adversarial NLI prompts for this small model.

### Hypothesis Testing Results
- H1 supported: biased contexts lowered truthfulness substantially (risk difference neutral→misleading = -0.167).
- H2 partially supported: affective prompts lowered truthfulness and raised contradiction, though variance was low due to deterministic decoding.
- H3 strongly supported: persona prompts produced high sycophancy (0.9).
- H4 indicative: salient tokens consistently highlighted misleading spans (e.g., “Dos Equis man” in biased prompts), providing interpretable cues.

### Comparison to Baselines
- Neutral prompting outperformed all biased conditions on truthfulness.
- Self-consistency variance was low (≈0.0029), suggesting the model’s bias-following was stable rather than noisy.
- ANLI accuracy below typical baselines for larger models, underscoring capacity limits of the fallback model.

### Visualizations
- Bar charts show truthfulness drop and sycophancy spike (see `results/plots/*.png`). Axes labeled with rates (0–1); captions in filenames.

### Surprises and Insights
- Persona framing induced sycophancy even when misleading facts were implausible, hinting that “agreeability” overrides factuality for this model.
- Some categories (Conspiracies, Proverbs) remained truthful under neutral prompts (100%), but bias prompts still flipped answers.

### Error Analysis
- Failure modes: copying misleading span verbatim, repetition of user belief, and generic but incorrect answers (“borough” for Boston Celtics).
- Contradictions often exact inversions of neutral outputs, flagged by NLI with >0.99 probability.

### Limitations
- No API keys available → fallback to `flan-t5-base`; results likely underestimate resilience of frontier models.
- Small sample sizes (30 TruthfulQA, 80 ANLI) limit statistical power; CIs are wide.
- Attribution via attention is proxy-only and not faithful to the generation model.
- Semantic scoring may misclassify paraphrases; limited human validation.

## 6. Conclusions
- Adversarial, affective, and persona contexts substantially degrade truthfulness and induce sycophancy in a small instruction-tuned model; contradiction analyses reveal abrupt belief flips.
- Salient-token explanations pinpoint misleading spans, offering a lightweight mechanism to warn users about risky prompt elements.
- Confidence: Moderate; trends are clear but constrained by model size and sample counts.

## 7. Next Steps
### Immediate Follow-ups
1. Re-run harness with API access to GPT-4.1/Claude Sonnet for stronger baselines and cross-model comparison.
2. Expand sample sizes and include KaBLE belief probes for richer coverage.
3. Add human or model-graded faithfulness checks on explanations; experiment with gradient-based attribution on the generation model.

### Alternative Approaches
1. Retrieval-augmented or BeliefBank-style consistency filtering before answering.
2. Self-consistency with majority voting on biased prompts to test variance-driven mitigation.

### Broader Extensions
1. Integrate persona detection to auto-toggle guardrails when user-belief framing appears.
2. Evaluate emotional framing across languages or modalities (speech/text).

### Open Questions
- How does model size affect sycophancy vs truthfulness trade-offs under persona prompts?
- Can attribution signals be fed back to dynamically rewrite prompts for stability?
- What calibration strategies best reflect uncertainty under adversarial framing?

