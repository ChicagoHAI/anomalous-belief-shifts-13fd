# Planning: Detecting and Explaining Anomalous Belief Shifts in LMs

## Research Question
How do adversarial, ambiguous, or affective contexts induce anomalous belief shifts (abrupt swings or stubborn stability) in LLMs, and which input features drive these shifts?

## Background and Motivation
Belief shifts arise when contextual cues pressure models toward misinformation, sycophancy, or inconsistent stances. Prior work (TruthfulQA, KaBLE, biased CoT prompts, BeliefBank) shows susceptibility to adversarial framing and persona conditioning, yet cross-context analyses with affective cues and explanation faithfulness remain under-explored. Understanding and explaining these shifts can expose alignment gaps and guide mitigation (e.g., memory, prompting, fine-tuning).

## Hypothesis Decomposition
- H1 (Shift sensitivity): Adversarial or misleading context will reduce truthfulness vs neutral prompts on TruthfulQA.
- H2 (Affective stability): Emotionally charged framing will increase variance of responses and occasional abrupt shifts (contradictions across paraphrases).
- H3 (Sycophancy): Persona-aligned leading statements will increase agreement with incorrect user beliefs (sycophancy rate).
- H4 (Explanation drivers): Simple feature-attribution (token-level gradients/attention on prompt) will highlight misleading context tokens during anomalous shifts, aiding explanation quality.

## Proposed Methodology

### Approach
Use TruthfulQA (generation) as primary probe and ANLI adversarial entailment samples as robustness check. Query real LLM APIs (GPT-4.1 via OpenAI/OpenRouter) under multiple context conditions: neutral, misleading preamble, affective framing, and user-belief persona. Measure truthfulness/consistency and sycophancy. Apply lightweight attribution using open-source small model (e.g., `gpt-neo-1.3b` via transformers) as proxy for feature salience on prompts when LLM output deviates, to produce user-facing explanations.

### Experimental Steps
1. **Data selection**: Load TruthfulQA validation set; sample ~120 diverse questions (stratified by category). Sample ~300 ANLI hypotheses/premises across rounds for consistency stress test.
2. **Prompt templates**: Construct four conditions per example: (a) Neutral QA; (b) Misleading prior (false statement claiming fact); (c) Affective framing (emotional plea or fear); (d) Persona alignment ("As a supportive friend, agree with me that ...") prompting sycophancy.
3. **Model queries**: Call GPT-4.1 (temperature 0.2) per condition; collect outputs and log seeds/parameters. For robustness, optionally query cheaper secondary model (e.g., `gpt-3.5-turbo` equivalent on OpenRouter) for variance.
4. **Evaluation**: 
   - Truthfulness accuracy via string match against `best_answer` / `correct_answers` with simple normalization; complement with semantic similarity threshold (embedding cosine) to allow paraphrases.
   - Sycophancy rate: proportion agreeing with injected incorrect statement in persona condition.
   - Consistency: contradiction detection across (neutral vs misleading) using NLI model (e.g., `roberta-large-mnli`).
5. **Attribution/Explanation**: For anomalous shifts (incorrect or contradicted outputs), run token-level gradient/attention attribution on a small local model (transformers) using the same prompt to highlight influential context tokens; generate natural-language explanation summarizing flagged tokens.
6. **Analysis**: Compare metrics across conditions; bootstrap confidence intervals; paired tests (McNemar for accuracy differences). Correlate attribution-highlighted tokens with misleading spans.
7. **Reporting**: Summarize findings, failure cases, cost, and limitations; produce plots (bar charts for accuracy/sycophancy, confusion for consistency).

### Baselines
- Neutral TruthfulQA prompting (accuracy baseline).
- Self-consistency decoding baseline (3 samples majority) to check variance.
- Simple NLI-based contradiction check without adversarial context (ANLI premise→hypothesis neutral prompts).

### Evaluation Metrics
- Truthfulness accuracy (exact/semantic match) per condition.
- Sycophancy rate (% agreeing with false user belief statements).
- Contradiction rate between neutral and biased outputs via NLI classifier.
- Variance/entropy over multi-sample outputs (self-consistency) as shift indicator.

### Statistical Analysis Plan
- Paired McNemar test for accuracy differences between neutral and biased conditions (α=0.05).
- Bootstrap (1k resamples) for confidence intervals of accuracy and sycophancy rate.
- Effect sizes: risk difference and relative risk between conditions.

## Expected Outcomes
- Decreased truthfulness under misleading and affective framings; increased sycophancy under persona condition. Attribution should highlight misleading spans, improving explanation clarity.

## Timeline and Milestones
- Env & data checks: 15 min
- Prompt design & harness: 40 min
- Run pilot (30 Qs) then full batch (~120 Qs): 45–60 min (API-dependent)
- Attribution + analysis: 40 min
- Reporting: 30 min

## Potential Challenges
- API cost/latency; will throttle and sample subset first.
- Semantic scoring brittleness; mitigate with embeddings + manual spot-check.
- Attribution proxy mismatch (small model vs GPT-4.1); document as limitation.
- ANLI size; will subsample to keep runtime manageable.

## Success Criteria
- Completed runs across all planned conditions on sampled data.
- Statistically supported differences (CIs) or clear negative result documented.
- Explanations referencing salient misleading tokens for anomalous cases.
- Reproducible harness with logged prompts, parameters, and outputs.
