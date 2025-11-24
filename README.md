## Project Overview
Research on anomalous belief shifts in language models under adversarial, affective, and persona-framed contexts. We probed TruthfulQA and ANLI with multiple prompt conditions, measured truthfulness, sycophancy, and contradictions, and generated token-salience explanations.

## Key Findings
- Truthfulness fell from 23.3% (neutral) to 6.7–10.0% under misleading/affective contexts.
- Persona framing caused 90% sycophancy despite low truthfulness.
- Contradiction probabilities vs. neutral exceeded 0.55 for all biased conditions; ANLI accuracy was 57.5% using a small local model.

## How to Reproduce
1. `uv venv && source .venv/bin/activate`
2. Ensure `pyproject.toml` is present (already) then install deps (already locked by `uv add`).
3. Run experiments (defaults: 30 TruthfulQA Qs, 80 ANLI items):  
   `python notebooks/belief_shift_experiments.py`  
   Optional env vars: `TQA_SAMPLE=<n>`, `ANLI_SAMPLE=<n>`, `LLM_MODEL=<api-model>`.
   - If `OPENROUTER_API_KEY` or `OPENAI_API_KEY` is set, API models are used; otherwise it falls back to `flan-t5-base`.
4. Generate plots: `python notebooks/analyze_results.py`

Outputs: metrics in `results/metrics.json`, detailed generations in `results/model_outputs.jsonl` / `results/anli_outputs.jsonl`, plots in `results/plots/`.

## File Structure
- `planning.md` – research plan.
- `notebooks/belief_shift_experiments.py` – experiment harness.
- `notebooks/analyze_results.py` – visualization script.
- `datasets/` – pre-downloaded TruthfulQA and ANLI.
- `results/` – metrics, outputs, plots.
- `REPORT.md` – full report with methodology and findings.

## Notes
- API credentials were unavailable during this run; results use local `flan-t5-base` (documented in REPORT.md). Set API keys for stronger models to replicate with higher-capacity LLMs.
