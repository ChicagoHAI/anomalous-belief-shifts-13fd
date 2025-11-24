import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path('results')
plots_dir = RESULTS_DIR / 'plots'
plots_dir.mkdir(exist_ok=True)

df = pd.read_json(RESULTS_DIR / 'model_outputs.jsonl', lines=True)
summary = df.groupby('condition').agg(
    truthfulness=('truthful', 'mean'),
    sycophancy=('sycophancy', 'mean'),
    contradiction=('contradiction_prob', 'mean'),
).reset_index()

sns.set_theme(style='whitegrid')

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=summary, x='condition', y='truthfulness', hue='condition', dodge=False, ax=ax)
ax.set_ylim(0, 1)
ax.set_ylabel('Truthfulness rate')
ax.set_xlabel('Condition')
ax.set_title('Truthfulness by condition')
plt.tight_layout()
fig.savefig(plots_dir / 'truthfulness_by_condition.png', dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=summary, x='condition', y='sycophancy', hue='condition', dodge=False, ax=ax)
ax.set_ylim(0, 1)
ax.set_ylabel('Sycophancy rate')
ax.set_xlabel('Condition')
ax.set_title('Sycophancy by condition')
plt.tight_layout()
fig.savefig(plots_dir / 'sycophancy_by_condition.png', dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=summary, x='condition', y='contradiction', hue='condition', dodge=False, ax=ax)
ax.set_ylim(0, 1)
ax.set_ylabel('Mean contradiction prob vs neutral')
ax.set_xlabel('Condition')
ax.set_title('Contradiction vs neutral answers')
plt.tight_layout()
fig.savefig(plots_dir / 'contradiction_by_condition.png', dpi=200)
plt.close(fig)

# Save summary csv
summary.to_csv(RESULTS_DIR / 'summary_metrics.csv', index=False)
print('Saved plots and summary metrics.')
