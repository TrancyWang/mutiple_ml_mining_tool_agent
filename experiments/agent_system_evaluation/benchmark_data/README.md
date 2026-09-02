# Benchmark datasets

`prepare_benchmark_datasets.py` downloads and normalizes the four datasets used
by the Gemini-only versus hybrid comparison. Run it from the project root:

```bash
conda run --no-capture-output -n trancy_tool \
  python experiments/agent_system_evaluation/prepare_benchmark_datasets.py
```

The benchmark is deliberately split into two groups:

| Dataset | Task | Why it is useful |
|---|---|---|
| UCI Bank Marketing | Tabular binary classification | 45,211 real customer records and a clear classification target |
| UCI Wine Quality | Tabular regression, optionally ordinal classification | 6,497 records (red + white) with a numeric quality target |
| ChnSentiCorp | Chinese binary sentiment | Fixed train/validation/test split: 9,600/1,200/1,200 |
| CLUE TNEWS | Chinese short-text topic classification | 15 news categories and official benchmark splits |

The script writes all downloaded data under this directory and records URLs,
dataset descriptions, counts, and license notes in `metadata.json`. The
ChnSentiCorp and CLUE source cards do not clearly declare a redistribution
license, so the paper should cite the sources and avoid republishing raw data
unless the terms are verified.

For CLUE TNEWS, the official test labels are hidden (`-1`) in the downloaded
test file. Use the official training/validation split for the reproducible
comparison, or create a new stratified holdout from training data; do not
report the hidden test file as a locally computed accuracy result.
