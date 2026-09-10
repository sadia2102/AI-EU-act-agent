# Thesis Evaluation (Multi-Model Comparison)

This directory contains the final evaluation reports and LaTeX/Word-ready tables for the multi-model comparison on 339 EU AI Act scenarios.

## Contents

- `multi_model_comparison_report.md` — Markdown report with aggregate and per-risk-category metrics.
- `multi_model_table.tex` — LaTeX table snippet for the thesis.
- `multi_model_table_for_word.html` — HTML table for Word.

## Reproducing the results

The evaluation scripts and data are in `../backend/`:

```bash
cd ../backend
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
# start ChromaDB and ingest text_chunks.tsv if needed
python ingest_tsv.py
# run the batch evaluation for a model
python scenario_batch_rag.py scenarios.json scenarios_output_MODEL.json
# evaluate the output
python scenario_evaluation.py scenarios.json scenarios_output_MODEL.json scenario_evaluation_MODEL.json
python analyze_classification.py scenario_evaluation_MODEL.json > classification_report_MODEL.txt
```

The four model outputs used in the comparison are:
- `scenarios_output_openai_gpt-5.4-mini.json`
- `scenarios_output_meta-llama_llama-3.1-70b-instruct.json`
- `scenarios_output_anthropic_claude-sonnet-5.json`
- `scenarios_output_mistralai_ministral-8b-2512.json`

The evaluation JSONs and classification reports are also in `../backend/`.
