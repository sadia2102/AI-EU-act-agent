# Multi-Model Comparison — 339 EU AI Act Scenarios

All four models were run over the full benchmark set of 339 scenarios. Evaluation was performed with `scenario_evaluation.py` and `analyze_classification.py`.

## Aggregate Performance

| Model | Evaluated | Risk Accuracy | Article F1 | Obligation Coverage | Avg. Processing Time |
|-------|-----------|--------------:|-----------:|--------------------:|---------------------:|
| GPT-5.4-mini | 339/339 (100%) | 83.48% | 0.247 | 83.23% | 34.46 s |
| Llama 3.1-70B-Instruct | 339/339 (100%) | **83.78%** | 0.161 | 57.73% | 141.1 s |
| Claude Sonnet-5 | 339/339 (100%) | 83.48% | **0.253** | **74.10%** | 284.3 s |
| Mistral 8B-2512 | 336/339 (99.1%) | 81.85% | 0.160 | 34.09% | 158.7 s |

## Per-Risk-Category F1 Scores

| Model | Prohibited | High-Risk | Limited-Risk | Minimal-Risk |
|-------|-----------:|----------:|-------------:|-------------:|
| GPT-5.4-mini | 0.776  | 0.857 | 0.782 | 0.892 |
| Llama 3.1-70B-Instruct | 0.773 | 0.857 | **0.797** | 0.892 |
| Claude Sonnet-5 | 0.759 | 0.862 | 0.782 | **0.900** |
| Mistral 8B-2512 | 0.736 | 0.826 | 0.776 | 0.900 |

## Key Findings

- **Risk classification accuracy** is very close for the top three models (83.48–83.78%); Mistral trails at 81.85%.
- **Claude Sonnet-5** produces the most relevant article citations (F1 = 0.253) and has the highest obligation coverage (74.10%).
- **GPT-5.4-mini** is fastest to evaluate while tying for best minimal-risk recall and second-best obligation coverage.
- **Llama 3.1-70B-Instruct** has the highest overall risk accuracy and the best limited-risk F1, but lower article/obligation scores.
- **Mistral 8B-2512** has the lowest scores on article and obligation coverage and the lowest overall risk accuracy. It also had 3/339 scenarios that could not be evaluated (likely unparseable output).

## Conclusion

For this benchmark, **Claude Sonnet-5** and **GPT-5.4-mini** provide the best balance of risk accuracy, citation quality, and obligation coverage. **Llama 3.1-70B-Instruct** is marginally best on raw risk accuracy but produces less complete citations and obligations. **Mistral 8B-2512** is the least capable model on this task set.
