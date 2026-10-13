# Experimental Results and Reproducibility

## 1. Experimental Dataset

The main evaluation uses 339 synthetic scenarios from the
AI Act Evaluation Benchmark.

Reference categories:

| Category | Count |
|---|---:|
| Prohibited | 70 |
| High-risk | 86 |
| Limited-risk | 84 |
| Minimal-risk | 99 |
| Total | 339 |

The dataset is used to evaluate risk-classification agreement,
article identification and obligation generation.

## 2. Main Model Comparison

The final master's thesis reports:

| Model | Correct | Evaluated | Accuracy | Article F1 | Obligation Count Coverage |
|---|---:|---:|---:|---:|---:|
| GPT-5.4-mini v7 | 283 | 339 | 83.48% | 0.247 | 83.23% |
| Llama 3.1-70B | 284 | 339 | 83.78% | 0.161 | 57.73% |
| Claude Sonnet-5 | 283 | 339 | 83.48% | 0.253 | 74.10% |
| Ministral 8B | 275 | 336 | 81.85% | 0.160 | 34.09% |

These are the final thesis reporting values.

The GPT-v7 values have been checked against the preserved
local output and evaluation files.

## 3. GPT Prompt-Version Development

The thesis records multiple development iterations:

| Version | Evaluated | Risk Accuracy | Obligation Count Coverage | Article F1 |
|---|---:|---:|---:|---:|
| v2 | 339 | 83.48% | 56.1% | 0.328 |
| v3 | 339 | 83.48% | 56.6% | 0.298 |
| v5 | 253 | 79.05% | 50.5% | 0.289 |
| v6 | 339 | 83.48% | 57.8% | 0.305 |
| v7 | 339 | 83.48% | 83.2% | 0.247 |

The v5 run was incomplete and should not be directly
compared with complete 339-scenario runs.

### Version 6

Version 6 tightened the minimal-risk classification
instructions.

Preserved evidence:

    scenarios_output_v6.json
    scenario_evaluation_v6.json

Full-run results:

    Evaluated: 339
    Correct: 283
    Incorrect: 56
    Accuracy: 83.48%
    Obligation count coverage: 57.84%
    Article F1: 0.3052
    Average processing time: 32.09 seconds

### Version 7

Version 7 extended the generation instructions to request
more obligations and relevant article numbers from
retrieved context. The implementation also includes a
per-scenario timeout.

Preserved evidence:

    scenarios_output_v7.json
    scenario_evaluation_v7.json

Full-run results:

    Evaluated: 339
    Correct: 283
    Incorrect: 56
    Accuracy: 83.48%
    Obligation count coverage: 83.23%
    Article F1: 0.2474
    Average processing time: 34.46 seconds

### Direct v6 versus v7 comparison

All 339 scenario IDs are present in both runs.

Only three final risk predictions changed:

| Scenario | Benchmark | v6 | v7 |
|---|---|---|---|
| scenario_161 | limited-risk | prohibited | minimal-risk |
| scenario_173 | limited-risk | prohibited | high-risk |
| scenario_201 | limited-risk | prohibited | minimal-risk |

All three cases remained incorrectly classified.

The improved obligation-count coverage did not produce
an overall improvement in risk-classification accuracy.

The results illustrate a trade-off between output
completeness and article-identification specificity.

## 4. Historical v6 Source Availability

The original version 6 Python implementation was
overwritten during subsequent development.

The complete v6 generated outputs and evaluation results
remain available.

These records support inspection and re-evaluation of
the original predictions.

They do not establish that the original v6 generation
process can be reproduced exactly from the final v7 code.

No reconstructed v6 prompt is presented as an original
historical source snapshot.

## 5. GPT-v7 Risk Confusion Summary

The verified final classification results are:

| Reference Category | Correct | Total |
|---|---:|---:|
| Prohibited | 52 | 70 |
| High-risk | 75 | 86 |
| Limited-risk | 61 | 84 |
| Minimal-risk | 95 | 99 |
| Total | 283 | 339 |

The largest sources of disagreement include confusion
between limited-risk and minimal-risk, as well as
misclassification of prohibited and high-risk cases.

The accompanying thesis contains detailed domain-level
and case-level failure analysis.

## 6. OpenAI Component Ablation

The final thesis reports:

| Condition | Accuracy |
|---|---:|
| Scenario-only OpenAI | 66.37% |
| Scenario-only + calibration | 83.78% |
| RAG without calibration | 54.87% |
| RAG + calibration | 83.48% |

The scenario-only raw experiment is preserved as:

    scenarios_output_openai_only_raw.json
    scenario_evaluation_openai_only_raw.json
    scenario_evaluation_openai_only_raw_report.md

The results indicate that deterministic calibration
has a substantial influence on the final risk label.

The tested RAG condition does not establish an
aggregate risk-classification improvement over the
scenario-only condition.

Retrieval nevertheless supplies external evidence for
explanations, citations, obligations and article extraction.

## 7. Development Samples

The archived files named v6_30_sample contain 29
successfully evaluated scenarios.

They are development records rather than the primary
full-dataset evaluation.

The sample uses its own scenario ordering and must not
be matched to the 339-scenario dataset by ID alone.

Its results must not replace the complete v6 evaluation.

## 8. Re-evaluation

From the backend directory, run:

    python scenario_evaluation.py \
      scenarios.json \
      ../experiments/V7/scenarios_output_v7.json \
      ../experiments/V7/recomputed_evaluation_v7.json

For v6, use the corresponding v6 files.

This procedure recalculates evaluation metrics from
stored predictions. It does not rerun the original LLM.

## 9. Interpretation and Limitations

The benchmark is synthetic and was used during development.

The reference labels should not be treated as independently
adjudicated legal truth.

Accuracy describes agreement with the benchmark.

Obligation-count coverage is not legal completeness.

Article F1 is based on overlap with reference article sets.

Citation-format validity does not establish source support.

The recorded experiments reflect a particular software,
model and retrieval configuration.

## 10. Original Experiment Preservation

The original JSON output and evaluation files should
remain unchanged.

Regenerated reports should use separate filenames.

Documentation should clearly distinguish:

- Historical recorded outputs
- Recomputed evaluation metrics
- New model-generation runs
- The final source-code snapshot
- Unavailable historical source-code versions

This separation preserves the provenance of the thesis
results and prevents later modifications from being
mistaken for the original experiments.
