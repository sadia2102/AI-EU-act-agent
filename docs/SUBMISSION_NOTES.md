# Master's Thesis Software Submission

## Purpose

This package contains the EU AI Act Legal Information Assistant
developed for the accompanying master's thesis.

It includes the application source code, benchmark data,
evaluation scripts, experimental results and documentation.

## Final Implementation

The submitted source code represents the final documented
implementation associated with the GPT-v7 research configuration.

The preserved v6 results are included as an intermediate
prompt-engineering experiment.

## Experimental Evidence

The package contains:

- Full GPT-v6 predictions and evaluation: 339 scenarios
- Full GPT-v7 predictions and evaluation: 339 scenarios
- Scenario-only OpenAI baseline
- Four-model benchmark outputs and evaluation reports
- The scenario benchmark
- Evaluation scripts

The v6 and v7 results have been checked against the reported
thesis metrics.

## Historical Version Limitation

The original v6 source-code snapshot was overwritten during
development.

Its output and evaluation JSON files are preserved.

These files permit retrospective analysis and re-evaluation
of v6 predictions, but not guaranteed exact regeneration
using the original v6 prompt.

The final source implementation is documented separately.

## Reproducibility

The package supports installation of the reviewed application
and re-evaluation of recorded benchmark predictions.

Full historical experiment regeneration additionally depends
on external model access, the retrieval environment and
historical configurations.

No claim is made that hosted LLM calls will reproduce
byte-identical outputs.

## Important Known Issues

- The older GitHub comparison report contains outdated GPT values.
- The model-comparison summary generator uses an incorrect JSON key.
- The classification-report script double-counts incorrect cases
  in its overall incorrect-total summary.
- The frontend may fall back to simulated responses.
- Some optional preprocessing dependencies are not listed in
  the current requirements file.

These issues are documented rather than concealed.

## Security

No real API keys or passwords should be included.

The repository and software archive should be checked for
credentials before distribution.

## Intended Use

This is academic research software.

The generated classifications, obligations and legal references
must not be treated as authoritative legal compliance decisions.

The accompanying thesis provides the full research context,
evaluation results, discussion and limitations.