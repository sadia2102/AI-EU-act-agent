# EU AI Act Legal Information Assistant

## Master's Thesis Software

**Thesis:** A Legal Information Assistant for AI Scenarios:
Exploring the EU AI Act with Retrieval-Augmented Generation

**University:** Friedrich-Alexander-Universität Erlangen-Nürnberg

**Degree:** M.Sc. Artificial Intelligence

**Supervisor:** Prof. Dr. Harald Köstler

**Repository:**
https://github.com/sadia2102/AI-EU-act-agent

## 1. Project Overview

This repository contains the software developed for a master's thesis
investigating Retrieval-Augmented Generation (RAG) for legal information
assistance under the European Union Artificial Intelligence Act
(Regulation (EU) 2024/1689).

The application analyzes descriptions of AI systems and generates:

- An AI Act risk-level classification
- An explanation of the classification
- Potentially applicable compliance obligations
- Relevant AI Act articles
- References to retrieved legal information

The four benchmark classification categories are:

1. Prohibited
2. High-risk
3. Limited-risk
4. Minimal-risk

The project is a research prototype, not a legal compliance
certification system. Its outputs require independent verification.

## 2. Research Objectives

The project investigates:

- Automated classification of AI deployment scenarios
- Retrieval of relevant legal passages
- Generation of source-attributed legal explanations
- Identification of applicable articles and obligations
- The effect of deterministic risk calibration
- Differences between language models
- The effect of iterative prompt engineering

The principal evaluation uses 339 synthetic AI deployment scenarios.

## 3. System Architecture

The benchmark-processing pipeline is:

    Benchmark scenario
           |
           v
    Query construction
           |
           v
    Query rewriting and expansion
           |
           v
    ChromaDB retrieval
           |
           v
    Candidate reranking
           |
           v
    Context construction
           |
           v
    Language-model generation
           |
           v
    Structured-output parsing
           |
           v
    Deterministic risk calibration
           |
           v
    Classification, obligations,
    articles and source references
           |
           v
    Benchmark evaluation

The live application uses a Next.js frontend and FastAPI backend.
It shares several components with the batch pipeline, but the
live and benchmark processing paths are not identical.

## 4. Technologies

| Component | Technology |
|---|---|
| Programming language | Python |
| Backend API | FastAPI |
| Frontend | Next.js, React, TypeScript |
| Vector database | ChromaDB |
| Embeddings | all-MiniLM-L6-v2 |
| Reranking | ms-marco-MiniLM-L-12-v2 |
| Generation | Configurable LLM backend |
| Evaluation | NumPy, scikit-learn, sentence-transformers |

The repository includes an OpenAI-compatible backend and an
Ollama option.

The historical experiments used specific model configurations;
changing the backend or model does not reproduce those experiments.

## 5. Repository Structure

    backend/
        rag_chat.py
        batch_rag.py
        scenario_batch_rag.py
        scenario_evaluation.py
        analyze_classification.py
        server.py
        ingest_tsv.py
        scenarios.json
        text_chunks.tsv
        requirements.txt

    src/
        app/
        components/
        lib/

    docs/
        SOFTWARE_DOCUMENTATION.md
        EXPERIMENTS_AND_EVALUATION.md
        SUBMISSION_NOTES.md

    experiments/
        V6/
        V7/
        baselines/
        development_samples/

    thesis/
        multi_model_comparison_report.md

The docs/ and experiments/ structure is used for the final
software submission. Some files may need to be copied from the
preserved local experimental outputs.

## 6. Installation

### Prerequisites

- Python 3 with virtual-environment support
- Node.js and npm
- ChromaDB
- Access to a supported language model
- The processed text corpus, text_chunks.tsv

### Python environment

From the repository root:

    cd backend
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

### Model configuration

The default GitHub configuration uses the OpenAI-compatible backend.

Set the relevant environment variables securely:

    LLM_BACKEND=openai
    OPENAI_MODEL=gpt-5.4-mini
    OPENAI_API_KEY=<your-secret-key>

Do not commit actual API credentials to the repository or include
them in the software submission.

The implementation also supports an alternative API base URL
through OPENAI_BASE_URL.

### Start ChromaDB

Start ChromaDB in a separate terminal:

    chroma run --path ./chroma_db --host localhost --port 8000

The application expects:

    Collection: research_chunks
    Host: localhost
    Port: 8000

### Populate the retrieval collection

From the backend directory:

    python ingest_tsv.py

WARNING: The current ingestion script deletes and recreates the
research_chunks collection. It must not be run against a database
containing data that needs to be preserved.

The script imports text_chunks.tsv and generates vector embeddings.

### Start the FastAPI server

From the backend directory:

    uvicorn server:app --host 127.0.0.1 --port 8008

Check the API:

    curl http://localhost:8008/health

A successful health response confirms that the API is reachable,
but does not independently verify the entire retrieval pipeline.

### Start the frontend

From the repository root:

    npm ci

Create a .env.local file containing:

    NEXT_PUBLIC_API_URL=http://localhost:8008

Then start Next.js:

    npm run dev

Open:

    http://localhost:3000

WARNING: The current frontend can fall back to simulated responses
if the backend request fails. Demo responses must not be confused
with actual RAG outputs.

## 7. Batch Processing

The benchmark-processing implementation is primarily contained
in batch_rag.py, with retrieval implemented in rag_chat.py and
risk-calibration functions imported from scenario_batch_rag.py.

An example command is:

    cd backend
    python batch_rag.py scenarios.json new_output.json

The command requires a working retrieval database and model
connection.

This generates new predictions. Results may differ from the
historical experiments because model behavior, external services,
software dependencies and retrieval state may change.

## 8. Evaluating Saved Results

Saved experimental predictions can be evaluated without repeating
the language-model generation step.

For example:

    cd backend

    python scenario_evaluation.py \
      scenarios.json \
      ../experiments/v7/scenarios_output_v7.json \
      ../experiments/v7/recomputed_evaluation_v7.json

A classification report can then be generated:

    python analyze_classification.py \
      ../experiments/v7/recomputed_evaluation_v7.json

The classification-report script has a known reporting issue
documented in SOFTWARE_DOCUMENTATION.md.

## 9. Main Experimental Results

The final thesis reports the following end-to-end performance:

| Model | Correct / Evaluated | Accuracy |
|---|---:|---:|
| GPT-5.4-mini v7 | 283/339 | 83.48% |
| Llama 3.1-70B-Instruct | 284/339 | 83.78% |
| Claude Sonnet-5 | 283/339 | 83.48% |
| Ministral 8B | 275/336 | 81.85% |

The GPT-v7 result is supported by the preserved full-run output
and evaluation files.

The thesis also investigates a scenario-only OpenAI baseline,
deterministic calibration, retrieval ablation and prompt evolution.

## 10. Prompt-Version Development

Version 6 and version 7 both achieved 83.48% accuracy on the
339-scenario benchmark.

Version 7 increased obligation-count coverage from approximately
57.8% to 83.2%, while article F1 decreased from approximately
0.305 to 0.247.

The original version 6 source-code snapshot was overwritten during
development. Its complete generated outputs and evaluation results
have been preserved, but exact regeneration of that historical
configuration is not guaranteed.

Version 7 is the final documented implementation.

See EXPERIMENTS_AND_EVALUATION.md for details.

## 11. Evaluation Limitations

The benchmark contains synthetic scenarios and was used during
development. Performance numbers measure agreement with benchmark
annotations rather than independently adjudicated legal correctness.

Deterministic post-processing substantially influences final
classification accuracy.

Obligation-count coverage measures generated quantity relative to
reference obligations, not verified legal completeness.

Citation-format validity does not establish substantive support
for every claim.

The final benchmark metrics should not be represented as measured
accuracy for arbitrary live chatbot interactions.

## 12. Reproducibility

The preserved experimental outputs support inspection and
re-evaluation of the reported results.

Exact regeneration additionally depends on historical prompts,
model versions, dependencies, retrieval content and configuration.

The original version 6 implementation is not separately available.

No guarantee is made that rerunning hosted LLM generation will
produce byte-identical outputs.

## 13. Software Status

The repository contains research software developed for the thesis.

The final uploaded Python scripts were compared with the
corresponding GitHub source files and found to match.

The complete application has not yet been verified through a
fresh end-to-end installation and execution test.

Known reporting and user-interface limitations are documented
separately.

## 14. Security and Data Handling

Never commit API keys, passwords or private credentials.

Exclude local virtual environments, node_modules, caches,
generated build files and local secret configuration files.

Review the license and redistribution conditions for third-party
benchmark materials and legal documents before public distribution.

## 15. Additional Documentation

- docs/SOFTWARE_DOCUMENTATION.md
- docs/EXPERIMENTS_AND_EVALUATION.md
- docs/SUBMISSION_NOTES.md

The accompanying master's thesis provides the research methodology,
experimental analysis, discussion and limitations in greater detail.
