#!/usr/bin/env python3
"""
Batch RAG processing for QA benchmark evaluation.

Processes multiple questions from a JSON benchmark file and generates
answers with citations using the existing rag_chat.py pipeline.

Usage:
  python batch_rag.py benchmark.json output.json [--fast]
  python batch_rag.py benchmark.json output.json --fast --model openai

Input JSON format:
{
  "questions": [
    {
      "id": "q1",
      "question": "What transparency obligations does the AI Act impose on GPAI providers?",
      "reference_answer": "Reference answer text for evaluation (optional)",
      "category": "transparency" (optional)
    },
    ...
  ]
}

Output JSON format:
{
  "metadata": {
    "processed_at": "2025-01-01T12:00:00Z",
    "total_questions": 5,
    "model": "gpt-5.4-mini",
    "fast_mode": false
  },
  "results": [
    {
      "id": "q1",
      "question": "...",
      "generated_answer": "...",
      "sources": [...],
      "context_used": "...",
      "processing_time": 2.5,
      "error": null
    },
    ...
  ]
}
"""

import json
import re
import signal
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from openai_deep_research import run_deep_research, DEFAULT_MODEL as DEEP_DEFAULT_MODEL
    DEEP_RESEARCH_AVAILABLE = True
except ImportError:
    run_deep_research = None  # type: ignore
    DEEP_DEFAULT_MODEL = None  # type: ignore
    DEEP_RESEARCH_AVAILABLE = False

# Hard per-scenario timeout (Unix-like systems)
SCENARIO_TIMEOUT_SECONDS = 900

def _scenario_timeout_handler(signum, frame):
    raise TimeoutError(f"Scenario processing exceeded {SCENARIO_TIMEOUT_SECONDS} seconds")

signal.signal(signal.SIGALRM, _scenario_timeout_handler)

# Import from existing rag_chat.py
from rag_chat import pre_retrieve, build_context, build_messages, chat_openai, chat_ollama, LLM_BACKEND, OPENAI_MODEL, OLLAMA_MODEL, OPENAI_API_KEY

# Reuse scenario heuristics/prompt helpers
from scenario_batch_rag import (
    build_scenario_query,
    extract_risk_level as scenario_extract_risk_level,
    extract_obligations as scenario_extract_obligations,
    extract_articles as scenario_extract_articles,
    calibrate_risk_level as scenario_calibrate_risk_level,
)


def load_benchmark(file_path: str) -> Dict[str, Any]:
    """Load benchmark JSON file and normalize entries into question format."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'questions' not in data and 'data' not in data:
        raise ValueError("Benchmark JSON must contain 'questions' or 'data' array")
    
    # Normalize to 'questions' list
    raw_items = data.get('questions') or data.get('data')
    normalized_questions = [_coerce_question_entry(item, idx + 1) for idx, item in enumerate(raw_items)]
    data['questions'] = normalized_questions
    data.pop('data', None)
    return data


def _coerce_question_entry(entry: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Ensure each entry has the fields batch_rag expects."""
    if 'question' in entry and entry['question']:
        return entry
    question_text = build_scenario_query(entry)
    obligations = entry.get('obligations') or []
    reference_answer = "\n".join(obligations) if isinstance(obligations, list) else str(obligations)
    return {
        "id": entry.get('id') or f"scenario_{idx}",
        "question": question_text,
        "reference_answer": reference_answer,
        "category": entry.get('risk_level') or entry.get('category'),
        "scenario": entry,
    }


def _build_structured_messages(question: str, context_block: str) -> List[Dict[str, str]]:
    system = (
        "You are an EU AI Act classification and compliance assistant. "
        "Use ONLY the provided 'Context' to answer the Question and for all citations. "
        "The following background describes EU AI Act risk categories to guide classification (do not cite this background unless it also appears in the provided Context):\n\n"
        "UNACCEPTABLE RISK — All AI systems considered a clear threat to the safety, livelihoods and rights of people are banned. "
        "The AI Act prohibits nine practices: (1) harmful AI-based manipulation and deception; "
        "(2) harmful AI-based exploitation of vulnerabilities; (3) social scoring; "
        "(4) individual criminal offence risk assessment or prediction; "
        "(5) untargeted scraping of the internet or CCTV material to create or expand facial recognition databases; "
        "(6) emotion recognition in workplaces and education institutions; "
        "(7) biometric categorisation to deduce certain protected characteristics; "
        "(8) real-time remote biometric identification for law enforcement purposes in publicly accessible spaces; "
        "(9) AI systems that generate non-consensual sexually explicit and intimate content or child sexual abuse (CSAM) material, such as AI 'nudification' apps. "
        "Prohibitions 1-8 became effective in February 2025. Prohibition 9 comes into effect in December 2026.\n\n"
        "HIGH RISK — AI use cases that can pose serious risks to health, safety or fundamental rights, including: "
        "AI safety components in critical infrastructures; AI solutions used in education institutions that may determine access to education and professional course; "
        "AI-based safety components of products; AI tools for employment, management of workers and access to self-employment; "
        "certain AI use cases giving access to essential private and public services; "
        "AI systems used for remote biometric identification, emotion recognition and biometric categorisation; "
        "AI use cases in law enforcement; AI use cases in migration, asylum and border control management; "
        "and AI solutions used in the administration of justice and democratic processes. "
        "Starting 2 December 2027, high-risk AI systems are subject to strict pre-market obligations: "
        "risk assessment and mitigation; high-quality datasets to minimise discriminatory outcomes; "
        "logging for traceability; detailed documentation; clear information to deployers; "
        "human oversight; and high levels of robustness, cybersecurity and accuracy.\n\n"
        "TRANSPARENCY RISK — Risks associated with a need for transparency around the use of AI. "
        "The AI Act introduces disclosure obligations to ensure humans are informed when necessary, e.g., when interacting with chatbots. "
        "Providers of generative AI must ensure AI-generated content is identifiable, and certain content such as deep fakes and text published to inform the public on matters of public interest must be clearly and visibly labelled. "
        "The transparency rules come into effect in August 2026.\n\n"
        "MINIMAL OR NO RISK — The AI Act does not introduce rules for AI deemed minimal or no risk, e.g., AI-enabled video games or spam filters.\n\n"
        "RISK-LEVEL MAPPING — For the `risk_level` JSON field, map the EU AI Act descriptions to these exact strings: "
        "Unacceptable/prohibited risk -> 'prohibited'; High risk -> 'high-risk'; Transparency risk -> 'limited-risk'; Minimal or no risk -> 'minimal-risk'. "
        "If a scenario does not clearly fit any category, use 'unknown'; do not default to 'minimal-risk'.\n\n"
        "CLASSIFICATION STEPS — Apply in this order:\n"
        "1. If the scenario matches any of the 9 prohibited practices, classify as 'prohibited'.\n"
        "2. Else if it matches a high-risk use case, classify as 'high-risk'.\n"
        "3. Else if it involves chatbots, deepfakes, AI-generated public-interest text, or other transparency/disclosure obligations, classify as 'limited-risk'.\n"
        "4. Only classify as 'minimal-risk' if the scenario is a simple, non-personal, non-interactive consumer utility (e.g. an AI-enabled video game or a spam filter) that does not collect personal data, influence decisions, monitor people, generate public-facing content, or match any higher-risk step; do not use minimal-risk as a broad low-stakes fallback.\n"
        "5. If uncertain, use 'unknown', never default to 'minimal-risk'.\n\n"
        "EXAMPLES — Use these as references:\n"
        "PROHIBITED: (1) A social scoring system that rates people's trustworthiness; (2) An emotion recognition system used in a classroom; (3) A nudification app that creates non-consensual intimate images.\n"
        "LIMITED-RISK: (1) A customer service chatbot that does not disclose it is an AI; (2) A news-publishing AI that does not clearly label AI-generated public-interest articles; (3) A deepfake generator used for entertainment without labelling.\n"
        "HIGH-RISK: (1) An AI system that screens job applicants for hiring decisions.\n"
        "MINIMAL-RISK: (1) An AI-enabled video game.\n\n"
        "Return a JSON object with this exact schema (no prose, no markdown fences):\n"
        "{\n"
        "  \"analysis\": string // concise 3-5 sentence summary with citations,\n"
        "  \"risk_level\": one of ['prohibited','high-risk','limited-risk','minimal-risk','unknown'],\n"
        "  \"obligations\": array of 5-10 short obligation bullets (each ending with a citation in the format [source; doc_id; chunk_index=K]),\n"
        "  \"articles\": array of all relevant Article numbers explicitly mentioned in the context as integers\n"
        "}"
    )
    user = (
        f"Question: {question}\n\n"
        f"Context:\n{context_block}\n\n"
        "Instructions:\n"
        "1. Base every statement on the context and cite like [source; doc_id; chunk_index=K].\n"
        "2. Choose the risk_level per EU AI Act categories.\n"
        "3. Obligations must be 5-10 actionable compliance steps, each drawn directly from the context and ending with a citation in the format [source; doc_id; chunk_index=K].\n"
        "4. Articles must be every relevant EU AI Act Article number explicitly mentioned in the context; include all that apply and do not add article numbers not cited.\n"
        "5. Output ONLY valid JSON."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_structured_response(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r'^```json\s*', '', cleaned, flags=re.IGNORECASE).rstrip('`')
    cleaned = cleaned.strip().rstrip('```').strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    obligations = data.get('obligations') or []
    if isinstance(obligations, str):
        obligations = [ob.strip() for ob in obligations.split('\n') if ob.strip()]
    elif isinstance(obligations, list):
        obligations = [str(ob).strip() for ob in obligations if str(ob).strip()]
    else:
        obligations = []

    articles_raw = data.get('articles') or []
    articles: List[int] = []
    if isinstance(articles_raw, list):
        for art in articles_raw:
            if isinstance(art, int):
                articles.append(art)
            else:
                match = re.search(r"(\d+)", str(art))
                if match:
                    articles.append(int(match.group(1)))

    data['obligations'] = obligations
    data['articles'] = sorted(set(articles))
    risk = str(data.get('risk_level', 'unknown')).strip().lower()
    allowed = {'prohibited', 'high-risk', 'limited-risk', 'minimal-risk', 'unknown'}
    data['risk_level'] = risk if risk in allowed else 'unknown'
    data['analysis'] = str(data.get('analysis', '')).strip()
    return data


def process_single_question(question_data: Dict[str, Any], fast: bool = False, deep_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Process a single question through the RAG pipeline."""
    question_id = question_data.get('id', f'q_{int(time.time())}')
    question_text = question_data['question']
    reference_answer = question_data.get('answer', '')  # Store reference answer for evaluation
    scenario_meta = question_data.get('scenario', {})
    deep_research_result = None

    start_time = time.time()
    signal.alarm(SCENARIO_TIMEOUT_SECONDS)
    
    try:
        # Step 1: Retrieve context
        prompt_question = question_text
        if scenario_meta:
            prompt_question = build_scenario_query(scenario_meta)
        print(f"Processing question {question_id}: {prompt_question[:100]}...")
        docs, metas, scores = pre_retrieve(prompt_question, fast=fast)
        
        if not docs:
            if deep_config:
                deep_research_result = _execute_deep_research(question_id, question_text, deep_config)
            return {
                "id": question_id,
                "question": question_text,
                "reference_answer": reference_answer,
                "scenario": scenario_meta,
                "generated_answer": "",
                "analysis": "",
                "generated_risk_level": "unknown",
                "generated_obligations": [],
                "relevant_articles": [],
                "sources": [],
                "context_used": "",
                "processing_time": time.time() - start_time,
                "error": "No context found",
                "deep_research": deep_research_result
            }
        
        # Step 2: Build context
        context_block = build_context(docs, metas)
        
        # Step 3: Generate structured answer
        messages = _build_structured_messages(question_text, context_block)
        
        if LLM_BACKEND == 'openai':
            # Use non-streaming version for batch processing
            from openai import OpenAI
            import os
            api_key = os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not found in environment or rag_chat.py config.")
            client = OpenAI(api_key=api_key, timeout=30.0, max_retries=0)
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.2,
                stream=False,
            )
            answer = response.choices[0].message.content or ""
        else:
            # Ollama non-streaming
            import requests
            payload = {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
            }
            response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=600)
            response.raise_for_status()
            answer = response.json().get("message", {}).get("content", "")
        
        structured = _parse_structured_response(answer)
        generated_risk_level = structured['risk_level'] if structured else 'unknown'
        generated_obligations = structured['obligations'] if structured else []
        relevant_articles = structured['articles'] if structured else []
        analysis_summary = (structured.get('analysis') if structured else None) or answer.strip()

        if (not generated_risk_level or generated_risk_level == 'unknown') and analysis_summary:
            extracted = scenario_extract_risk_level(analysis_summary)
            if extracted and extracted != 'unknown':
                generated_risk_level = extracted

        if not generated_obligations and analysis_summary:
            generated_obligations = scenario_extract_obligations(analysis_summary)

        if not relevant_articles and analysis_summary:
            relevant_articles = scenario_extract_articles(analysis_summary)

        if scenario_meta:
            generated_risk_level = scenario_calibrate_risk_level(generated_risk_level or 'unknown', scenario_meta)

        # Step 4: Extract sources for output
        sources = []
        for meta in metas:
            src = meta.get("source", "unknown")
            doc_id = meta.get("doc_id", "")
            chunk_idx = meta.get("chunk_index", "n/a")
            sources.append(f"[{src}; {doc_id}; chunk_index={chunk_idx}]")

        if deep_config:
            deep_research_result = _execute_deep_research(question_id, question_text, deep_config)

        return {
            "id": question_id,
            "question": question_text,
            "reference_answer": reference_answer,
            "scenario": scenario_meta,
            "generated_answer": answer.strip(),
            "analysis": analysis_summary,
            "generated_risk_level": generated_risk_level,
            "generated_obligations": generated_obligations,
            "relevant_articles": relevant_articles,
            "sources": sources,
            "context_used": context_block,
            "processing_time": time.time() - start_time,
            "error": None,
            "deep_research": deep_research_result
        }
        
    except Exception as e:
        deep_research_result = None
        if deep_config:
            deep_research_result = _execute_deep_research(question_id, question_text, deep_config)

        return {
            "id": question_id,
            "question": question_text,
            "reference_answer": reference_answer,
            "scenario": scenario_meta,
            "generated_answer": "",
            "analysis": "",
            "generated_risk_level": "unknown",
            "generated_obligations": [],
            "relevant_articles": [],
            "sources": [],
            "context_used": "",
            "processing_time": time.time() - start_time,
            "error": str(e),
            "deep_research": deep_research_result
        }
    finally:
        signal.alarm(0)


def process_batch(benchmark_data: Dict[str, Any], fast: bool = False, deep_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Process all questions in the benchmark."""
    questions = benchmark_data['questions']
    total_questions = len(questions)
    
    print(f"Processing {total_questions} questions...")
    
    results = []
    successful = 0
    failed = 0
    
    for i, question_data in enumerate(questions, 1):
        print(f"\nProgress: {i}/{total_questions}")
        result = process_single_question(question_data, fast=fast, deep_config=deep_config)
        results.append(result)
        
        if result['error']:
            failed += 1
            print(f"❌ Failed: {result['error']}")
        else:
            successful += 1
            print(f"✅ Success (answer length: {len(result['generated_answer'])} chars)")
    
    metadata = {
        "processed_at": datetime.utcnow().isoformat() + "Z",
        "total_questions": total_questions,
        "successful": successful,
        "failed": failed,
        "model": OPENAI_MODEL if LLM_BACKEND == 'openai' else OLLAMA_MODEL,
        "backend": LLM_BACKEND,
        "fast_mode": fast
    }

    if deep_config:
        metadata["deep_research"] = {
            "enabled": True,
            "rounds": deep_config.get("rounds"),
            "model": deep_config.get("model") or DEEP_DEFAULT_MODEL,
            "max_sources": deep_config.get("max_sources"),
            "followups": deep_config.get("followups"),
            "per_subq_docs": deep_config.get("per_subq_docs"),
            "max_context_chars": deep_config.get("max_context_chars"),
            "verbose": deep_config.get("verbose", False),
            "out_dir": deep_config.get("out_dir"),
        }
    
    return {
        "metadata": metadata,
        "results": results
    }


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


def create_example_benchmark():
    """Create an example benchmark JSON file."""
    example = {
        "questions": [
            {
                "id": "q1",
                "question": "What transparency obligations does the AI Act impose on GPAI providers?",
                "reference_answer": "The AI Act requires GPAI providers to provide detailed documentation about their models, including technical documentation, instructions for use, and information about data used in training.",
                "category": "transparency"
            },
            {
                "id": "q2", 
                "question": "What are the conformity assessment procedures for high-risk AI systems?",
                "reference_answer": "High-risk AI systems must undergo conformity assessment procedures including internal control, quality management systems, and third-party assessment for certain systems.",
                "category": "conformity"
            },
            {
                "id": "q3",
                "question": "What is lawful access under EU TDM?",
                "reference_answer": "Lawful access under EU Text and Data Mining (TDM) refers to the ability to access and use publicly available content for research and innovation purposes without additional authorization.",
                "category": "tdm"
            }
        ]
    }
    
    with open("example_benchmark.json", 'w', encoding='utf-8') as f:
        json.dump(example, f, indent=2, ensure_ascii=False)
    
    print("Created example_benchmark.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch RAG processing for QA benchmark evaluation")
    parser.add_argument("benchmark", help="Path to benchmark JSON file")
    parser.add_argument("output", help="Path to output JSON file")
    parser.add_argument("--fast", action="store_true", help="Use fast mode (fewer variants, no HyDE)")
    parser.add_argument("--create-example", action="store_true", help="Create example benchmark file")
    parser.add_argument("--deep-research", action="store_true", help="Also run openai_deep_research per question for comparison")
    parser.add_argument("--deep-rounds", type=int, default=None, help="Override number of deep research rounds (default from openai_deep_research)")
    parser.add_argument("--deep-model", type=str, default=None, help="Override deep research model (default from openai_deep_research)")
    parser.add_argument("--deep-max-sources", type=int, default=None, help="Maximum deep research sources to retain (default 20)")
    parser.add_argument("--deep-followups", type=int, default=None, help="Follow-up questions per round (default 3)")
    parser.add_argument("--deep-per-subq-docs", type=int, default=None, help="Documents per follow-up to include (default 3)")
    parser.add_argument("--deep-max-context-chars", type=int, default=None, help="Maximum stitched context length (default 2800 chars)")
    parser.add_argument("--deep-verbose", action="store_true", help="Use verbose deep research prompts/output")
    parser.add_argument("--deep-out-dir", type=str, default=None, help="Directory to store per-question deep research markdown reports")
    return parser.parse_args()


def _build_deep_config(args) -> Optional[Dict[str, Any]]:
    if not args.deep_research:
        return None
    if not DEEP_RESEARCH_AVAILABLE or run_deep_research is None:
        raise RuntimeError("openai_deep_research.py is not available. Install dependencies before using --deep-research.")

    config: Dict[str, Any] = {
        "fast": args.fast,
        "verbose": args.deep_verbose,
        "out_dir": args.deep_out_dir,
    }

    if args.deep_rounds is not None:
        config["rounds"] = args.deep_rounds
    if args.deep_model:
        config["model"] = args.deep_model
    if args.deep_max_sources is not None:
        config["max_sources"] = args.deep_max_sources
    if args.deep_followups is not None:
        config["followups"] = args.deep_followups
    if args.deep_per_subq_docs is not None:
        config["per_subq_docs"] = args.deep_per_subq_docs
    if args.deep_max_context_chars is not None:
        config["max_context_chars"] = args.deep_max_context_chars

    return config


def _execute_deep_research(question_id: str, question_text: str, deep_config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        out_dir = deep_config.get("out_dir")
        out_path = None
        if out_dir:
            out_dir_path = Path(out_dir)
            out_dir_path.mkdir(parents=True, exist_ok=True)
            out_path = str(out_dir_path / f"{question_id}_deep.md")

        kwargs: Dict[str, Any] = {}
        for key in ("rounds", "max_sources", "followups", "per_subq_docs", "max_context_chars"):
            if deep_config.get(key) is not None:
                kwargs[key] = deep_config[key]

        kwargs["fast"] = deep_config.get("fast", False)
        kwargs["model"] = deep_config.get("model") or DEEP_DEFAULT_MODEL
        kwargs["verbose"] = deep_config.get("verbose", False)
        if out_path:
            kwargs["out_path"] = out_path

        result = run_deep_research(question_text, **kwargs)
        result["question_id"] = question_id
        if out_path:
            result["report_path"] = out_path
        return result
    except Exception as exc:
        return {"error": str(exc)}


def main():
    args = parse_args()
    
    if args.create_example:
        create_example_benchmark()
        return

    try:
        deep_config = _build_deep_config(args)
    except RuntimeError as cfg_err:
        print(f"Error: {cfg_err}")
        return 1
    
    # Validate input file
    if not Path(args.benchmark).exists():
        print(f"Error: Benchmark file not found: {args.benchmark}")
        return
    
    try:
        # Load benchmark
        print(f"Loading benchmark from: {args.benchmark}")
        benchmark_data = load_benchmark(args.benchmark)
        
        # Process batch
        results = process_batch(benchmark_data, fast=args.fast, deep_config=deep_config)
        
        # Save results
        save_results(results, args.output)
        
        # Print summary
        metadata = results["metadata"]
        print(f"\n{'='*60}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total questions: {metadata['total_questions']}")
        print(f"Successful: {metadata['successful']}")
        print(f"Failed: {metadata['failed']}")
        print(f"Model: {metadata['model']} ({metadata['backend']})")
        print(f"Fast mode: {metadata['fast_mode']}")
        if metadata.get('deep_research', {}).get('enabled'):
            deep_meta = metadata['deep_research']
            print(f"Deep research: enabled (model: {deep_meta.get('model')}, rounds: {deep_meta.get('rounds')})")
        print(f"Processing time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if metadata['failed'] > 0:
            print(f"\n⚠️  {metadata['failed']} questions failed. Check output file for details.")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
