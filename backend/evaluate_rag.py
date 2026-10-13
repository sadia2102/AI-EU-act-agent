#!/usr/bin/env python3
"""
RAG Evaluation Script

Evaluates generated answers against benchmark QA pairs using multiple metrics:
- Semantic similarity (using sentence transformers)
- Factual consistency (using NLI/entailment)
- Citation quality and coverage
- Answer completeness
- Response time analysis

Usage:
  python evaluate_rag.py benchmark.json generated_results.json evaluation_report.json
  python evaluate_rag.py benchmark.json generated_results.json evaluation_report.json --detailed

Requirements:
  pip install sentence-transformers numpy scikit-learn
"""

import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import re

# For semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Semantic similarity will be skipped.")

# For factual consistency (NLI)
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    NLI_AVAILABLE = True
except ImportError:
    NLI_AVAILABLE = False
    print("Warning: transformers not available. Factual consistency will be skipped.")


class RAGEvaluator:
    def __init__(self, detailed: bool = False):
        self.detailed = detailed
        self.similarity_model = None
        self.nli_model = None
        self.nli_tokenizer = None
        
        # Initialize models if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            print("Loading semantic similarity model...")
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        if NLI_AVAILABLE:
            print("Loading NLI model for factual consistency...")
            self.nli_model = AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')
            self.nli_tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli')
    
    def extract_citations(self, text: str) -> List[str]:
        """Extract citation patterns from text."""
        # Pattern: [source; doc_id; chunk_index=K]
        citation_pattern = r'\[([^;]+);\s*([^;]+);\s*chunk_index=([^\]]+)\]'
        return re.findall(citation_pattern, text)
    
    def count_sentences(self, text: str) -> int:
        """Count sentences in text."""
        sentences = re.split(r'[.!?]+', text)
        return len([s.strip() for s in sentences if s.strip()])
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        if not self.similarity_model:
            return 0.0
        
        try:
            embeddings = self.similarity_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def factual_consistency(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Calculate factual consistency using NLI."""
        if not self.nli_model or not self.nli_tokenizer:
            return {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}
        
        try:
            inputs = self.nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.nli_model(**inputs)
            
            probabilities = torch.softmax(outputs.logits, dim=1)
            # Labels: 0=contradiction, 1=neutral, 2=entailment
            probs = probabilities[0].cpu().numpy()
            
            return {
                "entailment": float(probs[2]),
                "contradiction": float(probs[0]), 
                "neutral": float(probs[1])
            }
        except Exception as e:
            print(f"Error calculating factual consistency: {e}")
            return {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}
    
    def citation_quality_score(self, answer: str, context_used: str) -> Dict[str, Any]:
        """Evaluate citation quality and coverage."""
        citations = self.extract_citations(answer)
        context_citations = self.extract_citations(context_used)
        
        # Count sentences with and without citations
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        sentences_with_citations = 0
        for sentence in sentences:
            if self.extract_citations(sentence):
                sentences_with_citations += 1
        
        total_sentences = len(sentences)
        citation_coverage = sentences_with_citations / total_sentences if total_sentences > 0 else 0
        
        # Check if citations are from the provided context
        valid_citations = 0
        for citation in citations:
            if citation in context_citations:
                valid_citations += 1
        
        citation_validity = valid_citations / len(citations) if citations else 0
        
        return {
            "total_citations": len(citations),
            "unique_citations": len(set(citations)),
            "sentences_with_citations": sentences_with_citations,
            "total_sentences": total_sentences,
            "citation_coverage": citation_coverage,
            "citation_validity": citation_validity,
            "citations": citations
        }
    
    def evaluate_single_answer(self, question_data: Dict[str, Any], result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single generated answer against the benchmark."""
        question_id = question_data.get('id', 'unknown')
        question = question_data['question']
        reference_answer = question_data.get('answer', '') or question_data.get('reference_answer', '')
        generated_answer = result_data.get('generated_answer', '')
        context_used = result_data.get('context_used', '')
        processing_time = result_data.get('processing_time', 0)
        error = result_data.get('error')
        
        evaluation = {
            "id": question_id,
            "question": question,
            "has_error": error is not None,
            "error": error,
            "processing_time": processing_time
        }
        
        if error:
            return evaluation
        
        # Basic metrics
        evaluation.update({
            "generated_answer_length": len(generated_answer),
            "reference_answer_length": len(reference_answer) if reference_answer else 0,
            "has_reference_answer": bool(reference_answer)
        })
        
        # Citation quality
        citation_metrics = self.citation_quality_score(generated_answer, context_used)
        evaluation["citation_quality"] = citation_metrics
        
        # Semantic similarity (if reference answer exists)
        if reference_answer and self.similarity_model:
            similarity = self.semantic_similarity(reference_answer, generated_answer)
            evaluation["semantic_similarity"] = similarity
        
        # Factual consistency (if reference answer exists)
        if reference_answer and self.nli_model:
            consistency = self.factual_consistency(reference_answer, generated_answer)
            evaluation["factual_consistency"] = consistency
        
        # Answer completeness (basic heuristic)
        if generated_answer:
            # Check for structured answer with headings and bullets
            has_headings = bool(re.search(r'^#+\s|\n#+\s', generated_answer, re.MULTILINE))
            has_bullets = bool(re.search(r'^\s*[-*•]\s|\n\s*[-*•]\s', generated_answer, re.MULTILINE))
            has_summary = bool(re.search(r'summary|conclusion', generated_answer, re.IGNORECASE))
            
            evaluation["answer_structure"] = {
                "has_headings": has_headings,
                "has_bullets": has_bullets,
                "has_summary": has_summary
            }
        
        return evaluation
    
    def calculate_aggregate_scores(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate scores across all evaluations."""
        total_questions = len(evaluations)
        successful_evaluations = [e for e in evaluations if not e.get('has_error', False)]
        successful_count = len(successful_evaluations)
        
        if successful_count == 0:
            return {
                "total_questions": total_questions,
                "successful_evaluations": 0,
                "success_rate": 0.0,
                "error_rate": 1.0
            }
        
        # Basic metrics
        avg_processing_time = np.mean([e.get('processing_time', 0) for e in successful_evaluations])
        
        # Citation metrics
        citation_coverages = [e.get('citation_quality', {}).get('citation_coverage', 0) for e in successful_evaluations]
        citation_validities = [e.get('citation_quality', {}).get('citation_validity', 0) for e in successful_evaluations]
        avg_citation_coverage = np.mean(citation_coverages)
        avg_citation_validity = np.mean(citation_validities)
        
        # Semantic similarity
        similarities = [e.get('semantic_similarity', 0) for e in successful_evaluations if 'semantic_similarity' in e]
        avg_semantic_similarity = np.mean(similarities) if similarities else 0
        
        # Factual consistency
        entailments = [e.get('factual_consistency', {}).get('entailment', 0) for e in successful_evaluations if 'factual_consistency' in e]
        avg_entailment = np.mean(entailments) if entailments else 0
        
        # Answer structure
        with_headings = [e for e in successful_evaluations if e.get('answer_structure', {}).get('has_headings', False)]
        with_bullets = [e for e in successful_evaluations if e.get('answer_structure', {}).get('has_bullets', False)]
        with_summary = [e for e in successful_evaluations if e.get('answer_structure', {}).get('has_summary', False)]
        
        heading_rate = len(with_headings) / successful_count
        bullet_rate = len(with_bullets) / successful_count
        summary_rate = len(with_summary) / successful_count
        
        return {
            "total_questions": total_questions,
            "successful_evaluations": successful_count,
            "success_rate": successful_count / total_questions,
            "error_rate": (total_questions - successful_count) / total_questions,
            "avg_processing_time": float(avg_processing_time),
            "citation_metrics": {
                "avg_coverage": float(avg_citation_coverage),
                "avg_validity": float(avg_citation_validity)
            },
            "semantic_similarity": {
                "avg_score": float(avg_semantic_similarity),
                "evaluated_count": len(similarities)
            },
            "factual_consistency": {
                "avg_entailment": float(avg_entailment),
                "evaluated_count": len(entailments)
            },
            "answer_structure": {
                "heading_rate": float(heading_rate),
                "bullet_rate": float(bullet_rate),
                "summary_rate": float(summary_rate)
            }
        }
    
    def generate_detailed_report(self, evaluations: List[Dict[str, Any]], aggregate_scores: Dict[str, Any]) -> str:
        """Generate a detailed human-readable report."""
        report = []
        report.append("# RAG Evaluation Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append(f"- Total Questions: {aggregate_scores['total_questions']}")
        report.append(f"- Successful Evaluations: {aggregate_scores['successful_evaluations']}")
        report.append(f"- Success Rate: {aggregate_scores['success_rate']:.2%}")
        report.append(f"- Average Processing Time: {aggregate_scores['avg_processing_time']:.2f}s")
        report.append("")
        
        # Citation Quality
        report.append("## Citation Quality")
        report.append(f"- Average Citation Coverage: {aggregate_scores['citation_metrics']['avg_coverage']:.2%}")
        report.append(f"- Average Citation Validity: {aggregate_scores['citation_metrics']['avg_validity']:.2%}")
        report.append("")
        
        # Semantic Similarity
        if aggregate_scores['semantic_similarity']['evaluated_count'] > 0:
            report.append("## Semantic Similarity")
            report.append(f"- Average Similarity Score: {aggregate_scores['semantic_similarity']['avg_score']:.3f}")
            report.append(f"- Evaluated Pairs: {aggregate_scores['semantic_similarity']['evaluated_count']}")
            report.append("")
        
        # Factual Consistency
        if aggregate_scores['factual_consistency']['evaluated_count'] > 0:
            report.append("## Factual Consistency")
            report.append(f"- Average Entailment: {aggregate_scores['factual_consistency']['avg_entailment']:.3f}")
            report.append(f"- Evaluated Pairs: {aggregate_scores['factual_consistency']['evaluated_count']}")
            report.append("")
        
        # Answer Structure
        report.append("## Answer Structure")
        report.append(f"- Questions with Headings: {aggregate_scores['answer_structure']['heading_rate']:.2%}")
        report.append(f"- Questions with Bullets: {aggregate_scores['answer_structure']['bullet_rate']:.2%}")
        report.append(f"- Questions with Summary: {aggregate_scores['answer_structure']['summary_rate']:.2%}")
        report.append("")
        
        # Detailed results
        if self.detailed:
            report.append("## Detailed Results")
            for eval_result in evaluations:
                report.append(f"### Question {eval_result['id']}")
                report.append(f"**Question:** {eval_result['question']}")
                
                if eval_result.get('has_error'):
                    report.append(f"**Error:** {eval_result['error']}")
                else:
                    report.append(f"**Processing Time:** {eval_result['processing_time']:.2f}s")
                    report.append(f"**Answer Length:** {eval_result['generated_answer_length']} chars")
                    
                    if 'semantic_similarity' in eval_result:
                        report.append(f"**Semantic Similarity:** {eval_result['semantic_similarity']:.3f}")
                    
                    if 'factual_consistency' in eval_result:
                        fc = eval_result['factual_consistency']
                        report.append(f"**Factual Consistency:** Entailment={fc['entailment']:.3f}, Contradiction={fc['contradiction']:.3f}")
                    
                    cq = eval_result.get('citation_quality', {})
                    report.append(f"**Citation Coverage:** {cq.get('citation_coverage', 0):.2%}")
                    report.append(f"**Citation Validity:** {cq.get('citation_validity', 0):.2%}")
                
                report.append("")
        
        return "\n".join(report)


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load JSON file with error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures for benchmark files
        if 'data' in data and 'questions' not in data:
            data['questions'] = data.pop('data')
        
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise


def save_evaluation_report(evaluation_data: Dict[str, Any], output_path: str, detailed_report: str = None):
    """Save evaluation results and optional detailed report."""
    # Save JSON results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
    
    # Save detailed markdown report if requested
    if detailed_report:
        report_path = output_path.replace('.json', '_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(detailed_report)
        print(f"Detailed report saved to: {report_path}")
    
    print(f"Evaluation results saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RAG generated answers against benchmark")
    parser.add_argument("benchmark", help="Path to benchmark JSON file")
    parser.add_argument("results", help="Path to generated results JSON file")
    parser.add_argument("output", help="Path to evaluation output JSON file")
    parser.add_argument("--detailed", action="store_true", help="Generate detailed markdown report")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate input files
    if not Path(args.benchmark).exists():
        print(f"Error: Benchmark file not found: {args.benchmark}")
        return 1
    
    if not Path(args.results).exists():
        print(f"Error: Results file not found: {args.results}")
        return 1
    
    try:
        # Load data
        print(f"Loading benchmark from: {args.benchmark}")
        benchmark_data = load_json_file(args.benchmark)
        
        print(f"Loading results from: {args.results}")
        results_data = load_json_file(args.results)
        
        # Initialize evaluator
        evaluator = RAGEvaluator(detailed=args.detailed)
        
        # Evaluate each answer
        print("Evaluating answers...")
        evaluations = []
        
        # Create mapping of question text to result (more reliable than ID)
        results_map = {result['question']: result for result in results_data.get('results', [])}
        
        for question_data in benchmark_data['questions']:
            question_text = question_data['question']
            result_data = results_map.get(question_text, {})
            
            if not result_data:
                print(f"Warning: No result found for question: {question_text[:50]}...")
                result_data = {'error': 'No result found', 'generated_answer': '', 'processing_time': 0}
            
            evaluation = evaluator.evaluate_single_answer(question_data, result_data)
            evaluations.append(evaluation)
        
        # Calculate aggregate scores
        aggregate_scores = evaluator.calculate_aggregate_scores(evaluations)
        
        # Prepare final evaluation data
        evaluation_data = {
            "metadata": {
                "evaluated_at": datetime.utcnow().isoformat() + "Z",
                "benchmark_file": args.benchmark,
                "results_file": args.results,
                "detailed_report": args.detailed
            },
            "aggregate_scores": aggregate_scores,
            "individual_evaluations": evaluations
        }
        
        # Generate detailed report if requested
        detailed_report = None
        if args.detailed:
            detailed_report = evaluator.generate_detailed_report(evaluations, aggregate_scores)
            print("\n" + "="*60)
            print("EVALUATION SUMMARY")
            print("="*60)
            print(detailed_report.split("## Detailed Results")[0])  # Show summary part
        
        # Save results
        save_evaluation_report(evaluation_data, args.output, detailed_report)
        
        # Print final summary
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Success Rate: {aggregate_scores['success_rate']:.2%}")
        print(f"Avg Processing Time: {aggregate_scores['avg_processing_time']:.2f}s")
        print(f"Avg Citation Coverage: {aggregate_scores['citation_metrics']['avg_coverage']:.2%}")
        if aggregate_scores['semantic_similarity']['evaluated_count'] > 0:
            print(f"Avg Semantic Similarity: {aggregate_scores['semantic_similarity']['avg_score']:.3f}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
