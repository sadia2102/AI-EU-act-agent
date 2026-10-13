#!/usr/bin/env python3
"""
Scenario Evaluation Script for AI Regulation Compliance Analysis

Evaluates generated scenario analyses against benchmark scenarios using:
- Risk level classification accuracy
- Obligation coverage and relevance
- Article identification accuracy
- Citation quality and coverage
- Semantic similarity of obligations

Usage:
  python scenario_evaluation.py scenarios.json scenario_results.json scenario_evaluation.json
  python scenario_evaluation.py scenarios.json scenario_results.json scenario_evaluation.json --detailed
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


class ScenarioEvaluator:
    def __init__(self, detailed: bool = False):
        self.detailed = detailed
        self.similarity_model = None
        
        # Initialize similarity model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            print("Loading semantic similarity model...")
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def normalize_risk_level(self, risk_level: str) -> str:
        """Normalize risk level strings."""
        if not risk_level:
            return "unknown"
        
        normalized = risk_level.strip().lower()
        normalized = normalized.replace(' ', '-').replace('_', '-')
        synonyms = {
            'limited': 'limited-risk',
            'limitedrisk': 'limited-risk',
            'limited-risk': 'limited-risk',
            'minimal': 'minimal-risk',
            'minimalrisk': 'minimal-risk',
            'minimal-risk': 'minimal-risk',
            'high': 'high-risk',
            'highrisk': 'high-risk',
            'high-risk': 'high-risk',
            'prohibited': 'prohibited'
        }
        normalized = synonyms.get(normalized, normalized)
        valid_levels = ['prohibited', 'high-risk', 'limited-risk', 'minimal-risk']
        
        return normalized if normalized in valid_levels else "unknown"
    
    def evaluate_risk_classification(self, benchmark_risk: str, generated_risk: str) -> Dict[str, Any]:
        """Evaluate risk level classification accuracy."""
        benchmark_norm = self.normalize_risk_level(benchmark_risk)
        generated_norm = self.normalize_risk_level(generated_risk)
        
        is_correct = benchmark_norm == generated_norm
        is_partial_match = self._partial_risk_match(benchmark_norm, generated_norm)
        
        return {
            "benchmark_risk": benchmark_norm,
            "generated_risk": generated_norm,
            "is_correct": is_correct,
            "is_partial_match": is_partial_match,
            "risk_distance": self._risk_level_distance(benchmark_norm, generated_norm)
        }
    
    def _partial_risk_match(self, benchmark: str, generated: str) -> bool:
        """Check for partial risk level matches."""
        if benchmark == "unknown" or generated == "unknown":
            return False
        
        # Consider high-risk vs high risk as match
        benchmark_clean = benchmark.replace('-', '')
        generated_clean = generated.replace('-', '')
        
        return benchmark_clean == generated_clean
    
    def _risk_level_distance(self, benchmark: str, generated: str) -> int:
        """Calculate distance between risk levels (0 = exact match, higher = further apart)."""
        risk_hierarchy = {
            'prohibited': 0,
            'high-risk': 1,
            'limited-risk': 2,
            'minimal-risk': 3,
            'unknown': 999
        }
        
        return abs(risk_hierarchy.get(benchmark, 999) - risk_hierarchy.get(generated, 999))
    
    def evaluate_obligations(self, benchmark_obligations: List[str], generated_obligations: List[str]) -> Dict[str, Any]:
        """Evaluate obligation coverage and relevance."""
        if not benchmark_obligations:
            return {
                "benchmark_count": 0,
                "generated_count": len(generated_obligations),
                "coverage_ratio": 0.0,
                "semantic_similarity": 0.0,
                "obligation_matches": []
            }
        
        # Calculate coverage (capped at 1.0 to represent completeness)
        coverage_ratio = min(len(generated_obligations) / len(benchmark_obligations), 1.0) if benchmark_obligations else 0
        
        # Calculate semantic similarity
        semantic_similarity = 0.0
        obligation_matches = []
        
        if self.similarity_model and benchmark_obligations and generated_obligations:
            similarity_matrix = self._calculate_obligation_similarity(benchmark_obligations, generated_obligations)
            semantic_similarity = np.mean(similarity_matrix) if similarity_matrix.size > 0 else 0.0
            
            # Find best matches
            for i, bench_obl in enumerate(benchmark_obligations):
                if similarity_matrix.size > 0 and i < similarity_matrix.shape[0]:
                    best_match_idx = np.argmax(similarity_matrix[i])
                    best_similarity = similarity_matrix[i][best_match_idx]
                    if best_match_idx < len(generated_obligations):
                        obligation_matches.append({
                            "benchmark": bench_obl,
                            "generated": generated_obligations[best_match_idx],
                            "similarity": float(best_similarity)
                        })
        
        return {
            "benchmark_count": len(benchmark_obligations),
            "generated_count": len(generated_obligations),
            "coverage_ratio": coverage_ratio,
            "semantic_similarity": semantic_similarity,
            "obligation_matches": obligation_matches[:5]  # Top 5 matches
        }
    
    def _calculate_obligation_similarity(self, benchmark: List[str], generated: List[str]) -> np.ndarray:
        """Calculate similarity matrix between benchmark and generated obligations."""
        if not self.similarity_model or not benchmark or not generated:
            return np.array([])
        
        try:
            benchmark_embeddings = self.similarity_model.encode(benchmark)
            generated_embeddings = self.similarity_model.encode(generated)
            
            similarity_matrix = cosine_similarity(benchmark_embeddings, generated_embeddings)
            return similarity_matrix.astype(float)
        except Exception:
            return np.array([])
    
    def evaluate_articles(self, benchmark_articles: List[int], generated_articles: List[int]) -> Dict[str, Any]:
        """Evaluate article identification accuracy."""
        benchmark_set = set(benchmark_articles)
        generated_set = set(generated_articles)
        
        # Calculate metrics
        correct_articles = benchmark_set & generated_set
        missed_articles = benchmark_set - generated_set
        extra_articles = generated_set - benchmark_set
        
        precision = len(correct_articles) / len(generated_set) if generated_set else 0.0
        recall = len(correct_articles) / len(benchmark_set) if benchmark_set else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "benchmark_articles": sorted(list(benchmark_set)),
            "generated_articles": sorted(list(generated_set)),
            "correct_articles": sorted(list(correct_articles)),
            "missed_articles": sorted(list(missed_articles)),
            "extra_articles": sorted(list(extra_articles)),
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
    
    def evaluate_citations(self, sources: List[str], context_used: str) -> Dict[str, Any]:
        """Evaluate citation quality and coverage."""
        # Extract citations from context
        context_citations = self._extract_citations(context_used)
        
        # Count unique sources
        unique_sources = len(set(sources))
        
        # Check citation validity
        valid_citations = 0
        for citation in sources:
            if self._is_valid_citation_format(citation):
                valid_citations += 1
        
        citation_validity = valid_citations / len(sources) if sources else 0
        
        return {
            "total_citations": len(sources),
            "unique_sources": unique_sources,
            "valid_citations": valid_citations,
            "citation_validity": citation_validity,
            "context_citations": len(context_citations)
        }
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract citation patterns from text."""
        citation_pattern = r'\[([^;]+);\s*([^;]+);\s*chunk_index=([^\]]+)\]'
        return re.findall(citation_pattern, text)
    
    def _is_valid_citation_format(self, citation: str) -> bool:
        """Check if citation follows expected format."""
        pattern = r'\[([^;]+);\s*([^;]+);\s*chunk_index=([^\]]+)\]'
        return bool(re.match(pattern, citation))
    
    def evaluate_single_scenario(self, scenario_data: Dict[str, Any], result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single scenario analysis."""
        scenario_id = result_data.get('id', 'unknown')
        
        # Extract data
        benchmark_risk = scenario_data.get('risk_level', '')
        generated_risk = result_data.get('generated_risk_level', '')
        benchmark_obligations = scenario_data.get('obligations', [])
        generated_obligations = result_data.get('generated_obligations', [])
        benchmark_articles = scenario_data.get('related_articles', [])
        generated_articles = result_data.get('relevant_articles', [])
        sources = result_data.get('sources', [])
        context_used = result_data.get('context_used', '')
        processing_time = result_data.get('processing_time', 0)
        error = result_data.get('error')
        
        evaluation = {
            "id": scenario_id,
            "role": scenario_data.get('role', ''),
            "system_type": scenario_data.get('system_type', ''),
            "domain": scenario_data.get('domain', ''),
            "has_error": error is not None,
            "error": error,
            "processing_time": processing_time
        }
        
        if error:
            return evaluation
        
        # Risk classification evaluation
        risk_eval = self.evaluate_risk_classification(benchmark_risk, generated_risk)
        evaluation["risk_evaluation"] = risk_eval
        
        # Obligations evaluation
        obligations_eval = self.evaluate_obligations(benchmark_obligations, generated_obligations)
        evaluation["obligations_evaluation"] = obligations_eval
        
        # Articles evaluation
        articles_eval = self.evaluate_articles(benchmark_articles, generated_articles)
        evaluation["articles_evaluation"] = articles_eval
        
        # Citations evaluation
        citations_eval = self.evaluate_citations(sources, context_used)
        evaluation["citations_evaluation"] = citations_eval
        
        return evaluation
    
    def calculate_aggregate_scores(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate scores across all evaluations."""
        total_scenarios = len(evaluations)
        successful_evaluations = [e for e in evaluations if not e.get('has_error', False)]
        successful_count = len(successful_evaluations)
        
        if successful_count == 0:
            return {
                "total_scenarios": total_scenarios,
                "successful_evaluations": 0,
                "success_rate": 0.0,
                "error_rate": 1.0
            }
        
        # Risk classification metrics
        correct_classifications = sum(1 for e in successful_evaluations 
                                     if e.get('risk_evaluation', {}).get('is_correct', False))
        risk_accuracy = correct_classifications / successful_count
        
        # Average risk distance
        risk_distances = [e.get('risk_evaluation', {}).get('risk_distance', 999) 
                         for e in successful_evaluations]
        avg_risk_distance = np.mean([d for d in risk_distances if d < 999]) if risk_distances else 0
        
        # Obligations metrics
        coverage_ratios = [e.get('obligations_evaluation', {}).get('coverage_ratio', 0) 
                          for e in successful_evaluations]
        avg_coverage = np.mean(coverage_ratios) if coverage_ratios else 0
        
        semantic_similarities = [e.get('obligations_evaluation', {}).get('semantic_similarity', 0) 
                                for e in successful_evaluations]
        avg_semantic_similarity = np.mean(semantic_similarities) if semantic_similarities else 0
        
        # Articles metrics
        precisions = [e.get('articles_evaluation', {}).get('precision', 0) 
                     for e in successful_evaluations]
        avg_precision = np.mean(precisions) if precisions else 0
        
        recalls = [e.get('articles_evaluation', {}).get('recall', 0) 
                  for e in successful_evaluations]
        avg_recall = np.mean(recalls) if recalls else 0
        
        f1_scores = [e.get('articles_evaluation', {}).get('f1_score', 0) 
                    for e in successful_evaluations]
        avg_f1_score = np.mean(f1_scores) if f1_scores else 0
        
        # Citations metrics
        citation_validities = [e.get('citations_evaluation', {}).get('citation_validity', 0) 
                              for e in successful_evaluations]
        avg_citation_validity = np.mean(citation_validities) if citation_validities else 0
        
        avg_processing_time = np.mean([e.get('processing_time', 0) for e in successful_evaluations])
        
        return {
            "total_scenarios": total_scenarios,
            "successful_evaluations": successful_count,
            "success_rate": float(successful_count / total_scenarios),
            "error_rate": float((total_scenarios - successful_count) / total_scenarios),
            "avg_processing_time": float(avg_processing_time),
            "risk_classification": {
                "accuracy": float(risk_accuracy),
                "avg_distance": float(avg_risk_distance)
            },
            "obligations": {
                "avg_coverage": float(avg_coverage),
                "avg_semantic_similarity": float(avg_semantic_similarity)
            },
            "articles": {
                "avg_precision": float(avg_precision),
                "avg_recall": float(avg_recall),
                "avg_f1_score": float(avg_f1_score)
            },
            "citations": {
                "avg_validity": float(avg_citation_validity)
            }
        }
    
    def generate_detailed_report(self, evaluations: List[Dict[str, Any]], aggregate_scores: Dict[str, Any]) -> str:
        """Generate a detailed human-readable report."""
        report = []
        report.append("# Scenario Evaluation Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append(f"- Total Scenarios: {aggregate_scores['total_scenarios']}")
        report.append(f"- Successful Evaluations: {aggregate_scores['successful_evaluations']}")
        report.append(f"- Success Rate: {aggregate_scores['success_rate']:.2%}")
        report.append(f"- Average Processing Time: {aggregate_scores['avg_processing_time']:.2f}s")
        report.append("")
        
        # Risk Classification
        report.append("## Risk Classification Accuracy")
        report.append(f"- Accuracy: {aggregate_scores['risk_classification']['accuracy']:.2%}")
        report.append(f"- Average Distance: {aggregate_scores['risk_classification']['avg_distance']:.2f}")
        report.append("")
        
        # Obligations
        report.append("## Obligations Analysis")
        report.append(f"- Average Coverage: {aggregate_scores['obligations']['avg_coverage']:.2%}")
        report.append(f"- Average Semantic Similarity: {aggregate_scores['obligations']['avg_semantic_similarity']:.3f}")
        report.append("")
        
        # Articles
        report.append("## Article Identification")
        report.append(f"- Average Precision: {aggregate_scores['articles']['avg_precision']:.3f}")
        report.append(f"- Average Recall: {aggregate_scores['articles']['avg_recall']:.3f}")
        report.append(f"- Average F1 Score: {aggregate_scores['articles']['avg_f1_score']:.3f}")
        report.append("")
        
        # Citations
        report.append("## Citation Quality")
        report.append(f"- Average Validity: {aggregate_scores['citations']['avg_validity']:.2%}")
        report.append("")
        
        # Detailed results
        if self.detailed:
            report.append("## Detailed Results")
            for eval_result in evaluations:
                report.append(f"### Scenario {eval_result['id']}")
                report.append(f"**Role:** {eval_result['role']}")
                report.append(f"**System Type:** {eval_result['system_type']}")
                report.append(f"**Domain:** {eval_result['domain']}")
                
                if eval_result.get('has_error'):
                    report.append(f"**Error:** {eval_result['error']}")
                else:
                    report.append(f"**Processing Time:** {eval_result['processing_time']:.2f}s")
                    
                    # Risk classification
                    risk_eval = eval_result.get('risk_evaluation', {})
                    report.append(f"**Risk Classification:** {risk_eval.get('generated_risk', 'N/A')} "
                                f"(Benchmark: {risk_eval.get('benchmark_risk', 'N/A')}) "
                                f"✓" if risk_eval.get('is_correct') else "✗")
                    
                    # Obligations
                    obl_eval = eval_result.get('obligations_evaluation', {})
                    report.append(f"**Obligations Coverage:** {obl_eval.get('coverage_ratio', 0):.2%}")
                    report.append(f"**Obligation Similarity:** {obl_eval.get('semantic_similarity', 0):.3f}")
                    
                    # Articles
                    art_eval = eval_result.get('articles_evaluation', {})
                    report.append(f"**Article F1 Score:** {art_eval.get('f1_score', 0):.3f}")
                
                report.append("")
        
        return "\n".join(report)


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load JSON file with error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures for scenarios files
        if 'data' in data and 'scenarios' not in data:
            data['scenarios'] = data.pop('data')
        
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
    parser = argparse.ArgumentParser(description="Evaluate scenario analyses against benchmark")
    parser.add_argument("scenarios", help="Path to scenarios JSON file")
    parser.add_argument("results", help="Path to generated results JSON file")
    parser.add_argument("output", help="Path to evaluation output JSON file")
    parser.add_argument("--detailed", action="store_true", help="Generate detailed markdown report")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate input files
    if not Path(args.scenarios).exists():
        print(f"Error: Scenarios file not found: {args.scenarios}")
        return 1
    
    if not Path(args.results).exists():
        print(f"Error: Results file not found: {args.results}")
        return 1
    
    try:
        # Load data
        print(f"Loading scenarios from: {args.scenarios}")
        scenarios_data = load_json_file(args.scenarios)
        
        print(f"Loading results from: {args.results}")
        results_data = load_json_file(args.results)
        
        # Initialize evaluator
        evaluator = ScenarioEvaluator(detailed=args.detailed)
        
        # Evaluate each scenario
        print("Evaluating scenarios...")
        evaluations = []
        
        # Create mapping of scenario text to result
        results_map = {result['id']: result for result in results_data.get('results', [])}
        
        for i, scenario_data in enumerate(scenarios_data['scenarios']):
            scenario_id = f"scenario_{i + 1}"
            result_data = results_map.get(scenario_id, {})
            
            if not result_data:
                print(f"Warning: No result found for scenario {scenario_id}")
                result_data = {'error': 'No result found', 'generated_risk_level': '', 'generated_obligations': [], 'relevant_articles': [], 'sources': [], 'context_used': '', 'processing_time': 0}
            
            evaluation = evaluator.evaluate_single_scenario(scenario_data, result_data)
            evaluations.append(evaluation)
        
        # Calculate aggregate scores
        aggregate_scores = evaluator.calculate_aggregate_scores(evaluations)
        
        # Prepare final evaluation data
        evaluation_data = {
            "metadata": {
                "evaluated_at": datetime.utcnow().isoformat() + "Z",
                "scenarios_file": args.scenarios,
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
        print(f"Risk Classification Accuracy: {aggregate_scores['risk_classification']['accuracy']:.2%}")
        print(f"Article F1 Score: {aggregate_scores['articles']['avg_f1_score']:.3f}")
        print(f"Obligation Coverage: {aggregate_scores['obligations']['avg_coverage']:.2%}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
