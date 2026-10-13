#!/usr/bin/env python3
"""
Analyze scenario classification results and generate comprehensive performance report.
"""

import json
import sys

def main():
    # Load evaluation data from command line argument
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file = 'scenario_evaluation_30_improved_v2.json'
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    scenarios = data['individual_evaluations']
    skipped = 0
    
    # Initialize counters for precision and recall calculation
    # TP = True Positives (correctly predicted as class)
    # FP = False Positives (incorrectly predicted as class)
    # FN = False Negatives (actual class but predicted as something else)
    categories = {
        'prohibited': {'TP': 0, 'FP': 0, 'FN': 0, 'total_actual': 0, 'total_predicted': 0},
        'high-risk': {'TP': 0, 'FP': 0, 'FN': 0, 'total_actual': 0, 'total_predicted': 0},
        'limited-risk': {'TP': 0, 'FP': 0, 'FN': 0, 'total_actual': 0, 'total_predicted': 0},
        'minimal-risk': {'TP': 0, 'FP': 0, 'FN': 0, 'total_actual': 0, 'total_predicted': 0},
        'unknown': {'TP': 0, 'FP': 0, 'FN': 0, 'total_actual': 0, 'total_predicted': 0}
    }
    
    # Analyze each scenario
    for scenario in scenarios:
        risk_eval = scenario.get('risk_evaluation')
        if not risk_eval or scenario.get('has_error'):
            skipped += 1
            continue

        benchmark_risk = risk_eval.get('benchmark_risk')
        generated_risk = risk_eval.get('generated_risk')
        is_correct = risk_eval.get('is_correct')

        if benchmark_risk is None or generated_risk is None or is_correct is None:
            skipped += 1
            continue
        
        # Count total scenarios per category
        if benchmark_risk not in categories:
            categories[benchmark_risk] = {'TP': 0, 'FP': 0, 'FN': 0, 'total_actual': 0, 'total_predicted': 0}
        if generated_risk not in categories:
            categories[generated_risk] = {'TP': 0, 'FP': 0, 'FN': 0, 'total_actual': 0, 'total_predicted': 0}

        categories[benchmark_risk]['total_actual'] += 1
        categories[generated_risk]['total_predicted'] += 1
        
        # Calculate TP, FP, FN for each category
        if is_correct:
            categories[benchmark_risk]['TP'] += 1
        else:
            categories[benchmark_risk]['FN'] += 1
            categories[generated_risk]['FP'] += 1
    
    # Print detailed analysis
    print("=" * 60)
    print("SCENARIO CLASSIFICATION PERFORMANCE ANALYSIS")
    print("=" * 60)
    print()
    
    # Overall summary
    total_scenarios = sum(cat['TP'] + cat['FN'] for cat in categories.values())
    total_correct = sum(cat['TP'] for cat in categories.values())
    total_incorrect = sum(cat['FP'] + cat['FN'] for cat in categories.values())
    accuracy = (total_correct / total_scenarios) * 100 if total_scenarios else 0
    
    print(f"OVERALL PERFORMANCE:")
    print(f"Total Scenarios: {total_scenarios}")
    if skipped:
        print(f"Skipped (missing risk data or errors): {skipped}")
    print(f"Correctly Classified: {total_correct}")
    print(f"Incorrectly Classified: {total_incorrect}")
    print(f"Classification Accuracy: {accuracy:.2f}%")
    print()
    
    # Performance by category
    print("PERFORMANCE BY RISK CATEGORY:")
    print("-" * 50)
    print("Formula: Precision = TP/(TP+FP) | Recall = TP/(TP+FN)")
    print()
    
    for category, stats in categories.items():
        if stats['total_actual'] > 0 or stats['total_predicted'] > 0:
            # Calculate precision: TP/(TP+FP)
            precision = stats['TP'] / (stats['TP'] + stats['FP']) if (stats['TP'] + stats['FP']) > 0 else 0
            
            # Calculate recall: TP/(TP+FN)
            recall = stats['TP'] / (stats['TP'] + stats['FN']) if (stats['TP'] + stats['FN']) > 0 else 0
            
            # Calculate F1
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n{category.upper()}:")
            print(f"  True Positives (TP): {stats['TP']}")
            print(f"  False Positives (FP): {stats['FP']}")
            print(f"  False Negatives (FN): {stats['FN']}")
            print(f"  Total Actual: {stats['total_actual']}")
            print(f"  Total Predicted: {stats['total_predicted']}")
            print(f"  Precision: {precision:.3f} ({precision*100:.1f}%)")
            print(f"  Recall: {recall:.3f} ({recall*100:.1f}%)")
            print(f"  F1 Score: {f1:.3f}")
    
    print()
    print("=" * 60)

if __name__ == "__main__":
    main()
