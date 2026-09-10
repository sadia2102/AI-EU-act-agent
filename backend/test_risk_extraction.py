#!/usr/bin/env python3
"""
Test improved risk extraction logic on existing results without API calls.
"""

import json
import sys
from typing import Dict, Any

def extract_risk_level_improved(analysis: str) -> str:
    """Improved risk level extraction with stricter criteria."""
    import re
    
    # Priority 1: Look for explicit risk level statements
    risk_patterns = [
        r'risk level[:\s]*([a-z-]+)',
        r'classified as[:\s]*([a-z-]+)',
        r'risk classification[:\s]*([a-z-]+)',
        r'risk[:\s]*([a-z-]+)',
    ]
    
    valid_risk_levels = ['prohibited', 'high-risk', 'limited-risk', 'minimal-risk', 'high risk', 'limited risk', 'minimal risk']
    
    for pattern in risk_patterns:
        matches = re.findall(pattern, analysis, re.IGNORECASE)
        for match in matches:
            normalized = match.lower().replace(' ', '-')
            if normalized in valid_risk_levels:
                return normalized
    
    analysis_lower = analysis.lower()
    
    # Priority 2: Check for PROHIBITED practices (Article 5) - VERY STRICT CRITERIA
    prohibited_keywords = [
        'subliminal technique', 'subliminal techniques', 'subliminal influence',
        'manipulative technique', 'manipulative techniques', 'manipulative influence',
        'exploit vulnerability', 'exploiting vulnerabilities', 'exploit vulnerabilities',
        'social scoring', 'social score',
        'real-time remote biometric', 'real-time remote biometric identification'
    ]
    
    has_prohibited_keyword = any(phrase in analysis_lower for phrase in prohibited_keywords)
    
    has_manipulation_context = any(phrase in analysis_lower for phrase in [
        'distort behavior', 'materially distort', 'without awareness', 'without conscious',
        'nudge', 'influence decision', 'manipulate', 'exploit'
    ])
    
    if has_prohibited_keyword and has_manipulation_context:
        return 'prohibited'
    
    # Priority 3: Check for LIMITED-RISK systems (Article 50) - BEFORE high-risk
    if any(phrase in analysis_lower for phrase in [
        'content generation', 'content generator', 'generate content',
        'chatbot', 'conversational ai', 'conversational assistant',
        'recommendation system', 'recommender system', 'recommendation ai',
        'emotion recognition', 'deepfake', 'synthetic media',
        'article 50', 'transparency obligation', 'transparency requirement',
        'limited-risk', 'limited risk'
    ]):
        return 'limited-risk'
    
    # Priority 4: Medical/Healthcare systems are HIGH-RISK by default
    if any(phrase in analysis_lower for phrase in [
        'medical', 'diagnostic', 'healthcare', 'clinical', 'patient', 'treatment',
        'medical device', 'health monitoring', 'medical imaging', 'diagnosis',
        'medical diagnostic', 'health care'
    ]):
        return 'high-risk'
    
    # Priority 5: Look for HIGH-RISK systems (Annex III)
    if any(phrase in analysis_lower for phrase in [
        'inventory optimization', 'inventory management', 'logistics optimization',
        'warehouse management', 'supply chain optimization'
    ]):
        pass
    elif any(phrase in analysis_lower for phrase in [
        'surveillance', 'public space surveillance', 'biometric identification',
        'biometric screening', 'facial recognition', 'remote biometric',
        'annex iii', 'high-risk systems', 'article 6', 'conformity assessment',
        'fundamental rights impact assessment', 'employment', 'credit scoring', 'law enforcement',
        'education assessment', 'legal document', 'autonomous vehicle', 'border control',
        'immigration', 'insurance', 'criminal justice', 'risk management system', 
        'technical documentation', 'data governance', 'safety component'
    ]):
        return 'high-risk'
    
    # Priority 6: Default to minimal-risk
    if any(phrase in analysis_lower for phrase in [
        'minimal-risk', 'minimal risk', 'no specific obligations',
        'unregulated', 'gaming', 'spam filter', 'traffic optimization',
        'agricultural tool', 'inventory'
    ]):
        return 'minimal-risk'
    
    return 'unknown'

def main():
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        results_file = 'scenario_results_30_improved_v2.json'
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    scenarios_file = 'scenarios_30_sample.json'
    with open(scenarios_file, 'r') as f:
        scenarios_data = json.load(f)
    
    print("=" * 80)
    print("TESTING IMPROVED RISK EXTRACTION LOGIC")
    print("=" * 80)
    
    changes = []
    for i, result in enumerate(data['results']):
        scenario = scenarios_data['data'][i]
        old_risk = result['generated_risk_level']
        benchmark_risk = scenario['risk_level']
        
        # Extract analysis text from context_used
        analysis_text = result.get('context_used', '')
        new_risk = extract_risk_level_improved(analysis_text)
        
        if old_risk != new_risk:
            changes.append({
                'id': result['id'],
                'system_type': result['system_type'],
                'domain': result['domain'],
                'old_risk': old_risk,
                'new_risk': new_risk,
                'benchmark_risk': benchmark_risk,
                'old_correct': old_risk == benchmark_risk,
                'new_correct': new_risk == benchmark_risk
            })
    
    print(f"\nTotal scenarios: {len(data['results'])}")
    print(f"Risk level changes: {len(changes)}")
    
    if changes:
        print("\nCHANGES:")
        print("-" * 80)
        for change in changes:
            status = "✓ IMPROVED" if (not change['old_correct'] and change['new_correct']) else \
                    "✗ REGRESSED" if (change['old_correct'] and not change['new_correct']) else \
                    "~ CHANGED"
            print(f"\n{status}: {change['id']}")
            print(f"  System: {change['system_type']} in {change['domain']}")
            print(f"  Old: {change['old_risk']} (correct: {change['old_correct']})")
            print(f"  New: {change['new_risk']} (correct: {change['new_correct']})")
            print(f"  Benchmark: {change['benchmark_risk']}")
    
    # Calculate new accuracy
    correct_count = sum(1 for r in data['results'] if r['generated_risk_level'] == scenarios_data['data'][data['results'].index(r)]['risk_level'])
    old_accuracy = correct_count / len(data['results'])
    
    new_correct_count = sum(1 for c in changes if c['new_correct']) + sum(1 for r in data['results'] if r['id'] not in [c['id'] for c in changes] and r['generated_risk_level'] == scenarios_data['data'][data['results'].index(r)]['risk_level'])
    new_accuracy = new_correct_count / len(data['results'])
    
    print("\n" + "=" * 80)
    print(f"Old accuracy: {old_accuracy:.2%}")
    print(f"New accuracy: {new_accuracy:.2%}")
    print(f"Improvement: {(new_accuracy - old_accuracy):.2%}")
    print("=" * 80)

if __name__ == "__main__":
    main()
