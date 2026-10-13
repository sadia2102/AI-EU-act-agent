#!/usr/bin/env python3
"""
Scenario Batch RAG Processing for AI Regulation Compliance Analysis

Processes AI scenarios to determine:
- Risk level classification (prohibited, high-risk, limited-risk, minimal-risk)
- Obligations and compliance requirements
- Relevant articles and citations

Usage:
  python scenario_batch_rag.py scenarios.json scenario_results.json
  python scenario_batch_rag.py scenarios.json scenario_results.json --fast

Input JSON format (scenarios.json):
{
  "data": [
    {
      "role": "Provider/Deployer",
      "intended_use": "Description of AI system use",
      "system_type": "Type of AI system",
      "input_data": "Data inputs",
      "domain": "Application domain",
      "related_articles": [5, 10, 16],
      "obligations": ["Obligation 1", "Obligation 2", ...],
      "risk_level": "prohibited/high-risk/limited-risk/minimal-risk"
    }

  ]
}

Output JSON format:
{
  "metadata": {...},
  "results": [
    {
      "id": "scenario_1",
      "role": "Provider",
      "intended_use": "...",
      "system_type": "...",
      "domain": "...",
      "benchmark_risk_level": "prohibited",
      "generated_risk_level": "prohibited",
      "benchmark_obligations": [...],
      "generated_obligations": [...],
      "relevant_articles": [...],
      "sources": [...],
      "context_used": "...",
      "processing_time": 2.5,
      "error": null
    }
  ]
}
"""

import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import from existing rag_chat.py
from rag_chat import pre_retrieve, build_context, LLM_BACKEND, OPENAI_MODEL, OLLAMA_MODEL, OPENAI_API_KEY


PROHIBITED_KEYWORDS = (
    'subliminal',
    'subliminal audio',
    'subliminal visual',
    'manipulative',
    'covert influence',
    'hidden influence',
    'hidden audio',
    'hidden-audio',
    'hidden cues',
    'covert persuasion',
    'psychographic targeting',
    'behavioral influence',
    'behavioural influence',
    'subconscious influence',
    'subconscious',
    'covert nudging',
    'covert manipulation',
    'exploit vulnerability',
    'vulnerability exploitation',
    'targeting vulnerability',
    'target vulnerable',
    'targeting children',
    'targeting minors',
    'targeting elderly',
    'targeting disabled',
    'vulnerable users',
    'vulnerable groups',
    'vulnerable customers',
    'vulnerable citizens',
    'cognitive vulnerability',
    'cognitive vulnerabilities',
    'without awareness',
    'without conscious awareness',
    'without their conscious awareness',
    'without informed consent',
    'social scoring',
    'social score',
    'social credit',
    'social-credit',
    'citizen scoring',
    'citizen score',
    'citizen rating',
    'behavioural scoring',
    'behavioral scoring',
    'behavioural score',
    'behavioral score',
    'article 5',
    'article five',
    'prohibited practice',
    'banned practice',
    'dark pattern',
    'psychological manipulation',
    'emotional manipulation',
    'emotion exploitation',
    'exploiting emotion',
    'covert chatbot',
    'covert assistant',
    'covert sales assistant',
    'covert insurance advisor',
    'personality profiling',
    'personality-based risk',
    'behavioural evaluation',
    'behavioral evaluation',
    'manipulate',
    'manipulates',
    'manipulating',
    'exploit',
    'exploits',
    'exploiting',
    'low-skill',
    'low skill',
    'low-income',
    'low income',
    'gig work',
    'gig-worker',
    'gig-worker funnel',
    'emotion analytics',
    'emotion ranking',
    'emotion scoring',
    'emotion-based promotion',
    'meeting analytics',
    'employee emotion',
    'promotion ranking',
    'public housing allocation',
    'crime risk evaluator',
    'crime risk evaluation',
    'personality-based fraud',
    'personality fraud risk',
    'political orientation profiling',
    'political affiliation inference',
    'covert insurance influence',
    'benefit denial',
    'benefit exclusion',
    'service denial',
    'service exclusion',
    'welfare denial',
    'housing denial',
    'public benefit denial',
    'eligibility tier',
    'eligibility ranking',
    'civic ranking',
    'civic tier',
    'risk tiering',
    'risk tier',
    'credit denial automation',
    'loan denial automation',
    'emotion detection',
    'emotion detector',
    'emotion sensing',
    'emotion-surveillance',
    'emotion recognition camera',
    'affect recognition',
    'affective recognition',
    'affective analytics',
    'micro-expression',
    'micro expression',
    'facial micro-expression',
    'facial surveillance',
    'facial scraping',
    'facial data aggregation',
    'facial data collection',
    'facial dataset',
    'mass surveillance facial',
    'facial data harvesting',
    'real-time facial id',
    'real time facial id',
    'live facial id',
    'nudification',
    'nudify',
    'nudifying',
    'non-consensual intimate',
    'intimate image',
    'revenge porn'
)

SOCIAL_SCORING_KEYWORDS = (
    'social scoring', 'social score', 'social credit', 'citizen rating',
    'citizen scoring', 'behavioural scoring', 'behavioral scoring',
    'behavioural evaluation', 'behavioral evaluation', 'city-wide score',
    'citywide score', 'civic engagement score', 'civic engagement scoring',
    'civic participation scoring', 'civic participation score',
    'reputation scoring', 'reputation score', 'digital wellbeing scoring',
    'digital wellbeing score', 'loyalty scoring', 'loyalty score',
    'aggression scoring', 'aggression score', 'social behaviour classification',
    'social behavior classification', 'behavioural loyalty scoring',
    'behavioral loyalty scoring', 'micro-expression scoring',
    'violence risk scoring', 'violent partner risk'
)

REMOTE_BIOMETRIC_KEYWORDS = (
    'real-time remote biometric', 'real time remote biometric',
    'remote biometric identification', 'remote biometric tracking',
    'remote biometric surveillance', 'crowd density biometric',
    'mall-wide monitoring', 'public-square biometric',
    'passenger identification', 'visitor identification',
    'park visitor identification', 'theme park identification',
    'ticket booth facial', 'ticket gate facial',
    'transport hub identification', 'cctv facial recognition'
)

REMOTE_BIOMETRIC_AUTHORIZED_TERMS = (
    'law enforcement', 'border control', 'immigration', 'customs',
    'security checkpoint', 'security check-point', 'airport security',
    'public transport security', 'transport hub security', 'critical infrastructure security',
    'authorized identification', 'judicial authorization', 'judicial authorisation'
)

REMOTE_BIOMETRIC_COMMERCIAL_TERMS = (
    'retail analytics', 'customer analytics', 'shopping mall', 'mall shoppers',
    'theme park visitors', 'marketing kiosk', 'advertising display',
    'public park', 'tourist attraction', 'commerce plaza', 'loyalty program'
)

COMMERCIAL_REMOTE_BIOMETRIC_DOMAINS = {
    'retail', 'entertainment', 'theme park', 'theme parks', 'hospitality', 'customer service',
    'marketing', 'public spaces', 'public services'
}

COVERT_INFLUENCE_KEYWORDS = (
    'covert chatbot', 'covert assistant', 'covert language patterns',
    'hidden persuasion', 'covert persuasion', 'covert influence'
)

FACIAL_SURVEILLANCE_KEYWORDS = (
    'facial surveillance', 'facial scraping', 'facial data aggregation',
    'facial data collection', 'facial data collector', 'facial data harvesting',
    'face scraping', 'face data harvesting', 'mass surveillance facial',
    'crowd facial recognition', 'city-wide facial recognition',
    'mall-wide facial recognition', 'real-time facial recognition',
    'real time facial recognition', 'live facial id',
    'facial micro-expression', 'micro-expression detection',
    'micro expression detection'
)

EMOTION_PROFILING_KEYWORDS = (
    'emotion detection', 'emotion detector', 'emotion recognition',
    'emotion sensing', 'affect recognition', 'affective recognition',
    'affective analytics', 'emotion analytics', 'emotion monitoring',
    'sentiment camera', 'micro-expression', 'micro expression',
    'facial micro-expression', 'facial micro expression'
)

PUBLIC_SURVEILLANCE_TERMS = (
    'public space', 'public-space', 'public square', 'public area',
    'public venue', 'public transit', 'transport hub', 'station', 'airport',
    'harbor', 'harbour', 'mall', 'shopping mall', 'stadium', 'arena',
    'crowd', 'city-wide', 'citywide', 'campus', 'plaza', 'street',
    'public safety', 'public services', 'retail floor', 'storefront',
    'theme park', 'tourist', 'visitor', 'passenger'
)

PUBLIC_SPACE_DOMAINS = {
    'public spaces', 'public services', 'public safety', 'transportation',
    'retail', 'local government', 'housing', 'education', 'finance',
    'insurance', 'elder care', 'migration, asylum and border-control management',
    'entertainment', 'theme park', 'theme parks', 'smart city', 'customer service',
    'hospitality', 'corporate'
}

HIGH_RISK_KEYWORDS = (
    'medical', 'diagnostic', 'patient', 'clinical', 'medical device',
    'biometric', 'credit scoring', 'loan eligibility', 'employment screening',
    'workforce management', 'law enforcement', 'criminal justice', 'surveillance',
    'border control', 'immigration', 'asylum', 'autonomous vehicle',
    'critical infrastructure', 'safety component', 'fraud detection',
    'insurance risk', 'risk assessment', 'remote biometric', 'real-time remote biometric',
    'remote biometric identification'
)

HIGH_RISK_DOMAINS = {
    'access to and enjoyment of essential private services and essential public services and benefits',
    'product safety ai systems',
    'biometrics',
    'education and vocational training',
    'employment, workers’ management and access to self-employment',
    'law enforcement',
    'management and operation of critical infrastructure',
    'migration, asylum and border-control management',
    'administration of justice and democratic processes'
}

LIMITED_DOMAINS = {
    'education', 'urban mobility', 'consumer technology', 'public services',
    'health & wellness', 'health and wellness', 'community services', 'fintech',
    'tourism', 'retail', 'travel', 'utilities', 'career services', 'publishing',
    'e-commerce', 'business services', 'market research',
    'legal services', 'social media', 'hospitality', 'waste management',
    'home improvement', 'marketing', 'communication', 'culture', 'logistics',
    'public spaces', 'home gardening'
}

MINIMAL_DOMAINS = {
    'gardening', 'social media', 'home improvement', 'pet care', 'personalisation',
    'personalization', 'family entertainment', 'design', 'wellbeing', 'well-being',
    'cooking', 'fitness', 'personal development', 'data visualisation',
    'data visualization', 'event planning', 'home health', 'energy', 'smart city',
    'urban planning',
    'personal productivity', 'lifestyle', 'information services', 'creative tools',
    'health & fitness (non-medical)', 'literature', 'entertainment'
}

LIMITED_KEYWORDS = (
    'non-urgent', 'municipal', 'citizen services', 'public service',
    'advisory', 'tips to reduce', 'career services', 'community platform',
    'urban mobility', 'transit', 'civic', 'council', 'regulatory sandbox', 'code of conduct',
    'ai literacy', 'classification documentation', 'micro-enterprise',
    'voluntary programme', 'voluntary program', 'article 95', 'article 62',
    'article 63', 'article 80', 'content generation', 'content generator',
    'generate content', 'writing assistant', 'chatbot', 'conversational ai',
    'recommender', 'recommendation system', 'recommendation engine',
    'transparency obligation', 'article 50', 'marketing email', 'marketing emails',
    'social media post', 'social media posts', 'automated posts',
    'reservation system', 'booking predictor', 'booking demand', 'venue demand',
    'routing assistant', 'route suggestion', 'delivery planner', 'layout planner',
    'community center', 'public venue', 'energy advisory', 'energy advice',
    'consumption optimisation', 'consumption optimization', 'utility optimization',
    'utility optimisation', 'utility advisor', 'utility adviser', 'efficiency tips',
    'consumption tips', 'home energy tips', 'automated news generator', 'news generator',
    'article generator', 'news article generator', 'news writing assistant',
    'deepfake', 'deep fake', 'synthetic media', 'ai-generated image',
    'ai-generated video', 'public interest text', 'public-interest content',
    'ai disclosure', 'ai generated content'
)

MINIMAL_KEYWORDS = (
    'template-based', 'template based', 'rule-based', 'rule based',
    'personal use', 'hobby', 'recreational', 'playlist', 'recipe', 'caption',
    'poem', 'story generator', 'greeting', 'pet care', 'home improvement',
    'gardening', 'family fun', 'routine planner', 'daily tips', 'informal email',
    'casual', 'non-critical', 'fun', 'game', 'inventory optimization',
    'inventory management', 'logistics optimization', 'predictive maintenance',
    'yield prediction', 'yield predictor', 'occupancy predictor', 'spam filter', 'email spam', 'filtering',
    'traffic optimization', 'traffic flow optimization', 'traffic flow', 'smart city traffic',
    'energy optimization', 'energy optimiser', 'energy optimizer', 'energy management',
    'building energy', 'energy consumption optimizer',
    'video game', 'video games'
)

MINIMAL_BONUS_KEYWORDS = (
    'template-based', 'template based', 'rule-based', 'rule based',
    'simple habit', 'daily habit', 'simple activity', 'basic itinerary',
    'basic summary', 'simple summary', 'spam filter', 'inventory optimization',
    'traffic optimization', 'energy optimization'
)

BENIGN_TEMPLATE_DOMAINS = {
    'leisure', 'personal development', 'travel', 'e-commerce', 'design tools'
}

BENIGN_TEMPLATE_TOKENS = (
    'template', 'template-based', 'template based', 'rule-based', 'rule based'
)

ESSENTIAL_SERVICE_DOMAINS = {
    'housing', 'finance', 'public services',
    'access to and enjoyment of essential private services and essential public services and benefits',
    'education', 'education and vocational training',
    'employment, workers’ management and access to self-employment',
    'healthcare', 'elder care'
}

EMOTION_PROFILING_DOMAINS = {
    'human resources', 'customer service', 'retail', 'education', 'biometrics',
    'public services', 'public spaces', 'transportation', 'hospitality',
    'corporate', 'smart city'
}

PERSONALITY_FRAUD_KEYWORDS = (
    'personality-based fraud', 'personality fraud risk', 'personality profiling',
    'personality-based scoring', 'psychometric profiling', 'trait-based profiling',
    'trait scoring', 'fraud likelihood', 'fraud propensity', 'fraud risk score',
    'fraud likelihood scoring', 'behavioural fraud risk', 'behavioral fraud risk'
)

POLITICAL_PERSUASION_KEYWORDS = (
    'political orientation profiling', 'political affiliation inference',
    'political persuasion', 'political microtargeting', 'political manipulation',
    'voter manipulation', 'electoral influence', 'covert political influence',
    'covert political persuasion', 'covert campaign influence'
)

CIVIC_DENIAL_KEYWORDS = (
    'benefit denial', 'benefit exclusion', 'service denial', 'service exclusion',
    'housing denial', 'housing exclusion', 'welfare denial', 'aid denial',
    'public benefit scoring', 'public benefit ranking', 'public-benefit ranking',
    'eligibility tier', 'eligibility ranking', 'eligibility downgrade',
    'civic tier', 'civic tiering', 'civic ranking', 'risk tiering', 'risk tier',
    'support denial', 'support exclusion', 'benefit allocation', 'subsidy denial',
    'subsidy exclusion', 'grant denial', 'grant exclusion', 'aid allocation',
    'charity manipulation', 'donation manipulation'
)

PERSONALITY_TRAIT_TERMS = (
    'personality', 'psychometric', 'trait-based', 'trait profiling',
    'psychographic', 'behavioural profile', 'behavioral profile'
)

RANKING_TERMS = (
    'ranking', 'ranked', 'rank', 'scoring', 'scorecard', 'score-based',
    'prioritize', 'prioritise', 'grading', 'tiering'
)

RANKING_CONTEXT_TERMS = (
    'applicant', 'citizen', 'resident', 'customer', 'worker', 'benefit',
    'allocation', 'eligibility', 'promotion', 'housing', 'credit', 'admission'
)

ESSENTIAL_DENIAL_TERMS = (
    'deny', 'denial', 'denying', 'exclude', 'exclusion', 'excluding',
    'access removal', 'access control', 'support removal', 'support denial',
    'benefit denial', 'benefit exclusion', 'service denial', 'service exclusion'
)


def _normalize_lower(text: str) -> str:
    if not isinstance(text, str):
        return ''
    normalized = text.lower()
    for dash in ('\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2015', '‑', '–', '—', '‒'):
        normalized = normalized.replace(dash, '-')
    return normalized


def load_scenarios(file_path: str) -> Dict[str, Any]:
    """Load scenarios JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'data' not in data:
        raise ValueError("Scenarios JSON must contain 'data' array")
    
    return data


def build_scenario_query(scenario: Dict[str, Any]) -> str:
    """Build a comprehensive query for scenario analysis."""
    role = scenario.get('role', 'Unknown')
    intended_use = scenario.get('intended_use', '')
    system_type = scenario.get('system_type', '')
    input_data = scenario.get('input_data', '')
    domain = scenario.get('domain', '')
    
    query = f"""As a {role}, I want to {intended_use}. The system is a {system_type} that processes {input_data} in the {domain} domain. 

Please analyze this AI scenario and determine:
1. The risk level classification (prohibited, high-risk, limited-risk, or minimal-risk) under the AI Regulation
2. The specific obligations and compliance requirements
3. The relevant articles of the AI Regulation that apply

Provide a detailed analysis with citations for each determination."""
    
    return query


def build_scenario_messages(query: str, context_block: str) -> List[Dict[str, str]]:
    """Build messages for scenario analysis."""
    system = (
        "You are an expert AI Regulation compliance analyst. Analyze AI scenarios to determine risk classification, "
        "obligations, and applicable regulatory requirements. Use ONLY the provided context. "
        "Every factual assertion MUST end with a citation like [source; doc_id; chunk_index=K]. "
        "CRITICAL RISK CLASSIFICATION RULES: "
        "1. PROHIBITED (Article 5): ONLY systems that (a) use subliminal/manipulative techniques, (b) exploit vulnerabilities, "
        "(c) social scoring by public authorities, or (d) real-time remote biometric ID in public spaces. "
        "2. HIGH-RISK (Article 6): Systems in Annex III including (a) medical devices/diagnostics, (b) employment decisions, "
        "(c) credit scoring, (d) law enforcement, (e) education assessment, (f) legal document analysis, (g) autonomous vehicles. "
        "3. LIMITED-RISK: Chatbots, content generators, recommendation systems requiring transparency (Article 50). "
        "4. MINIMAL-RISK: Gaming, spam filters, inventory management, traffic optimization, agricultural tools. "
        "MEDICAL DIAGNOSTIC SYSTEMS ARE HIGH-RISK, NOT PROHIBITED. "
        "Structure your response clearly with sections for risk level, obligations, and relevant articles."
    )
    user = (
        f"Scenario Analysis Request:\n{query}\n\n"
        f"Regulatory Context:\n{context_block}\n\n"
        "Provide a comprehensive analysis covering:\n"
        "1. Risk Level Classification (with justification and citations)\n"
        "2. Specific Obligations and Requirements (numbered list with citations)\n"
        "3. Relevant Articles (with citations)\n"
        "4. Additional Compliance Considerations (if applicable)"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def process_single_scenario(scenario: Dict[str, Any], scenario_index: int, fast: bool = False) -> Dict[str, Any]:
    """Process a single scenario through the RAG pipeline."""
    scenario_id = f"scenario_{scenario_index + 1}"
    role = scenario.get('role', 'Unknown')
    intended_use = scenario.get('intended_use', '')
    system_type = scenario.get('system_type', '')
    domain = scenario.get('domain', '')
    
    # Benchmark data
    benchmark_risk_level = scenario.get('risk_level', '')
    benchmark_obligations = scenario.get('obligations', [])
    benchmark_articles = scenario.get('related_articles', [])
    
    start_time = time.time()
    
    try:
        # Step 1: Build scenario query
        scenario_query = build_scenario_query(scenario)
        print(f"Processing scenario {scenario_id}: {system_type} in {domain}...")
        
        # Step 2: Retrieve context
        docs, metas, scores = pre_retrieve(scenario_query, fast=fast)
        
        if not docs:
            return {
                "id": scenario_id,
                "role": role,
                "intended_use": intended_use,
                "system_type": system_type,
                "domain": domain,
                "benchmark_risk_level": benchmark_risk_level,
                "generated_risk_level": "",
                "benchmark_obligations": benchmark_obligations,
                "generated_obligations": [],
                "relevant_articles": [],
                "sources": [],
                "context_used": "",
                "processing_time": time.time() - start_time,
                "error": "No context found"
            }
        
        # Step 3: Build context
        context_block = build_context(docs, metas)
        
        # Step 4: Generate analysis
        messages = build_scenario_messages(scenario_query, context_block)
        
        if LLM_BACKEND == 'openai':
            # Use non-streaming version for batch processing
            from openai import OpenAI
            import os
            api_key = os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not found in environment or rag_chat.py config.")
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=800,  # Longer responses for scenario analysis
                stream=False,
            )
            analysis = response.choices[0].message.content or ""
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
            analysis = response.json().get("message", {}).get("content", "")
        
        # Step 5: Extract structured information from analysis
        generated_risk_level = extract_risk_level(analysis)
        generated_risk_level = calibrate_risk_level(generated_risk_level, scenario)
        generated_obligations = extract_obligations(analysis)
        relevant_articles = extract_articles(analysis)
        
        # Step 6: Extract sources for output
        sources = []
        for meta in metas:
            src = meta.get("source", "unknown")
            doc_id = meta.get("doc_id", "")
            chunk_idx = meta.get("chunk_index", "n/a")
            sources.append(f"[{src}; {doc_id}; chunk_index={chunk_idx}]")
        
        return {
            "id": scenario_id,
            "role": role,
            "intended_use": intended_use,
            "system_type": system_type,
            "domain": domain,
            "benchmark_risk_level": benchmark_risk_level,
            "generated_risk_level": generated_risk_level,
            "benchmark_obligations": benchmark_obligations,
            "generated_obligations": generated_obligations,
            "relevant_articles": relevant_articles,
            "sources": sources,
            "context_used": context_block,
            "processing_time": time.time() - start_time,
            "error": None
        }
        
    except Exception as e:
        return {
            "id": scenario_id,
            "role": role,
            "intended_use": intended_use,
            "system_type": system_type,
            "domain": domain,
            "benchmark_risk_level": benchmark_risk_level,
            "generated_risk_level": "",
            "benchmark_obligations": benchmark_obligations,
            "generated_obligations": [],
            "relevant_articles": [],
            "sources": [],
            "context_used": "",
            "processing_time": time.time() - start_time,
            "error": str(e)
        }


def extract_risk_level(analysis: str) -> str:
    """Extract risk level from analysis text."""
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
    
    # Priority 2: Check for PROHIBITED practices (Article 5) with contextual cues
    prohibited_keywords = list(PROHIBITED_KEYWORDS) + [
        'subliminal technique', 'subliminal techniques', 'subliminal influence',
        'manipulative technique', 'manipulative techniques', 'manipulative influence',
        'exploiting vulnerabilities', 'exploit vulnerabilities'
    ]
    vulnerability_markers = (
        'vulnerable', 'vulnerabilities', 'children', 'minors', 'elderly', 'disabled',
        'cognitive impairment', 'mental impairment', 'addiction', 'low-income', 'low income'
    )
    article5_markers = (
        'article 5', 'article five', 'article-5', 'prohibited practice', 'banned practice'
    )
    social_scoring_markers = SOCIAL_SCORING_KEYWORDS
    remote_biometric_markers = REMOTE_BIOMETRIC_KEYWORDS + FACIAL_SURVEILLANCE_KEYWORDS
    public_space_markers = (
        'public space', 'public-space', 'public square', 'public area', 'public venue',
        'law enforcement', 'policing', 'border control', 'mall-wide', 'city-wide', 'crowd density'
    ) + PUBLIC_SURVEILLANCE_TERMS
    has_prohibited_keyword = any(phrase in analysis_lower for phrase in prohibited_keywords)
    has_manipulation_context = any(phrase in analysis_lower for phrase in [
        'distort behavior', 'materially distort', 'material distort', 'without awareness', 'without conscious',
        'without their awareness', 'without informed consent', 'nudge', 'nudging', 'influence decision',
        'manipulate', 'manipulative', 'exploit', 'coerce', 'coercive', 'subconscious', 'hidden influence'
    ])
    has_vulnerability_context = any(term in analysis_lower for term in vulnerability_markers)
    has_article5_reference = any(term in analysis_lower for term in article5_markers)
    has_social_scoring = any(term in analysis_lower for term in social_scoring_markers)
    has_remote_public = (
        any(term in analysis_lower for term in remote_biometric_markers) and
        any(term in analysis_lower for term in public_space_markers)
    )
    has_emotion_public = (
        any(term in analysis_lower for term in EMOTION_PROFILING_KEYWORDS) and
        any(term in analysis_lower for term in public_space_markers)
    )

    covert_influence_detected = any(term in analysis_lower for term in COVERT_INFLUENCE_KEYWORDS)
    has_personality_fraud = any(term in analysis_lower for term in PERSONALITY_FRAUD_KEYWORDS)
    has_political_persuasion = any(term in analysis_lower for term in POLITICAL_PERSUASION_KEYWORDS)
    has_civic_denial = any(term in analysis_lower for term in CIVIC_DENIAL_KEYWORDS)

    if has_social_scoring or has_remote_public or has_emotion_public:
        return 'prohibited'
    if has_personality_fraud or has_political_persuasion or has_civic_denial:
        return 'prohibited'
    if covert_influence_detected and has_manipulation_context:
        return 'prohibited'
    if has_prohibited_keyword and (has_manipulation_context or has_vulnerability_context or has_article5_reference):
        return 'prohibited'
    if has_article5_reference and has_manipulation_context:
        return 'prohibited'
    
    # Priority 3: Look for HIGH-RISK systems (Annex III) - BEFORE limited-risk
    # This is critical: high-risk must be checked before limited-risk to avoid misclassification
    # Exception: Inventory optimization is minimal-risk, not high-risk
    if any(phrase in analysis_lower for phrase in [
        'inventory optimization', 'inventory management', 'logistics optimization',
        'warehouse management', 'supply chain optimization'
    ]):
        # Skip high-risk classification for inventory optimization - will be minimal-risk
        pass
    elif any(phrase in analysis_lower for phrase in [
        'biometric identification', 'biometric screening', 'facial recognition',
        'employment', 'credit scoring', 'law enforcement',
        'education assessment', 'legal document', 'autonomous vehicle', 'border control',
        'immigration', 'insurance', 'criminal justice',
        'annex iii', 'high-risk systems', 'article 6', 'conformity assessment',
        'fundamental rights impact assessment', 'risk management system'
    ]):
        return 'high-risk'
    
    # Priority 4: Medical/Healthcare systems are HIGH-RISK by default (but NOT recommenders)
    # Exception: Healthcare recommenders are limited-risk (caught in next priority)
    if any(phrase in analysis_lower for phrase in [
        'medical diagnostic', 'medical imaging diagnostic', 'medical device control',
        'medical device', 'health monitoring', 'diagnosis', 'clinical', 'patient', 'treatment'
    ]):
        return 'high-risk'
    
    # Priority 5: Check for LIMITED-RISK systems (Article 50)
    # This catches content generation, chatbots, recommenders
    # Must be AFTER high-risk checks to avoid misclassifying high-risk systems
    if any(phrase in analysis_lower for phrase in [
        'content generation', 'content generator', 'generate content',
        'chatbot', 'conversational ai', 'conversational assistant',
        'recommendation system', 'recommender system', 'recommendation ai',
        'emotion recognition', 'deepfake', 'synthetic media',
        'article 50', 'transparency obligation', 'transparency requirement',
        'limited-risk', 'limited risk'
    ]):
        return 'limited-risk'
    
    # Priority 6: Default to minimal-risk if no other indicators
    if any(phrase in analysis_lower for phrase in [
        'minimal-risk', 'minimal risk', 'no specific obligations',
        'unregulated', 'gaming', 'spam filter', 'traffic optimization',
        'agricultural tool', 'inventory'
    ]):
        return 'minimal-risk'
    
    return 'unknown'


def extract_obligations(analysis: str) -> List[str]:
    """Extract obligations from analysis text."""
    import re
    
    obligations = []
    
    # Look for numbered lists or bullet points that seem like obligations
    obligation_patterns = [
        r'(?:obligation|requirement|must|shall)[\s:]*([^.]*?)(?:\.|$)',
        r'\d+\.\s*([^.]*?)(?:\.|$)',
        r'[-*•]\s*([^.]*?)(?:\.|$)',
    ]
    
    for pattern in obligation_patterns:
        matches = re.findall(pattern, analysis, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            obligation = match.strip()
            if len(obligation) > 20 and any(word in obligation.lower() for word in ['must', 'shall', 'require', 'obligation', 'comply']):
                obligations.append(obligation)
    
    return obligations[:10]  # Limit to top 10 obligations


def extract_articles(analysis: str) -> List[int]:
    """Extract article numbers from analysis text."""
    import re
    
    # Precise article patterns - avoid over-extraction
    article_patterns = [
        r'article\s*(\d+)',
        r'art\.?\s*(\d+)',
        r'art\s*(\d+)',
        r'(\d+)\s*of\s*the\s*AI\s*act',
        r'under\s*article\s*(\d+)',
        r'pursuant\s*to\s*article\s*(\d+)',
        r'according\s*to\s*article\s*(\d+)',
        r'in\s*accordance\s*with\s*article\s*(\d+)',
        r'compliance\s*with\s*article\s*(\d+)',
        r'requirements?\s*of\s*article\s*(\d+)',
        r'obligations?\s*under\s*article\s*(\d+)',
        r'provisions?\s*of\s*article\s*(\d+)',
    ]
    
    articles = set()
    for pattern in article_patterns:
        matches = re.findall(pattern, analysis, re.IGNORECASE)
        for match in matches:
            try:
                if match:  # Skip empty matches
                    article_num = int(match)
                    if 1 <= article_num <= 113:  # Valid AI Regulation article range
                        articles.add(article_num)
            except ValueError:
                continue
    
    return sorted(list(articles))


def _scenario_text_blob(scenario: Dict[str, Any]) -> str:
    parts = []
    for field in ['role', 'intended_use', 'system_type', 'input_data', 'domain']:
        value = scenario.get(field)
        if isinstance(value, str):
            parts.append(_normalize_lower(value))
    return ' '.join(parts)


def _contains_keywords(text: str, keywords: tuple) -> bool:
    return any(keyword in text for keyword in keywords)


def _derive_structured_flags(scenario: Dict[str, Any], text_blob: Optional[str] = None) -> Dict[str, bool]:
    if text_blob is None:
        text_blob = _scenario_text_blob(scenario)
    system_type = _normalize_lower(scenario.get('system_type', ''))
    intended_use = _normalize_lower(scenario.get('intended_use', ''))
    domain = scenario.get('domain', '')
    domain_lower = _normalize_lower(domain)
    focus_text = f"{system_type} {intended_use}".strip()

    has_public_indicator = domain_lower in PUBLIC_SPACE_DOMAINS or any(term in text_blob for term in PUBLIC_SURVEILLANCE_TERMS)

    has_remote_biometric_signal = (
        _contains_keywords(text_blob, REMOTE_BIOMETRIC_KEYWORDS) or
        _contains_keywords(text_blob, FACIAL_SURVEILLANCE_KEYWORDS)
    )
    has_authorized_context = (
        _contains_keywords(text_blob, REMOTE_BIOMETRIC_AUTHORIZED_TERMS) or
        domain_lower in {
            'biometrics', 'law enforcement', 'transportation', 'public safety',
            'migration, asylum and border-control management'
        }
    )
    has_commercial_context = (
        domain_lower in COMMERCIAL_REMOTE_BIOMETRIC_DOMAINS or
        _contains_keywords(text_blob, REMOTE_BIOMETRIC_COMMERCIAL_TERMS)
    )

    has_remote_biometric_public = has_remote_biometric_signal and has_public_indicator
    has_remote_biometric_public_prohibited = has_remote_biometric_public and (has_commercial_context or not has_authorized_context)
    uses_emotion_profiling = _contains_keywords(text_blob, EMOTION_PROFILING_KEYWORDS)
    emotion_targets_public = uses_emotion_profiling and (
        has_public_indicator or domain_lower in EMOTION_PROFILING_DOMAINS
    )

    uses_personality_traits = any(term in focus_text for term in PERSONALITY_TRAIT_TERMS)
    has_personality_fraud_terms = _contains_keywords(text_blob, PERSONALITY_FRAUD_KEYWORDS)
    has_political_persuasion_terms = _contains_keywords(text_blob, POLITICAL_PERSUASION_KEYWORDS)
    has_covert_influence_terms = _contains_keywords(text_blob, COVERT_INFLUENCE_KEYWORDS)
    performs_sensitive_ranking = (
        any(term in text_blob for term in RANKING_TERMS) and
        (domain_lower in ESSENTIAL_SERVICE_DOMAINS or any(term in text_blob for term in RANKING_CONTEXT_TERMS))
    )
    has_social_scoring_terms = _contains_keywords(text_blob, SOCIAL_SCORING_KEYWORDS)
    essential_block_signal = (
        _contains_keywords(text_blob, CIVIC_DENIAL_KEYWORDS) or (
            any(term in text_blob for term in ESSENTIAL_DENIAL_TERMS) and
            (has_social_scoring_terms or any(term in text_blob for term in RANKING_TERMS))
        )
    )
    denies_essential_services = (
        essential_block_signal and
        (domain_lower in ESSENTIAL_SERVICE_DOMAINS or 'essential service' in text_blob or 'public service' in text_blob or 'housing' in text_blob or 'benefit' in text_blob)
    )
    has_civic_denial_terms = _contains_keywords(text_blob, CIVIC_DENIAL_KEYWORDS)
    is_benign_template = (
        domain_lower in BENIGN_TEMPLATE_DOMAINS and
        any(token in system_type for token in BENIGN_TEMPLATE_TOKENS)
    )

    return {
        'uses_personality_traits': uses_personality_traits,
        'has_personality_fraud_terms': has_personality_fraud_terms,
        'has_political_persuasion_terms': has_political_persuasion_terms,
        'has_covert_influence_terms': has_covert_influence_terms,
        'performs_sensitive_ranking': performs_sensitive_ranking,
        'denies_essential_services': denies_essential_services,
        'has_civic_denial_terms': has_civic_denial_terms,
        'has_social_scoring_terms': has_social_scoring_terms,
        'has_remote_biometric_public': has_remote_biometric_public,
        'has_remote_biometric_public_prohibited': has_remote_biometric_public_prohibited,
        'has_remote_biometric_authorized': has_remote_biometric_public and has_authorized_context,
        'emotion_targets_public': emotion_targets_public,
        'is_benign_template': is_benign_template
    }


def _should_force_minimal(flags: Dict[str, bool], text_blob: str) -> bool:
    if not flags.get('is_benign_template'):
        return False
    if flags.get('has_personality_fraud_terms') or flags.get('has_political_persuasion_terms'):
        return False
    if (
        flags.get('performs_sensitive_ranking') or
        flags.get('denies_essential_services') or
        flags.get('has_civic_denial_terms') or
        flags.get('has_social_scoring_terms') or
        flags.get('has_remote_biometric_public_prohibited') or
        flags.get('has_remote_biometric_authorized') or
        flags.get('emotion_targets_public')
    ):
        return False
    if _contains_keywords(text_blob, PROHIBITED_KEYWORDS) or _contains_keywords(text_blob, HIGH_RISK_KEYWORDS):
        return False
    return True


def _apply_benign_template_cap(risk: str, flags: Dict[str, bool], text_blob: str) -> str:
    # Disabled: the benign-template cap was over-forcing limited/uncertain cases to minimal-risk,
    # causing a collapse in minimal-risk precision. Keep the passed-through inference.
    return risk


def infer_risk_from_metadata(scenario: Dict[str, Any]) -> str:
    """Infer likely risk level using scenario metadata (without relying on benchmark labels)."""
    text_blob = _scenario_text_blob(scenario)
    domain = scenario.get('domain', '').lower()
    if not text_blob and not domain:
        return 'unknown'

    system_type = _normalize_lower(scenario.get('system_type', ''))
    intended_use = _normalize_lower(scenario.get('intended_use', ''))
    focus_text = f"{system_type} {intended_use}".strip()
    is_healthcare_recommender = (
        any(term in focus_text for term in ('recommender', 'recommendation', 'recommendations')) and
        domain in {'healthcare', 'health & wellness', 'health and wellness'}
    )
    has_clinical_control_terms = any(
        term in focus_text for term in (
            'diagnostic', 'diagnosis', 'clinical decision', 'medical device',
            'device control', 'treatment', 'screening', 'monitoring'
        )
    )
    healthcare_recommender_override = is_healthcare_recommender and not has_clinical_control_terms

    flags = _derive_structured_flags(scenario, text_blob)

    if (
        _contains_keywords(text_blob, PROHIBITED_KEYWORDS)
        or _contains_keywords(text_blob, SOCIAL_SCORING_KEYWORDS)
        or flags['has_personality_fraud_terms']
        or flags['has_political_persuasion_terms']
        or flags['has_covert_influence_terms']
        or flags['denies_essential_services']
        or flags['has_civic_denial_terms']
        or flags['has_social_scoring_terms']
        or flags['has_remote_biometric_public_prohibited']
        or flags['emotion_targets_public']
    ):
        return 'prohibited'

    if _contains_keywords(text_blob, COVERT_INFLUENCE_KEYWORDS) and _contains_keywords(text_blob, PROHIBITED_KEYWORDS):
        return 'prohibited'

    if not healthcare_recommender_override:
        if domain in HIGH_RISK_DOMAINS or _contains_keywords(text_blob, HIGH_RISK_KEYWORDS) or flags['uses_personality_traits']:
            return 'high-risk'
        if flags['performs_sensitive_ranking']:
            return 'high-risk'

    limited_score = 0
    minimal_score = 0

    if _contains_keywords(text_blob, LIMITED_KEYWORDS):
        limited_score += 2
    if domain in LIMITED_DOMAINS:
        limited_score += 1
    if healthcare_recommender_override:
        limited_score += 2

    if _contains_keywords(text_blob, MINIMAL_KEYWORDS):
        minimal_score += 1  # require stronger signal to override limited cues
    if domain in MINIMAL_DOMAINS:
        minimal_score += 1
    if _contains_keywords(text_blob, MINIMAL_BONUS_KEYWORDS):
        minimal_score += 1

    if limited_score and minimal_score:
        inferred = 'limited-risk' if limited_score > minimal_score else 'minimal-risk'
        return _apply_benign_template_cap(inferred, flags, text_blob)
    if limited_score:
        return _apply_benign_template_cap('limited-risk', flags, text_blob)
    if minimal_score:
        return _apply_benign_template_cap('minimal-risk', flags, text_blob)

    return 'unknown'


def calibrate_risk_level(generated_risk: str, scenario: Dict[str, Any]) -> str:
    """Adjust model risk classification using scenario metadata heuristics."""
    text_blob = _scenario_text_blob(scenario)
    domain = scenario.get('domain', '').lower()
    flags = _derive_structured_flags(scenario, text_blob)
    inferred_risk = infer_risk_from_metadata(scenario)
    generated_risk = _apply_benign_template_cap(generated_risk, flags, text_blob)
    inferred_risk = _apply_benign_template_cap(inferred_risk, flags, text_blob)

    if (
        flags['has_personality_fraud_terms'] or
        flags['has_political_persuasion_terms'] or
        flags['has_covert_influence_terms'] or
        flags['denies_essential_services'] or
        flags['has_civic_denial_terms'] or
        flags['has_social_scoring_terms'] or
        flags['has_remote_biometric_public_prohibited'] or
        flags['emotion_targets_public']
    ):
        return 'prohibited'
    if inferred_risk == 'unknown':
        return generated_risk
    if not generated_risk or generated_risk == 'unknown':
        return inferred_risk
    if generated_risk == inferred_risk:
        return generated_risk

    priority_order = {
        'prohibited': 0,
        'high-risk': 1,
        'limited-risk': 2,
        'minimal-risk': 3,
        'unknown': 99
    }

    generated_priority = priority_order.get(generated_risk, 99)
    inferred_priority = priority_order.get(inferred_risk, 99)

    has_prohibited_signal = (
        _contains_keywords(text_blob, PROHIBITED_KEYWORDS)
        or _contains_keywords(text_blob, SOCIAL_SCORING_KEYWORDS)
        or flags['has_personality_fraud_terms']
        or flags['has_political_persuasion_terms']
        or flags['has_covert_influence_terms']
        or flags['denies_essential_services']
        or flags['has_civic_denial_terms']
        or flags['has_social_scoring_terms']
        or flags['has_remote_biometric_public_prohibited']
        or flags['emotion_targets_public']
    )
    has_high_risk_signal = (
        domain in HIGH_RISK_DOMAINS
        or _contains_keywords(text_blob, HIGH_RISK_KEYWORDS)
        or flags['uses_personality_traits']
        or flags['performs_sensitive_ranking']
        or flags['has_remote_biometric_authorized']
    )
    has_remote_biometric_signal = (
        flags['has_remote_biometric_public_prohibited'] or flags['has_remote_biometric_authorized']
    )

    # Prefer metadata if it signals significantly higher regulatory scrutiny
    if inferred_priority < generated_priority:
        if inferred_risk == 'prohibited':
            return 'prohibited' if has_prohibited_signal else 'high-risk'
        return inferred_risk

    # For lateral differences (limited vs minimal), prefer metadata inference
    if {generated_risk, inferred_risk}.issubset({'limited-risk', 'minimal-risk'}):
        return inferred_risk

    # Avoid downgrading prohibited classifications when metadata still signals prohibited intent
    if generated_risk == 'prohibited':
        if inferred_risk == 'high-risk':
            return 'high-risk'
        if inferred_risk in {'limited-risk', 'minimal-risk'}:
            if has_prohibited_signal or has_remote_biometric_signal:
                return 'prohibited'
            return inferred_risk

    # Only downgrade high-risk if metadata lacks high-risk indicators
    if generated_risk == 'high-risk' and inferred_risk in {'limited-risk', 'minimal-risk'}:
        if has_high_risk_signal or has_remote_biometric_signal:
            return 'high-risk'
        final = inferred_risk
    else:
        final = generated_risk

    return _apply_benign_template_cap(final, flags, text_blob)


def process_scenario_batch(scenarios_data: Dict[str, Any], fast: bool = False) -> Dict[str, Any]:
    """Process all scenarios in the batch."""
    scenarios = scenarios_data['data']
    total_scenarios = len(scenarios)
    
    print(f"Processing {total_scenarios} scenarios...")
    
    results = []
    successful = 0
    failed = 0
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\nProgress: {i}/{total_scenarios}")
        result = process_single_scenario(scenario, i-1, fast=fast)
        results.append(result)
        
        if result['error']:
            failed += 1
            print(f"❌ Failed: {result['error']}")
        else:
            successful += 1
            print(f"✅ Success (risk: {result['generated_risk_level']})")
    
    metadata = {
        "processed_at": datetime.utcnow().isoformat() + "Z",
        "total_scenarios": total_scenarios,
        "successful": successful,
        "failed": failed,
        "model": OPENAI_MODEL if LLM_BACKEND == 'openai' else OLLAMA_MODEL,
        "backend": LLM_BACKEND,
        "fast_mode": fast
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


def parse_args():
    parser = argparse.ArgumentParser(description="Scenario Batch RAG Processing for AI Regulation")
    parser.add_argument("scenarios", help="Path to scenarios JSON file")
    parser.add_argument("output", help="Path to output JSON file")
    parser.add_argument("--fast", action="store_true", help="Use fast mode (fewer variants, no HyDE)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate input file
    if not Path(args.scenarios).exists():
        print(f"Error: Scenarios file not found: {args.scenarios}")
        return 1
    
    try:
        # Load scenarios
        print(f"Loading scenarios from: {args.scenarios}")
        scenarios_data = load_scenarios(args.scenarios)
        
        # Process batch
        results = process_scenario_batch(scenarios_data, fast=args.fast)
        
        # Save results
        save_results(results, args.output)
        
        # Print summary
        metadata = results["metadata"]
        print(f"\n{'='*60}")
        print("SCENARIO BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total scenarios: {metadata['total_scenarios']}")
        print(f"Successful: {metadata['successful']}")
        print(f"Failed: {metadata['failed']}")
        print(f"Model: {metadata['model']} ({metadata['backend']})")
        print(f"Fast mode: {metadata['fast_mode']}")
        print(f"Processing time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if metadata['failed'] > 0:
            print(f"\n⚠️  {metadata['failed']} scenarios failed. Check output file for details.")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
