import json
import os
import subprocess
from pathlib import Path

BENCHMARK = "scenarios.json"

MODELS = {
    "gpt-5.4-mini": "../experiments/V7/scenarios_output_v7.json",
    "llama-3.1-70b": "scenarios_output_meta-llama_llama-3.1-70b-instruct.json",
    "claude-sonnet-5": "scenarios_output_anthropic_claude-sonnet-5.json",
    "mistral-8b-2512": "scenarios_output_mistralai_ministral-8b-2512.json",
}

SUMMARY = []

for model, result_file in MODELS.items():
    if not Path(result_file).exists():
        print(f"Missing {result_file}; skipping {model}")
        continue

    eval_file = f"scenario_evaluation_{model}.json"
    report_file = f"classification_report_{model}.txt"

    print(f"\n=== Evaluating {model} ===")
    subprocess.run(["python", "scenario_evaluation.py", BENCHMARK, result_file, eval_file], check=False)

    print(f"=== Analyzing {model} ===")
    with open(report_file, "w", encoding="utf-8") as out:
        subprocess.run(["python", "analyze_classification.py", eval_file], stdout=out, stderr=subprocess.STDOUT)

    # Extract overall risk accuracy
    try:
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        overall = eval_data.get("overall_metrics", {})
        SUMMARY.append({
            "model": model,
            "result_file": result_file,
            "eval_file": eval_file,
            "report_file": report_file,
            "risk_level_accuracy": overall.get("risk_level_accuracy"),
        })
    except Exception as e:
        print(f"Could not parse summary for {model}: {e}")

with open("model_comparison_summary.json", "w", encoding="utf-8") as f:
    json.dump(SUMMARY, f, indent=2, ensure_ascii=False)

print("\n=== Summary ===")
for s in SUMMARY:
    print(f"{s['model']}: risk accuracy = {s['risk_level_accuracy']}")
