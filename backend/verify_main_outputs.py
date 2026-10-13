import json
from pathlib import Path

files = {
    "gpt-5.4-mini": "scenarios_output_openai_gpt-5.4-mini.json",
    "llama-3.1-70b": "scenarios_output_meta-llama_llama-3.1-70b-instruct.json",
    "claude-sonnet-5": "scenarios_output_anthropic_claude-sonnet-5.json",
    "mistral-8b-2512": "scenarios_output_mistralai_ministral-8b-2512.json",
}

for model, path in files.items():
    if not Path(path).exists():
        print(f"{model}: FILE NOT FOUND {path}")
        continue
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("metadata", {})
    results = data.get("results", [])
    failed = [r for r in results if r.get("error")]
    print(f"{model}: {len(results)} results, {len(results)-len(failed)} success, {len(failed)} failed")
    print(f"  metadata: successful={meta.get('successful')}, failed={meta.get('failed')}")
