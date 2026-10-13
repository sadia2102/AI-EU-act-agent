import json

in_path = "scenarios_output_mistralai_ministral-8b-2512.json"
out_path = "scenarios_mistral_failed.json"

with open(in_path, "r", encoding="utf-8") as f:
    data = json.load(f)

failed = [r for r in data["results"] if r.get("error")]
print(f"Extracted {len(failed)} failed scenarios")

with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"questions": failed}, f, indent=2, ensure_ascii=False)

print(f"Wrote {out_path}")
