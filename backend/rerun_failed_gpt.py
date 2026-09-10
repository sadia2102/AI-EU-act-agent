import json
import os
import subprocess

MAIN = "scenarios_output_openai_gpt-5.4-mini.json"
FAILED_IN = "scenarios_gpt_failed.json"
FAILED_OUT = "scenarios_output_openai_gpt-5.4-mini_failed.json"

def merge(source, target):
    with open(target, "r", encoding="utf-8") as f:
        tgt = json.load(f)
    with open(source, "r", encoding="utf-8") as f:
        src = json.load(f)
    src_by_id = {r["id"]: r for r in src["results"]}
    for i, r in enumerate(tgt["results"]):
        if r["id"] in src_by_id:
            tgt["results"][i] = src_by_id[r["id"]]
    with open(target, "w", encoding="utf-8") as f:
        json.dump(tgt, f, indent=2, ensure_ascii=False)

with open(MAIN, "r", encoding="utf-8") as f:
    data = json.load(f)
failed = [r for r in data["results"] if r.get("error")]
print(f"Extracted {len(failed)} failed scenarios")
with open(FAILED_IN, "w", encoding="utf-8") as f:
    json.dump({"questions": failed}, f, indent=2, ensure_ascii=False)

if os.path.exists(FAILED_OUT):
    os.remove(FAILED_OUT)
env = os.environ.copy()
subprocess.run(["python", "batch_rag.py", FAILED_IN, FAILED_OUT], env=env)

merge(FAILED_OUT, MAIN)

with open(MAIN, "r", encoding="utf-8") as f:
    main = json.load(f)
still = [r for r in main["results"] if r.get("error")]
print(f"Final: {len(main['results'])} total, {len(main['results'])-len(still)} success, {len(still)} failed")
