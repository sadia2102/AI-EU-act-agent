import json
import os
import subprocess

MAIN = "scenarios_output_meta-llama_llama-3.1-70b-instruct.json"
LAST3_IN = "scenarios_llama_last3.json"
LAST3_OUT = "scenarios_output_meta-llama_llama-3.1-70b-instruct_last3.json"

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
last3 = [r for r in data["results"] if r.get("error")]
print(f"Re-running {len(last3)} scenarios: {[r['id'] for r in last3]}")
with open(LAST3_IN, "w", encoding="utf-8") as f:
    json.dump({"questions": last3}, f, indent=2, ensure_ascii=False)

if os.path.exists(LAST3_OUT):
    os.remove(LAST3_OUT)
env = os.environ.copy()
subprocess.run(["python", "batch_rag.py", LAST3_IN, LAST3_OUT], env=env)

merge(LAST3_OUT, MAIN)

with open(MAIN, "r", encoding="utf-8") as f:
    main = json.load(f)
still = [r for r in main["results"] if r.get("error")]
print(f"Final: {len(main['results'])} total, {len(main['results'])-len(still)} success, {len(still)} failed")
