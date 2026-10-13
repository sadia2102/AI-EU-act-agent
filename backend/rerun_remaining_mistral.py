import json
import os
import subprocess
import time

MAIN = "scenarios_output_mistralai_ministral-8b-2512.json"
FAILED_167 = "scenarios_output_mistralai_ministral-8b-2512_failed.json"
REMAINING_IN = "scenarios_mistral_failed3.json"
REMAINING_OUT = "scenarios_output_mistralai_ministral-8b-2512_failed3.json"

def merge(source_json, target_json):
    with open(target_json, "r", encoding="utf-8") as f:
        target = json.load(f)
    with open(source_json, "r", encoding="utf-8") as f:
        source = json.load(f)
    source_by_id = {r["id"]: r for r in source["results"]}
    merged = 0
    for i, r in enumerate(target["results"]):
        if r["id"] in source_by_id:
            target["results"][i] = source_by_id[r["id"]]
            merged += 1
    with open(target_json, "w", encoding="utf-8") as f:
        json.dump(target, f, indent=2, ensure_ascii=False)
    return merged

# 1. Merge the 164 successful re-runs into the main file
print("Merging 167-re-run...")
print(f"  merged {merge(FAILED_167, MAIN)} entries")

# 2. Extract remaining failed scenarios (should be 3)
with open(MAIN, "r", encoding="utf-8") as f:
    main = json.load(f)
failed = [r for r in main["results"] if r.get("error")]
print(f"Remaining failed: {len(failed)}")
for r in failed:
    print(f"  {r['id']}: {(r.get('error') or '')[:120]}")
if not failed:
    print("All 339 succeeded. Done.")
    exit(0)

with open(REMAINING_IN, "w", encoding="utf-8") as f:
    json.dump({"questions": failed}, f, indent=2, ensure_ascii=False)
print(f"Wrote {REMAINING_IN}")

# 3. Re-run the remaining scenarios
if os.path.exists(REMAINING_OUT):
    os.remove(REMAINING_OUT)
env = os.environ.copy()
cmd = ["python", "batch_rag.py", REMAINING_IN, REMAINING_OUT]
print(f"Running: {' '.join(cmd)}")
subprocess.run(cmd, env=env)

# 4. Merge the last re-run
print("Merging final re-run...")
print(f"  merged {merge(REMAINING_OUT, MAIN)} entries")

# 5. Final check
with open(MAIN, "r", encoding="utf-8") as f:
    main = json.load(f)
still_failed = [r for r in main["results"] if r.get("error")]
print(f"Final: {len(main['results'])} total, {len(main['results'])-len(still_failed)} successful, {len(still_failed)} failed")
