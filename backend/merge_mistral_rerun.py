import json

main_path = "scenarios_output_mistralai_ministral-8b-2512.json"
rerun_path = "scenarios_output_mistralai_ministral-8b-2512_failed.json"
out_path = main_path

with open(main_path, "r", encoding="utf-8") as f:
    main = json.load(f)
with open(rerun_path, "r", encoding="utf-8") as f:
    rerun = json.load(f)

rerun_by_id = {r["id"]: r for r in rerun["results"]}
merged = 0
for i, r in enumerate(main["results"]):
    if r["id"] in rerun_by_id:
        main["results"][i] = rerun_by_id[r["id"]]
        merged += 1

main["metadata"]["merged_rerun"] = merged
main["metadata"]["rerun_model"] = rerun["metadata"].get("model", "unknown")

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(main, f, indent=2, ensure_ascii=False)

print(f"Merged {merged} re-run entries into {out_path}")
