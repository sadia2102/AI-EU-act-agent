import json
from datetime import datetime, timezone
from pathlib import Path

root = Path('/Users/sadiakhantumpa/Desktop/Thesis MSc')
subset_path = root / 'scenarios_failed_block_run4.json'
replay_path = root / 'scenario_results_failed_block_run4.json'
full_path = root / 'scenario_results_full_run4.json'

with open(subset_path, 'r', encoding='utf-8') as f:
    subset = json.load(f)
orig_ids = subset['original_ids']

with open(replay_path, 'r', encoding='utf-8') as f:
    replay = json.load(f)

if len(orig_ids) != len(replay['results']):
    raise SystemExit('Mismatch between original_ids and replay results count')

for idx, scenario_id in enumerate(orig_ids):
    replay['results'][idx]['id'] = scenario_id

with open(full_path, 'r', encoding='utf-8') as f:
    full = json.load(f)

replay_map = {result['id']: result for result in replay['results']}
updated_results = []
for result in full['results']:
    rid = result['id']
    if rid in replay_map:
        updated_results.append(replay_map[rid])
    else:
        updated_results.append(result)

full['results'] = updated_results
success = sum(1 for r in full['results'] if not r.get('error'))
failed = sum(1 for r in full['results'] if r.get('error'))
full['metadata']['successful'] = success
full['metadata']['failed'] = failed
full['metadata']['processed_at'] = datetime.now(timezone.utc).isoformat()

with open(full_path, 'w', encoding='utf-8') as f:
    json.dump(full, f, indent=2, ensure_ascii=False)

print(f'Patched {len(replay_map)} failed scenarios. Success: {success}, Failed: {failed}')
